"""Checks the drafted prompts against the model and writes the sets the experiments use.

    .venv/bin/python -m src.check_prompts

Input:  src/prompt_drafts.py (edit prompts there, then rerun this file).
Output: data/prompts/pass_at_k.json, data/prompts/swap.json, data/prompts/REVIEW.md.

No lens is used here, so nothing in this file can favor one lens over another.

Read-position choices, made per prompt, are explained in prompt_drafts.py. Pass@k prompts are kept if:
  * the intermediate is one token (with or without a leading space);
  * it isn't in the prompt and isn't the model's top next token;
  * where a prompt has a final answer, the answer's first token is in the model's top 3.
Each kept prompt gets a control token of nearby frequency rank (see choose_controls).

Swap templates: an argument is kept if it is one token and the model's top next token is
the first token of its answer. A trial is an ordered pair of kept arguments whose answers
start with different tokens.

Splits are made by group so validation and test share nothing: by intermediate word for
pass@k, and by template for swaps.
"""
from __future__ import annotations

import json
import random
from collections import Counter, defaultdict

import torch

from . import prompt_drafts
from .model import DATA, load_model

OUT = DATA / "prompts"
FREQUENCY_PATH = OUT / "token_frequency.pt"
ANSWER_RANK = 3
SEED = 0
N_PASS = {"val": 10, "test": 50}
N_SWAP = {"val": 3, "test": 9}
ARGS_PER_TEMPLATE = 4
DOMAINS = ("wiki", "stories", "python")


def one_token(tok, word: str) -> int | None:
    """The token for `word`, preferring the leading-space form, or None if it needs several."""
    for form in (" " + word, word):
        ids = tok(form, add_special_tokens=False)["input_ids"]
        if len(ids) == 1:
            return ids[0]
    return None


def first_token(tok, answer: str) -> int:
    return tok(answer, add_special_tokens=False)["input_ids"][0]


def word_key(text: str) -> str:
    """Pass@k compares tokens ignoring case and surrounding space."""
    return text.strip().lower()


def token_class(text: str) -> str:
    s = text.strip()
    if s.isdigit():
        return "digit"
    if s.isascii() and s.isalpha() and len(s) >= 3:
        return "Word" if s[0].isupper() else "word"
    return "other"


@torch.no_grad()
def next_token_logits(model, tok, prompt: str) -> torch.Tensor:
    ids = tok(prompt, return_tensors="pt").input_ids.to(next(model.parameters()).device)
    return model(input_ids=ids).logits[0, -1].float().cpu()


# ---------------------------------------------------------------------------------------
# Token frequency
# ---------------------------------------------------------------------------------------

def token_frequency(tok) -> torch.Tensor:
    """Per-token frequency in fit-side text, computed per domain and averaged across domains.

    wiki: the WikiText-2 train split. stories: the first 5,000 stories of the TinyStories
    train shard. python: the fit candidates. Validation and test text are never counted.
    """
    if FREQUENCY_PATH.exists():
        return torch.load(FREQUENCY_PATH, weights_only=True)
    from datasets import load_dataset

    vocab = len(tok)
    wiki = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    stories = load_dataset("roneneldan/TinyStories", split="train[:5000]", data_files={
        "train": "data/train-00000-of-00004-2d5a1467fff1081b.parquet"})
    streams = [torch.tensor(tok("".join(wiki["text"]), add_special_tokens=False)["input_ids"]),
               torch.tensor(tok("\n\n".join(stories["text"]), add_special_tokens=False)["input_ids"]),
               torch.load(DATA / "python_candidates.pt", weights_only=True)["fit"]["windows"].flatten()]
    per_domain = [torch.bincount(s, minlength=vocab).double() / len(s) for s in streams]
    frequency = torch.stack(per_domain).mean(0)
    OUT.mkdir(exist_ok=True)
    torch.save(frequency, FREQUENCY_PATH)
    return frequency


def choose_controls(tok, items: list[dict], frequency: torch.Tensor):
    """Give each prompt a control token: same token class, nearest frequency rank, unrelated.

    Unrelated here means: not in the prompt, and not any intermediate or answer in this set.
    Each control word is used at most once where possible.
    """
    order = frequency.argsort(descending=True).tolist()
    rank = {t: r for r, t in enumerate(order)}
    decoded = {t: tok.decode(t) for t in order[:60_000]}
    by_class = defaultdict(list)
    for t, text in decoded.items():
        by_class[token_class(text)].append(t)
    banned = {word_key(i["intermediate"]) for i in items} | {word_key(i.get("answer") or "") for i in items}
    used = set()
    for item in items:
        target = one_token(tok, item["intermediate"])
        klass = token_class(tok.decode(target))
        def fits(t, allow_reuse=False):
            key = word_key(decoded[t])
            return (key not in banned and key not in item["prompt"].lower()
                    and (allow_reuse or key not in used))
        pool = sorted(by_class[klass], key=lambda t: abs(rank[t] - rank[target]))
        # Digits are too few to give each prompt its own, so they may repeat.
        choice = next((t for t in pool if t != target and fits(t, klass == "digit")), None)
        if choice is None:
            # Every digit is an intermediate or answer somewhere; fall back to this prompt only.
            own = {word_key(item["intermediate"]), word_key(item.get("answer") or "")}
            choice = next(t for t in pool if word_key(decoded[t]) not in own
                          and word_key(decoded[t]) not in item["prompt"].lower())
        used.add(word_key(decoded[choice]))
        item["control"] = decoded[choice].strip()
        item["target_rank"], item["control_rank"] = rank[target], rank[choice]


# ---------------------------------------------------------------------------------------
# Pass@k
# ---------------------------------------------------------------------------------------

def check_pass_item(model, tok, draft: dict) -> dict:
    item = dict(draft)
    word, prompt = item["intermediate"], item["prompt"]
    logits = next_token_logits(model, tok, prompt)
    top = logits.topk(5).indices.tolist()
    item["model_top"] = [tok.decode(t) for t in top]
    problems = []
    if one_token(tok, word) is None:
        problems.append("intermediate is more than one token")
    if word.lower() in prompt.lower():
        problems.append("intermediate is in the prompt")
    if word_key(tok.decode(top[0])) == word_key(word):
        problems.append("intermediate is the top next token")
    if item.get("answer"):
        answer_ids = {first_token(tok, " " + item["answer"]), first_token(tok, item["answer"])}
        order = logits.argsort(descending=True).tolist()
        item["answer_rank"] = min(order.index(t) for t in answer_ids) + 1
        if item["answer_rank"] > ANSWER_RANK:
            problems.append(f"answer is rank {item['answer_rank']}")
    item["problems"] = problems
    return item


def split_pass(items: list[dict], domain: str) -> dict[str, list[dict]]:
    """Split one domain by intermediate word, then cap test, keeping a spread of difficulty.

    Validation words are pinned in prompt_drafts.VAL_WORDS, so editing drafts can't move a
    prompt that has already been scored with a lens into test.
    """
    rng = random.Random(SEED)
    pinned = {word_key(w) for w in prompt_drafts.VAL_WORDS[domain]}
    val = [i for i in items if word_key(i["intermediate"]) in pinned]
    test = [i for i in items if word_key(i["intermediate"]) not in pinned]
    rng.shuffle(test)
    by_difficulty = defaultdict(list)
    for item in test:
        by_difficulty[item["difficulty"]].append(item)
    capped = []
    while len(capped) < N_PASS["test"] and any(by_difficulty.values()):
        for level in sorted(by_difficulty):
            if by_difficulty[level] and len(capped) < N_PASS["test"]:
                capped.append(by_difficulty[level].pop(rng.randrange(len(by_difficulty[level]))))
    return {"val": val, "test": capped}


# ---------------------------------------------------------------------------------------
# Swaps
# ---------------------------------------------------------------------------------------

def check_template(model, tok, draft: dict) -> dict:
    """Decide which arguments work in one template, and list its trials."""
    template, results = draft["template"], []
    reference = None
    for arg, answer in draft["args"].items():
        ids = tok(template.replace("{arg}", arg), add_special_tokens=False)["input_ids"]
        if reference is None:
            reference = ids
        problem = None
        arg_token = None
        answer_token = first_token(tok, answer)
        if len(ids) != len(reference):
            problem = "argument is not one token here"
        else:
            # Tokens that vary between arguments must all be the same one: the argument.
            spot = template.split("{arg}")[0].rstrip(" ")
            pos = len(tok(spot, add_special_tokens=False)["input_ids"])
            arg_token = ids[pos]
            if word_key(tok.decode(arg_token)) != word_key(arg):
                problem = "argument is not one token here"
        logits = next_token_logits(model, tok, template.replace("{arg}", arg))
        top = int(logits.argmax())
        if problem is None and top != answer_token:
            problem = f"model says {tok.decode(top)!r}"
        results.append({"arg": arg, "answer": answer, "arg_token": arg_token,
                        "answer_token": answer_token, "top": tok.decode(top),
                        "prob": float(logits.softmax(-1)[answer_token]), "problem": problem})
    good = [r for r in results if r["problem"] is None][:ARGS_PER_TEMPLATE]
    trials = [{"source": s["arg"], "target": t["arg"]} for s in good for t in good
              if s is not t and s["answer_token"] != t["answer_token"]
              and s["arg_token"] != t["arg_token"]]
    return {**draft, "results": results, "kept": [r["arg"] for r in good], "trials": trials}


def split_swap(templates: list[dict]) -> dict[str, list[dict]]:
    """Validation gets one template per category where possible; the rest go to test."""
    rng = random.Random(SEED)
    usable = [t for t in templates if len(t["trials"]) >= 6]
    rng.shuffle(usable)
    usable.sort(key=lambda t: -len(t["trials"]))
    usable = usable[:sum(N_SWAP.values())]
    rng.shuffle(usable)
    val, seen = [], set()
    for template in usable:
        if len(val) < N_SWAP["val"] and template["category"] not in seen:
            val.append(template)
            seen.add(template["category"])
    for template in usable:
        if len(val) < N_SWAP["val"] and template not in val:
            val.append(template)
    return {"val": val, "test": [t for t in usable if t not in val]}


# ---------------------------------------------------------------------------------------
# Review document
# ---------------------------------------------------------------------------------------

def cell(text) -> str:
    return str(text).replace("\\", "\\\\").replace("\n", "\\n").replace("|", "\\|")


def write_review(checked, pass_sets, swap_checked, swap_sets):
    lines = ["# Prompt sets: for your review", "",
             "Generated by `python -m src.check_prompts` from `src/prompt_drafts.py`. Edit that file and rerun.",
             "",
             "## What to check", "",
             "These sets are unreviewed drafts, and experiments may run on them. Edits are cheap to rerun.", "",
             "1. **Stories read positions depart from the paper.** The paper reads at the closing period; this model's readout there is mostly sentence starters and formatting. So each story gets one added token. If the story describes a thing that ` The` would ask the model to name (the animal that says meow), it ends in a pronoun, so the target isn't the obvious next word. If the target is a setting, event or abstract concept, it keeps ` The`, because the obvious next word is something else in the scene. This was a judgment per prompt, not a rank cutoff. Please check the choices (`_NAMED_BY_DESCRIPTION` in `src/prompt_drafts.py`).",
             "2. **Lens exposure.** The read formats were chosen on validation prompts, except that the first ` The` and `def` formats were found by scoring small lenses on 8 stories drafts and 14 python drafts, some of them now in test. Those scores chose a format for all prompts, not individual prompts, and the `def` format was later dropped. No lens has otherwise seen test prompts.",
             "3. **Competence rule.** A final answer counts as right if its first token is in the model's top 3. Top-1 would drop many more wiki prompts.",
             "4. **Swap vectors are scaled to unit length.** The paper doesn't say whether it normalizes.",
             "5. **Templates split, entities shared.** Validation and test swap templates are different, but can share arguments such as France.",
             "6. **Difficulty labels and control tokens** are the agent's and an automatic pick. Controls may need a look for accidental relatedness.",
             "7. **Next-token overlap remains by design.** Wiki targets are bridges, different from the asked-for answer, even when they rank high as next tokens. Python `evaluate` targets are intermediate values, different from the final answer, but their digit controls are equally plausible, so pass@10 saturates there; pass@1 is the useful number.", "",
             "## Summary", "",
             "| Domain | Pass@k drafted | Passed checks | Validation | Test | Swap templates drafted | Validation | Test | Test trials |",
             "|---|---|---|---|---|---|---|---|---|"]
    for d in DOMAINS:
        drafted = [i for i in checked if i["domain"] == d]
        swaps = [t for t in swap_checked if t["domain"] == d]
        s = swap_sets[d]
        lines.append(f"| {d} | {len(drafted)} | {sum(not i['problems'] for i in drafted)} | {len(pass_sets[d]['val'])} "
                     f"| {len(pass_sets[d]['test'])} | {len(swaps)} | {len(s['val'])} | {len(s['test'])} "
                     f"| {sum(len(t['trials']) for t in s['test'])} |")
    for d in DOMAINS:
        lines += ["", f"## {d}: pass@k", "",
                  "| Split | Type | Difficulty | Source | Prompt | Intermediate | Answer (rank) | Control |",
                  "|---|---|---|---|---|---|---|---|"]
        for split in ("val", "test"):
            for i in pass_sets[d][split]:
                answer = f"{i['answer']} ({i['answer_rank']})" if i.get("answer") else ""
                lines.append(f"| {split} | {i['type']} | {i['difficulty']} | {i['source']} | `{cell(i['prompt'])}` "
                             f"| {cell(i['intermediate'])} | {cell(answer)} | {cell(i['control'])} |")
        lines += ["", f"## {d}: swap templates", "",
                  "| Split | Category | Source | Template | Arguments → answers | Trials |", "|---|---|---|---|---|---|"]
        for split in ("val", "test"):
            for t in swap_sets[d][split]:
                pairs = ", ".join(f"{a} → {t['args'][a].strip()}" for a in t["kept"])
                lines.append(f"| {split} | {t['category']} | {t['source']} | `{cell(t['template'])}` "
                             f"| {cell(pairs)} | {len(t['trials'])} |")
    lines += ["", "## Dropped drafts", "", "| Domain | Prompt or template | Why |", "|---|---|---|"]
    for i in checked:
        if i["problems"]:
            lines.append(f"| {i['domain']} | `{cell(i['prompt'])}` | {cell('; '.join(i['problems']))} |")
    for t in swap_checked:
        bad = [f"{r['arg']}: {r['problem']}" for r in t["results"] if r["problem"]]
        if bad:
            lines.append(f"| {t['domain']} | `{cell(t['template'])}` | {cell('; '.join(bad))} |")
    (OUT / "REVIEW.md").write_text("\n".join(lines) + "\n")


def main():
    model, tok = load_model()
    OUT.mkdir(exist_ok=True)
    frequency = token_frequency(tok)

    checked = [check_pass_item(model, tok, d) for d in prompt_drafts.PASS_AT_K]
    pass_sets = {}
    for domain in DOMAINS:
        kept = [i for i in checked if i["domain"] == domain and not i["problems"]]
        pass_sets[domain] = split_pass(kept, domain)
        chosen = pass_sets[domain]["val"] + pass_sets[domain]["test"]
        if chosen:
            choose_controls(tok, chosen, frequency)

    swap_checked = [check_template(model, tok, d) for d in prompt_drafts.SWAPS]
    swap_sets = {d: split_swap([t for t in swap_checked if t["domain"] == d]) for d in DOMAINS}

    (OUT / "pass_at_k.json").write_text(json.dumps(pass_sets, indent=1, ensure_ascii=False))
    (OUT / "swap.json").write_text(json.dumps(swap_sets, indent=1, ensure_ascii=False))
    write_review(checked, pass_sets, swap_checked, swap_sets)

    for domain in DOMAINS:
        p, s = pass_sets[domain], swap_sets[domain]
        print(domain, "pass@k val/test:", len(p["val"]), len(p["test"]),
              Counter(i["difficulty"] for i in p["test"]),
              "| swap val/test templates:", len(s["val"]), len(s["test"]),
              "test trials:", sum(len(t["trials"]) for t in s["test"]))


if __name__ == "__main__":
    main()
