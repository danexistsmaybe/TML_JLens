"""Scoring a lens on the prompt sets: pass@k and swap success.

    .venv/bin/python -m src.eval_prompts     # smoke test on VALIDATION prompts only

Pass@k: the best rank of the intermediate in the lens readout at the last position, over a
band of layers. A word counts in any casing, with or without a leading space. The control
token is scored the same way.

Swap (paper, "clamped lens-coordinate swap"): with V = [v_s v_t], c = pinv(V) h is read from
a clean forward pass, and at every position and every layer in the band the coordinates are
set to c + alpha (swap(c) - c). The trial passes if the target argument's answer becomes the
model's top next token.

v_c is J_l^T w_c scaled to unit length, where w_c is token c's row of the output embedding.
The paper doesn't say whether it normalizes. We do, so that swapping coordinates treats the
two tokens alike.
"""
from __future__ import annotations

import json

import torch

from .check_prompts import DOMAINS, OUT, word_key
from .corpus import load_windows
from .jlens import JLens
from .model import load_model, residuals
from .store import fit_lens

BAND = list(range(9, 23))  # the paper's workspace band (38-92% of depth), scaled to 24 blocks
KS = (1, 5, 10, 50)
CORPUS = {"wiki": "wikitext", "stories": "tinystories", "python": "python"}
SMOKE_WINDOWS = 30


def word_tokens(tok, word: str) -> list[int]:
    """Every single-token spelling of a word: casing, and with or without a leading space."""
    forms = {word, word.lower(), word.capitalize(), word.upper()}
    ids = set()
    for form in forms:
        for text in (form, " " + form):
            encoded = tok(text, add_special_tokens=False)["input_ids"]
            if len(encoded) == 1 and word_key(tok.decode(encoded[0])) == word_key(word):
                ids.add(encoded[0])
    return sorted(ids)


@torch.no_grad()
def best_ranks(model, tok, prompt: str, lens: JLens, words: dict[str, str], layers=BAND) -> dict[str, int]:
    """{name: best 0-based rank of that word in the readout at the last position, over layers}."""
    device = next(model.parameters()).device
    ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    acts, _ = residuals(model, ids, layers)
    token_sets = {name: torch.tensor(word_tokens(tok, w), device=device) for name, w in words.items()}
    best = {name: 10**9 for name in words}
    for layer in layers:
        logits = lens.logits(model, acts[layer][0, -1], layer)
        order = logits.argsort(descending=True)
        rank = torch.empty_like(order)
        rank[order] = torch.arange(len(order), device=device)
        for name, tokens in token_sets.items():
            best[name] = min(best[name], int(rank[tokens].min()))
    return best


def score_pass(model, tok, items: list[dict], lens: JLens, layers=BAND) -> dict:
    ranks = [best_ranks(model, tok, i["prompt"], lens, {"target": i["intermediate"], "control": i["control"]}, layers)
             for i in items]
    return {f"pass@{k}": {"target": sum(r["target"] < k for r in ranks) / len(ranks),
                          "control": sum(r["control"] < k for r in ranks) / len(ranks)} for k in KS}


def lens_vectors(model, lens: JLens, layer: int, tokens: list[int]) -> torch.Tensor:
    """[d, n] unit-length lens vectors, one column per token."""
    w = model.lm_head.weight[tokens].float()
    J = lens.jacobians[layer].to(w.device)
    v = w @ J  # row-vector form of J^T w
    return (v / v.norm(dim=-1, keepdim=True)).T


@torch.no_grad()
def swapped_logits(model, tok, prompt: str, lens: JLens, source: int, target: int, alpha: float, layers=BAND):
    """Next-token logits at the last position, under a clamped lens-coordinate swap.

    The coordinates are read from a clean forward pass and written at every layer. Reading
    them during the edited pass would be wrong: the residual stream carries each edit to the
    next layer, so the next layer's swap would undo it.
    """
    device = next(model.parameters()).device
    ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    clean, _ = residuals(model, ids, layers)

    def make_hook(layer):
        V = lens_vectors(model, lens, layer, [source, target])
        V_pinv = torch.linalg.pinv(V)
        c_clean = clean[layer].float() @ V_pinv.T  # [1, T, 2]
        wanted = c_clean + alpha * (c_clean.flip(-1) - c_clean)

        def hook(module, inputs, output):
            h = output if torch.is_tensor(output) else output[0]
            delta = (wanted - h.float() @ V_pinv.T) @ V.T  # [1, T, d]
            edited = h + delta.to(h.dtype)
            return edited if torch.is_tensor(output) else (edited, *output[1:])
        return hook

    handles = [model.model.layers[l].register_forward_hook(make_hook(l)) for l in layers]
    try:
        return model(input_ids=ids).logits[0, -1].float()
    finally:
        for handle in handles:
            handle.remove()


def run_swap_trial(model, tok, template: dict, trial: dict, lens: JLens, alpha: float, layers=BAND) -> dict:
    """Swap the prompt's argument (source) for another (target) and read the output."""
    args = {r["arg"]: r for r in template["results"]}
    source, target = args[trial["source"]], args[trial["target"]]
    prompt = template["template"].replace("{arg}", trial["source"])
    logits = swapped_logits(model, tok, prompt, lens, source["arg_token"], target["arg_token"], alpha, layers)
    logp = logits.log_softmax(-1)
    plain = model(input_ids=tok(prompt, return_tensors="pt").input_ids.to(logits.device)).logits[0, -1].float().log_softmax(-1)
    t, s = target["answer_token"], source["answer_token"]
    return {"pass": int(logits.argmax()) == t,
            "target_rank": int((logits > logits[t]).sum()),
            "logp_shift": float((logp[t] - logp[s]) - (plain[t] - plain[s]))}


def score_swap(model, tok, templates: list[dict], lens: JLens, alpha: float, layers=BAND) -> dict:
    results = [run_swap_trial(model, tok, t, trial, lens, alpha, layers) for t in templates for trial in t["trials"]]
    return {"trials": len(results), "success": sum(r["pass"] for r in results) / len(results),
            "mean_logp_shift": sum(r["logp_shift"] for r in results) / len(results)}


def main():
    """Fit a small lens on each domain and score the VALIDATION prompts. Test stays untouched."""
    model, tok = load_model()
    pass_sets = json.loads((OUT / "pass_at_k.json").read_text())
    swap_sets = json.loads((OUT / "swap.json").read_text())
    lenses = {"logit lens": JLens.identity(BAND, model.config.hidden_size)}
    for domain in DOMAINS:
        lenses[f"{domain} ({SMOKE_WINDOWS} windows)"] = fit_lens(
            model, load_windows(CORPUS[domain], "fit", n=SMOKE_WINDOWS), layers=list(range(23)))
    results = {}
    for domain in DOMAINS:
        for name, lens in lenses.items():
            entry = {"pass": score_pass(model, tok, pass_sets[domain]["val"], lens),
                     "swap_alpha_1": score_swap(model, tok, swap_sets[domain]["val"], lens, 1.0),
                     "swap_alpha_2": score_swap(model, tok, swap_sets[domain]["val"], lens, 2.0)}
            results.setdefault(domain, {})[name] = entry
            print(domain, "|", name, "| pass@10 target/control:", entry["pass"]["pass@10"]["target"],
                  entry["pass"]["pass@10"]["control"], "| swap a=1:", round(entry["swap_alpha_1"]["success"], 3),
                  "a=2:", round(entry["swap_alpha_2"]["success"], 3),
                  "| shift a=1:", round(entry["swap_alpha_1"]["mean_logp_shift"], 2), flush=True)
    (OUT / "smoke_validation.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
