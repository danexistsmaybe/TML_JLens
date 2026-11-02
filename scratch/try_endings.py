"""Compare read-position endings on VALIDATION prompts only (and model-only overlap on all).

Uses the 30-window lens of each prompt's domain. Test prompts never meet a lens here.

    .venv/bin/python scratch/try_endings.py
"""
import json
import sys

import torch

sys.path.insert(0, ".")
from src.corpus import load_windows
from src.eval_prompts import BAND, CORPUS, word_tokens
from src.jlens import JLens
from src.model import load_model, residuals
from src.store import fit_lens

model, tok = load_model()
sets = json.load(open("data/prompts/pass_at_k.json"))
items_by_domain = {d: sets[d]["val"] for d in sets}
logit = JLens.identity(BAND, model.config.hidden_size)
lenses = {d: fit_lens(model, load_windows(CORPUS[d], "fit", n=30), layers=BAND) for d in sets}


@torch.no_grad()
def score(item, text, lens):
    ids = tok(text, return_tensors="pt").input_ids.cuda()
    acts, logits = residuals(model, ids, BAND)
    next_logits = logits[0, -1]
    t = torch.tensor(word_tokens(tok, item["intermediate"])).cuda()
    c = torch.tensor(word_tokens(tok, item["control"])).cuda()
    best = {"t": 10**9, "c": 10**9}
    l12 = l18 = None
    for l in BAND:
        lg = lens.logits(model, acts[l][0, -1], l)
        rt = int((lg > lg[t].max()).sum())
        rc = int((lg > lg[c].max()).sum())
        best["t"], best["c"] = min(best["t"], rt), min(best["c"], rc)
        if l == 12:
            l12 = rt
        if l == 18:
            l18 = rt
    next_rank = int((next_logits > next_logits[t].max()).sum())
    return best["t"], best["c"], l12, l18, next_rank


def summarize(name, rows):
    n = len(rows)
    print(f"  {name:28s} pass@10 target {sum(r[0] < 10 for r in rows)}/{n}, control {sum(r[1] < 10 for r in rows)}/{n}"
          f" | pass@50 {sum(r[0] < 50 for r in rows)}/{n}, control {sum(r[1] < 50 for r in rows)}/{n}"
          f" | next-token top10 {sum(r[4] < 10 for r in rows)}/{n}"
          f" | L12 top10 {sum(r[2] < 10 for r in rows)}, L18 top10 {sum(r[3] < 10 for r in rows)}"
          f" | median best rank {sorted(r[0] for r in rows)[n // 2]}")


def run(domain, name, fn, items=None):
    items = items if items is not None else items_by_domain[domain]
    out = {}
    for lname, lens in (("own lens", lenses[domain]), ("logit", logit)):
        rows = [score(i, fn(i), lens) for i in items]
        summarize(f"{name} [{lname}]", rows)
        out[lname] = rows
    return out


sys.modules[__name__].__dict__["run"] = run


SUBJECT = {"soccer": " They", "queen": " She", "birthday": " She", "wedding": " They", "surprise": " He",
           "gift": " They", "bus": " They", "camping": " They", "rain": " He"}
NONNOUN = {"lie"}

print("STORIES (validation)")
stories = items_by_domain["stories"]
base = lambda i: i["prompt"][:-len(" The")]
run("stories", "The (current)", lambda i: i["prompt"])
run("stories", "period", base)
run("stories", "pronoun", lambda i: base(i) + (" The" if i["intermediate"] in NONNOUN else SUBJECT[i["intermediate"]]))
run("stories", "pron + was", lambda i: base(i) + (" The" if i["intermediate"] in NONNOUN else SUBJECT[i["intermediate"]] + " were" if SUBJECT.get(i["intermediate"]) == " They" else SUBJECT.get(i["intermediate"], " The") + " was"))
run("stories", "newline Q", lambda i: base(i) + "\n\nQ: What is this story about?\nA:")
run("stories", "Then", lambda i: base(i) + " Then")
run("stories", "open quote", lambda i: base(i) + ' "')
run("stories", "and", lambda i: base(i) + " And")

print("PYTHON implied (validation)")
imp = [i for i in items_by_domain["python"] if i["type"] == "implied"]
pb = lambda i: i["prompt"][:-len("\ndef")]
for nm, end in (("def (current)", "\ndef"), ("end of code", ""), ("result = f(", "\nresult = f("), ("f(", "\nf("), ("print(", "\nprint("), ("x =", "\nx ="), ("if", "\nif"), ("return", "\nreturn")):
    print(" ", repr(nm))
    run("python", nm, lambda i, end=end: pb(i) + end, items=imp)
print("PYTHON evaluate (validation)")
ev = [i for i in items_by_domain["python"] if i["type"] == "evaluate"]
run("python", "as is", lambda i: i["prompt"], items=ev)
