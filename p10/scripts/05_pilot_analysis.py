"""Pilot analysis: does the fitting corpus change the J-lens beyond sampling noise?

Lenses (per source layer, all built from the stored per-window Jacobians of script 04):
  logit   : J = I
  W_A, W_B: WikiText fit windows, two disjoint halves (40 each)
  T_A, T_B: TinyStories fit windows, two disjoint halves (40 each)
  Pool    : uniform pool of the first 20 of W_A + first 20 of T_A (matched 40-window budget)

Measurements:
  1. Matrix distance: ||J_WA - J_TA||_F (cross-corpus) vs ||J_WA - J_WB||_F and
     ||J_TA - J_TB||_F (within-corpus sampling noise), relative to ||J||_F.
  2. Readout agreement on held-out VAL windows of each domain (positions 16..126):
     top-1 agreement and top-10 overlap, cross-corpus pair vs within-corpus pairs.
  3. Derivative-prediction error: how well each lens predicts the per-window Jacobians
     of VAL windows from each domain: mean_j ||J_lens - J_j||_F^2 / ||J_j||_F^2.
  4. Controlled-token retrieval (DEVELOPMENT ONLY): the repo's multihop prompts; does the
     known intermediate (e.g. "Brazil") appear in a lens's top-k at the last prompt position?
  5. Qualitative readout table (viewer prototype) for one position per domain.

Outputs: results/pilot/*.csv|json|md, figures/fig2_*.png, figures/fig3_*.png
"""
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import P10, SKIP_FIRST, decode, load_manifest, load_model, residuals, window_ids  # noqa: E402
from jacobian import weighted_lens  # noqa: E402

PD = P10 / "results" / "pilot"
FIG = P10 / "figures"
N_EVAL_WIN = 10
model, tok = load_model()
L = model.config.num_hidden_layers
SRC = list(range(L - 1))
man = load_manifest()

per = {(c, r): torch.load(PD / f"perwindow_{c}_{r}.pt")["J"]
       for c in ["wikitext", "tinystories"] for r in ["fitA", "fitB", "val"]}
nA = per[("wikitext", "fitA")][0].shape[0]
half = nA // 2
lenses = {
    "W_A": weighted_lens(per[("wikitext", "fitA")]),
    "W_B": weighted_lens(per[("wikitext", "fitB")]),
    "T_A": weighted_lens(per[("tinystories", "fitA")]),
    "T_B": weighted_lens(per[("tinystories", "fitB")]),
    "Pool": weighted_lens({l: torch.cat([per[("wikitext", "fitA")][l][:half],
                                         per[("tinystories", "fitA")][l][:half]]) for l in SRC}),
}
torch.save({k: {l: v.half() for l, v in d.items()} for k, d in lenses.items()}, PD / "lenses_pilot.pt")
eye = torch.eye(model.config.hidden_size)

# ---------- 1. matrix distances
dist_rows = []
for l in SRC:
    def rel(a, b):
        return ((lenses[a][l] - lenses[b][l]).norm() / (0.5 * (lenses[a][l].norm() + lenses[b][l].norm()))).item()
    dist_rows.append(dict(layer=l, cross_WA_TA=rel("W_A", "T_A"), cross_WB_TB=rel("W_B", "T_B"),
                          within_W=rel("W_A", "W_B"), within_T=rel("T_A", "T_B"),
                          J_vs_identity=((lenses["W_A"][l] - eye).norm() / eye.norm()).item()))

# ---------- 2. readout agreement on val windows
G = {k: {l: v[l].cuda() for l in SRC} for k, v in lenses.items()}
pairs = {"cross (W_A vs T_A)": ("W_A", "T_A"), "cross (W_B vs T_B)": ("W_B", "T_B"),
         "within WikiText (W_A vs W_B)": ("W_A", "W_B"), "within TinyStories (T_A vs T_B)": ("T_A", "T_B"),
         "logit vs W_A": ("logit", "W_A")}
agree = {(d, p, l): [0.0, 0.0] for d in ["wikitext", "tinystories"] for p in pairs for l in SRC}
npos = {}
for d in ["wikitext", "tinystories"]:
    ids = torch.tensor(window_ids(man, d, "val")[:N_EVAL_WIN], device="cuda")
    n = 0
    for b in range(ids.shape[0]):
        res = residuals(model, ids[b:b + 1], SRC)
        pos = torch.arange(SKIP_FIRST, ids.shape[1] - 1, device="cuda")
        for l in SRC:
            h = res[l][0, pos]
            top = {"logit": decode(model, h).topk(10).indices}
            for k in lenses:
                top[k] = decode(model, h @ G[k][l].T).topk(10).indices
            for p, (a, c) in pairs.items():
                agree[(d, p, l)][0] += (top[a][:, 0] == top[c][:, 0]).float().sum().item()
                ov = (top[a][:, :, None] == top[c][:, None, :]).any(-1).float().sum(-1) / 10
                agree[(d, p, l)][1] += ov.sum().item()
        n += len(pos)
    npos[d] = n
agree_rows = [dict(eval_domain=d, pair=p, layer=l, top1_agreement=v[0] / npos[d], top10_overlap=v[1] / npos[d])
              for (d, p, l), v in agree.items()]

# ---------- 3. derivative-prediction error on val-window Jacobians
dpe_rows = []
for d in ["wikitext", "tinystories"]:
    Jv = per[(d, "val")]
    for l in SRC:
        tgt = Jv[l].float()                       # [n, d, d]
        den = tgt.flatten(1).norm(dim=1) ** 2
        row = dict(eval_domain=d, layer=l)
        for k in ["W_A", "T_A", "Pool"]:
            row[k] = (((lenses[k][l][None] - tgt).flatten(1).norm(dim=1) ** 2) / den).mean().item()
        row["logit(I)"] = (((eye[None] - tgt).flatten(1).norm(dim=1) ** 2) / den).mean().item()
        dpe_rows.append(row)

# ---------- 4. controlled-token retrieval on repo multihop prompts (dev only)
items = json.loads((P10 / "notes/raw/jlens_ref/data/evaluations/lens-eval-multihop.json").read_text())["items"]


def single_tok(word):
    c = [tok(" " + word, add_special_tokens=False)["input_ids"], tok(word, add_special_tokens=False)["input_ids"]]
    return [x[0] for x in c if len(x) == 1]


ranks = {k: [] for k in ["logit", *lenses]}
kept, model_correct = 0, 0
for it in items:
    tids = single_tok(it["intermediates"][0])
    if not tids:
        continue
    kept += 1
    ids = torch.tensor([tok(it["prompt"], add_special_tokens=False)["input_ids"]], device="cuda")
    res = residuals(model, ids, SRC)
    ans = single_tok(it["target"])
    model_correct += int(bool(ans) and res["logits"][0, -1].topk(10).indices.tolist().count(ans[0]) > 0)
    for k in ranks:
        best = 10 ** 9
        for l in SRC:
            h = res[l][0, -1]
            lg = decode(model, h if k == "logit" else h @ G[k][l].T)
            r = min(int((lg > lg[t]).sum().item()) for t in tids)
            best = min(best, r)
        ranks[k].append(best)
ks = [1, 5, 10, 50, 100]
retr = {k: {f"pass@{kk}": float(np.mean([r < kk for r in v])) for kk in ks} for k, v in ranks.items()}
retr["_n_items_single_token_intermediate"] = kept
retr["_n_model_answer_in_top10"] = model_correct

# ---------- 5. qualitative viewer table
qual = []
for d in ["wikitext", "tinystories"]:
    ids = torch.tensor([window_ids(man, d, "val")[1]], device="cuda")
    res = residuals(model, ids, SRC)
    p = 70
    qual.append(f"### {d}: `...{tok.decode(ids[0, p - 15:p + 1])!r}` -> next `{tok.decode(ids[0, p + 1])!r}`\n")
    qual.append("| layer | logit lens | J-lens (WikiText fit) | J-lens (TinyStories fit) |\n|---|---|---|---|")
    for l in [6, 9, 12, 15, 18, 21]:
        h = res[l][0, p]
        cells = [decode(model, h).topk(5).indices] + [decode(model, h @ G[k][l].T).topk(5).indices for k in ["W_A", "T_A"]]
        qual.append(f"| {l} | " + " | ".join(" · ".join(repr(tok.decode([t]))[1:-1] for t in c) for c in cells) + " |")
    qual.append("")

# ---------- save
def wcsv(name, rows):
    with open(PD / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


wcsv("matrix_distance.csv", dist_rows)
wcsv("readout_agreement.csv", agree_rows)
wcsv("derivative_prediction_error.csv", dpe_rows)
(PD / "multihop_retrieval_dev.json").write_text(json.dumps(retr, indent=2))
(PD / "readout_examples.md").write_text("# Lens readouts, same position, three lenses (top-5)\n\n" + "\n".join(qual))

# ---------- figures
COL = {"cross": "#c23b22", "W": "#1f6fb4", "T": "#d1621b", "logit": "#777777"}
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
x = [r["layer"] for r in dist_rows]
ax[0].plot(x, [r["cross_WA_TA"] for r in dist_rows], "o-", ms=3, c=COL["cross"], label="cross-corpus (W_A vs T_A)")
ax[0].plot(x, [r["cross_WB_TB"] for r in dist_rows], "o--", ms=3, c=COL["cross"], alpha=.5, label="cross-corpus (W_B vs T_B)")
ax[0].plot(x, [r["within_W"] for r in dist_rows], "s-", ms=3, c=COL["W"], label="within WikiText (W_A vs W_B)")
ax[0].plot(x, [r["within_T"] for r in dist_rows], "s-", ms=3, c=COL["T"], label="within TinyStories (T_A vs T_B)")
ax[0].set(title="Lens matrices: relative Frobenius distance", xlabel="source layer", ylabel="||J1-J2|| / mean||J||")
for i, d in enumerate(["wikitext", "tinystories"]):
    a = ax[i + 1]
    for p, c, ls in [("cross (W_A vs T_A)", COL["cross"], "-"), ("cross (W_B vs T_B)", COL["cross"], "--"),
                     ("within WikiText (W_A vs W_B)", COL["W"], "-"), ("within TinyStories (T_A vs T_B)", COL["T"], "-"),
                     ("logit vs W_A", COL["logit"], ":")]:
        r = [z for z in agree_rows if z["eval_domain"] == d and z["pair"] == p]
        a.plot([z["layer"] for z in r], [z["top1_agreement"] for z in r], marker="o", ms=3, ls=ls, c=c,
               alpha=.5 if ls == "--" else 1, label=p)
    a.set(title=f"Top-1 readout agreement on held-out {d} windows", xlabel="source layer", ylabel="top-1 agreement", ylim=(0, 1.02))
for a in ax:
    a.grid(alpha=.3)
ax[0].legend(fontsize=7)
ax[2].legend(fontsize=7, loc="lower right")
fig.suptitle(f"Qwen2.5-0.5B J-lens: fitting-corpus effect vs. sampling noise ({nA} windows/lens, 128 tok; eval {N_EVAL_WIN} val windows/domain)", fontsize=10)
fig.tight_layout()
fig.savefig(FIG / "fig2_corpus_effect_vs_sampling_noise.png", dpi=160)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for i, d in enumerate(["wikitext", "tinystories"]):
    r = [z for z in dpe_rows if z["eval_domain"] == d]
    for k, c in [("W_A", COL["W"]), ("T_A", COL["T"]), ("Pool", "#5a9e48"), ("logit(I)", COL["logit"])]:
        ax[i].plot([z["layer"] for z in r], [z[k] for z in r], "o-", ms=3, c=c, label=f"lens fit: {k}")
    ax[i].set(title=f"Derivative-prediction error on {d} val windows", xlabel="source layer",
              ylabel="mean ||J_lens - J_window||^2 / ||J_window||^2", yscale="log")
    ax[i].grid(alpha=.3)
    ax[i].legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "fig3_derivative_prediction_error.png", dpi=160)
print(json.dumps(retr, indent=1))
for r in dist_rows[::3]:
    print(r)
