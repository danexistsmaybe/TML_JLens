"""Slide-ready headline figure from the pilot CSVs (no GPU). -> figures/fig4_headline.png"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

P10 = Path(__file__).resolve().parents[1]
PD = P10 / "results" / "pilot"
ag = list(csv.DictReader(open(PD / "readout_agreement.csv")))
dpe = list(csv.DictReader(open(PD / "derivative_prediction_error.csv")))
LAYER = 12

fig, ax = plt.subplots(1, 2, figsize=(12, 4.3), gridspec_kw={"width_ratios": [1.5, 1]})
series = [("within-corpus  (same corpus, different samples)", ["within WikiText (W_A vs W_B)", "within TinyStories (T_A vs T_B)"], "#1f6fb4", "-"),
          ("cross-corpus  (WikiText fit vs TinyStories fit)", ["cross (W_A vs T_A)", "cross (W_B vs T_B)"], "#c23b22", "-"),
          ("logit lens vs J-lens", ["logit vs W_A"], "#888888", ":")]
layers = sorted({int(r["layer"]) for r in ag})
for name, pairs, c, ls in series:
    y = [np.mean([float(r["top1_agreement"]) for r in ag if r["pair"] in pairs and int(r["layer"]) == l]) for l in layers]
    ax[0].plot(layers, y, marker="o", ms=3.5, c=c, ls=ls, lw=2, label=name)
ax[0].axvspan(9, 22, color="#f2e6c9", alpha=.45, lw=0)
ax[0].text(9.3, 0.04, "paper's 'workspace' band (~38-92% depth)", fontsize=8, color="#7a6420")
ax[0].set(xlabel="source layer of Qwen2.5-0.5B", ylabel="top-1 readout agreement", ylim=(0, 1.02),
          title="Same activation, different fitting corpus -> different readout")
ax[0].legend(fontsize=8.5, loc="center left")
ax[0].grid(alpha=.3)

lenses = [("WikiText-fit", "W_A", "#1f6fb4"), ("TinyStories-fit", "T_A", "#d1621b"), ("Uniform pool", "Pool", "#5a9e48"), ("Logit lens (J=I)", "logit(I)", "#888888")]
doms = ["wikitext", "tinystories"]
w = 0.2
for i, (lab, k, c) in enumerate(lenses):
    vals = [float(next(r for r in dpe if r["eval_domain"] == d and int(r["layer"]) == LAYER)[k]) for d in doms]
    ax[1].bar(np.arange(2) + (i - 1.5) * w, vals, w, color=c, label=lab)
ax[1].set_xticks(range(2), ["WikiText prompts", "TinyStories prompts"])
ax[1].set(ylabel="relative Jacobian error (lower = better)", title=f"Predicting the true local Jacobian, layer {LAYER}")
ax[1].legend(fontsize=8)
ax[1].grid(alpha=.3, axis="y")
fig.suptitle("Pilot: Jacobian-lens readouts depend on the fitting corpus far beyond sampling noise "
             "(40 windows x 128 tokens per lens; held-out windows)", fontsize=10)
fig.tight_layout()
fig.savefig(P10 / "figures" / "fig4_headline.png", dpi=170)
