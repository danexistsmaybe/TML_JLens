"""How does one window's Jacobian depend on the window's length? (decides how to
combine positions inside a window when windows have mixed lengths; see .agent/DECISIONS.md)

    .venv/bin/python scratch/leveling_off.py              # compute + plot (~15 min on GPU)
    .venv/bin/python scratch/leveling_off.py --plot-only  # replot from the saved .pt

For one source token t, the estimator adds up its effect on every later token:
    S(t, T) = sum over t' in [t, T-2] of d h_target[t'] / d h_l[t]     (the "summed Jacobian")
The model is causal, so cutting a window down to its first T tokens leaves every
activation before T unchanged. Running window_jacobians(per_position=True) on longer and
longer prefixes of one window therefore gives S(t, T) for the same token t with more and
more later tokens included: h = T - 1 - t of them.

Rows of the figure (columns = the layer being nudged):
  1. Size of S(t, T) vs h, one line per token t (mean over windows).
  2. Cosine similarity between S(t, T) and the same token's S at its largest h:
     does including more later tokens change the matrix's direction, or only its size?
  3. Size of a whole window's Jacobian J^(w) when the window is cut to T tokens, relative to
     the full 128-token window, for three ways of combining positions:
       sum over targets, mean over sources   (what src/jlens.py does)
       mean over all (source, target) pairs
       mean over targets, then mean over sources
     A flat line means short and long windows count equally in the lens average.
     Solid = SKIP_FIRST 16, dashed = SKIP_FIRST 0.
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

SOURCES = (16, 40, 64, 96)          # fixed source positions tracked in rows 1-2
LAYERS = (6, 12, 18)
SKIPS = (16, 0)
LENGTHS = sorted({t + k for t in SOURCES for k in (2, 3, 5, 9)} | {24, 32, 48, 64, 80, 96, 112, 128})
DATA_OUT = REPO / "analysis" / "out" / "leveling_off.pt"
FIG_OUT = REPO / ".agent" / "figures" / "leveling_off.png"


def compute(n_per_corpus):
    from src import corpus
    from src.jlens import window_jacobians
    from src.model import load_model

    model, _ = load_model()
    windows = corpus.load_windows("wikitext", "fit", n_per_corpus) + corpus.load_windows("tinystories", "fit", n_per_corpus)
    W = len(windows)
    # norms[layer][w, i_source, i_T] (nan where the source is past the window), and the
    # summed Jacobians themselves at each T (needed for the cosine against the longest horizon)
    norms = {l: torch.full((W, len(SOURCES), len(LENGTHS)), float("nan")) for l in LAYERS}
    cos = {l: torch.full((W, len(SOURCES), len(LENGTHS)), float("nan")) for l in LAYERS}
    window_norm = {(l, s, est): torch.zeros(W, len(LENGTHS))
                   for l in LAYERS for s in SKIPS for est in ("sum", "pair", "source")}

    for w, win in enumerate(windows):
        ids = win.ids_tensor()
        summed = {}   # (layer, i_source, i_T) -> [d, d]
        for iT, T in enumerate(LENGTHS):
            # skip_first=0: targets are 0..T-2; only t' >= t contributes to source t anyway.
            per_pos = window_jacobians(model, ids[:, :T], LAYERS, skip_first=0, per_position=True)
            for l in LAYERS:
                B = per_pos[l].permute(1, 0, 2)                 # [T, d, d]: S(t, T) per source t
                for i, t in enumerate(SOURCES):
                    if t <= T - 2:
                        summed[l, i, iT] = B[t].clone()
                        norms[l][w, i, iT] = B[t].norm()
                for s in SKIPS:
                    src = torch.arange(s, T - 1)                # valid sources
                    n_targets = (T - 1 - src).float()           # targets t..T-2
                    Bs = B[src]
                    J_sum = Bs.mean(0)
                    n = len(src)
                    window_norm[l, s, "sum"][w, iT] = J_sum.norm()
                    window_norm[l, s, "pair"][w, iT] = (J_sum * 2 / (n + 1)).norm()
                    window_norm[l, s, "source"][w, iT] = (Bs / n_targets[:, None, None]).mean(0).norm()
            del per_pos
            print(f"  window {w + 1}/{W} ({win.corpus}) T={T}", flush=True)
        for l in LAYERS:
            for i in range(len(SOURCES)):
                iTs = [iT for iT in range(len(LENGTHS)) if (l, i, iT) in summed]
                ref = summed[l, i, iTs[-1]].flatten()
                for iT in iTs:
                    cos[l][w, i, iT] = torch.nn.functional.cosine_similarity(summed[l, i, iT].flatten(), ref, dim=0)
    DATA_OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"norms": norms, "cos": cos, "window_norm": window_norm, "lengths": LENGTHS,
                "sources": SOURCES, "layers": LAYERS, "windows": [w.sha256 for w in windows]}, DATA_OUT)


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    r = torch.load(DATA_OUT, weights_only=False)
    lengths = torch.tensor(r["lengths"], dtype=torch.float)
    surface, ink, muted, grid = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9"
    token_colors = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]       # one hue, light -> dark = earlier -> later token
    est_colors = {"sum": "#2a78d6", "pair": "#eb6834", "source": "#1baf7a"}
    est_labels = {"sum": "sum over targets, mean over sources (current code)",
                  "pair": "mean over all (source, target) pairs",
                  "source": "mean over targets, then mean over sources"}
    h_label = "number of later tokens included, $h$ (log scale)"
    rows = [
        ("1. How big is one token's summed Jacobian as more later tokens are included?",
         "size of $S_\\ell(t, T)$\n(Frobenius norm)"),
        ("2. Does including more later tokens change its direction, or only its size?",
         "cosine similarity to the same token's\n$S_\\ell$ with the most later tokens"),
        ("3. Does a window's contribution to the lens depend on the window's length?",
         "size of the window's Jacobian $J^{(w)}_\\ell$,\nrelative to a 128-token window (log)"),
    ]

    fig = plt.figure(figsize=(13, 12.5), facecolor=surface)
    subfigs = fig.subfigures(3, 1, hspace=0.04)
    for row, (sf, (title, ylabel)) in enumerate(zip(subfigs, rows)):
        sf.set_facecolor(surface)
        sf.suptitle(title, color=ink, x=0.01, ha="left", fontsize=12, fontweight="bold")
        axes = sf.subplots(1, len(r["layers"]))
        for c, (ax, l) in enumerate(zip(axes, r["layers"])):
            ax.set_title(f"nudging layer {l}", color=ink, fontsize=10)
            if row < 2:
                vals = r["norms" if row == 0 else "cos"][l].nanmean(0)   # mean over windows
                for i, t in enumerate(r["sources"]):
                    h = lengths - 1 - t
                    ok = ~torch.isnan(vals[i])
                    ax.plot(h[ok], vals[i][ok], color=token_colors[i], lw=2, marker="o", ms=4,
                            label=f"token at position {t}")
                ax.set_xscale("log")
                ax.set_xlabel(h_label, color=ink, fontsize=9)
                if row == 1:
                    ax.set_ylim(None, 1.01)
            else:
                for est, color in est_colors.items():
                    for s, ls in ((16, "-"), (0, "--")):
                        v = r["window_norm"][l, s, est].mean(0)
                        ok = lengths > s + 2
                        ax.plot(lengths[ok], v[ok] / v[-1], color=color, lw=2, ls=ls)
                ax.axhline(1, color=muted, lw=1)
                ax.set_yscale("log")
                ax.set_xlabel("window length $T$ (tokens)", color=ink, fontsize=9)
            if c == 0:
                ax.set_ylabel(ylabel, color=ink, fontsize=9)
            ax.set_facecolor(surface)
            ax.grid(color=grid, lw=0.6)
            ax.tick_params(colors=muted, labelsize=8)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color("#c3c2b7")
        if row == 0:
            axes[-1].legend(frameon=False, fontsize=8, title="source token", title_fontsize=8)
        if row == 2:
            handles = [Line2D([], [], color=c_, lw=2, label=est_labels[e]) for e, c_ in est_colors.items()]
            handles += [Line2D([], [], color=muted, lw=2, ls="-", label="SKIP_FIRST = 16"),
                        Line2D([], [], color=muted, lw=2, ls="--", label="SKIP_FIRST = 0")]
            sf.legend(handles=handles, frameon=False, fontsize=8, loc="lower center", ncol=3)
            sf.subplots_adjust(bottom=0.27)
        if row == 0:
            sf.text(0.99, 0.985, f"Qwen2.5-0.5B, mean over {len(r['windows'])} windows (2 WikiText, 2 TinyStories)",
                    ha="right", va="top", color=muted, fontsize=8)
    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_OUT, dpi=130, facecolor=surface)
    print(f"wrote {FIG_OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--n-per-corpus", type=int, default=2)
    args = ap.parse_args()
    if not args.plot_only:
        compute(args.n_per_corpus)
    plot()
