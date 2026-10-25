"""Animate how a J-lens is fit, using real numbers from our implementation.

    python analysis/visualize_fit.py            # -> analysis/out/fit_process.gif (+ .png of the last frame)

Four acts, one panel each (the active panel is highlighted):
  1. FORWARD PASS. One 128-token window flows up through the 24 blocks. Heatmap of the
     residual-stream norm ||h_l[p]|| per (layer, position).
  2. ONE BACKWARD PASS. A one-hot cotangent e_i is placed on the target layer (block 23)
     at every valid position (16..126). Backprop reveals, layer by layer from the top,
     ||d s_i / d h_l[p]||: how strongly each (layer, position) influences output
     coordinate i, summed over the current and later positions (attention is causal, so
     h_l[p] can only affect targets p' >= p). Bright columns are positions the output
     depends on strongly.
  3. FILLING ONE WINDOW'S JACOBIAN. Each backward pass yields 32 rows (32 batch copies,
     one row each); averaging over source positions gives J^(w) at layer 12, one block of
     rows at a time (28 passes for d = 896).
  4. AVERAGING WINDOWS INTO THE LENS. J_12 = mean of J^(w) over the demo's 16 windows
     (8 WikiText + 8 TinyStories). The curve shows how much each new window still changes
     the lens. The bars show the demo prompt read at layer 19 through the lens so far.

Uses the per-window Jacobians cached by `python -m src.demo` (computes them if missing).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

from src import corpus  # noqa: E402
from src.demo import PRESETS  # noqa: E402
from src.jlens import SKIP_FIRST, valid_positions  # noqa: E402
from src.model import decode, load_model, record_residuals, residuals, target_layer  # noqa: E402
from src.store import JacobianStore  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
J_LAYER, READ_LAYER, ROW_I, ROWS_PER_PASS = 12, 19, 100, 32


def compute(model, tok):
    L = target_layer(model)
    windows = corpus.load_windows("wikitext", "fit", 8) + corpus.load_windows("tinystories", "fit", 8)
    ids = windows[0].ids_tensor()
    T = ids.shape[1]

    # Act 1: residual norms from a plain forward pass.
    acts, _ = residuals(model, ids, range(L + 1))
    fwd = torch.stack([acts[l][0].norm(dim=-1) for l in range(L + 1)]).cpu().numpy()  # [24, T]

    # Act 2: one backward pass for output coordinate ROW_I.
    with torch.enable_grad(), record_residuals(model, range(L + 1), grad_root=0) as a:
        model.model(input_ids=ids)
        cot = torch.zeros_like(a[L])
        cot[0, valid_positions(T).to(ids.device), ROW_I] = 1.0
        grads = torch.autograd.grad(a[L], [a[l] for l in range(L)], grad_outputs=cot)
    bwd = np.zeros((L + 1, T))
    bwd[:L] = torch.stack([g[0].norm(dim=-1) for g in grads]).cpu().numpy()
    bwd[L] = cot[0, :, ROW_I].cpu().numpy()  # the cotangent itself, on the target layer

    # Acts 3 and 4: cached per-window Jacobians.
    store = JacobianStore(model)
    per = [store.get(w, [J_LAYER, READ_LAYER]) for w in windows]
    J_first = per[0][J_LAYER].float().numpy()

    prompt, _ = PRESETS["antonym"]
    p_ids = tok(prompt, return_tensors="pt").input_ids.cuda()
    p_acts, _ = residuals(model, p_ids, [READ_LAYER])
    h = p_acts[READ_LAYER][0, -1]

    running, changes, readouts, prev = [], [], [], None
    total12 = torch.zeros_like(per[0][J_LAYER], dtype=torch.float32)
    total19 = torch.zeros_like(total12)
    for k, J in enumerate(per, start=1):
        total12 += J[J_LAYER].float()
        total19 += J[READ_LAYER].float()
        cur = total12 / k
        changes.append(np.nan if prev is None else float((cur - prev).norm() / cur.norm()))
        prev = cur
        running.append(cur.numpy())
        probs = decode(model, h @ (total19 / k).cuda().T).softmax(-1)
        p, idx = probs.topk(5)
        readouts.append([(repr(tok.decode(int(i)))[1:-1], float(q)) for i, q in zip(idx, p)])
    return dict(fwd=fwd, bwd=bwd, J_first=J_first, running=running, changes=changes,
                readouts=readouts, corpora=[w.corpus for w in windows], T=T, L=L, prompt=prompt)


def animate(D):
    L, T = D["L"], D["T"]
    d = D["J_first"].shape[0]
    n_pass = int(np.ceil(d / ROWS_PER_PASS))
    acts_frames = [("fwd", l) for l in range(L + 1)] + [("fwd", L)] * 4 \
        + [("bwd", l) for l in range(L, -1, -1)] + [("bwd", 0)] * 4 \
        + [("fill", k) for k in range(1, n_pass + 1)] + [("fill", n_pass)] * 4 \
        + [("avg", k) for k in range(1, len(D["running"]) + 1)] + [("avg", len(D["running"]))] * 8

    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], width_ratios=[1.25, 1, 1], hspace=0.38, wspace=0.3)
    ax_heat = fig.add_subplot(gs[0, 0])
    ax_jw = fig.add_subplot(gs[0, 1])
    ax_jbar = fig.add_subplot(gs[0, 2])
    ax_chg = fig.add_subplot(gs[1, 0])
    ax_read = fig.add_subplot(gs[1, 1:])
    caption = fig.text(0.5, 0.965, "", ha="center", va="top", fontsize=12.5, wrap=True)

    fwd_log = np.log10(D["fwd"] + 1e-6)
    bwd_log = np.log10(D["bwd"] + 1e-6)
    vj = np.percentile(np.abs(D["J_first"]), 99)
    vbar = np.percentile(np.abs(D["running"][-1]), 99)

    def style(ax, active, title):
        ax.set_title(title, fontsize=10.5, fontweight="bold" if active else "normal",
                     color="black" if active else "0.55")
        for s in ax.spines.values():
            s.set_edgecolor("tab:red" if active else "0.75")
            s.set_linewidth(2.2 if active else 0.8)

    def draw(frame):
        act, k = frame
        for ax in (ax_heat, ax_jw, ax_jbar, ax_chg, ax_read):
            ax.clear()

        # Panel 1: forward / backward heatmap over (position, layer)
        if act in ("fwd", "bwd"):
            data = fwd_log if act == "fwd" else bwd_log
            masked = np.full_like(data, np.nan)
            if act == "fwd":
                masked[:k + 1] = data[:k + 1]
                vmin, vmax = np.percentile(fwd_log, [2, 99.5])
            else:
                masked[k:] = data[k:]
                finite = bwd_log[:L][bwd_log[:L] > -5]
                vmin, vmax = np.percentile(finite, [2, 99.5])
            ax_heat.imshow(masked, origin="lower", aspect="auto", cmap="magma" if act == "fwd" else "viridis",
                           vmin=vmin, vmax=vmax, extent=[-0.5, T - 0.5, -0.5, L + 0.5])
        else:
            ax_heat.imshow(bwd_log, origin="lower", aspect="auto", cmap="viridis", alpha=0.35,
                           vmin=-2, vmax=np.percentile(bwd_log, 99.5), extent=[-0.5, T - 0.5, -0.5, L + 0.5])
        ax_heat.axvspan(-0.5, SKIP_FIRST - 0.5, color="0.6", alpha=0.25)
        ax_heat.axvspan(T - 1.5, T - 0.5, color="0.6", alpha=0.25)
        for l, lab in [(J_LAYER, f"source l={J_LAYER}"), (L, f"target {L}")]:
            ax_heat.axhline(l, color="w", lw=0.8, ls="--")
            ax_heat.text(T - 3, l + 0.3, lab, ha="right", va="bottom", fontsize=8, color="w")
        ax_heat.set_xlabel("token position (grey = skipped: attention sinks, last token)")
        ax_heat.set_ylabel("layer (block output)")
        style(ax_heat, act in ("fwd", "bwd"),
              "1. forward: log ||h_l[p]||" if act == "fwd" else f"2. backward from e_{ROW_I}: log ||d s_{ROW_I} / d h_l[p]||")

        # Panel 2: J^(w) filling up row blocks
        J_show = np.full_like(D["J_first"], np.nan)
        rows = d if act == "avg" else (min(k * ROWS_PER_PASS, d) if act == "fill" else 0)
        J_show[:rows] = D["J_first"][:rows]
        ax_jw.imshow(J_show, cmap="RdBu_r", vmin=-vj, vmax=vj, interpolation="nearest")
        if act == "fill" and rows < d:
            ax_jw.axhspan(rows - ROWS_PER_PASS - 0.5, rows - 0.5, color="tab:red", alpha=0.25)
        ax_jw.set_xlabel("layer-12 coordinate j")
        ax_jw.set_ylabel("target coordinate i")
        style(ax_jw, act == "fill", f"3. J^(w) at layer {J_LAYER}, window 1: {rows}/{d} rows")

        # Panel 3: running lens average
        kk = k if act == "avg" else 0
        if kk:
            ax_jbar.imshow(D["running"][kk - 1], cmap="RdBu_r", vmin=-vbar, vmax=vbar, interpolation="nearest")
        else:
            ax_jbar.imshow(np.full((d, d), np.nan), cmap="RdBu_r")
        ax_jbar.set_xticks([])
        ax_jbar.set_yticks([])
        style(ax_jbar, act == "avg", f"4. lens J_{J_LAYER} = mean of {kk} window(s)")

        # Panel 4: change per added window
        if kk:
            xs = np.arange(1, kk + 1)
            cols = ["tab:blue" if c == "wikitext" else "tab:orange" for c in D["corpora"][:kk]]
            ax_chg.plot(xs, D["changes"][:kk], color="0.4", lw=1)
            ax_chg.scatter(xs, D["changes"][:kk], c=cols, zorder=3)
        ax_chg.set_xlim(0.5, len(D["running"]) + 0.5)
        ax_chg.set_ylim(0, np.nanmax(D["changes"]) * 1.1)
        ax_chg.set_xlabel("windows averaged (blue = WikiText, orange = TinyStories)")
        ax_chg.set_ylabel("||J_k - J_{k-1}|| / ||J_k||")
        style(ax_chg, act == "avg", "how much each new window moves the lens")

        # Panel 5: readout bars
        if kk:
            toks, ps = zip(*D["readouts"][kk - 1])
            ax_read.barh(range(5)[::-1], ps, color="tab:green")
            ax_read.set_yticks(range(5)[::-1], [f"'{t}'" for t in toks])
            ax_read.set_xlim(0, max(max(r[1] for r in rs) for rs in D["readouts"]) * 1.1)
        else:
            ax_read.set_yticks([])
        ax_read.set_xlabel("J-lens probability")
        style(ax_read, act == "avg", f"demo prompt read at layer {READ_LAYER} through the lens so far:  "
              + repr(D["prompt"].split("\n")[-1]))

        caption.set_text({
            "fwd": f"Forward pass: one 128-token window flows up through the blocks. Block {k} done.",
            "bwd": "Backward pass: put a 1 on coordinate i of the target layer at every valid position, then backprop.\n"
                   "The gradient reaching (layer l, position p) is row i of the bracket in src/jlens.py.",
            "fill": f"Each backward pass computes {ROWS_PER_PASS} rows at once (one per batch copy); "
                    f"averaging over valid source positions gives rows of J^(w). Pass {k}/{n_pass}.",
            "avg": f"The lens is the average of per-window Jacobians (uniform weights here). "
                   f"Window {kk}/{len(D['running'])}: {D['corpora'][kk - 1] if kk else ''}.",
        }[act])

    anim = FuncAnimation(fig, draw, frames=acts_frames, interval=250)
    OUT.mkdir(exist_ok=True)
    anim.save(OUT / "fit_process.gif", writer=PillowWriter(fps=4), dpi=80)
    draw(acts_frames[-1])
    fig.savefig(OUT / "fit_process_final.png", dpi=110)
    print(f"wrote {OUT / 'fit_process.gif'} and {OUT / 'fit_process_final.png'}")


if __name__ == "__main__":
    model, tok = load_model()
    animate(compute(model, tok))
