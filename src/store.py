"""Glue between corpora and the lens: compute each window's Jacobians once, cache them on
disk, and average them into a lens under any weighting.

    windows ──(window_jacobians, cached here)──> J^(w) per window ──(weights)──> JLens

Cache layout: JACOBIAN_ROOT/<window sha256>.pt = {layer: [d, d] float16}. One file per
window holds every layer that has been computed for it (~1.6 MB per layer). The cache is
keyed by window content, so a window fit once is reused by every lens that includes it.
The folder name records SKIP_FIRST, so changing it starts a new cache. If the model or the
estimator changes, delete the folder.

The cache lives on /localtmp, this machine's local disk, because it reaches hundreds of GB.
Nothing there is backed up, but every file can be recomputed. Set JLENS_JACOBIAN_ROOT to
put it elsewhere.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import torch

from .corpus import Window
from .jlens import SKIP_FIRST, JLens, window_jacobians

JACOBIAN_ROOT = Path(os.environ.get(
    "JLENS_JACOBIAN_ROOT", f"/localtmp/qeq4kh/tml-j-lens/jacobians_skip{SKIP_FIRST}"))


class JacobianStore:
    def __init__(self, model, root: Path = JACOBIAN_ROOT, rows_per_pass: int = 32, log=print):
        self.model = model
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.rows_per_pass = rows_per_pass
        self.log = log

    def path(self, window: Window) -> Path:
        return self.root / f"{window.sha256}.pt"

    def get(self, window: Window, layers) -> dict[int, torch.Tensor]:
        """{layer: [d, d]} for one window, computed (and cached) if missing."""
        layers = sorted(layers)
        path = self.path(window)
        stored = torch.load(path, weights_only=True) if path.exists() else {}
        missing = [l for l in layers if l not in stored]
        if missing:
            device = next(self.model.parameters()).device
            fresh = window_jacobians(self.model, window.ids_tensor(device), missing,
                                     rows_per_pass=self.rows_per_pass)
            stored.update({l: J.half() for l, J in fresh.items()})
            torch.save(stored, path)
        return {l: stored[l] for l in layers}

    def iterate(self, windows: list[Window], layers):
        """Yield each window's {layer: [d, d]} in order, logging progress for slow (uncached) ones."""
        t0, n_new = time.perf_counter(), 0
        for i, window in enumerate(windows):
            cached = self.path(window).exists()
            yield self.get(window, layers)
            if not cached:
                n_new += 1
                self.log(f"  window {i + 1}/{len(windows)} ({window.corpus}) computed, "
                         f"{(time.perf_counter() - t0) / n_new:.1f}s per new window")


def fit_lens(model, windows: list[Window], layers, weights: torch.Tensor | None = None,
             store: JacobianStore | None = None, info: dict | None = None) -> JLens:
    """Fit a J-lens on `windows` with per-window `weights` (None = uniform)."""
    store = store or JacobianStore(model)
    info = {"n_windows": len(windows),
            "corpora": sorted({w.corpus for w in windows}),
            "weighted": weights is not None, **(info or {})}
    return JLens.from_windows(store.iterate(windows, layers), weights=weights, info=info)
