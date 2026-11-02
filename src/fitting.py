"""Fits many lenses from cached Jacobians in one pass over the files.

A lens is an average of per-window Jacobians, and its weights may differ by layer. Reading the
files once per lens would mean reading hundreds of GB many times, so every lens is accumulated
during a single pass. Windows must already be cached (src/compute_jacobians.py); nothing here
runs the model.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import torch

from .corpus import Window
from .jlens import JLens
from .store import JACOBIAN_ROOT


def load_jacobians(window: Window, layers) -> dict[int, torch.Tensor]:
    stored = torch.load(JACOBIAN_ROOT / f"{window.sha256}.pt", weights_only=True)
    return {l: stored[l] for l in layers}


def load_stack(windows: list[Window], layer: int, workers: int = 8) -> torch.Tensor:
    """[N, d*d] float16: one layer of many windows, for trying many weightings cheaply."""
    with ThreadPoolExecutor(workers) as pool:
        rows = list(pool.map(lambda w: load_jacobians(w, [layer])[layer].flatten(), windows))
    return torch.stack(rows)


def fit_many(windows: list[Window], specs: dict[str, dict[int, torch.Tensor]], layers,
             batch_size: int = 64, workers: int = 8, log=None) -> dict[str, JLens]:
    """Fit one lens per entry of specs.

    specs: {name: {layer: [N] weights over `windows`}}. Weights are scaled to sum to 1 per layer.
    Windows no lens uses are never read.
    """
    layers = sorted(layers)
    names = list(specs)
    weight = {l: torch.stack([specs[n][l].double() / specs[n][l].double().sum() for n in names]).float()
              for l in layers}
    used = sum((weight[l] > 0).any(dim=0) for l in layers) > 0
    order = used.nonzero().flatten().tolist()
    d = None
    totals = {}
    with ThreadPoolExecutor(workers) as pool:
        for start in range(0, len(order), batch_size):
            batch = order[start:start + batch_size]
            loaded = list(pool.map(lambda i: load_jacobians(windows[i], layers), batch))
            if d is None:
                d = loaded[0][layers[0]].shape[0]
                totals = {l: torch.zeros(len(names), d * d) for l in layers}
            for l in layers:
                stack = torch.stack([J[l].flatten() for J in loaded]).float()  # [B, d*d]
                totals[l] += weight[l][:, batch] @ stack
            if log and (start // batch_size) % 20 == 0:
                log(f"  fit_many: {start + len(batch)}/{len(order)} windows")
    return {n: JLens({l: totals[l][k].reshape(d, d) for l in layers}, info={"name": n}) for k, n in enumerate(names)}
