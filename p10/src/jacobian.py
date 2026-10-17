"""Corpus-conditioned Jacobian-lens estimator (written for P10; reference repo used only
for reading).

For one window and each source layer l we compute

    J_l^(w) = mean_{p in valid} sum_{p' >= p, p' in valid} d h_final[p'] / d h_l[p]

(valid = positions SKIP_FIRST .. T-2), matching the paper's estimator ("cotangents summed
over target positions, then averaged over source positions"). Row i of J_l^(w) comes from
one backward pass with a one-hot cotangent e_i placed at every valid target position.
Rows are batched by replicating the window `dim_batch` times along the batch axis.

Per-window Jacobians are kept (not just their mean) so that any weighting scheme --
domain-specific, uniform pooled, similarity-weighted, random-weighted -- is a cheap
weighted average of the same stored matrices, at exactly matched compute.
"""
from __future__ import annotations

import math
import time

import torch

from common import SKIP_FIRST, n_layers, record


def valid_mask(T: int, skip_first: int = SKIP_FIRST) -> torch.Tensor:
    m = torch.zeros(T, dtype=torch.bool)
    m[skip_first : T - 1] = True
    return m


def window_jacobians(model, input_ids: torch.Tensor, source_layers, *, target_layer=None,
                     dim_batch: int = 64, skip_first: int = SKIP_FIRST,
                     per_position: bool = False):
    """input_ids: [1, T]. Returns {l: [d, d] fp32 CPU} (or, if per_position, {l: [d, T, d]}
    giving rows i of d h_final-summed / d h_l[p] for every p -- used only by tests)."""
    L = n_layers(model)
    target = L - 1 if target_layer is None else target_layer
    source_layers = sorted(source_layers)
    assert source_layers[-1] < target
    d = model.config.hidden_size
    T = input_ids.shape[1]
    vpos = valid_mask(T, skip_first).nonzero(as_tuple=True)[0].to(input_ids.device)
    out = {l: torch.zeros(d, T, d) if per_position else torch.zeros(d, d) for l in source_layers}
    n_pass = math.ceil(d / dim_batch)
    with torch.enable_grad(), record(model, [*source_layers, target], root=source_layers[0]) as acts:
        model.model(input_ids=input_ids.expand(dim_batch, -1))  # backbone only; no lm_head needed
        z = acts[target]
        srcs = [acts[l] for l in source_layers]
        b = torch.arange(dim_batch, device=z.device)
        cot = torch.zeros_like(z)
        for k, i0 in enumerate(range(0, d, dim_batch)):
            n = min(dim_batch, d - i0)
            cot.zero_()
            cot[b[:n, None], vpos[None, :], i0 + b[:n, None]] = 1.0
            grads = torch.autograd.grad(z, srcs, grad_outputs=cot, retain_graph=k < n_pass - 1)
            for l, g in zip(source_layers, grads):
                if per_position:
                    out[l][i0:i0 + n] = g[:n].float().cpu()
                else:
                    out[l][i0:i0 + n] = g[:n, vpos, :].float().mean(1).cpu()
            del grads
    return out


def fit_windows(model, windows, source_layers, *, dim_batch=64, device="cuda", log=print):
    """Per-window Jacobians for a list of token-id lists. Returns ({l: [N, d, d]} fp16, secs/window)."""
    per = {l: [] for l in source_layers}
    t0 = time.perf_counter()
    for j, ids in enumerate(windows):
        J = window_jacobians(model, torch.tensor([ids], device=device), source_layers, dim_batch=dim_batch)
        for l in source_layers:
            per[l].append(J[l].half())
        if j % 10 == 0:
            log(f"    window {j + 1}/{len(windows)}  {time.perf_counter() - t0:.1f}s elapsed")
    torch.cuda.synchronize()
    secs = (time.perf_counter() - t0) / max(len(windows), 1)
    return {l: torch.stack(v) for l, v in per.items()}, secs


def weighted_lens(per_window: dict[int, torch.Tensor], weights: torch.Tensor | None = None):
    """Weighted mean of per-window Jacobians (weights normalized to sum 1; None = uniform)."""
    out = {}
    for l, Js in per_window.items():
        w = torch.full((Js.shape[0],), 1.0 / Js.shape[0]) if weights is None else weights / weights.sum()
        out[l] = torch.einsum("n,nij->ij", w.float(), Js.float())
    return out
