"""Window features: phi_i, a window's mean residual at one layer. Forward passes only.

Positions are the same valid positions the Jacobian estimator uses (src/jlens.py), so a feature
describes the activations the lens is fit on. Results are cached on /localtmp, keyed by a name.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch

from .corpus import WINDOW_TOKENS, Window
from .jlens import SKIP_FIRST, valid_positions
from .model import DATA, block_outputs

FEATURE_ROOT = Path(os.environ.get("JLENS_FEATURE_ROOT", f"/localtmp/qeq4kh/tml-j-lens/features_skip{SKIP_FIRST}"))


def mean_residuals(model, ids: torch.Tensor, layers, batch_size: int = 64) -> dict[int, torch.Tensor]:
    """ids: [N, T] token ids. Returns {layer: [N, d] float32 on CPU}."""
    device = next(model.parameters()).device
    P = valid_positions(ids.shape[1]).to(device)
    parts = {l: [] for l in layers}
    for start in range(0, len(ids), batch_size):
        acts = block_outputs(model, ids[start:start + batch_size].to(device), layers)
        for l in layers:
            parts[l].append(acts[l][:, P].float().mean(dim=1).cpu())
    return {l: torch.cat(parts[l]) for l in layers}


def cached_features(model, name: str, ids: torch.Tensor, layers) -> dict[int, torch.Tensor]:
    """mean_residuals, saved under `name`. Delete the folder if the model or windows change."""
    path = FEATURE_ROOT / f"{name}.pt"
    if path.exists():
        stored = torch.load(path, weights_only=True)
        if all(l in stored for l in layers):
            return {l: stored[l] for l in layers}
    FEATURE_ROOT.mkdir(parents=True, exist_ok=True)
    fresh = mean_residuals(model, ids, layers)
    partial = path.with_name(f"{path.name}.{os.getpid()}.partial")
    torch.save(fresh, partial)
    partial.replace(path)
    return fresh


def window_ids(windows: list[Window]) -> torch.Tensor:
    return torch.tensor([w.input_ids for w in windows])


def load_candidates(corpus: str) -> dict:
    """{"windows": [N, 128], "offsets": [N]}: every usable fit-side window of a corpus."""
    return torch.load(DATA / f"{corpus}_candidates.pt", weights_only=True)["fit"]


def candidate_windows(corpus: str, indices: list[int]) -> list[Window]:
    """Candidates as Window objects, with the same content hash as the manifest uses."""
    from .extend_splits import window_hash

    c = load_candidates(corpus)
    out = []
    for i in indices:
        ids = c["windows"][i].tolist()
        assert len(ids) == WINDOW_TOKENS
        out.append(Window(corpus, "candidate", i, tuple(ids), f"{corpus}/candidates", int(c["offsets"][i]), window_hash(ids)))
    return out
