"""Our methods: similarity scores, similarity weighting, top-n selection, and their controls.

Each window i has a feature phi_i (src/features.py). A domain has a centroid mu, the mean feature
of half its validation windows. The score is the cosine of phi_i and mu, after subtracting the
pool's mean feature from both, so the score measures what makes a window differ from the pool.

    Similarity weighting: w_i proportional to exp(score_i / tau). We set tau through the
                          effective sample size, ESS = (sum w)^2 / sum w^2.
    Top-n selection:      equal weight on the n highest-scoring windows, so ESS = n.

Controls keep the same shape while ignoring the score:
    Random weights: the similarity weights shuffled among windows, so the ESS matches exactly.
    Random n:       equal weight on n random windows.
"""
from __future__ import annotations

import torch


def scores(features: torch.Tensor, mu: torch.Tensor, center: torch.Tensor | None) -> torch.Tensor:
    """Cosine of each feature with mu. features: [N, d]; mu and center: [d]. center=None: uncentered."""
    if center is not None:
        features, mu = features - center, mu - center
    return torch.nn.functional.cosine_similarity(features, mu[None], dim=-1)


def ess(weights: torch.Tensor) -> float:
    return float(weights.sum() ** 2 / (weights ** 2).sum())


def exp_weights(score: torch.Tensor, tau: float) -> torch.Tensor:
    """exp(score / tau), scaled to sum to 1, computed so that small tau doesn't overflow."""
    return torch.softmax(score.double() / tau, dim=0)


def similarity_weights(score: torch.Tensor, target_ess: float, tolerance: float = 1e-3) -> torch.Tensor:
    """exp(score / tau) with tau found by bisection, so that the ESS hits target_ess.

    ESS rises with tau, from about 1 as tau approaches 0 to N as tau grows without limit.
    """
    n = len(score)
    if not 1 <= target_ess <= n:
        raise ValueError(f"target ESS {target_ess} is outside [1, {n}]")
    if target_ess == n:
        return torch.full((n,), 1.0 / n, dtype=torch.float64)
    low, high = torch.tensor(1e-6).log(), torch.tensor(1e6).log()
    for _ in range(200):
        mid = (low + high) / 2
        value = ess(exp_weights(score, float(mid.exp())))
        if abs(value - target_ess) < tolerance * target_ess:
            break
        if value < target_ess:
            low = mid
        else:
            high = mid
    return exp_weights(score, float(mid.exp()))


def top_n_weights(score: torch.Tensor, n: int) -> torch.Tensor:
    weights = torch.zeros(len(score), dtype=torch.float64)
    weights[score.topk(n).indices] = 1.0 / n
    return weights


def shuffled(weights: torch.Tensor, seed: int) -> torch.Tensor:
    """The same weights assigned to windows at random, with the same permutation for every layer."""
    g = torch.Generator().manual_seed(seed)
    return weights[torch.randperm(len(weights), generator=g)]


def random_n_weights(size: int, n: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    weights = torch.zeros(size, dtype=torch.float64)
    weights[torch.randperm(size, generator=g)[:n]] = 1.0 / n
    return weights
