"""The Jacobian lens (J-lens): what it is, how one window's Jacobian is computed, and how
per-window Jacobians are averaged into a lens.

THE IDEA
--------
The logit lens reads layer l by decoding h_l as if it were the last layer:
    logit_lens(h_l) = decode(h_l).
That fails in early/middle layers because they don't use the final layer's "coordinates".
The J-lens first maps h_l into final-layer coordinates with a d x d matrix J_l:
    jlens(h_l) = decode(J_l @ h_l),
where J_l is the AVERAGE Jacobian of the target layer with respect to layer l:
    J_l = E[ d h_target / d h_l ]   (averaged over a fitting corpus).
With J_l = I you get the logit lens back. Since J_l is an average over a corpus, the
choice of corpus (and how its windows are weighted) can change J_l. That is P10's question.

THE ESTIMATOR (for one 128-token window w)
------------------------------------------
Let P be the "valid" positions (we skip the first SKIP_FIRST positions, whose activations
are atypical attention sinks, and the last position). Then

    J_l^(w) = mean over p in P of  [ sum over p' in P, p' >= p of  d h_target[p'] / d h_l[p] ]

  * sum over targets p' >= p: a change at position p can affect the current position and
    every later one (through attention), so the lens captures "things the model may say
    now or later". Targets before p are unaffected (causal attention) and contribute 0.
  * mean over sources p: one matrix per window, not one per position.
A lens averages J_l^(w) over windows, optionally with weights (see JLens.from_windows).

THE COMPUTATION TRICK
---------------------
A d x d Jacobian has d rows; row i is the gradient of output coordinate i. Define the
scalar
    s_i = sum over p' in P of h_target[p', i].
Its gradient with respect to h_l[p] is exactly row i of the bracket above, for EVERY
source position p at once. So one backward pass from s_i gives row i at all positions;
we then average over p in P. Doing this for i = 1..d gives the whole matrix.

In autograd terms, "backward from s_i" means: call autograd.grad on the target activation
with a "cotangent" (upstream gradient) that is 1 at coordinate i for every position in P
and 0 elsewhere.

To save time we compute B rows per backward pass: the window is copied B times along the
batch axis, and copy b gets the cotangent for coordinate i0 + b. Copies don't interact,
so each copy's gradient is its own row. Cost per window: 1 forward + ceil(d/B) backward
passes, and every source layer comes out of the same backward passes.
"""
from __future__ import annotations

import math

import torch

from .model import decode, record_residuals, target_layer

SKIP_FIRST = 16  # leading positions excluded from the estimator (attention sinks)


def valid_positions(T: int, skip_first: int = SKIP_FIRST) -> torch.Tensor:
    """Indices of the positions P used as sources and targets: skip_first .. T-2."""
    return torch.arange(skip_first, T - 1)


def window_jacobians(model, input_ids: torch.Tensor, layers, *, rows_per_pass: int = 32,
                     skip_first: int = SKIP_FIRST, per_position: bool = False) -> dict[int, torch.Tensor]:
    """Compute J_l^(w) for one window and every layer in `layers`.

    input_ids: [1, T] token ids of the window.
    Returns {layer: [d, d] float32 tensor on CPU}; entry [i, j] = d(target coord i) / d(layer coord j).
    per_position=True instead returns {layer: [d, T, d]} (the bracket for every source p,
    before averaging); only the correctness check uses this.
    """
    layers = sorted(layers)
    tgt = target_layer(model)
    assert layers[-1] < tgt, "source layers must come before the target layer"
    d = model.config.hidden_size
    T = input_ids.shape[1]
    P = valid_positions(T, skip_first).to(input_ids.device)
    B = rows_per_pass

    out = {l: torch.zeros(d, T, d) if per_position else torch.zeros(d, d) for l in layers}

    with torch.enable_grad(), record_residuals(model, [*layers, tgt], grad_root=layers[0]) as acts:
        # One forward pass on B copies of the window. Only the transformer blocks are
        # needed (the target is pre-norm), so we call model.model, not the LM head.
        model.model(input_ids=input_ids.expand(B, -1))
        h_target = acts[tgt]                      # [B, T, d]
        h_sources = [acts[l] for l in layers]     # each [B, T, d]

        copy = torch.arange(B, device=h_target.device)
        cotangent = torch.zeros_like(h_target)
        n_passes = math.ceil(d / B)
        for k, i0 in enumerate(range(0, d, B)):
            n = min(B, d - i0)  # the last pass may have fewer rows than copies
            # Copy b asks for row (i0 + b): put a 1 at coordinate i0 + b, at every valid position.
            cotangent.zero_()
            cotangent[copy[:n, None], P[None, :], i0 + copy[:n, None]] = 1.0
            grads = torch.autograd.grad(h_target, h_sources, grad_outputs=cotangent,
                                        retain_graph=k < n_passes - 1)  # keep the graph for the next pass
            for l, g in zip(layers, grads):       # g: [B, T, d]; g[b, p] = row (i0+b) at source p
                if per_position:
                    out[l][i0:i0 + n] = g[:n].float().cpu()
                else:
                    out[l][i0:i0 + n] = g[:n, P, :].float().mean(dim=1).cpu()  # mean over sources
    return out


class JLens:
    """A fitted lens: one d x d matrix J_l per layer, plus the readout.

    jacobians: {layer: [d, d] float32}. Use JLens.from_windows to build one from
    per-window Jacobians, or JLens.identity for the logit-lens baseline.
    """

    def __init__(self, jacobians: dict[int, torch.Tensor], info: dict | None = None):
        self.jacobians = {l: J.float() for l, J in jacobians.items()}
        self.layers = sorted(self.jacobians)
        self.info = info or {}  # free-form provenance: corpus, window count, weighting, ...

    def __repr__(self):
        return f"JLens(layers={self.layers[0]}..{self.layers[-1]}, info={self.info})"

    @classmethod
    def from_windows(cls, per_window_jacobians, weights: torch.Tensor | None = None, info=None) -> "JLens":
        """Weighted average of per-window Jacobians: J_l = sum_w weight_w * J_l^(w).

        per_window_jacobians: an iterable yielding one {layer: [d, d]} dict per window
            (e.g. JacobianStore.iterate). It is consumed one window at a time, so memory
            stays at one lens regardless of how many windows there are.
        weights: [N] non-negative, one per window (normalized here to sum to 1);
            None = uniform, i.e. the plain mean (the paper's estimator).
        """
        total, n = None, 0
        for w_idx, J_w in enumerate(per_window_jacobians):
            weight = 1.0 if weights is None else float(weights[w_idx])
            if total is None:
                total = {l: torch.zeros_like(J, dtype=torch.float32) for l, J in J_w.items()}
            for l, J in J_w.items():
                total[l] += weight * J.float()
            n += 1
        assert n > 0, "no windows"
        if weights is not None:
            assert len(weights) == n, f"{len(weights)} weights for {n} windows"
        norm = n if weights is None else float(weights.sum())
        return cls({l: J / norm for l, J in total.items()}, info=info)

    @classmethod
    def identity(cls, layers, d: int) -> "JLens":
        """J_l = I for every layer: the logit lens, expressed as a JLens."""
        return cls({l: torch.eye(d) for l in layers}, info={"name": "logit lens"})

    def transport(self, h: torch.Tensor, layer: int) -> torch.Tensor:
        """Map residual(s) at `layer`, shape [..., d], into target-layer coordinates: J_l @ h."""
        J = self.jacobians[layer].to(h.device)
        return h.float() @ J.T  # row-vector form of J @ h

    def logits(self, model, h: torch.Tensor, layer: int) -> torch.Tensor:
        """Lens readout as vocabulary logits: decode(J_l @ h)."""
        return decode(model, self.transport(h, layer))

    def save(self, path):
        torch.save({"J": {l: J.half() for l, J in self.jacobians.items()}, "info": self.info}, path)

    @classmethod
    def load(cls, path) -> "JLens":
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        return cls(ckpt["J"], info=ckpt["info"])
