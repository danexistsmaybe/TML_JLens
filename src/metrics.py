"""The window-based metrics: Jacobian error, readout agreement and +/- edit prediction error.

Pass@k and swap success, which use prompts, are in src/eval_prompts.py. Every function returns a
score per window, so that comparisons can be paired and confidence intervals can resample windows.
"""
from __future__ import annotations

import random

import torch

from .corpus import Window
from .fitting import load_jacobians
from .jlens import JLens, valid_positions
from .model import block_outputs, decode, record_residuals, target_layer

EDIT_POSITIONS_PER_WINDOW = 10
EDIT_FRACTIONS = (0.05, 0.10)  # edit size, as a share of the layer's typical activation norm


# ---------------------------------------------------------------------------------------
# Jacobian error
# ---------------------------------------------------------------------------------------

def jacobian_error(lens: JLens, windows: list[Window], layers) -> dict[int, list[float]]:
    """||J_lens - J_window||_F / ||J_window||_F, per layer and window."""
    out = {l: [] for l in layers}
    for window in windows:
        stored = load_jacobians(window, layers)
        for l in layers:
            J = stored[l].float()
            out[l].append(float((lens.jacobians[l].cpu() - J).norm() / J.norm()))
    return out


# ---------------------------------------------------------------------------------------
# Activations of whole windows, kept once so each lens needn't rerun the model
# ---------------------------------------------------------------------------------------

@torch.no_grad()
def window_activations(model, windows: list[Window], layers, batch_size: int = 32) -> dict[int, torch.Tensor]:
    """{layer: [N, T, d] float32 on CPU}."""
    device = next(model.parameters()).device
    ids = torch.tensor([w.input_ids for w in windows])
    parts = {l: [] for l in layers}
    for start in range(0, len(ids), batch_size):
        acts = block_outputs(model, ids[start:start + batch_size].to(device), layers)
        for l in layers:
            parts[l].append(acts[l].float().cpu())
    return {l: torch.cat(parts[l]) for l in layers}


# ---------------------------------------------------------------------------------------
# Readout agreement
# ---------------------------------------------------------------------------------------

@torch.no_grad()
def readout_agreement(model, lens: JLens, reference: JLens, activations: dict[int, torch.Tensor], k: int = 10) -> dict:
    """Top-k overlap and symmetric KL against a reference lens, at each window's valid positions.

    Returns {layer: {"overlap": [per window], "sym_kl": [per window]}}. Symmetric KL is
    (KL(p||q) + KL(q||p)) / 2, in nats.
    """
    device = next(model.parameters()).device
    out = {}
    for layer, acts in activations.items():
        P = valid_positions(acts.shape[1])
        overlap, sym_kl = [], []
        for window in acts:
            h = window[P].to(device)
            a, b = lens.logits(model, h, layer), reference.logits(model, h, layer)
            top_a, top_b = a.topk(k).indices, b.topk(k).indices
            overlap.append(float((top_a[:, :, None] == top_b[:, None, :]).any(-1).float().mean()))
            log_a, log_b = a.log_softmax(-1), b.log_softmax(-1)
            kl_ab = (log_a.exp() * (log_a - log_b)).sum(-1)
            kl_ba = (log_b.exp() * (log_b - log_a)).sum(-1)
            sym_kl.append(float(((kl_ab + kl_ba) / 2).mean()))
        out[layer] = {"overlap": overlap, "sym_kl": sym_kl}
    return out


# ---------------------------------------------------------------------------------------
# +/- edit prediction error
# ---------------------------------------------------------------------------------------

def edit_positions(window: Window, count: int = EDIT_POSITIONS_PER_WINDOW) -> list[int]:
    """Fixed for each window, so every lens is tested at the same positions."""
    P = valid_positions(len(window.input_ids)).tolist()
    return sorted(random.Random(f"edit-{window.sha256}").sample(P, count))


@torch.no_grad()
def typical_norms(activations: dict[int, torch.Tensor]) -> dict[int, float]:
    """Mean activation norm per layer, at valid positions. Sets the edit size in absolute terms."""
    return {l: float(a[:, valid_positions(a.shape[1])].norm(dim=-1).mean()) for l, a in activations.items()}


@torch.no_grad()
def edit_trials(model, lens: JLens, window: Window, layer: int, alphas: dict[float, float],
                positions: list[int] | None = None, directions: torch.Tensor | None = None) -> list[dict]:
    """Edit h_{l,t} +/- alpha v_c and compare the actual change in c's score with the lens's prediction.

    c is the lens's top reading at position t, and v_c = J^T w_c / ||J^T w_c||. The score is
    s_c = sum over valid t' >= t of w_c . h_final[t'], before the final norm, so it is what the
    lens estimates: the predicted change is +/- alpha ||J^T w_c||.

    alphas: {fraction of typical norm: absolute edit size}.
    positions, directions: tests may fix these; directions is [n_positions, d, d] exact J per position.
    Returns one record per position, size and sign.
    """
    device = next(model.parameters()).device
    ids = torch.tensor([window.input_ids], device=device)
    tgt = target_layer(model)
    T = ids.shape[1]
    P = valid_positions(T).to(device)
    positions = positions if positions is not None else edit_positions(window)
    pos = torch.tensor(positions, device=device)

    with record_residuals(model, [layer, tgt]) as acts:
        model.model(input_ids=ids)
    h_clean, final_clean = acts[layer].detach(), acts[tgt].detach().float()

    W = model.lm_head.weight.float()
    if directions is None:
        J = lens.jacobians[layer].to(device)
        readings = decode(model, (h_clean[0, pos].float() @ J.T)).argmax(-1)
        Jt_w = W[readings] @ J                            # [n, d]: row form of J^T w_c
        J_of = J[None].expand(len(pos), -1, -1)           # [n, d, d]
    else:
        J_of = directions.to(device)
        readings = decode(model, (J_of @ h_clean[0, pos].float()[..., None])[..., 0]).argmax(-1)
        Jt_w = (W[readings][:, None] @ J_of)[:, 0]
    lens_gain = Jt_w.norm(dim=-1)                         # ||J^T w_c||
    v = Jt_w / lens_gain[:, None]

    # One forward pass on one copy per edit. Edits at a position only change later positions.
    cases = [(i, frac, size, sign) for i in range(len(pos)) for frac, size in alphas.items() for sign in (1.0, -1.0)]
    which = torch.tensor([c[0] for c in cases], device=device)
    signed = torch.tensor([c[2] * c[3] for c in cases], device=device, dtype=torch.float32)
    copies = torch.arange(len(cases), device=device)

    def add_edit(module, inputs, output):
        h = output if torch.is_tensor(output) else output[0]
        edited = h.clone()
        edited[copies, pos[which]] += (signed[:, None] * v[which]).to(h.dtype)
        return edited if torch.is_tensor(output) else (edited, *output[1:])

    handle = model.model.layers[layer].register_forward_hook(add_edit)
    try:
        with record_residuals(model, [tgt]) as edited_acts:
            model.model(input_ids=ids.expand(len(cases), -1))
    finally:
        handle.remove()

    delta = edited_acts[tgt].float() - final_clean            # [B, T, d]
    in_P = torch.zeros(T, device=device, dtype=torch.bool)
    in_P[P] = True
    after = (torch.arange(T, device=device)[None] >= pos[which][:, None]) & in_P[None]  # [B, T]
    summed = (delta * after[..., None]).sum(dim=1)            # [B, d]
    actual = (summed * W[readings[which]]).sum(-1)
    predicted = signed * lens_gain[which]

    # Across the whole vocabulary: does the lens predict the direction of the output's change?
    predicted_final = (J_of[which] @ (signed[:, None] * v[which])[..., None])[..., 0]  # J v, [B, d]
    a = summed @ W.T
    b = predicted_final @ W.T
    a, b = a - a.mean(-1, keepdim=True), b - b.mean(-1, keepdim=True)
    correlation = (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1)).clamp_min(1e-12)

    records = []
    for j, (i, frac, _, sign) in enumerate(cases):
        records.append({"position": positions[i], "fraction": frac, "sign": sign,
                        "token": int(readings[i]), "predicted": float(predicted[j]), "actual": float(actual[j]),
                        "relative_error": float((predicted[j] - actual[j]).abs() / actual[j].abs().clamp_min(1e-12)),
                        "vocab_correlation": float(correlation[j])})
    return records
