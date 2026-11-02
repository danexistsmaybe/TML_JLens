"""Correctness checks for the hooks, the decode convention, and the Jacobian estimator.

    python -m tests.check_jacobian        # ~2 min on GPU; prints PASS/FAIL

  A. Hooks: our block-output hooks equal HF's hidden_states[l + 1].
  B. Decode: decode(target layer) reproduces the model's own logits.
  C. Jacobian vs finite differences (float64): nudge h_l[p] by +-eps*u along a random unit
     vector u; the change in sum_{p' in P} h_target[p'] must match (our row-by-row
     Jacobian at source p) @ u.
  D. The averaged estimator equals the mean over valid positions of the per-position Jacobians.

Rerun after any change to src/model.py or src/jlens.py.
"""
from __future__ import annotations

import os

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import torch  # noqa: E402

from src.corpus import load_windows  # noqa: E402
from src.jlens import valid_positions, window_jacobians  # noqa: E402
from src.model import decode, load_model, record_residuals, target_layer  # noqa: E402


def check_conventions(ids):
    model, _ = load_model(torch.float32)
    L = target_layer(model)
    with torch.no_grad(), record_residuals(model, range(L + 1)) as acts:
        out = model(input_ids=ids, output_hidden_states=True)
    hook_err = max((acts[l] - out.hidden_states[l + 1]).abs().max().item() for l in range(L))
    # HF's hidden_states[-1] already has the final norm applied, so B compares logits instead.
    decode_err = (decode(model, acts[L]) - out.logits.float()).abs().max().item()
    return hook_err, decode_err


def check_jacobian(ids, T=40, layer=8, p=25, eps=1e-5, trials=3):
    # HF's Qwen2RMSNorm upcasts to float32 internally, which would make a float64 finite
    # difference no more precise than float32 (relative error ~0.5). For this check only,
    # compute the norm in the input dtype (the same function, mathematically).
    from transformers.models.qwen2 import modeling_qwen2

    def rmsnorm_native(self, x):
        return self.weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.variance_epsilon))

    modeling_qwen2.Qwen2RMSNorm.forward = rmsnorm_native
    model, _ = load_model(torch.float64)
    L = target_layer(model)
    ids = ids[:, :T]
    J_pos = window_jacobians(model, ids, [layer], rows_per_pass=32, per_position=True)[layer]  # [d, T, d]
    J_avg = window_jacobians(model, ids, [layer], rows_per_pass=32)[layer]                     # [d, d]
    P = valid_positions(T)
    avg_err = (J_avg - J_pos[:, P, :].mean(1)).abs().max().item()

    def summed_target(delta):
        """sum over valid positions of h_target, after adding delta to h_layer[p]."""
        def nudge(module, inputs, output):
            h = (output if torch.is_tensor(output) else output[0]).clone()
            h[0, p] += delta
            return h if torch.is_tensor(output) else (h, *output[1:])
        handle = model.model.layers[layer].register_forward_hook(nudge)
        try:
            with torch.no_grad(), record_residuals(model, [L]) as acts:
                model.model(input_ids=ids)
                return acts[L][0, P].sum(0)
        finally:
            handle.remove()

    errs = []
    for _ in range(trials):
        u = torch.randn(model.config.hidden_size, dtype=torch.float64, device=ids.device)
        u /= u.norm()
        finite_diff = (summed_target(eps * u) - summed_target(-eps * u)) / (2 * eps)
        analytic = J_pos[:, p, :].double().to(ids.device) @ u
        errs.append(((finite_diff - analytic).norm() / analytic.norm()).item())
    return errs, avg_err


def main():
    torch.manual_seed(0)
    ids = load_windows("wikitext", "val", 1)[0].ids_tensor()
    hook_err, decode_err = check_conventions(ids)
    print(f"A hooks vs hidden_states  max abs err {hook_err:.2e}")
    print(f"B decode vs model logits  max abs err {decode_err:.2e}")
    torch.cuda.empty_cache()
    fd_errs, avg_err = check_jacobian(ids)
    print(f"C Jacobian vs finite diff relative err {', '.join(f'{e:.1e}' for e in fd_errs)}")
    print(f"D averaged vs per-position max abs err {avg_err:.2e}")
    ok = hook_err < 1e-4 and decode_err < 1e-3 and max(fd_errs) < 1e-5 and avg_err < 1e-6
    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
