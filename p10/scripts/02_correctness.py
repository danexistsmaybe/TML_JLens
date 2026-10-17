"""Small correctness checks (course §4: "a Jacobian versus finite differences").

  A. Hook convention: our block-output hooks equal HF hidden_states[l+1].
  B. Decode convention: decode(h_final) reproduces the model's own logits.
  C. Jacobian vs central finite differences (float64): perturb h_l[p] along a random
     unit vector u; the change in sum_{valid p'>=p} h_final[p'] must equal J_p u, where
     J_p is our per-source-position Jacobian.
  D. Averaged estimator == mean over valid positions of the per-position Jacobians.

Writes results/correctness.json. Run: .venv/bin/python scripts/02_correctness.py
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import P10, decode, load_manifest, load_model, record  # noqa: E402
from jacobian import valid_mask, window_jacobians  # noqa: E402

torch.manual_seed(0)
res = {}

# ---- A, B in fp32 on a real manifest window
model, tok = load_model(torch.float32)
ids = torch.tensor([load_manifest()["corpora"]["wikitext"]["val"][0]["input_ids"]], device="cuda")
with torch.no_grad(), record(model, range(model.config.num_hidden_layers)) as acts:
    out = model(input_ids=ids, output_hidden_states=True)
hook_err = max((acts[l] - out.hidden_states[l + 1]).abs().max().item() for l in range(23))
# HF applies the final norm to hidden_states[-1]; compare our pre-norm hook on the last block instead.
dec_err = (decode(model, acts[23]) - out.logits.float()).abs().max().item()
res["A_hook_vs_hidden_states_maxabs"] = hook_err
res["B_decode_vs_model_logits_maxabs"] = dec_err
print("A", hook_err, "B", dec_err)
del model
torch.cuda.empty_cache()

# ---- C, D in fp64 on a short window.
# HF's Qwen2RMSNorm upcasts to float32 internally, which would make an fp64 finite
# difference no more precise than fp32. For this check only, compute the norm in the
# input dtype (mathematically identical function).
from transformers.models.qwen2 import modeling_qwen2 as mq  # noqa: E402


def _rmsnorm_native(self, x):
    var = x.pow(2).mean(-1, keepdim=True)
    return self.weight * (x * torch.rsqrt(var + self.variance_epsilon))


mq.Qwen2RMSNorm.forward = _rmsnorm_native
model, tok = load_model(torch.float64)
T, layer, p = 40, 8, 25
ids = ids[:, :T]
Jpp = window_jacobians(model, ids, [layer], dim_batch=32, per_position=True)[layer]  # [d, T, d]
Javg = window_jacobians(model, ids, [layer], dim_batch=32)[layer]
vpos = valid_mask(T)
res["D_avg_vs_mean_of_per_position_maxabs"] = (Javg - Jpp[:, vpos, :].mean(1)).abs().max().item()

valid_targets = vpos.clone()
valid_targets[:p] = False  # p' >= p (causality makes earlier ones zero anyway)


def summed_final(delta):
    def hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        h = h.clone()
        h[0, p] += delta
        return (h, *out[1:]) if isinstance(out, tuple) else h
    hd = model.model.layers[layer].register_forward_hook(hook)
    try:
        with torch.no_grad(), record(model, [23]) as a:
            model.model(input_ids=ids)
            return a[23][0, vpos].sum(0).double()  # sum over all valid targets
    finally:
        hd.remove()


errs = []
for trial in range(3):
    u = torch.randn(model.config.hidden_size, dtype=torch.float64, device="cuda")
    u /= u.norm()
    eps = 1e-5
    fd = (summed_final(eps * u) - summed_final(-eps * u)) / (2 * eps)
    an = Jpp[:, p, :].double().cuda() @ u
    errs.append(((fd - an).norm() / an.norm()).item())
res["C_finite_difference_relerr"] = errs
res["C_settings"] = dict(dtype="float64", T=T, layer=layer, source_pos=p, eps=1e-5)
print("C", errs, "D", res["D_avg_vs_mean_of_per_position_maxabs"])
res["pass"] = bool(hook_err < 1e-4 and dec_err < 1e-3 and max(errs) < 1e-5
                   and res["D_avg_vs_mean_of_per_position_maxabs"] < 1e-6)
(P10 / "results" / "correctness.json").write_text(json.dumps(res, indent=2))
print("PASS" if res["pass"] else "FAIL", res)
