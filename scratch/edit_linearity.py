"""Is +/- edit error with an exact Jacobian nonlinearity, or a bug? Shrink the edit and see."""
import sys
sys.path.insert(0, ".")
import torch
from src.corpus import load_windows
from src.jlens import JLens, window_jacobians
from src.metrics import edit_positions, edit_trials, typical_norms, window_activations
from src.model import load_model
model, tok = load_model()
w = load_windows("wikitext", "fit", n=1)[0]
for layer in (12, 18):
    per = window_jacobians(model, w.ids_tensor("cuda"), [layer], per_position=True)[layer]
    pos = edit_positions(w, 5)
    exact = per[:, pos, :].permute(1, 0, 2)
    norm = typical_norms(window_activations(model, [w], [layer]))[layer]
    print("layer", layer, "typical norm", round(norm, 1))
    for frac in (1e-4, 1e-3, 0.01, 0.05, 0.1, 0.5):
        rec = edit_trials(model, JLens({layer: exact[0]}), w, layer, {frac: frac * norm}, positions=pos, directions=exact)
        errs = sorted(r["relative_error"] for r in rec)
        by = {}
        for r in rec:
            by.setdefault(r["position"], {})[r["sign"]] = r
        anti = sorted(abs((v[1.0]["actual"] - v[-1.0]["actual"]) / 2 - v[1.0]["predicted"]) / abs(v[1.0]["predicted"]) for v in by.values())
        print(f"  {frac:7.4f} median err {errs[len(errs)//2]:.4f} max {errs[-1]:.4f} | antisym median {anti[len(anti)//2]:.5f} max {anti[-1]:.5f} | corr min {min(r['vocab_correlation'] for r in rec):.4f}")
