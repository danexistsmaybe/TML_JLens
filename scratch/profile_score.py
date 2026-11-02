"""Where does scoring one lens spend its time?"""
import sys, time
sys.path.insert(0, ".")
import torch
from src import experiments as E
from src.eval_prompts import BAND, pass_ranks, swap_results
from src.fitting import fit_many
from src.metrics import edit_trials, jacobian_error, readout_agreement
from src.model import load_model
model, tok = load_model()
ws = E.fit_windows("wiki")[:8]
lens = E.on_gpu(fit_many(ws, {"a": {l: torch.ones(8) for l in BAND}}, BAND)["a"])
t = time.perf_counter(); ev = E.Evaluator(model, tok, "wiki", "val"); print("setup", round(time.perf_counter() - t, 1))
def timed(name, fn):
    t = time.perf_counter(); fn(); print(name, round(time.perf_counter() - t, 1), flush=True)
timed("jacobian_error", lambda: jacobian_error(lens, ev.windows, E.MAIN_LAYERS))
timed("pass", lambda: pass_ranks(model, ev.prepared, lens))
timed("swap", lambda: swap_results(model, tok, ev.swaps, lens))
timed("edits", lambda: [edit_trials(model, lens, w, l, ev.alphas[l]) for l in E.MAIN_LAYERS for w in ev.windows])
timed("readout", lambda: readout_agreement(model, lens, lens, ev.activations))
print("swap trials", sum(len(x["trials"]) for x in ev.swaps))
