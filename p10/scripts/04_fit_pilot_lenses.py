"""Pilot: per-window Jacobians on both corpora (all source layers 0..22, target = block 23).

  fit windows : first N_FIT of a seeded permutation of each corpus's 200-window fit pool,
                split into two disjoint halves A and B (the B half gives a within-corpus
                sampling-noise control for every cross-corpus comparison).
  val windows : first N_VAL val windows per corpus, used as "deployment-domain" samples of
                the true local Jacobian (derivative-prediction error).

Per-window Jacobians are saved (fp16) so every lens -- domain-specific, pooled, weighted --
is a weighted average of the same stored matrices.

Usage: .venv/bin/python scripts/04_fit_pilot_lenses.py [N_FIT] [N_VAL] [DIM_BATCH]
Outputs: results/pilot/perwindow_<corpus>_<role>.pt, results/pilot/timing.json
"""
import json
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import P10, load_manifest, load_model, window_ids  # noqa: E402
from jacobian import fit_windows  # noqa: E402

N_FIT = int(sys.argv[1]) if len(sys.argv) > 1 else 40
N_VAL = int(sys.argv[2]) if len(sys.argv) > 2 else 10
DIM_BATCH = int(sys.argv[3]) if len(sys.argv) > 3 else 32
SEED = 0
out_dir = P10 / "results" / "pilot"
out_dir.mkdir(parents=True, exist_ok=True)

model, tok = load_model()
SRC = list(range(model.config.num_hidden_layers - 1))
man = load_manifest()
timing = {"dim_batch": DIM_BATCH, "dtype": "float32", "seq_len": 128, "gpu": torch.cuda.get_device_name(0),
          "source_layers": SRC, "target_layer": model.config.num_hidden_layers - 1}
torch.cuda.reset_peak_memory_stats()
for corpus in ["wikitext", "tinystories"]:
    pool = window_ids(man, corpus, "fit")
    perm = list(range(len(pool)))
    random.Random(SEED).shuffle(perm)
    jobs = {"fitA": perm[:N_FIT // 2], "fitB": perm[N_FIT // 2:N_FIT], "val": list(range(N_VAL))}
    for role, idx in jobs.items():
        wins = [pool[i] for i in idx] if role != "val" else [window_ids(man, corpus, "val")[i] for i in idx]
        print(f"[{time.strftime('%H:%M:%S')}] {corpus}/{role}: {len(wins)} windows", flush=True)
        per, secs = fit_windows(model, wins, SRC, dim_batch=DIM_BATCH, log=lambda s: print(s, flush=True))
        torch.save({"J": per, "window_index": idx, "role": role, "corpus": corpus,
                    "fit_pool_seed": SEED if role != "val" else None},
                   out_dir / f"perwindow_{corpus}_{role}.pt")
        timing[f"{corpus}/{role}"] = {"windows": len(wins), "sec_per_window": secs}
        print(f"   {secs:.2f} s/window", flush=True)
timing["peak_mem_GB"] = torch.cuda.max_memory_allocated() / 1e9
(out_dir / "timing.json").write_text(json.dumps(timing, indent=2))
print(json.dumps(timing, indent=2))
