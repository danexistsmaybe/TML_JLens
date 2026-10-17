# P10: Does the J-lens fitting corpus change the interpretation? (Week 1 pilot)

Morning reading order:
1. `SUMMARY.md`: concepts, methods, and the problem (~15 min).
2. `CHECKLIST.md`: first-day checklist steps 1–5 with findings, the compute estimate, and **slide-ready results**.
3. `PROGRESS.md`: timestamped log of everything done overnight, including what went wrong.
4. This file, to rerun anything yourself.

## Layout
```
SUMMARY.md            paper synthesis (read first)
CHECKLIST.md          checklist steps 1-5, pilot findings, compute estimate, talk material
PROGRESS.md           overnight log
ENV.md, requirements.txt   environment record (uv venv, torch/CUDA, model/data revisions)
src/common.py         model loading, hooks, residual + decode conventions
src/jacobian.py       OUR corpus-conditioned / weighted J-lens estimator
scripts/01_make_splits.py         split manifest (run before any fitting)
scripts/02_correctness.py         hook/decode checks + Jacobian vs finite differences
scripts/03_first_start_logit_lens.py   brief's first-start experiment
scripts/04_fit_pilot_lenses.py    per-window Jacobians, both corpora
scripts/05_pilot_analysis.py      corpus effect vs sampling noise, derivative error, retrieval, viewer table
scripts/06_slide_figure.py        slide-ready headline figure (CPU only)
manifests/splits.json, preview.md
results/              csv/json/md outputs (results/pilot/*.pt are large per-window Jacobians)
figures/              fig1..fig4 PNGs (fig4_headline.png = main slide figure)
notes/                plain-text paper, reference repo files (reading only)
hf_cache/             model + dataset cache (HF_HOME)
```

## Rerun
```bash
cd ~/Claude/projects/cs6501-tml/p10
source .venv/bin/activate             # Python 3.11, torch cu124 (see ENV.md)
python scripts/01_make_splits.py      # ~1 min; deterministic (seed 0)
python scripts/02_correctness.py      # ~2 min; expects PASS
python scripts/03_first_start_logit_lens.py   # ~1 min -> figures/fig1
python scripts/04_fit_pilot_lenses.py 80 20 16  # ~30 min GPU; 200 windows x ~8.7 s
python scripts/05_pilot_analysis.py   # ~5 min -> figures/fig2, fig3, results/pilot/*
python scripts/06_slide_figure.py     # seconds -> figures/fig4_headline.png
```
`src/` is put on the path by each script; `HF_HOME` is set to `./hf_cache` in `src/common.py`.

## Conventions (the brief requires these to be stated)
- Model `Qwen/Qwen2.5-0.5B` (base), fp32, frozen. 24 blocks, d_model 896. Revision in ENV.md.
- **Layer l = output of decoder block l (0-indexed)** = HF `hidden_states[l+1]`.
- **Target = output of block 23, before the final RMSNorm.** Decode = `lm_head(norm(x))`, which reproduces model logits exactly (verified).
- Estimator: one-hot cotangent at every valid target position, sum over targets p' ≥ p, mean over valid source positions, mean (or weighted mean) over windows. Valid positions = 16..T-2 (skip attention-sink positions and the last position).
- Logit lens = J_l = I under the same decode.
- Windows: exactly 128 tokens, no special tokens added.
