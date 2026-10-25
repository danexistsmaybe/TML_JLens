"""Probe candidate demo prompts: J-lens vs logit lens top tokens at the last position.

    .venv/bin/python scratch/probe_demo_prompts.py "prompt one" "prompt two" ...

Uses the demo's lens (8 WikiText + 8 TinyStories windows, cached Jacobians).
Used 2026-10-25 to pick the demo prompts (see .agent/DECISIONS.md).
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import corpus  # noqa: E402
from src.jlens import JLens  # noqa: E402
from src.model import load_model, target_layer  # noqa: E402
from src.readout import print_side_by_side, top_tokens  # noqa: E402
from src.store import fit_lens  # noqa: E402

model, tok = load_model()
layers = list(range(target_layer(model)))
windows = corpus.load_windows("wikitext", "fit", 8) + corpus.load_windows("tinystories", "fit", 8)
jlens = fit_lens(model, windows, layers)
logit = JLens.identity(layers, model.config.hidden_size)
shown = layers[6:]
for prompt in sys.argv[1:]:
    print(f"\n=== {prompt!r}")
    print_side_by_side({"J-lens": top_tokens(model, tok, prompt, jlens, shown, k=6),
                        "logit lens": top_tokens(model, tok, prompt, logit, shown, k=4)}, shown)
