"""Demo: fit one J-lens on a small corpus, then read a prompt through it.

    python -m src.demo                          # default: French antonym prompt
    python -m src.demo --preset currency
    python -m src.demo --prompt "Your own prompt" --highlight word1 word2
    python -m src.demo --corpus wikitext --n-windows 20

Steps:
  1. Load Qwen2.5-0.5B (frozen).
  2. Pick fitting windows from the split manifest (default: 8 WikiText + 8 TinyStories).
  3. Compute each window's Jacobians (about 9 s per window on the first run; cached in
     data/jacobians/ afterwards) and average them uniformly into a J-lens.
  4. Read the prompt's last position at every layer through the J-lens and through the
     logit lens (J = I), side by side.

Why not the paper's Mars prompt? "The color of the fourth planet from the Sun is" needs
Mars as an intermediate, but this 0.5B model doesn't reliably answer "red" (it prefers
blue/blank), and "Mars" never surfaces at any layer. The presets below are prompts where the
small model gets the answer and the J-lens shows an intermediate the logit lens misses
(see .agent/DECISIONS.md, 2026-10-25). The Mars prompt is kept as a preset for comparison.
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

from . import corpus  # noqa: E402
from .jlens import JLens  # noqa: E402
from .model import load_model, target_layer  # noqa: E402
from .readout import print_side_by_side, top_tokens  # noqa: E402
from .store import fit_lens  # noqa: E402

PRESETS = {
    # Multilingual (as in the paper): the answer is French "grand", but the English concept
    # "big"/"large" shows up in the J-lens several layers before the output.
    "antonym": ('Le contraire de "chaud" est "froid".\nLe contraire de "petit" est "',
                ["big", "large", "grand", "grande"]),
    # Two-hop: Tokyo -> Japan -> yen. The J-lens surfaces "currency" and then "Japan"/"Japanese".
    "currency": ("The currency of the country whose capital is Tokyo is the",
                 ["currency", "japan", "japanese", "yen", "tokyo"]),
    # The paper's example. Does NOT work on Qwen2.5-0.5B: no "Mars" at any layer.
    "mars": ("The color of the planet fourth from the Sun is", ["mars", "red"]),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=PRESETS, default="antonym")
    ap.add_argument("--prompt", help="custom prompt (overrides --preset)")
    ap.add_argument("--highlight", nargs="*", default=None, help="tokens to mark with *stars*")
    ap.add_argument("--corpus", choices=[*corpus.CORPORA, "both"], default="both")
    ap.add_argument("--n-windows", type=int, default=16, help="total fitting windows")
    ap.add_argument("--first-layer", type=int, default=6, help="first layer to print")
    args = ap.parse_args()

    prompt, highlight = PRESETS[args.preset]
    if args.prompt:
        prompt, highlight = args.prompt, []
    if args.highlight is not None:
        highlight = args.highlight

    print("Loading model...")
    model, tok = load_model()
    layers = list(range(target_layer(model)))  # every block before the target: 0..22

    # The fitting corpus: a list of windows plus a weight per window.
    if args.corpus == "both":
        half = args.n_windows // 2
        windows = corpus.load_windows("wikitext", "fit", half) + \
            corpus.load_windows("tinystories", "fit", args.n_windows - half)
    else:
        windows = corpus.load_windows(args.corpus, "fit", args.n_windows)
    weights = corpus.uniform(windows)

    print(f"Fitting J-lens on {len(windows)} windows ({args.corpus})...")
    jlens = fit_lens(model, windows, layers, weights)
    logit_lens = JLens.identity(layers, model.config.hidden_size)

    print(f"\nPrompt: {prompt!r}\nTop-5 tokens at the last position, per layer:\n")
    shown = [l for l in layers if l >= args.first_layer]
    print_side_by_side({"J-lens": top_tokens(model, tok, prompt, jlens, shown),
                        "logit lens": top_tokens(model, tok, prompt, logit_lens, shown)},
                       shown, highlight=highlight)


if __name__ == "__main__":
    main()
