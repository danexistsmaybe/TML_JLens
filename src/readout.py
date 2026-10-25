"""Reading a prompt through one or more lenses: top-k tokens per layer at one position."""
from __future__ import annotations

import torch

from .jlens import JLens
from .model import residuals


@torch.no_grad()
def top_tokens(model, tok, prompt: str, lens: JLens, layers=None, position: int = -1, k: int = 5):
    """Run `prompt` and read `position` (default: last token) through `lens` at each layer.

    Returns {layer: [(token string, probability), ...]} with the k most likely tokens,
    plus the model's own top-k next tokens under the key "model".
    """
    layers = lens.layers if layers is None else layers
    ids = tok(prompt, return_tensors="pt").input_ids.to(next(model.parameters()).device)
    acts, model_logits = residuals(model, ids, layers)

    def topk(logits):
        probs = logits.softmax(-1)
        p, idx = probs.topk(k)
        return [(tok.decode(int(i)), float(q)) for i, q in zip(idx, p)]

    out = {l: topk(lens.logits(model, acts[l][0, position], l)) for l in layers}
    out["model"] = topk(model_logits[0, position])
    return out


def print_side_by_side(readouts: dict[str, dict], layers, highlight=(), width: int = 46):
    """Print one column per lens: readouts = {"J-lens": top_tokens(...), "logit lens": ...}.
    Tokens whose stripped, lower-cased text is in `highlight` are wrapped in *stars*."""
    highlight = {h.lower() for h in highlight}

    def fmt(entries):
        toks = []
        for t, _ in entries:
            s = repr(t)[1:-1]  # make whitespace/newlines visible
            toks.append(f"*{s}*" if t.strip().lower() in highlight else s)
        return " | ".join(toks)[:width]

    names = list(readouts)
    print("layer  " + "".join(f"{n:<{width + 2}}" for n in names))
    for l in layers:
        print(f"{l:>5}  " + "".join(f"{fmt(readouts[n][l]):<{width + 2}}" for n in names))
    print(f"model  {fmt(next(iter(readouts.values()))['model'])}")
