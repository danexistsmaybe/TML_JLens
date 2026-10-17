"""Shared model loading, residual-stream hooks, and decoding conventions for P10.

Conventions (record these in every report; they are the "residual convention" the
brief asks us to verify):
  * Layer l (0-indexed) = OUTPUT of decoder block l (post-residual-add), i.e. the
    tensor HF returns as hidden_states[l + 1].
  * Target layer = output of the final block (L-1), BEFORE the final RMSNorm.
  * Decode = lm_head(final_norm(x)). Applied to the target layer this reproduces the
    model's own logits exactly (checked in scripts/02_correctness.py).
  * Logit lens = decode(h_l). Jacobian lens = decode(J_l @ h_l).
"""
from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path

import torch

P10 = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(P10 / "hf_cache"))
MODEL_ID = "Qwen/Qwen2.5-0.5B"
SKIP_FIRST = 16  # leading positions excluded from Jacobian averages (attention sinks)


def load_model(dtype=torch.float32, device="cuda"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dtype).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


def blocks(model):
    return model.model.layers


def n_layers(model) -> int:
    return model.config.num_hidden_layers


@torch.no_grad()
def decode(model, x: torch.Tensor) -> torch.Tensor:
    """Final norm + unembedding, in fp32 logits."""
    return model.lm_head(model.model.norm(x.to(model.lm_head.weight.dtype))).float()


@contextmanager
def record(model, layers, root: int | None = None):
    """Capture block outputs at `layers` (dict filled during forward). If `root` is
    given, that block's output becomes a grad-requiring leaf so autograd graphs start
    there (model params are frozen)."""
    acts: dict[int, torch.Tensor] = {}
    handles = []
    idx = sorted(set(layers) | ({root} if root is not None else set()))

    def mk(i):
        def hook(mod, inp, out):
            t = out if torch.is_tensor(out) else out[0]
            if i == root:
                t.requires_grad_(True)
            acts[i] = t
        return hook

    try:
        for i in idx:
            handles.append(blocks(model)[i].register_forward_hook(mk(i)))
        yield acts
    finally:
        for h in handles:
            h.remove()


def residuals(model, input_ids: torch.Tensor, layers) -> dict[int, torch.Tensor]:
    """No-grad forward; returns {layer: [B, T, d]} block outputs plus model logits under key 'logits'."""
    with torch.no_grad(), record(model, layers) as acts:
        out = model(input_ids=input_ids)
        res = {k: v.detach() for k, v in acts.items()}
    res["logits"] = out.logits.float()
    return res


def load_manifest():
    return json.loads((P10 / "manifests" / "splits.json").read_text())


def window_ids(manifest, corpus: str, role: str) -> list[list[int]]:
    return [w["input_ids"] for w in manifest["corpora"][corpus][role]]
