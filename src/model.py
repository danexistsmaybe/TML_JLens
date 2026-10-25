"""Model loading, residual-stream hooks, and decoding.

Conventions used everywhere in this project:
  * "Layer l" (0-indexed) is the OUTPUT of decoder block l, after its residual add.
    This equals HF's hidden_states[l + 1]. Qwen2.5-0.5B has 24 blocks: layers 0..23.
  * The TARGET layer of the J-lens is the last block's output (layer 23), BEFORE the
    final RMSNorm.
  * decode(h) = lm_head(final_norm(h)). On the target layer this reproduces the model's
    own logits exactly (checked in src/check_jacobian.py).
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
os.environ.setdefault("HF_HOME", str(DATA / "hf_cache"))  # model + dataset downloads

MODEL_ID = "Qwen/Qwen2.5-0.5B"


def load_model(dtype=torch.float32, device="cuda"):
    """Load the frozen base model and its tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dtype).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


def n_layers(model) -> int:
    return model.config.num_hidden_layers


def target_layer(model) -> int:
    return n_layers(model) - 1


@torch.no_grad()
def decode(model, h: torch.Tensor) -> torch.Tensor:
    """Residual vector(s) [..., d] -> vocabulary logits [..., vocab] (final norm + unembedding)."""
    return model.lm_head(model.model.norm(h.to(model.lm_head.weight.dtype))).float()


@contextmanager
def record_residuals(model, layers, grad_root: int | None = None):
    """Capture block outputs during a forward pass.

    Yields a dict that fills up as the model runs: {layer: tensor [batch, T, d]}.

    grad_root: if given, that layer's output is turned into a tensor that requires grad.
    The model's weights are frozen, so normally nothing in the forward pass tracks
    gradients; marking one layer as the "root" makes autograd record every later
    operation, which is what lets us differentiate the target layer with respect to
    any layer >= grad_root.
    """
    acts: dict[int, torch.Tensor] = {}
    handles = []
    blocks = model.model.layers

    def make_hook(layer):
        def hook(module, inputs, output):
            h = output if torch.is_tensor(output) else output[0]
            if layer == grad_root:
                h.requires_grad_(True)
            acts[layer] = h
        return hook

    try:
        for layer in sorted(set(layers) | ({grad_root} if grad_root is not None else set())):
            handles.append(blocks[layer].register_forward_hook(make_hook(layer)))
        yield acts
    finally:
        for handle in handles:
            handle.remove()


@torch.no_grad()
def residuals(model, input_ids: torch.Tensor, layers) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
    """Plain forward pass. Returns ({layer: [batch, T, d]}, model logits [batch, T, vocab])."""
    with record_residuals(model, layers) as acts:
        logits = model(input_ids=input_ids).logits.float()
    return {l: h.detach() for l, h in acts.items()}, logits
