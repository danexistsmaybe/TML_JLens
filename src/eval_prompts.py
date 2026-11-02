"""Scoring a lens on the prompt sets: pass@k and swap success. Returns scores per prompt and trial.

Pass@k: the best rank of the intermediate in the lens readout at the last position, over a
band of layers. A word counts in any casing, with or without a leading space. The control
token is scored the same way.

Swap (paper, "clamped lens-coordinate swap"): with V = [v_s v_t], c = pinv(V) h is read from
a clean forward pass, and at every position and every layer in the band the coordinates are
set to c + alpha (swap(c) - c). The trial passes if the target argument's answer becomes the
model's top next token.

v_c is J_l^T w_c scaled to unit length, where w_c is token c's row of the output embedding.
The paper doesn't say whether it normalizes. We do, so that swapping coordinates treats the
two tokens alike.
"""
from __future__ import annotations

import math

import torch

from .check_prompts import word_key
from .jlens import JLens
from .model import residuals

BAND = list(range(9, 23))  # the paper's workspace band (38-92% of depth), scaled to 24 blocks
KS = (1, 5, 10, 50)
AUC_KS = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000)  # log-spaced, for the area under pass@k
CORPUS = {"wiki": "wikitext", "stories": "tinystories", "python": "python"}


def word_tokens(tok, word: str) -> list[int]:
    """Every single-token spelling of a word: casing, and with or without a leading space."""
    forms = {word, word.lower(), word.capitalize(), word.upper()}
    ids = set()
    for form in forms:
        for text in (form, " " + form):
            encoded = tok(text, add_special_tokens=False)["input_ids"]
            if len(encoded) == 1 and word_key(tok.decode(encoded[0])) == word_key(word):
                ids.add(encoded[0])
    return sorted(ids)


# ---------------------------------------------------------------------------------------
# Pass@k
# ---------------------------------------------------------------------------------------

@torch.no_grad()
def prepare_pass(model, tok, items: list[dict], layers=BAND, prefix: str = "") -> list[dict]:
    """Run each prompt once, so every lens reads the same stored activations."""
    device = next(model.parameters()).device
    prepared = []
    for item in items:
        ids = tok(prefix + item["prompt"], return_tensors="pt").input_ids.to(device)
        acts, _ = residuals(model, ids, layers)
        prepared.append({"h": {l: acts[l][0, -1] for l in layers},
                         "target": torch.tensor(word_tokens(tok, item["intermediate"]), device=device),
                         "control": torch.tensor(word_tokens(tok, item["control"]), device=device)})
    return prepared


@torch.no_grad()
def pass_ranks(model, prepared: list[dict], lens: JLens, layers=BAND) -> list[dict]:
    """Per prompt: {"target": {layer: rank}, "control": {layer: rank}}, 0-based."""
    results = []
    for p in prepared:
        ranks = {"target": {}, "control": {}}
        for layer in layers:
            logits = lens.logits(model, p["h"][layer], layer)
            for name in ("target", "control"):
                ranks[name][layer] = int((logits > logits[p[name]].max()).sum())
        results.append(ranks)
    return results


def pass_summary(ranks: list[dict], layers=BAND) -> dict:
    """Pass rates and normalized area under pass@k, over a set of layers (the best rank counts)."""
    best = {name: [min(r[name][l] for l in layers) for r in ranks] for name in ("target", "control")}
    summary = {}
    for name, values in best.items():
        summary[name] = {f"pass@{k}": sum(v < k for v in values) / len(values) for k in KS}
        summary[name]["auc"] = sum(auc_of_rank(v) for v in values) / len(values)
    return summary


def auc_of_rank(rank: int) -> float:
    """One prompt's area under pass@k against log k, scaled so that always ranking first gives 1."""
    logs = [math.log(k) for k in AUC_KS]
    passes = [float(rank < k) for k in AUC_KS]
    area = sum((logs[i + 1] - logs[i]) * (passes[i] + passes[i + 1]) / 2 for i in range(len(logs) - 1))
    return area / (logs[-1] - logs[0])


# ---------------------------------------------------------------------------------------
# Swaps
# ---------------------------------------------------------------------------------------

def lens_vectors(model, lens: JLens, layer: int, tokens: list[int]) -> torch.Tensor:
    """[d, n] unit-length lens vectors, one column per token."""
    w = model.lm_head.weight[tokens].float()
    J = lens.jacobians[layer].to(w.device)
    v = w @ J  # row-vector form of J^T w
    return (v / v.norm(dim=-1, keepdim=True)).T


@torch.no_grad()
def swapped_logits(model, tok, prompt: str, lens: JLens, source: int, target: int, alpha: float, layers=BAND,
                   clean: dict | None = None):
    """Next-token logits at the last position, under a clamped lens-coordinate swap.

    The coordinates are read from a clean forward pass and written at every layer. Reading
    them during the edited pass would be wrong: the residual stream carries each edit to the
    next layer, so the next layer's swap would undo it.

    clean: the prompt's stored activations, if the caller already has them.
    """
    device = next(model.parameters()).device
    ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    if clean is None:
        clean, _ = residuals(model, ids, layers)

    def make_hook(layer):
        V = lens_vectors(model, lens, layer, [source, target])
        # Pseudoinverse of a [d, 2] matrix with independent columns: (V^T V)^-1 V^T.
        V_pinv = torch.linalg.solve(V.T @ V, V.T)
        c_clean = clean[layer].float() @ V_pinv.T  # [1, T, 2]
        wanted = c_clean + alpha * (c_clean.flip(-1) - c_clean)

        def hook(module, inputs, output):
            h = output if torch.is_tensor(output) else output[0]
            delta = (wanted - h.float() @ V_pinv.T) @ V.T  # [1, T, d]
            edited = h + delta.to(h.dtype)
            return edited if torch.is_tensor(output) else (edited, *output[1:])
        return hook

    handles = [model.model.layers[l].register_forward_hook(make_hook(l)) for l in layers]
    try:
        return model(input_ids=ids).logits[0, -1].float()
    finally:
        for handle in handles:
            handle.remove()


def run_swap_trial(model, tok, template: dict, trial: dict, lens: JLens, alpha: float, layers=BAND,
                   stored: dict | None = None) -> dict:
    """Swap the prompt's argument (source) for another (target) and read the output.

    stored: {prompt: (clean activations, plain next-token log-probs)}, filled as prompts are met,
    so that a prompt's unedited forward pass runs once across all its trials.
    """
    args = {r["arg"]: r for r in template["results"]}
    source, target = args[trial["source"]], args[trial["target"]]
    prompt = template["template"].replace("{arg}", trial["source"])
    stored = {} if stored is None else stored
    if prompt not in stored:
        ids = tok(prompt, return_tensors="pt").input_ids.to(next(model.parameters()).device)
        clean, logits = residuals(model, ids, layers)
        stored[prompt] = (clean, logits[0, -1].log_softmax(-1))
    clean, plain = stored[prompt]
    logits = swapped_logits(model, tok, prompt, lens, source["arg_token"], target["arg_token"], alpha, layers, clean)
    logp = logits.log_softmax(-1)
    t, s = target["answer_token"], source["answer_token"]
    return {"pass": int(logits.argmax()) == t,
            "target_rank": int((logits > logits[t]).sum()),
            "logp_shift": float((logp[t] - logp[s]) - (plain[t] - plain[s]))}


def swap_results(model, tok, templates: list[dict], lens: JLens, alphas=(1.0, 2.0), layers=BAND) -> dict[str, list[dict]]:
    """{str(alpha): one record per trial, each naming its template}."""
    results, stored = {}, {}
    for alpha in alphas:
        records = []
        for t_index, template in enumerate(templates):
            for trial in template["trials"]:
                record = run_swap_trial(model, tok, template, trial, lens, alpha, layers, stored)
                records.append({"template": t_index, **record})
        results[str(alpha)] = records
    return results
