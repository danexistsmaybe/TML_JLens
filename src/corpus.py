"""Fitting corpora: where the windows come from, and how they are weighted.

A lens is fit from WINDOWS: chunks of exactly 128 tokens of plain text. Each window
contributes one Jacobian J_l^(w) (src/jlens.py), and the lens is their weighted average:

    J_l = sum_w weight_w * J_l^(w)        (weights >= 0, normalized to sum to 1)

So "the fitting corpus" is fully described by (a list of windows, a weight per window).
This file provides both halves:

  1. WINDOWS: the split manifest data/splits.json holds, per corpus, 200 "fit", 50 "val"
     and 50 "test" windows (token ids + provenance). It was built once, before any
     fitting, by build_splits() below; load_windows() reads it.
       wikitext    : WikiText-2 raw. fit <- train, val <- validation, test <- test.
       tinystories : TinyStories. fit <- train shard 0 (5,000 seeded stories);
                     val/test <- official validation split, partitioned BY STORY.
       python      : real Python files, split BY PROJECT; 500 fit windows. Built separately
                     by src/python_corpus.py, which also explains its cleaning steps.
  2. WEIGHTING SCHEMES: a weighting scheme is any function
         scheme(windows: list[Window]) -> torch.Tensor of shape [len(windows)], >= 0
     Weights are computed BEFORE any Jacobian is touched, so a scheme can only look at
     the windows themselves (their text, tokens, corpus label, or model activations you
     compute from them). Extra settings go in through functools.partial or a closure.
     To try a new scheme: write it here, pass it to store.fit_lens(..., weights=scheme(windows)).
     Per-window Jacobians are cached on disk, so re-weighting never recomputes them.
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass

import torch

from .model import DATA, MODEL_ID

SPLITS_PATH = DATA / "splits.json"
WINDOW_TOKENS = 128
CORPORA = ("wikitext", "tinystories", "python")
ROLES = ("fit", "val", "test")


@dataclass(frozen=True)
class Window:
    """One 128-token chunk of a corpus."""
    corpus: str            # "wikitext" or "tinystories"
    role: str              # "fit", "val" or "test"
    index: int             # position in the manifest list for (corpus, role)
    input_ids: tuple[int, ...]
    source: str            # dataset split it was cut from, e.g. "wikitext/train"
    token_offset: int      # where it starts in that split's token stream
    sha256: str            # short hash of the token ids; also the Jacobian cache key

    def ids_tensor(self, device="cuda") -> torch.Tensor:
        return torch.tensor([self.input_ids], device=device)

    def text(self, tok) -> str:
        return tok.decode(list(self.input_ids))


# ---------------------------------------------------------------------------------------
# 1. Windows
# ---------------------------------------------------------------------------------------

def load_windows(corpus: str, role: str = "fit", n: int | None = None, seed: int | None = None) -> list[Window]:
    """Windows of one corpus and role from the manifest.

    n: how many to take (None = all). seed: if given, take a random sample of n
    (reproducible); otherwise the first n in manifest order.
    """
    manifest = json.loads(SPLITS_PATH.read_text())
    entries = manifest["corpora"][corpus][role]
    windows = [Window(corpus, role, i, tuple(e["input_ids"]), e["source"], e["token_offset"], e["sha256"])
               for i, e in enumerate(entries)]
    if n is None:
        return windows
    if seed is None:
        return windows[:n]
    return random.Random(seed).sample(windows, n)


def build_splits(seed: int = 0, n_per_role: dict | None = None):
    """(Re)build data/splits.json. Run once, before any fitting; already done (2026-10-16).

    Each corpus split is concatenated into one token stream (no special tokens), cut into
    non-overlapping 128-token windows, and windows are sampled with a fixed seed.
    WikiText lines are joined in order; TinyStories stories are joined with a blank line.
    Rebuilding with the same seed and library versions reproduces the same windows.
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer

    n_per_role = n_per_role or {"fit": 200, "val": 50, "test": 50}
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    rng = random.Random(seed)

    def cut(text, n, source):
        ids = tok(text, add_special_tokens=False)["input_ids"]
        starts = sorted(rng.sample(range(0, len(ids) - WINDOW_TOKENS + 1, WINDOW_TOKENS), n))
        return [{"source": source, "token_offset": s, "input_ids": ids[s:s + WINDOW_TOKENS],
                 "sha256": hashlib.sha256(json.dumps(ids[s:s + WINDOW_TOKENS]).encode()).hexdigest()[:16]}
                for s in starts]

    manifest = {"seed": seed, "window_tokens": WINDOW_TOKENS, "model": MODEL_ID, "corpora": {}}

    wt = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    manifest["corpora"]["wikitext"] = {
        role: cut("".join(wt[split]["text"]), n_per_role[role], f"wikitext/{split}")
        for role, split in [("fit", "train"), ("val", "validation"), ("test", "test")]}

    ts = load_dataset("roneneldan/TinyStories", data_files={
        "train": "data/train-00000-of-00004-2d5a1467fff1081b.parquet",
        "validation": "data/validation-00000-of-00001-869c898b519ad725.parquet"})
    train_idx = sorted(rng.sample(range(len(ts["train"])), 5000))
    val_perm = list(range(len(ts["validation"])))
    rng.shuffle(val_perm)
    half = len(val_perm) // 2
    parts = {"fit": ("train", train_idx), "val": ("validation", sorted(val_perm[:half])),
             "test": ("validation", sorted(val_perm[half:]))}
    manifest["corpora"]["tinystories"] = {
        role: cut("\n\n".join(ts[split][i]["text"].strip() for i in idx), n_per_role[role],
                  f"tinystories/{split}/{role}")
        for role, (split, idx) in parts.items()}

    for corpus, roles in manifest["corpora"].items():  # no window may appear in two roles
        seen = {}
        for role, entries in roles.items():
            for e in entries:
                assert seen.setdefault(e["sha256"], role) == role, f"window shared across roles in {corpus}"
    SPLITS_PATH.write_text(json.dumps(manifest))


# ---------------------------------------------------------------------------------------
# 2. Weighting schemes: (windows) -> weights [N]
# ---------------------------------------------------------------------------------------

def uniform(windows: list[Window]) -> torch.Tensor:
    """Every window counts equally: the paper's plain average."""
    return torch.ones(len(windows))


def corpus_mix(windows: list[Window], mix: dict[str, float]) -> torch.Tensor:
    """Each corpus gets a total share of the weight, split evenly among its windows.
    Example: corpus_mix(ws, {"wikitext": 0.5, "tinystories": 0.5}) balances the two
    corpora even if they contribute different numbers of windows."""
    counts = {c: sum(w.corpus == c for w in windows) for c in mix}
    return torch.tensor([mix.get(w.corpus, 0.0) / counts[w.corpus] if counts.get(w.corpus) else 0.0
                         for w in windows])


def random_weights(windows: list[Window], seed: int = 0) -> torch.Tensor:
    """Random positive weights (an ablation: does *any* reweighting change the lens?)."""
    g = torch.Generator().manual_seed(seed)
    return torch.rand(len(windows), generator=g)
