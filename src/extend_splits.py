"""Grows each corpus's fit set to 1000 windows, and saves every usable fit window as a candidate.

    .venv/bin/python -m src.extend_splits

Existing manifest entries never change: new windows are appended after them, and val and test
are left alone. build_splits() can't be rerun for this, because all its choices share one random
generator, so asking for more windows would change the windows already drawn.

To find the same token streams, this replays that generator up to the point where it chose the
5,000 TinyStories fit stories, then checks that the replay reproduces the manifest exactly.

Output:
  * data/splits.json: fit grows to 1000 for wikitext, tinystories and python.
  * data/wikitext_candidates.pt, data/tinystories_candidates.pt: every non-overlapping fit
    window, in the format of data/python_candidates.pt. Ignored by git; rerunning recreates them.

Experiment 3 scores all candidates. Rerunning this file is safe: it gives the same result.
"""
from __future__ import annotations

import hashlib
import json
import random

import torch

from .corpus import SPLITS_PATH, WINDOW_TOKENS
from .model import DATA, MODEL_ID

FIT_SIZE = 1000
TINYSTORIES_FILE = "data/train-00000-of-00004-2d5a1467fff1081b.parquet"


def window_hash(ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16]


def candidates_from_stream(ids: list[int]) -> dict:
    """Every non-overlapping 128-token window of a stream, as build_splits cuts them."""
    starts = list(range(0, len(ids) - WINDOW_TOKENS + 1, WINDOW_TOKENS))
    return {"windows": torch.tensor([ids[s:s + WINDOW_TOKENS] for s in starts]), "offsets": torch.tensor(starts)}


def replay_streams(manifest: dict, tok) -> dict[str, list[int]]:
    """The wikitext and tinystories fit token streams, rebuilt and checked against the manifest."""
    from datasets import load_dataset

    counts = manifest["stream_window_counts"]
    rng = random.Random(manifest["seed"])
    # Each cut() call drew from a range of this many starts. Only the length affects the draw.
    for split, n in (("train", 200), ("validation", 50), ("test", 50)):
        rng.sample(range(counts[f"wikitext/{split}"]), n)

    wiki = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    stories = load_dataset("roneneldan/TinyStories", split="train", data_files={"train": TINYSTORIES_FILE})
    train_idx = sorted(rng.sample(range(len(stories)), 5000))

    streams = {"wikitext": tok("".join(wiki["text"]), add_special_tokens=False)["input_ids"],
               "tinystories": tok("\n\n".join(stories[i]["text"].strip() for i in train_idx),
                                  add_special_tokens=False)["input_ids"]}
    for corpus, stream in streams.items():
        for e in manifest["corpora"][corpus]["fit"][:200]:
            window = stream[e["token_offset"]:e["token_offset"] + WINDOW_TOKENS]
            assert window == e["input_ids"], f"replay of {corpus} does not match the manifest"
    return streams


def extend(entries: list[dict], candidates: dict, source: str, name: str, seed: int) -> list[dict]:
    """entries, followed by a seeded sample of unused candidates, up to FIT_SIZE."""
    used_offsets = {e["token_offset"] for e in entries}
    used_hashes = {e["sha256"] for e in entries}
    order = list(range(len(candidates["offsets"])))
    random.Random(f"{seed}-extend-{name}").shuffle(order)
    out = list(entries)
    for i in order:
        if len(out) >= FIT_SIZE:
            break
        offset = int(candidates["offsets"][i])
        ids = candidates["windows"][i].tolist()
        digest = window_hash(ids)
        if offset in used_offsets or digest in used_hashes:
            continue
        used_offsets.add(offset)
        used_hashes.add(digest)
        out.append({"source": source, "token_offset": offset, "input_ids": ids, "sha256": digest})
    assert len(out) == FIT_SIZE, f"{name}: only {len(out)} windows"
    return out


def main():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    manifest = json.loads(SPLITS_PATH.read_text())
    streams = replay_streams(manifest, tok)

    candidates = {corpus: candidates_from_stream(stream) for corpus, stream in streams.items()}
    for corpus, c in candidates.items():
        torch.save({"fit": c}, DATA / f"{corpus}_candidates.pt")
    candidates["python"] = torch.load(DATA / "python_candidates.pt", weights_only=True)["fit"]

    for corpus, c in candidates.items():
        entries = manifest["corpora"][corpus]["fit"]
        source = entries[0]["source"]
        manifest["corpora"][corpus]["fit"] = extend(entries, c, source, corpus, manifest["seed"])
        print(corpus, "candidates:", len(c["offsets"]), "fit:", len(manifest["corpora"][corpus]["fit"]), flush=True)
    SPLITS_PATH.write_text(json.dumps(manifest))


if __name__ == "__main__":
    main()
