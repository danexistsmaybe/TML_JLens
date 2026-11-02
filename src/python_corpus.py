"""Builds the python domain: 128-token windows of real Python code.

    .venv/bin/python -m src.python_corpus

Source: one shard of codeparrot/codeparrot-clean (Python files from GitHub, deduplicated).
Output, leaving the wikitext and tinystories entries of the manifest untouched:
  * data/splits.json gains corpora["python"] with fit, val and test windows, like the others.
  * data/python_candidates.pt holds every usable window per role, for selecting from a
    large pool (experiment 3). It is ignored by git; rerunning this file recreates it.

Code needs more care than prose in three ways, and each has a step below:
  1. BOILERPLATE. License text is everywhere: in leading comments, in docstrings, and
     further down the file. strip_boilerplate() removes it from files, and
     usable_window() rejects any window where some still survives.
  2. COPIED CODE. Repositories vendor and fork each other, so one file can appear under
     many repository names (e.g. Django inside some app's site-packages). Windows from
     near-identical files in fit and test would leak. We drop vendored paths, split by
     project, and drop any file that mostly repeats lines we have already kept.
  3. A FEW HUGE SOURCES. Ansible alone is 5% of the shard, so files per project are capped,
     and so are windows per file.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import random
import re
from collections import defaultdict

import torch

from .corpus import SPLITS_PATH, WINDOW_TOKENS
from .model import DATA, MODEL_ID

DATASET = "codeparrot/codeparrot-clean"
SHARD = "file-000000000001.json.gz"
REVISION = "main"  # the commit actually used is recorded in the manifest
SEED = 0

N_MANIFEST = {"fit": 500, "val": 50, "test": 50}  # windows stored in data/splits.json
TARGET_WINDOWS = {"fit": 20_000, "val": 5_000, "test": 5_000}  # candidates per role
ROLE_SHARE = {"fit": 0.8, "val": 0.1, "test": 0.1}  # share of PROJECTS sent to each role

MAX_FILES_PER_PROJECT = 20
MIN_FILE_TOKENS = 20         # skip near-empty files (e.g. an __init__.py with one import)
MAX_FILE_TOKENS = 50 * WINDOW_TOKENS  # one huge file can supply at most 50 windows
SHINGLE_LINES = 5            # a "shingle" is this many consecutive non-blank lines
MAX_REPEATED_SHINGLES = 0.1  # drop a file if more than this share of its shingles were seen
MIN_UNIQUE_TOKENS = 32       # reject windows that barely vary, such as large data tables

LICENSE_WORDS = re.compile(
    r"copyright|\(c\) *\d{4}|licen[sc]ed (under|to)|licen[sc]e[,:]|permission is hereby"
    r"|redistribution and use|redistribute it|all rights reserved|warrant(y|ies)"
    r"|general public licen[sc]e|apache licen[sc]e|mit licen[sc]e|bsd licen[sc]e"
    r"|spdx-license", re.IGNORECASE)
VENDORED_PATH = re.compile(
    r"(^|/)(site-packages|dist-packages|vendor|vendored|third[_-]?party|thirdparty"
    r"|node_modules|virtualenv|venv|env/lib|lib/python[\d.]*|build/lib|external|extern)(/|$)",
    re.IGNORECASE)
GENERATED_PATH = re.compile(r"(^|/)migrations/", re.IGNORECASE)  # Django's generated schemas
LEADING_STRING = re.compile(r"""\A[rRuUbB]{0,2}(\"\"\"|''')(.*?)\1[ \t]*\n?""", re.DOTALL)


# ---------------------------------------------------------------------------------------
# Cleaning one file
# ---------------------------------------------------------------------------------------

def strip_boilerplate(source: str) -> str:
    """Remove license text and file headers from Python source."""
    # 1. The top of the file. Repeatedly remove blank lines, comments, and any docstring
    #    containing license wording. A docstring without license wording is real
    #    documentation, so it stops the loop. Shebang and encoding lines are comments too.
    text = source
    while True:
        stripped = text.lstrip()
        if stripped.startswith("#"):
            newline = stripped.find("\n")
            text = "" if newline < 0 else stripped[newline + 1:]
            continue
        match = LEADING_STRING.match(stripped)
        if match and LICENSE_WORDS.search(match.group(2)):
            text = stripped[match.end():]
            continue
        text = stripped
        break

    # 2. The rest of the file. Remove any block of consecutive comment lines that
    #    contains license wording, such as a header placed after the module docstring.
    kept, block = [], []

    def flush_block():
        if not LICENSE_WORDS.search("\n".join(block)):
            kept.extend(block)
        block.clear()

    for line in text.split("\n"):
        if line.lstrip().startswith("#"):
            block.append(line)
        else:
            flush_block()
            kept.append(line)
    flush_block()
    return "\n".join(kept).strip() + "\n"


def shingles(text: str) -> set[bytes]:
    """Hashes of every run of SHINGLE_LINES consecutive non-blank lines (ignoring indentation)."""
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    # blake2b rather than hash(): Python's hash() of a string changes from run to run.
    return {hashlib.blake2b("\n".join(lines[i:i + SHINGLE_LINES]).encode(), digest_size=8).digest()
            for i in range(len(lines) - SHINGLE_LINES + 1)}


def project_of(repo_name: str) -> str:
    """The project behind a repository: "someuser/Django" -> "django", so forks stay together."""
    return repo_name.split("/")[-1].lower()


def role_of(project: str) -> str:
    """Assign a project to one role, from a hash of its name (stable across runs)."""
    digest = hashlib.sha256(f"{SEED}:{project}".encode()).digest()
    u = int.from_bytes(digest[:8], "big") / 2**64
    total = 0.0
    for role, share in ROLE_SHARE.items():
        total += share
        if u < total:
            return role
    return "test"


def usable_window(ids: list[int], text: str) -> bool:
    """A window must contain no leftover license text and enough distinct tokens."""
    return len(set(ids)) >= MIN_UNIQUE_TOKENS and not LICENSE_WORDS.search(text)


# ---------------------------------------------------------------------------------------
# Building the corpus
# ---------------------------------------------------------------------------------------

def load_files(path) -> tuple[dict[str, list[dict]], dict]:
    """Read the shard. Returns ({role: shuffled files}, counts of what was filtered out)."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        files = [json.loads(line) for line in f]
    stats = {"shard_files": len(files), "autogenerated": 0, "vendored_path": 0,
             "generated_path": 0, "over_project_cap": 0}

    by_project = defaultdict(list)
    for file in files:
        if str(file["autogenerated"]) == "True":
            stats["autogenerated"] += 1
        elif VENDORED_PATH.search(file["path"]):
            stats["vendored_path"] += 1
        elif GENERATED_PATH.search(file["path"]):
            stats["generated_path"] += 1
        else:
            by_project[project_of(file["repo_name"])].append(file)

    rng = random.Random(SEED)
    per_role = {role: [] for role in ROLE_SHARE}
    for project in sorted(by_project):
        group = by_project[project]
        rng.shuffle(group)
        stats["over_project_cap"] += max(0, len(group) - MAX_FILES_PER_PROJECT)
        per_role[role_of(project)].extend(group[:MAX_FILES_PER_PROJECT])
    for role_files in per_role.values():
        rng.shuffle(role_files)
    return per_role, stats


def cut_role(tok, files, target_windows, seen, stats) -> tuple[torch.Tensor, torch.Tensor]:
    """Add cleaned files to one token stream until it holds enough usable windows.

    `seen` holds line shingles from every file kept so far, across roles, so a later file
    that mostly repeats earlier code is dropped. Returns ([N, 128] windows, [N] offsets).
    """
    stream, windows, offsets, next_start, used = [], [], [], 0, set()
    for file in files:
        text = strip_boilerplate(file["content"])
        file_shingles = shingles(text)
        if file_shingles and len(file_shingles & seen) / len(file_shingles) > MAX_REPEATED_SHINGLES:
            stats["repeated_code"] = stats.get("repeated_code", 0) + 1
            continue
        ids = tok(text + "\n", add_special_tokens=False)["input_ids"]
        if len(ids) < MIN_FILE_TOKENS:
            stats["too_short"] = stats.get("too_short", 0) + 1
            continue
        seen |= file_shingles
        stats["files_used"] = stats.get("files_used", 0) + 1
        if len(ids) > MAX_FILE_TOKENS:
            stats["truncated_files"] = stats.get("truncated_files", 0) + 1
        stream.extend(ids[:MAX_FILE_TOKENS])

        # Non-overlapping windows, as in the other corpora; rejected windows leave a gap.
        while next_start + WINDOW_TOKENS <= len(stream):
            window = stream[next_start:next_start + WINDOW_TOKENS]
            if tuple(window) in used:
                stats["duplicate_windows"] = stats.get("duplicate_windows", 0) + 1
            elif usable_window(window, tok.decode(window)):
                used.add(tuple(window))
                windows.append(window)
                offsets.append(next_start)
            else:
                stats["rejected_windows"] = stats.get("rejected_windows", 0) + 1
            next_start += WINDOW_TOKENS
        if len(windows) >= target_windows:
            break
    assert len(windows) >= target_windows, f"only {len(windows)} windows; use more shards"
    return torch.tensor(windows[:target_windows]), torch.tensor(offsets[:target_windows])


def main():
    from huggingface_hub import HfApi, hf_hub_download
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    commit = HfApi().dataset_info(DATASET, revision=REVISION).sha
    path = hf_hub_download(DATASET, SHARD, repo_type="dataset", revision=commit)

    per_role, stats = load_files(path)
    seen: set[bytes] = set()
    candidates, role_stats = {}, {}
    for role in ROLE_SHARE:  # fit first, so val and test are checked against it
        role_stats[role] = {"files_available": len(per_role[role])}
        windows, offsets = cut_role(tok, per_role[role], TARGET_WINDOWS[role], seen, role_stats[role])
        candidates[role] = {"windows": windows, "offsets": offsets}
        print(role, role_stats[role], "windows:", len(windows), flush=True)

    # The manifest's windows are a seeded sample of the candidates.
    rng = random.Random(SEED)
    corpus = {}
    for role, n in N_MANIFEST.items():
        chosen = sorted(rng.sample(range(len(candidates[role]["windows"])), n))
        corpus[role] = []
        for i in chosen:
            ids = candidates[role]["windows"][i].tolist()
            corpus[role].append({"source": f"python/{role}",
                                 "token_offset": int(candidates[role]["offsets"][i]),
                                 "input_ids": ids,
                                 "sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16]})

    manifest = json.loads(SPLITS_PATH.read_text())
    manifest["corpora"]["python"] = corpus
    for role in ROLE_SHARE:
        manifest["stream_window_counts"][f"python/{role}"] = len(candidates[role]["windows"])
    manifest["python_source"] = {
        "dataset": DATASET, "commit": commit, "shard": SHARD, "seed": SEED,
        "max_files_per_project": MAX_FILES_PER_PROJECT, "max_repeated_shingles": MAX_REPEATED_SHINGLES,
        "min_unique_tokens": MIN_UNIQUE_TOKENS, "stats": stats, "role_stats": role_stats}
    SPLITS_PATH.write_text(json.dumps(manifest))
    torch.save(candidates, DATA / "python_candidates.pt")
    print("filtered:", stats)


if __name__ == "__main__":
    main()
