"""Computes and caches Jacobians (all 23 layers) for the windows the experiments need.

    .venv/bin/python -m src.compute_jacobians manifest          # every val, test and fit window
    .venv/bin/python -m src.compute_jacobians selected          # experiment 3's selected windows

Safe to stop and restart: each window is saved as it finishes, and finished windows are skipped.
Val and test come first, because the validation checks need them. Fit windows then alternate
between corpora, so an early stop still leaves balanced data.

--reverse walks the list from the other end, so a second process can share the work. Two
processes meeting in the middle at worst compute a window twice, and saves are atomic.
"""
from __future__ import annotations

import argparse
import itertools
import json
import time

from .corpus import CORPORA, load_windows
from .model import DATA, load_model
from .store import JacobianStore

ALL_LAYERS = list(range(23))
SELECTED_PATH = DATA / "selected_windows.json"  # written by src/experiments.py for experiment 3


def manifest_windows() -> list:
    first = [w for role in ("val", "test") for corpus in CORPORA for w in load_windows(corpus, role)]
    fit = [load_windows(corpus, "fit") for corpus in CORPORA]
    return first + [w for group in itertools.zip_longest(*fit) for w in group if w is not None]


def selected_windows() -> list:
    """Experiment 3's top candidates, as Window objects. Windows already cached are skipped."""
    from .features import candidate_windows

    selected = json.loads(SELECTED_PATH.read_text())
    return [w for corpus, indices in selected.items() for w in candidate_windows(corpus, indices)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("which", choices=["manifest", "selected"])
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--rows-per-pass", type=int, default=32)
    args = parser.parse_args()

    windows = manifest_windows() if args.which == "manifest" else selected_windows()
    if args.reverse:
        windows = windows[::-1]
    model, _ = load_model()
    store = JacobianStore(model, rows_per_pass=args.rows_per_pass)

    start, done = time.perf_counter(), 0
    for i, window in enumerate(windows):
        if store.path(window).exists():
            continue
        store.get(window, ALL_LAYERS)
        done += 1
        if done % 10 == 0:
            rate = (time.perf_counter() - start) / done
            print(f"{i + 1}/{len(windows)} {window.corpus}/{window.role}: {done} new, {rate:.1f}s per window", flush=True)
    print(f"finished: {len(windows)} windows, {done} new", flush=True)


if __name__ == "__main__":
    main()
