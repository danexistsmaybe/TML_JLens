"""The checks that must pass on validation before test is opened. Writes results/validation_checks.md.

    .venv/bin/python -m analysis.validation_checks

Needs: results/settings.json and validation results for experiments 1 and 2.

  1. Positive control: do the metrics separate two very different lenses (the two oracles in
     experiment 1; the python oracle and the uniform pool in experiment 2)?
  2. Short prompts or a neutral prefix: lenses never fit on positions 0-15, yet short prompts are
     read there. Rule, fixed before looking: switch only if the prefix raises the own-domain
     oracle's target-minus-control pass@10 by at least 20 points.
  3. Surviving swap trials per domain.
  4. Share of weight on each pool corpus at each setting, and the chosen settings.
  5. Edit error by size, so that the main size can be justified.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from analysis.summarize import MAIN_LAYERS, item_scores, load_raw, paired_bootstrap
from src import experiments as E
from src.check_prompts import DOMAINS, OUT as PROMPT_DIR
from src.eval_prompts import BAND, CORPUS, pass_ranks, pass_summary, prepare_pass, swap_results
from src.fitting import fit_many
from src.model import load_model

PREFIXES = {
    "wiki": "The following text is a short passage from an encyclopedia, written for readers who want to learn something new.\n\n",
    "stories": "The following text is a short story for children, written for readers who like to hear about small adventures.\n\n",
    "python": "# The following code is a short example of Python, written for readers who want to learn something new.\n\n",
}
PREFIX_THRESHOLD = 0.20
PAIRS = {("exp1", "wiki"): ("oracle:wiki", "oracle:stories"), ("exp1", "stories"): ("oracle:stories", "oracle:wiki"),
         ("exp2", "python"): ("oracle:python", "uniform_pool")}
CHECK_METRICS = ["jacobian_error", "pass10_gap", "pass50_gap", "auc_gap", "swap_1", "swap_2", "swap_shift",
                 "overlap", "sym_kl", "edit_error", "edit_antisym"]


def positive_control(root: Path) -> list[str]:
    lines = ["## 1. Positive control", "",
             "First lens minus second, with a 95% interval. A metric that can't separate these can't detect a smaller gain.", "",
             "| Deployed on | Comparison | Metric | Difference | Interval | Separates? |", "|---|---|---|---|---|---|"]
    for (exp, domain), (a, b) in PAIRS.items():
        raw = load_raw(root, int(exp[-1]), "val")[domain]
        x, y = item_scores(raw[a]), item_scores(raw[b])
        for metric in CHECK_METRICS:
            if metric not in x or metric not in y:
                continue
            value, low, high = paired_bootstrap(x[metric][0], y[metric][0], x[metric][1], 2000)
            lines.append(f"| {domain} | {a} − {b} | {metric} | {value:+.3f} | [{low:+.3f}, {high:+.3f}] "
                         f"| {'yes' if low > 0 or high < 0 else 'no'} |")
    return lines


def prefix_check(model, tok) -> tuple[list[str], bool]:
    universe = E.domain_universe()
    specs = {d: {l: E.uniform_over(E.universe_slice(d), len(universe)) for l in BAND} for d in DOMAINS}
    lenses = fit_many(universe, specs, BAND)
    sets = json.loads((PROMPT_DIR / "pass_at_k.json").read_text())
    swaps = json.loads((PROMPT_DIR / "swap.json").read_text())
    lines = ["## 2. Short prompts or a neutral prefix", "",
             f"Own-domain oracle on validation prompts. Rule: switch only if pass@10 gap rises by at least {int(PREFIX_THRESHOLD * 100)} points.", "",
             "| Domain | Form | Pass@10 | Control | Gap | Pass@50 | Control | AUC | Swap α=1 |", "|---|---|---|---|---|---|---|---|---|"]
    switch = False
    for domain in DOMAINS:
        lens = E.on_gpu(lenses[domain])
        items = sets[domain]["val"]
        gaps = {}
        for form, prefix in (("short", ""), ("prefix", PREFIXES[domain])):
            s = pass_summary(pass_ranks(model, prepare_pass(model, tok, items, prefix=prefix), lens))
            templates = [{**t, "template": prefix + t["template"]} for t in swaps[domain]["val"]]
            swap = swap_results(model, tok, templates, lens, alphas=(1.0,))["1.0"]
            rate = sum(r["pass"] for r in swap) / len(swap)
            t, c = s["target"], s["control"]
            gaps[form] = t["pass@10"] - c["pass@10"]
            lines.append(f"| {domain} | {form} | {t['pass@10']:.2f} | {c['pass@10']:.2f} | {gaps[form]:+.2f} | {t['pass@50']:.2f} "
                         f"| {c['pass@50']:.2f} | {t['auc']:.2f} | {rate:.2f} |")
        if gaps["prefix"] - gaps["short"] >= PREFIX_THRESHOLD:
            switch = True
    lines += ["", f"**Decision: {'use the prefix' if switch else 'keep short prompts'}.**"]
    return lines, switch


def trials() -> list[str]:
    swaps = json.loads((PROMPT_DIR / "swap.json").read_text())
    lines = ["## 3. Surviving swap trials", "", "| Domain | Validation | Test |", "|---|---|---|"]
    for domain in DOMAINS:
        counts = {s: sum(len(t["trials"]) for t in swaps[domain][s]) for s in ("val", "test")}
        lines.append(f"| {domain} | {counts['val']} | {counts['test']} |")
    return lines


def settings_table() -> list[str]:
    settings = json.loads(E.SETTINGS_PATH.read_text())
    mean = lambda xs: float(np.mean(xs))
    lines = ["## 4. Settings and weight share", "",
             "Chosen by Jacobian error on the second half of validation, averaged over layers 12 and 18 and seeds. "
             "Share is the weight on wiki windows; stories get the rest.", ""]
    for domain, s in settings["domains"].items():
        table = s["table"]
        lines += [f"### {domain}: ESS {s['ess']}, n {s['n']}. Uniform pool error {mean(table['uniform']):.4f}", "",
                  "| ESS or n | Similarity error | Top-n error | Similarity wiki share | Top-n wiki share |", "|---|---|---|---|---|"]
        for g in settings["grid"]:
            key = str(g)
            lines.append(f"| {g} | {mean(table['similarity'][key]):.4f} | {mean(table['top_n'][key]):.4f} "
                         f"| {mean(table['weight_share'][key]['similarity_wiki']):.2f} | {mean(table['weight_share'][key]['top_n_wiki']):.2f} |")
        lines.append("")
    return lines


def edit_sizes(root: Path) -> list[str]:
    lines = ["## 5. Edit error by size", "", "Own-domain oracle. The ± part cancels the edit's curvature.", "",
             "| Domain | Layer | Error at 5% | Error at 10% | ± part at 5% | ± part at 10% | Correlation at 10% |", "|---|---|---|---|---|---|---|"]
    for (exp, domain), (own, _) in PAIRS.items():
        s = item_scores(load_raw(root, int(exp[-1]), "val")[domain][own])
        for l in MAIN_LAYERS:
            get = lambda name: s[name][1](s[name][0])
            lines.append(f"| {domain} | {l} | {get(f'edit_error_L{l}'):.3f} | {get(f'edit_error_10pct_L{l}'):.3f} "
                         f"| {get(f'edit_antisym_L{l}'):.3f} | {get(f'edit_antisym_10pct_L{l}'):.3f} | {get(f'edit_corr_10pct_L{l}'):.3f} |")
    return lines


def main():
    model, tok = load_model()
    root = E.RAW
    prefix_lines, switch = prefix_check(model, tok)
    lines = ["# Validation checks", "", *positive_control(root), "", *prefix_lines, "", *trials(), "", *settings_table(), *edit_sizes(root)]
    path = E.RESULTS / "validation_checks.md"
    path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("prefix switch:", switch)


if __name__ == "__main__":
    main()
