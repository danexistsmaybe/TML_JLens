"""Turns raw per-item scores into the results tables.

    .venv/bin/python -m analysis.summarize --split test

For each experiment and deployment domain this writes, in results/tables/ (or results/tables_val/):
  * exp<N>.md: one main table per domain, plus paired comparisons of the methods.
  * exp<N>.csv: every number, including each layer separately and confidence intervals.

Averaging: a lens with several seeds is averaged over seeds, item by item. Confidence intervals
are 95% percentile bootstraps that resample whole windows, whole pass@k prompts or whole swap
templates. Each comparison resamples the same items on both sides, so it is paired.
The seeds' pools overlap, so seed spread understates true variability.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.eval_prompts import BAND, KS, auc_of_rank

MAIN_LAYERS = ["12", "18"]
MAIN_EDIT_FRACTION = 0.05  # moved from 10% before test: see EXPERIMENT_1.md section 4.5 and DECISIONS.md
BAND_KEYS = [str(l) for l in BAND]

ROWS = {
    1: ["logit", "uniform_pool", "random_weights", "random_n", "similarity", "top_n", "oracle:wiki", "oracle:stories",
        "oracle500_in_pool"],
    2: ["logit", "uniform_pool", "random_weights", "random_n", "similarity", "top_n", "oracle:python", "oracle500_half_b"],
    3: ["logit", "oracle", "random_weights_100", "random_n_100", "similarity_100", "top_n_100",
        "random_weights_500", "random_n_500", "similarity_500", "top_n_500", "wider_selection"],
}
LABELS = {
    "logit": "Logit lens", "uniform_pool": "Uniform pool", "random_weights": "Random weights", "random_n": "Random n",
    "similarity": "**Similarity weighting**", "top_n": "**Top-n selection**", "oracle:wiki": "Wiki oracle (1000)",
    "oracle:stories": "Stories oracle (1000)", "oracle:python": "Python oracle (1000)",
    "oracle500_in_pool": "Domain oracle (500, in pool)", "oracle500_half_b": "Python oracle (500)", "oracle": "Uniform fit (1000)",
    "random_weights_100": "Random weights, ESS 100", "random_n_100": "Random n = 100",
    "similarity_100": "**Similarity, ESS 100**", "top_n_100": "**Top-n, n = 100**",
    "random_weights_500": "Random weights, ESS 500", "random_n_500": "Random n = 500",
    "similarity_500": "**Similarity, ESS 500**", "top_n_500": "**Top-n, n = 500**",
    "wider_selection": "**Wider selection (top 1000)**",
}
COMPARISONS = {
    1: [("similarity", "uniform_pool"), ("top_n", "uniform_pool"), ("similarity", "random_weights"), ("top_n", "random_n"),
        ("similarity", "own_oracle"), ("top_n", "own_oracle"), ("uniform_pool", "logit"), ("own_oracle", "other_oracle")],
    2: [("similarity", "uniform_pool"), ("top_n", "uniform_pool"), ("similarity", "random_weights"), ("top_n", "random_n"),
        ("similarity", "own_oracle"), ("top_n", "own_oracle"), ("uniform_pool", "logit"), ("own_oracle", "uniform_pool")],
    3: [("similarity_100", "random_weights_100"), ("top_n_100", "random_n_100"), ("similarity_500", "random_weights_500"),
        ("top_n_500", "random_n_500"), ("top_n_500", "oracle"), ("wider_selection", "oracle"), ("wider_selection", "top_n_500"),
        ("oracle", "logit")],
}

# Lower is better for these metrics; a table reader needs to know.
METRICS = {
    "jacobian_error": ("Jacobian error", "lower"),
    "pass10_target": ("Pass@10", "higher"),
    "pass10_control": ("Pass@10 control", "lower"),
    "pass10_gap": ("Pass@10 gap", "higher"),
    "pass50_gap": ("Pass@50 gap", "higher"),
    "auc_gap": ("AUC gap", "higher"),
    "swap_1": ("Swap, α=1", "higher"),
    "swap_2": ("Swap, α=2", "higher"),
    "swap_shift": ("Log-prob shift, α=1", "higher"),
    "overlap": ("Top-10 overlap", "higher"),
    "sym_kl": ("Symmetric KL", "lower"),
    "edit_error": ("Edit error", "lower"),
    "edit_antisym": ("Edit error, ± part", "lower"),
    "edit_corr": ("Vocabulary correlation", "higher"),
}
MAIN_COLUMNS = ["jacobian_error", "pass10_target", "pass10_control", "auc_gap", "swap_1", "overlap", "sym_kl", "edit_error"]


def row_key(name: str, domain: str) -> str:
    """'similarity[wiki]@s0' -> 'similarity'; 'oracle[stories]' -> 'oracle:stories' in experiments 1 and 2."""
    base = name.split("@")[0]
    match = re.fullmatch(r"(.+)\[(.+)\]", base)
    if not match:
        return base
    head, tag = match.groups()
    return f"oracle:{tag}" if head == "oracle" else head


def load_raw(root: Path, experiment: int, split: str) -> dict[str, dict[str, list[dict]]]:
    """{domain: {row key: [one result per seed]}}."""
    out = defaultdict(lambda: defaultdict(list))
    folder = root / f"exp{experiment}" / split
    for path in sorted(folder.glob("*/*.json")):
        result = json.loads(path.read_text())
        key = row_key(result["name"], result["domain"])
        if experiment == 3 and key.startswith("oracle:"):
            key = "oracle"
        out[result["domain"]][key].append(result)
    return out


# ---------------------------------------------------------------------------------------
# Per-item scores for one lens, averaged over its seeds. Each returns arrays indexed by item.
# ---------------------------------------------------------------------------------------

def item_scores(results: list[dict]) -> dict[str, dict]:
    """{metric: {"values": [items, ...], "stat": fn}} with item as the resampling unit.

    Each entry is an array whose first axis is the resampling unit, and a function from a
    resampled array to the number reported.
    """
    def over_seeds(fn):
        return np.mean([fn(r) for r in results], axis=0)

    scores = {}

    def add(name, array, stat=lambda a: float(np.mean(a))):
        scores[name] = (np.asarray(array, dtype=float), stat)

    # Jacobian error: per window, per layer.
    add("jacobian_error", over_seeds(lambda r: np.mean([r["jacobian_error"][l] for l in MAIN_LAYERS], axis=0)))
    for l in MAIN_LAYERS:
        add(f"jacobian_error_L{l}", over_seeds(lambda r: r["jacobian_error"][l]))

    # Pass@k: per prompt, best rank over the band or at one layer.
    def ranks(r, name, layers):
        return np.array([min(p[name][l] for l in layers) for p in r["pass"]["ranks"]])

    for suffix, layers in (("", BAND_KEYS), ("_L12", ["12"]), ("_L18", ["18"])):
        for k in KS:
            target = over_seeds(lambda r: ranks(r, "target", layers) < k)
            control = over_seeds(lambda r: ranks(r, "control", layers) < k)
            add(f"pass{k}_target{suffix}", target)
            add(f"pass{k}_control{suffix}", control)
            add(f"pass{k}_gap{suffix}", target - control)
        auc_t = over_seeds(lambda r: [auc_of_rank(int(x)) for x in ranks(r, "target", layers)])
        auc_c = over_seeds(lambda r: [auc_of_rank(int(x)) for x in ranks(r, "control", layers)])
        add(f"auc_target{suffix}", auc_t)
        add(f"auc_control{suffix}", auc_c)
        add(f"auc_gap{suffix}", auc_t - auc_c)

    # Python mixes two prompt types, which behave very differently.
    types = results[0]["pass"].get("types")
    if types and len(set(types)) > 1:
        for kind in sorted(set(types)):
            keep = np.array([t == kind for t in types])
            for k in KS:
                target = over_seeds(lambda r: ranks(r, "target", BAND_KEYS)[keep] < k)
                control = over_seeds(lambda r: ranks(r, "control", BAND_KEYS)[keep] < k)
                add(f"pass{k}_target_{kind}", target)
                add(f"pass{k}_control_{kind}", control)

    # Swaps: trials grouped by template. Stored as [templates, 2]: sum and count, to pool trials.
    def by_template(r, alpha, field):
        records = r["swap"][alpha]
        count = max(x["template"] for x in records) + 1
        sums = np.zeros(count)
        counts = np.zeros(count)
        for x in records:
            sums[x["template"]] += float(x[field])
            counts[x["template"]] += 1
        return np.stack([sums, counts], axis=1)

    pooled = lambda a: float(a[:, 0].sum() / a[:, 1].sum())
    add("swap_1", over_seeds(lambda r: by_template(r, "1.0", "pass")), pooled)
    add("swap_2", over_seeds(lambda r: by_template(r, "2.0", "pass")), pooled)
    add("swap_shift", over_seeds(lambda r: by_template(r, "1.0", "logp_shift")), pooled)
    add("swap_shift_2", over_seeds(lambda r: by_template(r, "2.0", "logp_shift")), pooled)
    add("swap_rank", over_seeds(lambda r: by_template(r, "1.0", "target_rank")), pooled)

    # Readout agreement: per window.
    if "readout" in results[0]:
        add("overlap", over_seeds(lambda r: np.mean([r["readout"][l]["overlap"] for l in MAIN_LAYERS], axis=0)))
        add("sym_kl", over_seeds(lambda r: np.mean([r["readout"][l]["sym_kl"] for l in MAIN_LAYERS], axis=0)))
        for l in MAIN_LAYERS:
            add(f"overlap_L{l}", over_seeds(lambda r: r["readout"][l]["overlap"]))
            add(f"sym_kl_L{l}", over_seeds(lambda r: r["readout"][l]["sym_kl"]))

    # Edits: records per window. Median over the whole pool of records.
    if "edits" in results[0]:
        median_pool = lambda a: float(np.median(a))

        def edit_values(r, layers, fraction, field):
            windows = len(r["edits"][layers[0]])
            return np.array([[x[field] for l in layers for x in r["edits"][l][w] if x["fraction"] == fraction]
                             for w in range(windows)])

        def antisymmetric(r, layers, fraction):
            """|(actual+ - actual-)/2 - predicted| / |predicted|, which cancels the edit's curvature."""
            rows = []
            for w in range(len(r["edits"][layers[0]])):
                row = []
                for l in layers:
                    plus = {x["position"]: x for x in r["edits"][l][w] if x["fraction"] == fraction and x["sign"] > 0}
                    minus = {x["position"]: x for x in r["edits"][l][w] if x["fraction"] == fraction and x["sign"] < 0}
                    for p, x in plus.items():
                        row.append(abs((x["actual"] - minus[p]["actual"]) / 2 - x["predicted"]) / max(abs(x["predicted"]), 1e-12))
                rows.append(row)
            return np.array(rows)

        fractions = sorted({x["fraction"] for x in results[0]["edits"][MAIN_LAYERS[0]][0]})
        main = MAIN_EDIT_FRACTION if MAIN_EDIT_FRACTION in fractions else fractions[0]
        for fraction in fractions:
            tag = "" if fraction == main else f"_{int(fraction * 100)}pct"
            for suffix, layers in [("", MAIN_LAYERS), *((f"_L{l}", [l]) for l in MAIN_LAYERS)]:
                add(f"edit_error{tag}{suffix}", over_seeds(lambda r: edit_values(r, layers, fraction, "relative_error")), median_pool)
                add(f"edit_antisym{tag}{suffix}", over_seeds(lambda r: antisymmetric(r, layers, fraction)), median_pool)
                add(f"edit_corr{tag}{suffix}", over_seeds(lambda r: edit_values(r, layers, fraction, "vocab_correlation")), median_pool)
    return scores


def bootstrap(a, stat, resamples: int, seed: int = 0):
    """Point estimate and percentile interval."""
    rng = np.random.default_rng(seed)
    n = len(a)
    values = [stat(a[rng.integers(0, n, n)]) for _ in range(resamples)]
    return stat(a), float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def paired_bootstrap(a, b, stat, resamples: int, seed: int = 0):
    """Difference stat(a) - stat(b), resampling the same items on both sides."""
    rng = np.random.default_rng(seed)
    n = len(a)
    values = []
    for _ in range(resamples):
        i = rng.integers(0, n, n)
        values.append(stat(a[i]) - stat(b[i]))
    return stat(a) - stat(b), float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def fmt(value, low=None, high=None, digits=3):
    if value is None:
        return "–"
    text = f"{value:.{digits}f}"
    if low is not None:
        text += f" ± {(high - low) / 2:.{digits}f}"
    return text


# ---------------------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------------------

def build_experiment(root, experiment, split, out_dir, seeds, bootstrap_n) -> list[Path]:
    raw = load_raw(root, experiment, split)
    if not raw:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [f"# Experiment {experiment} ({'test' if split == 'test' else 'validation'})", "",
             "Cells are the mean ± half the 95% bootstrap interval, resampling windows, prompts or templates. "
             "Seeded lenses are averaged over seeds first. Jacobian error, overlap, KL and edit error average layers 12 and 18. "
             "Pass@k takes the best layer in 9–22. Gap = target minus control. "
             f"Edit error is the median relative error at {int(MAIN_EDIT_FRACTION * 100)}% of activation norm. "
             "Lower is better for Jacobian error, control pass rate, KL and edit error."]
    rows_csv = []
    for domain, by_row in raw.items():
        scores = {key: item_scores(results) for key, results in by_row.items()}
        order = [k for k in ROWS[experiment] if k in scores]
        columns = [c for c in MAIN_COLUMNS if any(c in scores[k] for k in order)]
        lines += ["", f"## Deployed on {domain}", "",
                  "| Lens | " + " | ".join(METRICS[c][0] for c in columns) + " |",
                  "|---|" + "---|" * len(columns)]
        for key in order:
            cells = []
            for column in columns:
                if column in scores[key]:
                    a, stat = scores[key][column]
                    cells.append(fmt(*bootstrap(a, stat, bootstrap_n)))
                else:
                    cells.append("–")
            lines.append(f"| {LABELS.get(key, key)} | " + " | ".join(cells) + " |")
            for metric, (a, stat) in scores[key].items():
                value, low, high = bootstrap(a, stat, bootstrap_n)
                rows_csv.append({"experiment": experiment, "split": split, "domain": domain, "lens": key,
                                 "metric": metric, "value": value, "low": low, "high": high, "kind": "lens"})

        own = f"oracle:{domain}" if experiment != 3 else "oracle"
        other = next((f"oracle:{d}" for d in ("wiki", "stories") if d != domain and f"oracle:{d}" in scores), None)
        names = {"own_oracle": own, "other_oracle": other}
        pairs = [(names.get(a, a), names.get(b, b)) for a, b in COMPARISONS[experiment]]
        pairs = [(a, b) for a, b in pairs if a in scores and b in scores]
        compare_columns = [c for c in ("jacobian_error", "pass10_gap", "pass50_gap", "auc_gap", "swap_1", "swap_shift", "overlap", "edit_error")
                           if any(c in scores[a] and c in scores[b] for a, b in pairs)]
        lines += ["", f"### Paired differences, {domain}", "",
                  "First lens minus second, ± half the 95% interval. **Bold** intervals exclude zero.", "",
                  "| Comparison | " + " | ".join(METRICS[c][0] for c in compare_columns) + " |",
                  "|---|" + "---|" * len(compare_columns)]
        for a, b in pairs:
            cells = []
            for column in compare_columns:
                if column in scores[a] and column in scores[b]:
                    (x, stat), (y, _) = scores[a][column], scores[b][column]
                    value, low, high = paired_bootstrap(x, y, stat, bootstrap_n)
                    text = f"{value:+.3f} ± {(high - low) / 2:.3f}"
                    excludes = low > 0 or high < 0
                    cells.append(f"**{text}**" if excludes else text)
                    rows_csv.append({"experiment": experiment, "split": split, "domain": domain, "lens": f"{a} - {b}",
                                     "metric": column, "value": value, "low": low, "high": high, "kind": "difference"})
                else:
                    cells.append("–")
            label = f"{LABELS.get(a, a)} − {LABELS.get(b, b)}".replace("**", "")
            lines.append(f"| {label} | " + " | ".join(cells) + " |")

    md, table_csv = out_dir / f"exp{experiment}.md", out_dir / f"exp{experiment}.csv"
    md.write_text("\n".join(lines) + "\n")
    with table_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_csv[0]))
        writer.writeheader()
        writer.writerows(rows_csv)
    return [md, table_csv]


def build(root: Path, split: str, out_dir: Path, seeds, bootstrap: int = 2000) -> list[Path]:
    written = []
    for experiment in (1, 2, 3):
        written += build_experiment(root, experiment, split, out_dir, seeds, bootstrap)
    return written


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1] / "results"
    out = root / ("tables" if args.split == "test" else "tables_val")
    for path in build(root / "raw", args.split, out, [0, 1, 2], args.bootstrap):
        print("wrote", path)


if __name__ == "__main__":
    main()
