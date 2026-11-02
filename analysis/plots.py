"""Preliminary figures from the result tables. Writes results/figures/ and results/tables/sensitivity.md.

    .venv/bin/python -m analysis.plots

Experiment figures show each lens with its 95% interval. A lens's color gives its role: gray for
baselines, light gray for random controls, blue and orange for our two methods, aqua for the oracle.
"""
from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.eval_prompts import auc_of_rank

ROOT = Path(__file__).resolve().parents[1] / "results"
TABLES, FIGURES = ROOT / "tables", ROOT / "figures"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
BASELINE, CONTROL, SIMILARITY, TOP_N, ORACLE = "#52514e", "#a3a29c", "#2a78d6", "#eb6834", "#1baf7a"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False, "axes.titlesize": 10,
    "axes.titleweight": "bold", "figure.facecolor": "white", "axes.facecolor": "white", "font.family": "DejaVu Sans",
})

METRICS = [("jacobian_error", "Jacobian error\n(lower is better)"), ("overlap", "Top-10 overlap with\ndomain oracle"),
           ("edit_error", "± edit error, 5%\n(lower is better)"), ("swap_1", "Swap success, α = 1"),
           ("swap_2", "Swap success, α = 2"), ("auc_gap", "Pass@k AUC,\ntarget − control")]
ROWS_12 = [("logit", "Logit lens", BASELINE), ("uniform_pool", "Uniform pool", BASELINE), ("random_weights", "Random weights", CONTROL),
           ("random_n", "Random n", CONTROL), ("similarity", "Similarity weighting", SIMILARITY), ("top_n", "Top-n selection", TOP_N),
           ("OWN", "Domain oracle", ORACLE)]
ROWS_3 = [("logit", "Logit lens", BASELINE), ("oracle", "Uniform fit (1000)", ORACLE), ("random_n_100", "Random n = 100", CONTROL),
          ("top_n_100", "Top-n, n = 100", TOP_N), ("random_weights_100", "Random weights, ESS 100", CONTROL), ("similarity_100", "Similarity, ESS 100", SIMILARITY),
          ("random_n_500", "Random n = 500", CONTROL), ("top_n_500", "Top-n, n = 500", TOP_N), ("random_weights_500", "Random weights, ESS 500", CONTROL),
          ("similarity_500", "Similarity, ESS 500", SIMILARITY), ("wider_selection", "Wider selection", "#6250d6")]


def read_table(experiment: int) -> dict:
    path = TABLES / f"exp{experiment}.csv"
    if not path.exists():
        return {}
    out = {}
    for row in csv.DictReader(path.open()):
        out[row["domain"], row["lens"], row["metric"]] = tuple(float(row[k]) for k in ("value", "low", "high"))
    return out


def clean(ax):
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)


def lens_figure(tables: list[tuple[dict, str]], rows, metrics, path: Path, title: str):
    """Rows of lenses, one panel per metric and domain. tables: [(table, domain)]."""
    fig, axes = plt.subplots(len(tables), len(metrics), figsize=(2.0 * len(metrics) + 1.6, 0.27 * len(rows) * len(tables) + 1.3),
                             sharey=True, squeeze=False)
    for i, (table, domain) in enumerate(tables):
        for j, (metric, label) in enumerate(metrics):
            ax = axes[i][j]
            for y, (key, _, color) in enumerate(rows):
                key = f"oracle:{domain}" if key == "OWN" else key
                if (domain, key, metric) not in table:
                    continue
                value, low, high = table[domain, key, metric]
                ax.plot([low, high], [y, y], color=color, linewidth=2, solid_capstyle="round", zorder=2)
                ax.plot(value, y, "o", markersize=6, color=color, markeredgecolor="white", markeredgewidth=1, zorder=3)
            ax.set_yticks(range(len(rows)), [r[1] for r in rows])
            clean(ax)
            if i == 0:
                ax.set_title(label, fontsize=9, fontweight="normal", color=MUTED)
        axes[i][0].set_ylabel(domain.capitalize(), fontsize=10, fontweight="bold", color=INK)
        axes[i][0].set_ylim(len(rows) - 0.5, -0.5)  # the first lens on top; shared across the row
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def difference_figure(table1: dict, table2: dict, path: Path):
    """Does each method beat the uniform pool, metric by metric? Both methods and all domains."""
    metrics = [("jacobian_error", "Jacobian error\n(negative is better)"), ("overlap", "Top-10 overlap"), ("edit_error", "± edit error\n(negative is better)"),
               ("swap_1", "Swap success, α = 1"), ("pass50_gap", "Pass@50 gap"), ("auc_gap", "Pass@k AUC gap")]
    groups = [("wiki", table1), ("stories", table1), ("python", table2)]
    methods = [("similarity", "Similarity", SIMILARITY), ("top_n", "Top-n", TOP_N)]
    fig, axes = plt.subplots(1, len(metrics), figsize=(11.5, 2.9), sharey=True)
    for ax, (metric, label) in zip(axes, metrics):
        for g, (domain, table) in enumerate(groups):
            for m, (key, name, color) in enumerate(methods):
                entry = table.get((domain, f"{key} - uniform_pool", metric))
                if entry is None:
                    continue
                value, low, high = entry
                y = g + (m - 0.5) * 0.3
                ax.plot([low, high], [y, y], color=color, linewidth=2, solid_capstyle="round", zorder=2)
                ax.plot(value, y, "o", markersize=6, color=color, markeredgecolor="white", markeredgewidth=1, zorder=3,
                        label=name if g == 0 else None)
        ax.axvline(0, color=MUTED, linewidth=1, zorder=1)
        ax.set_yticks(range(3), [g[0].capitalize() + (" (new domain)" if g[0] == "python" else "") for g in groups])
        ax.set_ylim(2.6, -0.6)
        ax.set_title(label, fontsize=9, fontweight="normal", color=MUTED)
        clean(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False, ncol=2, bbox_to_anchor=(0.995, 1.0))
    fig.suptitle("Our methods minus the uniform pool, with 95% paired intervals (test)", x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def settings_figure(path: Path):
    settings = json.loads((ROOT / "settings.json").read_text())
    grid = settings["grid"]
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.9))
    for ax, (domain, s) in zip(axes, settings["domains"].items()):
        table = s["table"]
        for key, name, color, chosen in (("similarity", "Similarity (by ESS)", SIMILARITY, s["ess"]), ("top_n", "Top-n (by n)", TOP_N, s["n"])):
            y = [np.mean(table[key][str(g)]) for g in grid]
            ax.plot(grid, y, color=color, linewidth=2, marker="o", markersize=4, label=name)
            ax.plot(chosen, np.mean(table[key][str(chosen)]), "o", markersize=9, markerfacecolor="white", markeredgecolor=color, markeredgewidth=2, zorder=4)
        uniform = np.mean(table["uniform"])
        ax.axhline(uniform, color=BASELINE, linewidth=1.5, linestyle="--", label="Uniform pool")
        ax.set_xscale("log")
        ax.set_xticks([10, 50, 100, 500], ["10", "50", "100", "500"])
        ax.minorticks_off()
        ax.set_title(domain.capitalize(), loc="left")
        ax.set_xlabel("Effective windows (ESS or n)")
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", colors=MUTED, length=0)
    axes[0].set_ylabel("Jacobian error, validation")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, loc="upper right", ncol=3, bbox_to_anchor=(0.995, 1.0))
    fig.suptitle("Choosing settings: large circles mark the setting used on test", x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def sensitivity():
    """Oracle quality against the number of fit windows. Returns {domain: {count: stats}}."""
    folder = ROOT / "raw" / "exp0" / "test"
    data = defaultdict(lambda: defaultdict(list))
    for path in folder.glob("*/*.json"):
        r = json.loads(path.read_text())
        count = int(re.match(r"count_(\d+)", r["name"]).group(1))
        band = [min(p["target"].values()) for p in r["pass"]["ranks"]]
        control = [min(p["control"].values()) for p in r["pass"]["ranks"]]
        data[r["domain"]][count].append({
            "jacobian_error": float(np.mean([np.mean(r["jacobian_error"][l]) for l in ("12", "18")])),
            "pass50_gap": float(np.mean([t < 50 for t in band]) - np.mean([c < 50 for c in control])),
            "auc_gap": float(np.mean([auc_of_rank(t) for t in band]) - np.mean([auc_of_rank(c) for c in control])),
        })
    return data


def sensitivity_outputs(path: Path):
    data = sensitivity()
    if not data:
        return
    lines = ["# Sample-count sensitivity (test)", "", "Own-domain uniform lens, fit on fewer windows. Means over 3 random subsets (one fit at 1000). Jacobian error averages layers 12 and 18.", "",
             "| Domain | Windows | Jacobian error | Pass@50 gap | AUC gap |", "|---|---|---|---|---|"]
    colors = {"wiki": SIMILARITY, "stories": TOP_N, "python": ORACLE}
    fig, ax = plt.subplots(figsize=(5.2, 3.1))
    for domain in ("wiki", "stories", "python"):
        counts = sorted(data[domain])
        y = []
        for c in counts:
            rows = data[domain][c]
            means = {k: np.mean([r[k] for r in rows]) for k in rows[0]}
            y.append(means["jacobian_error"])
            lines.append(f"| {domain} | {c} | {means['jacobian_error']:.3f} | {means['pass50_gap']:+.2f} | {means['auc_gap']:+.3f} |")
        ax.plot(counts, y, color=colors[domain], linewidth=2, marker="o", markersize=4)
        ax.annotate(domain.capitalize(), (counts[-1], y[-1]), xytext=(6, 0), textcoords="offset points", va="center", color=INK, fontsize=9)
    ax.set_xscale("log")
    ax.set_xticks(counts, [str(c) for c in counts])
    ax.minorticks_off()
    ax.set_xlim(right=2400)
    ax.set_xlabel("Windows in the lens")
    ax.set_ylabel("Jacobian error, test")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", colors=MUTED, length=0)
    ax.set_title("Jacobian error levels off by about 50 windows", loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    (TABLES / "sensitivity.md").write_text("\n".join(lines) + "\n")


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    table1, table2, table3 = read_table(1), read_table(2), read_table(3)
    if table1 and table2:
        lens_figure([(table1, "wiki"), (table1, "stories")], ROWS_12, METRICS, FIGURES / "exp1_lenses.png",
                    "Experiment 1: known domains. Methods reach the oracle on fit and readouts, but not pass@k or swaps (test, 95% intervals)")
        lens_figure([(table2, "python")], ROWS_12, METRICS, FIGURES / "exp2_lenses.png",
                    "Experiment 2: new domain. Methods help, but stay far from the python oracle (test, 95% intervals)")
        difference_figure(table1, table2, FIGURES / "methods_vs_uniform.png")
    settings_figure(FIGURES / "settings_choice.png")
    sensitivity_outputs(FIGURES / "sample_count.png")
    if table3:
        metrics3 = [m for m in METRICS if m[0] != "overlap"]
        lens_figure([(table3, d) for d in ("wiki", "stories", "python")], ROWS_3, metrics3, FIGURES / "exp3_lenses.png",
                    "Experiment 3: within a domain (test, 95% intervals)")


if __name__ == "__main__":
    main()
