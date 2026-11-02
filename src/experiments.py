"""The three experiments: which lenses exist, how settings are chosen, and how lenses are scored.

    .venv/bin/python -m src.experiments features           # window features for all manifest windows
    .venv/bin/python -m src.experiments select             # choose ESS and n on validation
    .venv/bin/python -m src.experiments wider              # score broad candidates; writes the selection
    .venv/bin/python -m src.experiments run 1 --split val  # fit and score experiment 1 on validation
    .venv/bin/python -m src.experiments run 1 --split test # the same on test. Run once settings are frozen.
    .venv/bin/python -m src.experiments sensitivity        # lens quality against number of windows

Plans are in .agent/EXPERIMENT_1.md, _2.md and _3.md. In short:
  1. Pool of wiki + stories, deployed on wiki and on stories.
  2. The same pool, deployed on python, which it doesn't contain.
  3. Within one domain: a fit set of 1000, or the top 1000 of a much larger candidate set.

Raw scores go to results/raw/ as one JSON file per lens; analysis/ turns them into tables and plots.
A lens that already has results is skipped, so an interrupted run can simply be restarted.
Existing results are only overwritten with --force.
"""
from __future__ import annotations

import argparse
import gc
import json
import random
import time
from pathlib import Path

import torch

from . import weighting as W
from .check_prompts import DOMAINS, OUT as PROMPT_DIR
from .corpus import load_windows
from .eval_prompts import BAND, CORPUS, pass_ranks, prepare_pass, swap_results
from .features import cached_features, candidate_windows, load_candidates, window_ids
from .fitting import fit_many, load_stack
from .jlens import JLens
from .metrics import EDIT_FRACTIONS, edit_trials, jacobian_error, readout_agreement, typical_norms, window_activations
from .model import DATA, REPO, load_model

MAIN_LAYERS = [12, 18]
GRID = [10, 25, 50, 100, 150, 200, 300, 400, 500, 750]
SEEDS = [0, 1, 2]
POOL_PER_CORPUS = 500
FIT_SIZE = 1000
EXP3_BUDGETS = [100, 500]
SAMPLE_COUNTS = [10, 50, 100, 200, 500]

RESULTS = REPO / "results"
RAW = RESULTS / "raw"
SETTINGS_PATH = RESULTS / "settings.json"
SELECTED_PATH = DATA / "selected_windows.json"
EXP1_DOMAINS, EXP2_DOMAINS, EXP3_DOMAINS = ["wiki", "stories"], ["python"], ["wiki", "stories", "python"]


def log(*args):
    print(time.strftime("%H:%M:%S"), *args, flush=True)


# ---------------------------------------------------------------------------------------
# Windows, pools and centroids
# ---------------------------------------------------------------------------------------

def fit_windows(domain: str) -> list:
    return load_windows(CORPUS[domain], "fit", n=FIT_SIZE)


def validation_halves(domain: str) -> tuple[list, list]:
    """Two random halves of validation: one gives the centroid, the other chooses settings.

    Random, because manifest order follows position in the source stream, so a plain cut would
    give the halves different stretches of text.
    """
    windows = load_windows(CORPUS[domain], "val")
    random.Random(f"val-halves-{domain}").shuffle(windows)
    return windows[:25], windows[25:]


def pool_indices(seed: int) -> dict[str, list[int]]:
    """For each pool corpus, the fit-set positions of its 500 pool windows under this seed."""
    return {d: sorted(random.Random(f"pool-{seed}-{d}").sample(range(FIT_SIZE), POOL_PER_CORPUS)) for d in EXP1_DOMAINS}


def domain_universe() -> list:
    """Every fit window of wiki, stories and python. Weight vectors index into this list."""
    return [w for d in DOMAINS for w in fit_windows(d)]


def universe_slice(domain: str) -> range:
    start = DOMAINS.index(domain) * FIT_SIZE
    return range(start, start + FIT_SIZE)


def pool_positions(seed: int) -> list[int]:
    """Positions in the universe of this seed's pool."""
    chosen = pool_indices(seed)
    return [universe_slice(d)[i] for d in EXP1_DOMAINS for i in chosen[d]]


def expand(partial: torch.Tensor, positions: list[int], size: int) -> torch.Tensor:
    full = torch.zeros(size, dtype=torch.float64)
    full[positions] = partial.double()
    return full


def features_of(model, domain: str, layers) -> dict:
    """Fit features and the two validation centroid sources, per layer."""
    fit = cached_features(model, f"{domain}_fit", window_ids(fit_windows(domain)), layers)
    centroid_half, _ = validation_halves(domain)
    mu = cached_features(model, f"{domain}_mu", window_ids(centroid_half), layers)
    return {"fit": {l: fit[l][:FIT_SIZE] for l in layers}, "mu": {l: mu[l].mean(0) for l in layers}}


# ---------------------------------------------------------------------------------------
# Choosing ESS and n on validation
# ---------------------------------------------------------------------------------------

def pool_scores(model, domain: str, seed: int, layers, centered: bool = True) -> dict[int, torch.Tensor]:
    """The deployment domain's similarity score for each window of this seed's pool, per layer."""
    chosen = pool_indices(seed)
    deploy = features_of(model, domain, layers)
    pool_features = {l: torch.cat([features_of(model, d, layers)["fit"][l][chosen[d]] for d in EXP1_DOMAINS]) for l in layers}
    return {l: W.scores(pool_features[l], deploy["mu"][l], pool_features[l].mean(0) if centered else None) for l in layers}


def select_settings(model):
    """Jacobian error on the settings half of validation, across seeds, for each ESS and n."""
    universe = domain_universe()
    stacks = {}
    for layer in MAIN_LAYERS:
        positions = sorted({p for s in SEEDS for p in pool_positions(s)})
        log(f"loading layer {layer} for {len(positions)} pool windows")
        stack = load_stack([universe[p] for p in positions], layer).float()
        stacks[layer] = (stack, {p: i for i, p in enumerate(positions)})

    settings = {"grid": GRID, "domains": {}, "chosen_on": "validation Jacobian error, layers 12 and 18, mean over seeds"}
    for domain in DOMAINS:
        _, settings_half = validation_halves(domain)
        targets = {l: load_stack(settings_half, l).float() for l in MAIN_LAYERS}
        target_norm = {l: targets[l].norm(dim=-1) for l in MAIN_LAYERS}
        table = {"similarity": {g: [] for g in GRID}, "top_n": {g: [] for g in GRID},
                 "uniform": [], "weight_share": {g: {} for g in GRID}}
        for seed in SEEDS:
            positions = pool_positions(seed)
            scores = pool_scores(model, domain, seed, MAIN_LAYERS)

            rows = {l: stacks[l][0][[stacks[l][1][p] for p in positions]] for l in MAIN_LAYERS}

            def error_of(weights_by_layer):
                errors = []
                for l in MAIN_LAYERS:
                    J = weights_by_layer[l].float() @ rows[l]
                    errors.append(float(((J[None] - targets[l]).norm(dim=-1) / target_norm[l]).mean()))
                return sum(errors) / len(errors)

            table["uniform"].append(error_of({l: torch.full((len(positions),), 1 / len(positions)) for l in MAIN_LAYERS}))
            for g in GRID:
                sim = {l: W.similarity_weights(scores[l], g) for l in MAIN_LAYERS}
                top = {l: W.top_n_weights(scores[l], g) for l in MAIN_LAYERS}
                table["similarity"][g].append(error_of(sim))
                table["top_n"][g].append(error_of(top))
                # Share of weight on each pool corpus (first half of the pool is wiki).
                wiki_sim = sum(float(sim[l][:POOL_PER_CORPUS].sum()) for l in MAIN_LAYERS) / len(MAIN_LAYERS)
                wiki_top = sum(float(top[l][:POOL_PER_CORPUS].sum()) for l in MAIN_LAYERS) / len(MAIN_LAYERS)
                table["weight_share"][g].setdefault("similarity_wiki", []).append(wiki_sim)
                table["weight_share"][g].setdefault("top_n_wiki", []).append(wiki_top)
        mean = lambda xs: sum(xs) / len(xs)
        best_ess = min(GRID, key=lambda g: mean(table["similarity"][g]))
        best_n = min(GRID, key=lambda g: mean(table["top_n"][g]))
        settings["domains"][domain] = {"ess": best_ess, "n": best_n, "table": table}
        log(domain, "ESS", best_ess, "n", best_n, "uniform error", round(mean(table["uniform"]), 4),
            "similarity", round(mean(table["similarity"][best_ess]), 4), "top_n", round(mean(table["top_n"][best_n]), 4))
    RESULTS.mkdir(exist_ok=True)
    SETTINGS_PATH.write_text(json.dumps(settings, indent=1))


# ---------------------------------------------------------------------------------------
# Lens definitions. Each returns {name: {layer: [N] weights over the universe}}.
# ---------------------------------------------------------------------------------------

def uniform_over(positions, size: int) -> torch.Tensor:
    return expand(torch.ones(len(positions)), list(positions), size)


def pool_lenses(model, domains: list[str], settings: dict, size: int, include_seeded_oracles: bool) -> dict:
    """Lenses over the mixed pool, shared by experiments 1 and 2."""
    specs = {}
    for seed in SEEDS:
        positions = pool_positions(seed)
        specs[f"uniform_pool@s{seed}"] = {l: uniform_over(positions, size) for l in BAND}
        for domain in domains:
            ess_value, n_value = settings["domains"][domain]["ess"], settings["domains"][domain]["n"]
            centered = pool_scores(model, domain, seed, BAND)
            plain = pool_scores(model, domain, seed, BAND, centered=False)
            sim = {l: W.similarity_weights(centered[l], ess_value) for l in BAND}
            specs[f"similarity[{domain}]@s{seed}"] = {l: expand(sim[l], positions, size) for l in BAND}
            specs[f"top_n[{domain}]@s{seed}"] = {l: expand(W.top_n_weights(centered[l], n_value), positions, size) for l in BAND}
            specs[f"random_weights[{domain}]@s{seed}"] = {l: expand(W.shuffled(sim[l], seed), positions, size) for l in BAND}
            specs[f"random_n[{domain}]@s{seed}"] = {
                l: expand(W.random_n_weights(len(positions), n_value, seed), positions, size) for l in BAND}
            # Private sanity check: the same methods without centering. Not in the main tables.
            specs[f"uncentered_similarity[{domain}]@s{seed}"] = {
                l: expand(W.similarity_weights(plain[l], ess_value), positions, size) for l in BAND}
            specs[f"uncentered_top_n[{domain}]@s{seed}"] = {
                l: expand(W.top_n_weights(plain[l], n_value), positions, size) for l in BAND}
        if include_seeded_oracles:
            chosen = pool_indices(seed)
            for domain in EXP1_DOMAINS:
                inside = [universe_slice(domain)[i] for i in chosen[domain]]
                outside = sorted(set(universe_slice(domain)) - set(inside))
                specs[f"oracle500_in_pool[{domain}]@s{seed}"] = {l: uniform_over(inside, size) for l in BAND}
                specs[f"oracle500_disjoint[{domain}]@s{seed}"] = {l: uniform_over(outside, size) for l in BAND}
    return specs


def experiment_specs(model, experiment: int, universe_size: int) -> dict:
    settings = json.loads(SETTINGS_PATH.read_text())
    if experiment == 1:
        specs = pool_lenses(model, EXP1_DOMAINS, settings, universe_size, include_seeded_oracles=True)
        for domain in EXP1_DOMAINS:
            specs[f"oracle[{domain}]"] = {l: uniform_over(universe_slice(domain), universe_size) for l in BAND}
        return specs
    if experiment == 2:
        specs = pool_lenses(model, EXP2_DOMAINS, settings, universe_size, include_seeded_oracles=False)
        specs["oracle[python]"] = {l: uniform_over(universe_slice("python"), universe_size) for l in BAND}
        order = list(universe_slice("python"))
        random.Random("python-halves").shuffle(order)
        specs["oracle500_half_a[python]"] = {l: uniform_over(sorted(order[:500]), universe_size) for l in BAND}
        specs["oracle500_half_b[python]"] = {l: uniform_over(sorted(order[500:]), universe_size) for l in BAND}
        return specs
    raise ValueError(experiment)


def experiment3_specs(model, domain: str, universe_size: int, wider: list[int]) -> dict:
    """wider: universe positions (see experiment3_universe) of the domain's top 1000 candidates."""
    positions = list(universe_slice(domain))
    feats = features_of(model, domain, BAND)
    score = {l: W.scores(feats["fit"][l], feats["mu"][l], feats["fit"][l].mean(0)) for l in BAND}
    specs = {f"oracle[{domain}]": {l: uniform_over(positions, universe_size) for l in BAND},
             f"wider_selection[{domain}]": {l: uniform_over(wider, universe_size) for l in BAND}}
    for budget in EXP3_BUDGETS:
        sim = {l: W.similarity_weights(score[l], budget) for l in BAND}
        specs[f"similarity_{budget}[{domain}]"] = {l: expand(sim[l], positions, universe_size) for l in BAND}
        specs[f"top_n_{budget}[{domain}]"] = {l: expand(W.top_n_weights(score[l], budget), positions, universe_size) for l in BAND}
        for seed in SEEDS:
            specs[f"random_weights_{budget}[{domain}]@s{seed}"] = {l: expand(W.shuffled(sim[l], seed), positions, universe_size) for l in BAND}
            specs[f"random_n_{budget}[{domain}]@s{seed}"] = {
                l: expand(W.random_n_weights(FIT_SIZE, budget, seed), positions, universe_size) for l in BAND}
    return specs


# ---------------------------------------------------------------------------------------
# Experiment 3: scoring broad candidates
# ---------------------------------------------------------------------------------------

def score_candidates(model):
    """Rank each domain's candidates by similarity, averaged over layers 12 and 18. Keep the top 1000."""
    selected = {}
    for domain in EXP3_DOMAINS:
        corpus = CORPUS[domain]
        ids = load_candidates(corpus)["windows"]
        log(domain, "scoring", len(ids), "candidates")
        features = cached_features(model, f"{domain}_candidates", ids, MAIN_LAYERS)
        mu = features_of(model, domain, MAIN_LAYERS)["mu"]
        score = sum(W.scores(features[l], mu[l], features[l].mean(0)) for l in MAIN_LAYERS) / len(MAIN_LAYERS)
        selected[corpus] = score.topk(FIT_SIZE).indices.tolist()
    SELECTED_PATH.write_text(json.dumps(selected))


def experiment3_universe() -> tuple[list, dict[str, list[int]]]:
    """The fit universe, then every selected window; and each domain's selected positions."""
    universe = domain_universe()
    selected = json.loads(SELECTED_PATH.read_text())
    index = {w.sha256: i for i, w in enumerate(universe)}
    positions = {}
    for domain in EXP3_DOMAINS:
        positions[domain] = []
        for window in candidate_windows(CORPUS[domain], selected[CORPUS[domain]]):
            if window.sha256 not in index:
                index[window.sha256] = len(universe)
                universe.append(window)
            positions[domain].append(index[window.sha256])
    return universe, positions


# ---------------------------------------------------------------------------------------
# Scoring one lens on one domain
# ---------------------------------------------------------------------------------------

class Evaluator:
    """Holds what doesn't depend on the lens: prompts, windows, activations."""

    def __init__(self, model, tok, domain: str, split: str, with_edits: bool = True):
        self.model, self.tok, self.domain, self.split = model, tok, domain, split
        pass_sets = json.loads((PROMPT_DIR / "pass_at_k.json").read_text())[domain]
        self.swaps = json.loads((PROMPT_DIR / "swap.json").read_text())[domain][split]
        self.items = pass_sets[split]
        self.prepared = prepare_pass(model, tok, self.items)
        self.windows = load_windows(CORPUS[domain], split) if split == "test" else validation_halves(domain)[1]
        self.activations = window_activations(model, self.windows, MAIN_LAYERS)
        reference_acts = window_activations(model, load_windows(CORPUS[domain], "val"), MAIN_LAYERS)
        norms = typical_norms(reference_acts)  # fixed on validation windows, for every split
        self.alphas = {l: {f: f * norms[l] for f in EDIT_FRACTIONS} for l in MAIN_LAYERS}
        self.with_edits = with_edits

    def score(self, lens: JLens, reference: JLens | None) -> dict:
        model, tok = self.model, self.tok
        start = time.perf_counter()
        out = {"jacobian_error": jacobian_error(lens, self.windows, MAIN_LAYERS)}
        out["pass"] = {"items": [i["intermediate"] for i in self.items], "ranks": pass_ranks(model, self.prepared, lens),
                       "types": [i["type"] for i in self.items]}
        out["swap"] = swap_results(model, tok, self.swaps, lens)
        if self.with_edits:
            out["edits"] = {l: [edit_trials(model, lens, w, l, self.alphas[l]) for w in self.windows] for l in MAIN_LAYERS}
        if reference is not None:
            out["readout"] = readout_agreement(model, lens, reference, self.activations)
        out["seconds_to_score"] = time.perf_counter() - start
        return out


def raw_path(experiment: int, split: str, domain: str, name: str) -> Path:
    safe = name.replace("[", "_").replace("]", "").replace("@", "_")
    return RAW / f"exp{experiment}" / split / domain / f"{safe}.json"


def on_gpu(lens: JLens) -> JLens:
    lens.jacobians = {l: J.cuda() for l, J in lens.jacobians.items()}
    return lens


def lens_applies(name: str, domain: str) -> bool:
    """A lens tied to one domain's centroid only runs on that domain. Fixed lenses run everywhere."""
    if "[" not in name:
        return True
    tag = name.split("[")[1].split("]")[0]
    if name.startswith(("oracle[", "oracle500")) and "half" not in name:
        return True if name.startswith("oracle[") else tag == domain
    return tag == domain


def reference_for(name_seed: str | None, domain: str, experiment: int, lenses: dict) -> JLens:
    """The lens each readout is compared with. It never shares windows with the pool."""
    if experiment == 2:
        return lenses["oracle500_half_a[python]"]
    return lenses[f"oracle500_disjoint[{domain}]@{name_seed or 's0'}"]


def run_experiment(model, tok, experiment: int, split: str, force: bool = False):
    universe = domain_universe()
    domains = {1: EXP1_DOMAINS, 2: EXP2_DOMAINS}[experiment]
    specs = experiment_specs(model, experiment, len(universe))
    log(f"experiment {experiment}: {len(specs)} lenses; fitting")
    start = time.perf_counter()
    lenses = fit_many(universe, specs, BAND, log=log)
    fit_seconds = time.perf_counter() - start
    lenses["logit"] = JLens.identity(BAND, model.config.hidden_size)
    for lens in lenses.values():
        on_gpu(lens)
    log(f"fit in {fit_seconds:.0f}s")

    for domain in domains:
        evaluator = None
        for name, lens in lenses.items():
            if not lens_applies(name, domain):
                continue
            path = raw_path(experiment, split, domain, name)
            if path.exists() and not force:
                continue
            evaluator = evaluator or Evaluator(model, tok, domain, split)
            seed = name.split("@")[1] if "@" in name else None
            result = evaluator.score(lens, reference_for(seed, domain, experiment, lenses))
            result.update(name=name, domain=domain, split=split, experiment=experiment,
                          seconds_to_fit_all=fit_seconds, lenses_in_fit=len(specs))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result))
            log(f"exp{experiment} {split} {domain} {name}: {result['seconds_to_score']:.0f}s")
    del lenses
    gc.collect()
    torch.cuda.empty_cache()


def run_experiment3(model, tok, split: str, force: bool = False):
    universe, wider = experiment3_universe()
    specs = {}
    for domain in EXP3_DOMAINS:
        specs.update(experiment3_specs(model, domain, len(universe), wider[domain]))
    log(f"experiment 3: {len(specs)} lenses; fitting")
    start = time.perf_counter()
    lenses = fit_many(universe, specs, BAND, log=log)
    fit_seconds = time.perf_counter() - start
    lenses["logit"] = JLens.identity(BAND, model.config.hidden_size)
    for lens in lenses.values():
        on_gpu(lens)

    for domain in EXP3_DOMAINS:
        evaluator = None
        for name, lens in lenses.items():
            if name != "logit" and f"[{domain}]" not in name:
                continue
            path = raw_path(3, split, domain, name)
            if path.exists() and not force:
                continue
            evaluator = evaluator or Evaluator(model, tok, domain, split)
            result = evaluator.score(lens, None)
            result.update(name=name, domain=domain, split=split, experiment=3,
                          seconds_to_fit_all=fit_seconds, lenses_in_fit=len(specs))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result))
            log(f"exp3 {split} {domain} {name}: {result['seconds_to_score']:.0f}s")


def run_sensitivity(model, tok, split: str):
    """Oracle lens quality against the number of windows, refit from the cache."""
    universe = domain_universe()
    specs = {}
    for domain in DOMAINS:
        for seed in SEEDS:
            rng = random.Random(f"sensitivity-{seed}-{domain}")
            order = list(universe_slice(domain))
            rng.shuffle(order)
            for count in SAMPLE_COUNTS:
                specs[f"count_{count}[{domain}]@s{seed}"] = {l: uniform_over(order[:count], len(universe)) for l in BAND}
        specs[f"count_1000[{domain}]"] = {l: uniform_over(universe_slice(domain), len(universe)) for l in BAND}
    lenses = fit_many(universe, specs, BAND, log=log)
    for lens in lenses.values():
        on_gpu(lens)
    for domain in DOMAINS:
        evaluator = None
        for name, lens in lenses.items():
            if f"[{domain}]" not in name:
                continue
            path = raw_path(0, split, domain, name)
            if path.exists():
                continue
            evaluator = evaluator or Evaluator(model, tok, domain, split, with_edits=False)
            result = {"name": name, "domain": domain, "jacobian_error": jacobian_error(lens, evaluator.windows, MAIN_LAYERS),
                      "pass": {"ranks": pass_ranks(model, evaluator.prepared, lens)}}
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result))
            log("sensitivity", name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["features", "select", "wider", "run", "sensitivity"])
    parser.add_argument("experiment", nargs="?", type=int, choices=[1, 2, 3])
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--force", action="store_true", help="overwrite existing results")
    args = parser.parse_args()
    model, tok = load_model()
    if args.command == "features":
        for domain in DOMAINS:
            features_of(model, domain, BAND)
            log(domain, "features done")
    elif args.command == "select":
        select_settings(model)
    elif args.command == "wider":
        score_candidates(model)
    elif args.command == "sensitivity":
        run_sensitivity(model, tok, args.split)
    elif args.experiment == 3:
        run_experiment3(model, tok, args.split, args.force)
    else:
        run_experiment(model, tok, args.experiment, args.split, args.force)


if __name__ == "__main__":
    main()
