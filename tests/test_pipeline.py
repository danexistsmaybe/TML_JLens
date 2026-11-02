"""Tests for the experiment pipeline. Validation prompts and windows only; test sets stay untouched.

    .venv/bin/python -m tests.test_pipeline            # everything
    .venv/bin/python -m tests.test_pipeline weighting  # names containing "weighting"

Each check prints PASS or FAIL with the numbers it saw. Exits nonzero if any fail.
Needs: the val windows' Jacobians, and the first 30 fit windows per domain.
"""
from __future__ import annotations

import json
import subprocess
import sys
import traceback

import torch

from src import experiments as E
from src import weighting as W
from src.check_prompts import DOMAINS, OUT as PROMPT_DIR
from src.corpus import CORPORA, load_windows
from src.eval_prompts import (BAND, CORPUS, auc_of_rank, pass_ranks, pass_summary, prepare_pass, swap_results,
                              swapped_logits, lens_vectors, word_tokens)
from src.features import candidate_windows, load_candidates, mean_residuals, window_ids
from src.fitting import fit_many, load_jacobians, load_stack
from src.jlens import JLens, valid_positions, window_jacobians
from src.metrics import edit_positions, edit_trials, jacobian_error, readout_agreement, typical_norms, window_activations
from src.model import load_model, record_residuals, residuals

CHECKS = []


def check(fn):
    CHECKS.append(fn)
    return fn


class Context:
    model = tok = None


def small_windows(domain: str, n: int = 30):
    return load_windows(CORPUS[domain], "fit", n=n)


# ---------------------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------------------

@check
def data_no_window_in_two_roles():
    seen = {}
    for corpus in CORPORA:
        for role in ("fit", "val", "test"):
            for w in load_windows(corpus, role):
                assert w.sha256 not in seen, f"{w.sha256} is in {seen[w.sha256]} and {corpus}/{role}"
                seen[w.sha256] = f"{corpus}/{role}"
    return f"{len(seen)} windows, all unique"


@check
def data_existing_windows_unchanged():
    old = json.loads(subprocess.check_output(["git", "show", "bd4e0b2:data/splits.json"]))
    new = json.loads(E.REPO.joinpath("data/splits.json").read_text())
    for corpus in old["corpora"]:
        for role, entries in old["corpora"][corpus].items():
            assert new["corpora"][corpus][role][:len(entries)] == entries, (corpus, role)
    return "every window from the committed manifest is unchanged, at the same index"


@check
def data_candidates_match_manifest_and_exclude_heldout():
    out = []
    for corpus in CORPORA:
        c = load_candidates(corpus)
        fit = load_windows(corpus, "fit")
        by_offset = {int(o): i for i, o in enumerate(c["offsets"])}
        windows = candidate_windows(corpus, [by_offset[w.token_offset] for w in fit[:50]])
        assert [w.sha256 for w in windows] == [w.sha256 for w in fit[:50]], corpus
        held_out = {tuple(w.input_ids) for role in ("val", "test") for w in load_windows(corpus, role)}
        overlap = sum(tuple(row.tolist()) in held_out for row in c["windows"])
        assert overlap == 0, f"{corpus}: {overlap} candidates are in val or test"
        out.append(f"{corpus} {len(c['windows'])}")
    return "candidates: " + ", ".join(out)


@check
def data_pools_and_halves():
    for seed in E.SEEDS:
        chosen = E.pool_indices(seed)
        assert all(len(set(v)) == 500 for v in chosen.values())
        assert chosen == E.pool_indices(seed), "pools must be reproducible"
        positions = E.pool_positions(seed)
        assert len(set(positions)) == 1000
    assert E.pool_indices(0) != E.pool_indices(1)
    universe = E.domain_universe()
    for domain in DOMAINS:
        slice_ = E.universe_slice(domain)
        assert all(universe[i].corpus == CORPUS[domain] for i in (slice_[0], slice_[-1]))
        a, b = E.validation_halves(domain)
        assert len(a) == 25 and len(b) == 25 and not ({w.sha256 for w in a} & {w.sha256 for w in b})
    return "pools 500 + 500 per seed, reproducible; validation halves disjoint"


@check
def prompts_splits_share_no_word_and_val_is_pinned():
    sets = json.loads((PROMPT_DIR / "pass_at_k.json").read_text())
    swaps = json.loads((PROMPT_DIR / "swap.json").read_text())
    from src.prompt_drafts import VAL_WORDS
    for domain in DOMAINS:
        val = {i["intermediate"].lower() for i in sets[domain]["val"]}
        test = {i["intermediate"].lower() for i in sets[domain]["test"]}
        assert not (val & test), (domain, val & test)
        assert val <= {w.lower() for w in VAL_WORDS[domain]}
        for item in sets[domain]["test"]:
            assert item["intermediate"].lower() not in item["prompt"].lower(), item["prompt"]
            assert item["control"].lower() not in item["prompt"].lower(), item["prompt"]
        val_t = {t["template"] for t in swaps[domain]["val"]}
        assert not (val_t & {t["template"] for t in swaps[domain]["test"]})
        assert sum(len(t["trials"]) for t in swaps[domain]["test"]) >= 50, domain
    return "; ".join(f"{d}: {len(sets[d]['val'])}+{len(sets[d]['test'])}" for d in DOMAINS)


# ---------------------------------------------------------------------------------------
# Weighting
# ---------------------------------------------------------------------------------------

@check
def weighting_ess_targets_are_hit():
    torch.manual_seed(0)
    score = torch.randn(1000) * 0.2
    worst = 0.0
    for target in E.GRID:
        weights = W.similarity_weights(score, target)
        worst = max(worst, abs(W.ess(weights) - target) / target)
        assert abs(float(weights.sum()) - 1) < 1e-9
        assert (weights.argsort() == score.argsort()).all(), "weights must follow the scores"
        assert abs(W.ess(W.shuffled(weights, 3)) - W.ess(weights)) < 1e-9
        assert abs(W.ess(W.top_n_weights(score, target)) - target) < 1e-9
        assert abs(W.ess(W.random_n_weights(1000, target, 1)) - target) < 1e-9
    assert worst < 2e-3, worst
    return f"worst relative ESS miss {worst:.1e}"


@check
def weighting_scores_center_and_select_top():
    features = torch.tensor([[2.0, 1.0], [2.0, -1.0], [0.0, 1.0]])
    mu = torch.tensor([2.0, 1.0])
    plain = W.scores(features, mu, None)
    assert abs(float(plain[0]) - 1) < 1e-6
    centered = W.scores(features, mu, torch.tensor([2.0, 0.0]))  # mu - center = (0, 1)
    assert abs(float(centered[0]) - 1) < 1e-6 and abs(float(centered[1]) + 1) < 1e-6
    top = W.top_n_weights(torch.tensor([0.1, 0.9, 0.5, 0.2]), 2)
    assert top.tolist() == [0.0, 0.5, 0.5, 0.0]
    return "cosine, centering and top-n match hand-worked answers"


@check
def weighting_similarity_separates_real_domains():
    """On real features, a stories centroid should prefer stories windows, and so on."""
    shares = {}
    for domain in E.EXP1_DOMAINS:
        for layer in E.MAIN_LAYERS:
            score = E.pool_scores(Context.model, domain, 0, [layer])[layer]
            top = W.top_n_weights(score, 100)
            wiki = float(top[:500].sum())
            shares[f"{domain}/L{layer}"] = round(wiki if domain == "wiki" else 1 - wiki, 3)
            assert shares[f"{domain}/L{layer}"] > 0.9, shares
    return f"share of top-100 from the deployment domain: {shares}"


@check
def lens_specs_have_the_right_structure():
    model = Context.model
    size = len(E.domain_universe())
    settings = {"domains": {d: {"ess": 200, "n": 100} for d in DOMAINS}}
    specs = E.pool_lenses(model, E.EXP1_DOMAINS, settings, size, include_seeded_oracles=True)
    assert len(specs) == 3 * (1 + 2 * 6 + 4), len(specs)
    for name, by_layer in specs.items():
        assert sorted(by_layer) == BAND, name
    for seed in E.SEEDS:
        pool = set(E.pool_positions(seed))
        for domain in E.EXP1_DOMAINS:
            sim = specs[f"similarity[{domain}]@s{seed}"][12]
            assert abs(W.ess(sim[sim > 0]) - 200) < 1, name
            assert set(sim.nonzero().flatten().tolist()) == pool
            assert int((specs[f"top_n[{domain}]@s{seed}"][12] > 0).sum()) == 100
            assert int((specs[f"random_n[{domain}]@s{seed}"][12] > 0).sum()) == 100
            rw = specs[f"random_weights[{domain}]@s{seed}"][12]
            assert abs(W.ess(rw[rw > 0]) - 200) < 1
            assert not torch.equal(rw, sim)
            inside = set(specs[f"oracle500_in_pool[{domain}]@s{seed}"][12].nonzero().flatten().tolist())
            outside = set(specs[f"oracle500_disjoint[{domain}]@s{seed}"][12].nonzero().flatten().tolist())
            assert len(inside) == len(outside) == 500 and not (inside & outside)
            assert inside <= pool and not (outside & pool), "the disjoint oracle must not touch the pool"
            assert inside | outside == set(E.universe_slice(domain))
        # Weights must differ across layers, because each layer has its own scores.
        assert not torch.equal(specs[f"top_n[wiki]@s{seed}"][12], specs[f"top_n[wiki]@s{seed}"][18])
    return f"{len(specs)} lens specs; disjoint oracles avoid the pool; ESS and n as asked"


# ---------------------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------------------

@check
def fitting_many_matches_direct_averaging():
    windows = small_windows("wiki", 12)
    layers = [12, 18]
    torch.manual_seed(0)
    weights = {l: torch.rand(12).double() for l in layers}
    specs = {"weighted": weights, "uniform": {l: torch.ones(12).double() for l in layers},
             "subset": {l: torch.tensor([1.0] * 3 + [0.0] * 9).double() for l in layers}}
    lenses = fit_many(windows, specs, layers, batch_size=5)
    worst = 0.0
    for name, by_layer in specs.items():
        for l in layers:
            direct = JLens.from_windows(({l: load_jacobians(w, [l])[l]} for w in windows), weights=by_layer[l].float())
            worst = max(worst, float((lenses[name].jacobians[l] - direct.jacobians[l]).norm() / direct.jacobians[l].norm()))
    assert worst < 1e-4, worst
    stack = load_stack(windows, 12)
    assert stack.shape == (12, 896 * 896)
    return f"worst relative difference {worst:.1e}"


@check
def fitting_cached_jacobians_are_correct():
    """A cached file must equal a fresh computation."""
    window = small_windows("stories", 1)[0]
    device = next(Context.model.parameters()).device
    fresh = window_jacobians(Context.model, window.ids_tensor(device), [12])[12]
    cached = load_jacobians(window, [12])[12].float()
    err = float((fresh - cached).norm() / fresh.norm())
    assert err < 2e-3, err
    return f"relative difference {err:.1e} (the cache stores float16)"


# ---------------------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------------------

@check
def metric_jacobian_error_on_known_cases():
    window = small_windows("wiki", 1)
    J = load_jacobians(window[0], [12])[12].float()
    same = jacobian_error(JLens({12: J}), window, [12])[12][0]
    doubled = jacobian_error(JLens({12: 2 * J}), window, [12])[12][0]
    assert same < 1e-6 and abs(doubled - 1) < 1e-5
    return f"self error {same:.1e}, doubled lens {doubled:.4f}"


@check
def metric_readout_agreement_on_known_cases():
    model = Context.model
    windows = small_windows("wiki", 2)
    acts = window_activations(model, windows, [12])
    identity = JLens.identity([12], 896)
    lens = fit_many(small_windows("wiki", 8), {"a": {12: torch.ones(8)}}, [12])["a"]
    same = readout_agreement(model, lens, lens, acts)[12]
    other = readout_agreement(model, identity, lens, acts)[12]
    assert min(same["overlap"]) == 1.0 and max(same["sym_kl"]) < 1e-4
    assert max(other["overlap"]) < 1.0 and min(other["sym_kl"]) > 0
    return f"self: overlap 1, KL ~0; logit vs J-lens: overlap {other['overlap'][0]:.2f}, KL {other['sym_kl'][0]:.1f}"


@check
def metric_edit_prediction_is_exact_for_exact_jacobian():
    """With a position's own exact Jacobian, the part of the change that flips with the edit's
    sign must match the prediction. A single-sign error also contains curvature, which grows
    with edit size (scratch/edit_linearity.py measures this across sizes)."""
    model = Context.model
    window = small_windows("wiki", 1)[0]
    device = next(model.parameters()).device
    layer = 12
    per_position = window_jacobians(model, window.ids_tensor(device), [layer], per_position=True)[layer]
    positions = edit_positions(window, 5)
    exact = per_position[:, positions, :].permute(1, 0, 2)  # [n, d, d]
    norm = typical_norms(window_activations(model, [window], [layer]))[layer]

    def antisymmetric_error(fraction):
        records = edit_trials(model, JLens({layer: exact[0]}), window, layer, {fraction: fraction * norm},
                              positions=positions, directions=exact)
        pairs = {}
        for r in records:
            pairs.setdefault(r["position"], {})[r["sign"]] = r
        errors = [abs((p[1.0]["actual"] - p[-1.0]["actual"]) / 2 - p[1.0]["predicted"]) / abs(p[1.0]["predicted"])
                  for p in pairs.values()]
        return max(errors), min(r["vocab_correlation"] for r in records)

    small_error, small_correlation = antisymmetric_error(0.01)
    large_error, _ = antisymmetric_error(0.5)
    assert small_error < 0.005, small_error
    assert small_correlation > 0.99, small_correlation
    assert large_error > small_error, "error should grow with edit size"
    return f"sign-flipping part matches the prediction within {small_error:.4f} at 1% of norm, {large_error:.2f} at 50%; correlation {small_correlation:.4f}"


@check
def metric_edit_batching_matches_a_single_full_pass():
    """The batched edit expands copies at the edited layer. Compare with a plain, unbatched run."""
    model = Context.model
    window = small_windows("wiki", 1)[0]
    device = next(model.parameters()).device
    layer, size = 12, 2.0
    lens = JLens.identity([12], 896)
    records = edit_trials(model, lens, window, layer, {0.05: size})
    r = records[0]
    assert r["sign"] == 1.0
    ids = window.ids_tensor(device)
    W_c = model.lm_head.weight[r["token"]].float()
    v = W_c / W_c.norm()  # the identity lens's direction

    def add(module, inputs, output):
        h = (output if torch.is_tensor(output) else output[0]).clone()
        h[0, r["position"]] += size * v
        return h if torch.is_tensor(output) else (h, *output[1:])

    with torch.no_grad(), record_residuals(model, [23]) as clean:
        model.model(input_ids=ids)
    base = clean[23].float()
    handle = model.model.layers[layer].register_forward_hook(add)
    try:
        with torch.no_grad(), record_residuals(model, [23]) as edited:
            model.model(input_ids=ids)
    finally:
        handle.remove()
    P = valid_positions(128).to(device)
    P = P[P >= r["position"]]
    actual = float(((edited[23].float() - base)[0, P] @ W_c).sum())
    gap = abs(actual - r["actual"]) / abs(actual)
    assert gap < 1e-3, (actual, r["actual"])
    return f"batched and single-pass changes agree within {gap:.1e}"


@check
def metric_edit_positions_are_fixed_and_valid():
    window = small_windows("wiki", 1)[0]
    a, b = edit_positions(window), edit_positions(window)
    valid = set(valid_positions(128).tolist())
    assert a == b and len(set(a)) == 10 and set(a) <= valid
    model = Context.model
    lens = JLens.identity([12], 896)
    records = edit_trials(model, lens, window, 12, {0.05: 1.0, 0.1: 2.0})
    assert len(records) == 10 * 2 * 2
    sizes = {(r["position"], r["fraction"], r["sign"]): r for r in records}
    r_plus, r_minus = sizes[(a[0], 0.1, 1.0)], sizes[(a[0], 0.1, -1.0)]
    assert abs(r_plus["predicted"] + r_minus["predicted"]) < 1e-3 and r_plus["token"] == r_minus["token"]
    return "10 valid positions per window, same for every lens; +/- predictions are mirror images"


@check
def pass_ranks_and_auc_on_known_cases():
    model, tok = Context.model, Context.tok
    prompt = "The capital of France is"
    ids = tok(prompt, return_tensors="pt").input_ids.to(next(model.parameters()).device)
    logits = model(input_ids=ids).logits[0, -1]
    top = tok.decode(int(logits.argmax())).strip()
    far = tok.decode(int(logits.argmin())).strip()
    if not word_tokens(tok, far):
        far = "photosynthesis"
    items = [{"prompt": prompt, "intermediate": top, "control": far}]
    prepared = prepare_pass(model, tok, items, layers=[23])
    lens = JLens.identity([23], 896)
    ranks = pass_ranks(model, prepared, lens, layers=[23])
    assert ranks[0]["target"][23] == 0, ranks
    summary = pass_summary(ranks, layers=[23])
    assert summary["target"]["pass@1"] == 1.0 and summary["control"]["pass@1"] == 0.0
    assert abs(auc_of_rank(0) - 1) < 1e-9 and auc_of_rank(10**6) == 0.0 and 0 < auc_of_rank(30) < 1
    assert auc_of_rank(3) > auc_of_rank(30) > auc_of_rank(300)
    return "the identity lens at the final layer ranks the model's own top token first; AUC is 1 at rank 0"


@check
def swap_alpha_zero_does_nothing_and_alpha_one_swaps_coordinates():
    model, tok = Context.model, Context.tok
    lens = fit_many(small_windows("wiki", 8), {"a": {l: torch.ones(8) for l in BAND}}, BAND)["a"]
    prompt = "Fact: The capital of France is"
    source, target = tok(" France")["input_ids"][0], tok(" Japan")["input_ids"][0]
    ids = tok(prompt, return_tensors="pt").input_ids.to(next(model.parameters()).device)
    plain = model(input_ids=ids).logits[0, -1].float()
    still = swapped_logits(model, tok, prompt, lens, source, target, 0.0)
    gap = float((plain - still).abs().max())
    assert gap < 1e-3, gap

    # After the edit, a layer's coordinates must equal the clean ones, swapped.
    layer = BAND[-1]
    clean, _ = residuals(model, ids, BAND)
    V = lens_vectors(model, lens, layer, [source, target])
    pinv = torch.linalg.pinv(V)  # the slow general form, as a check on the closed form in the code
    wanted = (clean[layer].float() @ pinv.T).flip(-1)
    seen = {}

    def spy(module, inputs, output):
        seen["h"] = output if torch.is_tensor(output) else output[0]

    # Registered before the edit hooks would see the unedited value, so it must be added after.
    handles = []
    original = model.model.layers[layer].register_forward_hook

    def register_with_spy(hook, *args, **kwargs):
        handle = original(hook, *args, **kwargs)
        handles.append(original(spy))
        return handle

    model.model.layers[layer].register_forward_hook = register_with_spy
    try:
        swapped = swapped_logits(model, tok, prompt, lens, source, target, 1.0)
    finally:
        model.model.layers[layer].register_forward_hook = original
        for handle in handles:
            handle.remove()
    got = seen["h"].float() @ pinv.T
    err = float((got - wanted).abs().max() / wanted.abs().max())
    assert err < 1e-3, err
    assert float((swapped - plain).abs().max()) > 0.1
    return f"alpha=0 changes logits by {gap:.1e}; alpha=1 coordinates match the swapped clean ones within {err:.1e}"


@check
def features_are_means_over_valid_positions():
    model = Context.model
    windows = small_windows("wiki", 3)
    ids = window_ids(windows)
    device = next(model.parameters()).device
    features = mean_residuals(model, ids, [12], batch_size=2)[12]
    acts, _ = residuals(model, ids.to(device), [12])
    expected = acts[12][:, valid_positions(128).to(device)].float().mean(1).cpu()
    err = float((features - expected).norm() / expected.norm())
    assert err < 1e-5, err
    return f"batched features match a direct computation within {err:.1e}"


# ---------------------------------------------------------------------------------------
# End to end, on validation prompts with small lenses
# ---------------------------------------------------------------------------------------

@check
def end_to_end_on_validation():
    """Every metric on every validation set, with small lenses. Prints a table for reading."""
    model, tok = Context.model, Context.tok
    lines = []
    for domain in DOMAINS:
        windows = small_windows(domain, 30)
        lens = fit_many(windows, {"a": {l: torch.ones(30) for l in BAND}}, BAND)["a"]
        half = fit_many(windows[:15], {"a": {l: torch.ones(15) for l in BAND}}, BAND)["a"]
        logit = JLens.identity(BAND, 896)
        for one in (lens, half, logit):
            E.on_gpu(one)
        evaluator = E.Evaluator(model, tok, domain, "val")
        scores = {name: evaluator.score(one, half) for name, one in (("30 windows", lens), ("logit", logit))}
        for name, s in scores.items():
            json.dumps(s)  # must be serializable
            summary = pass_summary(s["pass"]["ranks"])
            edits = [r for by_w in s["edits"][12] for r in by_w if r["fraction"] == 0.1]
            median = sorted(r["relative_error"] for r in edits)[len(edits) // 2]
            lines.append(f"    {domain:8s}{name:12s} jac {sum(s['jacobian_error'][12]) / 25:.3f}"
                         f" | pass@10 {summary['target']['pass@10']:.2f}/{summary['control']['pass@10']:.2f}"
                         f" | pass@50 {summary['target']['pass@50']:.2f}/{summary['control']['pass@50']:.2f}"
                         f" | auc {summary['target']['auc']:.2f}"
                         f" | swap {sum(r['pass'] for r in s['swap']['1.0']) / len(s['swap']['1.0']):.2f}"
                         f" | overlap {sum(s['readout'][12]['overlap']) / 25:.2f}"
                         f" | edit err L12 {median:.2f} | {s['seconds_to_score']:.0f}s")
        # The J-lens should fit Jacobians far better than the logit lens does.
        assert (sum(scores["30 windows"]["jacobian_error"][12]) < sum(scores["logit"]["jacobian_error"][12])), domain
    print("\n".join(lines))
    return "all metrics ran on all domains; the J-lens has lower Jacobian error than the logit lens"


@check
def dry_run_of_all_experiments(tmp=E.REPO / "results" / "dry_run"):
    """The real runner and analysis on 30 fit windows per domain, with results kept separately."""
    from analysis import summarize

    # Delete results/dry_run for a fresh start; finished lenses are otherwise skipped, as in a real run.
    saved = {k: getattr(E, k) for k in ("FIT_SIZE", "POOL_PER_CORPUS", "GRID", "SEEDS", "RAW", "SETTINGS_PATH",
                                        "SAMPLE_COUNTS", "EXP3_BUDGETS", "experiment3_universe")}
    E.FIT_SIZE, E.POOL_PER_CORPUS, E.GRID, E.SEEDS = 30, 10, [5, 10], [0, 1]
    E.SAMPLE_COUNTS, E.EXP3_BUDGETS = [5, 10], [5, 10]
    E.RAW, E.SETTINGS_PATH = tmp / "raw", tmp / "settings.json"
    tmp.mkdir(parents=True, exist_ok=True)
    # Experiment 3's selected windows have no Jacobians yet, so stand-ins are the fit windows.
    E.experiment3_universe = lambda: (E.domain_universe(), {d: list(E.universe_slice(d))[:10] for d in E.EXP3_DOMAINS})
    try:
        model, tok = Context.model, Context.tok
        E.select_settings(model)
        E.run_experiment(model, tok, 1, "val")
        E.run_experiment(model, tok, 2, "val")
        E.run_experiment3(model, tok, "val")
        E.run_sensitivity(model, tok, "val")
        tables = summarize.build(tmp / "raw", "val", tmp / "tables", E.SEEDS, bootstrap=100)
    finally:
        for key, value in saved.items():
            setattr(E, key, value)
    assert all(path.exists() for path in tables)
    print(tables[0].read_text()[:3000])
    return f"wrote {[p.name for p in tables]}"


def main():
    pattern = sys.argv[1] if len(sys.argv) > 1 else ""
    Context.model, Context.tok = load_model()
    failed = 0
    for fn in CHECKS:
        if pattern and pattern not in fn.__name__:
            continue
        try:
            print(f"PASS {fn.__name__}: {fn()}", flush=True)
        except Exception:
            failed += 1
            print(f"FAIL {fn.__name__}", flush=True)
            traceback.print_exc()
    print("all passed" if not failed else f"{failed} FAILED", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
