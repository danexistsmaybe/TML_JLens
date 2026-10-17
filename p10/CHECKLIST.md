# P10 first-day checklist (steps 1–5) and proposal-talk material

Everything here was produced overnight 2026-10-16. Numbers come from `results/`, and figures are in `figures/`. Steps 6 (division of work) and 7 (preparing the talk) were left to you as requested, but the "Talk material" section at the end gives you the raw ingredients.

---

## Step 1: Project and original component
- **Project:** P10, *Does the J-lens fitting corpus change the interpretation?* (D3 / C3).
- **Research question:** which Jacobian-lens readouts transfer between fitting corpora, and can corpus weighting improve interpretation on a new prompt domain?
- **Proposed original component: a domain-similarity-weighted J-lens estimator.** J_ℓ = Σ_i w_i J_ℓ^(i) over stored per-window Jacobians. Each weight is w_i ∝ exp(cos(φ_i, μ_deploy)/τ). φ_i is a window's mean frozen-model residual at a mid layer, μ_deploy is the centroid of unlabeled deployment-domain dev windows, and τ is chosen on dev data.
  - **Design point worth stating in the talk:** our estimator stores *per-window* Jacobians. That makes every lens (domain-specific, uniform-pooled, weighted, random-weighted, any sample count, any seed) an exact weighted average of the *same* matrices. Compute is matched across methods by construction, and the ablations cost almost nothing.

## Step 2: Readings and core data/model ✅
- Read: the J-lens / global-workspace paper (methods, layer bands, all methodological appendices, limitations), reference repo `fitting.py` / `lens.py` (read only), and the Tuned Lens paper. Synthesis in `SUMMARY.md`.
- Downloaded (details in `ENV.md`):

| Resource | Revision |
|---|---|
| Qwen/Qwen2.5-0.5B | `060db64` |
| WikiText-2 raw | `b08601e` |
| TinyStories (train shard 0 + validation) | `f54c09f` |

- Environment: uv venv, Python 3.11, torch 2.6.0+cu124, transformers 5.20.0, TITAN RTX 24 GB.

## Step 3: Split manifest, created before any fitting ✅
`manifests/splits.json` (seed 0) has, per corpus, **200 fit / 50 val / 50 test windows of exactly 128 tokens**:
- **WikiText:** fit ← official train, val ← validation, test ← test.
- **TinyStories:** fit ← train shard 0 (5,000 seeded stories); val/test ← the official validation split, **partitioned by story**.
- A hash check confirms no window appears in two roles. Decoded previews are in `manifests/preview.md`.

Still to do in Week 2: the controlled entity/property prompt sets, split by template/entity.

## Step 4: First-start experiment and pilot ✅

### 4a. Correctness check (`results/correctness.json`): **PASS**
- Our hooks equal HF `hidden_states[l+1]` exactly. `decode(h_final)` reproduces the model logits exactly.
- **Jacobian vs float64 central finite differences: relative error ~2×10⁻⁸** (3 random directions, layer 8, position 25).
- Our averaged estimator equals the mean of per-position Jacobians (max error 2×10⁻⁷).
- *Debugging story worth one sentence in the talk:* the first finite-difference check failed (relative error ~0.5). HF's Qwen2 RMSNorm silently upcasts to fp32, so the "fp64" check was really fp32 and finite differences were dominated by rounding. Computing the norm in native precision fixed it.

### 4b. Brief's first start: logit lens on 10 matched windows per corpus (`figures/fig1_logit_lens_by_corpus.png`)
- Same residual convention for every lens (block outputs, decode = `lm_head(norm(·))`).
- Logit-lens readouts are junk through ~layer 16. The same few tokens recur on every input ("уницип", " mắn", "UCCESS"). Agreement with the model's top-1 prediction:

| Layer | WikiText | TinyStories |
|---|---|---|
| 12 | 0.8% | 0.5% |
| 16 | 5% | 6% |
| 20 | 29% | 35% |

- The curves for the two corpora are nearly identical. The logit lens has no fitting corpus, so it is a corpus-independent baseline.

### 4c. Pilot J-lenses (`scripts/04`, `05`)
**Setup:** for each corpus, two disjoint halves of 40 fit windows (A, B), all 23 source layers, target = block 23. Evaluated on held-out val windows.

**Finding 1: the fitting corpus changes J-lens readouts far beyond sampling noise.**
Top-1 readout agreement on held-out windows, averaged over both evaluation domains (`figures/fig4_headline.png`):

| Layer | Within-corpus (A vs B) | Cross-corpus (WikiText fit vs TinyStories fit) | Logit vs J-lens |
|---|---|---|---|
| 6 | 0.77 | 0.22 | 0.00 |
| 12 | 0.80 | 0.21 | 0.00 |
| 18 | 0.88 | 0.40 | 0.12 |
| 21 | 0.96 | 0.74 | 0.51 |
| 22 | 0.97 | 0.80 | 0.60 |

The lens matrices tell the same story:

| Layer | Cross-corpus relative distance | Within-corpus relative distance | Ratio |
|---|---|---|---|
| 12 | 0.77 | 0.12–0.16 | ~5× |
| 21 | 0.22 | 0.03–0.05 | ~5× |

Disagreement is largest in early and middle layers and shrinks toward the output. This is the same depth profile where the paper says the J-lens adds the most over the logit lens.

**Finding 2: an in-domain lens predicts the true local Jacobian much better, and pooling sits in between.** Relative error of the lens against held-out per-window Jacobians at layer 12 (`figures/fig3_derivative_prediction_error.png`):

| Lens fit on | WikiText prompts | TinyStories prompts |
|---|---|---|
| WikiText | **0.39** | 0.58 |
| TinyStories | 0.82 | **0.21** |
| Uniform pool (matched 40 windows) | 0.49 | 0.30 |
| Logit lens (J = I) | 1.17 | 1.11 |

This is the gap a deployment-weighted pool should close, and it is the direct motivation for the original component.

**Finding 3, qualitative (viewer prototype, `results/pilot/readout_examples.md`).** Same TinyStories activation ("...John's mom told him to" → " trust"):

| Layer | Logit lens | WikiText-fit J-lens | TinyStories-fit J-lens |
|---|---|---|---|
| 12 | `./../../`, `rigesimal` (noise) | newline/quote tokens | **practice, practiced, obedient, disobed** |
| 15 | | | **push, caution, climbing** |
| 18 | emoji | | wait, stop, climb, pause / climb, stop, wait, take, push |

The in-domain lens surfaces meaningful intermediate concepts ("obedient", "caution") several layers earlier. **Caveat:** the WikiText lens is dominated by formatting tokens (newlines, quotes). This may be a token-frequency/formatting confound, which is exactly the brief's "control for token frequency and length" ablation.

**Finding 4 (development only): controlled-token retrieval on the repo's 84 multihop prompts with a single-token intermediate.** pass@10 (the intermediate, such as "Brazil", is in the lens top-10 at some layer):

| Lens | pass@10 |
|---|---|
| Logit lens | 0.27 |
| J-lens, WikiText fit | 0.38–0.39 |
| J-lens, TinyStories fit | 0.42–0.45 |
| J-lens, pool | 0.42 |

So the J-lens beats the logit lens at this small scale too. Corpus differences here are within noise, since the prompts belong to neither domain. Qwen-0.5B puts the final answer in its top-10 on only 27/84 of these prompts, so we need **easier controlled prompts** designed per domain.

## Step 5: Measurability and cost ✅
- **Measurability: yes.** The cross-corpus effect is 3–5× the within-corpus sampling noise on every metric, with only 40 windows per lens.
- **Measured cost:** 9.0–9.4 s per 128-token window for all 23 source layers; peak memory 6.2 GB at dim_batch 16 (18.9 GB at 64, same speed). Hardware: TITAN RTX, fp32. The entire overnight pilot used about **35 GPU-minutes**.
- **Estimate for the minimum scope** (total ≈ measured time per run × planned runs + overhead):

| Item | Windows | GPU-h |
|---|---|---|
| Per-window Jacobians, full fit pools (2 × 200) | 400 | 1.0 |
| Deployment Jacobians, val + test (2 × 100) | 200 | 0.5 |
| Readouts, controlled prompts, ± interventions (forward passes) | n/a | ≈0.5 |
| Weighting, seeds, sample-count sweep, ablations (reweighting stored matrices) | n/a | <0.2 |
| Debugging/rerun margin (×2–3) | n/a | 2–4 |
| **Total** | | **≈ 4–6 GPU-hours** |

  This is well under the brief's 20–50. **Headroom for extra credit:**
  - a third, synthetic arithmetic or code corpus (+~1 h)
  - 1000-window lenses as in the paper (+~4 h)
  - Qwen2.5-1.5B as a scale check (roughly 4–6× per window)
- **Disk:** per-window Jacobians are 1.6 MB per layer per window (fp16). Keeping all 23 layers for 600 windows is about 22 GB, which is fine here. Storing only the two chosen layers is about 2 GB.

---

## Proposed plan for the remaining weeks (follows the brief's milestones)
- **Layers:** 12 (early/mid workspace, where the corpus effect is largest) and 18 (late workspace, partial convergence). Pilot curves are available for all layers if you'd rather choose others.
- **Lenses:**
  - Baselines: logit, WikiText-only, TinyStories-only, uniform pool.
  - Proposed: similarity-weighted pool.
  - Ablations: random weights; frequency/length-controlled weighting; no weighting.
  - All at matched window budgets, with 3+ resampling seeds from the stored pools.
- **Metrics:**
  - controlled-token retrieval on new per-domain controlled prompts (templates such as "Lily has a red ball… she gave the ___", plus entity facts), split by template/entity
  - derivative-prediction error (cross-domain matrix)
  - readout agreement
  - ± activation-edit intervention check (predicted vs actual logit change along J-lens directions)
  - runtime
  - sample-count sensitivity (10 → 200 windows)
- **Week 2:** controlled prompts, full fit pools, intervention check code. **Week 3:** weighting rule on dev. **Week 4:** evaluation matrix on test. **Week 5:** frequency controls, seeds, failure analysis. **Week 6:** viewer and report.

## Unresolved assumptions and risks (good for the talk's last slide)
1. **Formatting confound.** Part of the cross-corpus gap may come from formatting and token frequency, such as WikiText's newline-heavy layout, rather than semantics. Plan: position masking (e.g. drop positions whose next token is whitespace or punctuation; the paper tried this for its own corpus) and frequency-matched weighting.
2. **Sample size.** Within-corpus agreement is only about 0.78 in middle layers at 40 windows, so the lenses themselves are still noisy. The sample-count sweep will show whether the cross-corpus gap persists at 200 windows.
3. **Model competence.** A 0.5B model is weak on multihop prompts, so controlled prompts must be easy enough that the model computes the intermediate.
4. **Different questions.** Derivative-prediction error and controlled-token retrieval measure different things. A lens can win on one and lose on the other.
5. **Convention choices.** Target = final block (not penultimate as in the paper's Claude runs), skip-first-16, sum-then-mean. These are documented, and one variant could be checked.

**Expected outcome:** weighting closes most of the in-domain vs out-of-domain gap on derivative error and improves controlled-token retrieval on the new domain. **Plausible failure:** the gap is mostly a formatting/frequency effect that frequency controls remove, giving the negative result "J-lens readouts are corpus-robust once format is controlled."

---

## Talk material (5 minutes)
- **Slide 1, question:** the J-lens claims to show "what the model is disposed to say," but it is an average over a fitting corpus. Does the analyst's corpus choice change the answer?
- **Slide 2, method:** J_ℓ = E[∂h_final/∂h_ℓ], one-hot cotangent trick, logit lens as J = I. The original component is a similarity-weighted pooled estimator over stored per-window Jacobians.
- **Slide 3, pilot (key figure):** `figures/fig4_headline.png`. Cross-corpus readout agreement is ~0.2 vs ~0.8 within-corpus in middle layers; the in-domain lens has 2–4× lower Jacobian error; the pool sits in between.
- **Slide 4, qualitative:** the TinyStories example table above ("obedient", "caution" vs noise).
- **Slide 5, plan:** baselines, ablations, compute (35 GPU-min pilot, ~5 GPU-h projected), risks, expected and failure outcomes.
- Backup: `fig1` (logit lens is corpus-independent junk before layer 16), `fig2` (all panels), `fig3`, and the correctness check.
