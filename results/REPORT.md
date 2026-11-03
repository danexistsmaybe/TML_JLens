# Preliminary results (test, 2026-11-02)

Prompts are unreviewed. Figures are in `figures/`; paired differences and per-layer numbers are in `tables/`.

## Main findings

1. **Known domains: the methods recover the domain's lens.** Jacobian error reaches the oracle (wiki 0.499 vs 0.498, stories 0.349 vs 0.347, from 0.560 and 0.420 for the uniform pool). Top-10 overlap rises from about 0.65 to about 0.94, near the 0.95–0.97 that a separate 500-window oracle reaches.
2. **Pass@k doesn't respond to the corpus.** The methods leave the AUC gap unchanged in every domain. Even the two oracles mostly can't be told apart, except stories at pass@50. So this metric can't detect a gain here, and we should report it as uninformative about corpus choice.
3. **Swaps go both ways.** Similarity against the uniform pool: wiki +0.04, python +0.09, stories −0.15. On stories, the wiki oracle beats the stories oracle (0.74 vs 0.58), so a matching corpus can hurt. The loss is spread across most templates and mostly fades at α = 2. We have no explanation yet.
4. **New domain: a real gain, far from the ceiling.** On python, Jacobian error falls from 0.762 to 0.701 against an oracle at 0.474, and nearly all the weight lands on wiki windows.
5. **Within a domain: selection doesn't beat random.** At ESS or n of 100, selected windows have higher Jacobian error than random ones in all three domains (for example, wiki 0.524 vs 0.501). Wider selection is worse than the uniform fit on Jacobian error (wiki 0.542 vs 0.498) and shows no swap gain. Python's AUC gap rises slightly with selection (+0.014 to +0.017), which is the one small positive, and it may come from the digit controls.
6. **Very few windows are enough.** An oracle fit on 10 windows has nearly the same AUC gap as one fit on 1000 (wiki 0.656 vs 0.652), and Jacobian error levels off by about 50 windows. That may explain why selection has so little room to help.
7. **Every J-lens beats the logit lens on the window metrics and on swaps.** Those are Jacobian error, overlap and edit error. The AUC gap also favors the J-lens in all three domains, though on python the margin is narrower and the pass@10 gap doesn't separate them.

## Runtime

About 9.4 s per window for Jacobians on one GPU; once those are cached, fitting 53 lenses at 14 layers took 52 s.

## Before relying on this

- **Prompts are unreviewed.** The pass@k and swap columns could change after review. Judge prompts on their text, without per-prompt scores, and later changes get logged and rerun.
- **Stories read positions depart from the paper.** Each story has one added token: a pronoun where “ The” would ask the model to name the described thing, otherwise “ The”. Those are judgment calls for you to check. Python implied prompts read at the end of the code, which is closer to the paper.
- **Python evaluate prompts saturate pass@10.** Their digit controls are as plausible as the targets, so control rates are high there. The CSV has per-type rows: on test, implied targets reach 37–41% at pass@50 under J-lenses with 0% control, against 13% for the logit lens.
- **Edit error.** The main size moved from 10% to 5% of activation norm before test, because an exact Jacobian already shows 24–33% error at 10%. With real lenses the two sizes give nearly the same error, so this changed little.
- **Intervals.** Paired differences aren't corrected for multiple comparisons. The three seeds' pools overlap, so seed spread understates true variability. Swap intervals rest on 9 templates and are wide, especially on wiki, where templates mostly score all or nothing.
- **Wider selection is topical.** Wiki picks favor road and river articles, and 420 of 1000 sit next to another pick in the source. That may be why it does poorly.
- **Scope.** One small model, Qwen2.5-0.5B. Jacobians were computed under NVIDIA driver 595, and scoring ran under 580. Correctness checks passed, and a cached Jacobian matched a fresh one to within 2e-4.

## Experiment 1: known domains

The pool holds 500 wiki and 500 stories windows. The lens is deployed on one of them, and an unlabeled validation sample picks the windows. Settings: ESS 400 and n = 400 for both.

### Deployed on wiki

| Lens | Jacobian error | Pass@10 | Control | Pass@50 gap | AUC gap | Swap α=1 | Swap α=2 | Top-10 overlap | Sym. KL | Edit error |
|---|---|---|---|---|---|---|---|---|---|---|
| Logit lens | 0.941 ± 0.003 | 0.16 ± 0.09 | 0.00 ± 0.00 | 0.62 ± 0.14 | 0.490 ± 0.063 | 0.12 ± 0.14 | 0.20 ± 0.19 | 0.057 ± 0.004 | 7.698 ± 0.128 | 1.51 ± 0.13 |
| Uniform pool | 0.560 ± 0.017 | 0.62 ± 0.14 | 0.02 ± 0.03 | 0.94 ± 0.07 | 0.650 ± 0.061 | 0.67 ± 0.28 | 0.40 ± 0.17 | 0.631 ± 0.008 | 0.625 ± 0.026 | 0.43 ± 0.06 |
| Random weights | 0.560 ± 0.017 | 0.62 ± 0.13 | 0.02 ± 0.03 | 0.94 ± 0.07 | 0.650 ± 0.061 | 0.67 ± 0.28 | 0.40 ± 0.17 | 0.631 ± 0.008 | 0.623 ± 0.026 | 0.43 ± 0.06 |
| Random n | 0.559 ± 0.017 | 0.62 ± 0.14 | 0.02 ± 0.03 | 0.94 ± 0.07 | 0.650 ± 0.061 | 0.67 ± 0.28 | 0.40 ± 0.17 | 0.632 ± 0.008 | 0.620 ± 0.026 | 0.42 ± 0.05 |
| **Similarity weighting** | 0.499 ± 0.016 | 0.57 ± 0.13 | 0.02 ± 0.03 | 0.90 ± 0.09 | 0.648 ± 0.061 | 0.71 ± 0.28 | 0.43 ± 0.17 | 0.933 ± 0.001 | 0.011 ± 0.000 | 0.46 ± 0.06 |
| **Top-n selection** | 0.499 ± 0.016 | 0.55 ± 0.13 | 0.00 ± 0.00 | 0.91 ± 0.08 | 0.647 ± 0.058 | 0.70 ± 0.27 | 0.43 ± 0.17 | 0.940 ± 0.001 | 0.009 ± 0.000 | 0.46 ± 0.06 |
| Wiki oracle (1000) | 0.498 ± 0.017 | 0.60 ± 0.13 | 0.00 ± 0.00 | 0.92 ± 0.07 | 0.652 ± 0.057 | 0.70 ± 0.26 | 0.43 ± 0.17 | 0.977 ± 0.001 | 0.001 ± 0.000 | 0.45 ± 0.06 |
| Stories oracle (1000) | 0.711 ± 0.017 | 0.60 ± 0.13 | 0.02 ± 0.03 | 0.92 ± 0.09 | 0.634 ± 0.066 | 0.63 ± 0.27 | 0.31 ± 0.18 | 0.374 ± 0.008 | 1.977 ± 0.070 | 0.69 ± 0.08 |
| Domain oracle (500) | 0.498 ± 0.017 | 0.61 ± 0.13 | 0.00 ± 0.00 | 0.92 ± 0.07 | 0.656 ± 0.057 | 0.70 ± 0.26 | 0.43 ± 0.17 | 0.953 ± 0.001 | 0.005 ± 0.000 | 0.46 ± 0.06 |

### Deployed on stories

| Lens | Jacobian error | Pass@10 | Control | Pass@50 gap | AUC gap | Swap α=1 | Swap α=2 | Top-10 overlap | Sym. KL | Edit error |
|---|---|---|---|---|---|---|---|---|---|---|
| Logit lens | 0.924 ± 0.001 | 0.16 ± 0.10 | 0.04 ± 0.05 | 0.28 ± 0.14 | 0.306 ± 0.097 | 0.07 ± 0.07 | 0.12 ± 0.06 | 0.056 ± 0.003 | 7.205 ± 0.087 | 1.54 ± 0.11 |
| Uniform pool | 0.420 ± 0.008 | 0.28 ± 0.12 | 0.02 ± 0.03 | 0.40 ± 0.14 | 0.361 ± 0.093 | 0.69 ± 0.13 | 0.56 ± 0.14 | 0.663 ± 0.008 | 0.469 ± 0.028 | 0.29 ± 0.03 |
| Random weights | 0.417 ± 0.008 | 0.27 ± 0.12 | 0.02 ± 0.03 | 0.40 ± 0.14 | 0.360 ± 0.092 | 0.69 ± 0.12 | 0.57 ± 0.15 | 0.669 ± 0.008 | 0.452 ± 0.027 | 0.29 ± 0.03 |
| Random n | 0.421 ± 0.008 | 0.27 ± 0.12 | 0.02 ± 0.03 | 0.40 ± 0.14 | 0.358 ± 0.092 | 0.67 ± 0.12 | 0.56 ± 0.15 | 0.662 ± 0.008 | 0.475 ± 0.028 | 0.29 ± 0.03 |
| **Similarity weighting** | 0.349 ± 0.013 | 0.30 ± 0.12 | 0.02 ± 0.03 | 0.46 ± 0.14 | 0.369 ± 0.094 | 0.54 ± 0.10 | 0.56 ± 0.15 | 0.937 ± 0.002 | 0.011 ± 0.000 | 0.32 ± 0.03 |
| **Top-n selection** | 0.349 ± 0.013 | 0.29 ± 0.12 | 0.02 ± 0.03 | 0.46 ± 0.14 | 0.368 ± 0.094 | 0.55 ± 0.11 | 0.56 ± 0.15 | 0.944 ± 0.001 | 0.008 ± 0.000 | 0.32 ± 0.03 |
| Wiki oracle (1000) | 0.587 ± 0.008 | 0.22 ± 0.12 | 0.02 ± 0.03 | 0.34 ± 0.15 | 0.343 ± 0.094 | 0.74 ± 0.09 | 0.62 ± 0.19 | 0.416 ± 0.009 | 1.834 ± 0.088 | 0.40 ± 0.06 |
| Stories oracle (1000) | 0.347 ± 0.012 | 0.30 ± 0.12 | 0.02 ± 0.03 | 0.46 ± 0.14 | 0.369 ± 0.095 | 0.58 ± 0.11 | 0.56 ± 0.15 | 0.983 ± 0.001 | 0.001 ± 0.000 | 0.30 ± 0.03 |
| Domain oracle (500) | 0.347 ± 0.012 | 0.28 ± 0.12 | 0.02 ± 0.03 | 0.46 ± 0.14 | 0.367 ± 0.094 | 0.57 ± 0.11 | 0.56 ± 0.15 | 0.966 ± 0.001 | 0.003 ± 0.000 | 0.30 ± 0.03 |

## Experiment 2: new domain

The same pool, deployed on python, which it doesn't contain. Settings: ESS 25, n = 50. The python oracle is out of the method's reach and marks the ceiling.

### Deployed on python

| Lens | Jacobian error | Pass@10 | Control | Pass@50 gap | AUC gap | Swap α=1 | Swap α=2 | Top-10 overlap | Sym. KL | Edit error |
|---|---|---|---|---|---|---|---|---|---|---|
| Logit lens | 0.942 ± 0.003 | 0.18 ± 0.11 | 0.10 ± 0.09 | 0.16 ± 0.10 | 0.218 ± 0.068 | 0.17 ± 0.16 | 0.21 ± 0.18 | 0.057 ± 0.006 | 8.863 ± 0.160 | 1.79 ± 0.18 |
| Uniform pool | 0.762 ± 0.014 | 0.30 ± 0.13 | 0.18 ± 0.11 | 0.32 ± 0.13 | 0.299 ± 0.068 | 0.26 ± 0.17 | 0.27 ± 0.16 | 0.236 ± 0.007 | 3.722 ± 0.087 | 0.99 ± 0.12 |
| Random weights | 0.763 ± 0.015 | 0.31 ± 0.13 | 0.19 ± 0.11 | 0.32 ± 0.13 | 0.295 ± 0.068 | 0.26 ± 0.17 | 0.26 ± 0.15 | 0.239 ± 0.006 | 3.703 ± 0.091 | 1.00 ± 0.12 |
| Random n | 0.762 ± 0.014 | 0.30 ± 0.13 | 0.18 ± 0.10 | 0.32 ± 0.13 | 0.299 ± 0.067 | 0.26 ± 0.17 | 0.27 ± 0.15 | 0.238 ± 0.007 | 3.690 ± 0.086 | 1.00 ± 0.11 |
| **Similarity weighting** | 0.701 ± 0.015 | 0.28 ± 0.13 | 0.22 ± 0.12 | 0.30 ± 0.12 | 0.295 ± 0.060 | 0.35 ± 0.21 | 0.31 ± 0.15 | 0.340 ± 0.008 | 2.781 ± 0.099 | 0.87 ± 0.11 |
| **Top-n selection** | 0.702 ± 0.015 | 0.30 ± 0.13 | 0.22 ± 0.12 | 0.29 ± 0.12 | 0.295 ± 0.061 | 0.36 ± 0.21 | 0.31 ± 0.16 | 0.335 ± 0.008 | 2.879 ± 0.106 | 0.87 ± 0.10 |
| Python oracle (1000) | 0.474 ± 0.026 | 0.34 ± 0.12 | 0.22 ± 0.12 | 0.32 ± 0.12 | 0.300 ± 0.067 | 0.48 ± 0.19 | 0.32 ± 0.14 | 0.979 ± 0.001 | 0.001 ± 0.000 | 0.57 ± 0.06 |
| Python oracle (500) | 0.475 ± 0.026 | 0.32 ± 0.13 | 0.22 ± 0.12 | 0.34 ± 0.13 | 0.303 ± 0.067 | 0.48 ± 0.19 | 0.33 ± 0.14 | 0.959 ± 0.001 | 0.004 ± 0.000 | 0.58 ± 0.06 |

## Experiment 3: within a domain

No domain shift. Windows close to the domain's centroid against random windows from the same domain, at fixed budgets. Wider selection takes the top 1000 of a much larger candidate set. Readout agreement doesn't apply here, because the uniform fit is itself the reference.

### Deployed on wiki

| Lens | Jacobian error | Pass@10 | Control | Pass@50 gap | AUC gap | Swap α=1 | Swap α=2 | Edit error |
|---|---|---|---|---|---|---|---|---|
| Logit lens | 0.941 ± 0.003 | 0.16 ± 0.09 | 0.00 ± 0.00 | 0.62 ± 0.14 | 0.490 ± 0.063 | 0.12 ± 0.14 | 0.20 ± 0.19 | 1.51 ± 0.13 |
| Uniform fit (1000) | 0.498 ± 0.017 | 0.60 ± 0.13 | 0.00 ± 0.00 | 0.92 ± 0.07 | 0.652 ± 0.057 | 0.70 ± 0.26 | 0.43 ± 0.17 | 0.45 ± 0.06 |
| Random weights, ESS 100 | 0.501 ± 0.018 | 0.61 ± 0.13 | 0.01 ± 0.01 | 0.92 ± 0.07 | 0.654 ± 0.058 | 0.70 ± 0.26 | 0.43 ± 0.17 | 0.46 ± 0.06 |
| Random n = 100 | 0.500 ± 0.017 | 0.59 ± 0.13 | 0.01 ± 0.01 | 0.91 ± 0.08 | 0.653 ± 0.056 | 0.70 ± 0.26 | 0.43 ± 0.17 | 0.47 ± 0.06 |
| **Similarity, ESS 100** | 0.524 ± 0.013 | 0.62 ± 0.13 | 0.00 ± 0.00 | 0.90 ± 0.09 | 0.641 ± 0.057 | 0.71 ± 0.29 | 0.43 ± 0.19 | 0.48 ± 0.07 |
| **Top-n, n = 100** | 0.527 ± 0.013 | 0.54 ± 0.13 | 0.00 ± 0.00 | 0.90 ± 0.09 | 0.632 ± 0.058 | 0.71 ± 0.29 | 0.42 ± 0.18 | 0.47 ± 0.07 |
| Random weights, ESS 500 | 0.498 ± 0.017 | 0.61 ± 0.13 | 0.00 ± 0.00 | 0.92 ± 0.07 | 0.654 ± 0.057 | 0.70 ± 0.26 | 0.43 ± 0.17 | 0.46 ± 0.06 |
| Random n = 500 | 0.498 ± 0.017 | 0.61 ± 0.13 | 0.00 ± 0.00 | 0.92 ± 0.07 | 0.653 ± 0.057 | 0.70 ± 0.26 | 0.43 ± 0.17 | 0.45 ± 0.06 |
| **Similarity, ESS 500** | 0.501 ± 0.014 | 0.56 ± 0.14 | 0.00 ± 0.00 | 0.90 ± 0.09 | 0.649 ± 0.060 | 0.70 ± 0.27 | 0.43 ± 0.17 | 0.47 ± 0.06 |
| **Top-n, n = 500** | 0.500 ± 0.014 | 0.56 ± 0.14 | 0.00 ± 0.00 | 0.90 ± 0.09 | 0.647 ± 0.059 | 0.70 ± 0.27 | 0.44 ± 0.17 | 0.47 ± 0.06 |
| **Wider selection (top 1000)** | 0.542 ± 0.013 | 0.64 ± 0.13 | 0.00 ± 0.00 | 0.90 ± 0.09 | 0.641 ± 0.057 | 0.71 ± 0.29 | 0.41 ± 0.17 | 0.51 ± 0.08 |

### Deployed on stories

| Lens | Jacobian error | Pass@10 | Control | Pass@50 gap | AUC gap | Swap α=1 | Swap α=2 | Edit error |
|---|---|---|---|---|---|---|---|---|
| Logit lens | 0.924 ± 0.001 | 0.16 ± 0.10 | 0.04 ± 0.05 | 0.28 ± 0.14 | 0.306 ± 0.097 | 0.07 ± 0.07 | 0.12 ± 0.06 | 1.54 ± 0.11 |
| Uniform fit (1000) | 0.347 ± 0.012 | 0.30 ± 0.12 | 0.02 ± 0.03 | 0.46 ± 0.14 | 0.369 ± 0.095 | 0.58 ± 0.11 | 0.56 ± 0.15 | 0.30 ± 0.03 |
| Random weights, ESS 100 | 0.349 ± 0.012 | 0.29 ± 0.12 | 0.02 ± 0.03 | 0.45 ± 0.14 | 0.366 ± 0.094 | 0.58 ± 0.10 | 0.57 ± 0.15 | 0.30 ± 0.03 |
| Random n = 100 | 0.349 ± 0.012 | 0.29 ± 0.12 | 0.02 ± 0.03 | 0.44 ± 0.14 | 0.365 ± 0.093 | 0.57 ± 0.10 | 0.58 ± 0.15 | 0.30 ± 0.03 |
| **Similarity, ESS 100** | 0.365 ± 0.015 | 0.28 ± 0.12 | 0.02 ± 0.03 | 0.44 ± 0.14 | 0.357 ± 0.094 | 0.54 ± 0.09 | 0.56 ± 0.16 | 0.32 ± 0.04 |
| **Top-n, n = 100** | 0.366 ± 0.015 | 0.28 ± 0.12 | 0.02 ± 0.03 | 0.46 ± 0.14 | 0.368 ± 0.095 | 0.54 ± 0.09 | 0.57 ± 0.16 | 0.31 ± 0.04 |
| Random weights, ESS 500 | 0.347 ± 0.012 | 0.29 ± 0.12 | 0.02 ± 0.03 | 0.46 ± 0.14 | 0.369 ± 0.095 | 0.57 ± 0.11 | 0.56 ± 0.15 | 0.31 ± 0.03 |
| Random n = 500 | 0.348 ± 0.012 | 0.30 ± 0.12 | 0.02 ± 0.03 | 0.46 ± 0.14 | 0.367 ± 0.094 | 0.57 ± 0.10 | 0.56 ± 0.15 | 0.30 ± 0.03 |
| **Similarity, ESS 500** | 0.352 ± 0.014 | 0.30 ± 0.13 | 0.02 ± 0.03 | 0.46 ± 0.14 | 0.369 ± 0.093 | 0.53 ± 0.11 | 0.58 ± 0.16 | 0.31 ± 0.03 |
| **Top-n, n = 500** | 0.352 ± 0.014 | 0.30 ± 0.13 | 0.02 ± 0.03 | 0.46 ± 0.14 | 0.369 ± 0.094 | 0.53 ± 0.11 | 0.58 ± 0.16 | 0.31 ± 0.03 |
| **Wider selection (top 1000)** | 0.365 ± 0.016 | 0.32 ± 0.12 | 0.02 ± 0.03 | 0.44 ± 0.14 | 0.369 ± 0.096 | 0.53 ± 0.08 | 0.57 ± 0.14 | 0.31 ± 0.03 |

### Deployed on python

| Lens | Jacobian error | Pass@10 | Control | Pass@50 gap | AUC gap | Swap α=1 | Swap α=2 | Edit error |
|---|---|---|---|---|---|---|---|---|
| Logit lens | 0.942 ± 0.003 | 0.18 ± 0.11 | 0.10 ± 0.09 | 0.16 ± 0.10 | 0.218 ± 0.068 | 0.17 ± 0.16 | 0.21 ± 0.18 | 1.79 ± 0.18 |
| Uniform fit (1000) | 0.474 ± 0.026 | 0.34 ± 0.12 | 0.22 ± 0.12 | 0.32 ± 0.12 | 0.300 ± 0.067 | 0.48 ± 0.19 | 0.32 ± 0.14 | 0.57 ± 0.06 |
| Random weights, ESS 100 | 0.476 ± 0.026 | 0.34 ± 0.12 | 0.22 ± 0.12 | 0.33 ± 0.13 | 0.301 ± 0.068 | 0.49 ± 0.19 | 0.34 ± 0.14 | 0.59 ± 0.06 |
| Random n = 100 | 0.476 ± 0.026 | 0.34 ± 0.12 | 0.22 ± 0.12 | 0.33 ± 0.13 | 0.301 ± 0.067 | 0.49 ± 0.19 | 0.33 ± 0.14 | 0.58 ± 0.06 |
| **Similarity, ESS 100** | 0.493 ± 0.029 | 0.34 ± 0.12 | 0.22 ± 0.12 | 0.34 ± 0.14 | 0.318 ± 0.065 | 0.47 ± 0.18 | 0.35 ± 0.14 | 0.64 ± 0.07 |
| **Top-n, n = 100** | 0.495 ± 0.029 | 0.34 ± 0.12 | 0.22 ± 0.12 | 0.38 ± 0.14 | 0.317 ± 0.065 | 0.47 ± 0.18 | 0.36 ± 0.14 | 0.63 ± 0.07 |
| Random weights, ESS 500 | 0.474 ± 0.026 | 0.34 ± 0.12 | 0.22 ± 0.12 | 0.33 ± 0.13 | 0.299 ± 0.067 | 0.49 ± 0.19 | 0.34 ± 0.14 | 0.59 ± 0.06 |
| Random n = 500 | 0.475 ± 0.026 | 0.34 ± 0.12 | 0.22 ± 0.12 | 0.33 ± 0.13 | 0.301 ± 0.067 | 0.48 ± 0.19 | 0.34 ± 0.14 | 0.58 ± 0.06 |
| **Similarity, ESS 500** | 0.481 ± 0.028 | 0.34 ± 0.12 | 0.22 ± 0.12 | 0.36 ± 0.13 | 0.314 ± 0.065 | 0.49 ± 0.19 | 0.36 ± 0.14 | 0.60 ± 0.07 |
| **Top-n, n = 500** | 0.482 ± 0.028 | 0.34 ± 0.12 | 0.22 ± 0.12 | 0.36 ± 0.13 | 0.314 ± 0.065 | 0.49 ± 0.19 | 0.36 ± 0.14 | 0.60 ± 0.07 |
| **Wider selection (top 1000)** | 0.498 ± 0.029 | 0.34 ± 0.13 | 0.22 ± 0.12 | 0.40 ± 0.14 | 0.315 ± 0.067 | 0.47 ± 0.18 | 0.36 ± 0.14 | 0.64 ± 0.08 |
