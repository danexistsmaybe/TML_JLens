# Validation checks

## 1. Positive control

First lens minus second, with a 95% interval. A metric that can't separate these can't detect a smaller gain.

| Deployed on | Comparison | Metric | Difference | Interval | Separates? |
|---|---|---|---|---|---|
| wiki | oracle:wiki − oracle:stories | jacobian_error | -0.232 | [-0.249, -0.214] | yes |
| wiki | oracle:wiki − oracle:stories | pass10_gap | +0.000 | [+0.000, +0.000] | no |
| wiki | oracle:wiki − oracle:stories | pass50_gap | +0.000 | [+0.000, +0.000] | no |
| wiki | oracle:wiki − oracle:stories | auc_gap | +0.009 | [-0.048, +0.077] | no |
| wiki | oracle:wiki − oracle:stories | swap_1 | +0.083 | [+0.000, +0.167] | no |
| wiki | oracle:wiki − oracle:stories | swap_2 | +0.056 | [-0.083, +0.250] | no |
| wiki | oracle:wiki − oracle:stories | swap_shift | +0.532 | [-0.682, +1.403] | no |
| wiki | oracle:wiki − oracle:stories | overlap | +0.594 | [+0.582, +0.606] | yes |
| wiki | oracle:wiki − oracle:stories | sym_kl | -1.900 | [-1.976, -1.817] | yes |
| wiki | oracle:wiki − oracle:stories | edit_error | -0.254 | [-0.355, -0.175] | yes |
| wiki | oracle:wiki − oracle:stories | edit_antisym | -0.094 | [-0.129, -0.038] | yes |
| stories | oracle:stories − oracle:wiki | jacobian_error | -0.246 | [-0.261, -0.229] | yes |
| stories | oracle:stories − oracle:wiki | pass10_gap | +0.000 | [+0.000, +0.000] | no |
| stories | oracle:stories − oracle:wiki | pass50_gap | +0.200 | [+0.000, +0.500] | no |
| stories | oracle:stories − oracle:wiki | auc_gap | +0.012 | [-0.105, +0.147] | no |
| stories | oracle:stories − oracle:wiki | swap_1 | +0.000 | [-0.167, +0.083] | no |
| stories | oracle:stories − oracle:wiki | swap_2 | +0.028 | [-0.083, +0.083] | no |
| stories | oracle:stories − oracle:wiki | swap_shift | +0.349 | [-0.268, +0.689] | no |
| stories | oracle:stories − oracle:wiki | overlap | +0.578 | [+0.566, +0.590] | yes |
| stories | oracle:stories − oracle:wiki | sym_kl | -1.953 | [-2.064, -1.840] | yes |
| stories | oracle:stories − oracle:wiki | edit_error | -0.087 | [-0.149, -0.023] | yes |
| stories | oracle:stories − oracle:wiki | edit_antisym | -0.070 | [-0.104, -0.044] | yes |
| python | oracle:python − uniform_pool | jacobian_error | -0.288 | [-0.315, -0.259] | yes |
| python | oracle:python − uniform_pool | pass10_gap | -0.200 | [-0.500, +0.000] | no |
| python | oracle:python − uniform_pool | pass50_gap | -0.100 | [-0.300, +0.000] | no |
| python | oracle:python − uniform_pool | auc_gap | -0.158 | [-0.250, -0.078] | yes |
| python | oracle:python − uniform_pool | swap_1 | +0.250 | [+0.083, +0.333] | yes |
| python | oracle:python − uniform_pool | swap_2 | +0.139 | [-0.083, +0.333] | no |
| python | oracle:python − uniform_pool | swap_shift | +3.314 | [+1.074, +5.749] | yes |
| python | oracle:python − uniform_pool | overlap | +0.752 | [+0.740, +0.763] | yes |
| python | oracle:python − uniform_pool | sym_kl | -3.696 | [-3.812, -3.566] | yes |
| python | oracle:python − uniform_pool | edit_error | -0.589 | [-0.694, -0.500] | yes |
| python | oracle:python − uniform_pool | edit_antisym | -0.167 | [-0.200, -0.124] | yes |

## 2. Short prompts or a neutral prefix

Own-domain oracle on validation prompts. Rule: switch only if pass@10 gap rises by at least 20 points.

| Domain | Form | Pass@10 | Control | Gap | Pass@50 | Control | AUC | Swap α=1 |
|---|---|---|---|---|---|---|---|---|
| wiki | short | 0.45 | 0.00 | +0.45 | 1.00 | 0.00 | 0.70 | 0.33 |
| wiki | prefix | 0.55 | 0.00 | +0.55 | 1.00 | 0.00 | 0.67 | 0.36 |
| stories | short | 0.30 | 0.00 | +0.30 | 0.40 | 0.00 | 0.42 | 0.67 |
| stories | prefix | 0.20 | 0.00 | +0.20 | 0.50 | 0.00 | 0.41 | 0.58 |
| python | short | 0.30 | 0.30 | +0.00 | 0.70 | 0.30 | 0.50 | 0.69 |
| python | prefix | 0.30 | 0.30 | +0.00 | 0.50 | 0.30 | 0.48 | 0.67 |

**Decision: keep short prompts.**

## 3. Surviving swap trials

| Domain | Validation | Test |
|---|---|---|
| wiki | 36 | 108 |
| stories | 36 | 72 |
| python | 36 | 102 |

## 4. Settings and weight share

Chosen by Jacobian error on the second half of validation, averaged over layers 12 and 18 and seeds. Share is the weight on wiki windows; stories get the rest.

### wiki: ESS 400, n 400. Uniform pool error 0.5410

| ESS or n | Similarity error | Top-n error | Similarity wiki share | Top-n wiki share |
|---|---|---|---|---|
| 10 | 0.5171 | 0.5232 | 1.00 | 1.00 |
| 25 | 0.4975 | 0.5058 | 1.00 | 1.00 |
| 50 | 0.4872 | 0.4901 | 1.00 | 1.00 |
| 100 | 0.4795 | 0.4797 | 1.00 | 1.00 |
| 150 | 0.4759 | 0.4753 | 1.00 | 1.00 |
| 200 | 0.4737 | 0.4729 | 1.00 | 1.00 |
| 300 | 0.4711 | 0.4705 | 1.00 | 1.00 |
| 400 | 0.4698 | 0.4697 | 1.00 | 1.00 |
| 500 | 0.4702 | 0.4710 | 0.96 | 1.00 |
| 750 | 0.4850 | 0.5015 | 0.78 | 0.67 |

### stories: ESS 400, n 400. Uniform pool error 0.4205

| ESS or n | Similarity error | Top-n error | Similarity wiki share | Top-n wiki share |
|---|---|---|---|---|
| 10 | 0.3576 | 0.3603 | 0.00 | 0.00 |
| 25 | 0.3519 | 0.3533 | 0.00 | 0.00 |
| 50 | 0.3493 | 0.3500 | 0.00 | 0.00 |
| 100 | 0.3472 | 0.3484 | 0.00 | 0.00 |
| 150 | 0.3460 | 0.3465 | 0.00 | 0.00 |
| 200 | 0.3451 | 0.3456 | 0.00 | 0.00 |
| 300 | 0.3440 | 0.3441 | 0.00 | 0.00 |
| 400 | 0.3435 | 0.3436 | 0.00 | 0.00 |
| 500 | 0.3439 | 0.3451 | 0.03 | 0.00 |
| 750 | 0.3594 | 0.3775 | 0.22 | 0.33 |

### python: ESS 25, n 50. Uniform pool error 0.7461

| ESS or n | Similarity error | Top-n error | Similarity wiki share | Top-n wiki share |
|---|---|---|---|---|
| 10 | 0.6931 | 0.6878 | 1.00 | 1.00 |
| 25 | 0.6843 | 0.6848 | 1.00 | 1.00 |
| 50 | 0.6845 | 0.6838 | 1.00 | 0.99 |
| 100 | 0.6885 | 0.6874 | 1.00 | 1.00 |
| 150 | 0.6920 | 0.6936 | 0.99 | 1.00 |
| 200 | 0.6950 | 0.6986 | 0.98 | 1.00 |
| 300 | 0.6994 | 0.7070 | 0.96 | 1.00 |
| 400 | 0.7029 | 0.7139 | 0.93 | 1.00 |
| 500 | 0.7060 | 0.7185 | 0.88 | 0.98 |
| 750 | 0.7156 | 0.7236 | 0.75 | 0.67 |

## 5. Edit error by size

Own-domain oracle. The ± part cancels the edit's curvature.

| Domain | Layer | Error at 5% | Error at 10% | ± part at 5% | ± part at 10% | Correlation at 10% |
|---|---|---|---|---|---|---|
| wiki | 12 | 0.716 | 0.706 | 0.572 | 0.577 | 0.865 |
| wiki | 18 | 0.220 | 0.215 | 0.200 | 0.197 | 0.927 |
| stories | 12 | 0.532 | 0.550 | 0.435 | 0.444 | 0.951 |
| stories | 18 | 0.244 | 0.255 | 0.193 | 0.194 | 0.947 |
| python | 12 | 0.512 | 0.526 | 0.453 | 0.452 | 0.787 |
| python | 18 | 0.441 | 0.410 | 0.313 | 0.312 | 0.857 |
