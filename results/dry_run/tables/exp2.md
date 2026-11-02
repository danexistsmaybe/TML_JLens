# Experiment 2 (validation)

Cells are the mean ± half the 95% bootstrap interval, resampling windows, prompts or templates. Seeded lenses are averaged over seeds first. Jacobian error, overlap, KL and edit error average layers 12 and 18. Pass@k takes the best layer in 9–22. Gap = target minus control. Edit error is the median relative error at 5% of activation norm. Lower is better for Jacobian error, control pass rate, KL and edit error.

## Deployed on python

| Lens | Jacobian error | Pass@10 | Pass@10 control | AUC gap | Swap, α=1 | Top-10 overlap | Symmetric KL | Edit error |
|---|---|---|---|---|---|---|---|---|
| Logit lens | 0.944 ± 0.004 | 0.200 ± 0.200 | 0.000 ± 0.000 | 0.357 ± 0.196 | 0.306 ± 0.417 | 0.053 ± 0.007 | 8.504 ± 0.141 | 1.766 ± 0.180 |
| Uniform pool | 0.755 ± 0.016 | 0.300 ± 0.300 | 0.100 ± 0.150 | 0.391 ± 0.124 | 0.458 ± 0.375 | 0.217 ± 0.011 | 3.743 ± 0.121 | 1.098 ± 0.120 |
| Random weights | 0.764 ± 0.016 | 0.350 ± 0.301 | 0.100 ± 0.150 | 0.399 ± 0.129 | 0.458 ± 0.375 | 0.211 ± 0.010 | 3.808 ± 0.123 | 1.121 ± 0.108 |
| Random n | 0.749 ± 0.016 | 0.300 ± 0.300 | 0.100 ± 0.150 | 0.385 ± 0.120 | 0.458 ± 0.375 | 0.222 ± 0.010 | 3.717 ± 0.120 | 1.056 ± 0.133 |
| **Similarity weighting** | 0.729 ± 0.017 | 0.350 ± 0.275 | 0.100 ± 0.150 | 0.371 ± 0.120 | 0.472 ± 0.354 | 0.245 ± 0.010 | 3.656 ± 0.131 | 1.076 ± 0.173 |
| **Top-n selection** | 0.737 ± 0.017 | 0.450 ± 0.314 | 0.100 ± 0.150 | 0.373 ± 0.127 | 0.486 ± 0.354 | 0.238 ± 0.009 | 3.847 ± 0.143 | 1.180 ± 0.192 |
| Python oracle (1000) | 0.474 ± 0.042 | 0.300 ± 0.300 | 0.300 ± 0.300 | 0.245 ± 0.111 | 0.694 ± 0.250 | 1.000 ± 0.000 | 0.000 ± 0.000 | 0.461 ± 0.048 |
| Python oracle (500) | nan ± nan | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.002 ± 0.001 | nan ± nan | nan ± nan |

### Paired differences, python

First lens minus second, ± half the 95% interval. **Bold** intervals exclude zero.

| Comparison | Jacobian error | Pass@10 gap | Pass@50 gap | AUC gap | Swap, α=1 | Log-prob shift, α=1 | Top-10 overlap | Edit error |
|---|---|---|---|---|---|---|---|---|
| Similarity weighting − Uniform pool | **-0.026 ± 0.002** | +0.050 ± 0.075 | -0.100 ± 0.150 | -0.020 ± 0.041 | +0.014 ± 0.021 | **+0.506 ± 0.342** | **+0.027 ± 0.004** | -0.022 ± 0.086 |
| Top-n selection − Uniform pool | **-0.018 ± 0.002** | +0.150 ± 0.188 | -0.100 ± 0.150 | -0.017 ± 0.053 | +0.028 ± 0.021 | **+0.544 ± 0.410** | **+0.021 ± 0.005** | **+0.082 ± 0.095** |
| Similarity weighting − Random weights | **-0.035 ± 0.003** | +0.000 ± 0.138 | -0.100 ± 0.150 | -0.028 ± 0.057 | +0.014 ± 0.021 | **+0.564 ± 0.321** | **+0.034 ± 0.004** | -0.046 ± 0.094 |
| Top-n selection − Random n | **-0.012 ± 0.002** | +0.150 ± 0.188 | -0.100 ± 0.150 | -0.012 ± 0.046 | +0.028 ± 0.021 | **+0.411 ± 0.354** | **+0.016 ± 0.004** | **+0.124 ± 0.080** |
| Similarity weighting − Python oracle (1000) | **+0.255 ± 0.029** | +0.250 ± 0.238 | +0.000 ± 0.000 | **+0.126 ± 0.059** | **-0.222 ± 0.104** | **-2.919 ± 2.039** | **-0.755 ± 0.010** | **+0.615 ± 0.151** |
| Top-n selection − Python oracle (1000) | **+0.263 ± 0.029** | **+0.350 ± 0.250** | +0.000 ± 0.000 | **+0.128 ± 0.054** | **-0.208 ± 0.104** | **-2.881 ± 1.971** | **-0.762 ± 0.009** | **+0.719 ± 0.168** |
| Uniform pool − Logit lens | **-0.189 ± 0.013** | +0.000 ± 0.226 | +0.200 ± 0.526 | +0.034 ± 0.212 | +0.153 ± 0.188 | **+2.526 ± 2.245** | **+0.165 ± 0.009** | **-0.668 ± 0.195** |
| Python oracle (1000) − Uniform pool | **-0.281 ± 0.030** | -0.200 ± 0.200 | -0.100 ± 0.150 | **-0.146 ± 0.090** | **+0.236 ± 0.125** | **+3.425 ± 2.381** | **+0.783 ± 0.011** | **-0.637 ± 0.093** |
