# Experiment 2 (validation)

Cells are the mean ± half the 95% bootstrap interval, resampling windows, prompts or templates. Seeded lenses are averaged over seeds first. Jacobian error, overlap, KL and edit error average layers 12 and 18. Pass@k takes the best layer in 9–22. Gap = target minus control. Edit error is the median relative error at 5% of activation norm. Lower is better for Jacobian error, control pass rate, KL and edit error.

## Deployed on python

| Lens | Jacobian error | Pass@10 | Pass@10 control | AUC gap | Swap, α=1 | Top-10 overlap | Symmetric KL | Edit error |
|---|---|---|---|---|---|---|---|---|
| Logit lens | 0.944 ± 0.005 | 0.200 ± 0.250 | 0.000 ± 0.000 | 0.357 ± 0.187 | 0.306 ± 0.417 | 0.052 ± 0.007 | 8.783 ± 0.178 | 1.766 ± 0.172 |
| Uniform pool | 0.746 ± 0.019 | 0.300 ± 0.300 | 0.100 ± 0.150 | 0.380 ± 0.135 | 0.444 ± 0.375 | 0.228 ± 0.011 | 3.697 ± 0.123 | 1.065 ± 0.113 |
| Random weights | 0.746 ± 0.020 | 0.333 ± 0.250 | 0.100 ± 0.150 | 0.385 ± 0.137 | 0.463 ± 0.375 | 0.231 ± 0.011 | 3.670 ± 0.122 | 1.068 ± 0.130 |
| Random n | 0.746 ± 0.019 | 0.300 ± 0.300 | 0.100 ± 0.150 | 0.385 ± 0.133 | 0.454 ± 0.375 | 0.230 ± 0.011 | 3.671 ± 0.123 | 1.083 ± 0.133 |
| **Similarity weighting** | 0.684 ± 0.021 | 0.367 ± 0.283 | 0.267 ± 0.267 | 0.344 ± 0.142 | 0.556 ± 0.292 | 0.338 ± 0.010 | 2.762 ± 0.119 | 1.035 ± 0.141 |
| **Top-n selection** | 0.684 ± 0.022 | 0.300 ± 0.300 | 0.200 ± 0.217 | 0.340 ± 0.138 | 0.556 ± 0.292 | 0.332 ± 0.010 | 2.855 ± 0.130 | 0.987 ± 0.143 |
| Python oracle (1000) | 0.458 ± 0.039 | 0.300 ± 0.300 | 0.300 ± 0.300 | 0.222 ± 0.135 | 0.694 ± 0.250 | 0.979 ± 0.001 | 0.001 ± 0.000 | 0.475 ± 0.054 |
| Python oracle (500) | 0.458 ± 0.039 | 0.300 ± 0.300 | 0.300 ± 0.300 | 0.227 ± 0.135 | 0.694 ± 0.250 | 0.961 ± 0.002 | 0.004 ± 0.000 | 0.469 ± 0.057 |

### Paired differences, python

First lens minus second, ± half the 95% interval. **Bold** intervals exclude zero.

| Comparison | Jacobian error | Pass@10 gap | Pass@50 gap | AUC gap | Swap, α=1 | Log-prob shift, α=1 | Top-10 overlap | Edit error |
|---|---|---|---|---|---|---|---|---|
| Similarity weighting − Uniform pool | **-0.062 ± 0.004** | -0.100 ± 0.250 | -0.100 ± 0.150 | -0.036 ± 0.059 | +0.111 ± 0.083 | **+1.150 ± 0.863** | **+0.111 ± 0.008** | -0.029 ± 0.064 |
| Top-n selection − Uniform pool | **-0.062 ± 0.004** | -0.100 ± 0.133 | -0.100 ± 0.150 | -0.040 ± 0.057 | +0.111 ± 0.083 | **+1.125 ± 0.844** | **+0.104 ± 0.009** | -0.077 ± 0.076 |
| Similarity weighting − Random weights | **-0.062 ± 0.004** | -0.133 ± 0.267 | -0.100 ± 0.150 | -0.041 ± 0.058 | +0.093 ± 0.083 | **+1.070 ± 0.734** | **+0.108 ± 0.008** | -0.033 ± 0.050 |
| Top-n selection − Random n | **-0.062 ± 0.004** | -0.100 ± 0.133 | -0.100 ± 0.150 | -0.045 ± 0.058 | +0.102 ± 0.083 | **+1.139 ± 0.837** | **+0.102 ± 0.009** | **-0.096 ± 0.078** |
| Similarity weighting − Python oracle (1000) | **+0.226 ± 0.027** | +0.100 ± 0.117 | +0.000 ± 0.000 | **+0.122 ± 0.056** | **-0.139 ± 0.042** | **-2.164 ± 1.475** | **-0.641 ± 0.010** | **+0.560 ± 0.117** |
| Top-n selection − Python oracle (1000) | **+0.226 ± 0.027** | +0.100 ± 0.133 | +0.000 ± 0.000 | **+0.118 ± 0.057** | **-0.139 ± 0.042** | **-2.188 ± 1.494** | **-0.647 ± 0.010** | **+0.512 ± 0.123** |
| Uniform pool − Logit lens | **-0.198 ± 0.015** | +0.000 ± 0.300 | +0.200 ± 0.500 | +0.023 ± 0.217 | +0.139 ± 0.167 | **+2.640 ± 2.435** | **+0.176 ± 0.010** | **-0.702 ± 0.184** |
| Python oracle (1000) − Uniform pool | **-0.288 ± 0.028** | -0.200 ± 0.250 | -0.100 ± 0.150 | **-0.158 ± 0.086** | **+0.250 ± 0.125** | **+3.314 ± 2.338** | **+0.752 ± 0.011** | **-0.589 ± 0.097** |
