# Experiment 2 (test)

Cells are the mean ± half the 95% bootstrap interval, resampling windows, prompts or templates. Seeded lenses are averaged over seeds first. Jacobian error, overlap, KL and edit error average layers 12 and 18. Pass@k takes the best layer in 9–22. Gap = target minus control. Edit error is the median relative error at 5% of activation norm. Lower is better for Jacobian error, control pass rate, KL and edit error.

## Deployed on python

| Lens | Jacobian error | Pass@10 | Pass@10 control | AUC gap | Swap, α=1 | Top-10 overlap | Symmetric KL | Edit error |
|---|---|---|---|---|---|---|---|---|
| Logit lens | 0.942 ± 0.003 | 0.180 ± 0.110 | 0.100 ± 0.090 | 0.218 ± 0.068 | 0.167 ± 0.162 | 0.057 ± 0.006 | 8.863 ± 0.160 | 1.786 ± 0.183 |
| Uniform pool | 0.762 ± 0.014 | 0.300 ± 0.130 | 0.180 ± 0.110 | 0.299 ± 0.068 | 0.265 ± 0.168 | 0.236 ± 0.007 | 3.722 ± 0.087 | 0.992 ± 0.120 |
| Random weights | 0.763 ± 0.015 | 0.307 ± 0.127 | 0.193 ± 0.107 | 0.295 ± 0.068 | 0.258 ± 0.172 | 0.239 ± 0.006 | 3.703 ± 0.091 | 0.999 ± 0.117 |
| Random n | 0.762 ± 0.014 | 0.300 ± 0.130 | 0.180 ± 0.103 | 0.299 ± 0.067 | 0.261 ± 0.170 | 0.238 ± 0.007 | 3.690 ± 0.086 | 1.002 ± 0.109 |
| **Similarity weighting** | 0.701 ± 0.015 | 0.280 ± 0.130 | 0.220 ± 0.120 | 0.295 ± 0.060 | 0.353 ± 0.207 | 0.340 ± 0.008 | 2.781 ± 0.099 | 0.871 ± 0.109 |
| **Top-n selection** | 0.702 ± 0.015 | 0.300 ± 0.130 | 0.220 ± 0.120 | 0.295 ± 0.061 | 0.359 ± 0.206 | 0.335 ± 0.008 | 2.879 ± 0.106 | 0.872 ± 0.103 |
| Python oracle (1000) | 0.474 ± 0.026 | 0.340 ± 0.120 | 0.220 ± 0.120 | 0.300 ± 0.067 | 0.480 ± 0.191 | 0.979 ± 0.001 | 0.001 ± 0.000 | 0.571 ± 0.061 |
| Python oracle (500) | 0.475 ± 0.026 | 0.320 ± 0.130 | 0.220 ± 0.120 | 0.303 ± 0.067 | 0.480 ± 0.191 | 0.959 ± 0.001 | 0.004 ± 0.000 | 0.583 ± 0.062 |

### Paired differences, python

First lens minus second, ± half the 95% interval. **Bold** intervals exclude zero.

| Comparison | Jacobian error | Pass@10 gap | Pass@50 gap | AUC gap | Swap, α=1 | Log-prob shift, α=1 | Top-10 overlap | Edit error |
|---|---|---|---|---|---|---|---|---|
| Similarity weighting − Uniform pool | **-0.061 ± 0.005** | -0.060 ± 0.070 | -0.020 ± 0.070 | -0.003 ± 0.022 | **+0.088 ± 0.053** | **+1.290 ± 0.456** | **+0.105 ± 0.007** | **-0.121 ± 0.083** |
| Top-n selection − Uniform pool | **-0.060 ± 0.004** | -0.040 ± 0.050 | -0.033 ± 0.060 | -0.004 ± 0.023 | **+0.095 ± 0.053** | **+1.252 ± 0.426** | **+0.099 ± 0.007** | **-0.120 ± 0.078** |
| Similarity weighting − Random weights | **-0.062 ± 0.005** | **-0.053 ± 0.057** | -0.020 ± 0.070 | +0.000 ± 0.020 | **+0.095 ± 0.050** | **+1.185 ± 0.403** | **+0.101 ± 0.006** | **-0.128 ± 0.075** |
| Top-n selection − Random n | **-0.060 ± 0.004** | -0.040 ± 0.047 | -0.033 ± 0.060 | -0.003 ± 0.023 | **+0.098 ± 0.052** | **+1.235 ± 0.407** | **+0.097 ± 0.007** | **-0.130 ± 0.077** |
| Similarity weighting − Python oracle (1000) | **+0.227 ± 0.020** | -0.060 ± 0.060 | -0.020 ± 0.070 | -0.005 ± 0.036 | **-0.127 ± 0.033** | **-2.500 ± 0.983** | **-0.638 ± 0.008** | **+0.300 ± 0.094** |
| Top-n selection − Python oracle (1000) | **+0.227 ± 0.020** | -0.040 ± 0.050 | -0.033 ± 0.060 | -0.005 ± 0.036 | **-0.121 ± 0.036** | **-2.538 ± 0.995** | **-0.644 ± 0.009** | **+0.301 ± 0.089** |
| Uniform pool − Logit lens | **-0.180 ± 0.013** | +0.040 ± 0.100 | **+0.160 ± 0.150** | **+0.081 ± 0.059** | **+0.098 ± 0.090** | **+3.560 ± 1.694** | **+0.179 ± 0.008** | **-0.794 ± 0.188** |
| Python oracle (1000) − Uniform pool | **-0.288 ± 0.021** | +0.000 ± 0.080 | +0.000 ± 0.090 | +0.001 ± 0.039 | **+0.216 ± 0.054** | **+3.790 ± 1.362** | **+0.743 ± 0.006** | **-0.421 ± 0.101** |
