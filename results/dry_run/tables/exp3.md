# Experiment 3 (validation)

Cells are the mean ± half the 95% bootstrap interval, resampling windows, prompts or templates. Seeded lenses are averaged over seeds first. Jacobian error, overlap, KL and edit error average layers 12 and 18. Pass@k takes the best layer in 9–22. Gap = target minus control. Edit error is the median relative error at 5% of activation norm. Lower is better for Jacobian error, control pass rate, KL and edit error.

## Deployed on python

| Lens | Jacobian error | Pass@10 | Pass@10 control | AUC gap | Swap, α=1 | Edit error |
|---|---|---|---|---|---|---|
| Logit lens | 0.944 ± 0.004 | 0.200 ± 0.200 | 0.000 ± 0.000 | 0.357 ± 0.196 | 0.306 ± 0.417 | 1.766 ± 0.180 |
| Uniform fit (1000) | 0.474 ± 0.042 | 0.300 ± 0.300 | 0.300 ± 0.300 | 0.245 ± 0.111 | 0.694 ± 0.250 | 0.461 ± 0.048 |
| **Wider selection (top 1000)** | 0.476 ± 0.044 | 0.300 ± 0.300 | 0.300 ± 0.300 | 0.257 ± 0.110 | 0.722 ± 0.250 | 0.483 ± 0.060 |

### Paired differences, python

First lens minus second, ± half the 95% interval. **Bold** intervals exclude zero.

| Comparison | Jacobian error | Pass@10 gap | Pass@50 gap | AUC gap | Swap, α=1 | Log-prob shift, α=1 | Edit error |
|---|---|---|---|---|---|---|---|
| Wider selection (top 1000) − Uniform fit (1000) | +0.002 ± 0.004 | +0.000 ± 0.000 | +0.000 ± 0.000 | +0.012 ± 0.017 | +0.028 ± 0.042 | +0.140 ± 0.237 | +0.022 ± 0.030 |
| Uniform fit (1000) − Logit lens | **-0.470 ± 0.038** | -0.200 ± 0.200 | +0.100 ± 0.502 | -0.112 ± 0.221 | **+0.389 ± 0.292** | **+5.951 ± 4.626** | **-1.305 ± 0.191** |

## Deployed on stories

| Lens | Jacobian error | Pass@10 | Pass@10 control | AUC gap | Swap, α=1 | Edit error |
|---|---|---|---|---|---|---|
| Logit lens | 0.923 ± 0.002 | 0.200 ± 0.276 | 0.000 ± 0.000 | 0.362 ± 0.179 | 0.028 ± 0.042 | 1.598 ± 0.230 |
| Uniform fit (1000) | 0.353 ± 0.014 | 0.300 ± 0.276 | 0.000 ± 0.000 | 0.372 ± 0.159 | 0.694 ± 0.250 | 0.370 ± 0.039 |
| **Wider selection (top 1000)** | 0.361 ± 0.015 | 0.200 ± 0.276 | 0.000 ± 0.000 | 0.373 ± 0.132 | 0.667 ± 0.292 | 0.394 ± 0.052 |

### Paired differences, stories

First lens minus second, ± half the 95% interval. **Bold** intervals exclude zero.

| Comparison | Jacobian error | Pass@10 gap | Pass@50 gap | AUC gap | Swap, α=1 | Log-prob shift, α=1 | Edit error |
|---|---|---|---|---|---|---|---|
| Wider selection (top 1000) − Uniform fit (1000) | **+0.008 ± 0.003** | -0.100 ± 0.150 | +0.000 ± 0.000 | +0.002 ± 0.046 | -0.028 ± 0.042 | -0.003 ± 0.104 | **+0.024 ± 0.021** |
| Uniform fit (1000) − Logit lens | **-0.570 ± 0.014** | +0.100 ± 0.150 | +0.100 ± 0.300 | +0.010 ± 0.129 | **+0.667 ± 0.208** | **+4.808 ± 0.939** | **-1.228 ± 0.228** |

## Deployed on wiki

| Lens | Jacobian error | Pass@10 | Pass@10 control | AUC gap | Swap, α=1 | Edit error |
|---|---|---|---|---|---|---|
| Logit lens | 0.940 ± 0.003 | 0.091 ± 0.136 | 0.000 ± 0.000 | 0.458 ± 0.090 | 0.000 ± 0.000 | 1.407 ± 0.230 |
| Uniform fit (1000) | 0.479 ± 0.022 | 0.455 ± 0.273 | 0.000 ± 0.000 | 0.679 ± 0.102 | 0.333 ± 0.417 | 0.415 ± 0.062 |
| **Wider selection (top 1000)** | 0.491 ± 0.018 | 0.364 ± 0.227 | 0.000 ± 0.000 | 0.644 ± 0.115 | 0.389 ± 0.458 | 0.390 ± 0.056 |

### Paired differences, wiki

First lens minus second, ± half the 95% interval. **Bold** intervals exclude zero.

| Comparison | Jacobian error | Pass@10 gap | Pass@50 gap | AUC gap | Swap, α=1 | Log-prob shift, α=1 | Edit error |
|---|---|---|---|---|---|---|---|
| Wider selection (top 1000) − Uniform fit (1000) | **+0.012 ± 0.006** | -0.091 ± 0.136 | -0.091 ± 0.136 | **-0.035 ± 0.031** | +0.056 ± 0.042 | -0.217 ± 0.458 | -0.025 ± 0.040 |
| Uniform fit (1000) − Logit lens | **-0.462 ± 0.019** | **+0.364 ± 0.251** | **+0.545 ± 0.273** | **+0.221 ± 0.063** | +0.333 ± 0.417 | **+5.828 ± 5.812** | **-0.992 ± 0.256** |
