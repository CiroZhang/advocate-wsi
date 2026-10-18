# Paired bootstrap
Paired slide-level bootstrap for the main comparisons (95% CI of the score difference, one-sided p).
Test slides are resampled within each fold (same draw for every seed of that fold); the statistic is the mean over runs,
matching Table 1. CAMELYON16 has one test set (3 seeds).
usage: python stats.py   (reads ../results, writes ../results/stats.{md,json})
| dataset | comparison | diff | 95% CI | p (one-sided) |
|---|---|---|---|---|
| C16 | attacked: ours - single selector | +0.617 | [+0.553, +0.673] | 0.0005 |
| C16 | clean: ours - single selector | -0.001 | [-0.017, +0.017] | 0.4563 |
| C16 | clean: ours - full slide | -0.027 | [-0.060, +0.003] | 0.0470 |
| C16 | attacked: ours - no presenter flags | +0.008 | [-0.004, +0.021] | 0.0905 |
| C17 | attacked: ours - single selector | +0.956 | [+0.898, +0.996] | 0.0005 |
| C17 | clean: ours - single selector | -0.050 | [-0.082, -0.024] | 0.0005 |
| C17 | clean: ours - full slide | -0.068 | [-0.110, -0.030] | 0.0010 |
| C17 | attacked: ours - no presenter flags | -0.004 | [-0.027, +0.018] | 0.3608 |
| RCC | attacked: ours - single selector | +0.845 | [+0.819, +0.867] | 0.0005 |
| RCC | clean: ours - single selector | +0.013 | [-0.007, +0.032] | 0.0920 |
| RCC | clean: ours - full slide | -0.013 | [-0.028, +0.002] | 0.0480 |
| RCC | attacked: ours - no presenter flags | -0.001 | [-0.015, +0.014] | 0.4738 |
| NSCLC | attacked: ours - single selector | +0.838 | [+0.818, +0.856] | 0.0005 |
| NSCLC | clean: ours - single selector | -0.035 | [-0.049, -0.022] | 0.0005 |
| NSCLC | clean: ours - full slide | -0.049 | [-0.063, -0.036] | 0.0005 |
| NSCLC | attacked: ours - no presenter flags | +0.003 | [-0.009, +0.014] | 0.3128 |
