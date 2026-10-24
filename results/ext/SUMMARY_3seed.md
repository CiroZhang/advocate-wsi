# EXT experiments, seeds [0, 1, 2] (5 folds each; mean over folds, then mean ± sd over seeds)

| | C17 κ | RCC BA | NSCLC BA |
|---|---|---|---|
| single selector, clean | 0.873 ± 0.026 (n=3) | 0.907 ± 0.023 (n=3) | 0.909 ± 0.008 (n=3) |
| single selector, attacked | -0.241 ± 0.011 (n=3) | 0.022 ± 0.002 (n=3) | 0.003 ± 0.001 (n=3) |
| single selector, weak scorer | 0.867 ± 0.030 (n=3) | 0.899 ± 0.013 (n=3) | 0.901 ± 0.003 (n=3) |
| advocates, clean | 0.823 ± 0.015 (n=3) | 0.920 ± 0.010 (n=3) | 0.874 ± 0.007 (n=3) |
| advocates, per-tile attack | 0.733 ± 0.019 (n=3) | 0.870 ± 0.006 (n=3) | 0.840 ± 0.019 (n=3) |
| advocates, adaptive attack | 0.646 ± 0.010 (n=3) | 0.695 ± 0.025 (n=3) | 0.666 ± 0.015 (n=3) |
| hybrid (4 tiles), clean | 0.856 ± 0.005 (n=3) | 0.927 ± 0.011 (n=3) | 0.899 ± 0.002 (n=3) |
| hybrid (4 tiles), adaptive attack | 0.713 ± 0.011 (n=3) | 0.716 ± 0.015 (n=3) | 0.683 ± 0.021 (n=3) |
| hybrid (K tiles), clean | 0.878 ± 0.011 (n=3) | 0.928 ± 0.004 (n=3) | 0.905 ± 0.008 (n=3) |
| hybrid (K tiles), per-tile attack | 0.771 ± 0.015 (n=3) | 0.870 ± 0.010 (n=3) | 0.811 ± 0.016 (n=3) |
| hybrid (K tiles), adaptive attack | 0.737 ± 0.025 (n=3) | 0.775 ± 0.016 (n=3) | 0.707 ± 0.029 (n=3) |
| independent scorers, clean | 0.840 ± 0.020 (n=3) | 0.932 ± 0.007 (n=3) | 0.873 ± 0.004 (n=3) |
| independent scorers, attacked | 0.713 ± 0.037 (n=3) | 0.883 ± 0.020 (n=3) | 0.845 ± 0.011 (n=3) |
| advocates, weak scorer | 0.774 ± 0.015 (n=3) | 0.932 ± 0.009 (n=3) | 0.874 ± 0.007 (n=3) |
| advocates, weak scorer, attacked | 0.592 ± 0.027 (n=3) | 0.875 ± 0.010 (n=3) | 0.831 ± 0.018 (n=3) |

- Over 3 seeds the hybrid presenter (advocates + neutral K-tile slot) **matches the single selector clean** (C17 .878 vs .873, RCC .928 vs .907, NSCLC .905 vs .909), while under the adaptive attack it keeps .71–.78 where the attacked single selector is at chance or below (−.24 / .02 / .00).
- Weak scorer (10% train slides) costs the single selector ≤.01: the fragility is adversarial, not a quality issue.
- Seed-0 numbers here differ slightly from SUMMARY.md (that file pooled folds; this one averages folds).
