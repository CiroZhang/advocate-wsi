# EXT experiments_atty, seeds [0, 1, 2] (5 folds each; mean over folds, then mean ± sd over seeds)

| | C17 κ | RCC BA | NSCLC BA |
|---|---|---|---|
| single selector, clean | 0.873 ± 0.026 (n=3) | 0.907 ± 0.023 (n=3) | 0.909 ± 0.008 (n=3) |
| single selector, attacked | -0.241 ± 0.011 (n=3) | 0.022 ± 0.002 (n=3) | 0.003 ± 0.001 (n=3) |
| single selector, weak scorer | 0.867 ± 0.030 (n=3) | 0.899 ± 0.013 (n=3) | 0.901 ± 0.003 (n=3) |
| advocates, clean | 0.823 ± 0.015 (n=3) | 0.920 ± 0.010 (n=3) | 0.874 ± 0.007 (n=3) |
| advocates, per-tile attack | 0.685 ± 0.017 (n=3) | 0.670 ± 0.001 (n=3) | 0.530 ± 0.023 (n=3) |
| advocates, adaptive attack | 0.589 ± 0.029 (n=3) | 0.433 ± 0.004 (n=3) | 0.153 ± 0.004 (n=3) |
| hybrid (4 tiles), clean | 0.854 ± 0.005 (n=3) | 0.933 ± 0.016 (n=3) | 0.898 ± 0.004 (n=3) |
| hybrid (4 tiles), adaptive attack | 0.658 ± 0.011 (n=3) | 0.661 ± 0.033 (n=3) | 0.618 ± 0.009 (n=3) |
| hybrid (K tiles), clean | 0.879 ± 0.014 (n=3) | 0.924 ± 0.013 (n=3) | 0.905 ± 0.001 (n=3) |
| hybrid (K tiles), per-tile attack | 0.730 ± 0.026 (n=3) | 0.829 ± 0.010 (n=3) | 0.783 ± 0.010 (n=3) |
| hybrid (K tiles), adaptive attack | 0.705 ± 0.025 (n=3) | 0.713 ± 0.031 (n=3) | 0.665 ± 0.022 (n=3) |
| independent scorers, clean | 0.840 ± 0.007 (n=3) | 0.915 ± 0.020 (n=3) | 0.872 ± 0.007 (n=3) |
| independent scorers, attacked | 0.677 ± 0.023 (n=3) | 0.675 ± 0.036 (n=3) | 0.505 ± 0.058 (n=3) |
| advocates, weak scorer | 0.788 ± 0.019 (n=3) | 0.916 ± 0.004 (n=3) | 0.879 ± 0.003 (n=3) |
| advocates, weak scorer, attacked | 0.511 ± 0.022 (n=3) | 0.661 ± 0.034 (n=3) | 0.509 ± 0.033 (n=3) |

The attacker may now also take the TRUE class's advocate, as well as the wrong-class advocates and the neutral slot. The judge is trained against all of these (ATTY=1).
- **Pure advocates break, as in the VLM runs.** Adaptive attack: C17 .59, RCC .43, NSCLC .15 (previously .65 / .70 / .67, when only wrong-class slots could be attacked).
- **Hybrid (K tiles) holds:** adaptive worst case .705 / .713 / .665 vs ≤.02 for the single selector. Clean .879 / .924 / .905 equals or beats the single selector (.873 / .907 / .909).
- Independent scorers alone don't help against this attacker (.68 / .68 / .51). The neutral slot is what provides the redundancy.
