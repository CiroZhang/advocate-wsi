# EXT experiments (seed 0, 5 folds; Narval jobs 6944241-3, 2026-10-24)
| | C17 κ | RCC BA | NSCLC BA |
|---|---|---|---|
| full-slide | .892 | .943 | .922 |
| single selector (ADMIL-style), clean / attacked | .892 / −.230 | .933 / .020 | .913 / .004 |
| single selector, weak scorer (10% train slides) | .875 | .914 | .900 |
| advocates, clean | .810 | .929 | .875 |
| advocates, per-tile white-box attack | .713 | .865 | .836 |
| advocates, adaptive (set-aware greedy) attack | .641 | .690 | .650 |
| hybrid (advocates + neutral presenter, K tiles), clean | .873 | .931 | .903 |
| hybrid, per-tile / adaptive attack | .770 / .739 | .873 / .757 | .829 / .719 |
| independent per-advocate scorers, clean / attacked | .844 / .714 | .923 / .861 | .869 / .833 |
| advocates, weak scorer, clean / attacked | .759 / .586 | .925 / .885 | .873 / .841 |

- Hybrid closes most of the clean gap (C17 −8→−2 κ, NSCLC −4→−1) while staying far above the attacked single selector.
- The adaptive attacker is much stronger than the per-tile one (−15 to −20 points on TCGA). Report adaptive numbers as the headline robustness.
- Independent scorers: no clean cost, so the shared-scorer single point of failure can be removed.
- Natural failure (weak scorer) does NOT break the single selector: the fragility is adversarial-only.
