# Zoom-crop findings on V*-Bench (191 questions; 2026-10-24, from cached Narval runs)

Matched comparison: the same grid tiles (ground-truth crop = the tile with max GT-box overlap, 2 random tiles, and 4 adversarial tiles = top wrong-option margin under Qwen2.5-VL, black-box for the other models) are shown to each model next to a 768² overview.

| model | overview only | + GT crop | + random crop | GT − random | 4 adv crops flip a correct answer |
|---|---|---|---|---|---|
| Qwen2.5-VL-7B (letter scoring) | .665 | .838 | .644 | **+.194** | .457 |
| Qwen3-VL-8B (letter scoring) | .691 | .838 | .644 | **+.194** | .447 |
| DeepEyes-7B, full-res overview (agent rollout; every tool call returns the substituted crop) | native .832 | .838 | .840 | **−.003** | .031 |
| DeepEyes-7B, 768² overview | native .681 | .707 | .702 | **+.005** | .031 |

- DeepEyes calls the zoom tool on 100% of questions, but what the tool returns does not matter. Even with a low-resolution overview, where the base model gains +.19 from the right crop, DeepEyes gains nothing. Its native .681 ≈ the base model with no zoom (.665).
- **Confabulated verification:** with a random crop that does not contain the object, 86% of DeepEyes answers still say the zoom confirms the attribute ("The dustpan is indeed blue, as shown in the zoomed-in image"). With the GT crop the rate is 81%, so the claim is the same either way.
- Base models are the opposite case: the zoom is load-bearing (+.19) and therefore hijackable (45% flips from 4 transfer tiles; 64–70% with white-box tile search, earlier runs).

Dilemma: a zoom is either load-bearing and attackable (base VLM + external selector) or decorative and safe (RL agent). Per-hypothesis advocates (vlm_debate2 `nway`: .806/.843 on V*, .705 HR-Bench; compromised single selector .33–.39) are one way to get load-bearing but robust zooms.

Caveats/todo: one RL agent so far; adversarial tiles for DeepEyes are transfer, not white-box (but GT and random also have no effect, so this is about reliance, not attack strength); letter-scoring (Qwen) vs generation (DeepEyes) protocols differ; lit check for prior "thinking-with-images faithfulness" work not done yet.

Files: deepeyes_v3_{full,lowres}_partial.jsonl (vlm/de_attack.py), vstar_partial.jsonl / vstar_q3_partial.jsonl (vlm/vstar_sel.py), *_deb2.jsonl (vlm/vlm_debate2.py).

## Update 2026-10-24 night: three RL agents (crop-substitution audit, vlm/agent_subst.py, 768² overview, V* 191 q)
| agent | native | +GT crop | +random crop | GT − random | blank crop | adv flips a correct answer | "confirms" on irrelevant crop | no answer on random crop |
|---|---|---|---|---|---|---|---|---|
| DeepEyes-7B (v3, de_attack.py) | .681 | .707 | .702 | +.005 | (v5 running) | .031 | 86% | 0% |
| PixelReasoner-RL-7B | .712 | .733 | .657 | +.076 | .665 | .140 | 12% | 0.5% |
| Mini-o3-7B (77/191 so far, 6 turns + forced answer) | .857 | .779 | .097 | +.68* | .026 | .879* | 2% | 86% |

\* Mini-o3 rarely answers under a wrong crop: it keeps re-zooming (5.8 calls on average) and ignores the forced-answer message. Its accuracy among answered questions is .90 (GT), .64 (random), .40 (blank). It treats the crop as evidence and searches when the crop is wrong, which is the behavior we want. The "flip" number is mostly abstention, not hijacking.

So the agents span a spectrum. DeepEyes is decorative (reliance ≈ 0, confabulated confirmation). PixelReasoner is weakly load-bearing (+.08). Mini-o3 is load-bearing but notices bad crops. Base Qwen is load-bearing and hijackable (+.19, 45% flips).
Note: the deepeyes_low_v4 array only ran the GT condition (sbatch split CONDS on its commas); it is rerun as deepeyes_low_v5. PixelReasoner with full-res overview is also running.

### Final V* table (supersedes the partial table above; vlm/agent_analyze.py)
| agent (overview) | native | GT crop | random crop | blank crop | GT − random | any of 4 adv crops flips a correct answer | no answer, random crop | "confirms" on irrelevant / blank crop |
|---|---|---|---|---|---|---|---|---|
| DeepEyes-7B (768²) | .681 | .717 | .694 | .691 | +.024 | .015 | 0% | 88% / 90% |
| PixelReasoner-RL-7B (768²) | .712 | .733 | .657 | .665 | +.076 | .140 | 0.5% | 12% / 18% |
| PixelReasoner-RL-7B (full res) | .827 | .838 | .804 | .812 | +.034 | .051 | 0% | 11% / 13% |
| Mini-o3-7B (768²) | .848 | .743 | .228 | .178 | +.52 | .691 | 70% | 2% / 2% |
| Qwen2.5-VL-7B, single crop (letter scoring) | .665 (no crop) | .838 | .644 | – | +.194 | .457 | – | – |

- DeepEyes answers the same with a blank grey crop (.691) as with the right one (.717), and 90% of its blank-crop answers claim the zoom confirms the answer.
- When Mini-o3 does answer under a wrong crop, it is right about as often as with the GT crop (.75 random / .79 blank vs .84 GT). It doesn't take the bait; it refuses to commit. That is a safe failure mode, but it costs coverage.
- HR-Bench-4K audit (rand/blank only, no GT boxes) submitted: jobs 7005600-2.

### HR-Bench-4K (200 q, 768² overview; random/blank crops only, since HR-Bench has no GT boxes) and paired bootstrap, native − substituted
| | V* native − blank [95% CI] | HR native − blank [95% CI] | HR native / random / blank | HR "confirms" on random / blank crop | HR no answer, random |
|---|---|---|---|---|---|
| DeepEyes | −.010 [−.037, +.010] | **+.065 [+.030, +.105]** | .695 / .630 / .630 | 69% / 75% | 1.5% |
| PixelReasoner | +.047 [+.010, +.084] | +.045 [+.010, +.080] | .655 / .610 / .610 | 15% / 13% | 1.5% |
| Mini-o3 | +.670 [+.597, +.733] | +.520 [+.455, +.595] | .700 / .230 / .180 | 2% / 1.5% | 68% |

To be careful: on HR-Bench, DeepEyes' own crops are worth +.065, so "decorative" holds on V* but only partly on HR-Bench. The claims that hold on both benchmarks:
1. Crop reliance of DeepEyes and PixelReasoner is small (≤ .075) compared with Mini-o3 (.5–.67).
2. DeepEyes' verification language is confabulated: 75–90% of its answers given a blank grey crop say the zoom confirms the answer.
Mini-o3 HR complete (200 q).

## Attacks on the advocate presenter (vlm/vlm_debate3.py; one compromised selection component, label-aware attacker; 95% bootstrap CI)
| presenter (judge sees overview + crops) | V* Qwen2.5-VL | V* Qwen3-VL | HR-4K Qwen2.5-VL |
|---|---|---|---|
| single selector, honest (coopn) | .812 | .911 | .690 |
| single selector, compromised (attn) | .366 [.30,.45] | .335 [.27,.40] | .390 [.33,.46] |
| advocates (nway), honest | .806 | .843 | .705 |
| advocates, TRUE option's advocate compromised (nway_cy) | .408 | .408 | .460 |
| advocates, true advocate silenced (nway_drop) | .419 | .387 | .475 |
| **hybrid** (advocates + neutral slot), honest | **.822** | **.880** | **.720** |
| hybrid, true advocate compromised (hyb_cy) | .733 [.67,.79] | .859 [.81,.91] | .675 [.61,.74] |
| hybrid, neutral slot compromised (hyb_cn) | .764 [.70,.82] | .801 [.75,.85] | .675 [.61,.74] |

- Being honest: **pure advocates are NOT robust** when the attacker controls the one advocate that matters (true option). Accuracy drops to .39–.48, about as bad as a compromised single selector, because the judge then sees only the wrong options' evidence.
- **The hybrid presenter is robust to any single compromised component.**
  - Worst case is .73 / .80 / .68, against .37 / .34 / .39 for a compromised single selector.
  - Clean accuracy is equal or better than the single selector (.822 vs .812, .720 vs .690), except Qwen3 (.880 vs .911).
- This matches pathology, where the hybrid closes the clean gap and holds .71–.78 under adaptive attack. **The method of the paper should be the hybrid (redundant, independent evidence channels), not advocates alone.**

## White-box crop attack on Mini-o3 (V*, 191/191; agent_subst.py CONDS=wb0..3)
The attacker scores every tile with Mini-o3 itself: its option log-probs when the first zoom returns that tile and it is made to answer. Every zoom call then returns one of the top-4 tiles.

| crop returned at every zoom | correct | committed wrong | abstains |
|---|---|---|---|
| GT | .750 | .131 | .119 |
| random | .220 | .077 | .702 |
| transfer adversarial (Qwen-chosen, adv0) | .256 | .179 | .565 |
| white-box adversarial (wb0) | .244 | .196 | .560 |

- Among natively correct questions, 1 white-box tile turns 17% into a committed wrong answer, and any of the top 4 tiles does so for 30%. The transfer tile does so for 16% (final, n=162 natively correct).
- By comparison, 4 transfer tiles flip 46% for base Qwen2.5-VL, and white-box search flips 64–70%.
- Mini-o3 is the hardest to hijack because it abstains (56–66%) instead of committing. It is safe but not useful under attack: accuracy drops to about .2 either way.
- The white-box surrogate finds a flipping tile for only 42% of questions. With the right answer present in the overview, Mini-o3 mostly prefers it.
