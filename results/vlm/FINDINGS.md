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
