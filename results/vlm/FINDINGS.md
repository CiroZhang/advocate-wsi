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
