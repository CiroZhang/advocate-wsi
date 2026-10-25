# CVPR 2027 plan (deadline Nov 16, 2026) — working notes, 2026-10-24

## Candidate story (strongest result so far)
**"Whoever zooms decides — unless nobody is looking."** In high-resolution and gigapixel vision, a selection step (a zoom tool, a crop
selector, a cheap tile selector) decides what the expensive model sees. We find two failure modes:
1. **Load-bearing selection is hijackable.** For base VLMs (Qwen2.5-VL, Qwen3-VL) the right crop is worth +.19 accuracy on V*, and the
   same crop channel lets 4 transfer-attack crops flip 45% of correct answers (64–70% with tile search). In WSI pipelines a compromised
   selector drives accuracy to chance on 4 datasets.
2. **RL-trained "thinking with images" agents don't use what they zoom on.** DeepEyes-7B calls its zoom tool on 100% of V* questions,
   but replacing every returned crop by the GT region, a random region, or an adversarial region changes 1–6% of answers. GT − random = 0
   even with a low-res overview, where the base model gains +.19. With a crop that does not show the object, 86% of its answers still
   say the zoom "confirms" the answer. Its zoom is decorative and its verification is confabulated.
3. **Fix:** make evidence selection load-bearing but not single-point: one advocate per hypothesis + a judge trained against an
   adversarial advocate. (Pathology: 4 datasets, 4–6× cheaper than full-slide; VLM: per-option crops, .81/.84 V*, .71 HR-Bench; a
   compromised single selector is .33–.39.)

## Must-have experiments (in order)
- [done] Mini-o3-7B crop substitution (768²): load-bearing (+.52 GT−random), abstains 70% under a wrong crop, 2% confabulation.
- [done] PixelReasoner-RL-7B (768² and full res): weak reliance (+.03 to +.08), 11–18% confabulation.
- [done/partial] HR-Bench-4K agent audit: DeepEyes' crops are worth +.065 there (not zero), but it still claims confirmation on 75% of blank crops. Mini-o3 HR running.
- [todo] "decision before looking": % of rollouts whose first turn already names the final answer, vs base models.
- [done, 3 seeds] pathology EXT jobs (adaptive attack, hybrid presenter, independent scorers, weak scorer).
- [todo] VLM advocates with a *cheap* selector (SigLIP per-option crop scoring) so both domains use the same cheap→expensive pipeline.
- [todo] lit check: faithfulness of thinking-with-images / visual tool use (2025–26) — not done (searches were not approved tonight).

## Risks
- Novelty of the "decorative zoom" finding (someone may have shown it in 2026). Need the lit check first thing.
- One agent so far; need ≥3 agents and 2 benchmarks for a CVPR-level claim.
- Protocol differences (letter scoring for base models vs generation for agents); report agent results with the agent's own protocol only.

## Status 2026-10-25 early morning (honest read)
- The strongest and most robust claim is **confabulated visual verification**: DeepEyes says the zoom confirms its answer on 75–90% of blank or irrelevant crops (V* and HR). The "zero reliance" claim holds on V* only (HR: +.065).
- The agents span a spectrum (decorative → weak → load-bearing-and-abstaining). The paper can frame it as an evaluation protocol (crop substitution) plus this spectrum, with selection robustness (advocates/hybrid) as the fix on the load-bearing side.
- Pathology hybrid: matches the single selector clean over 3 seeds; .71–.78 under adaptive attack vs ≤.02.

## Revised story after the lit check and the presenter attacks (2026-10-25)
**Title idea:** "Whoever picks the evidence decides: selection attacks on high-resolution vision and a redundant presenter."
1. **Threat:** an attacker controls one evidence-selection component (crop tool, tile selector, cheap scorer) and shows real, unmodified regions.
   - Base VLMs: .81 → .34–.39 on V* and HR-4K.
   - WSI MIL: chance on 4 datasets.
   - RL agents (prior work: Wang 2026, Liu 2025): agents that ignore their zoom are immune but get nothing from it (DeepEyes). Agents that use it (Mini-o3) are hijacked on 29% of correct answers, or abstain, so accuracy under attack is about .2.
2. **Method:** hybrid presenter = per-hypothesis advocates + a neutral slot, with the judge trained or prompted to weigh them.
   - Pure advocates fail when the true advocate is compromised (.41); the hybrid tolerates any single compromised component.
   - VLM: .73–.86 worst-case vs .34–.39; clean equal to the single selector.
   - WSI: .71–.78 adaptive worst-case vs ≤.02; clean equal over 3 seeds.
3. **Evidence that the fragility is adversarial-only:** a weak scorer costs the single selector ≤.01.

### Remaining gaps (in priority order)
- An adaptive attacker against the VLM hybrid that controls one component and chooses the tile knowing the hybrid's judge. Currently the attacker tile is chosen without seeing the judge's multi-crop input.
- WSI: compromise the neutral slot specifically (hybrid with a compromised neutral selector). Check what src/ext.py's adaptive attack already covers.
- Hybrid wrapper around an agent (optional).
- Figures: one teaser (threat → three domains → hybrid), a robustness bar chart across domains, and a clean-vs-worst-case scatter.
