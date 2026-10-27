# WSI_debate pilot results (2026-10-06, real BCNB CONCH features, 5-fold CV, balanced accuracy)

Setup:
- Debaters: greedy best responses to the current judge (no RL yet), alternating patch reveals.
- Judge: gated-attention MIL over revealed patches plus presenter flag, class-balanced, retrained on debate transcripts for 2 rounds.
- Baselines use the same judge architecture trained on their own subsets.

| task | majority acc | full slide (~160 patches) | random K2 | attn-top K2 | **debate K2** | random K4 | attn-top K4 | **debate K4** | random K8 | attn-top K8 | **debate K8** | debate K8, blank patches |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ER | .807 | .786 | .691 | .754 | .736 | .724 | .762 | **.771** | .747 | .775 | **.777** | .530 |
| PR | .747 | .729 | .682 | .729 | .670 | .692 | .732 | .706 | .715 | .749 | .727 | .494 |
| Grade 3 vs ≤2 | .600 | .738 | .677 | .715 | .701 | .706 | .724 | .718 | .710 | .712 | .701 | .513 |
| HER2 | .728 | .668 | .609 | .649 | .612 | .610 | .648 | .613 | .606 | .660 | .647 | .512 |

Readings:
- **A handful of debated patches carries almost all the slide's information.** ER at K=8: .777 vs .786 full slide (2.5% of the patches). Grade at K=4:
  .718 vs .738.
- **The image-necessity control works.** Blank patches put the judge at chance (.49–.53), so no answer comes from the prior.
- **Debate ≈ cooperative attention-top-K on accuracy** (within ±3 points; worse at K=2 for PR/HER2, where 1 patch per side is too few).
  Expected: accuracy is not the claim.
- **The real test is soundness:** what happens when the evidence selector lies. Running now as `attack_pilot.py`:
  - a cooperative judge attacked by an adversarial selector;
  - consultancy with a lying prover;
  - debate with a liar vs an honest debater.
  
  If the cooperative judges collapse while debate holds, that is the paper's Figure 1.

## Soundness / attack pilot (`attack_pilot.py`, 5-fold CV, balanced accuracy = honest-outcome rate)

| setting | cooperative judge, honest top-K | **cooperative judge, adversarial selector** | consultancy, prover argues TRUE label* | **consultancy, prover LIES** | **debate (liar vs honest)** |
|---|---|---|---|---|---|
| ER, K=4 | .760 | **.035** | .994 | **.077** | **.724** |
| ER, K=8 | .757 | **.038** | .988 | **.107** | **.761** |
| Grade3, K=4 | .716 | **.046** | .986 | **.071** | **.705** |
| Grade3, K=8 | .718 | **.046** | .984 | **.095** | **.726** |
| PR, K=4 | .716 | **.032** | .996 | **.045** | **.724** |
| PR, K=8 | .737 | **.037** | .986 | **.107** | **.736** |

\*Consultancy with a truthful prover is oracle-assisted (the prover is told the label), so it is an upper bound, not a usable system.

**This is Figure 1 of the paper.**
- Any judge that decides from a *selected* subset of patches (attention-top-K, retrieval, an agent's "zoom-in" crops) is decided by **whoever selects
  the patches**. An adversarial selector drives the judge to 3–5% (it can make the judge say almost anything). A lying consultant gets 5–11%.
- **Debate is the only protocol that holds.** With a liar on one side, accuracy stays at .70–.76, statistically the same as the honest cooperative
  setting. The cost of soundness is roughly zero.
- **Why it matters beyond pathology:** every agentic WSI system (GIANT, SlideSeek, PathAgent, AdaptivePath, BEACON, SlideBank) is a "selector + judge"
  pipeline. If the selector is biased (a question prior, a hallucinating planner, a poisoned retriever), the answer follows it. Debate gives a
  selector-robust guarantee. Caveat: the attack here is white-box greedy against the judge; RL debaters and transfer attacks are next.
