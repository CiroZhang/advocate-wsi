# WSI_debate: Adversarial evidence games make gigapixel answers image-necessary

**Target:** ICML 2027 (~Jan 22). Fallbacks: ICCV 2027 (Mar 8), NeurIPS 2027 (May). Direction 3: small specialised models that fight, done so that the fight is about *pixels*, not rhetoric.

## The problem in one line
WSI-QA models answer from question priors (CleanSlide 2609.32185, PathoArgus 2608.17607, PathBind 2607.21065). Multi-agent "debate" among LLMs makes this
worse, not better: debaters share the same prior, drift toward the majority, and the judge can be talked into anything ("Should we be going MAD?" 2311.17371;
"Can LLM agents really debate?" 2511.07784).

## The idea
Turn diagnosis into a **two-player evidence game** on the slide:
- Two small debaters are assigned opposite answers (e.g. ER+ vs ER−). They take turns *revealing patches* (coordinates), K in total (K = 2–8 out of ~10^4).
- The **judge never sees the question text, the slide, or any argument**. It sees only the revealed patches and who revealed each one.
- The judge is trained on debate transcripts, and the debaters are trained (RL or best-response search) to win in front of it.
- At test time the answer is whichever side wins.

Why this is the right structure:
1. **Image necessity by construction.** The judge's only input is pixels. With blank patches it is at chance (we test this), so no answer can come from the prior.
2. **Soundness against a lying expert.** The dishonest side must find real patches that support a false label, and the honest side gets to answer.
   For sparse evidence (a micrometastasis, a mitotic hotspot) this is a classic verifier-prover asymmetry, and it can be stated as a theorem
   (Merlin-Arthur classifiers: Wäldchen et al. 2206.00759; Irving et al. 1805.00899).
3. **The certificate is the answer.** The K revealed patches *are* the explanation, chosen adversarially. Attention maps, by contrast, are not faithful evidence.
4. **Small models suffice.** Debaters and judge are tiny MIL heads on frozen foundation features, so the compute is cheap and Killarney/A100 is enough.

## Why it is novel (checked 2026-10-06)
| Closest | What it does | Difference |
|---|---|---|
| Merlin-Arthur (2206.00759, 2306.04505, 2507.07532) | Prover + adversarial prover choose feature masks; the verifier classifies from the mask | Small images/concepts; one-sided masks. No two-sided debate over options, no gigapixel MIL, no medical QA, no prior-reliance analysis |
| Irving et al. 2018 (MNIST pixel debate) and follow-ups (2505.14627, LessWrong quantilized debate) | Debate with pixel reveals; blind judges | Toy images. Never on WSI or medical data, and never with trained patch-selecting debaters among ~10^4 candidates |
| WSI-Agents 2507.14680, PathoSage 2606.07549, UCAgents 2512.02485, MMedAgent-RL 2506.00555 | Multi-agent pathology/medical systems that verify claims | Prompted LLM personas that argue in text. The judge reads arguments, so the prior leaks |
| BEACON 2608.05757 / attention-MIL top-K | Choose informative patches | Cooperative selection; not robust to a dishonest selector; no soundness notion |

*No work trains adversarial patch-selecting debaters on WSIs or any medical images.*. (Merlin-Arthur is the conceptual precedent and must be positioned explicitly.)

## Why ICML would accept it
- **A clean, general claim.** "Debate restricted to evidence, judged by a question-blind verifier, yields answers that are image-necessary and sound against a
  dishonest prover." It touches AI-safety debate, interpretability (faithful explanations) and MIL, all ICML core topics, and gigapixel pathology is a
  compelling, realistic setting rather than MNIST.
- **Theory** (planned): for a threshold/sparse-evidence task, the honest debater wins whenever the evidence certificate exists in K patches. We bound the
  dishonest win rate by the judge's error on the worst-case K-subset (the Merlin-Arthur style bound, specialised to MIL), and give a sample-complexity
  statement for the judge.
- **Faithfulness for free**, measured against ground truth: CAMELYON16 lesion masks show whether the revealed patches are tumor.
- **Fits the 2026 conversation**: after CleanSlide/PathoArgus showed models ignore the slide, this is a mechanism that *cannot* ignore it.

## Experiments (minimal paper)
- **Data** (existing benchmarks only):
  - BCNB (ER/PR/HER2/grade; CONCH features already on the PC);
  - TCGA NSCLC LUAD/LUSC and RCC subtypes (SlideChat's released CONCH features);
  - CAMELYON16 (lesion masks).
- **Baselines** (same judge architecture):
  - full-slide ABMIL;
  - random-K;
  - attention-top-K;
  - single prover ("consultancy");
  - majority prior.
- **Metrics:**
  - accuracy / AUC vs K;
  - honest-win rate;
  - dishonest-debater win rate after training a stronger dishonest debater (soundness);
  - blank-patch control;
  - % of revealed patches inside lesions;
  - a judge-hacking check (retrain the judge after the debaters converge).
- **Success:**
  - with K ≤ 8 revealed patches, within 2–3 AUC points of full-slide MIL;
  - revealed patches are in lesions more often than attention-top-K;
  - a trained liar wins less than 20% of the time.

## Pilot (running now, real BCNB data; `pilot/debate_pilot.py`)
Best-response debaters (search, no RL yet) plus an iteratively retrained judge, 5-fold CV, K ∈ {2,4,8}, tasks ER/PR/Grade3/HER2.
**Results (`pilot/RESULTS.md`):**
- **Accuracy.** K=4–8 debated patches come within 1–4 points of full-slide MIL, with ~160 patches available; with blank patches the judge is at chance.
- **Soundness, the headline.** Judges that decide from *selected* patches collapse under a dishonest selector: attention-top-K judge 3–5%,
  lying consultant 5–11% balanced accuracy. Debate with a liar on one side holds at **.70–.76**, the same as honest cooperation, on ER, PR and grade for
  K = 4 and 8.

## Risks and mitigations
| Risk | Mitigation |
|---|---|
| Both sides show tumor and the debate is uninformative for non-sparse labels (e.g. grade) | Report it honestly. Debate is designed for "exists-evidence" questions; graded attributes use counting debates (who reveals more mitoses) |
| The judge gets hacked by the debaters | Retrain the judge from scratch on final transcripts, and use held-out judges |
| Close to Merlin-Arthur | Make the two-sided game, the MIL scale, the image-necessity guarantee and the medical faithfulness evaluation the explicit contributions |
