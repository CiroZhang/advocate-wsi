"""Usefulness (wrong->correct with the GT crop) vs attackability (correct->wrong with any of the same 4 Qwen2.5-surrogate tiles), V*."""
import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
D = "/Users/cirozhang/Desktop/idea/WSI_debate/narval_results/vlm_raw"; rd = lambda f: [json.loads(l) for l in open(f"{D}/{f}")]
am = lambda v: int(np.argmax(v)); q25 = {r['qid']: r for r in rd("vstar_partial.jsonl")}
def adv4(qid):
    r = q25[qid]; S = np.array(r['S_glob']); y = r['y']; m = np.max(np.delete(S, y, 1), 1) - S[:, y]; return list(np.argsort(-m)[:4])
pts = {}
for nm, f in (("Qwen2.5-VL-7B", "vstar_partial.jsonl"), ("Qwen3-VL-8B", "vstar_q3_partial.jsonl")):
    rs = rd(f); use, att = [], []
    for r in rs:
        S = np.array(r['S_glob']); y = r['y']; g = am(r['base']['global_only']) == y; pred = S.argmax(1)
        if g: att.append(any(pred[t] != y for t in adv4(r['qid'])))
        else: use.append(pred[int(np.argmax(r['gt_overlap']))] == y)
    pts[nm] = (np.mean(use), np.mean(att))
for nm, f in (("DeepEyes-7B (full-res)", "deepeyes_v3_full_partial.jsonl"), ("DeepEyes-7B (768² overview)", "deepeyes_v3_lowres_partial.jsonl")):
    rs = rd(f); nat = [r['native']['pred'] == r['y'] for r in rs]
    use = [r['subst']['gt']['pred'] == r['y'] for r, n in zip(rs, nat) if not n]
    att = [any(r['subst'][f'adv{i}']['pred'] != r['y'] for i in range(4)) for r, n in zip(rs, nat) if n]
    pts[nm] = (np.mean(use), np.mean(att))
fig, ax = plt.subplots(figsize=(4.6, 3.4))
OFF = {"Qwen2.5-VL-7B": (-20, 10), "Qwen3-VL-8B": (-95, -14), "DeepEyes-7B (full-res)": (4, 14), "DeepEyes-7B (768² overview)": (8, -4)}
for (nm, (u, a)), c in zip(pts.items(), ["C0", "C1", "C3", "C3"]):
    ax.scatter(u, a, s=70, color=c, marker="o" if "full" not in nm else "s", zorder=3); ax.annotate(nm, (u, a), textcoords="offset points", xytext=OFF[nm], fontsize=7.5)
ax.plot([0, 1], [0, 1], ls=":", color="0.6", lw=1)
ax.set_xlabel("usefulness: wrong→correct with GT crop"); ax.set_ylabel("attackability: correct→wrong\n(any of 4 adversarial crops)")
ax.set_xlim(-0.02, 1); ax.set_ylim(-0.02, 0.6); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig("/Users/cirozhang/Desktop/idea/WSI_debate/paper_v1/figs/fig_useful_attack.pdf"); plt.savefig("/Users/cirozhang/Desktop/idea/WSI_debate/figs/fig_useful_attack.png", dpi=180)
print({k: tuple(round(x, 3) for x in v) for k, v in pts.items()})
