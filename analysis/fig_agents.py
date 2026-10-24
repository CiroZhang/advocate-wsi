"""Decorative vs load-bearing zoom (V*-Bench, crop-substitution audit).
  fig_agents.pdf (full width): (a) accuracy when every zoom returns the GT / a random / a blank crop, per model;
                                 (b) behaviour under an irrelevant crop: abstains (no answer) vs claims the crop confirms the answer.
Reads results/agents/*.jsonl via vlm/agent_analyze.py; base-model numbers from results/vlm/FINDINGS.md (letter scoring, single crop).
usage: python analysis/fig_agents.py"""
import os, sys, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
B = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'); O = f'{B}/paper/figs'; os.makedirs(O, exist_ok=True)
sys.path.insert(0, f'{B}/vlm'); from agent_analyze import summarize
plt.rcParams.update({'font.size': 7, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
A = f'{B}/results/agents'
M = [('DeepEyes', summarize(f'{A}/deepeyes_low_merged.jsonl')), ('PixelReasoner', summarize(f'{A}/pixelreasoner_low_s*of12.jsonl')),
     ('Mini-o3', summarize(f'{A}/minio3_low_s*of12.jsonl'))]
BASE = dict(gt=.838, rand=.644, blank=float('nan'))  # Qwen2.5-VL-7B, overview + one crop (FINDINGS.md); no blank-crop run
fig, (a, b) = plt.subplots(1, 2, figsize=(6.0, 1.9), gridspec_kw=dict(width_ratios=[1.35, 1]))
C = {'gt': '#4C72B0', 'rand': '#DD8452', 'blank': '#999999'}; w = .26
names = ['Qwen2.5-VL\n(no RL)'] + [n for n, _ in M]
for k, (key, lab) in enumerate([('gt', 'GT crop'), ('rand', 'random crop'), ('blank', 'blank crop')]):
    vals = [BASE[key]] + [s[key] for _, s in M]
    a.bar([i + (k - 1) * w for i in range(len(vals))], vals, w, color=C[key], label=lab)
a.set_xticks(range(len(names))); a.set_xticklabels(names, fontsize=6); a.set_ylim(0, 1); a.set_ylabel('V* accuracy')
a.legend(fontsize=5.5, frameon=False, ncol=3, loc='upper center', bbox_to_anchor=(.5, 1.22)); a.set_title('(a)', fontsize=7, loc='left')
x = range(len(M))
b.bar([i - .18 for i in x], [s['noans_rand0'] for _, s in M], .36, color='#55A868', label='abstains')
b.bar([i + .18 for i in x], [s['confirm_on_irrelevant_crop'] for _, s in M], .36, color='#C44E52', label='"zoom confirms"')
b.set_xticks(list(x)); b.set_xticklabels([n for n, _ in M], fontsize=6); b.set_ylim(0, 1); b.set_ylabel('share, irrelevant crop')
b.legend(fontsize=5.5, frameon=False, loc='upper center', bbox_to_anchor=(.5, 1.22), ncol=2); b.set_title('(b)', fontsize=7, loc='left')
plt.tight_layout(); plt.savefig(f'{O}/fig_agents.pdf', bbox_inches='tight'); plt.savefig(f'{O}/fig_agents.png', dpi=200, bbox_inches='tight')
