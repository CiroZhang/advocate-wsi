"""Table: published MIL baselines (torchmil reference implementations, mil_baselines.py, 3 seeds) vs. ours (6 seeds), CAMELYON16 test.
AUC + BA (threshold logit>0 for everything) + A100 encoder s/slide (enc_timing.json) + aggregator ms (A100, per slide, median).
Prints LaTeX rows; also returns BASE dict used by fig_cheap.py."""
import csv, glob, json, os, numpy as np
from sklearn.metrics import roc_auc_score
B = 'narval_results/baselines'; D = 'narval_results/cheap'
NAMES = {'abmil': 'ABMIL~\\cite{ilse2018abmil}', 'clam_sb': 'CLAM-SB~\\cite{lu2021data}', 'dsmil': 'DSMIL~\\cite{li2021dual}',
         'transmil': 'TransMIL~\\cite{shao2021transmil}', 'dtfdmil': 'DTFD-MIL~\\cite{zhang2022dtfd}'}
T = json.load(open(f'{D}/enc_timing.json')); cu, cr = 1 / T['UNI_ViT-L16@224']['patches_per_s'], 1 / T['ResNet50@224']['patches_per_s']
N = json.load(open(f'{D}/cheap_s0.json'))['info']['mean_patches_test']
ENC = {'uni': ('UNI', N * cu), 'r50bt': ('R50', N * cr)}

def load_base():
    out = {}
    for enc in ENC:
        for m in NAMES:
            S = [json.load(open(f)) for f in sorted(glob.glob(f'{B}/c16_{enc}/{m}/summary_s*.json'))]
            if S: out[(enc, m)] = {k: np.array([s['testA'][k] for s in S]) for k in ('auc', 'balacc', 'ms_aggregator_median')} | {'balacc_vt': np.array([s.get('testA_valthr', s['testA'])['balacc'] for s in S])}
    return out
BASE = load_base()

def ours(keys):
    R = {k: [] for k in keys}
    for s in range(6):
        d = np.load(f'{D}/cheap_logits_s{s}.npz'); y = d['y']
        for k in keys:
            p = d[k] > 0; R[k].append((roc_auc_score(y, d[k]), (p[y == 1].mean() + (~p[y == 0]).mean()) / 2))
    return {k: np.array(v) for k, v in R.items()}

if __name__ == '__main__':
    f = lambda v: f'{v.mean():.3f}{{\\tiny$\\pm${v.std():.3f}}}'.replace('0.', '.')
    print('method & encoder & AUC & BA & BA (val thr.) & enc. s/slide & agg. ms \\\\')
    for enc in ENC:
        for m in NAMES:
            if (enc, m) in BASE:
                b = BASE[(enc, m)]
                print(f'{NAMES[m]} & {ENC[enc][0]} & {f(b["auc"])} & {f(b["balacc"])} & {f(b["balacc_vt"])} & {ENC[enc][1]:.1f} & {np.median(b["ms_aggregator_median"]):.1f} \\\\')
        print('\\midrule')
    O = ours(['full_uni', 'cheap2uni_K8', 'cheap2uni_liar_rollout_K8', 'cheap2uni_K2'])
    for k, lab, c in [('full_uni', 'our judge, full slide & UNI', N * cu), ('cheap2uni_K2', 'teaching round $B{=}2$ & R50$\\to$UNI', N * cr + 2 * cu),
                      ('cheap2uni_K8', 'teaching round $B{=}8$ & R50$\\to$UNI', N * cr + 8 * cu), ('cheap2uni_liar_rollout_K8', '\\quad + look-ahead DA & R50$\\to$UNI', N * cr + 8 * cu)]:
        print(f'{lab} & {f(O[k][:, 0])} & {f(O[k][:, 1])} & -- & {c:.1f} & -- \\\\')
