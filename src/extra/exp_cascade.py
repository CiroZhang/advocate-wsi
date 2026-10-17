"""Compute-matched baselines without students (CAMELYON16): (1) cascade: full-slide R50 ABMIL attention picks the top-B patches,
a UNI judge decides from them (cost = students' cost); (2) UNI ABMIL on a stride-s subgrid (reads 1/s^2 of the slide with UNI),
same judge trained on full slides and one retrained on subgrids. Also the cascade under attack (white-box selector for the wrong label
on the R50 ABMIL). Original docstring follows.
 Same training as exp_cheap.py; at TEST time the R50 students do not encode the whole
slide: they read a stride-s subgrid, then repeatedly read the neighbourhood (Chebyshev radius s) of the top-M patches for each label
(single-patch score under the cheap student model), and argue only over what they read. Reported: BA and fraction of patches read.
Original exp_cheap docstring follows.
 Students read the whole slide with a CHEAP encoder (ResNet50 Barlow-Twins features)
and argue with greedy picks against their own cheap teacher model; only the presented patches are encoded with the EXPENSIVE encoder
(UNI) for the teacher. Total encoder cost per slide: N * cost(R50) + B * cost(UNI), vs full-slide UNI ABMIL N * cost(UNI).
Reported per budget B = 2 Ks (test slides, balanced accuracy):
  full_uni / full_r50       : full-slide ABMIL on each encoder
  random_uni_K              : UNI judge on B random patches (cost B * UNI, no students)
  teach_uni_K               : students and teacher on UNI (paper's protocol)
  teach_r50_K               : students and teacher on R50 (all cheap)
  cheap2uni_K               : R50 students, UNI teacher  <- the compute-saving protocol
  cheap2uni_liar_rollout_K  : same, devil's advocate uses look-ahead rollouts against the cheap student model
  cheap2uni_blank_K         : UNI teacher with zeroed patches (flag-only control)
usage: python exp_cascade.py SEED"""
import sys, json, time, csv, numpy as np, torch
sys.path.insert(0, "/scratch/ciro/cvpr_lab/studentB_debate")
from common import *

seed = int(sys.argv[1]); KS = [1, 2, 4, 8]; R = int(os.environ.get('ROUNDS', 4)); EP_FULL, EP_SUB, EP_FT = 20, 40, 20; BUD = 400_000
D = "/scratch/ciro/cvpr_lab/data/camelyon16/torchmil"; OUT = "/scratch/ciro/cvpr_lab/studentB_debate/results"
rng = np.random.default_rng(seed); torch.manual_seed(seed)
rows = list(csv.DictReader(open(os.path.join(D, 'splits.csv'))))
names = [r['bag_name'] for r in rows if all(os.path.exists(os.path.join(D, f, r['bag_name'] + '.npy')) for f in ('features', 'features_r50bt'))]
split = {r['bag_name']: r['split'] for r in rows}; t0 = time.time()
XE = [torch.from_numpy(np.load(os.path.join(D, 'features', n + '.npy'))).half() for n in names]
XC = [torch.from_numpy(np.load(os.path.join(D, 'features_r50bt', n + '.npy'))).half() for n in names]
GR = [torch.from_numpy(np.load(os.path.join(D, 'coords', n + '.npy')) // 512).long() for n in names]
y = np.array([int(np.load(os.path.join(D, 'labels', n + '.npy')).reshape(-1)[0]) for n in names])
assert all(len(a) == len(b) for a, b in zip(XE, XC)), 'UNI and R50 bags not aligned'
tr = np.array([i for i, n in enumerate(names) if split[n] == 'train']); te = np.array([i for i, n in enumerate(names) if split[n] == 'test'])
print(f'loaded {len(names)} in {time.time() - t0:.0f}s; train {len(tr)} test {len(te)}; dims UNI {XE[0].shape[1]} R50 {XC[0].shape[1]}', flush=True)
L = {}
def fl(x, f): return flag(x.float(), f)
def rand_sub(xs, i, K, f):
    x = xs[i]; p = torch.randperm(len(x))[:K]
    ff = torch.zeros(len(p), 1) if f == 0 else (torch.randint(0, 2, (len(p), 1)) * 2 - 1).float()
    return torch.cat([x[p].float(), ff], 1)

def games(xs, ids, fn):
    """-> transcripts (n, T, d+1) on xs, logits, presented indices (n, T), flags (n, T)"""
    def g(X, M, k):
        st = fn(X, M, y[ids][k]); return st.transcript().cpu(), st.logit(), torch.stack(st.idx, 1), torch.stack(st.flags, 1)
    res = batched(g, [xs[i] for i in ids], bs=32, budget=BUD)
    return (torch.stack([r[0] for r in res]), np.array([r[1].item() for r in res]),
            torch.stack([r[2] for r in res]).cpu(), torch.stack([r[3] for r in res]).cpu())

def to_uni(ids, idx, flags):  # the presented patches, re-encoded with UNI
    return torch.stack([torch.cat([XE[i][idx[r]].float(), flags[r][:, None].float()], 1) for r, i in enumerate(ids)])

def train_debate(xs, Ks):
    jd = train_judge([rand_sub(xs, i, 2 * Ks, 1) for i in tr], y[tr], epochs=EP_SUB, seed=seed); bx, pres = [], []
    for r in range(R):
        T, lg, idx, f = games(xs, tr, lambda X, M, yb: play_debate(jd, X, M, Ks)); bx.append(T); pres.append((idx, f))
        print(f'   round {r} train balacc {balacc((lg > 0).astype(int), y[tr]):.3f}', flush=True)
        jd = train_judge(torch.cat(bx), np.tile(y[tr], len(bx)), epochs=EP_FT, judge=jd, seed=seed + r)
    idx, f = games(xs, tr, lambda X, M, yb: play_debate(jd, X, M, Ks))[2:]; pres.append((idx, f))
    return jd, pres



def grid(i, s): g = GR[i]; return (((g[:, 0] % s) == 0) & ((g[:, 1] % s) == 0)).nonzero().squeeze(1)
fu = train_judge([fl(XE[i], 0) for i in tr], y[tr], epochs=EP_FULL, bs=1, lr=2e-4, seed=seed)
fr = train_judge([fl(XC[i], 0) for i in tr], y[tr], epochs=EP_FULL, bs=1, lr=2e-4, seed=seed)
L['full_uni'] = judge_logits(fu, [fl(XE[i], 0) for i in te], bs=1); L['full_r50'] = judge_logits(fr, [fl(XC[i], 0) for i in te], bs=1)
for s in (2, 3, 4):
    L[f'uni_grid{s}_fulltrained'] = judge_logits(fu, [fl(XE[i][grid(i, s)], 0) for i in te], bs=1)
    fg = train_judge([fl(XE[i][grid(i, s)], 0) for i in tr], y[tr], epochs=EP_FULL, bs=1, lr=2e-4, seed=seed)
    L[f'uni_grid{s}'] = judge_logits(fg, [fl(XE[i][grid(i, s)], 0) for i in te], bs=1)
with torch.no_grad(): att = {i: fr.terms(fl(XC[i].to(DEV), 0))[1].cpu() for i in range(len(names))}
@torch.no_grad()
def attacked_idx(i, K):  # white-box: greedily choose K patches that push the R50 cascade selector's own ABMIL toward the wrong label
    st = play_single(fr, XC[i].float().to(DEV)[None], torch.ones(1, len(XC[i]), dtype=torch.bool, device=DEV), K,
                     torch.tensor([1 - y[i]], device=DEV), torch.zeros(1, device=DEV))
    return torch.stack(st.idx, 1)[0].cpu()
for K in (2, 4, 8, 16):
    sel = lambda i: att[i].topk(min(K, len(XC[i]))).indices
    jc = train_judge([fl(XE[i][sel(i)], 0) for i in tr], y[tr], epochs=EP_SUB, seed=seed)
    L[f'cascade_K{K}'] = judge_logits(jc, [fl(XE[i][sel(i)], 0) for i in te])
    L[f'cascade_attacked_K{K}'] = judge_logits(jc, [fl(XE[i][attacked_idx(i, K)], 0) for i in te])
    print(f'K{K} {time.time() - t0:.0f}s', {m: round(metrics(v, y[te])['balacc'], 3) for m, v in L.items()}, flush=True)
json.dump({'slide': {m: metrics(v, y[te]) for m, v in L.items()}, 'info': {'seed': seed}}, open(os.path.join(OUT, f'cascade_s{seed}.json'), 'w'), indent=1)
print('DONE', flush=True)
