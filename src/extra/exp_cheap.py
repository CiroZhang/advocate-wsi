"""Cheap students, expensive teacher (CAMELYON16). Students read the whole slide with a CHEAP encoder (ResNet50 Barlow-Twins features)
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
usage: python exp_cheap.py SEED"""
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

for nm, xs in (('uni', XE), ('r50', XC)):
    full = train_judge([fl(xs[i], 0) for i in tr], y[tr], epochs=EP_FULL, bs=1, lr=2e-4, seed=seed)
    L[f'full_{nm}'] = judge_logits(full, [fl(xs[i], 0) for i in te], bs=1)
    print(f'full_{nm}', metrics(L[f'full_{nm}'], y[te]), f'{time.time() - t0:.0f}s', flush=True)

for Ks in KS:
    K = 2 * Ks
    jr = train_judge([rand_sub(XE, i, K, 0) for i in tr], y[tr], epochs=EP_SUB, seed=seed)
    L[f'random_uni_K{K}'] = judge_logits(jr, [rand_sub(XE, i, K, 0) for i in te])
    ju, _ = train_debate(XE, Ks)
    L[f'teach_uni_K{K}'] = games(XE, te, lambda X, M, yb: play_debate(ju, X, M, Ks))[1]
    jc, pres = train_debate(XC, Ks)  # cheap students' own model
    _, L[f'teach_r50_K{K}'], idx_te, f_te = games(XC, te, lambda X, M, yb: play_debate(jc, X, M, Ks))
    # expensive teacher: warm start on random UNI subsets, then trained on the UNI re-encoding of every cheap round on training slides
    je = train_judge([rand_sub(XE, i, K, 1) for i in tr], y[tr], epochs=EP_SUB, seed=seed)
    TU = torch.cat([to_uni(tr, idx, f) for idx, f in pres])
    je = train_judge(TU, np.tile(y[tr], len(pres)), epochs=EP_FT * 2, judge=je, seed=seed + 50)
    Tte = to_uni(te, idx_te, f_te)
    L[f'cheap2uni_K{K}'] = judge_logits(je, Tte)
    L[f'cheap2uni_blank_K{K}'] = judge_logits(je, torch.cat([torch.zeros_like(Tte[..., :-1]), Tte[..., -1:]], -1))
    _, _, idx_l, f_l = games(XC, te, lambda X, M, yb: play_debate(jc, X, M, Ks, pol1=Mixed(greedy, make_rollout(8)), pol0=Mixed(greedy, make_rollout(8)), y=yb))
    L[f'cheap2uni_liar_rollout_K{K}'] = judge_logits(je, to_uni(te, idx_l, f_l))
    print(f'K{K} {time.time() - t0:.0f}s', {m: round(metrics(v, y[te])['balacc'], 3) for m, v in L.items() if m.endswith(f'_K{K}')}, flush=True)
    json.dump({'slide': {m: metrics(v, y[te]) for m, v in L.items()},
               'info': {'seed': seed, 'n_test': len(te), 'mean_patches_test': float(np.mean([len(XE[i]) for i in te]))}},
              open(os.path.join(OUT, f'cheap_s{seed}.json'), 'w'), indent=1)
np.savez(os.path.join(OUT, f'cheap_logits_s{seed}.npz'), y=y[te], **L)
print('DONE', flush=True)
