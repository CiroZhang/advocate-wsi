"""Truly cheap students vs an ADMIL-style selector (CAMELYON16). Cheap encoder CHEAP in {b0, r18, mnv3} (ImageNet, frozen, level-2 128px
reads from raw slides; cheap_feats.py) or r50bt (torchmil). Expensive encoder: UNI (torchmil features, same patches).
  full_uni / full_cheap          : full-slide ABMIL on each encoder
  admil_K                        : ADMIL (arXiv 2608.22066) style: a cheap attention head ("PriorNet") is distilled (KL) from the
                                   full-slide UNI ABMIL teacher's attention; at test the frozen UNI teacher reads only the top-K patches
  admil_attacked_K               : same, compromised selector: picks K patches pushing the cheap full-slide ABMIL toward the wrong label
  cascade_K / cascade_attacked_K : cheap full-slide ABMIL attention top-K -> UNI judge trained on its top-K (exp_cascade.py)
  cheap2uni_K                    : teaching round, cheap students -> UNI teacher (exp_cheap.py), B = K presented patches
  cheap2uni_liar_rollout_K       : same, look-ahead devil's advocate
Per-slide encoder cost: N * cheap + K * UNI for every selective method.
usage: CHEAP=b0 python exp_admil.py SEED"""
import sys, json, time, csv, numpy as np, torch, torch.nn as nn
import os
ROOT = os.environ.get('ADV_ROOT', '/scratch/ciro/cvpr_lab')  # data/ and studentB_debate/ live under here
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # common.py sits next to this script
from common import *

seed = int(sys.argv[1]); CH = os.environ.get('CHEAP', 'b0'); KS = [1, 2, 4, 8]; R = int(os.environ.get('ROUNDS', 4))
EP_FULL, EP_SUB, EP_FT = 20, 40, 20; BUD = 400_000; AK = [8, 16, 32, 128]
D = ROOT + "/data/camelyon16/torchmil"; CD = ROOT + f"/data/camelyon16/cheap/{CH}"
OUT = ROOT + "/studentB_debate/results"
cpath = (lambda n: os.path.join(D, 'features_r50bt', n + '.npy')) if CH == 'r50bt' else (lambda n: os.path.join(CD, n + '.npy'))
rng = np.random.default_rng(seed); torch.manual_seed(seed)
rows = list(csv.DictReader(open(os.path.join(D, 'splits.csv'))))
names = [r['bag_name'] for r in rows if os.path.exists(os.path.join(D, 'features', r['bag_name'] + '.npy')) and os.path.exists(cpath(r['bag_name']))]
split = {r['bag_name']: r['split'] for r in rows}; t0 = time.time()
XE = [torch.from_numpy(np.load(os.path.join(D, 'features', n + '.npy'))).half() for n in names]
XC = [torch.from_numpy(np.load(cpath(n)).astype(np.float32)).half() for n in names]
y = np.array([int(np.load(os.path.join(D, 'labels', n + '.npy')).reshape(-1)[0]) for n in names])
assert all(len(a) == len(b) for a, b in zip(XE, XC)), 'UNI and cheap bags not aligned'
# standardise cheap features (ImageNet features have large, uneven scales); statistics from training slides only
tr = np.array([i for i, n in enumerate(names) if split[n] == 'train']); te = np.array([i for i, n in enumerate(names) if split[n] == 'test'])
cat = torch.cat([XC[i].float() for i in tr]); mu, sd = cat.mean(0), cat.std(0) + 1e-4; del cat
XC = [((x.float() - mu) / sd).half() for x in XC]
print(f'{CH}: loaded {len(names)} in {time.time() - t0:.0f}s; train {len(tr)} test {len(te)}; dims UNI {XE[0].shape[1]} cheap {XC[0].shape[1]}', flush=True)
L = {}
def fl(x, f): return flag(x.float(), f)
def rand_sub(xs, i, K, f):
    x = xs[i]; p = torch.randperm(len(x))[:K]
    ff = torch.zeros(len(p), 1) if f == 0 else (torch.randint(0, 2, (len(p), 1)) * 2 - 1).float()
    return torch.cat([x[p].float(), ff], 1)

def games(xs, ids, fn):
    def g(X, M, k):
        st = fn(X, M, y[ids][k]); return st.transcript().cpu(), st.logit(), torch.stack(st.idx, 1), torch.stack(st.flags, 1)
    res = batched(g, [xs[i] for i in ids], bs=32, budget=BUD)
    return (torch.stack([r[0] for r in res]), np.array([r[1].item() for r in res]),
            torch.stack([r[2] for r in res]).cpu(), torch.stack([r[3] for r in res]).cpu())

def to_uni(ids, idx, flags):
    return torch.stack([torch.cat([XE[i][idx[r]].float(), flags[r][:, None].float()], 1) for r, i in enumerate(ids)])

def train_debate(xs, Ks):
    jd = train_judge([rand_sub(xs, i, 2 * Ks, 1) for i in tr], y[tr], epochs=EP_SUB, seed=seed); bx, pres = [], []
    for r in range(R):
        T, lg, idx, f = games(xs, tr, lambda X, M, yb: play_debate(jd, X, M, Ks)); bx.append(T); pres.append((idx, f))
        print(f'   round {r} train balacc {balacc((lg > 0).astype(int), y[tr]):.3f}', flush=True)
        jd = train_judge(torch.cat(bx), np.tile(y[tr], len(bx)), epochs=EP_FT, judge=jd, seed=seed + r)
    idx, f = games(xs, tr, lambda X, M, yb: play_debate(jd, X, M, Ks))[2:]; pres.append((idx, f))
    return jd, pres

# full-slide references
fu = train_judge([fl(XE[i], 0) for i in tr], y[tr], epochs=EP_FULL, bs=1, lr=2e-4, seed=seed)
fr = train_judge([fl(XC[i], 0) for i in tr], y[tr], epochs=EP_FULL, bs=1, lr=2e-4, seed=seed)
L['full_uni'] = judge_logits(fu, [fl(XE[i], 0) for i in te], bs=1); L['full_cheap'] = judge_logits(fr, [fl(XC[i], 0) for i in te], bs=1)
print({m: round(metrics(v, y[te])['balacc'], 3) for m, v in L.items()}, f'{time.time() - t0:.0f}s', flush=True)

# ADMIL-style PriorNet: gated-attention scorer on cheap features, KL-distilled from the UNI teacher's attention over all patches
class Prior(nn.Module):
    def __init__(s, d, h=128):
        super().__init__(); s.f = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Dropout(.1)); s.a = nn.Linear(h, h); s.b = nn.Linear(h, h); s.w = nn.Linear(h, 1)
    def forward(s, x): z = s.f(x); return s.w(torch.tanh(s.a(z)) * torch.sigmoid(s.b(z))).squeeze(-1)
with torch.no_grad(): ta = {i: fu.terms(fl(XE[i], 0).to(DEV))[1].float().cpu() for i in range(len(names))}
pn = Prior(XC[0].shape[1]).to(DEV); opt = torch.optim.AdamW(pn.parameters(), 5e-4, weight_decay=1e-4)
for ep in range(20):
    pn.train(); tot = 0
    for i in rng.permutation(tr):
        s = pn(XC[i].float().to(DEV)); p = ta[i].to(DEV).softmax(0)
        loss = (p * (p.clamp_min(1e-12).log() - s.log_softmax(0))).sum(); opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
pn.eval()
with torch.no_grad(): ps = {i: pn(XC[i].float().to(DEV)).cpu() for i in te}
print(f'PriorNet KL {tot / len(tr):.3f}; top-128 overlap with teacher', np.mean([len(set(ps[i].topk(min(128, len(ps[i]))).indices.tolist()) &
      set(ta[i].topk(min(128, len(ta[i]))).indices.tolist())) / min(128, len(ps[i])) for i in te]).round(3), flush=True)

@torch.no_grad()
def attacked_idx(i, K):  # compromised selector: white-box on the cheap full-slide ABMIL, pushing toward the wrong label
    st = play_single(fr, XC[i].float().to(DEV)[None], torch.ones(1, len(XC[i]), dtype=torch.bool, device=DEV), K,
                     torch.tensor([1 - y[i]], device=DEV), torch.zeros(1, device=DEV))
    return torch.stack(st.idx, 1)[0].cpu()
for K in AK:
    L[f'admil_K{K}'] = np.array([judge_logits(fu, [fl(XE[i][ps[i].topk(min(K, len(ps[i]))).indices], 0)], bs=1)[0] for i in te])
    L[f'admil_attacked_K{K}'] = np.array([judge_logits(fu, [fl(XE[i][attacked_idx(i, K)], 0)], bs=1)[0] for i in te])
    L[f'oracle_K{K}'] = np.array([judge_logits(fu, [fl(XE[i][ta[i].topk(min(K, len(ta[i]))).indices], 0)], bs=1)[0] for i in te])
    print(f'ADMIL K{K}', {m: round(metrics(v, y[te])['balacc'], 3) for m, v in L.items() if m.endswith(f'_K{K}')}, flush=True)

with torch.no_grad(): att = {i: fr.terms(fl(XC[i].to(DEV), 0))[1].cpu() for i in range(len(names))}
for Ks in KS:
    K = 2 * Ks
    sel = lambda i: att[i].topk(min(K, len(XC[i]))).indices
    jc0 = train_judge([fl(XE[i][sel(i)], 0) for i in tr], y[tr], epochs=EP_SUB, seed=seed)
    L[f'cascade_K{K}'] = judge_logits(jc0, [fl(XE[i][sel(i)], 0) for i in te])
    L[f'cascade_attacked_K{K}'] = judge_logits(jc0, [fl(XE[i][attacked_idx(i, K)], 0) for i in te])
    jc, pres = train_debate(XC, Ks)
    _, L[f'teach_cheap_K{K}'], idx_te, f_te = games(XC, te, lambda X, M, yb: play_debate(jc, X, M, Ks))
    je = train_judge([rand_sub(XE, i, K, 1) for i in tr], y[tr], epochs=EP_SUB, seed=seed)
    je = train_judge(torch.cat([to_uni(tr, idx, f) for idx, f in pres]), np.tile(y[tr], len(pres)), epochs=EP_FT * 2, judge=je, seed=seed + 50)
    L[f'cheap2uni_K{K}'] = judge_logits(je, to_uni(te, idx_te, f_te))
    _, _, idx_l, f_l = games(XC, te, lambda X, M, yb: play_debate(jc, X, M, Ks, pol1=Mixed(greedy, make_rollout(8)), pol0=Mixed(greedy, make_rollout(8)), y=yb))
    L[f'cheap2uni_liar_rollout_K{K}'] = judge_logits(je, to_uni(te, idx_l, f_l))
    print(f'K{K} {time.time() - t0:.0f}s', {m: round(metrics(v, y[te])['balacc'], 3) for m, v in L.items() if m.endswith(f'_K{K}')}, flush=True)
    json.dump({'slide': {m: metrics(v, y[te]) for m, v in L.items()},
               'info': {'seed': seed, 'cheap': CH, 'n_test': len(te), 'mean_patches_test': float(np.mean([len(XE[i]) for i in te]))}},
              open(os.path.join(OUT, f'admil_{CH}_s{seed}.json'), 'w'), indent=1)
np.savez(os.path.join(OUT, f'admil_{CH}_logits_s{seed}.npz'), y=y[te], names=np.array(names)[te], **L)
print('DONE', flush=True)
