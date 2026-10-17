"""Distilled cheap students (CAMELYON16). ADMIL's strength is that its cheap selector is distilled from the expensive teacher; ours is that
two opposed students present evidence. Here both: a cheap per-patch scorer g (MLP on cheap features) is regressed onto the full-slide UNI
ABMIL's per-patch evidence (instance logit c(f(x)) and attention), and the two students present, respectively, the top-Ks and bottom-Ks
patches under g. Only presented patches are encoded with UNI; the UNI teacher (gated ABMIL + presenter flag) is trained on the UNI
re-encoding of these rounds and of rounds where the liar is white-box on the current teacher (adversarial training, 3 rounds).
Attacks at test, all white-box on UNI features (strongest; the attacker pays for UNI on every patch):
  distill2uni_wb_K  : the wrong-label student replaced by a greedy white-box attacker on the UNI teacher
  admil_wb_K        : ADMIL-style selector replaced by a greedy white-box attacker on the frozen full UNI teacher (same budget K)
  admil_K, oracle_K : clean ADMIL-style selector (KL-distilled attention), teacher-attention oracle
usage: CHEAP=b0 python exp_distill.py SEED"""
import sys, json, time, csv, numpy as np, torch, torch.nn as nn
sys.path.insert(0, "/scratch/ciro/cvpr_lab/studentB_debate")
from common import *

seed = int(sys.argv[1]); CH = os.environ.get('CHEAP', 'b0'); KS = [1, 2, 4, 8]; EP_FULL, EP_SUB, EP_FT = 20, 40, 20
D = "/scratch/ciro/cvpr_lab/data/camelyon16/torchmil"; CD = f"/scratch/ciro/cvpr_lab/data/camelyon16/cheap/{CH}"
OUT = "/scratch/ciro/cvpr_lab/studentB_debate/results"
cpath = (lambda n: os.path.join(D, 'features_r50bt', n + '.npy')) if CH == 'r50bt' else (lambda n: os.path.join(CD, n + '.npy'))
rng = np.random.default_rng(seed); torch.manual_seed(seed)
rows = list(csv.DictReader(open(os.path.join(D, 'splits.csv'))))
names = [r['bag_name'] for r in rows if os.path.exists(os.path.join(D, 'features', r['bag_name'] + '.npy')) and os.path.exists(cpath(r['bag_name']))]
split = {r['bag_name']: r['split'] for r in rows}; t0 = time.time()
XE = [torch.from_numpy(np.load(os.path.join(D, 'features', n + '.npy'))).half() for n in names]
XC = [torch.from_numpy(np.load(cpath(n)).astype(np.float32)).half() for n in names]
y = np.array([int(np.load(os.path.join(D, 'labels', n + '.npy')).reshape(-1)[0]) for n in names])
assert all(len(a) == len(b) for a, b in zip(XE, XC))
tr = np.array([i for i, n in enumerate(names) if split[n] == 'train']); te = np.array([i for i, n in enumerate(names) if split[n] == 'test'])
cat = torch.cat([XC[i].float() for i in tr]); mu, sd = cat.mean(0), cat.std(0) + 1e-4; del cat
XC = [((x.float() - mu) / sd).half() for x in XC]
print(f'{CH}: loaded {len(names)} in {time.time() - t0:.0f}s', flush=True)
L = {}
def fl(x, f): return flag(x.float(), f)

fu = train_judge([fl(XE[i], 0) for i in tr], y[tr], epochs=EP_FULL, bs=1, lr=2e-4, seed=seed)
L['full_uni'] = judge_logits(fu, [fl(XE[i], 0) for i in te], bs=1)
with torch.no_grad():
    TA, TC = {}, {}
    for i in range(len(names)):
        z, a = fu.terms(fl(XE[i], 0).to(DEV)); TA[i] = a.float().cpu(); TC[i] = fu.c(z).squeeze(-1).float().cpu()

class MLP(nn.Module):
    def __init__(s, d, o, h=256):
        super().__init__(); s.n = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Dropout(.1), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, o))
    def forward(s, x): return s.n(x)
# one cheap network, two heads: [instance evidence c, attention a]; MSE on standardised c + KL on attention (ADMIL)
g = MLP(XC[0].shape[1], 2).to(DEV); opt = torch.optim.AdamW(g.parameters(), 5e-4, weight_decay=1e-4)
cs = torch.cat([TC[i] for i in tr]).std().item()
for ep in range(20):
    g.train()
    for i in rng.permutation(tr):
        o = g(XC[i].float().to(DEV)); p = TA[i].to(DEV).softmax(0)
        loss = ((o[:, 0] - TC[i].to(DEV) / cs) ** 2).mean() + (p * (p.clamp_min(1e-12).log() - o[:, 1].log_softmax(0))).sum()
        opt.zero_grad(); loss.backward(); opt.step()
g.eval()
with torch.no_grad(): G = {i: g(XC[i].float().to(DEV)).cpu() for i in range(len(names))}
from scipy.stats import spearmanr
print('evidence rank corr (test)', np.mean([spearmanr(G[i][:, 0], TC[i])[0] for i in te]).round(3), f'{time.time() - t0:.0f}s', flush=True)

def ev(i): return G[i][:, 0]  # distilled instance evidence (tumour-likeness under the UNI teacher)
def student_idx(i, Ks):
    e = ev(i); k = min(Ks, len(e)); return e.topk(k).indices, (-e).topk(k).indices
def present(i, i1, i0):
    return torch.cat([fl(XE[i][i1], 1), fl(XE[i][i0], -1)])
@torch.no_grad()
def wb_side(j, i, Ks, side):  # greedy white-box on the UNI judge j: patches whose single-patch logit (with this side's flag) pushes most to `side`
    s = judge_logits(j, fl(XE[i], 1 if side == 1 else -1)[:, None], bs=4096)
    s = torch.as_tensor(s); return (s if side == 1 else -s).topk(min(Ks, len(s))).indices
@torch.no_grad()
def adapt_logit(j, i, hon, fh, w, Ks, fw):
    """Adaptive white-box attacker on judge j (UNI features): the honest patches `hon` (flag fh) are shown first, then the attacker
    greedily adds Ks patches (flag fw), each maximising the judge's move toward the wrong label w given everything shown so far."""
    X = XE[i].float().to(DEV)[None]; st = State(j, X, torch.ones(1, X.shape[1], dtype=torch.bool, device=DEV))
    st.T = len(hon) + Ks; st.order = [torch.tensor([w], device=DEV)] * st.T
    for p in hon: st.add(torch.tensor([int(p)], device=DEV), torch.tensor([float(fh)], device=DEV))
    fwt = torch.tensor([float(fw)], device=DEV)
    for _ in range(Ks): st.add(greedy(st, w, fwt), fwt)
    return st.logit().item()
def present0(i, i1, i0): return torch.cat([fl(XE[i][i1], 0), fl(XE[i][i0], 0)])
@torch.no_grad()
def wb_side0(j, i, Ks, side):
    s = torch.as_tensor(judge_logits(j, fl(XE[i], 0)[:, None], bs=4096)); return (s if side == 1 else -s).topk(min(Ks, len(s))).indices
def admil_admil(i, K): return G[i][:, 1].topk(min(K, len(G[i]))).indices
@torch.no_grad()
def wb_select(i, K):  # compromised ADMIL selector: white-box greedy on the frozen full UNI teacher toward the wrong label
    st = play_single(fu, XE[i].float().to(DEV)[None], torch.ones(1, len(XE[i]), dtype=torch.bool, device=DEV), K,
                     torch.tensor([1 - y[i]], device=DEV), torch.zeros(1, device=DEV))
    return torch.stack(st.idx, 1)[0].cpu()

for K in (8, 16, 32):
    L[f'admil_K{K}'] = np.array([judge_logits(fu, [fl(XE[i][admil_admil(i, K)], 0)], bs=1)[0] for i in te])
    L[f'admil_wb_K{K}'] = np.array([judge_logits(fu, [fl(XE[i][wb_select(i, K)], 0)], bs=1)[0] for i in te])
for K in (8, 16, 32):
    L[f'union_admil_adapt_K{K}'] = np.array([adapt_logit(fu, i, admil_admil(i, K // 2), 0, 1 - y[i], K // 2, 0) for i in te])
for Ks in KS:
    K = 2 * Ks
    S = {i: student_idx(i, Ks) for i in range(len(names))}
    def rand_sub(i):
        p = torch.randperm(len(XE[i]))[:K]; return torch.cat([XE[i][p].float(), (torch.randint(0, 2, (len(p), 1)) * 2 - 1).float()], 1)
    je = train_judge([rand_sub(i) for i in tr], y[tr], epochs=EP_SUB, seed=seed); sets, ys = [], []
    for r in range(3):  # honest rounds + rounds where the wrong-label side is white-box on the current teacher
        sets += [present(i, *S[i]) for i in tr]; ys += list(y[tr])
        for i in tr:
            a1 = wb_side(je, i, Ks, 1) if y[i] == 0 else S[i][0]; a0 = wb_side(je, i, Ks, 0) if y[i] == 1 else S[i][1]
            sets.append(present(i, a1, a0)); ys.append(y[i])
        je = train_judge(sets, np.array(ys), epochs=EP_FT, judge=je, seed=seed + r)
    L[f'distill2uni_K{K}'] = judge_logits(je, [present(i, *S[i]) for i in te])
    L[f'distill2uni_wb_K{K}'] = judge_logits(je, [present(i, wb_side(je, i, Ks, 1) if y[i] == 0 else S[i][0],
                                                          wb_side(je, i, Ks, 0) if y[i] == 1 else S[i][1]) for i in te])
    hon = lambda i: S[i][0] if y[i] == 1 else S[i][1]
    L[f'distill2uni_adapt_K{K}'] = np.array([adapt_logit(je, i, hon(i), 1 if y[i] == 1 else -1, 1 - y[i], Ks, -1 if y[i] == 1 else 1) for i in te])
    j0 = train_judge([present0(i, *(lambda p: (p[:Ks], p[Ks:]))(torch.randperm(len(XE[i]))[:K])) for i in tr], y[tr], epochs=EP_SUB, seed=seed); sets, ys = [], []
    for r in range(3):  # identical training, but the teacher is not told which student argued which side
        sets += [present0(i, *S[i]) for i in tr]; ys += list(y[tr])
        for i in tr:
            a1 = wb_side0(j0, i, Ks, 1) if y[i] == 0 else S[i][0]; a0 = wb_side0(j0, i, Ks, 0) if y[i] == 1 else S[i][1]
            sets.append(present0(i, a1, a0)); ys.append(y[i])
        j0 = train_judge(sets, np.array(ys), epochs=EP_FT, judge=j0, seed=seed + r)
    L[f'noflag_K{K}'] = judge_logits(j0, [present0(i, *S[i]) for i in te])
    L[f'noflag_adapt_K{K}'] = np.array([adapt_logit(j0, i, hon(i), 0, 1 - y[i], Ks, 0) for i in te])
    print(f'K{K} {time.time() - t0:.0f}s', {m: round(metrics(v, y[te])['balacc'], 3) for m, v in L.items() if m.endswith(f'_K{K}') or m == 'full_uni'}, flush=True)
    json.dump({'slide': {m: metrics(v, y[te]) for m, v in L.items()}, 'info': {'seed': seed, 'cheap': CH}},
              open(os.path.join(OUT, f'distill2_{CH}_s{seed}.json'), 'w'), indent=1)
np.savez(os.path.join(OUT, f'distill2_{CH}_logits_s{seed}.npz'), y=y[te], **L)
print('DONE', flush=True)
