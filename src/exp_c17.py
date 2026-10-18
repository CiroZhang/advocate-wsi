"""CAMELYON17 multi-level teaching round: one cheap student per level (negative / ITC / micro / macro), each presents Kc tiles of
evidence for its level; only presented tiles are encoded with CONCH; a CONCH teacher (ABMIL over presented tiles, input = CONCH
feature + one-hot presenter + position relative to that presenter's first tile, in mm) picks the level.
Students share one cheap per-tile tumour-evidence scorer (MLP on cheap ImageNet features, distilled from a full-slide CONCH tumour
ABMIL's instance logits) and differ only in their search rule over (evidence, position):
  neg   : the Kc least tumour-like tiles
  itc   : the most tumour-like tile + its Kc-1 nearest tiles (an isolated focus)
  micro : the most tumour-like tile + the Kc-1 most tumour-like tiles within 2 mm of it (a compact cluster)
  macro : farthest-point sampling over the top-2% tumour-like tiles (spread of tumour)
Baselines: full-slide 4-class CONCH ABMIL; ADMIL-style (cheap attention distilled from the full 4-class teacher, top-K -> full teacher).
Attacks (white-box on the CONCH teacher, greedy single-tile scores): one wrong-level student replaced (ours) / the selector replaced (ADMIL).
Split: leave-one-centre-out (FOLD = held-out centre 0-4). Metrics: accuracy, balanced accuracy, quadratic kappa.
usage: CHEAP=b0 python exp_c17.py SEED FOLD"""
import sys, os, json, time, csv, numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn
ROOT = os.environ.get('ADV_ROOT', '/scratch/ciro/cvpr_lab')  # data/ and studentB_debate/ live under here
from sklearn.metrics import cohen_kappa_score, balanced_accuracy_score
seed, fold = int(sys.argv[1]), int(sys.argv[2]); CH = os.environ.get('CHEAP', 'b0'); Kc = int(os.environ.get('KC', 4)); SH = os.environ.get('SHIFT')  # stain-shifted cheap view for TEST slides only
D = ROOT + "/data/camelyon17"; OUT = ROOT + "/studentB_debate/results"; DEV = 'cuda' if torch.cuda.is_available() else 'cpu'; C = 4
LV = {'negative': 0, 'itc': 1, 'micro': 2, 'macro': 3}
torch.manual_seed(seed); rng = np.random.default_rng(seed); t0 = time.time()
lab, cen = {}, {}
for r in csv.DictReader(open(f'{D}/stages.csv')):
    if r['patient'].endswith('.tif'): lab[r['patient'][:-4]] = LV[r['stage'].strip().lower()]; cen[r['patient'][:-4]] = int(r['center'])
names = sorted(n for n in lab if os.path.exists(f'{D}/feats/{n}.pt'))
XE, XC, P = [], [], []
for n in names:
    d = torch.load(f'{D}/feats/{n}.pt'); XE.append(d['conch'].float()); XC.append(d[CH].float()); P.append(d['coords'].float() * d['mpp'] / 1000)  # mm
keep = [i for i in range(len(names)) if len(XE[i]) > 0]; names = [names[i] for i in keep]; XE = [XE[i] for i in keep]; XC = [XC[i] for i in keep]; P = [P[i] for i in keep]
y = np.array([lab[n] for n in names]); c = np.array([cen[n] for n in names])
tr = np.where(c != fold)[0]; te = np.where(c == fold)[0]
if SH is not None:
    for i in te: XC[i] = torch.load(f'{D}/feats_shift/s{SH}/{names[i]}.pt')[CH].float()
cat = torch.cat([XC[i] for i in tr]); mu, sd = cat.mean(0), cat.std(0) + 1e-4; XC = [(x - mu) / sd for x in XC]; del cat
print(f'{len(names)} slides, train {len(tr)} test {len(te)} (centre {fold}); test labels {np.bincount(y[te], minlength=C)}; {time.time() - t0:.0f}s', flush=True)

class ABMIL(nn.Module):
    def __init__(s, d, o, h=128, drop=.1):
        super().__init__(); s.f = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Dropout(drop)); s.a = nn.Linear(h, h); s.b = nn.Linear(h, h); s.w = nn.Linear(h, 1); s.c = nn.Linear(h, o)
    def terms(s, x): z = s.f(x); return z, s.w(torch.tanh(s.a(z)) * torch.sigmoid(s.b(z))).squeeze(-1)
    def forward(s, x): z, a = s.terms(x); return s.c((a.softmax(0)[:, None] * z).sum(0))
def train(sets, ys, o, epochs, model=None, lr=2e-4):
    m = model or ABMIL(sets[0].shape[1], o).to(DEV); opt = torch.optim.AdamW(m.parameters(), lr, weight_decay=1e-4)
    ys = np.asarray(ys); w = torch.tensor(len(ys) / (o * np.maximum(np.bincount(ys, minlength=o), 1)), dtype=torch.float, device=DEV)
    for _ in range(epochs):
        m.train()
        for k in rng.permutation(len(ys)):
            out = m(sets[k].to(DEV))
            loss = Fn.cross_entropy(out[None], torch.tensor([ys[k]], device=DEV), weight=w) if o > 1 else Fn.binary_cross_entropy_with_logits(out, torch.tensor([float(ys[k])], device=DEV), pos_weight=w[1:] / w[:1])
            opt.zero_grad(); loss.backward(); opt.step()
    return m.eval()
@torch.no_grad()
def predict(m, sets): return np.stack([m(s.to(DEV)).cpu().numpy() for s in sets])
def mets(lg):
    p = lg.argmax(1); yt = y[te]
    return {'acc': float((p == yt).mean()), 'balacc': float(balanced_accuracy_score(yt, p)), 'kappa': float(cohen_kappa_score(yt, p, weights='quadratic')),
            'tumour_balacc': float(balanced_accuracy_score(yt > 0, p > 0))}
L = {}
# full-slide references (expensive: every tile through CONCH)
fu4 = train([XE[i] for i in tr], y[tr], C, 20); L['full_conch'] = predict(fu4, [XE[i] for i in te])
fub = train([XE[i] for i in tr], (y[tr] > 0).astype(int), 1, 20)
print('full', mets(L['full_conch']), f'{time.time() - t0:.0f}s', flush=True)
# distilled cheap scorer: instance tumour evidence + 4-class attention (ADMIL PriorNet)
with torch.no_grad():
    TE, TA = {}, {}
    for i in range(len(names)):
        z, _ = fub.terms(XE[i].to(DEV)); TE[i] = fub.c(z).squeeze(-1).cpu(); TA[i] = fu4.terms(XE[i].to(DEV))[1].cpu()
g = nn.Sequential(nn.Linear(XC[0].shape[1], 256), nn.ReLU(), nn.Dropout(.1), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 2)).to(DEV)
opt = torch.optim.AdamW(g.parameters(), 5e-4, weight_decay=1e-4); es = torch.cat([TE[i] for i in tr]).std().item()
for ep in range(20):
    g.train()
    for i in rng.permutation(tr):
        o = g(XC[i].to(DEV)); p = TA[i].to(DEV).softmax(0)
        loss = ((o[:, 0] - TE[i].to(DEV) / es) ** 2).mean() + (p * (p.clamp_min(1e-12).log() - o[:, 1].log_softmax(0))).sum()
        opt.zero_grad(); loss.backward(); opt.step()
g.eval()
with torch.no_grad(): G = {i: g(XC[i].to(DEV)).cpu() for i in range(len(names))}

def fps(pts, k):
    sel = [0]; dmin = (pts - pts[0]).norm(dim=1)
    for _ in range(1, min(k, len(pts))): j = int(dmin.argmax()); sel.append(j); dmin = torch.minimum(dmin, (pts - pts[j]).norm(dim=1))
    return torch.tensor(sel)
def students(i):
    e, p = G[i][:, 0], P[i]; n = len(e); k = min(Kc, n); top = int(e.argmax()); dd = (p - p[top]).norm(dim=1)
    neg = (-e).topk(k).indices
    itc = dd.topk(k, largest=False).indices
    near = torch.where(dd <= 2.0)[0]; micro = near[e[near].topk(min(k, len(near))).indices]
    hi = e.topk(max(k, int(.02 * n))).indices; macro = hi[fps(p[hi], k)]
    return [neg, itc, micro, macro]
def present(i, S):
    rows = []
    for lv, idx in enumerate(S):
        oh = torch.zeros(len(idx), C); oh[:, lv] = 1; rel = (P[i][idx] - P[i][idx[0]]) / 5.0
        rows.append(torch.cat([XE[i][idx], oh, rel], 1))
    return torch.cat(rows)
@torch.no_grad()
def wb_replace(m, i, S, lv, keep_flags=True):  # student `lv` replaced by a white-box attacker: tiles whose single-tile teacher score for `lv` is highest
    n = len(XE[i]); oh = torch.zeros(n, C); oh[:, lv] = float(keep_flags); rel = torch.zeros(n, 2)
    X = torch.cat([XE[i], oh, rel], 1).to(DEV); z, a = m.terms(X); s = Fn.log_softmax(m.c(z), 1)[:, lv]
    idx = s.topk(min(Kc, n)).indices.cpu(); S2 = list(S); S2[lv] = idx; return S2
SS = {i: students(i) for i in range(len(names))}
# teacher: warm start on random tiles with random presenter labels, then 3 rounds of clean + one-liar (white-box) transcripts
def rand_pres(i):
    n = len(XE[i]); return [torch.randperm(n)[:min(Kc, n)] for _ in range(C)]
def run_teacher(keep_flags, tag):
    pr = (lambda i, S: present(i, S)) if keep_flags else (lambda i, S: (lambda X: torch.cat([X[:, :-6], torch.zeros(len(X), C), X[:, -2:]], 1))(present(i, S)))
    tch = train([pr(i, rand_pres(i)) for i in tr], y[tr], C, 10); sets, ys = [], []
    for r in range(3):
        for i in tr:
            sets.append(pr(i, SS[i])); ys.append(y[i])
            lv = int(rng.choice([l for l in range(C) if l != y[i]])); sets.append(pr(i, wb_replace(tch, i, SS[i], lv, keep_flags))); ys.append(y[i])
        tch = train(sets, ys, C, 5, model=tch)
    L[tag] = predict(tch, [pr(i, SS[i]) for i in te]); worst = []
    for i in te:
        cands = [predict(tch, [pr(i, wb_replace(tch, i, SS[i], lv, keep_flags))])[0] for lv in range(C) if lv != y[i]]
        worst.append(min(cands, key=lambda v: Fn.softmax(torch.tensor(v), 0)[y[i]].item()))
    L[tag + '_wb_worst'] = np.stack(worst)
run_teacher(True, 'teach'); run_teacher(False, 'noflag')
# ADMIL-style: top-K (K = C*Kc) by distilled attention -> full 4-class teacher; compromised selector: white-box toward a wrong level
K = C * Kc
L['admil'] = predict(fu4, [XE[i][G[i][:, 1].topk(min(K, len(XE[i]))).indices] for i in te])
@torch.no_grad()
def admil_wb(i):
    z, a = fu4.terms(XE[i].to(DEV)); lp = Fn.log_softmax(fu4.c(z), 1).cpu(); best = None
    for lv in range(C):
        if lv == y[i]: continue
        v = predict(fu4, [XE[i][lp[:, lv].topk(min(K, len(lp))).indices]])[0]
        if best is None or Fn.softmax(torch.tensor(v), 0)[y[i]] < Fn.softmax(torch.tensor(best), 0)[y[i]]: best = v
    return best
L['admil_wb_worst'] = np.stack([admil_wb(i) for i in te])
res = {k: mets(v) for k, v in L.items()}
for k, v in res.items(): print(k, {a: round(b, 3) for a, b in v.items()}, flush=True)
json.dump({'res': res, 'info': {'seed': seed, 'fold': fold, 'cheap': CH, 'Kc': Kc, 'shift': SH, 'n_train': int(len(tr)), 'n_test': int(len(te)),
           'mean_tiles_test': float(np.mean([len(XE[i]) for i in te]))}}, open(f'{OUT}/c17_{CH}_k{Kc}' + (f'_sh{SH}' if SH else '') + f'_f{fold}_s{seed}.json', 'w'), indent=1)
np.savez(f'{OUT}/c17_{CH}_k{Kc}' + (f'_sh{SH}' if SH else '') + f'_f{fold}_s{seed}.npz', y=y[te], names=np.array(names)[te], **L)
print('DONE', f'{time.time() - t0:.0f}s', flush=True)
