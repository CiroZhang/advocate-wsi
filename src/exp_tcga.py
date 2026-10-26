"""TCGA subtyping teaching round (RCC: KIRC/KIRP/KICH; NSCLC: LUAD/LUSC): one cheap student per subtype presents Kc tiles that look
most like its subtype; only presented tiles are encoded with CONCH; a CONCH teacher (ABMIL over presented tiles + one-hot presenter) decides.
Cheap scorer: MLP on cheap ImageNet features distilled from the full-slide CONCH subtype ABMIL (per-tile class logits + attention).
Student k: top-Kc tiles by distilled margin (logit_k - max other). Baselines: full-slide CONCH ABMIL; ADMIL-style top-K by distilled
attention -> full teacher. Attacks as exp_c17 (one wrong-subtype student / the selector replaced by a white-box attacker).
Split: 5-fold patient-level stratified CV (FOLD 0-4). usage: CHEAP=b0 KC=4 python exp_tcga.py SEED FOLD COHORT(rcc|nsclc)"""
import sys, os, json, time, csv, numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn
ROOT = os.environ.get('ADV_ROOT', '/scratch/ciro/cvpr_lab')  # data/ and studentB_debate/ live under here
from sklearn.metrics import cohen_kappa_score, balanced_accuracy_score, roc_auc_score
seed, fold, CO = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]; CH = os.environ.get('CHEAP', 'b0'); Kc = int(os.environ.get('KC', 4))
D = ROOT + f"/data/tcga_{CO}"; OUT = ROOT + "/studentB_debate/results"; DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
PR = {'rcc': ['TCGA-KIRC', 'TCGA-KIRP', 'TCGA-KICH'], 'nsclc': ['TCGA-LUAD', 'TCGA-LUSC']}[CO]; C = len(PR)
torch.manual_seed(seed); rng = np.random.default_rng(seed); t0 = time.time()
lab, case = {}, {}
for r in csv.DictReader(open(f'{D}/manifest.tsv'), delimiter='\t'): lab[r['file_name'][:-4]] = PR.index(r['project']); case[r['file_name'][:-4]] = r['case']
names = sorted(n for n in lab if n.split('-')[3][:2] == '01' and os.path.exists(f'{D}/feats/{n}.pt'))  # primary tumour slides only (drop -11 normal, -05)
XE, XC = [], []
for n in names:
    d = torch.load(f'{D}/feats/{n}.pt'); XE.append(d['conch'].float()); XC.append(d[CH].float())
keep = [i for i in range(len(names)) if len(XE[i]) > 0]; names = [names[i] for i in keep]; XE = [XE[i] for i in keep]; XC = [XC[i] for i in keep]
y = np.array([lab[n] for n in names])
cl = {case[n]: lab[n] for n in names}; crng = np.random.default_rng(1234); cf = {}  # fixed patient-level stratified folds (same for all seeds)
for k in range(C):
    cc = sorted(c for c in cl if cl[c] == k); crng.shuffle(cc)
    for j, c in enumerate(cc): cf[c] = j % 5
f = np.array([cf[case[n]] for n in names]); tr = np.where(f != fold)[0]; te = np.where(f == fold)[0]
cat = torch.cat([XC[i] for i in tr]); mu, sd = cat.mean(0), cat.std(0) + 1e-4; XC = [(x - mu) / sd for x in XC]; del cat
print(f'{len(names)} slides, train {len(tr)} test {len(te)} (fold {fold}); test labels {np.bincount(y[te], minlength=C)}; {time.time() - t0:.0f}s', flush=True)

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
    pr = torch.tensor(lg).softmax(1).numpy()
    auc = roc_auc_score(yt, pr[:, 1]) if C == 2 else roc_auc_score(yt, pr, multi_class='ovr')
    return {'acc': float((p == yt).mean()), 'balacc': float(balanced_accuracy_score(yt, p)), 'auc': float(auc)}
L = {}
# full-slide references (expensive: every tile through CONCH)
fu4 = train([XE[i] for i in tr], y[tr], C, 20); L['full_conch'] = predict(fu4, [XE[i] for i in te])
print('full', mets(L['full_conch']), f'{time.time() - t0:.0f}s', flush=True)
# distilled cheap scorer: per-tile subtype logits + attention (ADMIL PriorNet)
with torch.no_grad():
    TE, TA = {}, {}
    for i in range(len(names)):
        z, a = fu4.terms(XE[i].to(DEV)); TE[i] = fu4.c(z).cpu(); TA[i] = a.cpu()  # per-tile class logits, attention
g = nn.Sequential(nn.Linear(XC[0].shape[1], 256), nn.ReLU(), nn.Dropout(.1), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, C + 1)).to(DEV)
opt = torch.optim.AdamW(g.parameters(), 5e-4, weight_decay=1e-4); es = torch.cat([TE[i] for i in tr]).std().item()
for ep in range(20):
    g.train()
    for i in rng.permutation(tr):
        o = g(XC[i].to(DEV)); p = TA[i].to(DEV).softmax(0)
        loss = ((o[:, :C] - TE[i].to(DEV) / es) ** 2).mean() + (p * (p.clamp_min(1e-12).log() - o[:, C].log_softmax(0))).sum()
        opt.zero_grad(); loss.backward(); opt.step()
g.eval()
with torch.no_grad(): G = {i: g(XC[i].to(DEV)).cpu() for i in range(len(names))}

def students(i):
    e = G[i][:, :C]; k = min(Kc, len(e)); S = []
    for c in range(C):
        oth = torch.cat([e[:, :c], e[:, c + 1:]], 1).max(1).values; S.append((e[:, c] - oth).topk(k).indices)
    return S
def present(i, S):
    rows = []
    for lv, idx in enumerate(S):
        oh = torch.zeros(len(idx), C); oh[:, lv] = 1
        rows.append(torch.cat([XE[i][idx], oh], 1))
    return torch.cat(rows)
@torch.no_grad()
def wb_replace(m, i, S, lv, keep_flags=True):  # student `lv` replaced by a white-box attacker: tiles whose single-tile teacher score for `lv` is highest
    n = len(XE[i]); oh = torch.zeros(n, C); oh[:, lv] = float(keep_flags)
    X = torch.cat([XE[i], oh], 1).to(DEV); z, a = m.terms(X); s = Fn.log_softmax(m.c(z), 1)[:, lv]
    idx = s.topk(min(Kc, n)).indices.cpu(); S2 = list(S); S2[lv] = idx; return S2
SS = {i: students(i) for i in range(len(names))}
# teacher: warm start on random tiles with random presenter labels, then 3 rounds of clean + one-liar (white-box) transcripts
def rand_pres(i):
    n = len(XE[i]); return [torch.randperm(n)[:min(Kc, n)] for _ in range(C)]
def run_teacher(keep_flags, tag):
    pr = (lambda i, S: present(i, S)) if keep_flags else (lambda i, S: (lambda X: torch.cat([X[:, :-C], torch.zeros(len(X), C)], 1))(present(i, S)))
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
BONLY = os.environ.get('BONLY')  # budget-matched single selector only (skips the teachers)
if not BONLY: run_teacher(True, 'teach'); run_teacher(False, 'noflag')
# ADMIL-style: top-K (K = C*Kc) by distilled attention -> full 4-class teacher; compromised selector: white-box toward a wrong level
K = C * Kc
L['admil'] = predict(fu4, [XE[i][G[i][:, C].topk(min(K, len(XE[i]))).indices] for i in te])
# mechanism: same frozen full model, same budget K, only the selection rule differs (single attention top-K vs per-subtype advocates)
AD = {i: G[i][:, C].topk(min(K, len(XE[i]))).indices for i in te}; US = {i: torch.cat(SS[i]).unique() for i in te}
L['percls_full'] = predict(fu4, [XE[i][US[i]] for i in te])
# per-slide evidence analysis from the full model (defined independently of our method): per-tile subtype = argmax of fu4's
# per-tile class logits; heterogeneity = 1 - attention mass of the dominant per-tile subtype; coverage = #subtypes present in a selection
het, cov_ad, cov_us, ru_ad, ru_us = [], [], [], [], []
for i in te:
    tc = TE[i].argmax(1); a = TA[i].softmax(0); m = torch.stack([a[tc == c].sum() for c in range(C)])
    het.append(float(1 - m.max())); ru = int(torch.tensor(L['full_conch'][list(te).index(i)]).topk(2).indices[1])
    cov_ad.append(len(tc[AD[i]].unique())); cov_us.append(len(tc[US[i]].unique()))
    ru_ad.append(int((tc[AD[i]] == ru).any())); ru_us.append(int((tc[US[i]] == ru).any()))
EXTRA = dict(het=np.array(het), cov_admil=np.array(cov_ad), cov_ours=np.array(cov_us), runnerup_admil=np.array(ru_ad), runnerup_ours=np.array(ru_us))
print('coverage: subtypes in selection admil %.2f ours %.2f; runner-up evidence present admil %.2f ours %.2f; mean het %.3f' %
      (np.mean(cov_ad), np.mean(cov_us), np.mean(ru_ad), np.mean(ru_us), np.mean(het)), flush=True)
@torch.no_grad()
def admil_wb(i, KK=None):
    KK = KK or K
    z, a = fu4.terms(XE[i].to(DEV)); lp = Fn.log_softmax(fu4.c(z), 1).cpu(); best = None
    for lv in range(C):
        if lv == y[i]: continue
        v = predict(fu4, [XE[i][lp[:, lv].topk(min(KK, len(lp))).indices]])[0]
        if best is None or Fn.softmax(torch.tensor(v), 0)[y[i]] < Fn.softmax(torch.tensor(best), 0)[y[i]]: best = v
    return best
L['admil_wb_worst'] = np.stack([admil_wb(i) for i in te])
if BONLY:  # single selector at the hybrid presenters' budgets: K (= ours), K + Kc (hybrid-Kc), 2K (hybrid-K); clean and white-box attacked
    R = {}
    for KK in (K, K + Kc, 2 * K):
        R[f'admil_b{KK}'] = mets(predict(fu4, [XE[i][G[i][:, C].topk(min(KK, len(XE[i]))).indices] for i in te]))
        R[f'admil_b{KK}_wb_worst'] = mets(np.stack([admil_wb(i, KK) for i in te])); print(KK, R[f'admil_b{KK}'], R[f'admil_b{KK}_wb_worst'], flush=True)
    json.dump({'res': R, 'info': {'seed': seed, 'fold': fold, 'Kc': Kc, 'K': K}}, open(f'{OUT}/tcga_{CO}_{CH}_k{Kc}_f{fold}_s{seed}_budget.json', 'w'), indent=1); sys.exit(0)
EXT = os.environ.get('EXT'); SUF = ('_ext' + os.environ.get('SUFX', '')) if EXT else ''
if EXT:  # extended robustness experiments (ext.py): adaptive attack, hybrid presenter, independent per-advocate scorers, weak scorer
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import ext
    cx = ext.Ctx(XE=XE, extra=lambda i, idx: torch.zeros(len(idx), 0), E=0, C=C, y=y, tr=tr, te=te, train=train, predict=predict, rng=rng, DEV=DEV)
    def fit_g(sub, sd):  # the cheap scorer above, with its own seed and training slides
        torch.manual_seed(sd); r = np.random.default_rng(sd)
        h = nn.Sequential(nn.Linear(XC[0].shape[1], 256), nn.ReLU(), nn.Dropout(.1), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, C + 1)).to(DEV)
        op = torch.optim.AdamW(h.parameters(), 5e-4, weight_decay=1e-4)
        for ep in range(20):
            h.train()
            for i in r.permutation(sub):
                o = h(XC[i].to(DEV)); p = TA[i].to(DEV).softmax(0)
                loss = ((o[:, :C] - TE[i].to(DEV) / es) ** 2).mean() + (p * (p.clamp_min(1e-12).log() - o[:, C].log_softmax(0))).sum()
                op.zero_grad(); loss.backward(); op.step()
        h.eval()
        with torch.no_grad(): return {i: h(XC[i].to(DEV)).cpu() for i in range(len(names))}
    def adv_g(Gd, i):  # students() under scorer Gd
        e = Gd[i][:, :C]; k = min(Kc, len(e))
        return [(e[:, c] - torch.cat([e[:, :c], e[:, c + 1:]], 1).max(1).values).topk(k).indices for c in range(C)]
    top = lambda Gd, i, k: Gd[i][:, C].topk(min(k, len(XE[i]))).indices
    Gs = [fit_g(rng.choice(tr, len(tr)), 100 + c) for c in range(C)]  # bagged, independently seeded
    wsub = rng.choice(tr, max(3 * C, len(tr) // 10), replace=False); Gw = fit_g(wsub, 200)
    L['admil_weak'] = predict(fu4, [XE[i][top(Gw, i, K)] for i in te])
    N = range(len(names))
    assert all(torch.equal(u, v) for i in list(N)[:20] for u, v in zip(adv_g(G, i), SS[i]))  # refactored rule == students()
    ext.run(cx, [('x_base', {i: SS[i] for i in N}, True),
                 (f'x_hyb{Kc}', {i: SS[i] + [top(G, i, Kc)] for i in N}, True),
                 (f'x_hyb{K}', {i: SS[i] + [top(G, i, K)] for i in N}, True),
                 ('x_indep', {i: [adv_g(Gs[c], i)[c] for c in range(C)] for i in N}, False),
                 ('x_weak', {i: adv_g(Gw, i) for i in N}, False)], L, log=lambda t: print('ext', t, f'{time.time() - t0:.0f}s', flush=True))
res = {k: mets(v) for k, v in L.items()}
hq = np.quantile(EXTRA['het'], [1 / 3, 2 / 3]); grp = np.digitize(EXTRA['het'], hq)  # heterogeneity terciles within the test fold
for k in ['full_conch', 'teach', 'noflag', 'admil', 'percls_full']:
    p = L[k].argmax(1); res[k]['acc_by_het'] = [float((p[grp == g] == y[te][grp == g]).mean()) for g in range(3)]
for k, v in res.items(): print(k, {a: (round(b, 3) if not isinstance(b, list) else [round(x, 3) for x in b]) for a, b in v.items()}, flush=True)
json.dump({'res': res, 'info': {'seed': seed, 'fold': fold, 'cheap': CH, 'Kc': Kc, 'cohort': CO, 'n_train': int(len(tr)), 'n_test': int(len(te)),
           'mean_tiles_test': float(np.mean([len(XE[i]) for i in te])), 'ext': bool(EXT)}}, open(f'{OUT}/tcga_{CO}_{CH}_k{Kc}_f{fold}_s{seed}{SUF}.json', 'w'), indent=1)
np.savez(f'{OUT}/tcga_{CO}_{CH}_k{Kc}_f{fold}_s{seed}{SUF}.npz', y=y[te], names=np.array(names)[te], **L, **EXTRA)
torch.save({'names': [names[i] for i in te], 'students': [SS[i] for i in te], 'admil': [AD[i] for i in te], 'tile_subtype': [TE[i].argmax(1) for i in te],
            'att': [TA[i] for i in te]}, f'{OUT}/tcga_{CO}_{CH}_k{Kc}_f{fold}_s{seed}{SUF}_sel.pt')  # tile indices -> coords in feats/{name}.pt, for case review
print('DONE', f'{time.time() - t0:.0f}s', flush=True)
