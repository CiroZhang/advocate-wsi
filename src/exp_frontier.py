"""Matched-compute baselines for the accuracy-vs-cost frontier (same folds/labels as exp_tcga.py / exp_c17.py):
  cheap_full : gated ABMIL on the cheap B0 features of ALL tiles (no foundation model; cheaper than ours)
  rand_K     : CONCH ABMIL trained on all tiles, tested on K random tiles (K = ours' budget and 4x/16x it)
  rand_K_tr  : CONCH ABMIL trained AND tested on K random tiles (fresh draw per epoch)
usage: python exp_frontier.py SEED FOLD DATASET(rcc|c17|c16|nsclc)"""
import sys, os, json, time, csv, numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn
ROOT = os.environ.get('ADV_ROOT', '/scratch/ciro/cvpr_lab')  # data/ and studentB_debate/ live under here
from sklearn.metrics import cohen_kappa_score, balanced_accuracy_score, roc_auc_score
seed, fold, DS = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]; DEV = 'cuda' if torch.cuda.is_available() else 'cpu'; OUT = ROOT + "/studentB_debate/results"
torch.manual_seed(seed); rng = np.random.default_rng(seed); t0 = time.time()
if DS == 'rcc':
    D = ROOT + "/data/tcga_rcc"; PR = ['TCGA-KIRC', 'TCGA-KIRP', 'TCGA-KICH']; C = 3; KB = 12
    lab, case = {}, {}
    for r in csv.DictReader(open(f'{D}/manifest.tsv'), delimiter='\t'): lab[r['file_name'][:-4]] = PR.index(r['project']); case[r['file_name'][:-4]] = r['case']
    names = sorted(n for n in lab if n.split('-')[3][:2] == '01' and os.path.exists(f'{D}/feats/{n}.pt'))
elif DS == 'c16':  # official split (fold ignored); UNI features as exp_distill.py
    D = ROOT + "/data/camelyon16/torchmil"; CD = ROOT + "/data/camelyon16/cheap/b0"; C = 2; KB = 16
    rows = list(csv.DictReader(open(f'{D}/splits.csv'))); spl = {r['bag_name']: r['split'] for r in rows}
    names = [r['bag_name'] for r in rows if os.path.exists(f"{D}/features/{r['bag_name']}.npy") and os.path.exists(f"{CD}/{r['bag_name']}.npy")]
    lab = {n: int(np.load(f'{D}/labels/{n}.npy').reshape(-1)[0]) for n in names}
elif DS == 'nsclc':
    D = ROOT + "/data/tcga_nsclc"; PR = ['TCGA-LUAD', 'TCGA-LUSC']; C = 2; KB = 8
    lab, case = {}, {}
    for r in csv.DictReader(open(f'{D}/manifest.tsv'), delimiter='\t'): lab[r['file_name'][:-4]] = PR.index(r['project']); case[r['file_name'][:-4]] = r['case']
    names = sorted(n for n in lab if n.split('-')[3][:2] == '01' and os.path.exists(f'{D}/feats/{n}.pt'))
else:
    D = ROOT + "/data/camelyon17"; C = 4; KB = 16; LV = {'negative': 0, 'itc': 1, 'micro': 2, 'macro': 3}; lab, cen = {}, {}
    for r in csv.DictReader(open(f'{D}/stages.csv')):
        if r['patient'].endswith('.tif'): lab[r['patient'][:-4]] = LV[r['stage'].strip().lower()]; cen[r['patient'][:-4]] = int(r['center'])
    names = sorted(n for n in lab if os.path.exists(f'{D}/feats/{n}.pt'))
XE, XC = [], []
for n in names:
    if DS == 'c16': XE.append(torch.from_numpy(np.load(f'{D}/features/{n}.npy')).float()); XC.append(torch.from_numpy(np.load(f'{CD}/{n}.npy').astype(np.float32))); continue
    d = torch.load(f'{D}/feats/{n}.pt'); XE.append(d['conch'].float()); XC.append(d['b0'].float())
keep = [i for i in range(len(names)) if len(XE[i]) > 0]; names = [names[i] for i in keep]; XE = [XE[i] for i in keep]; XC = [XC[i] for i in keep]
y = np.array([lab[n] for n in names])
if DS == 'c16': f = np.array([0 if spl[n] == 'test' else 1 for n in names]); fold = 0
elif DS in ('rcc', 'nsclc'):  # identical fold assignment to exp_tcga.py
    cl = {case[n]: lab[n] for n in names}; crng = np.random.default_rng(1234); cf = {}
    for k in range(C):
        cc = sorted(c for c in cl if cl[c] == k); crng.shuffle(cc)
        for j, c in enumerate(cc): cf[c] = j % 5
    f = np.array([cf[case[n]] for n in names])
else: f = np.array([cen[n] for n in names])
tr = np.where(f != fold)[0]; te = np.where(f == fold)[0]
cat = torch.cat([XC[i] for i in tr]); mu, sd = cat.mean(0), cat.std(0) + 1e-4; XC = [(x - mu) / sd for x in XC]; del cat
print(f'{DS}: {len(names)} slides, train {len(tr)} test {len(te)} (fold {fold}); {time.time() - t0:.0f}s', flush=True)
class ABMIL(nn.Module):
    def __init__(s, d, o, h=128, drop=.1):
        super().__init__(); s.f = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Dropout(drop)); s.a = nn.Linear(h, h); s.b = nn.Linear(h, h); s.w = nn.Linear(h, 1); s.c = nn.Linear(h, o)
    def forward(s, x): z = s.f(x); a = s.w(torch.tanh(s.a(z)) * torch.sigmoid(s.b(z))).squeeze(-1); return s.c((a.softmax(0)[:, None] * z).sum(0))
def sub(x, k): return x[torch.randperm(len(x))[:k]] if len(x) > k else x
def train(get, ys, epochs=20):
    m = ABMIL(get(tr[0]).shape[1], C).to(DEV); opt = torch.optim.AdamW(m.parameters(), 2e-4, weight_decay=1e-4)
    w = torch.tensor(len(ys) / (C * np.maximum(np.bincount(ys, minlength=C), 1)), dtype=torch.float, device=DEV)
    for _ in range(epochs):
        m.train()
        for k in rng.permutation(len(tr)):
            loss = Fn.cross_entropy(m(get(tr[k]).to(DEV))[None], torch.tensor([ys[k]], device=DEV), weight=w); opt.zero_grad(); loss.backward(); opt.step()
    return m.eval()
@torch.no_grad()
def pred(m, get): return np.stack([m(get(i).to(DEV)).cpu().numpy() for i in te])
def mets(lg):
    p = lg.argmax(1); yt = y[te]; pr = torch.tensor(lg).softmax(1).numpy()
    r = {'acc': float((p == yt).mean()), 'balacc': float(balanced_accuracy_score(yt, p))}
    if DS == 'c17': r['kappa'] = float(cohen_kappa_score(yt, p, weights='quadratic'))
    elif C == 2: r['auc'] = float(roc_auc_score(yt, pr[:, 1]))
    else: r['auc'] = float(roc_auc_score(yt, pr, multi_class='ovr')); r['recall'] = [float((p[yt == c] == c).mean()) for c in range(C)]
    return r
L = {}; ytr = y[tr]
mc = train(lambda i: XC[i], ytr); L['cheap_full'] = pred(mc, lambda i: XC[i])
mf = train(lambda i: XE[i], ytr); L['full_conch'] = pred(mf, lambda i: XE[i])
for K in [KB, 4 * KB, 16 * KB]:
    L[f'rand_{K}'] = pred(mf, lambda i: sub(XE[i], K))
    L[f'rand_{K}_tr'] = pred(train(lambda i: sub(XE[i], K), ytr), lambda i: sub(XE[i], K))
res = {k: mets(v) for k, v in L.items()}
for k, v in res.items(): print(k, v, flush=True)
json.dump({'res': res, 'info': {'seed': seed, 'fold': fold, 'ds': DS, 'KB': KB}}, open(f'{OUT}/frontier_{DS}_f{fold}_s{seed}.json', 'w'), indent=1)
print('DONE', f'{time.time() - t0:.0f}s', flush=True)
