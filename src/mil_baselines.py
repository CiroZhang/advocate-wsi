"""Published MIL baselines (torchmil reference implementations) on precomputed patch features, shared by WSI_debate and Certified Zoom.

Input: a manifest CSV with columns slide,path,label,split[,dataset,center]; split in {train, testA, testB, ...} (any split not 'train'
is evaluated). 15% of train (stratified, seeded) is held out for epoch selection by validation AUC.
Features: .npy (N, d) or .npz (--key array). Output in OUT/<method>/:
  pred_s{seed}.csv : slide,dataset,center,label,split,score,decision,patches_read,patches_total,ms_aggregator,gpu
  summary_s{seed}.json : per split AUC / balanced accuracy / sensitivity / specificity, best epoch, train time
ms_aggregator = CUDA-synchronised forward time of the aggregator on one slide (features on GPU), median of 3 repeats.
Encoder time is added downstream from per-encoder throughput (patches_read / patches_per_s); full-read baselines read every patch.
usage: python mil_baselines.py --manifest M.csv --out DIR --method abmil --seed 0 [--key f_feat] [--epochs 30]"""
import argparse, csv, json, os, time, numpy as np, torch
from sklearn.metrics import roc_auc_score
from torchmil.models import ABMIL, CLAM_SB, TransMIL, DSMIL, DTFDMIL

ap = argparse.ArgumentParser()
ap.add_argument('--manifest', required=True); ap.add_argument('--out', required=True); ap.add_argument('--method', required=True)
ap.add_argument('--seed', type=int, default=0); ap.add_argument('--key', default=None); ap.add_argument('--epochs', type=int, default=30)
ap.add_argument('--lr', type=float, default=2e-4); ap.add_argument('--wd', type=float, default=1e-4); ap.add_argument('--val_frac', type=float, default=0.15)
a = ap.parse_args()
torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed); dev = 'cuda'
out = os.path.join(a.out, a.method); os.makedirs(out, exist_ok=True)
rows = list(csv.DictReader(open(a.manifest)))
t0 = time.time()
def load(p):
    x = np.load(p)
    if p.endswith('.npz'): x = x[a.key]
    return torch.from_numpy(np.asarray(x, dtype=np.float32)).half()
X = [load(r['path']) for r in rows]; Y = np.array([int(r['label']) for r in rows]); d = X[0].shape[1]
print(f'loaded {len(rows)} bags, dim {d}, {time.time() - t0:.0f}s', flush=True)
tr = np.array([i for i, r in enumerate(rows) if r['split'] == 'train'])
va = np.concatenate([rng.permutation(tr[Y[tr] == c])[:max(1, int(a.val_frac * (Y[tr] == c).sum()))] for c in (0, 1)])
tr = np.setdiff1d(tr, va); evals = sorted({r['split'] for r in rows} - {'train'})

MK = {'abmil': lambda: ABMIL(in_shape=(d,), gated=True), 'clam_sb': lambda: CLAM_SB(in_shape=(d,), gated=True,  # shared fc as in the original CLAM; without it the instance loss cannot affect the bag head
                                 feat_ext=torch.nn.Sequential(torch.nn.Linear(d, 512), torch.nn.ReLU(), torch.nn.Dropout(0.25))),
      'transmil': lambda: TransMIL(in_shape=(d,)), 'dsmil': lambda: DSMIL(in_shape=(d,)), 'dtfdmil': lambda: DTFDMIL(in_shape=(d,))}
model = MK[a.method]().to(dev)
NEEDS_MASK = a.method in ('dtfdmil', 'clam_sb')  # torchmil 1.0.2 DTFDMIL crashes on mask=None
if a.method == 'clam_sb':  # torchmil 1.0.2 SmoothTop1SVM/BCE get (k,2) logits vs (k,1) targets; use CE as in CLAM's ce option
    class InstCE(torch.nn.Module):
        def forward(self, lg, t): return torch.nn.functional.cross_entropy(lg, t.reshape(-1).long())
    model.inst_loss_fn = InstCE()
def M(x): return torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)

def fwd(x):  # bag logit for one slide
    o = model(x[None], M(x[None])) if NEEDS_MASK else model(x[None])
    return (o[0] if isinstance(o, tuple) else o).reshape(-1)[0]

@torch.no_grad()
def scores(ids, timing=False):
    model.eval(); s, ms = [], []
    for i in ids:
        x = X[i].to(dev).float()
        if timing:
            ts = []
            for _ in range(3):
                torch.cuda.synchronize(); t = time.perf_counter(); v = fwd(x); torch.cuda.synchronize(); ts.append((time.perf_counter() - t) * 1e3)
            ms.append(float(np.median(ts)))
        else: v = fwd(x)
        s.append(float(v))
    return np.array(s), np.array(ms)

def mets(s, y, thr=0.0):
    p = (s > thr).astype(int); sens = float(np.mean(p[y == 1] == 1)) if (y == 1).any() else float('nan'); spec = float(np.mean(p[y == 0] == 0)) if (y == 0).any() else float('nan')
    return {'auc': float(roc_auc_score(y, s)) if len(set(y)) == 2 else float('nan'), 'balacc': float(np.nanmean([sens, spec])), 'sens': sens, 'spec': spec, 'n': int(len(y))}

opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.wd); best, best_ep, best_state = -1, -1, None
pos = Y[tr].mean(); w = torch.tensor([(1 - pos) / pos], device=dev)  # class-balanced like our own judges
for ep in range(a.epochs):
    model.train()
    for i in rng.permutation(tr):
        x = X[i].to(dev).float()[None]; yy = torch.tensor([float(Y[i])], device=dev)
        _, ld = model.compute_loss(yy, x, M(x)) if NEEDS_MASK else model.compute_loss(yy, x)
        loss = sum(v for v in ld.values())
        if Y[i] == 1: loss = loss * w  # up-weight the minority (tumor) class
        opt.zero_grad(); loss.backward(); opt.step()
    sv, _ = scores(va); m = mets(sv, Y[va])['auc']
    if m > best: best, best_ep, best_state = m, ep, {k: v.detach().clone() for k, v in model.state_dict().items()}
    print(f'ep {ep} val auc {m:.3f} {time.time() - t0:.0f}s', flush=True)
model.load_state_dict(best_state); train_s = time.time() - t0
gpu = torch.cuda.get_device_name(0); summ = {'method': a.method, 'seed': a.seed, 'best_epoch': best_ep, 'val_auc': best, 'train_s': train_s, 'gpu': gpu, 'manifest': a.manifest}
with open(os.path.join(out, f'pred_s{a.seed}.csv'), 'w', newline='') as f:
    wr = csv.writer(f); wr.writerow(['slide', 'dataset', 'center', 'label', 'split', 'score', 'decision', 'patches_read', 'patches_total', 'ms_aggregator', 'gpu'])
    sv, _ = scores(va); cand = np.concatenate([[-np.inf], np.sort(sv)])  # threshold maximising val BA (reported alongside logit>0)
    thr = float(cand[int(np.argmax([mets(sv, Y[va], t)['balacc'] for t in cand]))]); summ['val_thr'] = thr
    for i, v in zip(va, sv):
        r = rows[i]; n = len(X[i]); wr.writerow([r['slide'], r.get('dataset', ''), r.get('center', ''), r['label'], 'val', f'{v:.5f}', int(v > 0), n, n, '', gpu])
    for sp in evals:
        ids = np.array([i for i, r in enumerate(rows) if r['split'] == sp])
        s, ms = scores(ids, timing=True); summ[sp] = mets(s, Y[ids]) | {'ms_aggregator_median': float(np.median(ms))}; summ[sp + '_valthr'] = mets(s, Y[ids], thr)
        for i, v, t in zip(ids, s, ms):
            r = rows[i]; n = len(X[i])
            wr.writerow([r['slide'], r.get('dataset', ''), r.get('center', ''), r['label'], sp, f'{v:.5f}', int(v > 0), n, n, f'{t:.3f}', gpu])
        print(sp, summ[sp], flush=True)
json.dump(summ, open(os.path.join(out, f'summary_s{a.seed}.json'), 'w'), indent=1)
print('DONE', flush=True)
