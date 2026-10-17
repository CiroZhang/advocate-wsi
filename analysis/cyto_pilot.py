"""Go/no-go for cervical cytology: can a CHEAP low-magnification view find the tiles holding the annotated abnormal cells?
Per slide: grid of 224 um tiles over cellular area; positive tile = contains the centre of an annotated cell (ASC-H/LSIL/HSIL/unlabelled).
Views: cheap5x = B0 on the tile at ~2 um/px (112 px), cheap2x = B0 at ~4 um/px (56 px -> 112), full = Phikon-v2 at 0.5 um/px (448 -> 224).
Probe: leave-one-slide-out logistic regression; plus an unsupervised 'dark nuclei' score from the cheap view.
Annotations are sparse (not every abnormal cell is boxed), so all ranking metrics are lower bounds.
usage: python cyto_pilot.py feats   # extract (cached in feats/)
       python cyto_pilot.py eval"""
import sys, os, glob, csv, numpy as np, torch, openslide, timm
from PIL import Image
H = os.path.dirname(os.path.abspath(__file__)); S = f'{H}/slides'; A = f'{H}/annot'; F = f'{H}/feats'; os.makedirs(F, exist_ok=True)
UM = 224.0; DEV = 'mps' if torch.backends.mps.is_available() else 'cpu'
def boxes(n):
    return [(r['cls'], float(r['x']) + float(r['w']) / 2, float(r['y']) + float(r['h']) / 2) for r in csv.DictReader(open(f'{A}/{n}.csv'))]
def grid(s, side0):
    lv = s.level_count - 1; ds = s.level_downsamples[lv]; th = np.asarray(s.read_region((0, 0), lv, s.level_dimensions[lv]).convert('RGB')).astype(float)
    cell = (th.min(2) < 200)  # anything not glass-white
    W, Hh = s.dimensions; out = []
    for y in range(0, Hh - side0, side0):
        for x in range(0, W - side0, side0):
            a = cell[int(y / ds):int((y + side0) / ds) + 1, int(x / ds):int((x + side0) / ds) + 1]
            if a.size and a.mean() > 0.02: out.append((x, y))
    return np.array(out)
def feats():
    b0 = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0).eval().to(DEV)
    from transformers import AutoModel
    ph = AutoModel.from_pretrained('owkin/phikon-v2').eval().to(DEV)
    IM = torch.tensor([.485, .456, .406], device=DEV)[:, None, None]; IS = torch.tensor([.229, .224, .225], device=DEV)[:, None, None]
    T = lambda ims, sz: torch.stack([torch.from_numpy(np.asarray(i.convert('RGB').resize((sz, sz)))).permute(2, 0, 1) for i in ims]).float().div(255).to(DEV).sub(IM).div(IS)
    for p in sorted(glob.glob(f'{S}/*.svs')):
        n = os.path.basename(p)[:-4]; o = f'{F}/{n}.npz'
        if os.path.exists(o) or not os.path.exists(f'{A}/{n}.csv'): continue
        s = openslide.OpenSlide(p); mpp = float(s.properties.get('openslide.mpp-x', .2613)); side0 = int(round(UM / mpp)); xy = grid(s, side0)
        l1 = s.get_best_level_for_downsample(side0 / 112 + 1e-3); l2 = s.get_best_level_for_downsample(side0 / 56 + 1e-3)
        r = lambda lv, q: s.read_region((int(q[0]), int(q[1])), lv, (int(side0 / s.level_downsamples[lv]),) * 2)
        c5, c2, fu, dk = [], [], [], []
        with torch.no_grad():
            for i in range(0, len(xy), 64):
                q = xy[i:i + 64]; a5 = [r(l1, t) for t in q]; a2 = [r(l2, t) for t in q]; a0 = [r(0, t) for t in q]
                c5.append(b0(T(a5, 112)).cpu()); c2.append(b0(T([im.resize((56, 56)) for im in a2], 112)).cpu())
                fu.append(ph(pixel_values=T(a0, 224)).last_hidden_state[:, 0].cpu())
                dk += [float((np.asarray(im.convert('L').resize((112, 112))) < 90).mean()) for im in a5]
        bx = boxes(n); pos = np.zeros(len(xy), int); cls = [''] * len(xy)
        for c, cx, cy in bx:
            k = np.where((xy[:, 0] <= cx) & (cx < xy[:, 0] + side0) & (xy[:, 1] <= cy) & (cy < xy[:, 1] + side0))[0]
            if len(k): pos[k[0]] = 1; cls[k[0]] = cls[k[0]] or c
        np.savez(o, xy=xy, side0=side0, c5=torch.cat(c5).numpy(), c2=torch.cat(c2).numpy(), full=torch.cat(fu).numpy(), dark=np.array(dk), pos=pos, cls=np.array(cls))
        print(n, len(xy), 'tiles', pos.sum(), 'positive', flush=True)
def evaluate():
    from sklearn.linear_model import LogisticRegression; from sklearn.preprocessing import StandardScaler; from sklearn.metrics import roc_auc_score
    D = {os.path.basename(f)[:-4]: dict(np.load(f)) for f in sorted(glob.glob(f'{F}/*.npz'))}; D = {k: v for k, v in D.items() if v['pos'].sum() > 0}
    print(f'{len(D)} slides, tiles/slide {np.mean([len(v["pos"]) for v in D.values()]):.0f}, positive tiles {sum(v["pos"].sum() for v in D.values())}')
    res = {}
    for view in ['c2', 'c5', 'full', 'dark']:
        rows = []
        for n, d in D.items():
            if view == 'dark': sc = d['dark']
            else:
                Xtr = np.concatenate([D[m][view] for m in D if m != n]); ytr = np.concatenate([D[m]['pos'] for m in D if m != n])
                ss = StandardScaler().fit(Xtr); sc = LogisticRegression(C=0.05, class_weight='balanced', max_iter=2000).fit(ss.transform(Xtr), ytr).decision_function(ss.transform(d[view]))
            y = d['pos']; o = np.argsort(-sc); N = len(y)
            rows.append(dict(slide=n, npos=int(y.sum()), auc=roc_auc_score(y, sc), hit16=int(y[o[:16]].any()), rec16=y[o[:16]].sum() / y.sum(),
                             rec1pct=y[o[:max(1, N // 100)]].sum() / y.sum(), first=int(np.argmax(y[o])) + 1, N=N))
        res[view] = rows
        print(f'{view:5s} AUROC {np.mean([r["auc"] for r in rows]):.3f}  slides with a positive in top-16 {sum(r["hit16"] for r in rows)}/{len(rows)}  '
              f'recall@16 {np.mean([r["rec16"] for r in rows]):.3f}  recall@top1% {np.mean([r["rec1pct"] for r in rows]):.3f}  median rank of first positive {np.median([r["first"] for r in rows]):.0f}')
    print('\nper slide (npos / tiles | first-positive rank: c2 c5 full dark)')
    for i, n in enumerate(D): print(n, res['c5'][i]['npos'], '/', res['c5'][i]['N'], '|', *[res[v][i]['first'] for v in ['c2', 'c5', 'full', 'dark']])
if __name__ == '__main__': feats() if sys.argv[1] == 'feats' else evaluate()
