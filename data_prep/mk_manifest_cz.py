"""Manifest for the Certified Zoom split (Killarney, read-only inputs): Phikon 20x f_feat, C16+C17 train -> C16 test (testA) / C17 test (testB).
Labels: C16 data/meta/reference.csv (image,class; positive = class != negative); C17 data17/meta/stages.csv (patient,stage,center; .tif rows).
Slides kept only if feats/<slide>.npz exists (C16 under data/, C17 under data17/)."""
import csv, os
R = "/scratch/ciro/car"; O = "/scratch/ciro/shared_baselines/manifests"; os.makedirs(O, exist_ok=True)
lab, cen, src = {}, {}, {}
for r in csv.DictReader(open(f"{R}/data/meta/reference.csv")):
    s = r['image'].replace('.tif', ''); lab[s] = int(r['class'].strip().lower() != 'negative'); src[s] = ('data', 'camelyon16')
for r in csv.DictReader(open(f"{R}/data17/meta/stages.csv")):
    if not r['patient'].endswith('.tif'): continue
    s = r['patient'][:-4]; lab[s] = int(r['stage'].strip().lower() != 'negative'); cen[s] = r['center']; src[s] = ('data17', 'camelyon17')
def ids(p): return [l.strip().replace('.tif', '') for l in open(f"{R}/{p}") if l.strip()]
splits = [('train', p) for p in ('data/meta/train.txt', 'data/meta/train_extra.txt', 'data17/meta/train17.txt', 'data17/meta/train17_neg.txt')]
splits += [('testA', 'data/meta/test.txt'), ('testB', 'data17/meta/test17.txt')]
n, miss = {}, {}
with open(f"{O}/cz_phikon20.csv", 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['slide', 'path', 'label', 'split', 'dataset', 'center']); seen = set()
    for sp, p in splits:
        if not os.path.exists(f"{R}/{p}"): print('missing list', p); continue
        for s in ids(p):
            if s in seen: continue
            if s not in lab: miss[sp] = miss.get(sp, 0) + 1; continue
            d, ds = src[s]; fp = f"{R}/{d}/feats/{s}.npz"
            if not os.path.exists(fp): miss[sp] = miss.get(sp, 0) + 1; continue
            seen.add(s); n[sp] = n.get(sp, 0) + 1; w.writerow([s, fp, lab[s], sp, ds, cen.get(s, '')])
print('kept', n, 'dropped (no label or feats)', miss)
