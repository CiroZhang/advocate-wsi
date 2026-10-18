"""Cheap-encoder features for the torchmil CAMELYON16 patches (same coords as features_UNI / features_r50bt), from raw slides.
Each torchmil patch is a 512x512 region at level 0 (40x, 0.243 um/px). Cheap students read it at pyramid level 2 (10x) as 128x128.
Encoders (timm, ImageNet weights, frozen): efficientnet_b0 (ADMIL's PriorNet backbone), resnet18, mobilenetv3_large_100.
Writes data/camelyon16/cheap/<enc>/<slide>.npy (fp16, N x d, same row order as coords) and timing/<slide>.json with
per-slide wall-clock: level-2 128px read time (all patches), level-0 512px read time (200-patch sample, scaled), encoder time per encoder.
usage: python cheap_feats.py --part i --nparts n"""
import argparse, json, os, time, numpy as np, torch, timm, openslide
ROOT = os.environ.get('ADV_ROOT', '/scratch/ciro/cvpr_lab')  # data/ and studentB_debate/ live under here
from torch.utils.data import Dataset, DataLoader
ap = argparse.ArgumentParser(); ap.add_argument('--part', type=int, default=0); ap.add_argument('--nparts', type=int, default=1)
ap.add_argument('--workers', type=int, default=8); a = ap.parse_args()
R = ROOT + '/data/camelyon16'; RAW = f'{R}/raw'; O = f'{R}/cheap'
ENCS = {'b0': 'efficientnet_b0.ra_in1k', 'r18': 'resnet18.a1_in1k', 'mnv3': 'mobilenetv3_large_100.ra_in1k'}
for e in ENCS: os.makedirs(f'{O}/{e}', exist_ok=True)
os.makedirs(f'{O}/timing', exist_ok=True)
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

class Patches(Dataset):
    def __init__(s, path, coords, level, size): s.path, s.c, s.level, s.size, s.sl = path, coords, level, size, None
    def __len__(s): return len(s.c)
    def __getitem__(s, i):
        if s.sl is None: s.sl = openslide.OpenSlide(s.path)
        x, y = s.c[i]
        im = s.sl.read_region((int(x), int(y)), s.level, (s.size, s.size)).convert('RGB')
        return torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1)

dev = 'cuda'
models = {e: timm.create_model(n, pretrained=True, num_classes=0).to(dev).eval().half() for e, n in ENCS.items()}
slides = sorted(l.split(',')[0] for l in open(f'{R}/splits.csv').read().split('\n')[1:] if l)[a.part::a.nparts]
done_dl = {l.split()[1][:-4] for l in open(f'{RAW}/dl.log') if l.startswith('ok ')}  # fully downloaded only
print(len(slides), 'slides in part', a.part, len(done_dl), 'downloaded', flush=True)
for b in slides:
    if os.path.exists(f'{O}/timing/{b}.json'): continue
    p = f'{RAW}/{b}.tif'
    if b not in done_dl: print('not downloaded yet', b, flush=True); continue
    try: sl = openslide.OpenSlide(p); assert sl.level_count > 2
    except Exception as ex: print('skip', b, ex, flush=True); continue
    c = np.load(f'{R}/torchmil/coords/{b}.npy'); t0 = time.time()
    # level-2 read of every patch (128 px)
    X = torch.cat(list(DataLoader(Patches(p, c, 2, 128), batch_size=256, num_workers=a.workers)))
    t_read2 = time.time() - t0
    # level-0 512 px read on a sample (the UNI pipeline's I/O), scaled to the slide
    smp = c[np.random.default_rng(0).permutation(len(c))[:200]]; t1 = time.time()
    for _ in DataLoader(Patches(p, smp, 0, 512), batch_size=50, num_workers=a.workers): pass
    t_read0 = (time.time() - t1) * len(c) / len(smp)
    tim = {'n': int(len(c)), 'read_l2_128_s': t_read2, 'read_l0_512_s_est': t_read0}
    for e, m in models.items():
        F = []; torch.cuda.synchronize(); t2 = time.time()
        with torch.no_grad():
            for i in range(0, len(X), 512):
                xb = (X[i:i + 512].to(dev).float() / 255 - MEAN.to(dev)) / STD.to(dev)
                F.append(m(xb.half()).float().cpu())
        torch.cuda.synchronize(); tim[f'enc_{e}_s'] = time.time() - t2
        np.save(f'{O}/{e}/{b}.npy', torch.cat(F).numpy().astype(np.float16))
    json.dump(tim, open(f'{O}/timing/{b}.json', 'w'))
    print(b, {k: round(v, 2) for k, v in tim.items()}, flush=True)
print('DONE', flush=True)
