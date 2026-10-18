"""Stain-shift stress test for the cheap selector: re-read each CAMELYON17 tile at the cheap level (same coords/level as c17_extract),
apply a per-slide random HED stain perturbation (Tellez et al. 2018 style: per-channel scale U[1-s,1+s] and shift U[-s,s] in HED space)
plus brightness U[1-s,1+s], encode with the same timm b0. Only the cheap view is shifted; CONCH features are untouched.
Output data/camelyon17/feats_shift/s{SEV}/{slide}.pt = {b0}. usage: python c17_shift.py TASK NTASKS  (severities 0, .05, .1, .2)"""
import os, sys, zlib, numpy as np, torch, openslide, timm
ROOT = os.environ.get('ADV_ROOT', '/scratch/ciro/cvpr_lab')  # data/ and studentB_debate/ live under here
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
D = ROOT + "/data/camelyon17"; SEVS = [0.0, 0.05, 0.1, 0.2]; BS = 256
i, n = int(sys.argv[1]), int(sys.argv[2])
b0 = timm.create_model('efficientnet_b0.ra_in1k', pretrained=True, num_classes=0).cuda().eval().half()
IM = torch.tensor([0.485, 0.456, 0.406], device='cuda').view(1, 3, 1, 1); IS = torch.tensor([0.229, 0.224, 0.225], device='cuda').view(1, 3, 1, 1)
RGB2HED = torch.linalg.inv(torch.tensor([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11], [0.27, 0.57, 0.78]]))  # skimage hed_from_rgb
HED2RGB = torch.linalg.inv(RGB2HED)
RGB2HED, HED2RGB = RGB2HED.cuda(), HED2RGB.cuda()
def shift(x, a, b, br):  # x (B,3,H,W) in [0,1]
    od = -torch.log(x.clamp(1e-6, 1)).permute(0, 2, 3, 1) @ RGB2HED
    od = od * a + b
    return torch.exp(-(od @ HED2RGB)).permute(0, 3, 1, 2).mul(br).clamp(0, 1)
names = sorted(f[:-3] for f in os.listdir(f'{D}/feats') if f.endswith('.pt'))[i::n]
for sev in SEVS: os.makedirs(f'{D}/feats_shift/s{sev}', exist_ok=True)
for sid in names:
    if all(os.path.exists(f'{D}/feats_shift/s{sev}/{sid}.pt') for sev in SEVS): continue
    d = torch.load(f'{D}/feats/{sid}.pt'); xy = d['coords'].numpy(); lv = d['level_cheap']
    if len(xy) == 0:
        for sev in SEVS: torch.save({'b0': torch.zeros(0, 1280, dtype=torch.half)}, f'{D}/feats_shift/s{sev}/{sid}.pt')
        continue
    s = openslide.OpenSlide(f'{D}/raw/{sid}.tif'); side_l = max(1, int(round(d['side0'] / s.level_downsamples[lv])))
    rd = lambda q: np.asarray(s.read_region((int(q[0]), int(q[1])), lv, (side_l, side_l)).convert('RGB').resize((112, 112), Image.BILINEAR))
    g = np.random.default_rng(zlib.crc32(sid.encode()))
    par = {sev: (torch.tensor(g.uniform(1 - sev, 1 + sev, 3), dtype=torch.float, device='cuda'), torch.tensor(g.uniform(-sev, sev, 3), dtype=torch.float, device='cuda'),
                 float(g.uniform(1 - sev, 1 + sev))) for sev in SEVS}
    F = {sev: [] for sev in SEVS}
    with torch.no_grad(), ThreadPoolExecutor(8) as ex:
        for b in range(0, len(xy), BS):
            x = torch.from_numpy(np.stack(list(ex.map(rd, xy[b:b + BS])))).cuda().permute(0, 3, 1, 2).float().div(255)
            for sev in SEVS:
                xs = shift(x, *par[sev]) if sev > 0 else x
                F[sev].append(b0(xs.sub(IM).div(IS).half()).cpu())
    for sev in SEVS: torch.save({'b0': torch.cat(F[sev]).half()}, f'{D}/feats_shift/s{sev}/{sid}.pt')
    print(sid, len(xy), flush=True)
print('DONE', flush=True)
