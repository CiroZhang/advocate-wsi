"""TCGA (GDC SVS, any cohort dir with raw/ + done.txt) -> tissue tiles (448 px at 0.5 mpp, as qa/tcga_extract.py; no subsampling) ->
expensive features: CONCH ViT-B/16 image embedding (pre-projection, 512) on the 448 px tile;
cheap features: timm ImageNet efficientnet_b0 / resnet18 on the same tile read at pyramid level 2 (4x downsampled; 112 px).
Array job: task i of n handles slides[i::n] among fully downloaded slides (raw/done.txt). Re-runnable (skips finished slides).
Output data/<cohort>/feats/{slide}.pt  usage: python tcga_feat.py TASK NTASKS COHORT_DIR = {coords (n,2) level-0 xy, side0, mpp, conch fp16, b0 fp16, r18 fp16, timing}."""
import os, sys, time, numpy as np, torch, openslide, timm
ROOT = os.environ.get('ADV_ROOT', '/scratch/ciro/cvpr_lab')  # data/ and studentB_debate/ live under here
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
R = ROOT; D = sys.argv[3]; OUT = f"{D}/feats"; os.makedirs(OUT, exist_ok=True)
T = 448; MINT = 0.4; BS = 128
i, n = int(sys.argv[1]), int(sys.argv[2])
from conch.open_clip_custom import create_model_from_pretrained
conch, _ = create_model_from_pretrained('conch_ViT-B-16', checkpoint_path=f"{R}/models/conch/pytorch_model.bin")
conch = conch.cuda().eval().half()
cheap = {e: timm.create_model(m, pretrained=True, num_classes=0).cuda().eval().half() for e, m in (('b0', 'efficientnet_b0.ra_in1k'), ('r18', 'resnet18.a1_in1k'))}
CM = torch.tensor([0.48145466, 0.4578275, 0.40821073], device='cuda', dtype=torch.half).view(1, 3, 1, 1)
CS = torch.tensor([0.26862954, 0.26130258, 0.27577711], device='cuda', dtype=torch.half).view(1, 3, 1, 1)
IM = torch.tensor([0.485, 0.456, 0.406], device='cuda', dtype=torch.half).view(1, 3, 1, 1)
IS = torch.tensor([0.229, 0.224, 0.225], device='cuda', dtype=torch.half).view(1, 3, 1, 1)

def mpp_of(s):
    for k in ('openslide.mpp-x', 'aperio.MPP'):
        if k in s.properties: return float(s.properties[k])
    return 0.25

def tiles(s, side0):
    W, H = s.dimensions; ds = 32
    th = np.asarray(s.get_thumbnail((W // ds, H // ds)).convert('RGB')).astype(np.float32); sx, sy = W / th.shape[1], H / th.shape[0]
    mx, mn = th.max(2), th.min(2); m = ((mx - mn) / (mx + 1e-6) > 0.07) & (mx < 245) & (mx > 30)
    xy = []
    for y in range(0, H - side0 + 1, side0):
        for x in range(0, W - side0 + 1, side0):
            a, b, c, d = int(y / sy), int((y + side0) / sy), int(x / sx), int((x + side0) / sx)
            if m[a:max(b, a + 1), c:max(d, c + 1)].mean() >= MINT: xy.append((x, y))
    return np.array(xy, dtype=np.int64).reshape(-1, 2)

def read(s, xy, lvl, side_l, out):
    im = s.read_region((int(xy[0]), int(xy[1])), lvl, (side_l, side_l)).convert('RGB')
    return np.asarray(im.resize((out, out), Image.BILINEAR) if side_l != out else im)

done = [l.strip().split('/')[-1][:-4] for l in open(f"{D}/done.txt") if l.strip()]
slides = sorted(set(done))[i::n]; t0 = time.time()
for sid in slides:
    o = f"{OUT}/{sid}.pt"
    if os.path.exists(o): continue
    try: s = openslide.OpenSlide(f"{D}/raw/{sid}.svs")
    except Exception as e: print('BAD', sid, repr(e)[:100], flush=True); continue
    if s.level_downsamples[-1] < 8: print('BAD', sid, 'no pyramid (levels %s)' % [round(x) for x in s.level_downsamples], flush=True); continue  # full-res decode would OOM
    try:
        mpp = mpp_of(s); side0 = int(round(T * 0.5 / mpp)); lv = s.get_best_level_for_downsample(side0 / 112 + 1e-3); dsl = s.level_downsamples[lv]  # coarsest level still >= 112 px per tile
        side_l = max(1, int(round(side0 / dsl)))
        ta = time.time(); xy = tiles(s, side0); tm = {'tiling_s': time.time() - ta, 'n': int(len(xy))}
        F = {'conch': [], 'b0': [], 'r18': []}; tm.update(read_l0_s=0., read_l2_s=0., enc_conch_s=0., enc_b0_s=0., enc_r18_s=0.)
        with torch.no_grad(), ThreadPoolExecutor(8) as ex:
            for b in range(0, len(xy), BS):
                c = xy[b:b + BS]
                t = time.time(); x0 = np.stack(list(ex.map(lambda q: read(s, q, 0, side0, T), c))); tm['read_l0_s'] += time.time() - t
                t = time.time(); x2 = np.stack(list(ex.map(lambda q: read(s, q, lv, side_l, 112), c))); tm['read_l2_s'] += time.time() - t
                x0 = torch.from_numpy(x0).cuda().permute(0, 3, 1, 2).half().div(255); x2 = torch.from_numpy(x2).cuda().permute(0, 3, 1, 2).half().div(255)
                torch.cuda.synchronize(); t = time.time(); F['conch'].append(conch.encode_image(x0.sub(CM).div(CS), proj_contrast=False, normalize=False).cpu())
                torch.cuda.synchronize(); tm['enc_conch_s'] += time.time() - t
                for e, m in cheap.items():
                    t = time.time(); F[e].append(m(x2.sub(IM).div(IS)).cpu()); torch.cuda.synchronize(); tm[f'enc_{e}_s'] += time.time() - t
        out = {k: (torch.cat(v).half() if v else torch.zeros(0, 1, dtype=torch.half)) for k, v in F.items()}
        torch.save({'coords': torch.from_numpy(xy), 'side0': side0, 'mpp': mpp, 'level_cheap': lv, **out, 'timing': tm}, o)
        print(sid, len(xy), side0, {k: round(v, 1) for k, v in tm.items()}, f'{time.time() - t0:.0f}s', flush=True)
    except Exception as e: print('BAD', sid, repr(e)[:100], flush=True)
print('DONE', flush=True)
