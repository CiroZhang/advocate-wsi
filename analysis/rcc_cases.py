"""Case sheets for pathologist review: for each slide, the tiles the single selector (ADMIL-style) sent to the foundation model and the
tiles each subtype advocate presented (seed 0 run where the slide was in the test fold). Tiles: 448 px at 0.5 mpp, shown at 192 px.
usage: python rcc_cases.py OUTDIR TAG:SLIDEPREFIX [...]   (run where openslide + data are available)"""
import sys, os, glob, torch, numpy as np, openslide
from PIL import Image, ImageDraw
D = "/scratch/ciro/cvpr_lab/data/tcga_rcc"; RES = "/scratch/ciro/cvpr_lab/studentB_debate/results"; out = sys.argv[1]; os.makedirs(out, exist_ok=True)
SEL = [torch.load(f) for f in sorted(glob.glob(f'{RES}/tcga_rcc_b0_k4_f*_s0_sel.pt'))]
S = ['ccRCC advocate', 'pRCC advocate', 'chRCC advocate']; T = 192
for arg in sys.argv[2:]:
    tag, pre = arg.split(':', 1)
    sel = next(s for s in SEL if any(n.startswith(pre) for n in s['names'])); j = next(k for k, n in enumerate(sel['names']) if n.startswith(pre)); name = sel['names'][j]
    d = torch.load(f'{D}/feats/{name}.pt'); xy, side0 = d['coords'].numpy(), d['side0']; sl = openslide.OpenSlide(f'{D}/raw/{name}.svs')
    tile = lambda i: sl.read_region((int(xy[i][0]), int(xy[i][1])), 0, (side0, side0)).convert('RGB').resize((T, T))
    rows = [('single selector (top-12)', sel['admil'][j].tolist())] + [(S[c], sel['students'][j][c].tolist()) for c in range(3)]
    W = 12 * T + 160; im = Image.new('RGB', (W, len(rows) * (T + 6) + 30), 'white'); dr = ImageDraw.Draw(im); dr.text((4, 4), f'{tag}  {name}', fill='black')
    for r, (lab, idx) in enumerate(rows):
        y0 = 30 + r * (T + 6); dr.text((4, y0 + T // 2), lab, fill='black')
        for k, i in enumerate(idx[:12]): im.paste(tile(i), (160 + k * T, y0))
    th = sl.get_thumbnail((1200, 1200)).convert('RGB'); sc = th.size[0] / sl.dimensions[0]; dt = ImageDraw.Draw(th)
    for (lab, idx), col in zip(rows, ['blue', 'green', 'orange', 'red']):
        for i in idx: x, y = xy[i] * sc; dt.rectangle([x, y, x + side0 * sc, y + side0 * sc], outline=col, width=3)
    im.save(f'{out}/{tag}_{pre}_tiles.jpg', quality=88); th.save(f'{out}/{tag}_{pre}_thumb.jpg', quality=85); print(tag, name, flush=True)
