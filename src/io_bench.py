"""Slide I/O benchmark (CPU only): time to read every torchmil patch of a slide at level 0 (512x512, the UNI pipeline's input)
vs level 2 (128x128, the cheap students' input), same 8-process pool, warm-up excluded. 10 test slides spread over sizes.
Writes cheap/io_bench.json."""
import json, time, numpy as np, openslide, multiprocessing as mp
import os
ROOT = os.environ.get('ADV_ROOT', '/scratch/ciro/cvpr_lab')  # data/ and studentB_debate/ live under here
R = ROOT + '/data/camelyon16'
def work(args):
    p, cs, lvl, sz = args; s = openslide.OpenSlide(p); t = 0
    for x, y in cs: s.read_region((int(x), int(y)), lvl, (sz, sz)).convert('RGB')
    return len(cs)
def bench(p, c, lvl, sz, pool, W=8):
    chunks = [(p, c[i::W], lvl, sz) for i in range(W)]; t = time.time(); pool.map(work, chunks); return time.time() - t
if __name__ == '__main__':
    te = sorted(l.split(',')[0] for l in open(f'{R}/splits.csv').read().split('\n')[1:] if l and l.endswith('test'))
    sizes = {b: len(np.load(f'{R}/torchmil/coords/{b}.npy')) for b in te}
    pick = [te[i] for i in np.argsort([sizes[b] for b in te])[np.linspace(10, len(te) - 1, 10).astype(int)]]
    out = {}
    with mp.Pool(8) as pool:
        for b in pick:
            p = f'{R}/raw/{b}.tif'; c = np.load(f'{R}/torchmil/coords/{b}.npy')
            bench(p, c[:64], 2, 128, pool)  # warm-up (process start, slide open)
            t2 = bench(p, c, 2, 128, pool); t0 = bench(p, c, 0, 512, pool)
            out[b] = {'n': int(len(c)), 'read_l2_128_s': t2, 'read_l0_512_s': t0}; print(b, out[b], flush=True)
    json.dump(out, open(f'{R}/cheap/io_bench.json', 'w'), indent=1); print('DONE')
