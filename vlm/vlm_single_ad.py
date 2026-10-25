"""Judge-aware adaptive attack on the SINGLE selector (matches the attacker used on the hybrid in vlm_debate3.py ADAPT=1).
The compromised selector controls all n unlabelled 'Zoomed-in view' tiles (same presentation as vlm_debate2 attn/coopn).
Start from attn (top-n tiles by strongest-wrong margin); then greedy coordinate descent: for each slot, try the top-M=8 candidate tiles
by wrong margin not already shown and keep the one that minimises the judge's p(true) on the full presentation. One pass over slots.
env: MODEL, DATA, SRC (cached partial jsonl), TAG. Output: {OUT}/{TAG}_singlead.jsonl"""
import os, sys, json, numpy as np
src = open("/scratch/ciro/cvpr_lab/hr_sel.py").read().split("res = []; t0")[0]
sys.argv = [sys.argv[0]]; exec(src)
SRC = os.environ["SRC"]; cache = {str(json.loads(l)['qid']): json.loads(l) for l in open(SRC)}
M = 8
outp = f"{OUT}/{os.environ['TAG']}_singlead.jsonl"
done = {json.loads(l)['qid'] for l in open(outp)} if os.path.exists(outp) else set()
fo = open(outp, 'a')
for q in load_items():
    qid = str(q['qid'])
    if qid in done or qid not in cache: continue
    c = cache[qid]; im = q['load'](); text = q['text']; y = q['y']; n = c['nopt']
    S = np.array(c['S_glob']); T = tiles(im); assert len(T) == len(S)
    glob = resize_max(im, GLOB_PIX); tim = lambda t: resize_max(im.crop(T[t]), TILE_PIX)
    def unl(ts): return ([glob] + [tim(t) for t in ts], text, ["Overview of the image:"] + ["Zoomed-in view:"] * len(ts))
    kw = max([k for k in range(n) if k != y], key=lambda k: S[:, k].max()); att = [int(t) for t in np.argsort(-(S[:, kw] - S[:, y]))]
    cand = [int(t) for t in np.argsort(-(np.max(np.delete(S, y, 1), 1) - S[:, y]))[:M + n]]
    cur = att[:n]; best = score([unl(cur)], n)[0]; p0 = best.tolist()
    for j in range(n):
        alt = [t for t in cand if t not in cur][:M]
        if not alt: continue
        sets = [cur[:j] + [t] + cur[j + 1:] for t in alt]; sc = score_many([unl(s) for s in sets], n, bs=4); i = int(sc[:, y].argmin())
        if sc[i, y] < best[y]: cur, best = sets[i], sc[i]
    torch.cuda.empty_cache()
    fo.write(json.dumps({'qid': qid, 'y': y, 'nopt': n, 'attn': p0, 'attn_ad': best.tolist(), 'tiles': cur}) + "\n"); fo.flush()
print('DONE', flush=True)
