"""Label-free crop debate + matched-budget controls (reviewer #2/#3/#4). Reuses hr_sel.py helpers and the cached tile scores (SRC partial).
Per question (judge = same VLM, global 768^2 view always shown):
  nway   : every option gets an advocate showing its best tile (argmax S_glob[:,k]); labelled; random order. No label used.
  top2   : the two options most likely under the global view each show their best tile (1v1); labelled; random order. No label used.
  coop2 / coopn : cooperative label-free selector, 2 or n tiles with the highest judge confidence, unlabelled (matched crop budget).
  att2 / attn   : worst-case attacker (label-aware), 2 or n tiles maximizing the strongest wrong option's margin, unlabelled.
env: MODEL, DATA, SRC (cached partial jsonl), TAG."""
import os, sys, json, hashlib, numpy as np
src = open("/scratch/ciro/cvpr_lab/hr_sel.py").read().split("res = []; t0")[0]
sys.argv = [sys.argv[0]]; exec(src)
SRC = os.environ["SRC"]; cache = {str(json.loads(l)['qid']): json.loads(l) for l in open(SRC)}
outp = f"{OUT}/{os.environ['TAG']}_deb2.jsonl"
done = {json.loads(l)['qid'] for l in open(outp)} if os.path.exists(outp) else set()
fo = open(outp, 'a')
for q in load_items():
    qid = str(q['qid'])
    if qid in done or qid not in cache: continue
    c = cache[qid]; im = q['load'](); text = q['text']; y = q['y']; n = c['nopt']
    S = np.array(c['S_glob']); G = np.array(c['base']['global_only']); T = tiles(im); assert len(T) == len(S)
    glob = resize_max(im, GLOB_PIX); tim = lambda t: resize_max(im.crop(T[t]), TILE_PIX)
    rng = np.random.default_rng(int(hashlib.md5(qid.encode()).hexdigest()[:8], 16))
    adv = {k: int(S[:, k].argmax()) for k in range(n)}
    def labelled(opts):
        o = list(rng.permutation(opts))
        return ([glob] + [tim(adv[k]) for k in o], text, ["Overview of the image:"] + [f"Evidence presented to support option ({L[k]}):" for k in o]), [int(k) for k in o]
    def unl(ts): return ([glob] + [tim(t) for t in ts], text, ["Overview of the image:"] + ["Zoomed-in view:"] * len(ts))
    conf = np.exp(S).max(1); coop = list(np.argsort(-conf))
    kw = max([k for k in range(n) if k != y], key=lambda k: S[:, k].max()); marg = S[:, kw] - S[:, y]; att = list(np.argsort(-marg))
    top2 = [int(k) for k in np.argsort(-G)[:2]]
    b_nway, o_nway = labelled(list(range(n))); b_top2, o_top2 = labelled(top2)
    items = {'nway': b_nway, 'top2': b_top2, 'coop2': unl(coop[:2]), 'coopn': unl(coop[:n]), 'att2': unl(att[:2]), 'attn': unl(att[:n])}
    sc = [score([it], n)[0] for it in items.values()]; torch.cuda.empty_cache()
    r = {'qid': qid, 'y': y, 'nopt': n, 'top2': top2, 'top2_has_y': y in top2, 'order_nway': o_nway, 'order_top2': o_top2,
         **{k: v.tolist() for k, v in zip(items, sc)}}
    fo.write(json.dumps(r) + "\n"); fo.flush()
print('DONE', flush=True)
