"""Attacks on the advocate presenter itself (completes vlm_debate2.py). Same judge, cache and prompt as vlm_debate2 (global 768^2 view always shown).
Threat model = the attacker controls ONE evidence-selection component and knows the label:
  nway_cy   : the advocate of the TRUE option is compromised: it shows the attacker's best tile (max strongest-wrong-option margin) labelled as
              evidence for the true option; the other advocates are honest (argmax S_glob[:, k]).
  nway_drop : the true option's advocate is silenced (shows nothing); the others are honest.
  hyb       : nway + a neutral slot (unlabelled 'Zoomed-in view': the label-free selector's most confident tile), cf. the pathology hybrid presenter.
  hyb_cy    : hyb with the true option's advocate compromised as in nway_cy.
  hyb_cn    : hyb with the neutral slot compromised (shows the attacker's best tile); advocates honest.
  (single selector compromised = vlm_debate2 attn, for comparison.)
env: MODEL, DATA, SRC (cached partial jsonl), TAG. Output: {OUT}/{TAG}_deb3.jsonl"""
import os, sys, json, hashlib, numpy as np
src = open("/scratch/ciro/cvpr_lab/hr_sel.py").read().split("res = []; t0")[0]
sys.argv = [sys.argv[0]]; exec(src)
SRC = os.environ["SRC"]; cache = {str(json.loads(l)['qid']): json.loads(l) for l in open(SRC)}
outp = f"{OUT}/{os.environ['TAG']}_deb3.jsonl"
done = {json.loads(l)['qid'] for l in open(outp)} if os.path.exists(outp) else set()
fo = open(outp, 'a')
for q in load_items():
    qid = str(q['qid'])
    if qid in done or qid not in cache: continue
    c = cache[qid]; im = q['load'](); text = q['text']; y = q['y']; n = c['nopt']
    S = np.array(c['S_glob']); T = tiles(im); assert len(T) == len(S)
    glob = resize_max(im, GLOB_PIX); tim = lambda t: resize_max(im.crop(T[t]), TILE_PIX)
    rng = np.random.default_rng(int(hashlib.md5(qid.encode()).hexdigest()[:8], 16)); order = [int(k) for k in rng.permutation(n)]  # same seed as vlm_debate2
    adv = {k: int(S[:, k].argmax()) for k in range(n)}
    kw = max([k for k in range(n) if k != y], key=lambda k: S[:, k].max()); att = int(np.argmax(S[:, kw] - S[:, y]))
    coop = int(np.exp(S).max(1).argmax())
    def present(tile_of, neutral=None):
        ims, labs = [glob], ["Overview of the image:"]
        for k in order:
            if tile_of.get(k) is None: continue
            ims.append(tim(tile_of[k])); labs.append(f"Evidence presented to support option ({L[k]}):")
        if neutral is not None: ims.append(tim(neutral)); labs.append("Zoomed-in view:")
        return (ims, text, labs)
    items = {'nway_cy': present(adv | {y: att}), 'nway_drop': present(adv | {y: None}), 'hyb': present(adv, coop),
             'hyb_cy': present(adv | {y: att}, coop), 'hyb_cn': present(adv, att)}
    sc = [score([it], n)[0] for it in items.values()]; torch.cuda.empty_cache()
    r = {'qid': qid, 'y': y, 'nopt': n, 'order': order, 'att_tile': att, 'coop_tile': coop, **{k: v.tolist() for k, v in zip(items, sc)}}
    fo.write(json.dumps(r) + "\n"); fo.flush()
print('DONE', flush=True)
