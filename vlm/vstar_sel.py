"""Paper A, natural-image track: "whoever zooms decides" on V*-Bench with Qwen2.5-VL-7B.
Judge = VLM answering the MCQ from evidence crops (optionally + a low-res global view). Exact option-letter scoring.
Phase 1 (cache): for every question, score every candidate tile (grid, 2 scales) alone and with the global view.
Phase 2 (selectors): GT-box crop, random, VLM self-grounding zoom (natural agent), confirmation-seeking zoom
(grounding the wrong option's sentence), adversarial query attack (best wrong tile; and 'hide' = excluding GT-overlapping tiles),
and 1-vs-1 crop debate (each side shows the tile best supporting its option; judge sees both crops labelled).
usage: python vstar_sel.py [limit]"""
import os, sys, json, re, math, time, random, numpy as np, torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
R = "/scratch/ciro/cvpr_lab/hf"; VS = f"{R}/vstar_bench"; OUT = "/scratch/ciro/cvpr_lab/paperA_vstar"; os.makedirs(OUT, exist_ok=True)
LIM = int(sys.argv[1]) if len(sys.argv) > 1 else 10**9
L = "ABCD"; TILE_PIX = 448 * 448; GLOB_PIX = 768 * 768
torch.manual_seed(0); random.seed(0)
proc = AutoProcessor.from_pretrained(f"{R}/Qwen2.5-VL-7B-Instruct", min_pixels=128 * 28 * 28, max_pixels=1280 * 28 * 28)
proc.tokenizer.padding_side = 'left'
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(f"{R}/Qwen2.5-VL-7B-Instruct", torch_dtype=torch.bfloat16, device_map="cuda").eval()
LID = [proc.tokenizer.encode(c, add_special_tokens=False)[0] for c in L]

def resize_max(im, pix):
    w, h = im.size; s = min(1.0, math.sqrt(pix / (w * h))); return im.resize((max(28, int(w * s)), max(28, int(h * s)))) if s < 1 else im

def msgs(images, text, labels=None):
    c = []
    for i, im in enumerate(images):
        if labels: c.append({"type": "text", "text": labels[i]})
        c.append({"type": "image"})
    c.append({"type": "text", "text": text}); return [{"role": "user", "content": c}]

@torch.no_grad()
def score(batch, nopt):
    """batch: list of (images, text, labels) -> (B, nopt) log-probs over option letters."""
    texts = [proc.apply_chat_template(msgs(ims, t, lb), add_generation_prompt=True) for ims, t, lb in batch]
    ims = [im for b in batch for im in b[0]]
    enc = proc(text=texts, images=ims if ims else None, return_tensors="pt", padding=True).to("cuda")
    lg = model(**enc).logits[:, -1, LID[:nopt]].float()
    return torch.log_softmax(lg, -1).cpu().numpy()

def score_many(items, nopt, bs=8):
    return np.concatenate([score(items[i:i + bs], nopt) for i in range(0, len(items), bs)]) if items else np.zeros((0, nopt))

@torch.no_grad()
def ground(im, phrase):
    """Natural zoom: ask the VLM itself where to look. Returns box in original coords or None."""
    small = resize_max(im, 1280 * 28 * 28)
    t = proc.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": f"Locate {phrase} in the image. Output its bounding box as JSON: [x1, y1, x2, y2]."}]}], add_generation_prompt=True)
    enc = proc(text=[t], images=[small], return_tensors="pt").to("cuda")
    out = model.generate(**enc, max_new_tokens=48, do_sample=False)
    s = proc.tokenizer.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True)
    m = re.search(r"\[\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\]", s)
    nums = [float(x) for x in m.groups()] if m else []
    if len(nums) < 4: return None, s
    gh, gw = (enc.image_grid_thw[0, 1].item() * 14, enc.image_grid_thw[0, 2].item() * 14)
    sx, sy = im.size[0] / gw, im.size[1] / gh
    return [nums[0] * sx, nums[1] * sy, nums[2] * sx, nums[3] * sy], s

def crop_around(im, box, minside):
    x1, y1, x2, y2 = box; cx, cy = (x1 + x2) / 2, (y1 + y2) / 2; s = max(minside, x2 - x1, y2 - y1) * 1.0
    W, H = im.size; l = int(min(max(0, cx - s / 2), max(0, W - s))); t = int(min(max(0, cy - s / 2), max(0, H - s)))
    return (l, t, int(min(W, l + s)), int(min(H, t + s)))

def tiles(im):
    W, H = im.size; out = []
    for frac in (0.25, 0.5):
        s = int(min(W, H) * frac); st = s // 2
        for t in range(0, max(1, H - s + 1), st):
            for l in range(0, max(1, W - s + 1), st): out.append((l, t, l + s, t + s))
    return out

def overlap(a, b):  # fraction of b (gt box xywh->xyxy) inside a
    bx = (b[0], b[1], b[0] + b[2], b[1] + b[3])
    iw = max(0, min(a[2], bx[2]) - max(a[0], bx[0])); ih = max(0, min(a[3], bx[3]) - max(a[1], bx[1]))
    return iw * ih / max(1, b[2] * b[3])

qs = [json.loads(l) for l in open(f"{VS}/test_questions.jsonl")][:LIM]
res = []; t0 = time.time()
part = f"{OUT}/vstar_partial.jsonl"
done = {json.loads(l)['qid'] for l in open(part)} if os.path.exists(part) else set()
fo = open(part, 'a')
for qi, q in enumerate(qs):
    if q['question_id'] in done: continue
    im = Image.open(f"{VS}/{q['image']}").convert("RGB"); ann = json.load(open(f"{VS}/{q['image'].replace('.jpg', '.json')}"))
    text = q['text']; nopt = len(re.findall(r"^\([A-D]\)", text, re.M)); y = L.index(q['label'])
    opts = re.findall(r"^\([A-D]\) (.*)$", text, re.M)
    glob = resize_max(im, GLOB_PIX); T = tiles(im)
    gt = ann['bbox']; ov = np.array([max(overlap(tb, g) for g in gt) for tb in T])
    tile_ims = [resize_max(im.crop(tb), TILE_PIX) for tb in T]
    # phase 1: per-tile scores, alone and with the global view
    S_alone = score_many([([ti], text, None) for ti in tile_ims], nopt)
    S_glob = score_many([([glob, ti], text, ["Overview of the image:", "Zoomed-in view:"]) for ti in tile_ims], nopt)
    base = {'full_hi': score([([resize_max(im, 1280 * 28 * 28)], text, None)], nopt)[0], 'global_only': score([([glob], text, None)], nopt)[0]}
    # natural zoom: the VLM grounds the target object(s) and the wrong options
    obj = " and ".join(ann.get('target_object', [])) or "the object asked about"
    bx, raw = ground(im, f"the {obj}")
    zoom = {}
    if bx:
        cb = crop_around(im, bx, int(min(im.size) * 0.25)); zi = resize_max(im.crop(cb), TILE_PIX)
        zoom['self'] = (score([([zi], text, None)], nopt)[0], score([([glob, zi], text, ["Overview of the image:", "Zoomed-in view:"])], nopt)[0], max(overlap(cb, g) for g in gt))
    for k in range(nopt):
        if k == y: continue
        b2, _ = ground(im, f"the region showing that {opts[k].rstrip('.')}" if len(opts[k].split()) > 2 else f"the {obj} that is {opts[k]}")
        if b2:
            cb = crop_around(im, b2, int(min(im.size) * 0.25)); zi = resize_max(im.crop(cb), TILE_PIX)
            zoom[f'confirm_{k}'] = (score([([zi], text, None)], nopt)[0], score([([glob, zi], text, ["Overview of the image:", "Zoomed-in view:"])], nopt)[0], max(overlap(cb, g) for g in gt))
    # debate 1v1 (vs the strongest wrong option), with global view: each side shows its best tile (query access); judge sees both
    wrong = [k for k in range(nopt) if k != y]
    deb = {}
    for setting, S in (('alone', S_alone), ('glob', S_glob)):
        kw = max(wrong, key=lambda k: S[:, k].max()); th = int(S[:, y].argmax()); tl = int(S[:, kw].argmax())
        for order in ((y, th, kw, tl), (kw, tl, y, th)):
            ims = [tile_ims[order[1]], tile_ims[order[3]]]; lab = [f"Evidence presented to support option ({L[order[0]]}):", f"Evidence presented to support option ({L[order[2]]}):"]
            if setting == 'glob': ims = [glob] + ims; lab = ["Overview of the image:"] + lab
            deb[f'{setting}_{"honestfirst" if order[0] == y else "liarfirst"}'] = score([(ims, text, lab)], nopt)[0]
    r = {'qid': q['question_id'], 'cat': q['category'], 'y': y, 'nopt': nopt, 'n_tiles': len(T), 'gt_overlap': ov.tolist(),
         'S_alone': S_alone.tolist(), 'S_glob': S_glob.tolist(), 'base': {k: v.tolist() for k, v in base.items()},
         'zoom': {k: [v[0].tolist(), v[1].tolist(), v[2]] for k, v in zoom.items()}, 'ground_raw': raw,
         'debate': {k: v.tolist() for k, v in deb.items()}}
    fo.write(json.dumps(r) + "\n"); fo.flush()
    print(qi, q['category'], 'tiles', len(T), f'{time.time() - t0:.0f}s', flush=True)
print('DONE', flush=True)
