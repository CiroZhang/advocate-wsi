"""Crop-substitution audit of RL-trained 'thinking with images' agents on V*-Bench (generalizes de_attack.py to several agents).
Each agent runs its own official multi-turn zoom protocol. Per question:
  native  : the agent's own rollout (real crops).
  subst_* : identical prompt, but EVERY zoom call returns a substituted image instead of the requested region:
            gt = grid tile with max GT-box overlap; rand0/rand1 = random tiles; adv0..adv3 = top wrong-option-margin tiles under Qwen2.5-VL
            (black-box transfer, from vstar_partial.jsonl); blank = a uniform grey image (no visual content at all).
If the zoom is load-bearing, gt > rand > blank and adv flips answers; if it is decorative, all are equal to native.
We also store the first-turn text (before any zoom) to measure whether the answer is committed before looking.
env: AGENT=deepeyes|minio3, INIT_PIX (overview pixel budget), TAG, SHARD/NSHARD (question shards), CH (rollouts per batch).
usage: AGENT=minio3 TAG=minio3_full python agent_subst.py"""
import os, sys, json, re, math, time, random, hashlib, numpy as np, torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
R = os.environ.get('HF_ROOT', "/scratch/ciro/cvpr_lab/hf"); VS = f"{R}/vstar_bench"; OUT = os.environ.get('OUT', "/scratch/ciro/cvpr_lab/paperA_vstar")
AGENT = os.environ.get('AGENT', 'minio3'); TAG = os.environ.get('TAG', AGENT); CH = int(os.environ.get('CH', 3))
SHARD, NSHARD = int(os.environ.get('SHARD', 0)), int(os.environ.get('NSHARD', 1))
L = "ABCD"; random.seed(0); torch.manual_seed(0)

# ------------------------------------------------------------------------------------------------ agent protocols
if AGENT == 'deepeyes':  # official DeepEyes V* eval (as de_attack.py v3)
    MP = "DeepEyes-7B"; MAX_TURNS, MAX_NEW = 4, 768; INIT_PIX = int(os.environ.get("INIT_PIX", 16384 * 28 * 28)); OBS_PIX = 16384 * 28 * 28
    SYS = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'deepeyes_sys.txt')).read() if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'deepeyes_sys.txt')) else None
    UP = "\nThink first, call **image_zoom_in_tool** if needed, then answer. Format strictly as:  <think>...</think>  <tool_call>...</tool_call> (if tools needed)  <answer>...</answer> "
    STOPS = ["</tool_call>", "</answer>"]; CALL = "<tool_call>"; REL = False
    def user_prompt(q, stem, opts): return f"Question: {stem}\nOptions: \n" + "".join(f"{L[k]}. {o}\n" for k, o in enumerate(opts)) + "\n" + UP
    def tool_msg(turn, crop): return ["<tool_response>", crop, UP, "</tool_response>"]
    def parse_call(s):
        m = re.search(r"bbox_2d\"?\s*:\s*\[\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)", s)
        return ([float(v) for v in m.groups()], 0) if m else None
elif AGENT == 'minio3':  # Mini-o3 (github Mini-o3/Mini-o3: verl/trainer/constants.py, vllm_rollout_spmd.py; relative coords, raw-image crops)
    MP = "Mini-o3-7B-v1"; MAX_TURNS, MAX_NEW = int(os.environ.get('MAX_TURNS', 8)), 1024; INIT_PIX = int(os.environ.get("INIT_PIX", 2000000)); OBS_PIX = 2000000
    SYS = ("You are a helpful assistant. Answer the user's question based on the image provided. Output your thinking process within the <think> and </think> tags. "
           "Whenever you find anything unclear, you can zoom in a specific region in the given image to see more clearly by outputing <grounding>{\"bbox_2d\": [x0, y0, x1, y1], "
           "\"source\": \"original_image\"}</grounding>, where (x0, y0) and (x1, y1) are the top-left and bottom-right coordinates of the region that you want to zoom in, "
           "respectively (suppose the width and height of the image are 1.0), and 'source' refers to the image that you zoom in and could be either 'original_image' or "
           "'observation_i'. Once the final answer is confirmed, put it within <answer> and </answer>.")
    STOPS = ["</grounding>", "</answer>"]; CALL = "<grounding>"; REL = True
    def user_prompt(q, stem, opts): return q['text']
    def tool_msg(turn, crop):
        return [f"After the above Action {turn}, here is the the zoom-in image (Observation {turn + 1}):\n", crop,
                ".\nContinue your reasoning process inside <think> and </think>. If needed, you can continue to zoom in on the original image or any of the observations, "
                "by outputting <grounding> and </grounding> as before. If the final answer is confirmed, put your final answer inside <answer> and </answer>."]
    def parse_call(s):
        m = re.search(r"<grounding>\s*\{\s*\"bbox_2d\"\s*:\s*\[\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\]\s*,\s*\"source\"\s*:\s*[\"']([^\"']*)[\"']", s)
        if not m: return None
        src = m.group(5); o = 0 if src == 'original_image' else int(re.match(r"observation_(\d+)", src).group(1)) if re.match(r"observation_(\d+)", src) else 0
        return [float(v) for v in m.groups()[:4]], o
else: raise ValueError(AGENT)
proc = AutoProcessor.from_pretrained(f"{R}/{MP}"); proc.tokenizer.padding_side = 'left'
model = AutoModelForImageTextToText.from_pretrained(f"{R}/{MP}", dtype=torch.bfloat16, device_map="cuda").eval()
if AGENT == 'deepeyes' and SYS is None:  # reuse the exact system prompt of de_attack.py
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'de_attack.py')).read(); SYS = re.search(r'SYS = """(.*?)"""', src, re.S).group(1)

# ------------------------------------------------------------------------------------------------ helpers
def smart_resize(h, w, factor=28, min_pixels=4 * 28 * 28, max_pixels=16384 * 28 * 28):
    hb, wb = max(factor, round(h / factor) * factor), max(factor, round(w / factor) * factor)
    if hb * wb > max_pixels:
        b = math.sqrt(h * w / max_pixels); hb, wb = math.floor(h / b / factor) * factor, math.floor(w / b / factor) * factor
    elif hb * wb < min_pixels:
        b = math.sqrt(min_pixels / (h * w)); hb, wb = math.ceil(h * b / factor) * factor, math.ceil(w * b / factor) * factor
    return hb, wb
def fit(im, max_pixels):
    h, w = smart_resize(im.size[1], im.size[0], max_pixels=max_pixels); return im.resize((w, h), Image.BICUBIC)
def to_box(v, src_im, shown_size):  # model coordinates -> pixel box in src_im
    W, H = src_im.size
    if REL: x1, y1, x2, y2 = v[0] * W, v[1] * H, v[2] * W, v[3] * H
    else: sx, sy = W / shown_size[0], H / shown_size[1]; x1, y1, x2, y2 = v[0] * sx, v[1] * sy, v[2] * sx, v[3] * sy
    x1, x2 = sorted((min(max(x1, 0), W), min(max(x2, 0), W))); y1, y2 = sorted((min(max(y1, 0), H), min(max(y2, 0), H)))
    if x2 - x1 < 28: x2 = min(W, x1 + 28); x1 = max(0, x2 - 28)
    if y2 - y1 < 28: y2 = min(H, y1 + 28); y1 = max(0, y2 - 28)
    return [x1, y1, x2, y2]
def parse_answer(s, opts):
    m = re.search(r"<answer>(.*?)</answer>", s, re.S)
    a = (m.group(1) if m else s[-200:]).strip()
    m2 = re.match(r"^\(?([A-D])[\).:\s]|^\(?([A-D])\)?$", a)
    if m2: return L.index(m2.group(1) or m2.group(2))
    for k, o in enumerate(opts):
        if o.lower().rstrip('.') in a.lower(): return k
    m3 = re.search(r"\b([A-D])\b", a)
    return L.index(m3.group(1)) if m3 else -1
def tiles(im):  # identical to vstar_sel.py
    W, H = im.size; out = []
    for frac in (0.25, 0.5):
        s = int(min(W, H) * frac); st = s // 2
        for t in range(0, max(1, H - s + 1), st):
            for l in range(0, max(1, W - s + 1), st): out.append((l, t, l + s, t + s))
    return out
def overlap(a, b):
    bx = (b[0], b[1], b[0] + b[2], b[1] + b[3])
    iw = max(0, min(a[2], bx[2]) - max(a[0], bx[0])); ih = max(0, min(a[3], bx[3]) - max(a[1], bx[1]))
    return iw * ih / max(1, b[2] * b[3])
def to_chat(msgs):
    out, ims = [{"role": "system", "content": SYS}], []
    for role, parts in msgs:
        if role == "assistant": out.append({"role": role, "content": parts[0]}); continue
        c = []
        for p in parts:
            if isinstance(p, str): c.append({"type": "text", "text": p})
            else: c.append({"type": "image"}); ims.append(p)
        out.append({"role": role, "content": c})
    return out, ims
@torch.no_grad()
def gen(batch):
    texts, ims = [], []
    for msgs in batch:
        m, i = to_chat(msgs); texts.append(proc.apply_chat_template(m, add_generation_prompt=True)); ims += i
    enc = proc(text=texts, images=ims, return_tensors="pt", padding=True).to("cuda")
    out = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False, stop_strings=STOPS, tokenizer=proc.tokenizer)  # RL rollouts were cut at the tool stop string
    res = []
    for o in out:
        t = proc.tokenizer.decode(o[enc.input_ids.shape[1]:], skip_special_tokens=False).split("<|im_end|>")[0].split("<|im_start|>")[0]
        cut = [t.index(st) + len(st) for st in STOPS if st in t]
        if cut: t = t[:min(cut)]
        res.append(t.replace("<|endoftext|>", ""))
    return res
def rollout(im, prompt, crop_fn, n):
    """n parallel rollouts; crop_fn(j, src_image, box) -> PIL returned to rollout j. Returns final texts, first-turn texts, requested boxes."""
    ov = fit(im, INIT_PIX); convs = [[("user", [ov, prompt])] for _ in range(n)]
    obs = [[(im, ov.size)] for _ in range(n)]  # observation 0 = original image (shown at ov.size); later = crops
    active = list(range(n)); final, first, boxes, trace = [""] * n, [""] * n, [[] for _ in range(n)], [[] for _ in range(n)]
    for turn in range(MAX_TURNS):
        if not active: break
        outs = gen([convs[j] for j in active]); nxt = []
        for j, o in zip(active, outs):
            final[j] = o; trace[j].append(o)
            if turn == 0: first[j] = o
            if "</answer>" in o or CALL not in o: continue
            pc = parse_call(o)
            if pc is None: continue
            v, src = pc; src = min(src, len(obs[j]) - 1); s_im, s_sz = obs[j][src]
            bb = to_box(v, s_im, s_sz); boxes[j].append((src, bb))
            cr = crop_fn(j, s_im, bb); crs = fit(cr, OBS_PIX); obs[j].append((cr, crs.size))
            convs[j] += [("assistant", [o]), ("user", tool_msg(turn + 1, crs))]
            nxt.append(j)
        active = nxt
    return final, first, boxes, trace

# ------------------------------------------------------------------------------------------------ main
qs = [json.loads(l) for l in open(f"{VS}/test_questions.jsonl")][SHARD::NSHARD]
cache = {json.loads(l)['qid']: json.loads(l) for l in open(f"{OUT}/vstar_partial.jsonl")}  # Qwen2.5-VL-7B tile scores (attacker's surrogate)
part = f"{OUT}/{TAG}_s{SHARD}of{NSHARD}.jsonl"
done = {json.loads(l)['qid'] for l in open(part)} if os.path.exists(part) else set()
fo = open(part, 'a'); t0 = time.time()
for qi, q in enumerate(qs):
    if q['question_id'] in done: continue
    im = Image.open(f"{VS}/{q['image']}").convert("RGB"); ann = json.load(open(f"{VS}/{q['image'].replace('.jpg', '.json')}"))
    y = L.index(q['label']); opts = re.findall(r"^\([A-D]\) (.*)$", q['text'], re.M); stem = q['text'].split("\n")[0]
    prompt = user_prompt(q, stem, opts)
    T = tiles(im); c = cache[q['question_id']]; S = np.array(c['S_glob']); assert len(S) == len(T)
    ov = np.array([max(overlap(t, g) for g in ann['bbox']) for t in T])
    margin = np.max(np.delete(S, y, 1), 1) - S[:, y]
    rr = random.Random(int(hashlib.md5(str(q['question_id']).encode()).hexdigest()[:8], 16))
    subst = {f'adv{i}': int(t) for i, t in enumerate(np.argsort(-margin)[:4])} | {f'rand{i}': int(t) for i, t in enumerate(rr.sample(range(len(T)), 2))} | {'gt': int(ov.argmax()), 'blank': -1}
    grey = Image.new('RGB', (448, 448), (128, 128, 128))
    fin, fst, bx, tr = rollout(im, prompt, lambda j, s, b: s.crop(tuple(int(v) for v in b)), 1); torch.cuda.empty_cache()
    rec = {'qid': q['question_id'], 'cat': q['category'], 'y': y, 'agent': AGENT, 'init_pix': INIT_PIX,
           'native': {'pred': parse_answer(fin[0], opts), 'pred_first': parse_answer(fst[0], opts) if '<answer>' in fst[0] else None, 'n_tool': len(bx[0]),
                      'boxes': bx[0], 'box_gt_overlap': [max(overlap(b, g) for g in ann['bbox']) for s_, b in bx[0] if s_ == 0], 'first': fst[0][-600:], 'text': fin[0][-400:],
                      'trace': [t[-500:] for t in tr[0]]}}
    names = list(subst); outs = []
    for c0 in range(0, len(names), CH):
        nms = names[c0:c0 + CH]
        fn = lambda j, s, b, nms=nms: grey if subst[nms[j]] < 0 else im.crop(T[subst[nms[j]]])
        try: outs += list(zip(*rollout(im, prompt, fn, len(nms))))
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            for nm in nms: outs += list(zip(*rollout(im, prompt, lambda j, s, b, nm=nm: grey if subst[nm] < 0 else im.crop(T[subst[nm]]), 1)))
        torch.cuda.empty_cache()
    rec['subst'] = {nm: {'tile': subst[nm], 'tile_gt_overlap': float(ov[subst[nm]]) if subst[nm] >= 0 else 0.0,
                         'surrogate_margin': float(margin[subst[nm]]) if subst[nm] >= 0 else None,
                         'pred': parse_answer(o[0], opts), 'n_tool': len(o[2]), 'text': o[0][-300:]} for nm, o in zip(names, outs)}
    fo.write(json.dumps(rec) + "\n"); fo.flush()
    print(qi, q['category'], 'native', rec['native']['pred'] == y, rec['native']['n_tool'], {k: v['pred'] == y for k, v in rec['subst'].items()}, f'{time.time() - t0:.0f}s', flush=True)
print('DONE', flush=True)
