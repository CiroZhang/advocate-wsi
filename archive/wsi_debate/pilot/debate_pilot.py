"""H pilot: adversarial evidence debate on BCNB CONCH patch features (real data, 5-fold CV).
Two debaters (search-based best responses to the current judge) alternately reveal patches arguing for opposite
labels; a question-blind judge sees ONLY the revealed patches (+ who revealed each). The judge is retrained on
debate transcripts for R rounds. Baselines with the same judge architecture: random-K, attention-top-K, full-slide ABMIL."""
import os, sys, json, numpy as np, pandas as pd, torch, torch.nn as nn, collections as C
torch.set_num_threads(6); torch.manual_seed(0); rng = np.random.default_rng(0)
F = r"D:\wsiagent\data\BCNB_patch_feat"
bc = pd.read_csv(r"D:\wsiagent\pilot\SlideBench-VQA-BCNB.csv")
lab = C.defaultdict(dict)
for _, r in bc.iterrows():
    v = str(r[r.Answer]).strip().lower()
    if r.Task in ('ER', 'PR', 'HER2'): lab[r.Task][str(r.Slide)] = int(v == 'positive')
    if r.Task == 'Histological grading': lab['Grade3'][str(r.Slide)] = int(v == '3')
slides = sorted({str(s) for s in bc.Slide}, key=int)
feats = {s: torch.tensor(pd.read_csv(os.path.join(F, f"{s}.csv")).iloc[:, :512].values, dtype=torch.float32) for s in slides}
DEV = 'cpu'

class Judge(nn.Module):  # gated-attention MIL over revealed patches; input = feature + presenter flag (+1 for side 1, -1 for side 0, 0 neutral)
    def __init__(s, d=513, h=128):
        super().__init__(); s.f = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Dropout(0.1))
        s.a = nn.Linear(h, h); s.b = nn.Linear(h, h); s.w = nn.Linear(h, 1); s.c = nn.Linear(h, 1)
    def forward(s, x):  # x: (B, K, 513) -> logit (B,)
        z = s.f(x); a = s.w(torch.tanh(s.a(z)) * torch.sigmoid(s.b(z))).softmax(1)
        return s.c((a * z).sum(1)).squeeze(-1)

def with_flag(x, flag): return torch.cat([x, torch.full((x.shape[0], 1), float(flag))], 1)

def train_judge(sets, ys, epochs=15, judge=None):
    j = judge or Judge(); opt = torch.optim.AdamW(j.parameters(), 3e-4, weight_decay=1e-4)
    pos = np.mean(ys); w = torch.tensor([(1 - pos) / max(pos, 1e-3)])
    lossf = nn.BCEWithLogitsLoss(pos_weight=w)  # class-balanced -> judge is prior-neutral
    for _ in range(epochs):
        j.train()
        for i in rng.permutation(len(sets)):
            loss = lossf(j(sets[i][None]), torch.tensor([float(ys[i])])); opt.zero_grad(); loss.backward(); opt.step()
    return j.eval()

@torch.no_grad()
def debate(j, x, K):  # alternate turns; side 1 maximizes judge logit, side 0 minimizes; returns revealed set (k, 513)
    n = x.shape[0]; used = set(); rev = []
    order = [1, 0] * (K // 2); rng.shuffle(order) if False else None
    for t, side in enumerate(order):
        cand = [i for i in range(n) if i not in used]
        flag = 1.0 if side == 1 else -1.0
        base = torch.stack(rev) if rev else torch.zeros(0, 513)
        sets = torch.stack([torch.cat([base, with_flag(x[i:i + 1], flag)], 0) for i in cand])
        lg = j(sets); pick = cand[int(lg.argmax() if side == 1 else lg.argmin())]
        used.add(pick); rev.append(with_flag(x[pick:pick + 1], flag)[0])
    return torch.stack(rev)

def balacc(p, y):
    p, y = np.array(p), np.array(y); return np.mean([np.mean(p[y == c] == c) for c in (0, 1) if (y == c).any()])

res = {}
tasks = sys.argv[1].split(',') if len(sys.argv) > 1 else ['ER', 'PR', 'Grade3', 'HER2']
Ks = [2, 4, 8]
for task in tasks:
    d = lab[task]; ss = [s for s in slides if s in d]; y = np.array([d[s] for s in ss])
    folds = np.array_split(rng.permutation(len(ss)), 5)
    out = C.defaultdict(lambda: ([], []))
    for k in range(5):
        te = folds[k]; tr = np.concatenate([folds[m] for m in range(5) if m != k])
        # full-slide ABMIL (all patches, neutral flag) and its attention-top-K
        full = train_judge([with_flag(feats[ss[i]], 0) for i in tr], y[tr], epochs=10)
        with torch.no_grad():
            for i in te:
                out['full'][0].append(int(full(with_flag(feats[ss[i]], 0)[None]).item() > 0)); out['full'][1].append(y[i])
        def topk(i, K):
            x = with_flag(feats[ss[i]], 0)
            with torch.no_grad():
                z = full.f(x); a = full.w(torch.tanh(full.a(z)) * torch.sigmoid(full.b(z))).squeeze(-1)
            return x[a.topk(min(K, len(x))).indices]
        def randk(i, K):
            x = with_flag(feats[ss[i]], 0); return x[torch.randperm(len(x))[:K]]
        for K in Ks:
            for name, fn in (('random', randk), ('attn_top', topk)):
                jj = train_judge([fn(i, K) for i in tr], y[tr], epochs=10)
                with torch.no_grad():
                    for i in te:
                        out[f'{name}_K{K}'][0].append(int(jj(fn(i, K)[None]).item() > 0)); out[f'{name}_K{K}'][1].append(y[i])
            # debate: judge bootstrapped on random subsets with random flags, then retrained on debate transcripts
            jd = train_judge([with_flag(feats[ss[i]], 0)[torch.randperm(len(feats[ss[i]]))[:K]] for i in tr], y[tr], epochs=5)
            for r in range(2):
                trans = [debate(jd, feats[ss[i]], K) for i in tr]
                jd = train_judge(trans, y[tr], epochs=6, judge=jd)
            with torch.no_grad():
                for i in te:
                    t = debate(jd, feats[ss[i]], K)
                    out[f'debate_K{K}'][0].append(int(jd(t[None]).item() > 0)); out[f'debate_K{K}'][1].append(y[i])
                    # blank control: flags only, zeroed features -> judge must fall back to chance-level
                    tb = t.clone(); tb[:, :512] = 0
                    out[f'debate_blank_K{K}'][0].append(int(jd(tb[None]).item() > 0)); out[f'debate_blank_K{K}'][1].append(y[i])
        print(task, 'fold', k, {m: round(balacc(*v), 3) for m, v in out.items()}, flush=True)
    res[task] = {m: {'acc': float(np.mean(np.array(v[0]) == np.array(v[1]))), 'balacc': float(balacc(*v))} for m, v in out.items()}
    res[task]['prior_acc'] = float(max(y.mean(), 1 - y.mean()))
    json.dump(res, open(rf"D:\wsiagent\debate_{'_'.join(tasks)}.json", 'w'), indent=1)
    print(task, json.dumps(res[task]), flush=True)
