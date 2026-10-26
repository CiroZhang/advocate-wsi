"""Extended robustness experiments, shared by exp_tcga.py and exp_c17.py (run with EXT=1; results go to *_ext.{json,npz}).
A presentation is a list of slots (index tensors into the slide's tiles); slot s is tagged with a one-hot flag of size NF = #slots.
  adaptive attack : the compromised slot picks its tiles greedily, one at a time, to minimise the judge's p(true class) given the
                    honest slots' tiles (set-aware; candidates = top-M tiles by single-tile score). Stronger than the per-tile attack.
  hybrid          : the C advocates plus one neutral slot showing the single selector's top tiles (by distilled attention);
                    any one slot, including the neutral one, may be compromised.
Configs that need new cheap scorers (independent per-advocate scorers, weak scorer) are built by the calling script."""
import os, numpy as np, torch, torch.nn.functional as Fn

class Ctx:  # XE, extra(i, idx) -> (len(idx), E) extra columns, E, C, y, tr, te, train, predict, rng, DEV
    def __init__(s, **kw): s.__dict__.update(kw)

def onehot(n, k, NF): o = torch.zeros(n, NF); o[:, k] = 1; return o
def slot_rows(cx, i, idx, s, NF): return torch.cat([cx.XE[i][idx], onehot(len(idx), s, NF), cx.extra(i, idx)], 1)
def rows(cx, i, S): return torch.cat([slot_rows(cx, i, idx, s, len(S)) for s, idx in enumerate(S)])

@torch.no_grad()
def tile_scores(cx, m, i, s, NF, lv):  # single-tile log p(lv) when a tile is shown under slot s's flag
    n = len(cx.XE[i]); X = torch.cat([cx.XE[i], onehot(n, s, NF), torch.zeros(n, cx.E)], 1).to(cx.DEV)
    z, _ = m.terms(X); return Fn.log_softmax(m.c(z), 1)[:, lv].cpu()
def wb(cx, m, i, S, s, lv):  # per-tile white-box attacker (as in the main experiments)
    S2 = list(S); S2[s] = tile_scores(cx, m, i, s, len(S), lv).topk(min(len(S[s]), len(cx.XE[i]))).indices; return S2

@torch.no_grad()
def batched(m, X3):  # gated-ABMIL forward over a batch of equal-size bags [B, n, d]
    z = m.f(X3); a = m.w(torch.tanh(m.a(z)) * torch.sigmoid(m.b(z))).squeeze(-1); return m.c((a.softmax(1)[..., None] * z).sum(1))
@torch.no_grad()
def adaptive(cx, m, i, S, s, lv, M=128):
    NF = len(S); k = len(S[s]); yt = int(cx.y[i])
    cand = tile_scores(cx, m, i, s, NF, lv).topk(min(M, len(cx.XE[i]))).indices.tolist()
    oth = [slot_rows(cx, i, idx, t, NF) for t, idx in enumerate(S) if t != s]
    base = (torch.cat(oth) if oth else torch.zeros(0, cx.XE[i].shape[1] + NF + cx.E)).to(cx.DEV); chosen = []
    for _ in range(min(k, len(cand))):
        rest = [c for c in cand if c not in chosen]
        A = torch.stack([slot_rows(cx, i, torch.tensor(chosen + [c]), s, NF) for c in rest]).to(cx.DEV)
        lp = batched(m, torch.cat([base[None].expand(len(rest), -1, -1), A], 1)).log_softmax(1)[:, yt]
        chosen.append(rest[int(lp.argmin())])
    S2 = list(S); S2[s] = torch.tensor(chosen); return S2

ATTY = bool(os.environ.get('ATTY'))  # also let the attacker take the TRUE class's advocate (argue any wrong class from that slot); used in training and evaluation
ANYSLOT = False  # redundancy baselines: slots are unlabelled, so the attacker may take any slot and argue any wrong class
def combos(cx, i, NF):  # (slot, target class) pairs an attacker may take: a wrong-class advocate argues its class; the neutral slot argues any wrong class
    yt = int(cx.y[i])
    if ANYSLOT: return [(s, c) for s in range(NF) for c in range(cx.C) if c != yt]
    out = [(c, c) for c in range(cx.C) if c != yt]
    if ATTY and NF >= cx.C: out += [(yt, c) for c in range(cx.C) if c != yt]
    return out + [(cx.C, c) for c in range(cx.C) if c != yt] if NF > cx.C else out

def train_judge(cx, SS):  # as run_teacher in the main scripts: warm start on random tiles, then 3 rounds of clean + one-attacked transcripts
    NF = len(SS[cx.tr[0]])
    def rand_pres(i): n = len(cx.XE[i]); return [torch.randperm(n)[:len(idx)] for idx in SS[i]]
    m = cx.train([rows(cx, i, rand_pres(i)) for i in cx.tr], cx.y[cx.tr], cx.C, 10); sets, ys = [], []
    for r in range(3):
        for i in cx.tr:
            sets.append(rows(cx, i, SS[i])); ys.append(cx.y[i])
            cb = combos(cx, i, NF); s, lv = cb[int(cx.rng.integers(len(cb)))]
            sets.append(rows(cx, i, wb(cx, m, i, SS[i], s, lv))); ys.append(cx.y[i])
        m = cx.train(sets, ys, cx.C, 5, model=m)
    return m

def evaluate(cx, m, SS, tag, L, adapt=True):
    """L[tag] clean, L[tag_wb] worst per-tile attack, L[tag_ad] worst of per-tile and greedy set attack (over all slot/target pairs)."""
    NF = len(SS[cx.te[0]]); L[tag] = cx.predict(m, [rows(cx, i, SS[i]) for i in cx.te])
    for kind, atts in [('wb', [wb])] + ([('ad', [wb, adaptive])] if adapt else []):  # the adaptive attacker also has the per-tile option
        worst = []
        for i in cx.te:
            cands = [cx.predict(m, [rows(cx, i, att(cx, m, i, SS[i], s, lv))])[0] for s, lv in combos(cx, i, NF) for att in atts]
            worst.append(min(cands, key=lambda v: Fn.softmax(torch.tensor(v), 0)[int(cx.y[i])].item()))
        L[f'{tag}_{kind}'] = np.stack(worst)

def train_iso(cx, SS):  # isolate-then-aggregate judge: one bag per slot (flag of size 1), trained on honest and white-box-attacked single slots
    def rand1(i, k): return [torch.randperm(len(cx.XE[i]))[:k]]
    m = cx.train([rows(cx, i, rand1(i, len(SS[i][0]))) for i in cx.tr], cx.y[cx.tr], cx.C, 10); sets, ys = [], []
    for r in range(3):
        for i in cx.tr:
            yt = int(cx.y[i]); idx = SS[i][int(cx.rng.integers(len(SS[i])))]; lv = int(cx.rng.choice([c for c in range(cx.C) if c != yt]))
            sets += [rows(cx, i, [idx]), rows(cx, i, wb(cx, m, i, [idx], 0, lv))]; ys += [yt, yt]
        m = cx.train(sets, ys, cx.C, 5, model=m)
    return m

def evaluate_iso(cx, m, SS, tag, L):
    """Each slot judged alone; aggregate = mean of per-slot log-probs. Attack: one slot replaced (per-tile or greedy set attack against
    that slot alone, any wrong class); L[tag] clean, L[tag_ad] per-slide worst case."""
    lp = lambda i, S: [torch.tensor(cx.predict(m, [rows(cx, i, [idx])])[0]).log_softmax(0) for idx in S]
    clean, worst = [], []
    for i in cx.te:
        P = lp(i, SS[i]); clean.append(torch.stack(P).mean(0).numpy()); yt = int(cx.y[i]); best = None
        for s in range(len(SS[i])):
            for lv in range(cx.C):
                if lv == yt: continue
                for att in (wb, adaptive):
                    Q = list(P); Q[s] = lp(i, [att(cx, m, i, [SS[i][s]], 0, lv)[0]])[0]; v = torch.stack(Q).mean(0).numpy()
                    if best is None or Fn.softmax(torch.tensor(v), 0)[yt] < Fn.softmax(torch.tensor(best), 0)[yt]: best = v
        worst.append(best)
    L[tag] = np.stack(clean); L[f'{tag}_ad'] = np.stack(worst)

def run(cx, configs, L, log=print):
    """configs: list of (tag, SS dict over all slides, adapt flag[, mode]); mode 'any' = unlabelled slots (attacker takes any slot),
    'iso' = isolate-then-aggregate (each slot judged alone, any slot attackable)."""
    global ANYSLOT
    for tag, SS, adapt, *mode in configs:
        mode = mode[0] if mode else None; ANYSLOT = mode in ('any', 'iso')
        if mode == 'iso': m = train_iso(cx, SS); evaluate_iso(cx, m, SS, tag, L)
        else: m = train_judge(cx, SS); evaluate(cx, m, SS, tag, L, adapt)
        ANYSLOT = False; log(tag)
