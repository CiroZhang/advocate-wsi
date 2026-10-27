"""Extended robustness experiments, shared by exp_tcga.py and exp_c17.py (run with EXT=1; results go to *_ext.{json,npz}).
A presentation is a list of slots (index tensors into the slide's tiles); slot s is tagged with a one-hot flag of size NF = #slots.
  adaptive attack : the compromised slot picks its tiles greedily, one at a time, to minimise the judge's p(true class) given the
                    honest slots' tiles (set-aware; candidates = top-M tiles by single-tile score). Stronger than the per-tile attack.
  hybrid          : the C advocates plus one neutral slot showing the single selector's top tiles (by distilled attention);
                    any one slot, including the neutral one, may be compromised.
Configs that need new cheap scorers (independent per-advocate scorers, weak scorer) are built by the calling script."""
import os, math, numpy as np, torch, torch.nn.functional as Fn

class Ctx:  # XE, extra(i, idx) -> (len(idx), E) extra columns, E, C, y, tr, te, train, predict, rng, DEV
    def __init__(s, **kw): s.__dict__.update(kw)

def onehot(n, k, NF): o = torch.zeros(n, NF); o[:, k] = 1; return o
def slot_rows(cx, i, idx, s, NF): return torch.cat([cx.XE[i][idx], onehot(len(idx), s, NF), cx.extra(i, idx)], 1)
def rows(cx, i, S): return torch.cat([slot_rows(cx, i, idx, s, len(S)) for s, idx in enumerate(S)])

@torch.no_grad()
def tile_scores(cx, m, i, s, NF, lv):  # single-tile log p(lv) when a tile is shown under slot s's flag
    n = len(cx.XE[i]); X = torch.cat([cx.XE[i], onehot(n, s, NF), torch.zeros(n, cx.E)], 1).to(cx.DEV)
    if not hasattr(m, 'terms'): return torch.cat([Fn.log_softmax(m.batch(X[j:j + 4096, None]), 1)[:, lv] for j in range(0, n, 4096)]).cpu()  # archs.MIL
    z, _ = m.terms(X); return Fn.log_softmax(m.c(z), 1)[:, lv].cpu()
def wb(cx, m, i, S, s, lv):  # per-tile white-box attacker (as in the main experiments)
    S2 = list(S); S2[s] = tile_scores(cx, m, i, s, len(S), lv).topk(min(len(S[s]), len(cx.XE[i]))).indices; return S2

@torch.no_grad()
def batched(m, X3):  # gated-ABMIL forward over a batch of equal-size bags [B, n, d]
    if hasattr(m, 'batch'): return torch.cat([m.batch(X3[j:j + 64]) for j in range(0, len(X3), 64)])  # archs.MIL
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

AGG = {'': lambda P: P.mean(0),  # mean of per-slot log-probs (as in RobustRAG's keyword/decoding aggregation)
       '_clip': lambda P: P.clamp(min=math.log(.05)).mean(0),  # bounded: one slot can move the mean by at most log(20)/m
       '_vote': lambda P: Fn.one_hot(P.argmax(1), P.shape[1]).float().sum(0) + 1e-3 * P.exp().mean(0)}  # majority vote, ties by mean prob

def evaluate_iso(cx, m, SS, tag, L):
    """Each slot judged alone, then aggregated (AGG: mean log-prob, clipped log-prob, majority vote). Attack: one slot replaced (per-tile
    or greedy set attack against that slot alone, any wrong class); L[tag+agg] clean, L[tag+agg+'_ad'] per-slide worst case."""
    lp = lambda i, S: [torch.tensor(cx.predict(m, [rows(cx, i, [idx])])[0]).log_softmax(0) for idx in S]
    clean, worst = {g: [] for g in AGG}, {g: [] for g in AGG}
    for i in cx.te:
        P = lp(i, SS[i]); yt = int(cx.y[i]); cands = []
        for s in range(len(SS[i])):
            for lv in range(cx.C):
                if lv == yt: continue
                for att in (wb, adaptive):
                    Q = list(P); Q[s] = lp(i, [att(cx, m, i, [SS[i][s]], 0, lv)[0]])[0]; cands.append(torch.stack(Q))
        for g, f in AGG.items():
            clean[g].append(f(torch.stack(P)).numpy())
            worst[g].append(min((f(Q).numpy() for Q in cands), key=lambda v: Fn.softmax(torch.tensor(v), 0)[yt].item()))
    for g in AGG: L[tag + g] = np.stack(clean[g]); L[f'{tag}{g}_ad'] = np.stack(worst[g])

def run(cx, configs, L, log=print):
    """configs: list of (tag, SS dict over all slides, adapt flag[, mode]); mode 'any' = unlabelled slots (attacker takes any slot),
    'iso' = isolate-then-aggregate (each slot judged alone, any slot attackable)."""
    global ANYSLOT
    for tag, SS, adapt, *mode in configs:
        mode = mode[0] if mode else None; ANYSLOT = mode in ('any', 'iso')
        if mode == 'iso': m = train_iso(cx, SS); evaluate_iso(cx, m, SS, tag, L)
        else: m = train_judge(cx, SS); evaluate(cx, m, SS, tag, L, adapt)
        ANYSLOT = False; log(tag)

def arch_run(cx, arch, train, top, SS, Kc, K, mets, out, t0, info):
    """Generality (ARCH=...): the same selector (top), advocates (SS) and attacks with another MIL judge from archs.py.
    full: judge on every tile; single: judge on the selector's top-K; single_ad: worst of per-tile and greedy set attack (top-128
    candidates) replacing the selector, over wrong classes; x_red / x_hyb{K}: plain redundancy (any slot) and hybrid, as in run()."""
    import json, time; from archs import MIL
    C, XE, y, te, DEV = cx.C, cx.XE, cx.y, cx.te, cx.DEV
    new = lambda d: MIL(d, C, arch).to(DEV); cx.train = lambda sets, ys, o, ep, model=None: train(sets, ys, o, ep, model=model or new(sets[0].shape[1]))
    J = cx.train([XE[i] for i in cx.tr], y[cx.tr], C, 20); R = {'full': mets(cx.predict(J, [XE[i] for i in te])), 'single': mets(cx.predict(J, [XE[i][top(i, K)] for i in te]))}
    worst = []
    with torch.no_grad():
        for i in te:
            n = len(XE[i]); X = XE[i].to(DEV); P1 = torch.cat([J.batch(X[j:j + 4096, None]).log_softmax(1) for j in range(0, n, 4096)]).cpu(); cands = []
            for lv in range(C):
                if lv == y[i]: continue
                cands.append(cx.predict(J, [XE[i][P1[:, lv].topk(min(K, n)).indices]])[0]); pool = P1[:, lv].topk(min(128, n)).indices.tolist(); ch = []
                for _ in range(min(K, n)):
                    rest = [c for c in pool if c not in ch]; lp = batched(J, torch.stack([X[ch + [c]] for c in rest])).log_softmax(1)[:, int(y[i])]
                    ch.append(rest[int(lp.argmin())])
                cands.append(cx.predict(J, [XE[i][ch]])[0])
            worst.append(min(cands, key=lambda v: Fn.softmax(torch.tensor(v), 0)[int(y[i])].item()))
    R['single_ad'] = mets(np.stack(worst)); print(arch, R, f'{time.time() - t0:.0f}s', flush=True)
    N = range(len(XE)); LA = {}
    def red(i, k): t = top(i, (C + 1) * k); return [t[j::C + 1] if j < len(t) else t[:1] for j in range(C + 1)]
    run(cx, [('x_red', {i: red(i, Kc) for i in N}, True, 'any'), (f'x_hyb{K}', {i: SS[i] + [top(i, K)] for i in N}, True)], LA,
        log=lambda t: print('arch', t, f'{time.time() - t0:.0f}s', flush=True))
    R.update({k: mets(v) for k, v in LA.items()}); json.dump({'res': R, 'info': info}, open(out, 'w'), indent=1)

def bbox_run(cx, fu, att, top, SS, Kc, K, mets, out, t0, info):
    """Black-box, label-free attacker (BBOX=1): it knows neither the label nor the judge. It pushes every slide toward one fixed
    target class c, choosing tiles with the cheap scorer only (att(i, c, k) = the advocate rule for c with budget k), and takes the
    single selector or one slot of a multi-slot presenter (worst over slot positions). Reported per target class c (all test
    slides, including those of class c); *_bb_mean / *_bb_min = mean / min over c. fu = full-slide judge used by the single selector."""
    import json, time
    C, XE, te = cx.C, cx.XE, cx.te; R = {'single': mets(cx.predict(fu, [XE[i][top(i, K)] for i in te]))}
    per = [mets(cx.predict(fu, [XE[i][att(i, c, K)] for i in te])) for c in range(C)]
    for c, r in enumerate(per): R[f'single_bb{c}'] = r
    def multi(tag, SSd):
        global ANYSLOT
        ANYSLOT = True; m = train_judge(cx, SSd); ANYSLOT = False; NF = len(SSd[te[0]])
        R[tag] = mets(cx.predict(m, [rows(cx, i, SSd[i]) for i in te]))
        for c in range(C):
            worst = []
            for i in te:
                cands = []
                for s in range(NF):
                    S2 = list(SSd[i]); S2[s] = att(i, c, len(SSd[i][s])); cands.append(cx.predict(m, [rows(cx, i, S2)])[0])
                worst.append(min(cands, key=lambda v: Fn.softmax(torch.tensor(v), 0)[int(cx.y[i])].item()))
            R[f'{tag}_bb{c}'] = mets(np.stack(worst))
        print('bbox', tag, f'{time.time() - t0:.0f}s', flush=True)
    N = range(len(XE))
    def red(i, k): t = top(i, (C + 1) * k); return [t[j::C + 1] if j < len(t) else t[:1] for j in range(C + 1)]
    multi('x_red', {i: red(i, Kc) for i in N}); multi(f'x_hyb{K}', {i: SS[i] + [top(i, K)] for i in N})
    for tag in ['single', 'x_red', f'x_hyb{K}']:
        for met in R['single']:
            v = [R[f'{tag}_bb{c}'][met] for c in range(C)]; R.setdefault(f'{tag}_bb_mean', {})[met] = float(np.mean(v)); R.setdefault(f'{tag}_bb_min', {})[met] = float(np.min(v))
    json.dump({'res': R, 'info': info}, open(out, 'w'), indent=1)
