"""Teaching rounds with a restricted teacher on CAMELYON16 (same data/encoders as exp_c16w.py).
Two students see the whole slide; each argues one label (the one arguing the wrong label is the devil's advocate) and presents the
patch that most moves the teacher toward its label (greedy, white-box). The teacher (gated ABMIL over presented patches + presenter flag)
sees ONLY presented patches. Every protocol opens with one patch per student (random order); cap TMAX presented patches.
  A_K          : no questions; both students present until K patches (K = 2, 4, 8, 16)
  trail(tau)   : while teacher confidence < tau, the trailing student presents its next patch
  both(tau)    : while < tau, both students present their next patch
  verify(tau)  : while < tau, the teacher points at the leading student's most-attended patch and asks both students to show its
                 surroundings: trailing then leading student each present a neighbour (8-neighbourhood) of their choice
  rand(tau)    : control, while < tau, two random patches (no student choice)
  learned(lam) : learned question policy over {stop, trail, both, verify}: per-action value regression (myopic gain in log p(truth),
                 label-free features of the teacher state) on held-out training slides; act = argmax(value - lam * patches)
Stress 'weak': the honest student picks randomly (devil's advocate stays greedy).
The teacher and the policy are trained on 75% / 25% of the training slides; full-slide ABMIL (all training slides) is the reference.
usage: python exp_teach.py SEED"""
import sys, json, time, csv, numpy as np, torch
sys.path.insert(0, "/scratch/ciro/cvpr_lab/studentB_debate")
from common import *

seed = int(sys.argv[1]); R = int(os.environ.get('ROUNDS', 3)); EP_FULL, EP_SUB, EP_FT = 20, 40, 20; BUD = 400_000; TMAX = 16
FEAT = os.environ.get("FEAT", "features"); ETAG = os.environ.get("ETAG", "")
D = "/scratch/ciro/cvpr_lab/data/camelyon16/torchmil"; OUT = "/scratch/ciro/cvpr_lab/studentB_debate/results"
rng = np.random.default_rng(seed); torch.manual_seed(seed)
TAUS = [0.7, 0.8, 0.9, 0.95, 0.99]; LAMS = [0.0, 0.005, 0.01, 0.02, 0.05]

rows = list(csv.DictReader(open(os.path.join(D, 'splits.csv'))))
names = [r['bag_name'] for r in rows if os.path.exists(os.path.join(D, FEAT, r['bag_name'] + '.npy'))]
split = {r['bag_name']: r['split'] for r in rows}
t0 = time.time()
xs = [torch.from_numpy(np.load(os.path.join(D, FEAT, n + '.npy'))).half() for n in names]
P = [torch.from_numpy(np.load(os.path.join(D, 'patch_labels', n + '.npy')).astype(np.int64).reshape(-1)) for n in names]
GR = [torch.from_numpy(np.load(os.path.join(D, 'coords', n + '.npy')) // 512).long() for n in names]
y = np.array([int(np.load(os.path.join(D, 'labels', n + '.npy')).reshape(-1)[0]) for n in names])
tr = np.array([i for i, n in enumerate(names) if split[n] == 'train']); te = np.array([i for i, n in enumerate(names) if split[n] == 'test'])
perm = rng.permutation(tr); npol = len(tr) // 4; pol_ids, ttr = np.sort(perm[:npol]), np.sort(perm[npol:])
print(f'loaded {len(names)} in {time.time() - t0:.0f}s; teacher-train {len(ttr)} policy {len(pol_ids)} test {len(te)}', flush=True)
DIM = xs[0].shape[1]
def fl(x, f): return flag(x.float(), f)


# ----------------------------------------------------------------------------------------------------- game state with masked moves
class TS(State):
    def __init__(s, j, X, M, G, y):
        super().__init__(j, X, M, y); B = X.shape[0]; dv = X.device
        s.G = G; s.ri = torch.zeros(B, 0, dtype=torch.long, device=dv); s.rf = torch.zeros(B, 0, device=dv)
        s.rm = torch.zeros(B, 0, dtype=torch.bool, device=dv); s.n = torch.zeros(B, dtype=torch.long, device=dv)
        s.prev = torch.zeros(B, device=dv)

    def clone(s):
        n = object.__new__(TS); n.__dict__.update(s.__dict__)
        n.used, n.S, n.W, n.M0, n.idx, n.flags = s.used.clone(), s.S.clone(), s.W.clone(), s.M0.clone(), [], []
        for k in ('ri', 'rf', 'rm', 'n', 'prev'): setattr(n, k, getattr(s, k).clone())
        return n

    def add_m(s, pick, f, act):
        Bi = torch.arange(len(pick), device=pick.device)
        Z, A = s.terms_for(f); z, a = Z[Bi, pick], A[Bi, pick]
        m = torch.maximum(s.M0, a); e0 = torch.exp(s.M0 - m); e1 = torch.exp(a - m)
        s.S = torch.where(act[:, None], e0[:, None] * s.S + e1[:, None] * z, s.S)
        s.W = torch.where(act, e0 * s.W + e1, s.W); s.M0 = torch.where(act, m, s.M0)
        s.used[Bi[act], pick[act]] = True
        s.ri = torch.cat([s.ri, pick[:, None]], 1); s.rf = torch.cat([s.rf, f[:, None]], 1); s.rm = torch.cat([s.rm, act[:, None]], 1)
        s.n = s.n + act.long()

    def tr_out(s):  # transcript (B, T, DIM+1), mask (B, T)
        Bi = torch.arange(s.X.shape[0], device=s.X.device)[:, None]
        return torch.cat([s.X[Bi, s.ri], s.rf[..., None]], -1), s.rm

    def rev_att(s):  # teacher attention logits of presented patches (B, T), -inf where not presented
        a = torch.where(s.rf == 1, s.A[1].gather(1, s.ri), s.A[0].gather(1, s.ri))
        return a.masked_fill(~s.rm, -float('inf'))


def pick_for(st, side, allow, weak):
    """side (B,) label argued; allow (B, N) candidate mask; weak (B,) bool -> random pick among allowed."""
    lg = st.cand_logits(side.float() * 2 - 1)
    ok = (allow & ~st.used).any(1)
    lg = torch.where(ok[:, None] & ~allow, torch.full_like(lg, float('nan')), lg)
    sc = torch.nan_to_num(torch.where(side[:, None] == 1, lg, -lg), nan=-float('inf'))
    pk = sc.argmax(1)
    r = torch.rand_like(sc).masked_fill(torch.isinf(sc), -1.0)
    return torch.where(weak, r.argmax(1), pk)


def nb_allow(st):
    """Neighbourhood of the leading student's most-attended presented patch."""
    lead = (st.logit() > 0).long(); lf = lead.float() * 2 - 1
    a = st.rev_att().masked_fill(st.rf != lf[:, None], -float('inf'))
    has = torch.isfinite(a).any(1); a = torch.where(has[:, None], a, st.rev_att())
    p = st.ri.gather(1, a.argmax(1, keepdim=True)).squeeze(1)
    Bi = torch.arange(len(p), device=p.device)
    g = st.G[Bi, p]; d = (st.G - g[:, None]).abs().max(-1).values
    return d == 1


def step(st, a, weak_mode):
    """a (B,): 0 stop, 1 trail, 2 both, 3 verify, 4 random control. Presentations beyond TMAX are dropped."""
    lead = (st.logit() > 0).long(); trail = 1 - lead; free = ~st.used
    weak = lambda side: (side == st.y) & weak_mode
    allow = torch.where((a == 3)[:, None], nb_allow(st) & free, free)
    rnd = a == 4
    for side, act in ((trail, (a >= 1) & (a <= 3)), (lead, (a == 2) | (a == 3))):
        act = act & (st.n < TMAX)
        if act.any(): st.add_m(pick_for(st, side, allow & ~st.used, weak(side)), side.float() * 2 - 1, act)
    for _ in range(2):
        act = rnd & (st.n < TMAX)
        if act.any():
            r = torch.rand(st.used.shape, device=DEV).masked_fill(st.used, -1.0).argmax(1)
            st.add_m(r, (torch.randint(0, 2, (len(r),), device=DEV) * 2 - 1).float(), act)


def opening(st, weak_mode):
    B = st.X.shape[0]; first = torch.randint(0, 2, (B,), device=DEV); one = torch.ones(B, dtype=torch.bool, device=DEV)
    for side in (first, 1 - first):
        st.add_m(pick_for(st, side, ~st.used, (side == st.y) & weak_mode), side.float() * 2 - 1, one)


def state_feats(st):
    lg = st.logit(); n = st.n.float()
    w = st.rev_att().softmax(1).nan_to_num(0.0); lf = ((lg > 0).float() * 2 - 1)[:, None]
    wl = (w * (st.rf == lf)).sum(1); ent = -(w * (w + 1e-9).log()).sum(1)
    nl = ((st.rf == lf) & st.rm).sum(1).float()
    return torch.stack([lg.abs(), n / TMAX, nl / TMAX, (n - nl) / TMAX, wl, ent, w.max(1).values, (lg.abs() - st.prev).abs()], 1)


class Policy(nn.Module):
    def __init__(s): super().__init__(); s.f = nn.Sequential(nn.Linear(8, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 3))
    def forward(s, x): return s.f(x)  # predicted gain of trail / both / verify (stop = 0)


COST = torch.tensor([0., 1., 2., 2.])


def run(j, ids, mode, param=None, weak_mode=False, policy=None, collect=False, behaviour=False):
    """Returns per-slide logit, #presented, transcripts (padded), masks, lesion fraction of presented patches by the tumor-arguing student;
    with collect=True also (features, targets) for policy training."""
    sub = [xs[i] for i in ids]; yy = y[ids]; Gs = [GR[i] for i in ids]; Ps = [P[i] for i in ids]; F, Tg = [], []
    def g(X, M, k):
        G = torch.full((X.shape[0], X.shape[1], 2), -10 ** 6, dtype=torch.long, device=DEV)
        Pp = torch.zeros(X.shape[0], X.shape[1], device=DEV)
        for r, q in enumerate(k): G[r, :len(Gs[q])] = Gs[q].to(DEV); Pp[r, :len(Ps[q])] = Ps[q].float().to(DEV)
        yb = torch.as_tensor(yy[k], device=DEV); wm = torch.full((len(k),), weak_mode, dtype=torch.bool, device=DEV)
        st = TS(j, X, M, G, yb); opening(st, wm)
        if behaviour: stop_at = torch.randint(2, TMAX + 1, (len(k),), device=DEV)
        for _ in range(TMAX):
            c = torch.sigmoid(st.logit().abs()); go = st.n < TMAX
            if mode == 'A': a = torch.where(st.n < param, 2, 0)
            elif mode in ('trail', 'both', 'verify', 'rand'):
                a = torch.where((c < param) & go, {'trail': 1, 'both': 2, 'verify': 3, 'rand': 4}[mode], 0)
            elif mode == 'learned':
                v = torch.cat([torch.zeros(len(k), 1, device=DEV), policy(state_feats(st))], 1) - param * COST.to(DEV)
                a = torch.where(go, v.argmax(1), 0)
            elif mode == 'behaviour':
                a = torch.where(go & (st.n < stop_at), torch.randint(1, 4, (len(k),), device=DEV), 0)
            if collect:
                yt = yb.float() * 2 - 1; base = Fn.logsigmoid(st.logit() * yt).clamp(min=-4)
                gains = []
                for aa in (1, 2, 3):
                    c2 = st.clone(); step(c2, torch.full_like(a, aa), wm)
                    gains.append(Fn.logsigmoid(c2.logit() * yt).clamp(min=-4) - base)
                ok = go & (a > 0) if mode == 'behaviour' else go
                F.append(state_feats(st)[ok].cpu()); Tg.append(torch.stack(gains, 1)[ok].cpu())
            if not (a > 0).any(): break
            st.prev = st.logit().abs(); step(st, a, wm)
        Tr, Mk = st.tr_out()
        pos = (st.rf == 1) & st.rm; hit = (Pp.gather(1, st.ri) * pos).sum(1) / pos.sum(1).clamp(min=1)
        return st.logit(), st.n, Tr.cpu(), Mk.cpu(), hit
    res = batched(g, sub, bs=32, budget=BUD)
    T = max(r[2].shape[0] for r in res)
    padT = lambda t, v: torch.cat([t, torch.full((T - t.shape[0], *t.shape[1:]), v, dtype=t.dtype)], 0)
    out = dict(logit=np.array([r[0].item() for r in res]), n=np.array([r[1].item() for r in res]),
               T=torch.stack([padT(r[2], 0.0) for r in res]), M=torch.stack([padT(r[3], False) for r in res]),
               hit=np.array([r[4].item() for r in res]))
    if collect: out['F'], out['G'] = torch.cat(F), torch.cat(Tg)
    return out


def prefix_masks(Mk, gen):
    """Random prefix truncation (>= 2 presented) of each transcript mask."""
    cs = Mk.long().cumsum(1); tot = cs[:, -1]
    L = (2 + (torch.rand(len(Mk), generator=gen) * (tot - 1).clamp(min=1).float()).long()).clamp(max=tot)
    return Mk & (cs <= L[:, None])


# ----------------------------------------------------------------------------------------------------- full-slide reference + warm start
full = train_judge([fl(xs[i], 0) for i in tr], y[tr], epochs=EP_FULL, bs=1, lr=2e-4, seed=seed)
L = {'full': judge_logits(full, [fl(xs[i], 0) for i in te], bs=1)}
print('full', metrics(L['full'], y[te]), f'{time.time() - t0:.0f}s', flush=True)
with torch.no_grad(): att = {i: full.terms(fl(xs[i].to(DEV), 0))[1].cpu() for i in range(len(names))}
def warm(i, K):
    x = xs[i]; p = att[i].topk(min(K, len(x))).indices if rng.random() < 0.5 else torch.randperm(len(x))[:K]
    return torch.cat([x[p].float(), (torch.randint(0, 2, (len(p), 1)) * 2 - 1).float()], 1)
jd = train_judge([warm(i, int(rng.choice([2, 4, 8, 16]))) for i in ttr for _ in range(2)], np.repeat(y[ttr], 2), epochs=EP_SUB, seed=seed)

# ---- on-policy teacher training on random question sequences (random prefixes)
gen = torch.Generator().manual_seed(seed); BX, BM, BY = [], [], []
for r in range(R):
    for _ in range(2):
        o = run(jd, ttr, 'behaviour', behaviour=True)
        BX.append(o['T']); BM.append(prefix_masks(o['M'], gen)); BY.append(y[ttr])
    T = max(b.shape[1] for b in BX)
    X_ = torch.cat([torch.cat([b, torch.zeros(b.shape[0], T - b.shape[1], b.shape[2])], 1) for b in BX])
    M_ = torch.cat([torch.cat([m, torch.zeros(m.shape[0], T - m.shape[1], dtype=torch.bool)], 1) for m in BM])
    jd = train_judge(X_, np.concatenate(BY), epochs=EP_FT, judge=jd, seed=seed + r, masks=M_)
    o = run(jd, ttr, 'A', 4); print(f'  round {r} train A4 balacc {balacc((o["logit"] > 0).astype(int), y[ttr]):.3f} {time.time() - t0:.0f}s', flush=True)

# ---- question policy on held-out training slides (two DAgger-style passes: behaviour, then own policy)
pol = Policy().to(DEV); Fs, Gs_ = [], []
for it in range(2):
    for wm in (False, True):
        o = run(jd, pol_ids, 'behaviour' if it == 0 else 'learned', param=0.01, weak_mode=wm, policy=pol, collect=True, behaviour=it == 0)
        Fs.append(o['F']); Gs_.append(o['G'])
    Fx, Gx = torch.cat(Fs).to(DEV), torch.cat(Gs_).to(DEV)
    opt = torch.optim.AdamW(pol.parameters(), 1e-3, weight_decay=1e-4)
    for ep in range(300):
        for idx in torch.randperm(len(Fx), device=DEV).split(256):
            loss = Fn.mse_loss(pol(Fx[idx]), Gx[idx]); opt.zero_grad(); loss.backward(); opt.step()
    print(f'  policy pass {it}: {len(Fx)} states, mse {loss.item():.4f}, mean gains {Gx.mean(0).tolist()}', flush=True)
pol.eval()

# ----------------------------------------------------------------------------------------------------- evaluation
RES = {}
with torch.no_grad():
    for wm in (False, True):
        tag = '_weak' if wm else ''
        prot = [('A', K, f'A_K{K}') for K in (2, 4, 8, 16)]
        prot += [(m, t, f'{m}_tau{t}') for m in ('trail', 'both', 'verify', 'rand') for t in TAUS]
        prot += [('learned', l, f'learned_lam{l}') for l in LAMS]
        for mode, prm, nm in prot:
            o = run(jd, te, mode, prm, weak_mode=wm, policy=pol)
            L[nm + tag] = o['logit']
            tum = y[te] == 1
            RES[nm + tag] = metrics(o['logit'], y[te]) | {'patches': float(o['n'].mean()), 'patches_sd': float(o['n'].std()),
                                                         'pos_lesion_frac_tumor': float(o['hit'][tum].mean())}
            print(nm + tag, {k: round(v, 3) for k, v in RES[nm + tag].items() if k in ('balacc', 'auc', 'patches', 'pos_lesion_frac_tumor')}, f'{time.time() - t0:.0f}s', flush=True)
RES['full'] = metrics(L['full'], y[te]) | {'patches': float(np.mean([len(xs[i]) for i in te]))}
json.dump({'res': RES, 'info': {'seed': seed, 'feat': FEAT, 'n_test': len(te), 'tmax': TMAX}},
          open(os.path.join(OUT, f'teach{ETAG}_s{seed}.json'), 'w'), indent=1)
np.savez(os.path.join(OUT, f'teach{ETAG}_logits_s{seed}.npz'), y=y[te], **L)
print('DONE', flush=True)
