"""Student B: GPU-batched evidence-debate library (BCNB / CAMELYON16 patch features).

Judge = gated-attention MIL over revealed patches; input = feature + presenter flag (+1 side 1, -1 side 0, 0 neutral).
Greedy debaters use an exact incremental form of the attention pool, so one move costs O(N) per slide:
  logit(rev + j) = c( (S*exp(M0-m) + exp(a_j-m) z_j) / (W*exp(M0-m) + exp(a_j-m)) ),  m = max(M0, a_j).
"""
import os, math, numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn
from sklearn.metrics import roc_auc_score

DEV = os.environ.get('DEV', 'cuda' if torch.cuda.is_available() else 'cpu')
torch.set_num_threads(int(os.environ.get('NTHREADS', 4)))
NEG = -1e9


# ----------------------------------------------------------------------------------------------------- data
def load_bcnb():
    d = torch.load(os.environ.get('ADV_ROOT', '/scratch/ciro/cvpr_lab') + "/data/bcnb_conch_cache_A.pt")  # Student A's cache of D:\wsiagent\data\BCNB_patch_feat (read-only reuse)
    return d['feats'], d['lab'], d['slides']


def pad(xs, dev=DEV):
    """list of (n_i, d) -> (B, N, d) padded, (B, N) bool valid"""
    N = max(len(x) for x in xs); d = xs[0].shape[1]
    X = torch.zeros(len(xs), N, d, device=dev); M = torch.zeros(len(xs), N, dtype=torch.bool, device=dev)
    for i, x in enumerate(xs): X[i, :len(x)] = x.to(dev); M[i, :len(x)] = True
    return X, M


def balacc(p, y):
    p, y = np.asarray(p), np.asarray(y)
    return float(np.mean([np.mean(p[y == c] == c) for c in (0, 1) if (y == c).any()]))


def metrics(logit, y):
    logit, y = np.asarray(logit, dtype=float), np.asarray(y)
    ok = ~np.isnan(logit); logit, y = logit[ok], y[ok]
    p = (logit > 0).astype(int)
    auc = float(roc_auc_score(y, logit)) if len(set(y)) == 2 else float('nan')
    return {'balacc': balacc(p, y), 'acc': float(np.mean(p == y)), 'auc': auc, 'n': int(len(y))}


# ----------------------------------------------------------------------------------------------------- judges
class Judge(nn.Module):
    """arch: 'gated' (pilot ABMIL), 'mean' (MLP + mean pool), 'max' (MLP + max pool), 'tf' (1-layer transformer + attn pool)."""
    def __init__(s, d=513, h=128, arch='gated', drop=0.1):
        super().__init__(); s.arch = arch
        s.f = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Dropout(drop))
        if arch in ('gated', 'tf'):
            s.a = nn.Linear(h, h); s.b = nn.Linear(h, h); s.w = nn.Linear(h, 1)
        if arch == 'tf':
            s.t = nn.TransformerEncoderLayer(h, 4, 2 * h, drop, batch_first=True)
        s.c = nn.Linear(h, 1)

    def terms(s, x):  # per-patch embedding and attention score (gated arch only)
        z = s.f(x); return z, s.w(torch.tanh(s.a(z)) * torch.sigmoid(s.b(z))).squeeze(-1)

    def forward(s, x, mask=None):
        if mask is None: mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        z = s.f(x)
        if s.arch == 'tf': z = s.t(z, src_key_padding_mask=~mask)
        if s.arch in ('gated', 'tf'):
            a = s.w(torch.tanh(s.a(z)) * torch.sigmoid(s.b(z))).squeeze(-1).masked_fill(~mask, NEG).softmax(1)
            p = (a[..., None] * z).sum(1)
        elif s.arch == 'mean':
            p = (z * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        elif s.arch == 'max':
            p = z.masked_fill(~mask[..., None], NEG).max(1).values
        return s.c(p).squeeze(-1)


def train_judge(sets, ys, epochs=20, judge=None, bs=16, lr=5e-4, seed=0, arch='gated', masks=None):
    """sets: list of (k_i, 513) tensors or a (n, K, 513) tensor. Class-balanced BCE -> prior-neutral judge."""
    g = torch.Generator().manual_seed(seed); torch.manual_seed(seed)
    d_in = (sets.shape[-1] if torch.is_tensor(sets) else sets[0].shape[-1])
    j = judge or Judge(d=d_in, arch=arch).to(DEV)
    opt = torch.optim.AdamW(j.parameters(), lr, weight_decay=1e-4)
    ys = np.asarray(ys, dtype=float); pos = ys.mean()
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(1 - pos) / max(pos, 1e-3)], device=DEV))
    Y = torch.tensor(ys, dtype=torch.float32, device=DEV)
    fixed = torch.is_tensor(sets)
    for _ in range(epochs):
        j.train()
        for idx in torch.randperm(len(ys), generator=g).split(bs):
            if fixed:
                x = sets[idx].to(DEV); m = None if masks is None else masks[idx].to(DEV)
            else:
                x, m = pad([sets[i] for i in idx.tolist()])
            loss = lossf(j(x, m), Y[idx.to(DEV)]); opt.zero_grad(); loss.backward(); opt.step()
    return j.eval()


@torch.no_grad()
def judge_logits(j, sets, bs=64, masks=None):
    out = []
    for i in range(0, len(sets), bs):
        if torch.is_tensor(sets):
            out.append(j(sets[i:i + bs].to(DEV), None if masks is None else masks[i:i + bs].to(DEV)))
        else:
            out.append(j(*pad(sets[i:i + bs])))
    return torch.cat(out).cpu().numpy()


def flag(x, f):
    return torch.cat([x, torch.full((*x.shape[:-1], 1), float(f), device=x.device)], -1)


@torch.no_grad()
def attention_scores(j, x):  # x: (n, 512) one slide, neutral flag
    return j.terms(flag(x.to(DEV), 0))[1]


# ----------------------------------------------------------------------------------------------------- game state
class State:
    """Batched game on B slides. Tracks the judge's pooled state for the revealed set exactly."""
    def __init__(s, j, X, M, y=None):
        s.j, s.X, s.M, s.y = j, X, M, y
        B, N, _ = X.shape
        with torch.no_grad():
            s.Z, s.A = {}, {}
            for side, fv in ((1, 1.0), (0, -1.0)):
                z, a = j.terms(flag(X, fv)); s.Z[side], s.A[side] = z, a.masked_fill(~M, NEG)
            s.Zn, s.An = j.terms(flag(X, 0.0)); s.An = s.An.masked_fill(~M, NEG)  # neutral flag (cooperative selectors)
        h = s.Z[1].shape[-1]
        s.used = ~M.clone(); s.S = torch.zeros(B, h, device=X.device); s.W = torch.zeros(B, device=X.device)
        s.M0 = torch.full((B,), -float('inf'), device=X.device); s.idx, s.flags = [], []

    def clone(s):
        n = object.__new__(State); n.__dict__.update(s.__dict__)
        n.used, n.S, n.W, n.M0 = s.used.clone(), s.S.clone(), s.W.clone(), s.M0.clone()
        n.idx, n.flags = list(s.idx), list(s.flags); return n

    def terms_for(s, fl):  # fl: (B,) in {+1,-1,0}
        Z = torch.where((fl == 1)[:, None, None], s.Z[1], torch.where((fl == -1)[:, None, None], s.Z[0], s.Zn))
        A = torch.where((fl == 1)[:, None], s.A[1], torch.where((fl == -1)[:, None], s.A[0], s.An))
        return Z, A

    @torch.no_grad()
    def cand_logits(s, fl):
        """(B, N) judge logit if each candidate is revealed next with flag fl; used/invalid -> nan."""
        Z, A = s.terms_for(fl)
        m = torch.maximum(s.M0[:, None], A)
        e0 = torch.exp(s.M0[:, None] - m); e1 = torch.exp(A - m)
        num = e0[..., None] * s.S[:, None, :] + e1[..., None] * Z
        lg = s.j.c(num / (e0 * s.W[:, None] + e1)[..., None]).squeeze(-1)
        return lg.masked_fill(s.used, float('nan'))

    def add(s, pick, fl):
        B = torch.arange(len(pick), device=pick.device)
        Z, A = s.terms_for(fl)
        z, a = Z[B, pick], A[B, pick]
        m = torch.maximum(s.M0, a)
        e0 = torch.exp(s.M0 - m); e1 = torch.exp(a - m)
        s.S = e0[:, None] * s.S + e1[:, None] * z; s.W = e0 * s.W + e1; s.M0 = m
        s.used[B, pick] = True; s.idx.append(pick); s.flags.append(fl)

    @torch.no_grad()
    def logit(s):
        return s.j.c(s.S / s.W[:, None]).squeeze(-1)

    def transcript(s, blank=False):
        B = torch.arange(s.X.shape[0], device=s.X.device)
        xs = torch.stack([s.X[B, i] for i in s.idx], 1)
        if blank: xs = torch.zeros_like(xs)
        return torch.cat([xs, torch.stack(s.flags, 1)[..., None].float()], -1)  # (B, T, 513)

    def lesion_hits(s, P):  # P: (B, N) patch labels -> (B, T) labels of revealed patches
        B = torch.arange(s.X.shape[0], device=s.X.device)
        return torch.stack([P[B, i] for i in s.idx], 1)


# ----------------------------------------------------------------------------------------------------- selection policies
# A policy is a callable pol(st, side) -> (B,) patch index, where side in {1, 0} is the label it argues for.
# Flags: side 1 -> +1, side 0 -> -1. Cooperative/attack selectors use the neutral flag (handled by callers).
def greedy(st, side, fl=None):
    fl = fl if fl is not None else torch.full((st.X.shape[0],), 1.0 if side == 1 else -1.0, device=st.X.device)
    lg = st.cand_logits(fl)
    lg = torch.nan_to_num(lg, nan=-float('inf') if side == 1 else float('inf'))
    return lg.argmax(1) if side == 1 else lg.argmin(1)


def confident(st, side, fl):
    """V-IP-style label-blind cooperative querier: reveal the patch that makes the judge most confident (max |logit|)."""
    lg = torch.nan_to_num(st.cand_logits(fl).abs(), nan=-1.0)
    return lg.argmax(1)


def random_pol(st, side, fl=None):
    r = torch.rand(st.used.shape, device=st.X.device).masked_fill(st.used, -1.0)
    return r.argmax(1)


def side_flag(st, side):
    return torch.full((st.X.shape[0],), 1.0 if side == 1 else -1.0, device=st.X.device)


def make_rollout(M=8):
    """Stronger player: for its top-M greedy candidates, simulate the rest of the debate with both sides greedy,
    and pick the candidate with the best final logit for its side (one-step lookahead with full greedy rollout)."""
    def pol(st, side, fl=None):
        fl = side_flag(st, side) if fl is None else fl
        lg = st.cand_logits(fl)
        lg = torch.nan_to_num(lg, nan=-float('inf') if side == 1 else float('inf'))
        top = (lg if side == 1 else -lg).topk(min(M, st.X.shape[1]), 1).indices  # (B, M)
        finals = []
        for m in range(top.shape[1]):
            c = st.clone(); c.add(top[:, m], fl)
            for t in range(len(c.idx), st.T):
                s_t = st.order[t]  # (B,) side to move
                p1 = greedy(c, 1); p0 = greedy(c, 0)
                c.add(torch.where(s_t == 1, p1, p0), torch.where(s_t == 1, 1.0, -1.0).float())
            finals.append(c.logit())
        F = torch.stack(finals, 1)  # (B, M)
        best = F.argmax(1) if side == 1 else F.argmin(1)
        return top[torch.arange(len(best), device=best.device), best]
    return pol


class Mixed:
    """Per-slide policy choice: liar policy on slides where this side's label is wrong, honest policy otherwise."""
    def __init__(s, honest, liar): s.h, s.l = honest, liar
    def __call__(s, st, side, fl=None):
        ph = s.h(st, side); pl = s.l(st, side) if s.l is not s.h else ph
        return torch.where(st.y == side, ph, pl)


@torch.no_grad()
def play_debate(j, X, M, K_side, pol1=greedy, pol0=greedy, y=None, first=None, gen=None):
    """Alternating debate, K_side reveals per side. first: (B,) side that moves first (default random)."""
    B = X.shape[0]
    st = State(j, X, M, None if y is None else torch.as_tensor(y, device=X.device))
    if first is None:
        first = (torch.rand(B, generator=gen) < 0.5).long().to(X.device) if gen is not None else torch.randint(0, 2, (B,), device=X.device)
    st.T = 2 * K_side
    st.order = [first if t % 2 == 0 else 1 - first for t in range(st.T)]
    for t in range(st.T):
        s_t = st.order[t]
        p1 = pol1(st, 1); p0 = pol0(st, 0)
        st.add(torch.where(s_t == 1, p1, p0), torch.where(s_t == 1, 1.0, -1.0).float())
    return st


@torch.no_grad()
def play_single(j, X, M, K, target, fl_val, y=None, pol=greedy):
    """One selector picks all K patches pushing toward `target` (B,) with presenter flag fl_val (B,) (0 = neutral)."""
    st = State(j, X, M, None if y is None else torch.as_tensor(y, device=X.device))
    st.T = K; st.order = [target] * K
    for _ in range(K):
        p1 = pol(st, 1, fl_val); p0 = pol(st, 0, fl_val)
        st.add(torch.where(target == 1, p1, p0), fl_val)
    return st


def batched(fn, xs, bs=64, budget=None):
    """Run a game fn(X, M, sl) on chunks of slides sorted by size (chunk = bs slides, or <= budget padded patches);
    returns list of results in original order."""
    n = [len(x) for x in xs]; order = np.argsort(n); res = [None] * len(xs); chunks, cur = [], []
    for k in order:
        if cur and (len(cur) >= bs or (budget and (len(cur) + 1) * n[k] > budget)): chunks.append(cur); cur = []
        cur.append(k)
    if cur: chunks.append(cur)
    for ids in chunks:
        X, M = pad([xs[k].float() for k in ids])
        out = fn(X, M, ids)
        for r, k in enumerate(ids): res[k] = None if out is None else tuple(o[r] for o in out)
    return res
