"""Soundness pilot: is the judge manipulable by a dishonest evidence selector?
(a) cooperative judge trained on attention-top-K patches; at test time an ADVERSARY chooses all K patches to push the judge to the WRONG label.
(b) consultancy judge (one prover, random side during training); test with the prover always arguing the WRONG label.
(c) debate judge (as in debate_pilot) with BOTH sides best-responding: the liar argues the wrong label, the honest side the right one.
Report accuracy (= honest outcome rate). Real BCNB CONCH features, 5-fold CV."""
import os, sys, json, numpy as np, pandas as pd, torch, collections as C
sys.argv = [sys.argv[0]]; exec(open('debate_pilot.py').read().split('res = {}')[0])  # reuse data, Judge, train_judge, debate, helpers
torch.set_num_threads(10)
@torch.no_grad()
def adversary(j, x, K, target, flag):  # greedy: choose K patches maximizing judge prob of `target`
    used = set(); rev = []
    for _ in range(K):
        cand = [i for i in range(x.shape[0]) if i not in used]
        base = torch.stack(rev) if rev else torch.zeros(0, 513)
        sets = torch.stack([torch.cat([base, with_flag(x[i:i + 1], flag)], 0) for i in cand])
        lg = j(sets); pick = cand[int(lg.argmax() if target == 1 else lg.argmin())]
        used.add(pick); rev.append(with_flag(x[pick:pick + 1], flag)[0])
    return torch.stack(rev)
res = {}
for task in ['ER', 'Grade3', 'PR']:
    d = lab[task]; ss = [s for s in slides if s in d]; y = np.array([d[s] for s in ss]); folds = np.array_split(rng.permutation(len(ss)), 5)
    for K in [4, 8]:
        R = C.defaultdict(lambda: ([], []))
        for k in range(5):
            te = folds[k]; tr = np.concatenate([folds[m] for m in range(5) if m != k])
            full = train_judge([with_flag(feats[ss[i]], 0) for i in tr], y[tr], epochs=10)
            def topk(i):
                x = with_flag(feats[ss[i]], 0)
                with torch.no_grad():
                    z = full.f(x); a = full.w(torch.tanh(full.a(z)) * torch.sigmoid(full.b(z))).squeeze(-1)
                return x[a.topk(min(K, len(x))).indices]
            jc = train_judge([topk(i) for i in tr], y[tr], epochs=10)
            # consultancy: prover assigned a random side, best-responds to the current judge; judge retrained on its transcripts
            jcons = train_judge([with_flag(feats[ss[i]], 0)[torch.randperm(len(feats[ss[i]]))[:K]] for i in tr], y[tr], epochs=5)
            for r in range(2):
                sides = rng.integers(0, 2, len(tr))
                trans = [adversary(jcons, feats[ss[i]], K, int(s), 1.0 if s == 1 else -1.0) for i, s in zip(tr, sides)]
                jcons = train_judge(trans, y[tr], epochs=6, judge=jcons)
            jd = train_judge([with_flag(feats[ss[i]], 0)[torch.randperm(len(feats[ss[i]]))[:K]] for i in tr], y[tr], epochs=5)
            for r in range(2):
                jd = train_judge([debate(jd, feats[ss[i]], K) for i in tr], y[tr], epochs=6, judge=jd)
            with torch.no_grad():
                for i in te:
                    yt = int(y[i]); wrong = 1 - yt
                    R['coop_honest'][0].append(int(jc(topk(i)[None]).item() > 0)); R['coop_honest'][1].append(yt)
                    R['coop_attacked'][0].append(int(jc(adversary(jc, feats[ss[i]], K, wrong, 0.0)[None]).item() > 0)); R['coop_attacked'][1].append(yt)
                    R['consult_liar'][0].append(int(jcons(adversary(jcons, feats[ss[i]], K, wrong, 1.0 if wrong == 1 else -1.0)[None]).item() > 0)); R['consult_liar'][1].append(yt)
                    R['consult_honest'][0].append(int(jcons(adversary(jcons, feats[ss[i]], K, yt, 1.0 if yt == 1 else -1.0)[None]).item() > 0)); R['consult_honest'][1].append(yt)
                    R['debate'][0].append(int(jd(debate(jd, feats[ss[i]], K)[None]).item() > 0)); R['debate'][1].append(yt)
            print(task, K, 'fold', k, {m: round(balacc(*v), 3) for m, v in R.items()}, flush=True)
        res[f'{task}_K{K}'] = {m: {'acc': float(np.mean(np.array(v[0]) == np.array(v[1]))), 'balacc': float(balacc(*v))} for m, v in R.items()}
        json.dump(res, open(r"D:\wsiagent\attack_pilot.json", 'w'), indent=1)
