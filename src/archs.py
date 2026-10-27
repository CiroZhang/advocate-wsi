"""Judge architectures for the generality experiment (ARCH=... in exp_tcga.py / exp_c17.py): the selector and attacks are unchanged,
only the MIL judge differs. Multi-class re-implementations (torchmil's reference models are binary-only):
  mean  : MLP + mean pooling            max   : MLP + max pooling (per class)
  trans : TransMIL-style, 2 transformer layers + class token (train bags capped at 2048 random tiles)
  dsmil : DSMIL-style, per-class critical instance + attention of every tile to it; logit = mean of max-instance and bag logits
forward(x [n, d]) -> [o]; batch(x [B, n, d]) -> [B, o] (equal-size bags, as the attacks use). A single-tile bag gives the per-tile score."""
import torch, torch.nn as nn

class MIL(nn.Module):
    def __init__(s, d, o, arch, h=128, drop=.1):
        super().__init__(); s.arch = arch; s.f = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Dropout(drop))
        if arch == 'trans':
            s.cls = nn.Parameter(torch.zeros(1, 1, h))
            s.t = nn.TransformerEncoder(nn.TransformerEncoderLayer(h, 4, 2 * h, drop, batch_first=True, norm_first=True), 2, enable_nested_tensor=False)
        if arch == 'dsmil':
            s.ic = nn.Linear(h, o); s.q = nn.Linear(h, h); s.v = nn.Linear(h, h); s.bc = nn.Conv1d(o, o, h, groups=o)
        s.c = nn.Linear(h, o)
    def batch(s, x):
        z = s.f(x)
        if s.arch == 'mean': return s.c(z.mean(1))
        if s.arch == 'max': return s.c(z).max(1).values
        if s.arch == 'trans':
            if s.training and z.shape[1] > 2048: z = z[:, torch.randperm(z.shape[1], device=z.device)[:2048]]
            return s.c(s.t(torch.cat([s.cls.expand(len(z), -1, -1), z], 1))[:, 0])
        if s.arch == 'dsmil':
            ins = s.ic(z); crit = z.gather(1, ins.argmax(1)[..., None].expand(-1, -1, z.shape[2]))  # [B, o, h]
            a = (s.q(z) @ s.q(crit).transpose(1, 2) / z.shape[2] ** .5).softmax(1)  # [B, n, o]
            b = a.transpose(1, 2) @ s.v(z)  # [B, o, h]
            return (ins.max(1).values + s.bc(b).squeeze(-1)) / 2
        raise ValueError(s.arch)
    def forward(s, x): return s.batch(x[None])[0]
