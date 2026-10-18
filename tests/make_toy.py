"""Toy TCGA-style cohort with random features, for the smoke test (no slides or real features needed).
Each slide has a few 'evidence' tiles shifted toward its class in both the expensive (conch) and cheap (b0) feature spaces.
usage: python make_toy.py ROOT   -> ROOT/data/tcga_rcc/{manifest.tsv, feats/*.pt}"""
import sys, os, torch
root = sys.argv[1]; D = f'{root}/data/tcga_rcc'; os.makedirs(f'{D}/feats', exist_ok=True); os.makedirs(f'{root}/studentB_debate/results', exist_ok=True)
PR = ['TCGA-KIRC', 'TCGA-KIRP', 'TCGA-KICH']; g = torch.Generator().manual_seed(0)
ME, MC = torch.randn(3, 64, generator=g) * 2, torch.randn(3, 32, generator=g) * 2
with open(f'{D}/manifest.tsv', 'w') as f:
    f.write('file_name\tproject\tcase\n')
    for i in range(45):
        c = i % 3; n = f'TCGA-T{c}-{i:04d}-01Z-00-DX1'; t = 40 + i % 7; ev = torch.randperm(t, generator=g)[:4]
        E, Cf = torch.randn(t, 64, generator=g), torch.randn(t, 32, generator=g); E[ev] += ME[c]; Cf[ev] += MC[c]
        torch.save({'conch': E, 'b0': Cf}, f'{D}/feats/{n}.pt'); f.write(f'{n}.svs\t{PR[c]}\tTCGA-T{c}-{i:04d}\n')
print('toy cohort at', D)
