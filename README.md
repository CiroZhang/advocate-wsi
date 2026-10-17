# advocate-wsi

One Advocate per Diagnosis: robust cheap evidence selection for whole-slide image classification.

- `paper/` – two-column preprint draft (`latexmk -pdf main.tex`); figures in `paper/figs/`
- `analysis/` – tables and figures from `results/`: `compute_table.py` (accuracy vs cost), `fig_paper.py`, `tcga_agg.py`, `c17_cases.py`, `rcc_cases.py`, `cyto_pilot.py` (cervical go/no-go, private data, not in paper)
- `src/` – experiments: `exp_distill.py` (CAMELYON16), `exp_c17.py` (CAMELYON17), `exp_tcga.py` (TCGA-RCC/NSCLC), `exp_frontier.py` (matched-compute baselines), `exp_admil.py`, `mil_baselines.py` (published MIL), `enc_timing.py`/`io_bench.py` (cost); `src/extra/` = earlier variants not in the paper
- `data_prep/` – TCGA manifest/download, feature extraction (CONCH + EfficientNet-B0 cheap view), stain shift
- `slurm/` – Narval job scripts (paths are hard-coded to /scratch/ciro/cvpr_lab)
- `results/` – per-run JSON/NPZ (selected-tile `*_sel.pt` files are git-ignored, ~280 MB)

Regenerate: `cd analysis && python compute_table.py && python fig_paper.py`
