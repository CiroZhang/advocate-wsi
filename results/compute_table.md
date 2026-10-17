# Accuracy vs compute per cancer type
Accuracy vs. per-slide compute for every method and cancer type (mean over runs).
Compute model (A100, seconds/slide, measured during feature extraction / io_bench; tiling = tissue mask + grid):
  full slide        tiling + full-res read(all) + FM(all)
  cheap full slide  tiling + low-mag read(all) + B0(all)
  random K          tiling + K * per-tile(full-res read + FM)
  ADMIL / ours      tiling + low-mag read(all) + B0(all) + K * per-tile(full-res read + FM)
  + escalate f      above + f * (full-res read(all) + FM(all))     [lowest-confidence fraction f -> full-slide model]
usage: python compute_table.py  (reads ../results, writes ../results/compute_table.{md,json})
### C16: breast LN metastasis (2-class) (BA, 3 runs)

| method | BA | attacked | s/slide | speed-up |
|---|---|---|---|---|
| full slide (UNI, all tiles) | 0.969 |  | 33.3 | 1.0x |
| cheap full slide (B0, all tiles) | 0.713 |  | 5.4 | 6.1x |
| UNI on 16 random tiles (trained on random) | 0.666 |  | 0.0 | 761.4x |
| UNI on 64 random tiles (trained on random) | 0.688 |  | 0.2 | 190.3x |
| UNI on 256 random tiles (trained on random) | 0.785 |  | 0.7 | 47.6x |
| ADMIL-style (16 tiles) | 0.943 | 0.315 | 5.5 | 6.1x |
| ours (16 tiles) | 0.942 | 0.931 | 5.5 | 6.1x |
| ADMIL-style + escalate 10% | 0.946 |  | 8.8 | 3.8x |
| ours + escalate 10% | 0.935 |  | 8.8 | 3.8x |
| ADMIL-style + escalate 20% | 0.956 |  | 12.1 | 2.7x |
| ours + escalate 20% | 0.942 |  | 12.1 | 2.7x |

### C17: breast LN stage (4-class, ordinal) (kappa, 15 runs)

| method | kappa | attacked | s/slide | speed-up |
|---|---|---|---|---|
| full slide (CONCH, all tiles) | 0.890 |  | 24.4 | 1.0x |
| cheap full slide (B0, all tiles) | 0.567 |  | 6.6 | 3.7x |
| CONCH on 16 random tiles (trained on random) | 0.669 |  | 2.8 | 8.6x |
| CONCH on 64 random tiles (trained on random) | 0.816 |  | 3.3 | 7.3x |
| CONCH on 256 random tiles (trained on random) | 0.869 |  | 5.3 | 4.6x |
| ADMIL-style (16 tiles) | 0.873 | -0.241 | 6.7 | 3.6x |
| ours (16 tiles) | 0.823 | 0.715 | 6.7 | 3.6x |
| ADMIL-style + escalate 10% | 0.878 |  | 8.9 | 2.7x |
| ours + escalate 10% | 0.835 |  | 8.9 | 2.7x |
| ADMIL-style + escalate 20% | 0.881 |  | 11.1 | 2.2x |
| ours + escalate 20% | 0.845 |  | 11.1 | 2.2x |

### RCC: kidney RCC subtype (3-class) (BA, 15 runs)

| method | BA | attacked | s/slide | speed-up |
|---|---|---|---|---|
| full slide (CONCH, all tiles) | 0.933 |  | 32.4 | 1.0x |
| cheap full slide (B0, all tiles) | 0.898 |  | 7.8 | 4.1x |
| CONCH on 12 random tiles (trained on random) | 0.920 |  | 1.7 | 19.0x |
| CONCH on 48 random tiles (trained on random) | 0.929 |  | 2.0 | 16.6x |
| CONCH on 192 random tiles (trained on random) | 0.939 |  | 2.9 | 11.0x |
| ADMIL-style (12 tiles) | 0.907 | 0.022 | 7.9 | 4.1x |
| ours (12 tiles) | 0.920 | 0.866 | 7.9 | 4.1x |
| ADMIL-style + escalate 10% | 0.922 |  | 11.0 | 2.9x |
| ours + escalate 10% | 0.936 |  | 11.0 | 2.9x |
| ADMIL-style + escalate 20% | 0.927 |  | 14.1 | 2.3x |
| ours + escalate 20% | 0.934 |  | 14.1 | 2.3x |

### NSCLC: lung NSCLC subtype (2-class) (BA, 15 runs)

| method | BA | attacked | s/slide | speed-up |
|---|---|---|---|---|
| full slide (CONCH, all tiles) | 0.923 |  | 26.3 | 1.0x |
| cheap full slide (B0, all tiles) | 0.850 |  | 6.6 | 4.0x |
| CONCH on 8 random tiles (trained on random) | 0.895 |  | 1.4 | 18.5x |
| CONCH on 32 random tiles (trained on random) | 0.917 |  | 1.6 | 16.7x |
| CONCH on 128 random tiles (trained on random) | 0.918 |  | 2.2 | 12.1x |
| ADMIL-style (8 tiles) | 0.909 | 0.003 | 6.6 | 4.0x |
| ours (8 tiles) | 0.874 | 0.841 | 6.6 | 4.0x |
| ADMIL-style + escalate 10% | 0.919 |  | 9.1 | 2.9x |
| ours + escalate 10% | 0.899 |  | 9.1 | 2.9x |
| ADMIL-style + escalate 20% | 0.921 |  | 11.6 | 2.3x |
| ours + escalate 20% | 0.908 |  | 11.6 | 2.3x |
