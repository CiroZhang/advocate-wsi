#!/usr/bin/env bash
# End-to-end smoke test on a toy cohort (CPU, ~1 min): advocates + judge + attacks + baselines, then the analysis scripts.
# Checks that the code runs, not the numbers. usage: bash tests/smoke_test.sh
set -euo pipefail
cd "$(dirname "$0")/.."
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
python tests/make_toy.py "$T"
ADV_ROOT="$T" CHEAP=b0 KC=2 python src/exp_tcga.py 0 0 rcc
python -c "import json,sys; r=json.load(open(sys.argv[1]))['res']; assert {'full_conch','teach','teach_wb_worst','admil','admil_wb_worst'} <= set(r); print('exp_tcga OK:', {k: round(v['balacc'], 2) for k, v in r.items()})" "$T/studentB_debate/results/tcga_rcc_b0_k2_f0_s0.json"
ADV_ROOT="$T" EXT=1 CHEAP=b0 KC=2 python src/exp_tcga.py 0 1 rcc >/dev/null && echo "exp_tcga EXT=1 OK"
ADV_ROOT="$T" EXT=1 CHEAP=b0 python src/exp_c17.py 0 0 >/dev/null && echo "exp_c17 EXT=1 OK"
ADV_ROOT="$T" EXT=1 ATTY=1 SUFX=_atty CHEAP=b0 KC=2 python src/exp_tcga.py 0 1 rcc >/dev/null && ls "$T"/studentB_debate/results/*_atty.json >/dev/null && echo "exp_tcga EXT=1 ATTY=1 OK"
ADV_ROOT="$T" BONLY=1 CHEAP=b0 KC=2 python src/exp_tcga.py 0 2 rcc >/dev/null && ls "$T"/studentB_debate/results/tcga_rcc_b0_k2_f2_s0_budget.json >/dev/null && echo "exp_tcga BONLY=1 OK"
ADV_ROOT="$T" BONLY=1 CHEAP=b0 python src/exp_c17.py 0 1 >/dev/null && ls "$T"/studentB_debate/results/c17_b0_k4_f1_s0_budget.json >/dev/null && echo "exp_c17 BONLY=1 OK"
ADV_ROOT="$T" EXT=1 RED=1 SUFX=_red CHEAP=b0 KC=2 python src/exp_tcga.py 0 3 rcc >/dev/null && python -c "import json,sys; r=json.load(open(sys.argv[1]))['res']; assert {'x_red','x_red_ad','x_rediso','x_rediso_ad','x_hybiso_ad','x_rediso_vote_ad','x_rediso_clip_ad'} <= set(r); print('RED OK', {k: round(v['balacc'],2) for k,v in r.items() if k.startswith('x_')})" "$T"/studentB_debate/results/tcga_rcc_b0_k2_f3_s0_ext_red.json
ADV_ROOT="$T" EXT=1 RED=1 SUFX=_red CHEAP=b0 python src/exp_c17.py 0 2 >/dev/null && echo "exp_c17 RED=1 OK"
for A in mean max trans dsmil; do ADV_ROOT="$T" ARCH=$A CHEAP=b0 KC=2 python src/exp_tcga.py 0 4 rcc >/dev/null && python -c "import json,sys; r=json.load(open(sys.argv[1]))['res']; assert {'full','single','single_ad','x_red','x_red_ad','x_hyb6_ad'} <= set(r), set(r); print('ARCH', sys.argv[2], {k: round(v['balacc'],2) for k,v in r.items()})" "$T"/studentB_debate/results/tcga_rcc_b0_k2_f4_s0_arch$A.json $A; done
ADV_ROOT="$T" ARCH=dsmil CHEAP=b0 python src/exp_c17.py 0 3 >/dev/null && ls "$T"/studentB_debate/results/c17_b0_k4_f3_s0_archdsmil.json >/dev/null && echo "exp_c17 ARCH OK"
ADV_ROOT="$T" BBOX=1 CHEAP=b0 KC=2 python src/exp_tcga.py 0 4 rcc >/dev/null && python -c "import json,sys; r=json.load(open(sys.argv[1]))['res']; assert {'single_bb_mean','x_red_bb_min','x_hyb6_bb_mean'} <= set(r), set(r); print('BBOX', {k: round(v['balacc'],2) for k,v in r.items() if 'bb_' in k or k in ('single','x_red')})" "$T"/studentB_debate/results/tcga_rcc_b0_k2_f4_s0_bbox.json
ADV_ROOT="$T" BBOX=1 CHEAP=b0 python src/exp_c17.py 0 4 >/dev/null && ls "$T"/studentB_debate/results/c17_b0_k4_f4_s0_bbox.json >/dev/null && echo "exp_c17 BBOX OK"
(cd analysis && python -c "import compute_table, stats" >/dev/null && echo "analysis imports OK")
echo "SMOKE TEST PASSED"
