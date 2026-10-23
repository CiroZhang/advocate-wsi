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
(cd analysis && python -c "import compute_table, stats" >/dev/null && echo "analysis imports OK")
echo "SMOKE TEST PASSED"
