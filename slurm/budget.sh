#!/bin/bash
# Budget-matched single selector (BONLY=1): ADMIL-style top-K at K, K+Kc, 2K tiles, clean and white-box attacked; skips the judges.
# Results: studentB_debate/results/*_budget.json. usage: sbatch --export=ALL,EXP=tcga,CO=rcc,SEED=0 slurm/budget.sh | EXP=c17
#SBATCH --account=def-xli135
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=200G
#SBATCH --time=4:00:00
#SBATCH --array=0-4
#SBATCH --job-name=budget
#SBATCH --output=/scratch/ciro/cvpr_lab/runs/%x_%A_%a.log
echo "start $(date +%T) $(hostname) EXP=$EXP CO=$CO SEED=$SEED"
module load StdEnv/2023 python/3.11
source /scratch/ciro/cvpr_lab/venv/bin/activate
cd /scratch/ciro/cvpr_lab/studentB_debate/v2
if [ "$EXP" = c17 ]; then BONLY=1 CHEAP=b0 python -u exp_c17.py ${SEED:-0} $SLURM_ARRAY_TASK_ID
else BONLY=1 CHEAP=b0 python -u exp_tcga.py ${SEED:-0} $SLURM_ARRAY_TASK_ID ${CO:-rcc}; fi
