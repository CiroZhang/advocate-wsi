#!/bin/bash
# Killarney: isolated redundancy with mean / clipped / vote aggregation (RED=iso). One array task per fold.
# usage: sbatch --export=ALL,EXP=tcga,CO=rcc,SEED=0 slurm/iso_k.sh   |   sbatch --export=ALL,EXP=c17,SEED=0 slurm/iso_k.sh
#SBATCH --account=aip-xli135
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=120G
#SBATCH --time=6:00:00
#SBATCH --array=0-4
#SBATCH --job-name=iso
#SBATCH --output=/scratch/ciro/cvpr_lab_k/runs/%x_%A_%a.log
source /cvmfs/soft.computecanada.ca/config/profile/bash.sh; module load StdEnv/2023 python/3.11
source /scratch/ciro/cvpr_lab_k/venv/bin/activate
python -c "import torch; assert torch.cuda.is_available()" || exit 1
echo "start $(date +%T) $(hostname) EXP=$EXP CO=$CO SEED=$SEED "
cd /scratch/ciro/cvpr_lab_k/studentB_debate/v2
if [ "$EXP" = c17 ]; then EXT=1 RED=iso SUFX=_iso CHEAP=b0 python -u exp_c17.py ${SEED:-0} $SLURM_ARRAY_TASK_ID
else EXT=1 RED=iso SUFX=_iso CHEAP=b0 python -u exp_tcga.py ${SEED:-0} $SLURM_ARRAY_TASK_ID ${CO:-rcc}; fi
