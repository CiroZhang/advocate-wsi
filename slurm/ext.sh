#!/bin/bash
# Extended robustness experiments (src/ext.py): adaptive attack, hybrid presenter, independent scorers, weak scorer.
# Scripts live in studentB_debate/v2/ so the originals are untouched; results: studentB_debate/results/*_ext.{json,npz}.
# usage: sbatch --export=ALL,EXP=tcga,CO=rcc,SEED=0 slurm/ext.sh   |   sbatch --export=ALL,EXP=c17,SEED=0 slurm/ext.sh
#SBATCH --account=def-xli135
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=200G
#SBATCH --time=16:00:00
#SBATCH --array=0-4
#SBATCH --job-name=ext
#SBATCH --output=/scratch/ciro/cvpr_lab/runs/%x_%A_%a.log
echo "start $(date +%T) $(hostname) EXP=$EXP CO=$CO SEED=$SEED"
module load StdEnv/2023 python/3.11
source /scratch/ciro/cvpr_lab/venv/bin/activate
cd /scratch/ciro/cvpr_lab/studentB_debate/v2
if [ "$EXP" = c17 ]; then EXT=1 CHEAP=b0 python -u exp_c17.py ${SEED:-0} $SLURM_ARRAY_TASK_ID
else EXT=1 CHEAP=b0 python -u exp_tcga.py ${SEED:-0} $SLURM_ARRAY_TASK_ID ${CO:-rcc}; fi
