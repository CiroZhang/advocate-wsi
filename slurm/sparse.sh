#!/bin/bash
#SBATCH --account=def-xli135
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=110G
#SBATCH --time=8:00:00
#SBATCH --array=0-2
#SBATCH --job-name=sparse
#SBATCH --output=/scratch/ciro/cvpr_lab/runs/%x_%A_%a.log
echo "start $(date +%T) $(hostname)"
module load StdEnv/2023 python/3.11
source /scratch/ciro/cvpr_lab/venv/bin/activate
cd /scratch/ciro/cvpr_lab/studentB_debate
python -u exp_sparse.py $SLURM_ARRAY_TASK_ID
