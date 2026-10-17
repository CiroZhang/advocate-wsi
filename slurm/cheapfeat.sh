#!/bin/bash
#SBATCH --account=def-xli135
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --array=0-9
#SBATCH --output=/scratch/ciro/cvpr_lab/runs/cheapfeat_%A_%a.log
module load StdEnv/2023 python/3.11 openslide/4.0.0
source /scratch/ciro/cvpr_lab/venv/bin/activate
export HF_HOME=/scratch/ciro/cvpr_lab/hf_cache HF_HUB_OFFLINE=1
cd /scratch/ciro/cvpr_lab/studentB_debate
echo start $(date +%T)
python cheap_feats.py --part $SLURM_ARRAY_TASK_ID --nparts 10 --workers 8
