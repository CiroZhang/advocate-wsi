#!/bin/bash
#SBATCH --account=def-xli135
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=5:00:00
#SBATCH --array=0-29%6
#SBATCH --job-name=milbase
#SBATCH --output=/scratch/ciro/cvpr_lab/runs/%x_%A_%a.log
M=(abmil clam_sb transmil dsmil dtfdmil); E=(uni r50bt)
i=$SLURM_ARRAY_TASK_ID; meth=${M[$((i % 5))]}; enc=${E[$((i / 5 % 2))]}; seed=$((i / 10))
echo "start $(date +%T) $meth $enc seed $seed"
module load StdEnv/2023 python/3.11
source /scratch/ciro/cvpr_lab/venv/bin/activate
cd /scratch/ciro/cvpr_lab/studentB_debate
python -u mil_baselines.py --manifest /scratch/ciro/shared_baselines/manifests/c16_$enc.csv --out /scratch/ciro/shared_baselines/c16_$enc --method $meth --seed $seed
