#!/bin/bash
# Crop-substitution audit of a thinking-with-images agent (vlm/agent_subst.py), sharded over V* questions.
# usage: sbatch --export=ALL,AGENT=minio3,TAG=minio3_full,NSHARD=4 --array=0-3 slurm/agent.sh   (INIT_PIX=589824 for the 768^2 overview)
#SBATCH --account=def-xli135
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --job-name=agent
#SBATCH --output=/scratch/ciro/cvpr_lab/runs/%x_%A_%a.log
echo "start $(date +%T) $(hostname) AGENT=$AGENT TAG=$TAG shard $SLURM_ARRAY_TASK_ID/$NSHARD"
module load StdEnv/2023 python/3.11
source /scratch/ciro/cvpr_lab/venv/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd /scratch/ciro/cvpr_lab/vlm2
SHARD=${SHARD:-$SLURM_ARRAY_TASK_ID} python -u agent_subst.py
