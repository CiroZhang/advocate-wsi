#!/bin/bash
# Attacks on the advocate presenter (vlm/vlm_debate3.py). One job per (model, dataset).
# usage: sbatch --export=ALL,MODEL=Qwen2.5-VL-7B-Instruct,DATA=vstar,SRC=/scratch/ciro/cvpr_lab/paperA_vstar/vstar_partial.jsonl,TAG=vstar_q25 slurm/deb3.sh
#SBATCH --account=def-xli135
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --job-name=deb3
#SBATCH --output=/scratch/ciro/cvpr_lab/runs/%x_%j.log
echo "start $(date +%T) $(hostname) MODEL=$MODEL DATA=$DATA TAG=$TAG"
module load StdEnv/2023 python/3.11
[ "$DATA" = hrbench4k ] && module load gcc arrow/23.0.1
source /scratch/ciro/cvpr_lab/venv/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd /scratch/ciro/cvpr_lab/vlm2
python -u vlm_debate3.py
