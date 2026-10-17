#!/bin/bash
#SBATCH --account=def-xli135
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=1:30:00
#SBATCH --job-name=enctime
#SBATCH --output=/scratch/ciro/cvpr_lab/runs/%x_%j.log
module load StdEnv/2023 python/3.11
source /scratch/ciro/cvpr_lab/venv/bin/activate
cd /scratch/ciro/cvpr_lab/studentB_debate
python -u enc_timing.py
