#!/bin/bash
#SBATCH --account=def-xli135
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --output=/scratch/ciro/cvpr_lab/runs/io_%j.log
module load StdEnv/2023 python/3.11 openslide/4.0.0
source /scratch/ciro/cvpr_lab/venv/bin/activate
cd /scratch/ciro/cvpr_lab/studentB_debate
python -u io_bench.py
