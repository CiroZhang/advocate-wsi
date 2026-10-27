#!/bin/bash
# Killarney: generality over MIL judge architectures (ARCH mode, src/archs.py). One array task per fold.
# usage: sbatch --export=ALL,EXP=tcga,CO=rcc,SEED=0,ARCH=mean slurm/arch_k.sh   |   sbatch --export=ALL,EXP=c17,SEED=0,ARCH=trans slurm/arch_k.sh
#SBATCH --account=aip-xli135
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=120G
#SBATCH --time=6:00:00
#SBATCH --array=0-4
#SBATCH --job-name=arch
#SBATCH --output=/scratch/ciro/cvpr_lab_k/runs/%x_%A_%a.log
source /cvmfs/soft.computecanada.ca/config/profile/bash.sh; module load StdEnv/2023 python/3.11
source /scratch/ciro/cvpr_lab_k/venv/bin/activate
python -c "import torch; assert torch.cuda.is_available()" || exit 1
echo "start $(date +%T) $(hostname) EXP=$EXP CO=$CO SEED=$SEED ARCH=$ARCH"
cd /scratch/ciro/cvpr_lab_k/studentB_debate/v2
if [ "$EXP" = c17 ]; then CHEAP=b0 python -u exp_c17.py ${SEED:-0} $SLURM_ARRAY_TASK_ID
else CHEAP=b0 python -u exp_tcga.py ${SEED:-0} $SLURM_ARRAY_TASK_ID ${CO:-rcc}; fi
