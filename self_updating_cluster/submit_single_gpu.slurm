#!/bin/bash
#SBATCH --job-name=self-updating-1gpu
#SBATCH --partition=ruixu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=350G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

# Load USC CARC's Conda module, then activate the environment created by
# setup_cluster_environment.sh.
module purge
module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate GNN

nvidia-smi
export SELF_UPDATING_GPU_IDS=0
export SELF_UPDATING_CPU_THREADS="${SLURM_CPUS_PER_TASK}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
python run_cycle.py
