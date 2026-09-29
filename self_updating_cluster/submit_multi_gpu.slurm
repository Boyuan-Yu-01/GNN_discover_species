#!/bin/bash
#SBATCH --job-name=self-updating-2gpu
#SBATCH --partition=ruixu
#SBATCH --nodelist=c01-08
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:l40s:2
#SBATCH --mem=350G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out

set -eo pipefail
cd "${SLURM_SUBMIT_DIR}"

# Load USC CARC's Conda module, then activate the environment created by
# setup_cluster_environment.sh.
module purge
module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate GNN

nvidia-smi
export SELF_UPDATING_GPU_IDS=0,1
export SELF_UPDATING_CPU_THREADS="${SLURM_CPUS_PER_TASK}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export SELF_UPDATING_SHOW_PROGRESS_BARS=0
/home1/boyuanyu/.conda/envs/GNN/bin/python run_cycle.py
