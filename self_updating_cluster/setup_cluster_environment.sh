#!/bin/bash
# Create the CUDA-enabled GNN environment on USC Endeavour.
set -euo pipefail

module purge
module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx GNN; then
    conda env update --name GNN --file environment_cluster.yml --prune
else
    conda env create --file environment_cluster.yml
fi

conda activate GNN
python -c "import torch, rdkit, openpyxl, networkx, matplotlib, PIL, tqdm; print('CUDA available:', torch.cuda.is_available()); print('Visible GPUs:', torch.cuda.device_count())"
