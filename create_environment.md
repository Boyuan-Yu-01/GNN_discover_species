## Create a anaconda virtual environment named "GNN"

### Create Environment
```batch
conda create -n GNN python=3.11 -y
conda activate GNN
```

### Install Packages
```batch
python -m pip install --upgrade pip
python -m pip install torch torchvision
python -m pip install torch_geometric
python -m pip install networkx matplotlib numpy
```

### Varify Installation, shoudl print "GNN environment is ready"
```batch
python -c "import torch; import torch_geometric; import networkx; import matplotlib; import numpy; print('GNN environment is ready')"
```