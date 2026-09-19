# Two-stage molecule training

- **stageI/**: reference species, training and test scripts, walkthroughs,
  Excel reference, and saved training runs/checkpoints.
- **stageII/**: fixed-network exploration, tests, and generated candidate inventories.

## Run stage I

From this directory:

```sh
cd stageI
conda activate GNN
python train_stage1.py
python test_stage1.py
```

The scripts retain configurable relative paths. Run them from stageI.
Existing training outputs are under stageI/output_stage1/.
See [the stage I README](stageI/README.md) for parameters and results.

## Run stage II

From this directory:

```sh
cd stageII
conda activate GNN
python test_stage2.py
python explore_stage2.py
```

See [the stage II README](stageII/README.md) for sampling parameters, output
classification, and saved candidate records.
