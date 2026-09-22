# Self-improving molecule discovery

This workspace has its own Stage I training files, Stage II PUCT exploration,
and Stage III comparison with the FFCMII reference workbook.
The original folders outside this workspace remain unchanged.

## Layout

```text
self_improving/
├── README.md
├── FFCMII.yaml
├── FFCMII_species_reference.xlsx         # All 96 FFCMII species
├── stageI/
│   ├── train_stage1.py
│   ├── test_stage1.py
│   ├── species_graphs.py
│   ├── species_graphs.json
│   ├── species_training_reference.xlsx   # Original training reference
│   ├── compact_json.py
│   ├── README.md
│   ├── README_train_stage1.md
│   ├── README_test_stage1.md
│   └── output_stage1/                    # Copied training run and checkpoint
├── stageII/
│   ├── explore_stage2.py                 # PUCT and ordinary sampling
│   ├── test_stage2.py
│   ├── README.md
│   ├── output/                          # Latest exploration; overwritten each run
│   └── output_stage2/                   # Earlier saved comparison
└── stageIII/
    ├── compare_stage3.py                # FFCMII matching and green/red/blue plots
    ├── test_stage3.py
    ├── README.md
    └── output/                          # Latest comparison; overwritten each run
```

Stage I is a copy of the existing training code, reference data, documentation,
and saved training run. Stage II reads its checkpoint and reference JSON from
this local `stageI` folder. The PUCT comparison outputs were moved into
`stageII/output_stage2`. Historical run configurations retain their original
recorded output paths.

The root Excel workbook lists all 96 FFCMII species. The copied Stage I still
uses the original active training set; organizing the folders does not retrain
the model on FFCMII or add new atom/charge encodings.

## Run

From this `self_improving` directory:

```sh
conda activate GNN
cd stageI
python test_stage1.py
python train_stage1.py
```

Training writes a new timestamped directory under `stageI/output_stage1`.
To explore newly trained weights, set `CHECKPOINT_PATH` in
`stageII/explore_stage2.py` to that new checkpoint. Its current default uses the
local 300-epoch run `20260921_221023_510656`. Stage II tests read the same path
setting; no separate test checkpoint path needs to be updated.

From this `self_improving` directory, run exploration with:

```sh
conda activate GNN
cd stageII
python test_stage2.py
python explore_stage2.py
```

Stage II defaults to `GENERATION_MODE = "free"`: each attempt starts empty,
chooses its atom counts, and selects STOP without an exact target formula.
The existing 10-atom and 24-action caps remain. The frozen formula-conditioned
network supplies an averaged policy across reference formulas; no retraining
is performed.

Edit `EXPLORATION_RUNS` and `ATTEMPTS_PER_RUN` in `stageII/explore_stage2.py`.
The defaults produce 10 × 1,000 = 10,000 attempts.
Set `GENERATION_MODE = "formula"` to restore `COMPOSITIONS` and
`SAMPLES_PER_COMPOSITION`. Full-reference recovery stopping remains optional
through `STOP_WHEN_ALL_SPECIES_RECOVERED`.

Stage II saves new results directly to `stageII/output/` and replaces its previous
results on every run, including stale media and per-run folders.

After exploration, run Stage III from this `self_improving` directory:

```sh
cd stageIII
python compare_stage3.py
```

Stage III saves `stageIII/output/final_structures.png`: FFCMII matches outside
training appear first in green, unmatched structures follow in red, and training
species appear last in blue. Larger paginated plots and compact JSON
reports are saved alongside it. Matching compares atom elements and bond orders,
ignoring atom numbering; electronic states remain unresolved.

Paths remain editable and relative to the working directory. The scripts retain
classes first, parameters below, and direct execution without a main guard.

See [Stage I](stageI/README.md) for training,
[Stage II](stageII/README.md) for PUCT settings, rewards, logs, and saved results,
and [Stage III](stageIII/README.md) for matching rules and comparison outputs.
