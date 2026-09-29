# Self-updating molecular generation

Run from this directory:

```sh
conda activate GNN
cd /Users/boyuanyu/Documents/research/GNN/GNN_discover_species/self_updating
python run_cycle.py
```

- Edit experiment parameters in `run_cycle.py`. Library defaults and configuration
  checks live in `codebase/config.py`.
- The runner uses direct calls, without a main guard. Importing it starts a run.
- Defaults: initial training from scratch, then **3 update rounds**; **100,000
  attempts per generation method per round**; **100 fine-tuning epochs**.
- Training uses MPS. Use `TRAINING_DEVICE = 'cpu'` for CPU training.
  Generation uses CPU. There is no silent device fallback.
- `SAVE_PLOTS` controls PNG reports. No MP4 output.
- Paths are relative to your working directory.
- `LOG_LEVEL` in `run_cycle.py` controls console and text-log detail:
  `0` is silent, `1` shows stage banners and progress bars only, and `2` keeps
  detailed progress, metrics, and saved logs. Use the same value for a resumed
  run because it is part of the saved run configuration.

## Inputs and JSON conversion

- `input/given_species.xlsx`: the supplied training-reference workbook.
  It contains 32 entries; the default CO exclusion leaves 31 active species.
- `input/FFCM2_CHO_species_reference.xlsx`: 93 reference entries for validation.
- Both use the `Graph definitions` sheet. The reader uses the explicit node and
  bond columns, preserves supplied formal charges, and leaves absent charge
  information unspecified. The current training model rejects charged graphs.
- Before replacing any previous run, the pipeline checks input structures and
  settings. It then writes `output/inputs/given_species.json`,
  `negative_species.json`, and `reference_species.json`.
- These snapshots are hashed as a completed phase. Resume rejects modified or
  missing snapshots, even if the original workbook is unchanged.
- Optional negative input is a JSON **list** of records containing
  `structure_id` and `graph`; an empty list is valid.
- CSV and JSON graph dictionaries remain supported. CSV columns are
  `species_key`, `graph_json`, and optional `chemical_name`/`state_notes`.

## Make a workbook from SMILES or InChI

`codebase/species_exporter.py` contains `SpeciesExporter`. Use it from this
directory with the GNN environment (RDKit and openpyxl are already installed):

```python
from codebase.species_exporter import SpeciesExporter

SPECIES = [
    {"species_key": "CH4", "smiles": "C"},
    {"species_key": "OH", "smiles": "[OH]"},
    {"species_key": "H2O", "inchi": "InChI=1S/H2O/h1H2"},
    {"species_key": "CH3CHO", "smiles": "CC=O", "chemical_name": "Acetaldehyde"},
]
OUTPUT_PATH = "input/my_species.xlsx"

exporter = SpeciesExporter(SPECIES)
exporter.write_xlsx(OUTPUT_PATH)
```

- Supply exactly one `smiles` or `inchi` string per entry. Keys must be unique.
- The species key is a label; the supplied structure determines the atoms/bonds.
- `Species reference` contains atom-circle plots, base atom indices, bonds,
  attached-H counts, SMILES and the original input strings.
- `Graph definitions` contains compact arrays compatible with the existing
  workbook reader, including formal charges and RDKit radical counts.
- All hydrogens are explicit in graphs/plots. Aromatic inputs use a Kekulé
  representation with single/double bonds. H and H2 use empty attached-H arrays.
- Invalid strings and disconnected fragments raise an error before writing.
  The output path is your choice; writing to an existing file replaces it.
- Exporting charged species such as CO (`[C-]#[O+]`) is supported. The current
  network still trains neutral CHO graphs only; isotope/stereo labels and radical
  counts are retained for reference but are not used by the graph reader.
- Set `POSITIVE_SPECIES_PATH` or `REFERENCE_PATH` in `run_cycle.py` to use the new
  workbook as the corresponding input. Importing the exporter starts no training.

### FFCM2 example

Run the supplied example from `self_updating`:

```sh
conda activate GNN
python run_species_export.py
```

- `run_species_export.py` defines all 96 entries from the
  [Stanford FFCM2 species table](https://web.stanford.edu/group/haiwanglab/FFCM2/docs/TrialModel/Species/),
  checked on 2026-09-27. It runs offline, without a CSV or website download.
- Edit `SPECIES` and `OUTPUT_PATH` in that script. It creates a `SpeciesExporter`
  object, then writes the workbook to `OUTPUT_PATH` (currently `test_generator.xlsx`).
- The full table includes HE, AR and N2. This output is separate from the existing
  CHO-only input workbook and the given-species workbook.
- `EXCITED_` and `SINGLET_` are retained in `Source InChI` and recorded as source
  state annotations. Only these prefixes are removed before parsing. The graph
  sheet also saves `Normalized InChI`; connectivity does not resolve spin/state.
- Source keys are retained even when structures coincide, such as CH3CHO and
  C2H4O in the published table. This is an export of the source table, without
  correcting names or deduplicating chemical identities.

## One update round

### Consistent stage names

- **Stage I**: initial supervised imitation training. It runs once when the
  cycle starts without an existing checkpoint.
- **Stage II-A**: PUCT exploration with frozen model weights.
- **Stage II-B**: direct-sampling exploration with the same frozen weights.
- **Stage III**: validate and classify generated structures, then update the
  positive, negative, and unresolved datasets.
- **Stage IV**: fine-tune on the updated datasets and select the checkpoint
  used by the next round.

Stages II-A through IV repeat for every update round. Console banners include
the round number, for example `ROUND 2/3 | STAGE II-A | PUCT exploration`.

1. Freeze the parent model and current positive set.
2. Run PUCT and direct sampling with independent search trees and seeded runs.
   Direct sampling uses temperature 1.0 and no uniform prior mixture.
3. Merge duplicate structures, keeping method-specific appearance counts and
   representative construction traces.
4. Validate candidates. Retain existing positives and add accepted species.
5. Select the most frequent negative species for training.
6. Fine-tune with positive imitation and negative complete-route unlikelihood.
7. Compare the candidate with its parent and verify with a separate evaluation
   seed. Keep the parent weights when the candidate fails selection.
8. Save the selected checkpoint and carry the updated datasets into the next round.

All positive species remain in training. Initial training samples
`INITIAL_TRAJECTORY_FRACTION = 0.20` of each species' enumerated routes per epoch.
Fine-tuning uses `POSITIVE_TRAJECTORY_FRACTION = 1.00`. Negative routes use
`NEGATIVE_TRAJECTORY_FRACTION = 0.20`. Route counts are rounded up, and routes
are sampled without replacement within an epoch.

The positive and negative loss definitions are unchanged. There is one optimizer
update per epoch; route batches control memory rather than the update frequency.
`LAMBDA_NEGATIVE` controls the negative contribution.

## Negative selection

```text
hardness = direct appearances + PUCT_HARDNESS_WEIGHT × PUCT appearances
selected count = min(ceil(HARD_NEGATIVE_PERCENT × eligible negative species), HARD_NEGATIVE_MAX_COUNT)
```

- Defaults: **20%** of negative species, capped at **50** species;
  `PUCT_HARDNESS_WEIGHT = 0.25`.
- Counts accumulate across rounds. Imported counts without method provenance
  receive the same weight as PUCT counts.
- Every selected negative is used each epoch, with its configured route fraction.
- Unselected negatives remain available for evaluation and later rounds.
- `datasets/hard_negative_selection.json` records scores, ranking and selection.

## Checkpoints and resume

Set `RESUME = True` to continue the same run in the same output folder.

- Completed phases are verified and reused.
- Interrupted training resumes **after the last completely saved epoch**.
  Work within an interrupted epoch is repeated.
- Initial training and fine-tuning restore model weights, Adam state, route-sampling
  generators, Python/PyTorch random states, and MPS random state when applicable.
  Fine-tuning also restores evaluation history and checkpoint-selection state.
- `latest_growth_gnn.pt` is written atomically after every epoch.
- `CHECKPOINT_EVERY` and `INITIAL_CHECKPOINT_EVERY` control archived
  `checkpoints/epoch_XXXX.pt` copies with full continuation state.
  If the latest file is absent, resume uses the newest archived epoch.
- Metrics written after the saved epoch are discarded before continuing.
- Inputs, settings and code must match the saved run. `TOTAL_ROUNDS` can increase
  to extend a completed cycle.
- Pending validation jobs are collected without repeating completed generation.
  Interrupted generation currently restarts that method's campaign.
- CPU regression tests compare resumed and uninterrupted training exactly.
  MPS uses PyTorch's available deterministic operations with warnings for
  unsupported deterministic kernels; exact bitwise equality is not guaranteed.

To start a **new cycle from selected weights**, use a different `OUTPUT_FOLDER`,
set `CHECKPOINT_PATH` to a previous `checkpoints/round_001.pt`, and leave
`RESUME = False`. Set `POSITIVE_SPECIES_PATH = None` to inherit the positive
and negative sets embedded in that checkpoint. An explicit negative path can
override its bundled negatives. Old checkpoints without embedded datasets need
explicit input files.

Top-level round checkpoints are selected models with their datasets. Epoch
checkpoints additionally hold the optimizer and continuation state. A fresh cycle
starts a new optimizer; continuing an interrupted run restores the saved one.

Runs created before this resume implementation require a fresh run or a new
cycle from an existing model. Their previous phase signatures are incompatible.

## Validation and future DFT

- Validators return `status`: completed, pending or failed; and `decision`:
  accepted, rejected or unknown.
- Reference matches are accepted under element/bond matching rules. Missing
  references are unknown, rather than proof of chemical invalidity.
- `UNMATCHED_AS_NEGATIVE = True` is the current experiment's explicit policy.
  It labels completed reference misses as negative while retaining the evidence.
- Failed/pending jobs and other unknown outcomes stay outside negative training.
- Electronic metadata is preserved; unsupported charged/spin-specified graphs
  do not enter training.
- Comparison plots show green accepted additions, red unvalidated structures,
  and blue given species, in that order. Each category is sorted by cumulative
  appearances, highest first. Red does not by itself prove chemical invalidity.

Implement a future validator through `Validator.configuration()`,
`submit(record, workdir)` and, for asynchronous jobs,
`collect(record, previous, workdir)`. Stable job directories support idempotent
submission and result caching. Keep original candidates and optimized geometries
separate. `DFTValidator` remains an explicit placeholder; no DFT engine is implemented.

## Reused work

- Both training phases use the same symmetry-pruned route enumerator.
- Routes are cached by exact graph and action settings under `output/route_cache/`
  and reused across rounds/resume. Budgets raise errors rather than truncate routes.
- A bounded CPU cache stores route states, target actions and legal masks.
  `REPLAY_CACHE_SIZE` controls its route capacity. Learned features are recomputed
  after weight updates.
- Graph convolutions remain shared across compositions.
- Each generation campaign loads its model/reference once and merges returned
  attempts in memory. Per-run files remain available for inspection.
- Both PNG renderers share the same RDKit coordinate calculation.

## Output layout

```text
output/
├── inputs/                       # Hashed normalized input JSON snapshots
├── config.json
├── run_state.json
├── summary.json
├── pipeline.log
├── validation_cache.json
├── validation_jobs/
├── route_cache/                  # Reused exhaustive route files and hashes
├── checkpoints/
│   ├── initial_training.pt       # Or starting_checkpoint.pt
│   ├── round_001.pt
│   └── manifest.json
├── initial_training/
│   ├── trained_growth_gnn.pt
│   ├── latest_growth_gnn.pt
│   └── checkpoints/epoch_XXXX.pt
└── round_001/                    # Then round_002/, etc.
    ├── inputs/
    ├── generation/
    │   ├── puct/
    │   ├── direct/
    │   └── candidates.json
    ├── validation/              # Evidence, groups, summary, final_structures.png
    ├── datasets/                # Positive, negative, unresolved and selected sets
    ├── training/
    │   ├── trained_growth_gnn.pt # Selected weights for the next round
    │   ├── latest_growth_gnn.pt  # Latest epoch and continuation state
    │   └── checkpoints/epoch_XXXX.pt
    └── summary.json
```

`RESUME = False` replaces only pipeline-owned outputs in a recognized run folder.
It refuses to clear unrelated nonempty folders. JSON numeric vectors remain on
one line, and logs have no timestamp prefix.

## Modules and verification

| Module | Responsibility |
|---|---|
| `config.py` | Shared defaults and configuration checks |
| `checkpoint.py` | Atomic checkpoints, random states and epoch continuation |
| `molecule.py` | Input records, graph identity and construction environment |
| `model.py` | GNN and CPU/MPS handling |
| `trajectories.py` | Enumeration, persistent route cache and bounded replay cache |
| `trainer.py` | Initial training and positive/negative fine-tuning |
| `generator.py` | Frozen policy, PUCT/direct campaigns and result merging |
| `dataset.py` | Structure registry, labeling and negative selection |
| `evaluator.py` | Reconstruction, recurrence and model selection |
| `reporter.py` | Logs, progress output and PNG rendering |
| `storage.py` | Compact JSON, atomic state updates and hashes |
| `pipeline.py` | Round orchestration, input snapshots and resume |
| `validators/` | Validator contract, reference matching and DFT placeholder |

Dependencies: PyTorch, NetworkX, NumPy, Matplotlib, Pillow, tqdm and RDKit.
Workbook reading uses standard-library XML/ZIP support.

Run the small regression suite without launching the production cycle:

```sh
python -B -m unittest discover -s tests -v
```
