# Self-updating molecular generation: codebase plan

## Objective

Replace the numbered-stage scripts with reusable classes and one running script.
Keep molecule generation, validation, dataset updates and training separate so
FFCM2 matching can later be replaced by DFT or another validation method.

This document proposes the interface and implementation plan. The example runner
below is not yet implemented.

## 1. Proposed Python files

```text
self_updating/
├── run_cycle.py
└── codebase/
    ├── molecule.py
    ├── model.py
    ├── trajectories.py
    ├── trainer.py
    ├── generator.py
    ├── dataset.py
    ├── evaluator.py
    ├── reporter.py
    ├── pipeline.py
    └── validators/
        ├── base.py
        ├── reference.py
        └── dft.py
```

| File                      | Responsibility                                                                         | Existing code to reuse                    |
| ------------------------- | -------------------------------------------------------------------------------------- | ----------------------------------------- |
| `molecule.py`             | Molecular graph representation, legal actions, graph matching and duplicate removal.   | Stage I environment and reference classes |
| `model.py`                | GNN architecture, shared graph features, action probabilities and checkpoint loading.  | `GrowthGNN`                               |
| `trajectories.py`         | Enumerate, cache, sample and replay construction routes.                               | Stages I and IV                           |
| `trainer.py`              | Initial positive training and subsequent positive/negative fine-tuning.                | Stages I and IV                           |
| `generator.py`            | PUCT and direct sampling through one interface; retain complete action traces.         | Stages II and V                           |
| `dataset.py`              | Maintain positive, negative and unresolved datasets across rounds.                     | Stage III updater                         |
| `evaluator.py`            | Measure reconstruction, coverage and negative recurrence; select the next checkpoint.  | Stage IV evaluation                       |
| `reporter.py`             | Compact JSON, logs, progress bars and PNG plots.                                       | Existing reporting classes                |
| `pipeline.py`             | Coordinate generation → validation → dataset update → training → checkpoint selection. | New                                       |
| `validators/base.py`      | Define the common validation interface and result format.                              | New                                       |
| `validators/reference.py` | Compare structures against a supplied reference workbook or graph file.                | Stages III and VI                         |
| `validators/dft.py`       | Future adapter for preparing, submitting and interpreting DFT calculations.            | Implement after requirements are defined  |
| `run_cycle.py`            | Editable parameters, object creation and execution.                                    | Replaces separate stage runners           |

- Library files contain definitions only and can be imported normally.
- The runner defines editable parameters, creates objects and calls their methods.
- Use direct execution without `if __name__ == "__main__":`.
- Preserve existing checkpoint compatibility and numerical behavior during extraction.

## 2. Replaceable validation

The pipeline asks what the validator concluded about a structure. It does not
need to know whether the evidence came from FFCM2, DFT or another method.

### Common result format

```python
{
    "structure_id": "G00123",
    "status": "completed",
    "decision": "accepted",
    "reason": "Matched a reference structure",
    "validator": "reference",
    "details": {"matched_species": ["CH3OOH"]},
}
```

| Field | Allowed values | Purpose |
|---|---|---|
| `status` | `completed`, `pending`, `failed` | Did validation finish? |
| `decision` | `accepted`, `rejected`, `unknown` | What conclusion does the evidence support? |

- Pending and failed validations use `decision="unknown"`.
- A failed calculation must not automatically become a negative training example.
- “Given species” describes dataset membership, not a validation decision. Store it separately.
- Support submitting jobs and collecting results later, because future calculations may not finish immediately.

### Separate validation from training-label policy

For reference matching:

- Match → accepted.
- No match → unknown.
- An explicit labeling setting may treat unmatched structures as negatives,
  reproducing the current experiment.

For DFT:

- Completed calculation → interpret the results using defined acceptance criteria.
- Inconclusive result → unknown.
- Calculation failure → failed, with an unknown decision.

This separation prevents the current assumption that unmatched structures are
negative examples from silently carrying into every future validator.

## 3. One complete round

1. **Load** the current selected model and datasets.
2. **Generate** molecules using PUCT and direct sampling, with the same frozen
   model and known-species set.
3. **Deduplicate** structures across both methods. Validate each unique structure
   once while retaining which methods and routes generated it.
4. **Validate** new structures. Reuse compatible cached validation results.
5. **Update datasets:**
   - Retain previous positives and add accepted structures.
   - Add rejected structures to negatives.
   - Apply the explicit unmatched-as-negative policy when enabled.
   - Keep remaining unknown, pending and failed cases separate.
   - Resolve label conflicts before training.
6. **Fine-tune** from the current model using configured positive and negative sampling.
7. **Evaluate** the candidate checkpoint against its parent on the same evaluation setup.
8. **Select** the next model, retaining the parent if the candidate fails the selection criteria.
9. Repeat until the configured number of rounds is complete.

Validation caches must distinguish structures and validator settings. A changed
reference, calculation method or acceptance rule can require revalidation.

## 4. Molecular representation for future DFT
<span style="color:red">DFT will not be implemented at this moment.</span>

The molecule record should support:

- Atom types and bond orders.
- Atom-level formal charges, when known.
- Total charge and spin multiplicity, when known.
- Optional 3D coordinates.
- Provenance linking a calculation to its original generated graph.

Unknown charge or spin remains explicitly unknown. If a calculation changes
geometry or connectivity, retain the original candidate and resulting structure
separately.

Adding these fields prepares the data format. Teaching the current GNN to generate
charged species would require a separate model/action-space change.

## 5. Example running script

The following is a proposed interface, not an executable implementation yet.
Paths remain relative to the runner's working directory and can be edited.

```python
from codebase.pipeline import SelfUpdatingPipeline
from codebase.validators.reference import ReferenceValidator


# PARAMETERS
POSITIVE_SPECIES_PATH = "../stageIII/species_update/expanded_species_graphs.json"
NEGATIVE_SPECIES_PATH = "../stageIII/species_update/unmatched_species.json"
CHECKPOINT_PATH = "../stageIV/output/trained_growth_gnn.pt"

REFERENCE_PATH = "../stageIII/FFCM2_CHO_reference.xlsx"
OUTPUT_FOLDER = "output"

TOTAL_ROUNDS = 3
TRAINING_DEVICE = "mps"

EXPLORATION_RUNS = 100
ATTEMPTS_PER_RUN = 1000

NEGATIVE_SPECIES_PER_EPOCH = 96
LAMBDA_NEGATIVE = 1.0


# CREATE OBJECTS
validator = ReferenceValidator(
    reference_path=REFERENCE_PATH,
    sheet_name="Graph definitions",
)

pipeline = SelfUpdatingPipeline(
    positive_path=POSITIVE_SPECIES_PATH,
    negative_path=NEGATIVE_SPECIES_PATH,
    checkpoint_path=CHECKPOINT_PATH,
    validator=validator,
    output_folder=OUTPUT_FOLDER,
    generation_config={
        "methods": ["puct", "direct"],
        "runs": EXPLORATION_RUNS,
        "attempts_per_run": ATTEMPTS_PER_RUN,
        "puct_c": 2.0,
        "puct_uniform_fraction": 0.25,
    },
    training_config={
        "device": TRAINING_DEVICE,
        "epochs": 100,
        "positive_trajectory_fraction": 0.20,
        "negative_trajectory_fraction": 0.20,
        "negative_species_per_epoch": NEGATIVE_SPECIES_PER_EPOCH,
        "lambda_negative": LAMBDA_NEGATIVE,
    },
    labeling_config={
        # Explicitly reproduce the current training assumption.
        "unmatched_as_negative": True,
    },
)


# RUN
pipeline.run(rounds=TOTAL_ROUNDS)
```

- The generation budget applies per method per round.
- Direct sampling uses temperature 1.0 and no PUCT uniform mixture.
- The device setting above applies to training; generation device support should
  be configured separately rather than inferred from this setting.
- Later, replace `ReferenceValidator` with a configured `DFTValidator` and change
  the labeling policy. Generation and training continue through the same interface.
- DFT engine configuration and acceptance criteria must be defined before implementing that adapter.

## 6. Output organization

```text
output/
├── round_001/
│   ├── generation/
│   │   ├── puct/
│   │   └── direct/
│   ├── validation/
│   ├── datasets/
│   └── training/
├── round_002/
└── summary.json
```

- Round folders represent successive learning rounds, not timestamped runs.
- Keep compact numeric arrays, logs without timestamp prefixes and configurable relative paths.
- A fresh run replaces generated output in the configured destination.
- An explicit resume option preserves completed work and pending validation jobs.
- Record the model, datasets and validation settings used by each round so results remain traceable.

## 7. Recommended implementation order

1. **Extract existing classes**, preserving numerical behavior and checkpoint compatibility.
2. **Introduce the common validator interface**, initially using FFCM2.
3. **Implement one complete round**, including merging both generation methods.
4. **Add repeated rounds, validation caching and resume support.**
5. **Implement the DFT adapter** after choosing the calculation engine and acceptance criteria.

This delivers a reusable working cycle first, with a clear interface for future
validation methods.
