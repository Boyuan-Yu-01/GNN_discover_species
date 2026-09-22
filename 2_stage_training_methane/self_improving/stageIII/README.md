# Stage III: compare predictions with FFCMII

Stage III reads the saved Stage II structures and compares them with all 96 entries
in `../FFCMII_species_reference.xlsx`, using its `Graph definitions` sheet.
The output PNG uses three groups in this order:

1. **Green:** structures in FFCMII but outside the training set.
2. **Red:** structures outside FFCMII and the training set.
3. **Blue:** training-set structures, placed at the end.

Original structure IDs and the order within each group are preserved. Training
membership takes precedence if a training species is absent from FFCMII.

## Run

From `self_improving`:

```sh
conda activate GNN
cd stageIII
python compare_stage3.py
```

Run this again after a new Stage II exploration. It replaces the previous Stage III
reports in `output/`, including removing stale generated plot pages. The workbook
and Stage II outputs are read only. This step compares saved results; it does not
run exploration or update the neural network.

## How matching works

1. Read explicit atom symbols, bonds, bond orders, and reference formal charges
   from the workbook's graph arrays.
2. Read every completed structure in `../stageII/output/unique_structures.json`.
3. Find graph isomorphisms: atom elements and their bonds must correspond,
   regardless of atom numbering. The molecular formula alone is insufficient.
4. Read training membership from Stage II's saved `in_species_training_reference`
   field (or `category == "reference_match"` for older outputs). This preserves
   the membership used during exploration, including training exclusions.
5. Record every matching reference key, sort green/red/blue, and draw the plots.

Current Stage II predictions contain no formal charges or electronic states.
FFCMII membership therefore means a **connectivity match**, including bond orders. If a
prediction supplies formal charges, those charges must also match. Missing
charges are left unknown. States with identical connectivity, such as `CH2` and
`CH2(S)`, are listed together; Stage III cannot resolve their electronic state.

## Outputs

| File | Contents |
| --- | --- |
| `output/final_structures.png` | All predicted structures, green then red then blue |
| `output/pages/structures_001.png`, etc. | Larger panels, 50 structures per page by default |
| `output/compared_structures.json` | All original structure records plus FFCMII match information, in plot order |
| `output/ffcmii_matches.json` | Only matching structure records |
| `output/comparison.csv` | Structure IDs, formulas, matched keys, original categories, and occurrence counts |
| `output/summary.json` | Counts, matched/unmatched reference keys, and plot order |
| `output/config.json` | Settings and input file hashes |
| `output/comparison.log` | Saved progress log |

The current saved comparison contains **1,239 structures: 17 green, 1,192 red,
and 30 blue**. The green and blue groups together contain the 47 FFCMII matches.
These counts describe structures, not electronic
states or repeated construction attempts. The overview is tall; use the 25
paginated images for easier inspection.

## Settings and tests

Edit the parameters below the classes in `compare_stage3.py`:

- `PREDICTIONS_PATH`, `REFERENCE_WORKBOOK`, `REFERENCE_SHEET`: input locations.
- `COMPACT_JSON_SOURCE`: the shared Stage I compact JSON writer.
- `OUTPUT_FOLDER`: output location, default `output`.
- `OVERVIEW_COLUMNS`: columns in the complete overview, default 10.
- `STRUCTURES_PER_PAGE`, `PAGE_COLUMNS`: detailed page layout, defaults 50 and 5.

Paths are relative to the working directory. The script follows the existing
style: classes, editable parameters, then direct function calls, without a main
guard. It uses the existing GNN environment and Python's built-in XLSX reader
implementation; no new Excel library is required.

From `stageIII`, run:

```sh
python test_stage3.py
```

Seven tests cover workbook loading, graph relabeling, distinct isomers and bond
orders, formal charges, ambiguous electronic states, green/red/blue plotting,
invalid inputs, and end-to-end output generation without modifying the inputs.
