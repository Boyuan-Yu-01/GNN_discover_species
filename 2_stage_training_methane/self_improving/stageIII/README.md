# Stage III: compare predictions with FFCMII

Stage III reads the saved Stage II structures and compares them with the 93 CHO entries
in `FFCM2_CHO_reference.xlsx`, using its `Graph definitions` sheet.
`FFCM2_species_reference.xlsx` retains all 96 entries; the CHO subset excludes HE, AR, and N2.
The output PNG uses three groups in this order:

1. **Green — validated:** structures matching the reference but outside the training set.
2. **Red — unvalidated:** structures without a reference match and outside the training set.
3. **Blue — given / recovered:** training-set structures, placed at the end.

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
reports in `output/`. Only one PNG is generated: `final_structures.png`.
Generated pages left by earlier versions are removed. The workbook
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
| `output/ffcmii_matches.json` | All generated species in three sets: `validated`, `unvalidated`, and `given` |
| `output/comparison.csv` | Structure IDs, formulas, matched keys, original categories, and occurrence counts |
| `output/summary.json` | Counts, matched/unmatched reference keys, and plot order |
| `output/config.json` | Settings and input file hashes |
| `output/comparison.log` | Saved progress log |

`ffcmii_matches.json` contains all generated species in `validated` (green),
`unvalidated` (red), and `given` (blue). Each entry records its
structure ID, species keys (empty when unknown), formula, occurrence count,
base atom indices, base bonds, attached H, full graph, and reference state notes.
All matching reference aliases are retained. Every generated structure appears
in exactly one group. The given/recovered group includes only training species
actually generated, not the entire training set.

Counts describe distinct structures, not electronic states or repeated attempts;
see `summary.json` for the current run. Validation means a reference graph
match with the matching rules above.

Only three JSON reports are written: `ffcmii_matches.json`, `summary.json`, and
`config.json`. Rerunning removes the obsolete `species_classification.json` and
`compared_structures.json` reports. Full construction and search details remain
available in Stage II's `unique_structures.json`.

JSON numeric arrays stay on one line, including long run/index lists. Short bond
tables stay together, and long lists of species names/IDs use several entries
per line. Objects remain indented for readability, with each species field and
each atom/bond record on its own line. Stage III explicitly uses `pack=False`
to retain this layout; Stage I and Stage II use the denser default layout.

## Expand known species after comparison

Run `python update_species.py` from `stageIII` after running the comparison.
The script reads `../stageI/species_graphs.json` and `output/ffcmii_matches.json`.
Its class is followed by editable path parameters and a direct function call.

It writes to the configurable `species_update/` folder, replacing prior exports:

| File | Contents |
| --- | --- |
| `expanded_species_graphs.json` | Every original known graph plus distinct graphs from the validated (green) group; uses Stage I's species-key -> nodes/edges format |
| `unmatched_species.json` | Complete Stage III records for the unvalidated (red) generated candidates |
| `update_species.log` | Input paths, additions, duplicate skips, name conflicts, and final counts; also printed to the console |

Recovered (blue) species normally belong to the known set used for exploration;
original species not recovered during exploration are also retained. If a later
reference correction changes a blue graph, the updater logs it as historical
and does not reintroduce it into the corrected expanded set or automatically
add it to the unmatched file. Unmatched
here means generated unvalidated candidates, not missing reference species.
No unvalidated candidate is added to the known set.

Graph comparison ignores atom numbering and preserves elements, bond orders,
and supplied charges. An already-known graph is skipped. If a reference name
already exists with a different graph, both are kept: the new key receives its
structure ID, for example `HCCO__M00080`. Such entries can represent alternative
bond/resonance descriptions of the same species, not new chemical identities.
Reference aliases and source structure IDs are stored on new graph entries.
Electronic-state aliases do not create multiple copies of the same graph.

Latest run reviewed on 2026-09-24:

- Stage II: 100 independent PUCT runs, 100,000 attempts; reference reward 0.5, candidate reward 1.0.
- Stage III: 1,013 distinct graphs = 17 validated (green) + 966 unvalidated (red) + 30 given/recovered (blue).
- The 47 reference-matched graphs match 51 reference keys because some keys share graph connectivity; electronic states remain unresolved.
- Expanded set: **48 graphs = 31 original + 17 additions**; unmatched set: **966 graphs**.
- C2H6 was not recovered, but remains in the expanded set as prior knowledge.
- HCCO and CH2CHO match the corrected reference; no suffixed alternatives or duplicate expanded graphs are present.
- Input hashes, graph classifications, all CSV rows and PNG panels, and both updater exports agree with the current inputs.

These counts describe the current saved run and change with future exploration.

The original known file is preserved. To use the expanded set for future
training, set Stage I's `REFERENCE_JSON_PATH` to
`../stageIII/species_update/expanded_species_graphs.json` when running from
`stageI`. Training settings still control excluded species and size/valence
limits. This script does not change those settings or run training.

## Comparison settings

Edit the parameters below the classes in `compare_stage3.py`:

- `PREDICTIONS_PATH`, `REFERENCE_WORKBOOK`, `REFERENCE_SHEET`: input locations.
- `COMPACT_JSON_SOURCE`: the shared Stage I compact JSON writer.
- `OUTPUT_FOLDER`: output location, default `output`.
- `OVERVIEW_COLUMNS`: columns in the complete overview, default 10.

Paths are relative to the working directory. The script follows the existing
style: classes, editable parameters, then direct function calls, without a main
guard. It uses the existing GNN environment and Python's built-in XLSX reader
implementation; no new Excel library is required.
