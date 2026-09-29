# Stage V: generate from the Stage IV model

- Run from `self_improving/stageV/`: `python predict_stage5.py`.
- Uses `../stageIV/output/trained_growth_gnn.pt`, Stage IV's selected model.
  If Stage IV retained the parent checkpoint, this file contains the parent weights.
  Finish Stage IV before running; the model and reference must stay fixed.
- Uses `../stageIII/species_update/expanded_species_graphs.json` as the known set
  and the source of composition contexts (currently 48 graphs, 40 compositions).
- Runs two campaigns in order: **PUCT**, then **direct network sampling**.
- Edit parameters below the class. Default: 100 runs × 1,000 attempts **per method**,
  totaling 200,000 attempts. Both use the same model, reference and seed schedule.
- PUCT uses `PUCT_C = 2.0`, 25% uniform prior mixing, known reward 0.5 and
  candidate reward 1.0. Direct sampling uses temperature 1.0 and no uniform mixing.
- Free growth averages action probabilities across reference compositions; atom
  counts are not prescribed. Both methods reuse graph features across contexts.
- Weights remain frozen. Inference uses CPU, as in Stage II.
- Reuses Stage II classes without executing their bottom-level calls.
- Saves Stage II-style JSON inventories, route traces, logs, PNG and MP4 in:
  - `output/puct/`
  - `output/direct/`
- Each method has per-run details under `runs/`. Rerunning replaces generated
  files in these method folders. Earlier stages' outputs are untouched.
- Next: run `../stageVI/compare_stage6.py` from the Stage VI directory.
- Verification: two temporary runs of six attempts per method passed, including
  PNG/MP4 generation and Stage VI classification. Full production runs have not
  been started by this implementation task.

## Molecular drawing

- The shared Stage II renderer now uses RDKit 2D coordinates instead of spring
  coordinates. This prevents the NC4H5 chain bond from passing behind an unrelated
  carbon and falsely resembling a ring. RDKit must be installed for media output.
- Layout uses exactly the saved atoms and bond orders, without adding hydrogens
  or inferring electronic states. Existing graph matches and training are unaffected.
