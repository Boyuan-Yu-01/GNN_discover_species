# Stage IV: fine-tune after exploration

## Run

- Run from `self_improving/stageIV/`:

```sh
conda activate GNN
python train_stage4.py
```

- Edit parameters below the classes in `train_stage4.py`.
- Set **`LAMBDA_NEGATIVE = 1.0`** to start. Use `0.0` for a positive-only comparison.
- Set **`NEGATIVE_SPECIES_PER_EPOCH = 96`** to limit negative training to a fresh
  subset each epoch. Lower it for less work; set it to at least 966 to use the
  entire current negative set. `NEGATIVE_TRAJECTORY_FRACTION` independently
  controls the routes sampled within each selected species.
- Set a different `OUTPUT_FOLDER` for each comparison you want to retain; rerunning replaces that folder's generated outputs.
- Training starts from `CHECKPOINT_PATH` each time. Automatic resume is not implemented.
- Console progress bars show completed epochs and positive/negative routes within each epoch, with speed and estimated time remaining. Saved logs retain plain messages without timestamp prefixes.
- `TRAINING_DEVICE = "mps"` selects the Apple GPU; use `"cpu"` explicitly for a CPU run. The startup log reports the selected device. No silent CPU fallback is used.
- Graph preparation and seeded sampling remain on CPU. The GNN forward/backward passes use MPS. Only the small selected-log-probability vectors move to CPU for stable float64 route losses; gradients still propagate to the GPU model.
- Saved model and optimizer tensors are on CPU for portability. MPS permits warnings for operations without deterministic kernels; results need not be identical across devices.
- Restart the script to change devices; an already-running CPU job cannot switch devices mid-run.
- Shared graph-convolution optimization: encode and convolve each partial state once, then score its representation under all positive composition contexts. This applies to negative training and free-growth evaluation. Features stay attached to autograd and are recomputed on every call.
- A warmed-up benchmark of eight negative routes across 40 contexts at forward batch size 256 measured CPU `0.259 → 0.032 s/batch` and MPS `0.767 → 0.023 s/batch` after this change. Each graph-convolution layer processed 72 graph inputs instead of 2,880. These are small-batch measurements; full-epoch speed depends on graph complexity, sampling and evaluation.


## Current status

- Stage II used the current checkpoint: 100 runs, 100,000 attempts, 1,013 unique graphs.
- PUCT rewards: given species `0.5`, unknown structures `1.0`, failed attempts `0.0`; stopping uses the attempt count only.
- Stage III compared those graphs with the 93-species CHO reference:
  - 30 given/recovered graphs.
  - 17 additional validated graphs.
  - 966 unmatched graphs.
- `update_species.py` has completed; the refreshed inputs are ready:
  - **48 known graphs = 31 existing + 17 additions**.
  - **966 unmatched graphs**, matching the latest Stage III unvalidated set.
  - C2H6 was not recovered in exploration; it remains among the 31 existing graphs.
- New graph keys: C2H3CHO, CH3OOH, C2H3OH, C2H2OH, C3H3, C3H5, C2H5O2, AC3H4, PC3H4, H2CC, C2H4OH, C2H5OH, CH3COCH3, C3H6, C3H5OH, C2H5CHO, H2C4O.

## Inputs

- Paths are relative to `stageIV/`:
  - Model: `../stageI/output_stage1/trained_growth_gnn.pt`.
  - Positive examples: `../stageIII/species_update/expanded_species_graphs.json`.
  - Negative examples: `../stageIII/species_update/unmatched_species.json`.
- The expanded file includes the original species; no separate original-species JSON is required.
- Treat unmatched graphs as **invalid for this training round**, as requested. This is a working label, not proof of chemical invalidity.

## Training approach

- Fine-tune the existing checkpoint; do not initialize a new network.
- Reuse Stage I's GNN, environment, action masks, and symmetry-equivalent targets.
- Read architecture and limits from the checkpoint; currently: `[32, 64, 32, 32, 16]`, 10 atoms, 24 actions; CO remains excluded.
- Validate graphs and remove duplicates before training. Positive labels take precedence if a graph appears in both datasets.
- Reject unsupported formal charges instead of silently ignoring them.
- Enumerate the same route family as Stage I: final bond orders are introduced directly, followed by STOP. Optional intermediate bond-order upgrades are not included.
- Store exhaustive routes in compressed `route_cache/` files; load sampled routes as needed. Symmetry pruning preserves Stage I's distinct action sequences.
- Positive examples:
  - Construct legal trajectories, including STOP.
  - Sample 20% of each species' available trajectories per epoch, rounded up, without replacement.
  - Average loss over steps, trajectories, then species so each species receives equal weight.
  - Stop preparation and report the graph if exhaustive enumeration exceeds its configured budget.
- Negative examples:
  - Uniformly sample up to `NEGATIVE_SPECIES_PER_EPOCH` graphs without replacement each epoch. The current setting is 96 of the 966; graphs can repeat across epochs.
  - Within each selected graph, sample `ceil(total routes × NEGATIVE_TRAJECTORY_FRACTION)` distinct routes without replacement.
  - The default negative fraction is 20%, independently editable from the positive fraction.
  - With `LAMBDA_NEGATIVE = 0`, skip negative route preparation and training; still measure negative recurrence.
  - Penalize the probability of each complete rejected route, including STOP.
  - Do not label every shared fragment or individual construction action invalid.
  - Normalize negative loss separately so the larger negative dataset does not dominate by count.
  - Average over the selected graphs: each route has weight `1 / (selected graphs × sampled routes for that graph)`. This estimates the full negative-set mean without reducing lambda's effective weight merely because fewer graphs are selected. Individual epochs have more sampling variation.
  - Species sampling uses a separate seeded random generator; positive sampling remains independent. Logs record the selected negative IDs each epoch.
  - All negative routes are still enumerated during initial preparation. The cap reduces per-epoch training and route-file reads, not that initial preparation cost. Evaluation still checks against the full negative set.

## Loss

- Combined objective: `L = L_positive + lambda_negative * L_negative`.
- Positive step loss: `-log(sum of probabilities of acceptable equivalent actions)`.
- Negative route loss: `-log(1 - P(route))`; minimizing it lowers the rejected route's probability.
- A negative route uses its exact action at each step, including STOP; positive steps retain symmetry-equivalent action targets.
- Average negative route losses within each graph, then over graphs. Average positive losses over steps, routes, then graphs.
- For negative routes, compute `P(route)` from the product of action probabilities, using Stage II's probability average over the expanded set's composition contexts. Shared graph features preserve all 40 contexts; averaging remains over probabilities, not logits.
- Compute in log space with numerical safeguards. Do not negate ordinary cross-entropy.
- Penalizing only final STOP is insufficient when it is the only legal action; complete-route loss can affect earlier choices.
- Long routes may have very small probabilities and weak gradients. Check actual recurrence of rejected graphs; low loss alone does not establish success.
- No teacher-consistency loss, new classifier, or PPO is proposed in this version.

## Initial parameters

- Learning rate: `0.0003`; fresh Adam optimizer initialized from the loaded model weights.
- Epochs: `100`; positive and negative trajectory fractions: `0.20` each.
- Negative loss weight: `1.0`; use `0` for a positive-only comparison run.
- Current route counts: **22,613 positive** and **4,725,390 negative**.
- At 20% with all graphs: **4,544 positive + 945,412 negative routes per epoch**.
  The current 96-of-966 negative subset reduces expected negative routes to
  approximately **94,000 per epoch**, with variation because graph route counts differ.
  Positive routes remain 4,544. This is about 9.9% of the previous negative route
  workload in expectation, not a measured full-training speedup.
- Route batch size: `8`; forward batch size: `256` states including formula-context copies.
- Gradients accumulate over batches; perform one Adam update per epoch.
- Enumeration limits: `1,000,000` routes and `10,000,000` visited enumeration states per graph; `20,000,000` routes overall. Exceeding a limit raises an error, never silently truncates the pool.
- Evaluate every `10` epochs; `100` samples per composition and `500` free-growth attempts; seed `12345`.
- Selection tolerances: reconstruction and completion may decrease by at most `0.02` (two percentage points); positive species coverage must not decrease.
- Keep parameters editable below the classes, relative paths, and direct calls without a main guard.

## Evaluation

- Evaluate the parent checkpoint on the corrected expanded set before updating weights.
- Track positive species coverage, reconstruction frequency, correct composition, STOP completion, and negative-graph recurrence.
- Watch for collapse to small molecules or nontermination as ways to avoid rejected outputs.
- Compare positive-only training against positive-plus-negative training using identical positive samples, budgets, and evaluation seeds.
- The script selects evaluated checkpoints only if they retain parent coverage and meet the reconstruction/completion tolerances.
- With a positive lambda, free-policy negative recurrence must decrease from the parent baseline; if the baseline is zero, it must remain zero.
- With lambda zero, rank eligible checkpoints by positive coverage and reconstruction.
- For positive lambda, rank eligible checkpoints by lowest negative recurrence, then positive coverage and reconstruction.
- Recheck the selected model and parent with a second fixed seed. If the selected model fails this check, export parent weights as the fallback (`selected_epoch = 0`).
- Evaluation samples the network directly; it does not run PUCT or establish experimentally valid chemistry. The positive-only comparison remains a separate run.
- These metrics measure the chosen labels and reconstruction, not experimentally established chemistry.

## Outputs and next round

- Save directly under configurable `stageIV/output/`, replacing that run's previous outputs:
  - `trained_growth_gnn.pt`: selected model.
  - `latest_growth_gnn.pt`: final trained weights, optimizer and random-generator states; automatic resume is not implemented.
  - `training.log` and `metrics.csv`: positive loss, negative loss, lambda-weighted loss, route counts, selected negative species count, gradient norms and epoch times. The log also lists the selected negative IDs.
  - `config.json`, `dataset_summary.json`, and `evaluation.json`: compact reports.
    Route counts in `dataset_summary.json` use `sampled_when_species_selected`, since unselected negative graphs contribute no routes in that epoch.
  - `route_cache/`: compressed JSON-lines route pools, regenerated for each run.
- Keep comparison experiments in separate output folders; preserve the parent checkpoint and input datasets.
- Stage II checkpoint compatibility was verified with a small temporary exploration.
- For the next exploration, set Stage II `CHECKPOINT_PATH = "../stageIV/output/trained_growth_gnn.pt"` and `REFERENCE_JSON_PATH = "../stageIII/species_update/expanded_species_graphs.json"`. Adjust the checkpoint path if using a custom output folder.
- Use the selected model and expanded reference together for the next round; compare models with the same PUCT settings and reference set.
- **PUCT limitation:** its current novelty reward can still favor rejected graphs. Stage IV changes model probabilities, not that reward; measure this conflict separately.
- After exploration, run Stage III comparison and `update_species.py` to prepare the next round's datasets.

## Verification performed

- Verified fresh, reproducible negative species subsets, no replacement, per-graph
  weights, exact expected full-set mean loss, small/empty sets, independent positive
  sampling and invalid cap rejection. Temporary CPU and MPS runs with a one-of-two
  negative cap passed for lambda 0 and 1, including saved species RNG states and
  Stage II checkpoint loading; existing production checkpoints were preserved.
- Exhaustive preparation completed for all 48 positive and 966 negative graphs with the default limits.
- Compared route enumeration with Stage I on small molecules, CH3CHO, C2H6, and a cyclic graph.
- Checked negative-loss values/gradients, probability averaging, percentage sampling, normalization, deduplication and overlap removal.
- Completed temporary two-epoch runs with lambda `0` and `1`, then loaded both selected checkpoints through Stage II.
- Verified Stage I and Stage IV training on the actual Apple GPU outside the sandbox, including float64 loss transfers and CPU-portable checkpoint loading.
- Compared the shared graph calculation against the previous implementation on CPU and MPS: action probabilities, negative route losses, parameter gradients, mask handling, head chunk boundaries, and unchanged ordinary Stage I/II forward behavior.
- Temporary two-epoch runs passed after the optimization for lambda `0` and `1`, including Stage II loading of their saved checkpoints.
- Full 100-epoch training has not been run; parent checkpoint and production outputs remain unchanged.
