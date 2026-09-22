# Stage 1: learn the reference species

This implementation follows grow_train_animation_v4_1.py's class organization,
atom codes, node_types/current_bonds/bonds state, GrowthGNN naming, configurable
hidden layer widths, and model-saving approach.

It trains one shared composition-conditioned GNN on all 31 active reference
species. CO is excluded. Stage 2 novelty optimization is not implemented here.

## Run

Use the GNN environment, which contains PyTorch and NetworkX. From this folder:

    conda activate GNN
    python train_stage1.py

All parameters are below the classes in train_stage1.py. It executes directly,
without a main guard. Consequently, importing that script starts a training run.
It reads species_graphs.json directly and does not import species_graphs.py.

Paths are relative to your current working directory. OUTPUT_FOLDER is editable;
the default creates a timestamped run directory under output_stage1. An existing
run directory is rejected rather than overwritten.

Optional environment overrides, useful for short runs:

    STAGE1_EPOCHS=30 STAGE1_EVAL_EVERY=15 STAGE1_EVAL_SAMPLES=10 python train_stage1.py

STAGE1_OUTPUT_FOLDER may also specify a new relative output directory.

## What is learned

The policy receives:
- A partial molecular graph, with atom types, bond orders and available valence.
- Desired composition in C, H, O order.
- Current composition and its difference from the desired composition.

The species name, reference adjacency, and diagram coordinates are not inputs.
Node numbers only index the graph and actions.

Actions:
- START: choose an element for an empty graph.
- GROW: add H, O or C to a node, choosing a single/double/triple bond.
- CONNECT: add or increase a bond between existing nodes.
- STOP: finish the current molecule, including radicals.

Unlike version 4's single-bond GROW action, this GROW action can select the bond
order directly. Generic valence masks still apply. Each undirected CONNECT
action is included once. STOP is available even when more bonds are possible.

The desired composition does not force action masks. For example, a request for
CH4 still allows chemically legal C/O additions; the policy must learn to select H.
OH with composition O1H1 has a demonstrated STOP target; with O1H2, the demonstrated
continuation adds H. Isomers sharing a composition are multiple acceptable outputs.

## Demonstrations and loss

Before training, enumerate all distinct action sequences for each reference,
deduplicate interchangeable atom labels, and validate every final graph.
Set `TRAJECTORY_SAMPLE_FRACTION = 0.20` in the parameters section: each epoch
samples `ceil(total_sequences * fraction)` sequences per species **without
replacement**. Draw a fresh subset next epoch with a persistent seeded RNG.
CH3CHO has 100 unique sequences, so 20% selects 20 per epoch.

Average losses over steps, then sampled sequences within each species, then
species. Each step has weight `1 / (S * K_s * T_s,k)`, preserving equal species
weight even when their sequence counts differ. Valid fractions satisfy
`0 < fraction <= 1`. Exhaustive enumeration can become expensive for larger
future references; the current 31 reference species are small enough.

The supervised loss is the negative log probability of the demonstrated action.
Probabilities of symmetry-equivalent attachment choices are summed. There is no
mixture reward, discounted terminal reward, Gaussian-noise argmax, or novelty
bonus in this stage.

The GNN uses explicit single/double/triple message channels rather than version
4's repeated edges. Its grow_head, connect_head and termination_layer, plus a new
start_head, feed the same masked categorical distribution.

Evaluation samples or takes the maximum from that exact distribution.
Every generated action is checked by the environment before execution.

## Logs and output files

training.log contains the same messages printed to the terminal:
1. Configuration, source-file hash, reference counts and architecture.
2. A complete example action sequence for every reference species.
3. Every epoch's loss, teacher-action accuracy, correct-action probability,
   gradient norm and duration.
4. Periodic greedy generation results for each composition and each species.
5. Final action-by-action greedy traces for OH, H2O and CH4.
6. Final sampled generation results and output paths.
7. A traceback if training fails after logger initialization.

Other outputs:
- config.json: complete run settings.
- formation_sequences.json: all unique action sequences and per-species total/sample counts.
- training_examples.json: one readable validated trajectory per reference.
- metrics.csv: numerical per-epoch training metrics.
- baseline_greedy.json / .csv: untrained-model generation.
- greedy_epoch_NNNN.json / .csv: periodic evaluations; the final JSON includes
  every greedy rollout's full action/probability trace.
- trained_growth_gnn.pt: latest evaluated model, optimizer state, epoch,
  architecture/configuration, action list and reference JSON hash.
- generated_species_inventory.json / .csv: final sampled outputs, including
  base atom indices, base bonds, attached-H lists, reference membership,
  composition correctness, termination reason and full graphs (JSON).

This checkpoint is a saved model; automatic resume is not implemented. These
weights are not compatible with version 4's different feature/action dimensions.

## Reading the metrics

Teacher-action accuracy measures predictions at demonstrated partial graphs.
It is not the same as free-generation success. Multiple valid construction
orders can also limit this single-example action metric.

Reconstruction requires all three:
1. The model chose STOP before the step cap.
2. The generated composition matches the requested composition.
3. The output graph is isomorphic to a reference for that composition.

A chemically permitted graph or matching formula alone is not a success.
These checks implement the configured valence rules, not a complete
thermochemical/electronic validation.

The 31 species occupy 29 distinct compositions. CH3O/CH2OH and CH2CHO/CH3CO
share conditions. One greedy output per condition can therefore cover at most
29 of the 31 species. Sampled evaluation is required to measure coverage of
both isomers.

Coverage counts distinct correct reference outputs, not the fraction of all
samples that are correct. Evaluate both metrics together. These are
seen-reference reconstruction measurements, not held-out chemical generalization.

Default training: 300 epochs, 100 sampled evaluations per composition (2,900
total samples). Increase the sample budget for more precise frequency estimates.
Results are deterministic under the tested CPU environment and fixed seeds;
bitwise reproducibility across different PyTorch versions is not promised.

## Validation performed

Run the focused checks from this directory:

    python test_stage1.py

Checks cover all 31 reference reconstructions, STOP for radicals, competing
actions, valence masks, size bounds, unique undirected bonds, isomer distinction,
symmetric action probabilities, node-permutation equivariance, finite gradients,
checkpoint round-trip, and invalid reference rejection.

A 300-epoch validation run is saved in output_stage1/validation_300_epochs:
- Initial greedy reconstruction: 1/29 conditions (3.4%).
- Final greedy reconstruction: 24/29 conditions (82.8%).
- Sampled reconstruction: 52.5% over 2,900 outputs.
- Sampled coverage: 31/31 species.

This run verifies the full workflow and demonstrates learning. It does not meet
a near-perfect reconstruction target and should not yet be treated as a finished
discovery model.

## Walkthroughs and JSON formatting

- [Training walkthrough](README_train_stage1.md)
- [Testing walkthrough](README_test_stage1.md)

All JSON exporters use CompactJSON from compact_json.py. Small atom/bond records
and short lists stay on one line; nested species, steps and samples remain indented.
Formatting changes do not alter the JSON schema or stored values.

Existing validation_300_epochs results describe the earlier fixed-pool sampling method; they are historical, not measurements of percentage sampling.

## Validation of exhaustive percentage sampling

Run: `output_stage1/validation_exhaustive_20pct/`, 300 epochs, fraction 0.20,
seed 12345. All 11 tests pass. The 31 references contain 439 unique sequences;
rounding upward per species selects 102 sequences per epoch. CH3CHO contributes
20 of its 100 sequences.

Final sampled reconstruction: 56.8%; reference coverage: 31/31. These are
reconstruction results on the training reference set, not held-out validation
or evidence of new molecule quality. Logs, all formation sequences, metrics,
generated molecules and the trained checkpoint are saved in that run folder.
