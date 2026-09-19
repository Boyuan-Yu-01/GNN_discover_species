# How train_stage1.py works

`train_stage1.py` follows this sequence:

```text
Read reference JSON
        ↓
Build and verify construction examples
        ↓
Create one shared GNN
        ↓
Train it to predict demonstrated actions
        ↓
Let it generate molecules independently
        ↓
Save logs, results, and model weights
```

Here is how the pieces connect.

## 1. Execution starts at the bottom

The classes above define the operations. The parameters below them configure the run:

```python
GNN_HIDDEN_DIMS = [32, 64, 32, 32, 16]
TOTAL_EPOCHS = 300
TRAJECTORY_SAMPLE_FRACTION = 0.20
```

The final calls start training:

```python
trainer = Stage1Trainer(training_config)
trainer.run()
```

`Stage1Trainer` creates the output folder and logger. `run()` calls `train()`, records a traceback if anything fails, and closes the log afterward.

## 2. Load the 31 reference species

Inside `train()`, `ReferenceSpecies` reads:

```python
REFERENCE_JSON_PATH = "species_graphs.json"
```

It reconstructs each molecule as a NetworkX graph:

- Nodes contain atom types.
- Edges contain bond orders.

It checks for invalid node IDs, duplicate bonds, disconnected graphs, and excessive valence. CO is excluded.

It also groups species by composition:

```text
(C=1, H=3, O=1) → CH3O, CH2OH
```

That grouping matters because the model receives **composition**, not the species name. Both isomers are acceptable outputs for that condition.

## 3. Build a fixed action list

`ActionSpace` gives every possible action a fixed position in the network's output vector.

The actions are:

| Action | Meaning |
|---|---|
| `START(atom)` | Place the first atom |
| `GROW(node, atom, order)` | Attach a new atom to an existing node |
| `CONNECT(u, v, order)` | Create or upgrade a bond to the specified final order |
| `STOP` | Finish the molecule |

For example:

```python
("GROW", 0, 0, 1)
```

means:

> Add an H atom—element code `0`—to node `0` with a single bond.

**Node ID and element code are different things**, even though both are integers.

## 4. The environment tracks the molecule

`AdvancedMoleculeEnv` retains version 4's main state variables:

```python
self.node_types
self.current_bonds
self.bonds
```

For OH, constructed with oxygen first:

```python
node_types    = [1, 0]       # O, H
current_bonds = [1, 1]       # Bond-order sum at each atom
bonds         = {(0, 1): 1}  # One single O-H bond
```

`get_valid_actions()` determines which actions satisfy the configured valence rules.

At this OH state:

- Adding another H to O is allowed.
- Adding another atom to the already bonded H is blocked.
- STOP is allowed.

**The environment does not decide whether OH or H₂O is the intended output.** The model makes that decision using the composition input.

## 5. Generate teaching sequences

`ReferenceSpecies.enumerate_sequences()` exhaustively explores valid reference construction routes. It deduplicates complete action tuples, so atom labels do not create duplicate examples. `build_demonstrations()` replays every unique route, checks legal targets and verifies the final graph.

Bond orders are created directly, and STOP occurs when the reference is complete. These are graph-building routes, not physical reaction pathways. `formation_sequences.json` saves all routes and their total/sample counts.

One possible H₂O sequence is:

```text
Empty → O → OH → H2O → STOP
```

Another starts from hydrogen:

```text
Empty → H → HO → H2O → STOP
```

For each step, the script stores:

```text
Current partial graph
Desired composition
Correct next action
Equivalent correct actions, if any
```

It saves the current graph **before** applying the action. Otherwise, the input would accidentally show the result the network is supposed to predict.

Every sequence is replayed and checked against its reference graph. A wrong construction sequence cannot enter training.

## 6. Convert each partial graph into tensors

`GrowthGNN.encode()` prepares the network inputs.

### Atom features

Each atom has five values:

```text
[is_H, is_O, is_C, bond_order_sum / 4, remaining_valence / 4]
```

For the oxygen in OH:

```text
[0, 1, 0, 0.25, 0.25]
```

It is oxygen, currently uses one bond-order unit, and has one remaining under the configured limit of two.

### Bond information

There are three adjacency channels:

```text
Channel 0: single bonds
Channel 1: double bonds
Channel 2: triple bonds
```

### Composition context

The model receives three vectors, each ordered **C, H, O**:

```text
Desired composition
Current composition
Desired − current composition
```

For OH while building H₂O, these are:

```text
Desired:   [0, 2, 1]
Current:   [0, 1, 1]
Remaining: [0, 1, 0]
```

These counts are scaled by `MAX_ATOMS`.

Graphs are padded to the same size for batching. A node mask distinguishes real atoms from padding.

## 7. The shared GNN scores actions

`GrowthGNN.forward()` passes atom features through the configured message-passing layers.

Each `BondMessageLayer` combines:

- Information about the atom itself.
- Information from its single-bond neighbors.
- Information from its double-bond neighbors.
- Information from its triple-bond neighbors.

The resulting atom embeddings summarize their local structural surroundings.

The model then combines a pooled graph summary with the composition context and produces scores through:

```python
start_head
grow_head
connect_head
termination_layer
```

All scores are concatenated into **one action vector**. Illegal actions receive `-inf`, giving them zero probability after softmax.

The same output distribution is used for training and generation.

## 8. Train using demonstrated partial graphs

Each epoch selects `K_s = ceil(N_s * TRAJECTORY_SAMPLE_FRACTION)` trajectories for species `s`, without replacement within that epoch. Here `N_s` is its total unique sequence count. For CH3CHO, `ceil(100 * 0.20) = 20`. A species with one sequence still supplies one. The seeded RNG persists across epochs, producing fresh selections; sequences may repeat between epochs.

All their partial-graph examples are processed together, followed by one optimizer update.

The main calculation is:

```python
logits = model(*inputs)
log_probs = F.log_softmax(logits, dim=-1)

target_log_prob = torch.logsumexp(
    log_probs.masked_fill(~targets, -torch.inf),
    dim=1,
)

loss = -(weights * target_log_prob).sum()
```

This means:

> Increase the total probability assigned to the demonstrated action and any symmetry-equivalent alternatives.

Each action receives weight `1 / (S * K_s * T_s,k)`:

- `s`: species index; `S`: number of species (31).
- `k`: selected trajectory index; `K_s`: number selected for that species.
- `t`: action step index; `T_s,k`: steps in that trajectory, including START and STOP.

Average action losses within a sequence, sequence losses within a species, then species losses. Each species therefore contributes total weight `1/S`, even when it supplies more sequences. The target probability sums the demonstrated action and its symmetry-equivalent alternatives, rather than all possible future construction choices.

Then:

```python
loss.backward()
clip_grad_norm_(...)
optimizer.step()
```

computes gradients, limits unusually large updates, and adjusts the shared network weights.

During this process, the next input comes from the demonstration. A wrong prediction does not alter it. This is **teacher forcing**.

## 9. Evaluate independent generation

Periodically, `evaluate()` starts fresh empty molecules and lets the model choose its own actions.

Now, unlike training, a wrong action changes what the model sees next.

Two modes are used:

- **Greedy:** choose the highest-probability action.
- **Sampled:** draw an action according to the probabilities.

Success requires:

```text
Explicit STOP
    AND
Requested composition
    AND
Exact reference connectivity and bond orders
```

Graph matching ignores arbitrary node numbering.

For a shared composition such as CH₃O/CH₂OH, either reference isomer is accepted. Sampled evaluation checks whether both appear.

## 10. Save the run

The main outputs are:

| File | Purpose |
|---|---|
| `training.log` | Same readable messages shown in the terminal |
| `training_examples.json` | One illustrated-by-data construction sequence per species |
| `metrics.csv` | Loss and action-prediction metrics for every epoch |
| `greedy_epoch_....json/.csv` | Periodic independent-generation results |
| `generated_species_inventory.json/.csv` | Final sampled molecules and reference matches |
| `formation_sequences.json` | All unique sequences, total counts and sampled counts per species |
| `trained_growth_gnn.pt` | Network weights, optimizer state, configuration, and action list |

**The key distinction throughout the script is:** training teaches the correct next action from a known partial graph; evaluation checks whether those learned decisions can build a complete reference molecule without guidance.
