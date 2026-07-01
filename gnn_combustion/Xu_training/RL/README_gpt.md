# Reinforcement Learning GNN Molecule Growth Demo

This directory contains `grow_train_animation.py`, a small reinforcement-learning style experiment that trains a graph neural network policy to grow and connect H/O atoms into chemically valid small species.

The script simulates repeated molecule-building episodes, rewards final inventories that belong to a predefined H/O combustion species set, and saves visualizations of the discovered molecular structures.

## Target Species

The environment treats the following formulas as valid:

```python
{"H2", "O2", "OH", "HO2", "H2O", "H2O2", "H", "O"}
```

Atoms are represented with two node types:

- `ATOM_H = 0`
- `ATOM_O = 1`

Valency constraints are enforced during action selection:

- H: maximum valency 1
- O: maximum valency 2

## Main Components

### `AdvancedMoleculeEnv`

The molecule-building environment. It starts each episode with two H atoms and two O atoms, then lets the policy:

- grow a new H or O atom from an existing atom,
- connect two existing atoms with a bond,
- terminate the episode.

The environment tracks disconnected molecular fragments using a union-find style mapping, then evaluates the final inventory as a list of formulas.

#### Internal State

`AdvancedMoleculeEnv` stores the molecule as lightweight Python lists and only converts it to a PyTorch Geometric graph when the model needs an input.

| Attribute         | Type                    | Meaning                                                                                                   |
| ----------------- | ----------------------- | --------------------------------------------------------------------------------------------------------- |
| `node_types`      | `list[int]`             | Atom type for each local node. `0` means H and `1` means O.                                               |
| `current_bonds`   | `list[int]`             | Current valency count for each local node. Used to prevent invalid growth or bond formation.              |
| `edges`           | `list[tuple[int, int]]` | Global node-id bonds in the current episode.                                                              |
| `terminated`      | `bool`                  | Whether the current growth episode has stopped.                                                           |
| `offset`          | `int`                   | Global node-id offset, allowing many epoch graphs to share one visualization graph without id collisions. |
| `node_to_species` | `dict[int, set[int]]`   | Union-find style mapping from each global atom id to the connected component/species it belongs to.       |

#### `__init__(node_offset=0)`

Initializes one growth episode.

- Starts with four disconnected atoms: `H, H, O, O`.
- Sets all bond counts to zero.
- Creates one singleton species set per atom.
- Stores `node_offset` so local atom ids can be translated into global visualization ids.

The initial state is intentionally simple: the agent must decide whether to connect existing atoms, grow new atoms, or stop.

#### `get_pyg_data()`

Converts the current molecule state into a `torch_geometric.data.Data` object.

Returned fields:

| Field        |                Shape | Meaning                                                                   |
| ------------ | -------------------: | ------------------------------------------------------------------------- |
| `x`          |     `[num_nodes, 2]` | One-hot atom features. H is `[1.0, 0.0]`; O is `[0.0, 1.0]`.              |
| `edge_index` | `[2, 2 * num_edges]` | Undirected PyG edge list. Each bond is mirrored as `u -> v` and `v -> u`. |

If there are no bonds yet, `edge_index` is an empty tensor with shape `[2, 0]`.

This method is the bridge between the chemistry environment and the GNN policy.

#### `get_valid_action_masks()`

Builds masks that block chemically invalid actions before the policy acts.

Returns:

```python
grow_mask, edge_connect_mask
```

| Return value | Shape | Meaning |
|---|---:|---|
| `grow_mask` | `[num_nodes]` | `1.0` if a node can accept a newly grown atom, otherwise `0.0`. |
| `edge_connect_mask` | `[num_nodes, num_nodes]` | `1.0` if two existing nodes can be bonded, otherwise `0.0`. |

A growth action is valid only if the parent atom has remaining valency.

A connection action is valid only if:

- the two atoms are not the same atom,
- both atoms have remaining valency,
- the bond does not already exist.

These masks are later converted into large negative logit penalties inside `GrowthGNN.forward()`.

#### `step(action_class, u_global, v_global=None, new_atom_type=None)`

Applies one action to the environment and updates graph state.

Action classes:

| `action_class` | Action | Required arguments | Effect |
|---:|---|---|---|
| `0` | Grow | `u_global`, `new_atom_type` | Adds a new H/O atom and bonds it to `u_global`. |
| `1` | Connect | `u_global`, `v_global` | Adds a bond between two existing atoms. |
| `2` | Terminate | `u_global` placeholder | Marks the episode as complete. |

For growth:

- appends the new atom type to `node_types`,
- gives the new atom one bond,
- increments the parent atom bond count,
- adds the new bond to `edges`,
- inserts the new atom into the parent atom's species set.

For connection:

- adds the bond to `edges`,
- increments both atoms' bond counts,
- merges two species sets if the bond joins previously disconnected fragments.

After any non-termination action, the method recomputes valid masks. If no growth or connection action remains, the episode auto-terminates.

Returns:

```python
terminated: bool
```

#### `evaluate_inventory()`

Converts connected atom groups into formulas and checks whether every resulting species is valid.

Workflow:

1. Collect unique connected components from `node_to_species`.
2. Count H and O atoms in each component.
3. Convert each component into a formula string.
4. Apply special formatting for `H2O`, `HO2`, and `H2O2`.
5. Check each formula against `VALID_SPECIES`.

Returns:

```python
formulas, all_legal
```

| Return value | Type | Meaning |
|---|---|---|
| `formulas` | `list[str]` | Final species inventory, such as `["H2", "O2"]` or `["H2O", "O"]`. |
| `all_legal` | `bool` | `True` only if every formula appears in `VALID_SPECIES`. |

The training loop uses `all_legal` to assign the episode reward:

```python
reward = 1.0 if success else -1.0
```

### `GrowthGNN`

The policy model built with PyTorch Geometric.

It uses two `GCNConv` layers to embed the current molecular graph and has three decision heads:

- `grow_head`: scores adding H or O to each valid node.
- `connect_head`: scores bonding each valid atom pair.
- `termination_layer`: scores whether the current graph should stop growing.

Invalid growth or connection actions are masked with large negative logits so the policy does not choose chemically impossible moves.

#### Architecture

`GrowthGNN` is a multi-head policy network over graph states.

| Layer/head | Input | Output | Purpose |
|---|---:|---:|---|
| `conv1` | `2` atom features | `hidden_dim` | First graph convolution over local atom neighborhoods. |
| `conv2` | `hidden_dim` | `hidden_dim` | Second graph convolution for deeper structural context. |
| `grow_head` | `hidden_dim` | `2` | Scores growing either H or O from each node. |
| `connect_head` | `2 * hidden_dim` | `1` | Scores forming a bond between each ordered atom pair. |
| `termination_layer` | `hidden_dim` graph embedding | `1` | Scores whether the current episode should stop. |

The default hidden size is:

```python
GrowthGNN(hidden_dim=32)
```

#### `__init__(hidden_dim=32)`

Defines all neural modules used by the policy.

Key design choices:

- The input atom feature size is fixed at `2`, matching H/O one-hot features.
- The two GCN layers let node embeddings depend on nearby bonded atoms.
- The grow head is node-local: each atom receives two scores, one for adding H and one for adding O.
- The connect head is pairwise: each possible `(u, v)` atom pair receives one bond-formation score.
- The termination head is graph-level: it pools all node embeddings into one graph embedding before scoring completion.

If more atom types are added later, both the input feature size and grow output size must be updated.

#### `forward(data, grow_mask, edge_mask)`

Computes action logits for the current molecule graph.

Inputs:

| Argument | Expected shape | Meaning |
|---|---:|---|
| `data.x` | `[num_nodes, 2]` | H/O one-hot atom features. |
| `data.edge_index` | `[2, num_directed_edges]` | PyG graph connectivity. |
| `grow_mask` | `[num_nodes]` | Validity mask for growing from each node. |
| `edge_mask` | `[num_nodes, num_nodes]` | Validity mask for connecting each atom pair. |

Forward pass:

1. Run graph message passing:

   ```python
   h = leaky_relu(conv1(x, edge_index))
   h = leaky_relu(conv2(h, edge_index))
   ```

2. Compute growth logits:

   ```python
   grow_logits = grow_head(h)
   ```

   Shape: `[num_nodes, 2]`.

   Entry `[i, 0]` scores adding H to node `i`; entry `[i, 1]` scores adding O to node `i`.

3. Compute connection logits:

   ```python
   pair_features = concat(h_i, h_j)
   connect_logits = connect_head(pair_features)
   ```

   Shape: `[num_nodes, num_nodes]`.

   Entry `[i, j]` scores connecting node `i` to node `j`.

4. Compute termination logit:

   ```python
   graph_latent = global_mean_pool(h, batch=None)
   term_logit = termination_layer(graph_latent)
   ```

   Shape: `[1, 1]` for a single graph.

5. Apply masks by adding a large negative value to invalid actions:

   ```python
   invalid_logit_penalty = -1e9
   ```

   This makes invalid growth and connection moves effectively impossible after softmax or argmax selection.

Returns:

```python
grow_logits, connect_logits, term_logit
```

| Return value | Shape | Used for |
|---|---:|---|
| `grow_logits` | `[num_nodes, 2]` | Choosing which atom to grow from and whether to add H or O. |
| `connect_logits` | `[num_nodes, num_nodes]` | Choosing which existing atom pair to bond. |
| `term_logit` | `[1, 1]` | Deciding whether to terminate the episode. |

In the current training loop, growth and connection logits are perturbed with Gaussian noise for exploration, while `term_logit` is passed through a sigmoid and terminates the episode when the probability exceeds `0.90`.

### Training Loop

The training logic lives mainly in lines 217-361 of `grow_train_animation.py`. It is organized as repeated independent growth episodes. Each episode starts from the same initial atom pool, lets the GNN policy choose structural actions, evaluates the final molecular inventory, and updates the policy from the resulting success or failure.

This is closer to a compact policy-gradient-style toy loop than to supervised training: there are no labeled target graphs. The model is rewarded when the final disconnected fragments are all valid H/O species.

#### 1. Model And Optimizer Setup

Lines 217-218 create the policy network and optimizer:

```python
gnn_model = GrowthGNN(hidden_dim=32)
optimizer = torch.optim.Adam(gnn_model.parameters(), lr=0.01)
```

The GNN uses a shared hidden dimension of `32`. The Adam optimizer updates all GNN parameters, including:

- `conv1`
- `conv2`
- `grow_head`
- `connect_head`
- `termination_layer`

#### 2. Global Visualization State

Lines 220-228 initialize global bookkeeping:

| Variable                    | Purpose                                                                                          |
| --------------------------- | ------------------------------------------------------------------------------------------------ |
| `global_G`                  | A NetworkX graph containing all epoch structures for visualization.                              |
| `global_pos`                | Final 2D coordinates for every atom node.                                                        |
| `epoch_snapshots`           | Per-epoch graph snapshots used later for images/GIF generation.                                  |
| `global_discovery_registry` | Initializes a registry for valid species, although the current loop does not actively update it. |
| `total_epochs`              | Number of independent training episodes. Default is `50`.                                        |
| `global_node_counter`       | Tracks the next global node id so each epoch has unique atom ids.                                |

Lines 230-236 choose visualization scaling based on `total_epochs`. This does not affect learning; it only controls node size, label size, and canvas padding in the output plots.

#### 3. Epoch Initialization

Each epoch begins at line 240:

```python
for epoch in range(1, total_epochs + 1):
```

At the start of every epoch:

```python
optimizer.zero_grad()
env = AdvancedMoleculeEnv(node_offset=global_node_counter)
```

The environment is reset to four disconnected atoms:

```text
H, H, O, O
```

The `node_offset` ensures that this epoch's local atom ids map to unique global ids in `global_G`.

Lines 244-257 add the initial four atoms to the global visualization graph and reserve their ids:

```python
epoch_initial_nodes = [
    (global_node_counter + 0, ATOM_H),
    (global_node_counter + 1, ATOM_H),
    (global_node_counter + 2, ATOM_O),
    (global_node_counter + 3, ATOM_O)
]
```

The training-specific variables are then initialized:

| Variable           | Purpose                                                                            |
| ------------------ | ---------------------------------------------------------------------------------- |
| `steps`            | Counts structural decisions inside the current episode.                            |
| `action_log_probs` | Stores log probabilities of selected grow/connect actions for policy loss.         |
| `epoch_step_data`  | Stores step-level graph snapshots, although final plotting uses `epoch_snapshots`. |

#### 4. Inner Action Loop

The actual molecule growth happens in lines 263-334:

```python
while not env.terminated and steps < 8:
```

The episode stops when either:

- the environment terminates,
- the policy chooses termination,
- no valid growth or connection actions remain,
- or the hard cap of `8` structural steps is reached.

At each step, the current environment is converted into a PyTorch Geometric graph:

```python
pyg_data = env.get_pyg_data()
grow_mask, edge_mask = env.get_valid_action_masks()
```

The masks encode chemical validity before the model acts:

- `grow_mask`: which atoms still have free valency for growing a new atom.
- `edge_mask`: which existing atom pairs can still be connected.

The GNN then evaluates the current graph:

```python
grow_logits, connect_logits, term_logit = gnn_model(pyg_data, grow_mask, edge_mask)
```

These outputs represent:

| Output           |                    Shape | Meaning                                             |
| ---------------- | -----------------------: | --------------------------------------------------- |
| `grow_logits`    |         `[num_nodes, 2]` | Scores adding H or O to each existing atom.         |
| `connect_logits` | `[num_nodes, num_nodes]` | Scores bonding each ordered pair of existing atoms. |
| `term_logit`     |                 `[1, 1]` | Scores stopping the episode.                        |

#### 5. Action Scoring And Exploration

Lines 269-275 flatten growth and connection logits into a common action space:

```python
flat_grow = grow_logits.flatten()
flat_conn = connect_logits.flatten()
combined_logits = torch.cat([flat_grow, flat_conn])
probs = F.softmax(combined_logits, dim=-1)
```

`combined_logits` and `probs` show the intended unified action distribution, but the current code does not sample directly from `probs`. Instead, it uses noisy argmax selection.

Lines 278-281 add Gaussian noise to encourage exploration:

```python
noise_grow = torch.randn_like(grow_logits) * 0.4
noise_conn = torch.randn_like(connect_logits) * 0.4
max_grow_val = torch.max(grow_logits + noise_grow).item()
max_conn_val = torch.max(connect_logits + noise_conn).item()
```

The noise scale is `0.4`. Larger noise would make the search more exploratory; smaller noise would make it more greedy.

#### 6. Termination Decision

Lines 283-286 evaluate the termination head:

```python
prob_stop = torch.sigmoid(term_logit).item()
if prob_stop > 0.90 or (sum(grow_mask) == 0 and torch.sum(edge_mask) == 0):
    env.step(action_class=2, u_global=env.offset)
    break
```

The episode terminates if:

- the learned stop probability exceeds `0.90`,
- or there are no valid grow/connect actions left.

The termination decision is trained later using binary cross entropy against final success or failure.

#### 7. Grow Action

If the best noisy grow score is at least as large as the best noisy connect score, lines 291-309 perform a grow action.

The selected flat index is decoded as:

```python
u_local = flat_idx // 2
chosen_atom = flat_idx % 2
```

This works because each node has two growth choices:

| `chosen_atom` | Added atom |
|---:|---|
| `0` | H |
| `1` | O |

The log probability of the selected grow action is saved:

```python
action_log_probs.append(
    torch.log(F.softmax(grow_logits.flatten(), dim=-1)[flat_idx] + 1e-8)
)
```

Then the new atom is added to both:

- `global_G`, for visualization,
- `env`, for the actual chemical state.

```python
global_G.add_node(new_global_idx, element="H" if chosen_atom == 0 else "O")
env.step(action_class=0, u_global=u_global, new_atom_type=chosen_atom)
global_G.add_edge(u_global, new_global_idx)
```

The environment updates valency counts, edge lists, and connected species membership.

#### 8. Connect Action

If the best noisy connection score is larger, lines 310-326 perform a bond-connection action.

The selected flat index is decoded back into two atom ids:

```python
num_nodes = connect_logits.size(0)
u_global = (flat_idx // num_nodes) + env.offset
v_global = (flat_idx % num_nodes) + env.offset
```

The log probability of this connection is saved:

```python
action_log_probs.append(
    torch.log(F.softmax(connect_logits.flatten(), dim=-1)[flat_idx] + 1e-8)
)
```

Then the bond is applied:

```python
env.step(action_class=1, u_global=u_global, v_global=v_global)
global_G.add_edge(u_global, v_global)
```

The environment handles valency updates and merges species sets if this bond joins two previously disconnected fragments.

#### 9. Step Snapshot

Lines 328-334 store a step-level snapshot:

```python
current_state = {
    'step': steps,
    'edges': list(global_G.edges()),
    'nodes': list(global_G.nodes(data=True))
}
epoch_step_data.append(current_state)
steps += 1
```

This records the evolving graph after each structural move. In the current script, these step snapshots are collected but not used in the final image/GIF generation; the later `epoch_snapshots` list drives the final visual outputs.

#### 10. Final Evaluation

After the episode ends, lines 336-340 recompute the final graph state and evaluate the species inventory:

```python
final_pyg = env.get_pyg_data()
g_mask, e_mask = env.get_valid_action_masks()
_, _, final_term_logit = gnn_model(final_pyg, g_mask, e_mask)

formulas_list, success = env.evaluate_inventory()
```

`evaluate_inventory()` checks whether every connected fragment corresponds to one of the allowed formulas in `VALID_SPECIES`.

Example valid inventories:

```text
H2 + O2
H2O + O
OH + OH
HO2 + H
```

If any fragment is outside `VALID_SPECIES`, the episode is considered unsuccessful.

#### 11. Reward

Lines 342-343 assign a scalar reward:

```python
reward = 1.0 if success else -1.0
```

This means the current reward function is binary:

| Final inventory               | Reward |
| ----------------------------- | -----: |
| All fragments valid           | `+1.0` |
| At least one invalid fragment | `-1.0` |

The reward does not currently distinguish between different valid products. For example, `H2 + O2` and `H2O + O` both receive the same positive reward.

#### 12. Termination Loss

Lines 345-347 train the termination head:

```python
target_term = torch.tensor([[1.0 if success else 0.0]], dtype=torch.float)
term_loss = F.binary_cross_entropy_with_logits(final_term_logit, target_term)
```

The termination head is trained to output:

- `1.0` when the final structure is successful,
- `0.0` when the final structure is unsuccessful.

Conceptually, this teaches the model whether a completed graph state looks worth stopping on.

#### 13. Policy Loss

Lines 349-352 compute a simple policy-style objective:

```python
policy_loss = 0
if len(action_log_probs) > 0:
    policy_loss = -torch.stack(action_log_probs).mean() * reward
```

The effect depends on the reward:

| Reward | Effect on selected actions |
|---:|---|
| `+1.0` | Increases probability of actions taken in successful episodes. |
| `-1.0` | Decreases probability of actions taken in failed episodes. |

Because the loss uses the mean of `action_log_probs`, every grow/connect action in an episode receives the same final reward signal.

Important detail: termination choices are not added to `action_log_probs`. The termination behavior is trained separately through `term_loss`.

#### 14. Total Loss And Optimizer Step

Lines 354-357 combine the two losses and update the model:

```python
total_loss = term_loss + 0.5 * policy_loss
total_loss.backward()
optimizer.step()
```

The combined objective is:

```text
total_loss = termination_loss + 0.5 * policy_loss
```

The `0.5` factor reduces the strength of the policy-gradient term relative to the termination classification term.

#### 15. Console Output

Lines 359-361 print a compact training log:

```python
mixture_str = " + ".join(formulas_list)
print(f"Epoch {epoch:02d}/{total_epochs} | System Output: {mixture_str:<22} | Training Loss: {total_loss.item():.5f}")
```

Each epoch reports:

- epoch number,
- final species inventory,
- total training loss.

Example:

```text
Epoch 01/50 | System Output: H2 + O2                | Training Loss: 0.31452
```

#### Training Summary

The overall training cycle is:

```text
initialize H,H,O,O
-> convert graph to PyG Data
-> GNN scores grow/connect/terminate actions
-> masks block invalid chemistry
-> noisy argmax selects grow or connect
-> environment updates bonds/species sets
-> final inventory is checked against VALID_SPECIES
-> reward is +1 or -1
-> termination loss + policy loss update the GNN
```

#### Practical Notes

- `combined_logits` and `probs` are computed, but the current code uses noisy argmax rather than sampling from the combined categorical distribution.
- The action mask logic is what keeps the generated structures chemically plausible under the simple valency model.
- The reward is sparse and delayed until the end of an episode.
- Since all selected actions share one final reward, the model does not know which individual action caused success or failure.
- The hard step limit `steps < 8` prevents unbounded molecule growth.
- Training is stochastic because exploration noise is sampled at every step and no random seed is set.

## Outputs

Running the script creates:

- `individual_plots/epoch_<N>_final.png`: one final molecule layout image per epoch.
- `relaxed_8species_growth.gif`: an animated global canvas showing the accumulated epoch outputs.

Both output paths are relative to the directory where the script is launched. For clean output placement, run the script from this `RL` directory.

## Configuration

The most important knob is inside the `if __name__ == "__main__":` block:

```python
total_epochs = 50
```

Changing this controls how many independent growth episodes are simulated and visualized. The script automatically adjusts node sizes, label sizes, and canvas padding for larger epoch counts.

Other useful constants:

- `VALID_SPECIES`: controls which final species receive positive reward.
- `MAX_VALENCY`: controls the allowed valency of each atom type.
- `hidden_dim=32`: controls the GNN embedding size.
- `lr=0.01`: optimizer learning rate.
- `steps < 8`: maximum structural actions per episode.

## Notes And Limitations

- This is a compact proof-of-concept, not a full combustion chemistry generator.
- The reward checks formula validity only; it does not include thermodynamics, kinetics, reaction rates, stoichiometric balance, or pathway likelihood.
- The policy currently uses noisy argmax-style action selection rather than sampling directly from the full categorical distribution.
- Results are not seeded, so repeated runs may produce different growth histories.
