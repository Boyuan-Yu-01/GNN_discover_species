# Code Interpretation: `Xu_grow_train_animation_v2.py`

This document explains the script function by function and block by block. The script simulates repeated molecule-growth trials, uses a graph neural network policy to choose atom-growth and bond-connection actions, checks whether the resulting molecular fragments are valid H/O species, trains from the success/failure signal, and saves static images plus a GIF animation of the generated structures.
## High-Level Purpose

The script models a small chemical graph-building environment using hydrogen and oxygen atoms. Each epoch starts from four atoms:

```text
H, H, O, O
```

The model then repeatedly chooses actions such as:

```text
grow a new atom from an existing atom
connect two existing atoms
terminate the construction
```

After the construction ends, the environment checks whether the disconnected molecular fragments correspond to allowed species:

```python
VALID_SPECIES = {"H2", "O2", "OH", "HO2", "H2O", "H2O2", "H", "O"}
```

The GNN is trained with a simple reward signal:

```text
valid inventory   -> reward = +1
invalid inventory -> reward = -1
```

The script also lays out each epoch's generated structure in 2D and exports:

```text
individual_plots/epoch_<n>_final.png
relaxed_8species_growth.gif
```

## Imports

```python
import os
```

Used to create the output directory for standalone image files.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
```

Used for neural-network modeling, tensors, activation functions, loss functions, and optimization.

```python
import numpy as np
```

Used for geometric layout calculations, especially 2D coordinates for molecule placement.

```python
import matplotlib.pyplot as plt
import matplotlib.animation as animation
```

Used to draw PNG images and generate the final GIF animation.

```python
import networkx as nx
```

Used to store and draw the molecule graph globally across epochs.

```python
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
```

Used to convert molecule graphs into PyTorch Geometric graph objects and to define the graph convolutional neural network.

```python
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
```

Suppresses selected warning messages so the console output is cleaner.

## Global Chemistry Configuration

```python
VALID_SPECIES = {"H2", "O2", "OH", "HO2", "H2O", "H2O2", "H", "O"}
```

Defines which final molecular formulas are considered chemically valid by this toy environment.

```python
ATOM_H = 0
ATOM_O = 1
```

Encodes hydrogen and oxygen as integer classes. This is useful because neural networks and tensors work naturally with numeric labels.

```python
MAX_VALENCY = {ATOM_H: 1, ATOM_O: 2}
```

Defines the maximum allowed bond count for each atom type:

```text
H can have at most 1 bond
O can have at most 2 bonds
```

This is a simplified valency model. It is used to reject invalid graph actions.

## Class: `AdvancedMoleculeEnv`

`AdvancedMoleculeEnv` is the chemical graph environment. It stores atom types, bonds, valency usage, molecule-fragment membership, and whether the current construction has terminated.

The name mentions union-find tracking. The implementation does not use a formal union-find data structure with parent pointers, but it uses shared Python sets to track which atoms belong to the same connected molecular fragment.

### `__init__(self, node_offset=0)`

Initializes one molecule-growth environment.

```python
self.node_types = [ATOM_H, ATOM_H, ATOM_O, ATOM_O]
```

Starts every episode with four atoms:

```text
node 0: H
node 1: H
node 2: O
node 3: O
```

If `node_offset` is not zero, the global node IDs shift by that offset.

```python
self.current_bonds = [0, 0, 0, 0]
```

Tracks how many bond units each atom currently uses. A single bond consumes 1. A double bond consumes 2.

```python
self.bonds = {}
```

Stores bonds as a dictionary:

```python
{(u, v): bond_order}
```

where `(u, v)` is a sorted global-node pair and `bond_order` is typically 1 or 2.

```python
self.success = False
self.terminated = False
```

Stores whether the environment has succeeded and whether the episode has ended.

```python
self.offset = node_offset
```

Stores the global node offset. This allows each epoch to use unique global node IDs while the local environment still works with local indices.

```python
self.node_to_species = {
    self.offset + i: {self.offset + i} for i in range(4)
}
```

Initially, each atom is its own molecular fragment. For example, if `offset = 100`, then:

```python
{
  100: {100},
  101: {101},
  102: {102},
  103: {103}
}
```

When bonds connect atoms, these sets are merged.

### `get_pyg_data(self)`

Converts the current molecule graph into a PyTorch Geometric `Data` object.

```python
x = [[1.0, 0.0] if t == ATOM_H else [0.0, 1.0] for t in self.node_types]
```

Creates one-hot atom features:

```text
H -> [1.0, 0.0]
O -> [0.0, 1.0]
```

These become node features for the GNN.

```python
x = torch.tensor(x, dtype=torch.float)
```

Converts the Python list into a PyTorch tensor.

```python
edge_list = []
for (u, v), order in self.bonds.items():
    for _ in range(order):
        edge_list.append([u - self.offset, v - self.offset])
        edge_list.append([v - self.offset, u - self.offset])
```

Builds a directed edge list from the bond dictionary. Each chemical bond is added in both directions because `GCNConv` uses directed edge indices to pass messages.

If the bond order is 2, the code adds duplicate edges. This is a simple way to make double bonds appear stronger to the GNN, because message passing sees two edge entries instead of one.

```python
edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous() if edge_list else torch.empty((2, 0), dtype=torch.long)
```

Creates the PyTorch Geometric `edge_index` tensor with shape:

```text
[2, number_of_edges]
```

If there are no bonds, it creates an empty edge tensor.

```python
return Data(x=x, edge_index=edge_index)
```

Returns the graph object consumed by the GNN.

### `get_valid_action_masks(self)`

Computes which actions are currently legal.

The function returns two masks:

```python
node_grow_mask, edge_connect_mask
```

The GNN uses these masks to strongly suppress invalid actions.

```python
num_nodes = len(self.node_types)
```

Gets the current number of atoms.

```python
node_grow_mask = [
    1.0 if self.current_bonds[i] < MAX_VALENCY[self.node_types[i]] else 0.0
    for i in range(num_nodes)
]
```

For each atom, this says whether a new atom can be grown from it. If the atom still has unused valency, growth is allowed.

Example:

```text
H with 0 bonds -> can grow
H with 1 bond  -> cannot grow
O with 1 bond  -> can grow
O with 2 bonds -> cannot grow
```

```python
edge_connect_mask = torch.zeros((num_nodes, num_nodes, 2), dtype=torch.float)
```

Creates a mask for connecting pairs of existing atoms. The shape is:

```text
(source_node, target_node, bond_type)
```

The last dimension has size 2:

```text
index 0 -> single bond action
index 1 -> double bond action
```

```python
if i == j: continue
```

Prevents an atom from bonding to itself.

```python
if current_bonds[i] < max_valency[i] and current_bonds[j] < max_valency[j]:
    edge_connect_mask[i, j] = 1.0
```

Initially marks both single and double bond choices as possible if both atoms still have any remaining valency.

The code then refines this choice based on current bond order.

```python
pair = tuple(sorted((u_global, v_global)))
order = self.bonds.get(pair, 0)
```

Uses sorted global indices to identify the bond independent of direction.

```python
if order == 0 and valency allows:
    edge_connect_mask[i, j, 0] = 1.0
```

Allows a single bond only if no bond already exists between the pair.

```python
if order < 2 and capacity condition:
    edge_connect_mask[i, j, 1] = 1.0
```

Allows the double-bond action if the current bond order is below 2 and the atoms have enough valency.

Important implementation note: this mask is conceptually trying to encode valid single/double actions, but the double-bond capacity logic is imperfect. In `step`, the final valency check is stricter and ultimately decides whether the action is accepted.

```python
return torch.tensor(node_grow_mask, dtype=torch.float), edge_connect_mask
```

Returns masks for the GNN policy.

### `step(self, action_class, u_global, v_global=None, new_atom_type=None, bond_order_inc=0)`

Applies one action to the environment.

The supported action classes are:

```text
0 -> grow a new atom
1 -> connect two existing atoms
2 -> terminate successfully
```

The function returns:

```python
(success, terminated)
```

#### Termination Action

```python
if action_class == 2:
    self.terminated = True
    self.success = True
    return self.success, self.terminated
```

If the model chooses termination, the environment stops immediately and marks the termination action as successful. The final chemical validity is still evaluated later by `evaluate_inventory()`.

#### Growth Action

```python
u_local = u_global - self.offset
```

Converts a global node ID into a local node index.

```python
new_global_idx = len(self.node_types) + self.offset
```

Assigns the new atom a global ID.

```python
self.node_types.append(new_atom_type)
```

Adds the new atom type to the environment.

```python
self.current_bonds.append(1)
self.current_bonds[u_local] += 1
```

The new atom starts with one bond to its parent. The parent atom also uses one additional bond.

```python
self.bonds[(u_global, new_global_idx)] = 1
```

Stores a single bond between the parent atom and the new atom.

```python
u_species_set = self.node_to_species[u_global]
u_species_set.add(new_global_idx)
self.node_to_species[new_global_idx] = u_species_set
```

Adds the new atom to the same molecular fragment as the parent atom.

```python
return True, False
```

Growth succeeded, but the episode does not terminate.

#### Connection Action

```python
pair = tuple(sorted((u_global, v_global)))
current_order = self.bonds.get(pair, 0)
order_increment = bond_order_inc
```

Identifies the pair and reads the current bond order.

```python
if current_order + order_increment > 2:
    return False, False
```

Rejects actions that would create a bond order above 2.

```python
u_remaining = MAX_VALENCY[self.node_types[u_local]] - self.current_bonds[u_local]
v_remaining = MAX_VALENCY[self.node_types[v_local]] - self.current_bonds[v_local]
```

Computes remaining valency capacity for both atoms.

```python
if order_increment <= u_remaining and order_increment <= v_remaining:
```

Only accepts the action if both atoms have enough unused valency.

```python
self.bonds[pair] = current_order + order_increment
self.current_bonds[u_local] += order_increment
self.current_bonds[v_local] += order_increment
```

Updates the bond order and valency usage.

```python
if self.node_to_species[u_global] is not self.node_to_species[v_global]:
    merged_set = self.node_to_species[u_global].union(self.node_to_species[v_global])
    for idx in merged_set:
        self.node_to_species[idx] = merged_set
```

Merges the two molecular fragments if the new bond connects previously separate fragments.

```python
return True, False
```

The connection succeeded, but the episode continues.

If the valency check fails, the function returns:

```python
return False, False
```

### `evaluate_inventory(self)`

Checks whether all final molecular fragments are valid species.

```python
unique_sets = []
for s in self.node_to_species.values():
    if s not in unique_sets:
        unique_sets.append(s)
```

Collects unique molecular fragments. Each fragment is a set of atom IDs.

```python
num_H = sum(...)
num_O = sum(...)
```

Counts how many H and O atoms are in each fragment.

```python
H_part = f"H{num_H}" if num_H > 1 else ("H" if num_H == 1 else "")
O_part = f"O{num_O}" if num_O > 1 else ("O" if num_O == 1 else "")
```

Builds formula pieces such as:

```text
H
H2
O
O2
```

```python
if num_H == 1 and num_O == 1:
    formula = "OH"
else:
    formula = f"{H_part}{O_part}"
```

Special-cases one H and one O as `OH`, not `HO`.

```python
if formula in VALID_SPECIES:
    formulas.append(formula)
else:
    invalid_formula.append(formula)
    all_legal = False
```

Marks the inventory as legal only if every fragment belongs to `VALID_SPECIES`.

```python
return formulas, invalid_formula, all_legal
```

Returns:

```text
valid formulas
invalid formulas
whether the whole inventory is legal
```

## Class: `GrowthGNN`

`GrowthGNN` is the policy model. It reads the current molecule graph and scores possible structural actions.

It does not directly know chemistry rules. Chemistry constraints are supplied through action masks from the environment.

### `__init__(self, hidden_dim=32)`

Defines the neural network layers.

```python
self.conv1 = GCNConv(2, hidden_dim)
```

First graph convolution. Input feature size is 2 because atoms are represented as:

```text
H -> [1, 0]
O -> [0, 1]
```

```python
self.conv2 = GCNConv(hidden_dim, hidden_dim)
```

Second graph convolution. This lets node embeddings contain information from nearby bonded atoms.

```python
self.grow_head = nn.Linear(hidden_dim, 2)
```

For each node, predicts two growth logits:

```text
grow H from this node
grow O from this node
```

```python
self.connect_head = nn.Linear(hidden_dim * 2, 2)
```

For each ordered node pair, predicts two connection logits:

```text
single bond
double bond
```

The input size is `hidden_dim * 2` because the model concatenates the embedding of node `i` and node `j`.

```python
self.termination_layer = nn.Linear(hidden_dim, 1)
```

Predicts one termination logit from a graph-level embedding.

### `forward(self, data, grow_mask, edge_mask)`

Computes action logits.

```python
h = F.leaky_relu(self.conv1(data.x, data.edge_index), 0.1)
h = F.leaky_relu(self.conv2(h, data.edge_index), 0.1)
```

Runs two graph-convolution layers with leaky-ReLU activations.

`h` has shape:

```text
(number_of_nodes, hidden_dim)
```

```python
grow_logits = self.grow_head(h) + (grow_mask.unsqueeze(1) - 1.0) * 1e9
```

Computes growth-action scores and applies the grow mask.

If a mask value is:

```text
1 -> valid action, add 0
0 -> invalid action, add -1e9
```

So invalid growth actions become practically impossible after softmax or argmax.

```python
num_nodes = h.size(0)
h_i = h.unsqueeze(1).expand(-1, num_nodes, -1)
h_j = h.unsqueeze(0).expand(num_nodes, -1, -1)
```

Creates all ordered pairs of node embeddings.

```python
pair_features = torch.cat([h_i, h_j], dim=-1)
```

Concatenates each pair into a feature vector:

```text
[embedding_i, embedding_j]
```

```python
connect_logits = self.connect_head(pair_features) + (edge_mask - 1.0) * 1e9
```

Computes connection-action logits and masks invalid bond actions.

The shape is:

```text
(number_of_nodes, number_of_nodes, 2)
```

```python
term_logit = self.termination_layer(global_mean_pool(h, batch=None))
```

Computes a graph-level termination score from the mean of all node embeddings.

```python
return grow_logits, connect_logits, term_logit
```

Returns all action scores.

## Function: `get_spiral_coordinates(index, spacing=35.0)`

Maps each epoch index to a 2D coordinate on a square spiral.

The purpose is visual layout: each training epoch gets its own cluster position, so the final GIF can show many generated molecule systems without complete overlap.

```python
if index == 0:
    return 0.0, 0.0
```

Epoch 0 is placed at the origin.

```python
x = y = 0
dx = 0
dy = -1
```

Initializes an integer grid walk.

```python
step_limit = max_steps = 1
turns = 0
```

Controls when the spiral changes direction and when the straight-line segment length increases.

```python
for _ in range(index):
    x += dx
    y += dy
    step_limit -= 1
```

Moves one grid step at a time.

```python
if step_limit == 0:
    dx, dy = -dy, dx
    turns += 1
    if turns % 2 == 0:
        max_steps += 1
    step_limit = max_steps
```

Rotates the direction vector and increases the segment length every two turns. This creates a compact spiral pattern.

```python
return float(x * spacing), float(y * spacing)
```

Scales the integer grid coordinate into drawing coordinates.

## Main Block

The main block runs only when the script is executed directly:

```python
if __name__ == "__main__":
```

It will not run automatically if the file is imported as a module.

### Model and Optimizer

```python
gnn_model = GrowthGNN(hidden_dim=32)
optimizer = torch.optim.Adam(gnn_model.parameters(), lr=0.01)
```

Creates the policy GNN and an Adam optimizer.

The model has hidden node embeddings of size 32. The learning rate is 0.01.

### Global Graph and Layout State

```python
global_G = nx.Graph()
global_pos = {}
epoch_snapshots = []
```

`global_G` stores every atom and bond from every epoch.

`global_pos` stores each global node's 2D drawing position.

`epoch_snapshots` stores enough information to draw each animation frame later.

```python
global_discovery_registry = {sp: 0 for sp in VALID_SPECIES}
```

Creates a dictionary for tracking discovered species, but this variable is not used later in the current script.

```python
total_epochs = 50
global_node_counter = 0
```

Runs 50 independent construction attempts. `global_node_counter` ensures each epoch gets unique node IDs.

### Dynamic Visualization Styling

```python
if total_epochs <= 10:
    ...
elif total_epochs <= 25:
    ...
else:
    ...
```

Adjusts node sizes, label size, and plot margins depending on how many epochs will be drawn. More epochs require smaller nodes and more spacing.

## Per-Epoch Training Loop

```python
for epoch in range(1, total_epochs + 1):
```

Runs one molecule-growth episode per epoch.

```python
optimizer.zero_grad()
env = AdvancedMoleculeEnv(node_offset=global_node_counter)
```

Clears old gradients and creates a fresh environment. The offset ensures the new episode uses node IDs after all previous nodes.

```python
x_center, y_center = get_spiral_coordinates(epoch - 1, spacing=35.0)
cluster_center = np.array([x_center, y_center])
```

Computes where this epoch's molecules should be drawn.

```python
epoch_initial_nodes = [
    (global_node_counter + 0, ATOM_H),
    (global_node_counter + 1, ATOM_H),
    (global_node_counter + 2, ATOM_O),
    (global_node_counter + 3, ATOM_O)
]
```

Defines the four initial atoms in global-node coordinates.

```python
for idx, a_type in epoch_initial_nodes:
    global_G.add_node(idx, element="H" if a_type == ATOM_H else "O")
```

Adds these atoms to the global visualization graph.

```python
global_node_counter += 4
```

Advances the global node counter past the initial atoms.

## Per-Step Action Loop

```python
while not env.terminated and steps < 8:
```

Each epoch can take at most 8 structural steps.

```python
pyg_data = env.get_pyg_data()
grow_mask, edge_mask = env.get_valid_action_masks()
```

Gets the current molecule graph and valid action masks.

```python
grow_logits, connect_logits, term_logit = gnn_model(pyg_data, grow_mask, edge_mask)
```

Runs the GNN to score growth, connection, and termination actions.

```python
flat_grow = grow_logits.flatten()
flat_conn = connect_logits.flatten()
combined_logits = torch.cat([flat_grow, flat_conn])
probs = F.softmax(combined_logits, dim=-1)
```

Combines all structural action logits into one distribution. In the current script, `probs` is computed but not used for sampling. The actual choice is made by noisy argmax.

```python
noise_grow = torch.randn_like(grow_logits) * 0.4
noise_conn = torch.randn_like(connect_logits) * 0.4
```

Adds random exploration noise to action scores.

```python
max_grow_val = torch.max(grow_logits + noise_grow).item()
max_conn_val = torch.max(connect_logits + noise_conn).item()
```

Finds the best noisy growth action and best noisy connection action.

```python
prob_stop = torch.sigmoid(term_logit).item()
```

Converts the termination logit into a probability-like value between 0 and 1.

```python
if prob_stop > 0.90 or (sum(grow_mask) == 0 and torch.sum(edge_mask) == 0):
    env.step(action_class=2, u_global=env.offset)
    break
```

Terminates if the model strongly wants to stop or no structural actions remain.

## Growth Action Branch

The model grows a new atom if:

```python
max_grow_val >= max_conn_val
```

```python
flat_idx = torch.argmax(grow_logits + noise_grow).item()
```

Finds the highest-scoring growth action.

```python
action_log_probs.append(torch.log(F.softmax(grow_logits.flatten(), dim=-1)[flat_idx] + 1e-8))
```

Stores the log probability of the selected growth action. This is later used in the policy loss.

```python
u_local = flat_idx // 2
chosen_atom = flat_idx % 2
```

Decodes the flattened action index:

```text
u_local     -> parent atom
chosen_atom -> 0 for H, 1 for O
```

```python
u_global = u_local + env.offset
new_global_idx = len(env.node_types) + env.offset
```

Converts local node IDs to global node IDs.

```python
global_G.add_node(new_global_idx, element="H" if chosen_atom == 0 else "O")
env.step(action_class=0, u_global=u_global, new_atom_type=chosen_atom)
global_G.add_edge(u_global, new_global_idx)
```

Updates both the visualization graph and the chemical environment.

## Connection Action Branch

The model connects existing atoms if:

```python
max_conn_val > max_grow_val
```

```python
flat_idx = torch.argmax(connect_logits + noise_conn).item()
```

Finds the highest-scoring connection action.

```python
action_log_probs.append(torch.log(F.softmax(connect_logits.flatten(), dim=-1)[flat_idx] + 1e-8))
```

Stores the log probability for policy-gradient-like training.

```python
num_nodes = connect_logits.size(0)
u_local = (flat_idx // 2) // num_nodes
v_local = (flat_idx // 2) % num_nodes
bond_type = flat_idx % 2
```

Decodes the flattened connection index.

Because `connect_logits` has shape:

```text
(num_nodes, num_nodes, 2)
```

the flattened index contains:

```text
source node
target node
bond type
```

where:

```text
bond_type = 0 -> single bond
bond_type = 1 -> double bond
```

```python
bond_order = 1 + bond_type
```

Maps bond type to bond order:

```text
0 -> 1
1 -> 2
```

```python
success, terminated = env.step(
    action_class=1,
    u_global=u_global,
    v_global=v_global,
    bond_order_inc=bond_order
)
```

Attempts to add the bond in the environment.

```python
if success:
    global_G.add_edge(u_global, v_global)
```

Only updates the visualization graph if the environment accepts the bond.

## Step Snapshot

```python
current_state = {
    'step': steps,
    'edges': list(global_G.edges()),
    'nodes': list(global_G.nodes(data=True))
}
epoch_step_data.append(current_state)
```

Stores per-step graph state, but `epoch_step_data` is not used later in the current script.

```python
steps += 1
```

Advances the step counter.

## Final Evaluation and Training

```python
final_pyg = env.get_pyg_data()
g_mask, e_mask = env.get_valid_action_masks()
_, _, final_term_logit = gnn_model(final_pyg, g_mask, e_mask)
```

Runs the final graph through the model to get a final termination logit.

```python
formulas_list, invalid_formulas_list, success = env.evaluate_inventory()
```

Converts connected atom groups into molecular formulas and checks validity.

```python
reward = 1.0 if success else -1.0
```

Defines the episode reward.

```python
target_term = torch.tensor([[1.0 if success else 0.0]], dtype=torch.float)
term_loss = F.binary_cross_entropy_with_logits(final_term_logit, target_term)
```

Trains the termination head to output:

```text
1 if the final inventory is valid
0 if the final inventory is invalid
```

```python
policy_loss = 0
if len(action_log_probs) > 0:
    policy_loss = -torch.stack(action_log_probs).mean() * reward
```

This is a simple policy-gradient-style objective.

If `reward = +1`, the loss encourages the selected actions by making their log probabilities larger.

If `reward = -1`, the loss discourages the selected actions.

```python
total_loss = term_loss + 0.5 * policy_loss
total_loss.backward()
optimizer.step()
```

Combines termination loss and policy loss, then updates the GNN.

## Molecular Layout Generation

After each episode, the script computes drawing coordinates.

```python
unique_molecular_groups = []
for s in env.node_to_species.values():
    if s not in unique_molecular_groups:
        unique_molecular_groups.append(list(s))
```

Gets each disconnected molecular fragment.

```python
num_molecules = len(unique_molecular_groups)
```

Counts how many molecular fragments exist in the final inventory.

For multiple fragments:

```python
mol_angle = (2 * np.pi * mol_idx) / num_molecules
fragment_separation = 6.5 + (num_atoms_in_mol * 0.5)
mol_center = cluster_center + ...
```

Places each molecule around the epoch's cluster center.

For atoms inside a molecule:

```python
atom_angle = (2 * np.pi * atom_idx) / num_atoms_in_mol
dynamic_bond_radius = 3.2 + ...
global_pos[node_global_idx] = mol_center + ...
```

Places atoms in a small circular arrangement around the molecular center.

This is a visualization layout, not a physically optimized molecular geometry.

## Epoch Snapshot

```python
epoch_snapshots.append({
    'epoch_num': epoch,
    'node_offset': env.offset,
    'max_node_idx': len(global_G.nodes()),
    'edges': list(global_G.edges()),
    'text': f"Epoch {epoch}: {mixture_str}"
})
```

Stores enough data to later draw the epoch's final structure and animation frame.

```python
global_node_counter = len(global_G.nodes())
```

Updates the global node counter for the next epoch.

## Output 1: Individual Epoch PNG Files

```python
output_dir = "individual_plots"
os.makedirs(output_dir, exist_ok=True)
```

Creates the output directory if needed.

For each epoch snapshot:

```python
fig_ind, ax_ind = plt.subplots(figsize=(6, 6))
```

Creates one standalone figure.

```python
start_node = snap['node_offset']
end_node = snap['max_node_idx']
```

Finds the node-ID range belonging to this epoch.

```python
epoch_h = [...]
epoch_o = [...]
epoch_edges = [...]
```

Extracts H nodes, O nodes, and edges for this epoch only.

```python
nx.draw_networkx_edges(...)
nx.draw_networkx_nodes(...)
nx.draw_networkx_labels(...)
```

Draws the molecule graph.

```python
fig_ind.savefig(f"{output_dir}/epoch_{snap['epoch_num']}_final.png", bbox_inches='tight', dpi=150)
```

Saves the image.

## Output 2: Progressive GIF Animation

```python
fig, ax = plt.subplots(figsize=(12, 12))
```

Creates the main animation canvas.

```python
all_final_coords = np.array(list(global_pos.values()))
global_x_min, global_y_min = np.min(all_final_coords, axis=0) - padding_margin
global_x_max, global_y_max = np.max(all_final_coords, axis=0) + padding_margin
```

Computes global axis limits so the animation frame does not jump as more epochs appear.

### Nested Function: `update_canvas(frame_idx)`

This function draws one animation frame.

```python
ax.clear()
snapshot = epoch_snapshots[frame_idx]
max_visible_node = snapshot['max_node_idx']
```

Clears the plot and chooses the frame's visible graph state.

```python
h_nodes = [...]
o_nodes = [...]
```

Selects visible H and O atoms up to the current frame.

```python
visible_edges = [...]
```

Selects visible bonds up to the current frame.

```python
nx.draw_networkx_edges(...)
nx.draw_networkx_nodes(...)
nx.draw_networkx_labels(...)
```

Draws the visible part of the global graph.

```python
ax.set_xlim(global_x_min, global_x_max)
ax.set_ylim(global_y_min, global_y_max)
ax.axis('off')
```

Keeps the camera fixed and hides axes.

### GIF Saving

```python
ani = animation.FuncAnimation(fig, update_canvas, frames=len(epoch_snapshots), interval=1600, repeat=False)
```

Creates a Matplotlib animation with one frame per epoch.

```python
output_filename = "relaxed_8species_growth.gif"
ani.save(output_filename, writer='pillow', fps=0.62)
```

Saves the final GIF using Pillow.

## Important Data Flow

The main data flow is:

```text
Environment state
    -> PyTorch Geometric graph
    -> GNN action logits
    -> masked/noisy action choice
    -> environment update
    -> final species evaluation
    -> reward/loss
    -> optimizer update
    -> visualization output
```

## Key Implementation Notes

The script is a useful prototype, but several details are important for interpretation.

1. The GNN computes a termination logit, but structural actions are chosen by noisy argmax rather than sampling from the combined probability distribution.

2. `combined_logits` and `probs` are computed but not used for choosing actions.

3. `epoch_step_data` is stored but not used in the final animation.

4. `global_discovery_registry` is initialized but not used.

5. `node_to_species` uses shared sets to track molecular fragments. This works here, but it is not a standard union-find implementation.

6. The chemical validity check is formula-based only. It does not check detailed bond topology beyond valency constraints already applied during actions.

7. The layout is schematic. The atom positions are for visualization and are not molecular mechanics or quantum-chemistry geometries.

8. Double bonds are represented in the GNN graph by repeated edges. That is a simple encoding, but another common option would be to store bond order as an edge attribute.

## Most Important Functions to Read First

For understanding the script, read in this order:

1. `AdvancedMoleculeEnv.step`

This defines what actions actually do to the chemical graph.

2. `AdvancedMoleculeEnv.get_valid_action_masks`

This defines which actions are allowed or forbidden.

3. `GrowthGNN.forward`

This defines how the GNN scores actions.

4. The `while not env.terminated and steps < 8` loop

This connects the model to the environment and explains how actions are selected.

5. `AdvancedMoleculeEnv.evaluate_inventory`

This defines what counts as success or failure.
