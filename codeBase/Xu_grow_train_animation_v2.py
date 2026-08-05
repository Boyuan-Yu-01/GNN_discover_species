"""
Toy graph-growth experiment for discovering simple H/O molecular species.

The script treats atoms as graph nodes and bonds as graph edges. A graph neural
network scores possible graph-edit actions, the environment rejects chemically
invalid moves using simple valency rules, and the final molecular fragments are
rewarded if their formulas belong to a small valid-species set.

This is a learning/visualization prototype. The generated 2D positions are for
plotting only; they are not physical molecular geometries.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# Universe Configuration
# The toy chemistry universe is limited to hydrogen/oxygen species.
VALID_SPECIES = {"H2", "O2", "OH", "HO2", "H2O", "H2O2", "H", "O"}

# Atom classes used by the neural network and environment.
# H is encoded as 0 and O is encoded as 1.
ATOM_H = 0
ATOM_O = 1

# Simplified valency rule used to mask/reject invalid actions.
MAX_VALENCY = {ATOM_H: 1, ATOM_O: 2}

# ==========================================
# MOLECULE ENVIRONMENT (UNION-FIND TRACKING)
# ==========================================
class AdvancedMoleculeEnv:
    """
    Environment that stores and updates one molecule-growth episode.

    Each episode starts from H, H, O, O. The environment accepts graph-edit
    actions from the policy model, applies valency checks, updates the bond
    dictionary, and tracks which atoms belong to the same molecular fragment.

    node_offset lets each training epoch use unique global node IDs while this
    local environment still indexes atoms from 0 internally.
    """

    def __init__(self, node_offset=0):
        # Start every episode with two hydrogens and two oxygens.
        self.node_types = [ATOM_H, ATOM_H, ATOM_O, ATOM_O] 

        # current_bonds[i] counts how much valency atom i has already used.
        # A single bond adds 1 and a double bond adds 2.
        self.current_bonds = [0, 0, 0, 0]

        # Changed from list to dict: {(u, v): bond_order}
        # u and v are global node IDs. bond_order is usually 1 or 2.
        self.bonds = {}

        self.success = False
        self.terminated = False
        self.offset = node_offset

        # Track connected molecular fragments. Each atom initially belongs to
        # its own one-atom fragment; bonding merges these sets.
        self.node_to_species = {
            self.offset + i: {self.offset + i} for i in range(4)
        }

    def get_pyg_data(self):
        """Convert the current molecule graph into a PyTorch Geometric graph."""

        # Node features are one-hot atom identities:
        # H -> [1, 0], O -> [0, 1].
        x = [[1.0, 0.0] if t == ATOM_H else [0.0, 1.0] for t in self.node_types]
        x = torch.tensor(x, dtype=torch.float)

        # Mirror bonds based on order for GNN input
        # PyTorch Geometric expects edge_index in local node coordinates.
        # Each undirected chemical bond is represented as two directed edges.
        # Double bonds are represented by duplicated edge entries.
        edge_list = []
        for (u, v), order in self.bonds.items():
            for _ in range(order):
                edge_list.append([u - self.offset, v - self.offset])
                edge_list.append([v - self.offset, u - self.offset])
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous() if edge_list else torch.empty((2, 0), dtype=torch.long)
        return Data(x=x, edge_index=edge_index)

    def get_valid_action_masks(self):
        """
        Build masks that mark currently legal growth and connection actions.

        The policy network adds a large negative number to invalid logits, so
        masked actions are almost impossible to select. The final safety check
        still happens in step(), because masks can be imperfect.
        """

        num_nodes = len(self.node_types)

        # Growth is allowed from atoms that still have free valency.
        node_grow_mask = [1.0 if self.current_bonds[i] < MAX_VALENCY[self.node_types[i]] else 0.0 for i in range(num_nodes)]
        
        # Output shape: (num_nodes, num_nodes, 2) -> (pair_i_j, [single, double])
        edge_connect_mask = torch.zeros((num_nodes, num_nodes, 2), dtype=torch.float)
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i == j: continue # 1. Prevent self-loops

                # 2. Prevent exceeding valency
                if (self.current_bonds[i] < MAX_VALENCY[self.node_types[i]] and 
                    self.current_bonds[j] < MAX_VALENCY[self.node_types[j]]):
                    edge_connect_mask[i, j] = 1.0

                # Use global IDs for the bond dictionary, but local IDs for
                # valency arrays.
                u_global, v_global = i + self.offset, j + self.offset
                pair = tuple(sorted((u_global, v_global)))
                order = self.bonds.get(pair, 0)
                
                # Single bond valid if no bond exists and valency allows
                if order == 0 and self.current_bonds[i] < MAX_VALENCY[self.node_types[i]] and self.current_bonds[j] < MAX_VALENCY[self.node_types[j]]:
                    edge_connect_mask[i, j, 0] = 1.0
                # Double bond valid if order < 2 and valency allows
                if order < 2 and (self.current_bonds[i] + (1 if order==0 else 0) <= MAX_VALENCY[self.node_types[i]]) and \
                                 (self.current_bonds[j] + (1 if order==0 else 0) <= MAX_VALENCY[self.node_types[j]]):
                    edge_connect_mask[i, j, 1] = 1.0
        return torch.tensor(node_grow_mask, dtype=torch.float), edge_connect_mask

    def step(self, action_class, u_global, v_global=None, new_atom_type=None, bond_order_inc=0):
        """
        Apply one graph-edit action.

        action_class:
          0 -> grow a new atom from u_global
          1 -> add/increase a bond between u_global and v_global
          2 -> terminate the episode

        Returns:
          (success, terminated)
        """

        # Termination is treated as a successful action choice here. The final
        # molecule inventory is evaluated separately after the episode ends.
        if action_class == 2:
            self.terminated = True
            self.success = True
            return self.success, self.terminated

        u_local = u_global - self.offset

        if action_class == 0:
            # Add a new atom and connect it to the selected parent atom with a
            # single bond.
            new_global_idx = len(self.node_types) + self.offset
            self.node_types.append(new_atom_type)
            self.current_bonds.append(1)
            self.current_bonds[u_local] += 1
            self.bonds[(u_global, new_global_idx)] = 1

            # ADD LOG HERE
            print(f"DEBUG: Growth action added atom {new_global_idx} and bond to {u_global}")
            
            # The new atom belongs to the same molecular fragment as the atom
            # it was grown from.
            u_species_set = self.node_to_species[u_global]
            u_species_set.add(new_global_idx)
            self.node_to_species[new_global_idx] = u_species_set
            return True, False # (Success=True, Terminated=False)

        elif action_class == 1:
            # Connecting two existing atoms may create a new bond or increase
            # an existing bond order, subject to valency limits.
            pair = tuple(sorted((u_global, v_global)))
            current_order = self.bonds.get(pair, 0)
            order_increment = bond_order_inc  # This is passed in as 1 or 2 depending on the action

            # STRICT CHECK
            if current_order + order_increment > 2:
                print("DEBUG: Rejected! Exceeds bond order limit of 2.")
                return False, False # (Success=False, Terminated=False)

            # Only proceed if we are actually adding capacity
            if order_increment > 0:
                # Calculate remaining capacity for both atoms
                u_local = u_global - self.offset
                v_local = v_global - self.offset

                u_remaining = MAX_VALENCY[self.node_types[u_local]] - self.current_bonds[u_local]
                v_remaining = MAX_VALENCY[self.node_types[v_local]] - self.current_bonds[v_local]
                
                # Check valency capacity
                if order_increment <= u_remaining and order_increment <= v_remaining:
                    # Update bond order state
                    self.bonds[pair] = current_order + order_increment
                    
                    # Update valency count: 
                    # cost is exactly the new units added to the bond order
                    self.current_bonds[u_local] += order_increment
                    self.current_bonds[v_local] += order_increment

                    # If this bond connects two separate molecular fragments,
                    # merge their species-tracking sets.
                    if self.node_to_species[u_global] is not self.node_to_species[v_global]:
                        merged_set = self.node_to_species[u_global].union(self.node_to_species[v_global])
                        for idx in merged_set: self.node_to_species[idx] = merged_set
                    
                    print(f"DEBUG: Successfully added {order_increment} bond(s) between {u_global} and {v_global}. "
                          f"Current order: {self.bonds[pair]}")
                    return True, False # (Success=True, Terminated=False)
                else:
                    print(f"DEBUG: Rejected bond! Insufficient valency.")
                    return False, False # (Success=False, Terminated=False)
                
        return False, self.terminated

    def evaluate_inventory(self):
        """
        Convert connected atom groups into formulas and check validity.

        This is a formula-level check. It verifies that each connected fragment
        belongs to VALID_SPECIES, but it does not perform full chemical
        graph-isomorphism or geometry validation.
        """

        # Collect unique connected components from node_to_species.
        unique_sets = []
        for s in self.node_to_species.values():
            if s not in unique_sets: unique_sets.append(s)

        formulas, invalid_formula, all_legal = [], [], True
        for group in unique_sets:
            # Count atoms in the connected component.
            num_H = sum(1 for idx in group if self.node_types[idx - self.offset] == ATOM_H)
            num_O = sum(1 for idx in group if self.node_types[idx - self.offset] == ATOM_O)
            #formula = f"{'H' if num_H==1 else ('H'+str(num_H) if num_H>1 else '')}{'O' if num_O==1 else ('O'+str(num_O) if num_O>1 else '')}"
            # Calculate H and O parts
            H_part = f"H{num_H}" if num_H > 1 else ("H" if num_H == 1 else "")
            O_part = f"O{num_O}" if num_O > 1 else ("O" if num_O == 1 else "")
    
            # Correct the ordering for specific radicals or molecules
            if num_H == 1 and num_O == 1:
                formula = "OH"
            else:
                # Default order for others
                formula = f"{H_part}{O_part}"

            # Print for debugging
            if formula in VALID_SPECIES: formulas.append(formula)
            else: 
                print(f"DEBUG: Invalid structure found: {formula} (H={num_H}, O={num_O})")
                invalid_formula.append(formula)
                all_legal = False
        return formulas, invalid_formula, all_legal

# ==========================================
# POLICY GNN MODEL
# ==========================================
class GrowthGNN(nn.Module):
    """
    Graph neural network policy that scores possible molecule-edit actions.

    It produces three outputs:
      1. grow_logits: which atom should grow H/O
      2. connect_logits: which atom pair should form single/double bonds
      3. term_logit: whether the episode should terminate
    """

    def __init__(self, hidden_dim=32):
        super().__init__()

        # Two graph convolution layers create node embeddings from atom type
        # features and current bond connectivity.
        self.conv1 = GCNConv(2, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        # For every node, score two growth choices: add H or add O.
        self.grow_head = nn.Linear(hidden_dim, 2) 

        # Output 2 values per pair (Single=0, Double=1)
        # For every ordered node pair, score single-bond and double-bond actions.
        self.connect_head = nn.Linear(hidden_dim * 2, 2)

        # A graph-level head predicts whether construction should stop.
        self.termination_layer = nn.Linear(hidden_dim, 1)
        
    def forward(self, data, grow_mask, edge_mask):
        """Run the policy network and apply validity masks to action logits."""

        # Message passing over the current molecule graph.
        h = F.leaky_relu(self.conv1(data.x, data.edge_index), 0.1)
        h = F.leaky_relu(self.conv2(h, data.edge_index), 0.1)

        # Invalid growth actions receive approximately -infinity logits.
        grow_logits = self.grow_head(h) + (grow_mask.unsqueeze(1) - 1.0) * 1e9
        
        # Build all ordered pairs of node embeddings.
        num_nodes = h.size(0)
        h_i = h.unsqueeze(1).expand(-1, num_nodes, -1)
        h_j = h.unsqueeze(0).expand(num_nodes, -1, -1)
        pair_features = torch.cat([h_i, h_j], dim=-1)

        # connect_logits shape: (N, N, 2)
        # Invalid connection actions also receive approximately -infinity logits.
        connect_logits = self.connect_head(pair_features) + (edge_mask - 1.0) * 1e9
        
        # Pool node embeddings into one graph embedding for stop/continue choice.
        term_logit = self.termination_layer(global_mean_pool(h, batch=None))
        return grow_logits, connect_logits, term_logit

# ==========================================
# 2D SPIRAL MAPPER 
# ==========================================
def get_spiral_coordinates(index, spacing=35.0):
    """
    Return a 2D position for an epoch cluster on a square spiral.

    This keeps many generated molecule systems visible in one global plot
    instead of placing every epoch at the same location.
    """

    if index == 0: return 0.0, 0.0
    x = y = 0
    dx = 0; dy = -1
    step_limit = max_steps = 1
    turns = 0
    for _ in range(index):
        x += dx; y += dy
        step_limit -= 1
        if step_limit == 0:
            dx, dy = -dy, dx
            turns += 1
            if turns % 2 == 0: max_steps += 1
            step_limit = max_steps
    return float(x * spacing), float(y * spacing)

# ==========================================
# MAIN TRAINING WORKSPACE
# ==========================================
if __name__ == "__main__":
    # Create the policy model and optimizer. The model is trained online:
    # every generated molecule system immediately contributes one loss update.
    gnn_model = GrowthGNN(hidden_dim=32)
    optimizer = torch.optim.Adam(gnn_model.parameters(), lr=0.01)
    
    # NetworkX stores all atoms/bonds from all epochs for final visualization.
    # global_pos stores fixed 2D drawing coordinates for those global nodes.
    global_G = nx.Graph()
    global_pos = {}
    epoch_snapshots = [] 
    
    # Currently initialized for future analysis, but not used later.
    global_discovery_registry = {sp: 0 for sp in VALID_SPECIES}
    
    # ADJUSTABLE: Change this to 20, 30, or 50 to scale layout automatically
    total_epochs = 50  

    # Ensures each epoch gets a unique block of global node IDs.
    global_node_counter = 0
    
    # Dynamic styling rules to support large epoch counts cleanly
    if total_epochs <= 10:
        o_size, h_size, font_sz, padding_margin = 800, 700, 6.0, 8.5
    elif total_epochs <= 25:
        o_size, h_size, font_sz, padding_margin = 450, 380, 4.5, 12.0
    else:
        o_size, h_size, font_sz, padding_margin = 250, 200, 3.5, 18.0
        
    print(f"Simulating {total_epochs} Training Runs with Dynamic Structural Layout Scaling...")
    
    for epoch in range(1, total_epochs + 1):
        # Each epoch is one independent graph-growth attempt.
        optimizer.zero_grad()
        env = AdvancedMoleculeEnv(node_offset=global_node_counter)
        
        # Place this epoch's final molecules at a unique location in the
        # composite plot.
        x_center, y_center = get_spiral_coordinates(epoch - 1, spacing=35.0)
        cluster_center = np.array([x_center, y_center])
        
        # Add the four starting atoms to the global visualization graph.
        epoch_initial_nodes = [
            (global_node_counter + 0, ATOM_H),
            (global_node_counter + 1, ATOM_H),
            (global_node_counter + 2, ATOM_O),
            (global_node_counter + 3, ATOM_O)
        ]
        
        for idx, a_type in epoch_initial_nodes:
            global_G.add_node(idx, element="H" if a_type == ATOM_H else "O")
            
        global_node_counter += 4
        
        steps = 0
        action_log_probs = [] # Track log probabilities of chosen structural steps
        epoch_step_data = [] # List to store graph state at each step
        
        while not env.terminated and steps < 8:
            # Convert the current environment state into GNN inputs.
            pyg_data = env.get_pyg_data()
            grow_mask, edge_mask = env.get_valid_action_masks()
            
            # The model scores all legal/illegal actions; masks suppress the
            # illegal ones inside the logits.
            grow_logits, connect_logits, term_logit = gnn_model(pyg_data, grow_mask, edge_mask)
            
            # Combine structural options into a single distribution for action sampling
            flat_grow = grow_logits.flatten()
            flat_conn = connect_logits.flatten()
            combined_logits = torch.cat([flat_grow, flat_conn])
            
            # Apply a softmax categorical distribution over valid structural moves
            # Note: probs is computed for inspection but the current policy
            # chooses actions by noisy argmax below.
            probs = F.softmax(combined_logits, dim=-1)
            
            # (To retain exploration while updating, we sample or take argmax)
            # Random noise prevents the same highest-logit action from being
            # chosen every time early in training.
            noise_grow = torch.randn_like(grow_logits) * 0.4
            noise_conn = torch.randn_like(connect_logits) * 0.4
            max_grow_val = torch.max(grow_logits + noise_grow).item()
            max_conn_val = torch.max(connect_logits + noise_conn).item()
            
            # Stop if the termination head is confident or no legal structural
            # actions remain.
            prob_stop = torch.sigmoid(term_logit).item()
            if prob_stop > 0.90 or (sum(grow_mask) == 0 and torch.sum(edge_mask) == 0):
                env.step(action_class=2, u_global=env.offset)
                break

            # Determines which type of structural action the model takes at each time step.
            # Grow a new node or connect between existing nodes based on the highest logit value, 
            # with added noise for exploration.    
            if max_grow_val >= max_conn_val:
                flat_idx = torch.argmax(grow_logits + noise_grow).item()
                # Track the log probability of choosing this growth path
                action_log_probs.append(torch.log(F.softmax(grow_logits.flatten(), dim=-1)[flat_idx] + 1e-8))
                
                # Decode the node and atom type from the flat index
                # Use integer division (//) to find the node index (every node has 2 possible atom types)
                # Use modulo (%) to determine the atom type (0 for H, 1 for O)
                u_local = flat_idx // 2
                chosen_atom = flat_idx % 2

                # Convert local index to global index for environment consistency
                u_global = u_local + env.offset
                new_global_idx = len(env.node_types) + env.offset
                
                # Update graph and environment state
                global_G.add_node(new_global_idx, element="H" if chosen_atom == 0 else "O")
                env.step(action_class=0, u_global=u_global, new_atom_type=chosen_atom)
                global_G.add_edge(u_global, new_global_idx)
            else:
                flat_idx = torch.argmax(connect_logits + noise_conn).item()
                # Track the log probability of choosing this bonding connection
                action_log_probs.append(torch.log(F.softmax(connect_logits.flatten(), dim=-1)[flat_idx] + 1e-8))
                
                # Decode the flat index back into two distinct node indices (u, v)
                # The matrix is (num_nodes, num_nodes), so use floor division and modulo
                num_nodes = connect_logits.size(0)
                u_local = (flat_idx // 2) // num_nodes
                v_local = (flat_idx // 2) % num_nodes 
                bond_type = flat_idx % 2 # 0 -> Single, 1 -> Double
                u_global = u_local + env.offset
                v_global = v_local + env.offset
                
                # Update the environment and visual graph state
                # This checks valency and updates the edge list
                bond_type = flat_idx % 2 # 0 -> Single, 1 -> Double
                bond_order = 1 + bond_type # Map 0->1, 1->2
                success, terminated =env.step(action_class=1, u_global=u_global, v_global=v_global, bond_order_inc=bond_order)

                # Update the NetworkX graph so the visualization reflects the new bond
                if success:
                    global_G.add_edge(u_global, v_global)
                # if terminated:
                #     break
                if bond_type == 1:
                    print(f"DEBUG: Was attempting double bond between {u_global} and {v_global}")
                else:
                    print(f"DEBUG: Was attempting single bond between {u_global} and {v_global}")
                
            current_state = {
                'step': steps,
                'edges': list(global_G.edges()),
                'nodes': list(global_G.nodes(data=True))
            }
            epoch_step_data.append(current_state)
            steps += 1
            
        final_pyg = env.get_pyg_data()
        g_mask, e_mask = env.get_valid_action_masks()
        _, _, final_term_logit = gnn_model(final_pyg, g_mask, e_mask)
        
        formulas_list, invalid_formulas_list, success = env.evaluate_inventory()
        
        # Calculate Reward (Success vs Failure)
        # This reward is intentionally simple: the whole final inventory is
        # either valid or invalid.
        reward = 1.0 if success else -1.0 # Small penalty for failing, big reward for success
        # Even better: Add a bonus for each valid bond created
        #reward += (number_of_valid_bonds * 0.2)
        
        # Termination Loss
        # Teach the stop head to predict whether the final structure was valid.
        target_term = torch.tensor([[1.0 if success else 0.0]], dtype=torch.float)
        term_loss = F.binary_cross_entropy_with_logits(final_term_logit, target_term)
        
        # Policy Loss: Penalize choices that lead to chemical errors, reward valid configurations
        # For positive reward, the selected actions are reinforced.
        # For negative reward, the selected actions are discouraged.
        policy_loss = 0
        if len(action_log_probs) > 0:
            policy_loss = -torch.stack(action_log_probs).mean() * reward
            
        # Total combined loss optimization
        # The 0.5 factor keeps the policy-gradient-like term from dominating
        # the termination classification loss.
        total_loss = term_loss + 0.5 * policy_loss
        total_loss.backward()
        optimizer.step()
        
        # ADDED: Clear loss formatting printed to console log
        mixture_str = " + ".join(formulas_list)
        if invalid_formulas_list:
            mixture_str += " | Invalid: " + ", ".join(invalid_formulas_list)
        print(f"Epoch {epoch:02d}/{total_epochs} | System Output: {mixture_str:<22} | Training Loss: {total_loss.item():.5f}")
        
        # ====================================================
        # COMPACT GEOMETRY POSITION GENERATOR
        # ====================================================
        # The generated positions are schematic drawing positions. They are
        # designed for readable plots, not physical molecular geometry.
        unique_molecular_groups = []
        for s in env.node_to_species.values():
            if s not in unique_molecular_groups:
                unique_molecular_groups.append(list(s))
                
        num_molecules = len(unique_molecular_groups)
        
        for mol_idx, atom_group in enumerate(unique_molecular_groups):
            num_atoms_in_mol = len(atom_group)
            
            if num_molecules > 1:
                mol_angle = (2 * np.pi * mol_idx) / num_molecules
                fragment_separation = 6.5 + (num_atoms_in_mol * 0.5)
                mol_center = cluster_center + np.array([fragment_separation * np.cos(mol_angle), 
                                                        fragment_separation * np.sin(mol_angle)])
            else:
                mol_center = cluster_center
                
            for atom_idx, node_global_idx in enumerate(atom_group):
                if num_atoms_in_mol > 1:
                    atom_angle = (2 * np.pi * atom_idx) / num_atoms_in_mol
                    dynamic_bond_radius = 3.2 + (max(0, num_atoms_in_mol - 3) * 0.9)
                    global_pos[node_global_idx] = mol_center + np.array([dynamic_bond_radius * np.cos(atom_angle), 
                                                                         dynamic_bond_radius * np.sin(atom_angle)])
                else:
                    global_pos[node_global_idx] = mol_center

        epoch_snapshots.append({
            'epoch_num': epoch,
            'node_offset': env.offset,
            'max_node_idx': len(global_G.nodes()),
            'edges': list(global_G.edges()),
            'text': f"Epoch {epoch}: {mixture_str}"
        })
        global_node_counter = len(global_G.nodes())

    # ========================================================
    # OUTPUT 1: STANDALONE INDIVIDUAL PLOTS
    # ========================================================
    # Save one PNG per epoch, showing only that epoch's final molecule system.
    print("\nSaving standalone image files for each epoch layout...")
    output_dir = "individual_plots"
    os.makedirs(output_dir, exist_ok=True)
    
    for snap in epoch_snapshots:
        fig_ind, ax_ind = plt.subplots(figsize=(6, 6))
        ax_ind.set_facecolor('#fcfcfc')
        ax_ind.set_title(snap['text'], fontsize=11, fontweight='bold', color='#1e272c', pad=10)
        
        start_node = snap['node_offset']
        end_node = snap['max_node_idx']
        
        epoch_h = [n for n in range(start_node, end_node) if global_G.nodes[n]['element'] == "H"]
        epoch_o = [n for n in range(start_node, end_node) if global_G.nodes[n]['element'] == "O"]
        epoch_edges = [(u, v) for u, v in snap['edges'] if start_node <= u < end_node and start_node <= v < end_node]
        
        if epoch_edges:
            nx.draw_networkx_edges(global_G, global_pos, edgelist=epoch_edges, width=1.5, edge_color="#7f8c8d", ax=ax_ind)
        if epoch_o:
            # Standalone plots keep structural high detail scaling
            nx.draw_networkx_nodes(global_G, global_pos, nodelist=epoch_o, node_color="#ff4d4d", 
                                   node_size=800, edgecolors="black", linewidths=0.8, ax=ax_ind)
        if epoch_h:
            nx.draw_networkx_nodes(global_G, global_pos, nodelist=epoch_h, node_color="#a6c8ff", 
                                   node_size=700, edgecolors="black", linewidths=0.8, ax=ax_ind)
            
        # Index-only formatting
        epoch_nodes = epoch_h + epoch_o
        clean_labels = {n: f"({n})" for n in epoch_nodes}
        nx.draw_networkx_labels(global_G, global_pos, labels=clean_labels, font_size=6.5, 
                                font_weight="bold", font_family="sans-serif", ax=ax_ind)
        
        coords = np.array([global_pos[n] for n in epoch_nodes])
        x_min, y_min = np.min(coords, axis=0) - 6.0
        x_max, y_max = np.max(coords, axis=0) + 6.0
        
        ax_ind.set_xlim(x_min, x_max)
        ax_ind.set_ylim(y_min, y_max)
        ax_ind.axis('off')
        
        fig_ind.savefig(f"{output_dir}/epoch_{snap['epoch_num']}_final.png", bbox_inches='tight', dpi=150)
        plt.close(fig_ind)

    # ========================================================
    # OUTPUT 2: PROGRESSIVE MESH CANVAS (GIF GENERATION)
    # ========================================================
    # Build one global animation where more epoch clusters appear over time.
    print("\nCompiling full-grid composite plot map...")
    fig, ax = plt.subplots(figsize=(12, 12))
    
    all_final_coords = np.array(list(global_pos.values()))
    global_x_min, global_y_min = np.min(all_final_coords, axis=0) - padding_margin
    global_x_max, global_y_max = np.max(all_final_coords, axis=0) + padding_margin

    def update_canvas(frame_idx):
        """Draw one frame of the final GIF animation."""

        ax.clear()
        snapshot = epoch_snapshots[frame_idx]
        max_visible_node = snapshot['max_node_idx']
        
        ax.set_title(snapshot['text'], fontsize=14, fontweight='bold', color='#1e272c', pad=15)
        ax.set_facecolor('#fcfcfc')
        
        h_nodes = [n for n, attr in global_G.nodes(data=True) if attr['element'] == "H" and n < max_visible_node]
        o_nodes = [n for n, attr in global_G.nodes(data=True) if attr['element'] == "O" and n < max_visible_node]
        
        visible_edges = [(u, v) for u, v in snapshot['edges'] if u < max_visible_node and v < max_visible_node]
        if visible_edges:
            nx.draw_networkx_edges(global_G, global_pos, edgelist=visible_edges, width=1.2, edge_color="#7f8c8d", ax=ax)
            
        if o_nodes:
            nx.draw_networkx_nodes(global_G, global_pos, nodelist=o_nodes, node_color="#ff4d4d", 
                                   node_size=o_size, edgecolors="black", linewidths=0.8, ax=ax)
        if h_nodes:
            nx.draw_networkx_nodes(global_G, global_pos, nodelist=h_nodes, node_color="#a6c8ff", 
                                   node_size=h_size, edgecolors="black", linewidths=0.8, ax=ax)
            
        visible_nodes = h_nodes + o_nodes
        global_clean_labels = {n: f"({n})" for n in visible_nodes}
        nx.draw_networkx_labels(global_G, global_pos, labels=global_clean_labels, font_size=font_sz, 
                                font_weight="bold", font_family="sans-serif", ax=ax)
        
        ax.set_xlim(global_x_min, global_x_max)
        ax.set_ylim(global_y_min, global_y_max)
        ax.axis('off')

    ani = animation.FuncAnimation(fig, update_canvas, frames=len(epoch_snapshots), interval=1600, repeat=False)
    
    output_filename = "relaxed_8species_growth.gif"
    print(f"Saving single-plot grid animation sequence: {output_filename} ...")
    ani.save(output_filename, writer='pillow', fps=0.62)
    print("Everything processed successfully!")
    plt.close()
