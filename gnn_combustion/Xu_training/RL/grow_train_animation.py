import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx   # used for graph bookkeeping and visualization
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
import warnings

from asyncio import graph

warnings.filterwarnings("ignore", category=UserWarning)

# Universe Configuration
VALID_SPECIES = {"H2", "O2", "OH", "HO2", "H2O", "H2O2", "H", "O"}
ATOM_H = 0
ATOM_O = 1
MAX_VALENCY = {ATOM_H: 1, ATOM_O: 2}

# ==========================================
# MOLECULE ENVIRONMENT (UNION-FIND TRACKING)
# ==========================================
class AdvancedMoleculeEnv:
    def __init__(self, node_offset=0):
        self.node_types = [ATOM_H, ATOM_H, ATOM_O, ATOM_O] 
        self.current_bonds = [0, 0, 0, 0]
        self.edges = []
        self.terminated = False
        # Offset allows multiple epochs to share the same graph space without ID collisions.
        self.offset = node_offset
        # Uses Union-Find logic to track which atoms belong to the same disconnected molecule.
        self.node_to_species = {
            self.offset + 0: {self.offset + 0}, 
            self.offset + 1: {self.offset + 1}, 
            self.offset + 2: {self.offset + 2}, 
            self.offset + 3: {self.offset + 3}
        }

    def get_pyg_data(self): # encoding edge & nodal information for GNN consumption
        # Transforms the internal list-based state into a PyTorch Geometric 'Data' object.
        # This allows the GNN to consume the graph structure (nodes + edges) directly.
        x = [[1.0, 0.0] if t == ATOM_H else [0.0, 1.0] for t in self.node_types]    # one-hot atom features
        x = torch.tensor(x, dtype=torch.float)
        # Constructs the edge_index needed for Graph Convolutional message passing.
        # It mirrors edges to ensure the adjacency matrix is undirected.
        if len(self.edges) > 0:
            edge_list = [[u - self.offset, v - self.offset] for u, v in self.edges] + \
                        [[v - self.offset, u - self.offset] for u, v in self.edges] # shape: 2E by 2
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous() # shape: 2 by 2E
            # '.t' changes the shape from (2E, 2) to (2, 2E) 
            # '.contiguous' ensures the tensor is stored in a contiguous block of memory for efficiency.
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        return Data(x=x, edge_index=edge_index)

    def get_valid_action_masks(self):   # hash boundary: hard coding outermost electrons
        num_nodes = len(self.node_types)

        # THE GATEKEEPER: Prevents the model from making chemically impossible moves.
        # It sets a mask to 0.0 if an atom has reached its MAX_VALENCY limit.
        node_grow_mask = [1.0 if self.current_bonds[i] < MAX_VALENCY[self.node_types[i]] else 0.0 for i in range(num_nodes)]
        
        # Validates potential connections (edges) by checking valency and existing bonds.
        # Only bonds between atoms with spare capacity and that aren't already connected are allowed.
        edge_connect_mask = torch.zeros((num_nodes, num_nodes), dtype=torch.float)
        # refactor local to global indices (node level)
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i == j: continue
                u_global, v_global = i + self.offset, j + self.offset
                already_bonded = (u_global, v_global) in self.edges or (v_global, u_global) in self.edges
                if (self.current_bonds[i] < MAX_VALENCY[self.node_types[i]] and 
                    self.current_bonds[j] < MAX_VALENCY[self.node_types[j]] and not already_bonded):
                    edge_connect_mask[i, j] = 1.0
        return torch.tensor(node_grow_mask, dtype=torch.float), edge_connect_mask

    def step(self, action_class, u_global, v_global=None, new_atom_type=None):
        # Executes the agent's chosen action and updates the environment state.
        # action_class 0: Growth (adds a new atom to the graph).
        # action_class 1: Connection (bonds two existing atoms).
        # action_class 2: Termination (ends the episode).
        if action_class == 2:
            self.terminated = True
            return self.terminated

        u_local = u_global - self.offset

        if action_class == 0:   # growth
            # environment adds a new atom and immediately bonds it to node "u_global"
            new_global_idx = len(self.node_types) + self.offset
            self.node_types.append(new_atom_type)
            self.current_bonds.append(1)
            self.current_bonds[u_local] += 1
            self.edges.append((u_global, new_global_idx))
            
            u_species_set = self.node_to_species[u_global]
            u_species_set.add(new_global_idx)
            self.node_to_species[new_global_idx] = u_species_set

        elif action_class == 1:
            v_local = v_global - self.offset
            self.edges.append((u_global, v_global))
            self.current_bonds[u_local] += 1
            self.current_bonds[v_local] += 1
            
            if self.node_to_species[u_global] is not self.node_to_species[v_global]:
                merged_set = self.node_to_species[u_global].union(self.node_to_species[v_global])
                for idx in merged_set:
                    self.node_to_species[idx] = merged_set
        
        grow_mask, edge_mask = self.get_valid_action_masks()
        # When connecting, it merges atom sets into a single species group.
        # If no further moves are possible (masks are empty), it auto-terminates the epoch.
        if sum(grow_mask) == 0 and torch.sum(edge_mask) == 0:
            self.terminated = True
        return self.terminated

    def evaluate_inventory(self):
        # THE JUDGE: Identifies the final chemical species built during the epoch.
        # It iterates through unique species groups and checks if they exist in VALID_SPECIES.
        # This is where the model earns its +1.0 or -1.0 reward.
        unique_sets = []
        for s in self.node_to_species.values():
            if s not in unique_sets: unique_sets.append(s)
                
        formulas = []
        all_legal = True
        for group in unique_sets:
            num_H = sum(1 for idx in group if self.node_types[idx - self.offset] == ATOM_H)
            num_O = sum(1 for idx in group if self.node_types[idx - self.offset] == ATOM_O)
            
            h_str = f"H{num_H}" if num_H > 1 else ("H" if num_H == 1 else "")
            o_str = f"O{num_O}" if num_O > 1 else ("O" if num_O == 1 else "")
            formula = f"{h_str}{o_str}" if num_H >= num_O else f"{o_str}{h_str}"
            
            if num_H == 2 and num_O == 1: formula = "H2O"
            if num_H == 1 and num_O == 2: formula = "HO2"
            if num_H == 2 and num_O == 2: formula = "H2O2"
            
            formulas.append(formula)
            if formula not in VALID_SPECIES: all_legal = False
                
        return formulas, all_legal

# ==========================================
# POLICY GNN MODEL WITH ENTROPY SHIFTING
# ==========================================
class GrowthGNN(nn.Module):
    def __init__(self, hidden_dim=32):
        super().__init__()
        # Two GCN layers allow the agent to consider local atomic neighbors 
        # (e.g., an O atom can "see" that it is bonded to an H).
        self.conv1 = GCNConv(2, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        # Grow head: Predicts whether to add an H or an O atom to a specific node.
        self.grow_head = nn.Linear(hidden_dim, 2) 

        # Connect head: Predicts whether to form a bond between any two nodes.
        self.connect_head = nn.Linear(hidden_dim * 2, 1)

        # Termination layer: Predicts the graph-level readiness to finish.
        self.termination_layer = nn.Linear(hidden_dim, 1)
        
    def forward(self, data, grow_mask, edge_mask):
        x, edge_index = data.x, data.edge_index
        num_nodes = x.size(0)

        # Message Passing: Updates node features based on local chemical structure.
        # h is the result of all atoms in the graph, each row of h represent a node's embedding
        h = F.leaky_relu(self.conv1(x, edge_index), 0.1) 
        h = F.leaky_relu(self.conv2(h, edge_index), 0.1)
        
        # Grow Action: Adds a large negative number (-1e9) to masked positions 
        # to ensure the agent never chooses an invalid chemical growth.
        grow_logits = self.grow_head(h) + (grow_mask.unsqueeze(1) - 1.0) * 1e9
        
        # Connect Action: Creates pair features by concatenating embeddings of 
        # all possible node combinations (i, j). 
        # It then applies the connect head to predict bond formation likelihood.
        h_i = h.unsqueeze(1).expand(-1, num_nodes, -1)
        h_j = h.unsqueeze(0).expand(num_nodes, -1, -1)
        pair_features = torch.cat([h_i, h_j], dim=-1)
        # Applies the same masking logic here to prevent invalid bond connections.
        connect_logits = self.connect_head(pair_features).squeeze(-1) + (edge_mask - 1.0) * 1e9
        
        # Global Pooling: Summarizes the entire molecule's state into a single 
        # latent vector to decide if the molecule is "complete" and ready to terminate.
        graph_latent = global_mean_pool(h, batch=None)
        term_logit = self.termination_layer(graph_latent)
        return grow_logits, connect_logits, term_logit

# ==========================================
# 2D SPIRAL MAPPER 
# ==========================================
def get_spiral_coordinates(index, spacing=35.0):
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
    gnn_model = GrowthGNN(hidden_dim=32)
    optimizer = torch.optim.Adam(gnn_model.parameters(), lr=0.01)
    
    global_G = nx.Graph()
    global_pos = {}
    epoch_snapshots = [] 
    
    global_discovery_registry = {sp: 0 for sp in VALID_SPECIES}
    
    # ADJUSTABLE: Change this to 20, 30, or 50 to scale layout automatically
    total_epochs = 50  
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
        # The environment is reset to four disconnected atoms:
        optimizer.zero_grad()
        env = AdvancedMoleculeEnv(node_offset=global_node_counter)
        
        x_center, y_center = get_spiral_coordinates(epoch - 1, spacing=35.0)    # for plotting
        cluster_center = np.array([x_center, y_center])
        
        # add the initial four atoms to the global visualization graph and reserve their ids:
        epoch_initial_nodes = [
            (global_node_counter + 0, ATOM_H),
            (global_node_counter + 1, ATOM_H),
            (global_node_counter + 2, ATOM_O),
            (global_node_counter + 3, ATOM_O)
        ]
        
        for idx, a_type in epoch_initial_nodes:
            global_G.add_node(idx, element="H" if a_type == ATOM_H else "O")
            
        global_node_counter += 4

       # Initialize training-specific variables for this epoch 
        steps = 0
        action_log_probs = [] # Track log probabilities of chosen structural steps
        epoch_step_data = [] # List to store graph state at each step for visualization
        
        while not env.terminated and steps < 8:
            pyg_data = env.get_pyg_data()
            grow_mask, edge_mask = env.get_valid_action_masks()
            
            grow_logits, connect_logits, term_logit = gnn_model(pyg_data, grow_mask, edge_mask)
            
            # Combine structural options into a single distribution for action sampling
            flat_grow = grow_logits.flatten()
            flat_conn = connect_logits.flatten()
            combined_logits = torch.cat([flat_grow, flat_conn])
            
            # Apply a softmax categorical distribution over valid structural moves
            probs = F.softmax(combined_logits, dim=-1)
            
            # (To retain exploration while updating, we sample or take argmax)
            noise_grow = torch.randn_like(grow_logits) * 0.4
            noise_conn = torch.randn_like(connect_logits) * 0.4
            max_grow_val = torch.max(grow_logits + noise_grow).item()
            max_conn_val = torch.max(connect_logits + noise_conn).item()
            
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
                u_global = (flat_idx // num_nodes) + env.offset
                v_global = (flat_idx % num_nodes) + env.offset
                
                # Update the environment and visual graph state
                # This checks valency and updates the edge list
                env.step(action_class=1, u_global=u_global, v_global=v_global)

                # Update the NetworkX graph so the visualization reflects the new bond
                global_G.add_edge(u_global, v_global)
                
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
        
        formulas_list, success = env.evaluate_inventory()
        
        # Calculate Reward (Success vs Failure)
        reward = 1.0 if success else -1.0
        
        # Termination Loss
        target_term = torch.tensor([[1.0 if success else 0.0]], dtype=torch.float)
        term_loss = F.binary_cross_entropy_with_logits(final_term_logit, target_term)
        
        # Policy Loss: Penalize choices that lead to chemical errors, reward valid configurations
        policy_loss = 0
        if len(action_log_probs) > 0:
            policy_loss = -torch.stack(action_log_probs).mean() * reward
            
        # Total combined loss optimization
        total_loss = term_loss + 0.5 * policy_loss
        total_loss.backward()
        optimizer.step()
        
        # ADDED: Clear loss formatting printed to console log
        mixture_str = " + ".join(formulas_list)
        print(f"Epoch {epoch:02d}/{total_epochs} | System Output: {mixture_str:<22} | Training Loss: {total_loss.item():.5f}")
        
        # ====================================================
        # COMPACT GEOMETRY POSITION GENERATOR
        # ====================================================
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
    print("\nCompiling full-grid composite plot map...")
    fig, ax = plt.subplots(figsize=(12, 12))
    
    all_final_coords = np.array(list(global_pos.values()))
    global_x_min, global_y_min = np.min(all_final_coords, axis=0) - padding_margin
    global_x_max, global_y_max = np.max(all_final_coords, axis=0) + padding_margin

    def update_canvas(frame_idx):
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