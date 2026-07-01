import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
import numpy as np
import os

# Create a directory for outputs if it doesn't exist
output_dir = "plots"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# ==========================================
# 1. ARCHITECTURE (Mirroring ex_bo_4node.py)
# ==========================================

class CombustionConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='mean') 
        self.lin = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index, edge_attr):
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return self.lin(out)

    def message(self, x_j, edge_attr):
        weight = torch.exp(-edge_attr * 2.0) 
        return x_j * weight

class CombustionGNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = CombustionConv(in_channels=2, out_channels=64)
        self.bond_predictor = nn.Sequential(
            nn.Linear(193, 64), nn.LeakyReLU(0.1),
            nn.Linear(64, 32), nn.LeakyReLU(0.1),
            nn.Linear(32, 1), nn.Softplus() 
        )
        self.existence_predictor = nn.Sequential(
            nn.Linear(64, 32), nn.LeakyReLU(0.1),
            nn.Linear(32, 1), nn.Sigmoid() 
        )

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        h = F.leaky_relu(self.conv1(x, edge_index, edge_attr), 0.1)
        row, col = edge_index
        h_u, h_v = h[row], h[col]
        # Commutative operators
        edge_features = torch.cat([h_u+h_v, h_u*h_v, torch.abs(h_u-h_v), edge_attr.view(-1, 1)], dim=1)
        return self.bond_predictor(edge_features), self.existence_predictor(h)

# ==========================================
# 2. DATA BUILDING (Fix: Use distances directly from coordinates)
# ==========================================

def build_4node_data_from_pos(x_vals, pos_config):
    """
    Builds PyG Data object. Edge attributes (distances) are calculated 
    DIRECTLY from the provided coordinates.
    """
    num_nodes = len(x_vals)
    # Fixed edge_index for 4-node complete graph (bi-directional)
    edge_index = torch.tensor([[0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3],
                               [1, 0, 2, 0, 3, 0, 2, 1, 3, 1, 3, 2]], dtype=torch.long)
    
    edge_attr = []
    for i in range(edge_index.shape[1]):
        u, v = edge_index[0, i].item(), edge_index[1, i].item()
        p1, p2 = np.array(pos_config[u]), np.array(pos_config[v])
        # Direct Euclidean distance from specified coordinates
        dist = np.linalg.norm(p1 - p2)
        edge_attr.append([dist])
        
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    x_vectors = [[1.0, 0.0] if val == 0 else [0.0, 1.0] for val in x_vals]
    return Data(x=torch.tensor(x_vectors, dtype=torch.float), edge_index=edge_index, edge_attr=edge_attr)

# ==========================================
# 3. VISUALIZATION (Fix: Accepting and plotting all nodes)
# ==========================================

def plot_molecule_direct(name, x_vals, pe, pb, edge_index, pos_config):
    num_nodes = len(x_vals)
    G = nx.Graph()
    atom_labels = {i: ('H' if x_vals[i] == 0 else 'O') for i in range(num_nodes)}
    
    # 1. DEFINE CUSTOM COLORMAP: Yellow -> Lime Green -> Light Blue
    colors = ["#fdfd96", "#32cd32", "#add8e6"] # Soft yellow, Lime, Light Blue
    custom_cmap = LinearSegmentedColormap.from_list("combustion_cmap", colors)

    # Add nodes with metadata
    for i in range(num_nodes):
        G.add_node(i, label=f"{atom_labels[i]}\n{pe[i].item():.2f}", score=pe[i].item())

    plt.figure(figsize=(8, 8)) 
    pos = {i: np.array(pos_config[i]) for i in range(num_nodes)}
    
    # 2. Add edges and labels
    edge_labels = {}
    valid_edges = []
    for i in range(edge_index.shape[1]):
        u, v = edge_index[0, i].item(), edge_index[1, i].item()
        if u < v:
            bo = pb[i].item()
            if bo > 0.05:
                G.add_edge(u, v, weight=bo)
                valid_edges.append((u, v))
                edge_labels[(u, v)] = f"{bo:.2f}"

    # 3. DRAW NODES (Fixed size: 1000 for better spacing)
    node_scores = [G.nodes[n]['score'] for n in G.nodes]
    nodes = nx.draw_networkx_nodes(
        G, pos, 
        node_color=node_scores, 
        cmap=custom_cmap, 
        node_size=1200,      # Smaller node size to un-squeeze labels
        edgecolors='black',
        vmin=0.0, vmax=1.0   # Fix color scale for existence probability
    )
    nx.draw_networkx_labels(G, pos, labels=nx.get_node_attributes(G, 'label'), font_size=11, font_weight='bold')

    # 4. DRAW EDGES (Proportional to Bond Order)
    # Extract weights from the graph in the order of valid_edges
    widths = [G[u][v]['weight'] * 5.0 for u, v in valid_edges]
    nx.draw_networkx_edges(G, pos, edgelist=valid_edges, width=widths, edge_color='orange', alpha=0.6)
    #nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)

    # 5. CUSTOM OFFSET EDGE LABELS
    ax = plt.gca()
    for (u, v), label in edge_labels.items():
        p1 = pos[u]
        p2 = pos[v]
        
        # Calculate midpoint
        midpoint = (p1 + p2) / 2
        
        # Calculate edge vector and its normal (perpendicular) vector
        edge_vec = p2 - p1
        edge_len = np.linalg.norm(edge_vec)
        if edge_len == 0: continue
        
        # Normal vector (rotate 90 degrees)
        normal = np.array([-edge_vec[1], edge_vec[0]]) / edge_len
        offset_distance = 0.25

        # --- MANUAL ADJUSTMENT FOR H2O + O ---
        # If the molecule is H2O + O and the label is pointing "inward", flip it.
        if name == 'H2O + O':
            # Check if the normal vector is pointing towards the center (x=0)
            # If the midpoint is on the left and normal points right, or vice versa, flip it.
            if (midpoint[0] < 0 and normal[0] > 0) or (midpoint[0] > 0 and normal[0] < 0):
                normal = -normal 
            offset_distance = 0.15 # Slightly more room for the H2O labels
        
        # Determine displacement: offset by ~0.2 units away from the edge
        # Adjust 0.25 to increase/decrease the distance from the edge
        offset_pos = midpoint + normal * offset_distance
        
        plt.text(
            offset_pos[0], offset_pos[1],
            label,
            fontsize=12,
            fontweight='bold',
            color='darkred',
            ha='center',
            va='center',
            bbox=dict(facecolor='white', alpha=0, edgecolor='none') # No background box
        )

    # 5. SCALE AND AXES
    plt.title(f"GNN Prediction: {name}", fontsize=15, fontweight='bold')
    # Colorbar is now strictly 0.0 to 1.0
    cbar = plt.colorbar(nodes, shrink=0.8, ticks=[0, 0.25, 0.5, 0.75, 1.0])
    cbar.ax.tick_params(labelsize=15)
    cbar.set_label('Node Probability within a Species', fontsize=15, fontweight='bold')

    plt.axis('equal') 
    plt.xlim(-3.2, 3.2) 
    plt.ylim(-3.2, 3.2)
    #plt.grid(True, linestyle='--', alpha=0.3)
    #plt.show()

    safe_name = name.replace(" ", "_").replace("+", "plus").replace("(", "").replace(")", "")
    filename_svg = os.path.join(output_dir, f"{safe_name}.svg")
    filename_png = os.path.join(output_dir, f"{safe_name}.png")
    plt.savefig(filename_svg, dpi=300, bbox_inches='tight')
    plt.savefig(filename_png, dpi=300, bbox_inches='tight')
    plt.close() # Free up memory
    print(f"Saved figure to {filename_svg} and {filename_png}")

# ==========================================
# 4. EXECUTION
# ==========================================

# 1. Load Model
model = CombustionGNN()
try:
    model.load_state_dict(torch.load("ex_bo_4node.pth"))
    model.eval()
    print("Model loaded successfully.")
except FileNotFoundError:
    print("Error: 'ex_bo_4node.pth' not found. Please run training first.")
    exit()

# 2. Define Test Cases with DIRECT 2D Coordinates (Units: Å)
# Format: (Name, NodeFeatures[0/H, 1/O], Coordinates[(x0, y0), (x1, y1), ...])
test_cases = [
    (
        'H2 + O2', 
        [0, 0, 1, 1], 
        [(-1.5, 0.37), (-1.5, -0.37), (1.5, 0.605), (1.5, -0.605)] # H-H dist 0.74, O-O dist 1.21
    ),
    (
        'H2O + O', 
        [0, 0, 1, 1], 
        [(-0.755, -1.0), (0.755, -1.0), (0.0, -0.4), (0.0, 2.2)] # O-H dist 0.96, H-H 1.51, Lone O 3.5 away
    ),
    (
        'OH + OH', 
        [0, 1, 0, 1], 
        [(-1.5, 0.485), (-1.5, -0.485), (1.5, 0.485), (1.5, -0.485)] # Two OH pairs, 0.97 bond each
    ),
    (
        'H2O2', 
        [0, 1, 1, 0], 
        [(-0.79527, -0.98584), (-0.73, 0.0), (0.73, 0.0), (0.79527, 0.98584)] # H-O-O-H chain with 1.46 O-O bond
    ),
    (
        'HO2 + H', 
        [0, 1, 1, 0], 
        [(-1.2, 0.5), (-0.4, 0.0), (0.93, 0.0), (0.0, -2.5)] # H-O (0.97), O-O (1.33), Lone H (2.5)
    ),
    (
        'H2 + O + O', 
        [0, 0, 1, 1], 
        [(-1.5, 0.37), (-1.5, -0.37), (1.5, 1.5), (1.5, -1.5)] # H2 pair (0.74) and two separated O radicals
    ),
    (
        'O2 + H + H', 
        [1, 1, 0, 0], 
        [(-1.5, 0.605), (-1.5, -0.605), (1.5, 1.5), (1.5, -1.5)] # O2 pair (1.21) and two separated H radicals
    ),
    (
        'OH + O + H', 
        [0, 1, 1, 0], 
        [(-1.5, 0.485), (-1.5, -0.485), (1.5, 0.0), (0.0, -2.5)] # One OH (0.97) and isolated O and H
    ),
    (
        'H + H + O + O', 
        [0, 0, 1, 1], 
        [(-2.0, 2.0), (2.0, 2.0), (-2.0, -2.0), (2.0, -2.0)] # All radicals widely separated
    ),
    (
        'H + H + O + O (closer 1)', 
        [0, 0, 1, 1], 
        [(-1.0, 1.0), (1.0, 1.0), (-1.0, -1.0), (1.0, -1.0)] # All radicals widely separated
    ),
    (
        'H + H + O + O (closer 2)', 
        [0, 0, 1, 1], 
        [(-0.8, 0.8), (0.8, 0.8), (-0.8, -0.8), (0.8, -0.8)] # All radicals widely separated
    ),
    (
        'H2 (stretched)', 
        [0, 0, 1, 1], 
        [(-1.5, 0.82), (-1.5, -0.82), (1.5, 1.5), (1.5, -1.5)] # H2 pair (0.74) and two separated O radicals
    ),
    (
        'O2 (stretched)', 
        [1, 1, 0, 0], 
        [(-1.5, 0.83), (-1.5, -0.83), (1.5, 1.5), (1.5, -1.5)] # O2 pair (1.21) and two separated H radicals
    ),
    (
        'OH (stretched)', 
        [0, 1, 1, 0], 
        [(-1.5, 0.83), (-1.5, -0.83), (1.5, 0.0), (0.0, -2.5)] # One OH (0.97) and isolated O and H
    ),
    (
        'H2O (stretched)', 
        [0, 0, 1, 1], 
        [(-0.755, -1.0), (1.650, -1.0), (0.0, -0.4), (0.0, 2.2)] # O-H dist 0.96, H-H 1.51, Lone O 3.5 away
    ),
    (
        'H2O (stretched 2)', 
        [0, 0, 1, 1], 
        [(-0.154, 1.800), (0.95, 0.0), (0.0, 0.0), (0.5, -2.8)] # H-H 2.1, O-H 0.93/1.95, Lone O 3.5 away
    ),
    (
        'HO2 (stretched)', 
        [0, 1, 1, 0], 
        [(-1.2, 0.5), (-0.4, 0.0), (1.35, 0.0), (0.0, -2.5)] # H-O (0.97), O-O (1.33), Lone H (3.5)
    ),
    (
        'H2O2 (stretched)', 
        [0, 1, 1, 0], 
        [(-0.89527, -0.98584), (-0.83, 0.0), (0.83, 0.0), (0.89527, 0.98584)]
    )
]

# 3. Run and Plot
with torch.no_grad():
    for name, x_vals, pos_config in test_cases:
        # Step 1: Calculate interatomic distances from these coordinates
        data = build_4node_data_from_pos(x_vals, pos_config)
        
        # Step 2: GNN Prediction
        pb, pe = model(data)
        
        # Step 3: Visual Validation
        print(f"\n--- Running: {name} ---")
        print(f"Nodes defined in configuration: {len(pos_config)}")
        print(f"Nodes in x_vals: {len(x_vals)}")
        print("Predicted existence probabilities:", [f"{p.item():.4f}" for p in pe.squeeze()])
        print("Predicted bond orders:", [f"{p.item():.4f}" for p in pb.squeeze()])
        
        # FIXED: Passing coordinates directly to the plot function
        plot_molecule_direct(name, x_vals, pe, pb, data.edge_index, pos_config)