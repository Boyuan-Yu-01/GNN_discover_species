import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
#from torch_geometric.utils import to_networkx
#import networkx as nx
#import matplotlib.pyplot as plt

class CombustionConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        # aggr='add' means we sum up the messages from all neighboring atoms
        super().__init__(aggr='add') 
        # This turns 'in' (1) into 'out' (16)
        self.lin = nn.Linear(in_channels, out_channels)
        
        # A small network to process distances (edge_attr)
        # It takes 1 value (distance) and turns it into a scaling factor
        self.distance_mlp = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, in_channels)
        )

    def forward(self, x, edge_index, edge_attr):
        # 1. Start message passing
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        # 2. Transform the result to the 16-dimensional space
        out = self.lin(out) # lin is a learnable layer that transforms the propagated features
        return out

    def message(self, x_j, edge_attr):
        # x_j is the feature of the neighbor atom
        # We multiply the neighbor's info by a weight based on distance
        weight = self.distance_mlp(edge_attr)
        return x_j * weight

    def update(self, aggr_out):
        # aggr_out is the sum of messages. We could add more layers here.
        return aggr_out
    
class CombustionGNN(torch.nn.Module):
    ## MessagePassing inherits from torch.nn.Module
    ## Every MessagePassing layer is an nn.Module
    def __init__(self):
        super().__init__()
        # Our message passing layer from before
        self.conv1 = CombustionConv(in_channels=1, out_channels=16)
        
        # Bond Order Predictor (Takes features of 2 nodes: 16+16 = 32)
        self.bond_predictor = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1) # Output: Bond Order
        )
        
        # Existence Predictor (Takes feature of 1 node: 16)
        self.existence_predictor = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid() # Output: Probability (0 to 1)
        )

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # 1. Update node features via Message Passing
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        
        # 2. Predict Existence for each node
        existence_prob = self.existence_predictor(x)
        
        # 3. Predict Bond Order for each edge
        # We get the features of the two atoms forming each edge
        row, col = edge_index
        edge_features = torch.cat([x[row], x[col]], dim=1)
        bond_order = self.bond_predictor(edge_features)
        
        return bond_order, existence_prob

# 1. Define 4 atoms: H, H, O, O (using 0 for H, 1 for O)
# Shape: [num_nodes, num_features]
x = torch.tensor([[0], [0], [1], [1]], dtype=torch.float)

# 2. Define connectivity (The edges)
# Let's say Node 0 is bonded to Node 2 (H-O) 
# and Node 1 is bonded to Node 3 (H-O)
edge_index = torch.tensor([
    [0, 2, 1, 3], # Sources
    [2, 0, 3, 1]  # Targets (making it undirected)
], dtype=torch.long)

# 3. Define the Bond Lengths (The "Edge Attributes")
# Suppose these are 0.97 Angstroms
edge_attr = torch.tensor([[0.97], [0.97], [0.97], [0.97]], dtype=torch.float)

# 4. Create the Data Object
data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

print("--- Combustion Graph Check ---")
print(f"Number of Atoms: {data.num_nodes}")
print(f"Number of Edges: {data.num_edges}")
print(f"Edge Attributes (Distances) shape: {data.edge_attr.shape}")
print("Graph is ready for training!")

# Message passing
# 1. Initialize our layer
# Input is 1 feature (atom type), Output is 1 feature
conv = CombustionConv(in_channels=1, out_channels=1)

# 2. Re-use your 'data' from the previous script
# x: [0, 0, 1, 1] (H, H, O, O)
# edge_attr: [0.97, 0.97, 0.97, 0.97]
new_x = conv(data.x, data.edge_index, data.edge_attr)

print("Original node features (Atom Types):\n", data.x)
print("\nNew node features after Message Passing:\n", new_x)

# run bond orders
model = CombustionGNN()
bond_orders, existence = model(data)

print("--- Model Outputs ---")
print(f"Predicted Bond Orders for each edge:\n{bond_orders}")
print(f"Predicted Existence Probability for each node:\n{existence}")

""" # 5. Convert to NetworkX
# we set to_undirected=True to avoid drawing double arrows
G = to_networkx(data, to_undirected=True)

# 6. Define colors and labels based on atom types
color_map = []
labels = {}
for i, atom_type in enumerate(data.x):
    if atom_type == 0: # Hydrogen
        color_map.append('skyblue')
        labels[i] = f"H{i}"
    else: # Oxygen
        color_map.append('salmon')
        labels[i] = f"O{i}"

# 7. Draw the graph
plt.figure(figsize=(6, 4))
pos = nx.spring_layout(G) # Calculates a visually pleasing layout

nx.draw(G, pos, 
        with_labels=True, 
        labels=labels, 
        node_color=color_map, 
        node_size=1000, 
        edge_color='gray',
        font_weight='bold')

# Draw edge labels (the bond lengths/distances)
edge_labels = {(u, v): f"{data.edge_attr[i].item():.2f}Å" for i, (u, v) in enumerate(G.edges())}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

plt.title("4-Node Combustion Species (H-O Pairs)")
plt.show() """