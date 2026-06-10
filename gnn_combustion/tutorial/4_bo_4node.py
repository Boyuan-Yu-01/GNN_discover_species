import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing

# ==========================================
# 1. MODEL DEFINITION
# ==========================================

class CombustionConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        #super().__init__(aggr='add') 
        super().__init__(aggr='mean') 
        self.lin = nn.Linear(in_channels, out_channels)
        #self.distance_mlp = nn.Sequential(
        #    nn.Linear(1, 64),
        #    nn.LeakyReLU(0.1), # Changed from ReLU
        #    nn.Linear(64, in_channels) 
        #)

    def forward(self, x, edge_index, edge_attr):
        # Message passing uses the raw interatomic distance directly.
        # Larger distances should reduce the message weight.
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return self.lin(out)

    def message(self, x_j, edge_attr):
        # Decay message strength with distance.
        weight = torch.exp(-edge_attr * 2.0) # Adjust decay rate as needed
        return x_j * weight

class CombustionGNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Use 2 in_channels for your [H, O] one-hot encoding
        self.conv1 = CombustionConv(in_channels=2, out_channels=64)
        
        # 64 (sum) + 64 (prod) + 64 (diff) + 1 (dist) = 193 inputs
        # notice the reason for the above-mentioned 'engineered' features is that they are same for whichever i j order one may take. The sum, abs. diff., multiplication are commutative operations.
        self.bond_predictor = nn.Sequential(
            nn.Linear(193, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, 1),
            nn.Softplus() 
        )
        
        self.existence_predictor = nn.Sequential(
            nn.Linear(64, 32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, 1),
            nn.Sigmoid() 
        )

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        h = F.leaky_relu(self.conv1(x, edge_index, edge_attr), 0.1)
        
        row, col = edge_index
        h_u, h_v = h[row], h[col]
        
        # COMMUTATIVE OPERATORS: Force undirected logic
        sum_feat = h_u + h_v
        prod_feat = h_u * h_v
        diff_feat = torch.abs(h_u - h_v)
        dist = edge_attr.view(-1, 1)
        
        # Concatenate and predict
        edge_features = torch.cat([sum_feat, prod_feat, diff_feat, dist], dim=1)
        return self.bond_predictor(edge_features)

# ==========================================
# 2. DATASET GENERATION
# ==========================================

def create_mini_dataset(num_samples=1000):
    dataset = []
    # Equilibrium lengths (R0) and decay constant (b)
    #targets = {'HH': 0.74, 'OO': 1.21, 'OH': 0.97}
    targets = {'HH': 0.74, 'OO': 1.48, 'OH': 0.97}
    b = 0.357 # Pauling constant
    
    atom_features = {
        'H': [1.0, 0.0],
        'O': [0.0, 1.0]
    }

    sample_types = [
        ('H2 + O2', ['H', 'H', 'O', 'O'], [(0, 1), (2, 3)]),
        ('H2 + O + O', ['H', 'H', 'O', 'O'], [(0, 1)]),
        ('O2 + H + H', ['O', 'O', 'H', 'H'], [(0, 1)]),
        ('H2O + O', ['H', 'H', 'O', 'O'], [(0, 2), (1, 2)]),
        ('OH + OH', ['O', 'H', 'O', 'H'], [(0, 1), (2, 3)]),
        ('OH + O + H', ['O', 'H', 'O', 'H'], [(0, 1)]),
        ('H2O2', ['H', 'O', 'O', 'H'], [(0, 1), (1, 2), (2, 3)]),
        ('HO2 + H', ['H', 'O', 'O', 'H'], [(0, 1), (1, 2)]),
        ('H + H + O + O', ['H', 'H', 'O', 'O'], []) # All radical species
    ]

    for _ in range(num_samples):
        # 1. Pick a template and permute it
        _, base_atom_types, base_bonded_pairs = random.choice(sample_types) # pick up the 2nd and 3rd element in the tuple
        perm = torch.randperm(4).tolist()   # choose a random permutation of [0,1,2,3] and convert to python list
        
        # Map atoms and bond pairs to new positions
        atom_types = [base_atom_types[i] for i in perm] # permute the atom types
        index_map = {old: new for new, old in enumerate(perm)}
        bonded_pairs = [(index_map[u], index_map[v]) for u, v in base_bonded_pairs]

        # 2. Define fixed edge index and create symmetric distance map
        # edge_index = torch.tensor([
        #     [0,1, 1,0, 0,2, 2,0, 0,3, 3,0, 1,2, 2,1, 1,3, 3,1, 2,3, 3,2]
        # ], dtype=torch.long).view(2, -1)
        # Row 0 = Source (u), Row 1 = Target (v)
        edge_index = torch.tensor([
            [0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3], # u
            [1, 0, 2, 0, 3, 0, 2, 1, 3, 1, 3, 2]  # v
        ], dtype=torch.long)            # all possible pairs of edges (undirected)

        dist_map = {}
        for i in range(12):
            u, v = edge_index[0, i].item(), edge_index[1, i].item() # u_i, v_i
            pair_key = tuple(sorted((u, v))) # Ensures (0,1) and (1,0) use same key
            
            if pair_key not in dist_map:
                if (u, v) in bonded_pairs or (v, u) in bonded_pairs:
                    # Bonded distance: random variation around equilibrium
                    dist_map[pair_key] = random.uniform(0.7, 1.6)
                else:
                    # Non-bonded distance: significantly further away
                    dist_map[pair_key] = random.uniform(2.2, 3.5)

        # 3. Build edge_attr and y_bond using the symmetric distances
        x = torch.tensor([atom_features[t] for t in atom_types], dtype=torch.float) # This is a 4 x 2 tensor
        edge_attr = torch.zeros((12, 1))
        y_bond = torch.zeros((12, 1))
        
        ## This for loop is used to define the "ground truth" of the bond value (calculated using Pauling's formula) based on the assigned distance for each edge. the y_bond is the eventual "ground truth" for the bond order.
        for i in range(12):
            u, v = edge_index[0, i].item(), edge_index[1, i].item()
            dist = dist_map[tuple(sorted((u, v)))]
            edge_attr[i] = dist

            # Pauling math tied directly to the assigned distance
            if atom_types[u] == 'H' and atom_types[v] == 'H': pair = 'HH'
            elif atom_types[u] == 'O' and atom_types[v] == 'O': pair = 'OO'
            else: pair = 'OH'

            bo_val = torch.exp(torch.tensor((targets[pair] - dist) / b))
            y_bond[i] = torch.clamp(bo_val, max=2.5, min=0.0)

        # # 4. Define y_exist based on bonded_pairs
        # y_exist = torch.zeros((4, 1))
        # bonded_atoms = set()
        # for u, v in bonded_pairs:
        #     bonded_atoms.add(u); bonded_atoms.add(v)
        # for node in bonded_atoms:
        #     y_exist[node] = 1.0

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y_bond=y_bond)
        dataset.append(data)
        
    return dataset

# Create 20 samples and wrap in a DataLoader
my_dataset = create_mini_dataset(1000)
train_loader = DataLoader(my_dataset, batch_size=16, shuffle=True)  # shuffle once per epoch

# # Debug: Check y_exist distribution in dataset
# print("Sample y_exist values from dataset:")
# for i in range(5):
#     print(f"  Sample {i}: {my_dataset[i].y_exist.squeeze().tolist()}")

# ==========================================
# 3. TRAINING INFRASTRUCTURE
# ==========================================

model = CombustionGNN()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion_bond = nn.MSELoss()
criterion_exist = nn.BCELoss()

def train():
    model.train()
    total_loss = 0.0
    for batch in train_loader:
        optimizer.zero_grad()
        
        # Only predict bond orders
        pred_bond = model(batch)
        
        # Single loss function: MSE for Bond Order
        loss = criterion_bond(pred_bond, batch.y_bond)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

# ==========================================
# 4. EXECUTION
# ==========================================

print("Starting training loop...")
for epoch in range(1, 201):
    avg_loss = train()
    if epoch % 10 == 0:
        print(f"Epoch: {epoch:03d}, Loss: {avg_loss:.4f}")

print("\nTraining complete. Model is now ready for testing!")

# ==========================================
# 5. TEST
# ==========================================

def build_4node_data(x_vals, bonded_distances):
    # edge_index = torch.tensor([
    #     [0,1, 1,0, 0,2, 2,0, 0,3, 3,0, 1,2, 2,1, 1,3, 3,1, 2,3, 3,2]
    # ], dtype=torch.long).view(2, -1)
    # Row 0 = Source (u), Row 1 = Target (v)
    edge_index = torch.tensor([
        [0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3], # u
        [1, 0, 2, 0, 3, 0, 2, 1, 3, 1, 3, 2]  # v
    ], dtype=torch.long)
    edge_attr = []
    for u, v in zip(edge_index[0].tolist(), edge_index[1].tolist()):
        edge_attr.append([bonded_distances.get((u, v), bonded_distances.get((v, u), 3.5))])
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    return Data(x=torch.tensor(x_vals, dtype=torch.float),
                edge_index=edge_index,
                edge_attr=edge_attr)

print("\n--- Test Cases ---")
model.eval()
with torch.no_grad():
    cases = [
        ('H2 + O2', [0, 0, 1, 1], {(0, 1): 0.74, (2, 3): 1.21}),
        ('H2O + O', [0, 0, 1, 1], {(0, 2): 0.97, (1, 2): 0.97}),
        ('OH + OH', [0, 1, 0, 1], {(0, 1): 0.97, (2, 3): 0.97}),
        ('H2O2', [0, 1, 1, 0], {(0, 1): 0.97, (1, 2): 1.47, (2, 3): 0.97}),
        ('HO2 + H', [0, 1, 1, 0], {(0, 1): 0.97, (1, 2): 1.33}),
        ('H2 + O + O', [0, 0, 1, 1], {(0, 1): 0.74}),
        ('O2 + H + H', [1, 1, 0, 0], {(0, 1): 1.21}),
        ('OH + O + H', [0, 1, 1, 0], {(0, 1): 0.97}),
        ('H + H + O + O', [0, 0, 1, 1], {})
    ]

    for name, x_vals, distances in cases:
        x_vectors = [
            [1.0, 0.0] if val == 0 else
            [0.0, 1.0]
            for val in x_vals
        ]
        data = build_4node_data(x_vectors, distances)
        pb = model(data)
        print(f"\n{name}")
        print(f"Node features: {x_vals}")
        print("Predicted bond orders:", [f"{p.item():.4f}" for p in pb.squeeze()])

# ==========================================
# 6. SAVE
# ==========================================
torch.save(model.state_dict(), "combustion_gnn_bo_small.pth")
print("Model weights saved to combustion_gnn_bo_small.pth")