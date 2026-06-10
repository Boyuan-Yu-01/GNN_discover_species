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
        super().__init__(aggr='add') 
        self.lin = nn.Linear(in_channels, out_channels)
        self.distance_mlp = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, in_channels) 
        )

    def forward(self, x, edge_index, edge_attr):
        # Message passing
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        # Transform the result to the 16-dimensional space
        return self.lin(out)

    def message(self, x_j, edge_attr):
        weight = self.distance_mlp(edge_attr)
        return x_j * weight

class CombustionGNN(torch.nn.Module):
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
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        # 2. Predict Existence for each node
        existence_prob = self.existence_predictor(x)
        # 3. Predict Bond Order for each edge
        # We get the features of the two atoms forming each edge
        row, col = edge_index
        edge_features = torch.cat([x[row], x[col]], dim=1)
        bond_order = self.bond_predictor(edge_features)
        
        return bond_order, existence_prob

# ==========================================
# 2. DATASET GENERATION
# ==========================================

def create_mini_dataset(num_samples=100):
    dataset = []
    for _ in range(num_samples):
        x = torch.tensor([[0], [0], [1], [1]], dtype=torch.float)
        edge_index = torch.tensor([[0, 2, 1, 3, 2, 0, 3, 1],
                                   [2, 0, 3, 1, 0, 2, 1, 3]], dtype=torch.long)
        
        # Physics: Bonded (~0.97A) vs Dissociated (~2.5A)
        # the initialization of the dataset is uniform for edge_attr, y_bond, and y_exist.
        # if no bond, then no bond between any atoms, and the edge_attr, y_bond, y_exist are the same for all edges
        is_bonded = random.choice([True, False])
        dist = 0.97 + (random.random() * 0.1) if is_bonded else 2.5 + random.random()# for formed bond, the distance is around 0.97A; for non-formed bond, the distance is around 2.5A
        
        edge_attr = torch.full((8, 1), dist)    # make a 8 x 1 tensor filled with scalar 'dist'
        y_bond = torch.full((8, 1), 1.0 if is_bonded else 0.0)
        y_exist = torch.full((4, 1), 1.0 if is_bonded else 0.0)
        
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, 
                    y_bond=y_bond, y_exist=y_exist)
        dataset.append(data)
    return dataset

# Create 20 samples and wrap in a DataLoader
my_dataset = create_mini_dataset(100)
train_loader = DataLoader(my_dataset, batch_size=16, shuffle=True)

# ==========================================
# 3. TRAINING INFRASTRUCTURE
# ==========================================

model = CombustionGNN()
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion_bond = nn.MSELoss()
criterion_exist = nn.BCELoss()

def train():
    model.train()
    total_loss = 0
    
    for batch in train_loader:
        optimizer.zero_grad()
        
        # 1. Forward Pass: bond order prediction and existence prediction using combustion GNN
        pred_bond, pred_exist = model(batch)
        
        # 2. Calculate "Average of Averages": batch.y_bond and batch.y_exist are the ground truth target tensors created when building each data sample
        loss_bond = criterion_bond(pred_bond, batch.y_bond)
        loss_exist = criterion_exist(pred_exist, batch.y_exist)
        
        # Combine losses (you can adjust the 1.0 weight later)
        loss = loss_bond + 1.0 * loss_exist
        
        # 3. Backward Pass
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    return total_loss / len(train_loader)

# ==========================================
# 4. EXECUTION
# ==========================================

print("Starting training loop...")
for epoch in range(1, 301):
    avg_loss = train()
    if epoch % 10 == 0:
        print(f"Epoch: {epoch:03d}, Loss: {avg_loss:.4f}")

print("\nTraining complete. Model is now ready for testing!")

# ==========================================
# 5. TEST
# ==========================================

print("Now testing the model...")
# 1. Create a "Tight Bond" test (0.90 A)
tight_x = torch.tensor([[0], [0], [1], [1]], dtype=torch.float)
tight_edge_index = torch.tensor([[0, 2], [2, 0]], dtype=torch.long)
tight_edge_attr = torch.tensor([[0.90], [0.90]], dtype=torch.float)
tight_data = Data(x=tight_x, edge_index=tight_edge_index, edge_attr=tight_edge_attr)

# 2. Create a "Broken Bond" test (3.50 A)
broken_edge_attr = torch.tensor([[3.50], [3.50]], dtype=torch.float)
broken_data = Data(x=tight_x, edge_index=tight_edge_index, edge_attr=broken_edge_attr)

model.eval()
with torch.no_grad():
    b_tight, e_tight = model(tight_data)
    b_broken, e_broken = model(broken_data)

print("--- The Acid Test ---")
print(f"Dist 0.90A -> Bond Order: {b_tight[0].item():.4f}, Exist: {e_tight[0].item():.4f}")
print(f"Dist 3.50A -> Bond Order: {b_broken[0].item():.4f}, Exist: {e_broken[0].item():.4f}")

# ==========================================
# 6. SAVE
# ==========================================
torch.save(model.state_dict(), "combustion_gnn_mini.pth")
print("Model weights saved to combustion_gnn_mini.pth")