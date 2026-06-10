"""Sketch of the CombustionConv layer.

CombustionConv is a message-passing layer built on top of PyTorch Geometric.
It updates each node by collecting information from its neighbors, but unlike
plain graph convolution it also uses edge attributes, here interpreted as an
interatomic distance, to scale each incoming message.

The flow is:
1. Each neighbor feature x_j is sent through message().
2. The distance_mlp turns a 1D edge distance into a learned weight vector.
3. The neighbor feature is multiplied by that weight.
4. All messages are summed by the MessagePassing base class.
5. The result is projected to the requested output size with a linear layer.

The example at the bottom creates a tiny 3-node graph and runs one forward
pass so you can see how to instantiate and call the layer.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


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
        ## MessagePassing uses edge_index to know which nodes are connected. For each edge, it calls "message(...)"
        # 1. Start message passing
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        # 2. Transform the result to the 16-dimensional space
        out = self.lin(out)
        return out

    def message(self, x_j, edge_attr):
        # x_j is the feature of the neighbor atom
        # We multiply the neighbor's info by a weight based on distance
        weight = self.distance_mlp(edge_attr)
        return x_j * weight

    def update(self, aggr_out):
        # aggr_out is the sum of messages. We could add more layers here.
        return aggr_out


if __name__ == "__main__":
    # Tiny example graph with three nodes.
    x = torch.tensor([[0.0], [1.0], [0.5]], dtype=torch.float)
    # three nodes, and one feature per node

    # Undirected connections written as paired directed edges.
    edge_index = torch.tensor([
        [0, 1, 1, 2],
        [1, 0, 2, 1],
    ], dtype=torch.long)
    # four edges: 0->1, 1->0, 1->2, 2->1

    # One distance value per edge.
    edge_attr = torch.tensor([
        [0.97],
        [0.97],
        [1.25],
        [1.25],
    ], dtype=torch.float)
    # one scalar (distance) per edge

    conv = CombustionConv(in_channels=1, out_channels=4)
    out = conv(x, edge_index, edge_attr)

    print("Input node features:")
    print(x)
    print("\nOutput node features:")
    print(out)