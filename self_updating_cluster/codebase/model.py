"""Shared graph policy; existing Stage I and IV checkpoints remain loadable."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiGPUDataParallel(nn.DataParallel):
    """Expose model helpers such as ``encode`` during single-process evaluation."""

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

class TrainingDevice:
    """Select the requested device explicitly; keep checkpoints portable to CPU."""

    @staticmethod
    def resolve(name, gpu_ids=None):
        if name not in ("cpu", "mps", "cuda"):
            raise ValueError("TRAINING_DEVICE must be 'cpu', 'mps', or 'cuda'")
        if name == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError(
                "Apple GPU (MPS) is unavailable in this process. Run in your GNN terminal "
                "with an MPS-enabled PyTorch build, or explicitly set TRAINING_DEVICE = 'cpu'. "
                "Training has not started; no silent CPU fallback is used.")
        if name == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable. Load a CUDA-enabled PyTorch module on the cluster.")
            ids = [0] if gpu_ids is None else gpu_ids
            available = torch.cuda.device_count()
            if any(index >= available for index in ids):
                raise RuntimeError(f"Requested GPU IDs {ids}; only {available} CUDA GPU(s) are visible.")
            torch.cuda.set_device(ids[0])
            return torch.device("cuda", ids[0])
        return torch.device(name)

    @staticmethod
    def parallelize(model, device, gpu_ids=None):
        """Place a model on CUDA and split tensor forward batches across GPUs."""
        model = model.to(device)
        ids = [0] if gpu_ids is None else gpu_ids
        if device.type == "cuda" and len(ids) > 1:
            return MultiGPUDataParallel(model, device_ids=ids, output_device=ids[0])
        return model

    @staticmethod
    def unwrap(model):
        return model.module if isinstance(model, nn.DataParallel) else model

    @classmethod
    def cpu_data(cls, value):
        # Also handles Adam state, nested lists and action tuples.
        if torch.is_tensor(value):
            return value.detach().cpu().clone()
        if isinstance(value, dict):
            return {key: cls.cpu_data(item) for key, item in value.items()}
        if isinstance(value, list):
            return [cls.cpu_data(item) for item in value]
        if isinstance(value, tuple):
            return tuple(cls.cpu_data(item) for item in value)
        return value

    @staticmethod
    def synchronize(device):
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize(device)


class BondMessageLayer(nn.Module):
    """Explicit bond-order messages replace duplicated edges in version 4."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.self_layer = nn.Linear(input_dim, output_dim)
        # Each bond order has its own learned linear transformation of neighbor features.
        self.bond_layers = nn.ModuleList(
            nn.Linear(input_dim, output_dim, bias=False) for _ in range(3))
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, features, adjacency, node_mask):
        # Keep a contribution from the atom itself, then add messages from its neighbors.
        # bmm performs one adjacency-times-features multiplication per graph in the batch.
        output = self.self_layer(features)
        for bond_type, layer in enumerate(self.bond_layers):
            output = output + torch.bmm(adjacency[:, bond_type], layer(features))
        # Zero padded atom slots after every layer so padding cannot affect graph pooling.
        return F.leaky_relu(self.norm(output), 0.1) * node_mask.unsqueeze(-1)


class GrowthGNN(nn.Module):
    """One composition-conditioned GNN for every species, retaining v4 head names."""

    def __init__(self, hidden_dims, action_space, atom_symbols, max_valency):
        super().__init__()
        if not hidden_dims or any(type(width) is not int or width <= 0 for width in hidden_dims):
            raise ValueError("HIDDEN_DIMS must contain positive integers")
        self.action_space = action_space
        self.atom_symbols = dict(atom_symbols)
        self.max_valency = dict(max_valency)
        self.atom_index = {atom: i for i, atom in enumerate(action_space.atom_types)}
        # Default node vector: [is_H, is_O, is_C, bond_order_sum/4, remaining_valence/4].
        # This input dimension is separate from the editable hidden-layer widths.
        self.feature_dim = len(atom_symbols) + 2
        dims = [self.feature_dim] + list(hidden_dims)
        self.convs = nn.ModuleList(BondMessageLayer(a, b) for a, b in zip(dims[:-1], dims[1:]))
        h = hidden_dims[-1]
        # Target counts, current counts, difference: three C/H/O vectors.
        self.context_layer = nn.Sequential(nn.Linear(h + 9, h), nn.LeakyReLU(0.1))
        # START scores one choice per element; GROW scores 3 elements x 3 bond orders
        # at each potential attachment node. CONNECT scores 3 final orders per pair.
        self.start_head = nn.Linear(h, len(atom_symbols))
        self.grow_head = nn.Sequential(nn.Linear(2 * h, h), nn.LeakyReLU(0.1),
                                       nn.Linear(h, len(atom_symbols) * 3))
        # Symmetric pair features: h_i+h_j and abs(h_i-h_j).
        self.connect_head = nn.Sequential(nn.Linear(3 * h, h), nn.LeakyReLU(0.1),
                                          nn.Linear(h, 3))
        self.termination_layer = nn.Linear(h, 1)
        # Pair indices move with the model and are saved in its state, but are not learned.
        self.register_buffer("pair_u", torch.tensor([u for u, _ in action_space.pairs], dtype=torch.long))
        self.register_buffer("pair_v", torch.tensor([v for _, v in action_space.pairs], dtype=torch.long))

    def encode(self, environments, conditions):
        count, size = len(environments), self.action_space.max_atoms
        # Pad all graphs to MAX_ATOMS so different molecule sizes can share one batch.
        # Shapes: nodes [B,N,5], adjacency [B,3,N,N], node_mask [B,N],
        # contexts [B,9], valid_mask [B,A]; B=batch size, N=MAX_ATOMS, A=action count.
        nodes = torch.zeros(count, size, self.feature_dim)
        adjacency = torch.zeros(count, 3, size, size)
        node_mask = torch.zeros(count, size)
        contexts = torch.zeros(count, 9)
        valid_mask = torch.zeros(count, len(self.action_space.actions), dtype=torch.bool)
        for batch, (env, condition) in enumerate(zip(environments, conditions)):
            for u, atom in enumerate(env.node_types):
                nodes[batch, u, self.atom_index[atom]] = 1
                nodes[batch, u, -2] = env.current_bonds[u] / 4
                nodes[batch, u, -1] = (self.max_valency[atom] - env.current_bonds[u]) / 4
                node_mask[batch, u] = 1
            # The molecular bond is undirected, so messages flow in both directions.
            # This is separate from counting the CONNECT action only once.
            for (u, v), order in env.bonds.items():
                adjacency[batch, order - 1, u, v] = 1
                adjacency[batch, order - 1, v, u] = 1
            current = env.composition()
            # Supply target counts, current counts, and target-minus-current counts.
            # All count vectors use C,H,O order; dividing by MAX_ATOMS sets their scale.
            # Example at OH: target H2O still has one H remaining; target OH has zero.
            contexts[batch] = torch.tensor(
                list(condition) + list(current) + [a - b for a, b in zip(condition, current)]
            ) / size
            valid_mask[batch] = env.get_valid_action_masks(self.action_space)
        device = next(self.parameters()).device
        return tuple(t.to(device) for t in (nodes, adjacency, node_mask, contexts, valid_mask))

    def graph_features(self, nodes, adjacency, node_mask):
        """Compute features that depend only on the partial graph, not composition."""
        h = nodes
        for conv in self.convs:
            h = conv(h, adjacency, node_mask)
        # Average real-node embeddings into a graph summary. clamp_min(1) also handles
        # the empty graph, whose zero summary is used to predict the START element.
        pooled = h.sum(dim=1) / node_mask.sum(dim=1, keepdim=True).clamp_min(1)
        return h, pooled

    def action_logits(self, h, pooled, contexts, valid_mask):
        """Score actions for each requested composition using existing graph features."""
        context = self.context_layer(torch.cat((pooled, contexts), dim=-1))
        # Combine each local atom embedding with the same graph/composition context.
        per_node_context = context.unsqueeze(1).expand(-1, h.size(1), -1)
        grow_logits = self.grow_head(torch.cat((h, per_node_context), dim=-1)).flatten(1)
        # Sum and absolute difference make pair features invariant to swapping u and v.
        h_i, h_j = h[:, self.pair_u], h[:, self.pair_v]
        pair_context = context.unsqueeze(1).expand(-1, len(self.action_space.pairs), -1)
        connect_logits = self.connect_head(
            torch.cat((h_i + h_j, torch.abs(h_i - h_j), pair_context), dim=-1)
        ).flatten(1)
        # A logit is an unnormalized action score. Concatenate every head before
        # softmax, so START/GROW/CONNECT/STOP share one probability distribution.
        logits = torch.cat((self.start_head(context), grow_logits, connect_logits,
                            self.termination_layer(context)), dim=1)
        # Softmax assigns exactly zero probability to -infinity entries (illegal actions).
        return logits.masked_fill(~valid_mask, -torch.inf)

    def forward(self, nodes, adjacency, node_mask, contexts, valid_mask):
        # Preserve the original Stage I/II interface and all checkpoint parameter names.
        h, pooled = self.graph_features(nodes, adjacency, node_mask)
        return self.action_logits(h, pooled, contexts, valid_mask)

    def log_probs_over_contexts(self, environments, conditions, batch_size):
        """Return [graph, composition, action] log probabilities with shared GNN work.

        Graphs are encoded and convolved once per partial state. Only the action
        heads repeat across compositions. Features stay attached to autograd so
        all composition branches contribute to graph-layer gradients. Nothing is
        cached between calls or weight updates.
        """
        if not environments or not conditions or type(batch_size) is not int or batch_size < 1:
            raise ValueError("Expected nonempty graphs/conditions and a positive batch size")
        device = next(self.parameters()).device
        requested = torch.tensor(conditions, dtype=torch.float32, device=device)
        count = len(conditions)
        results = []
        for start in range(0, len(environments), batch_size):
            graphs = environments[start:start+batch_size]
            # The first condition is only a placeholder for encode(); its current
            # count columns are independent of that requested composition.
            nodes, adjacency, node_mask, context, valid_mask = self.encode(
                graphs, [conditions[0]] * len(graphs))
            h, pooled = self.graph_features(nodes, adjacency, node_mask)
            current = torch.round(context[:, 3:6] * self.action_space.max_atoms)
            chunks = []
            for offset in range(0, len(graphs) * count, batch_size):
                rows = torch.arange(offset, min(offset+batch_size, len(graphs)*count), device=device)
                graph_ids = rows // count
                requested_counts = requested[rows % count]
                current_counts = current[graph_ids]
                # Subtract integer counts before scaling to retain encode()'s
                # floating-point convention for the target-minus-current vector.
                differences = requested_counts - current_counts
                contexts = torch.cat((requested_counts, current_counts, differences), dim=1) / self.action_space.max_atoms
                logits = self.action_logits(h[graph_ids], pooled[graph_ids], contexts, valid_mask[graph_ids])
                chunks.append(logits.log_softmax(-1))
            results.append(torch.cat(chunks).reshape(len(graphs), count, -1))
        return torch.cat(results)
