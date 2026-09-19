"""Stage 1: one shared graph policy learns all active reference species.

Run from the directory containing species_graphs.json:
    python train_stage1.py

Classes come first, followed by editable parameters and direct execution.
Paths are relative to the working directory. CO remains excluded.
"""
# File formats and logging below support readable run records and checkpoints.
import csv
import hashlib
import json
import logging
import math
import os
import random
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

# NetworkX stores/checks molecular connectivity; PyTorch learns action probabilities.
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

from compact_json import CompactJSON


# Console/file logging replaces version 4's Tee while keeping the same message content.
class TrainingLogger:
    """Write identical, immediately flushed messages to the console and a file."""

    def __init__(self, path):
        self.logger = logging.Logger("stage1", level=logging.INFO)
        # The two handlers receive the same timestamped message; handlers flush after emitting.
        formatter = logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")
        for handler in (logging.StreamHandler(sys.stdout),
                        logging.FileHandler(path, mode="w", encoding="utf-8")):
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def info(self, message):
        self.logger.info(message)

    def exception(self, message):
        self.logger.exception(message)

    def close(self):
        # Iterate over a copy because removing handlers changes the original list.
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)


class AdvancedMoleculeEnv:
    """Version-4-style node_types/current_bonds/bonds for ONE molecule.

    Action tuples:
      ("START", atom_type)
      ("GROW", attachment_node, atom_type, bond_order)
      ("CONNECT", node_u, node_v, desired_final_bond_order)
      ("STOP",)
    Masks depend on chemistry and the generic size bound, never the target.
    """

    def __init__(self, atom_symbols, max_valency, max_atoms):
        self.atom_symbols = dict(atom_symbols)
        self.max_valency = dict(max_valency)
        self.max_atoms = max_atoms
        # Node IDs are local list indices: node_types[u] is the element code for node u.
        # current_bonds[u] is a bond-order SUM, not the number of neighboring atoms.
        # Example: carbon with one double bond and one single bond has sum 3.
        self.node_types = []
        self.current_bonds = []
        self.bonds = {}
        self.terminated = False

    # Demonstrations need independent snapshots: later actions must not modify earlier states.
    def copy(self):
        other = AdvancedMoleculeEnv(self.atom_symbols, self.max_valency, self.max_atoms)
        other.node_types = self.node_types.copy()
        other.current_bonds = self.current_bonds.copy()
        other.bonds = self.bonds.copy()
        other.terminated = self.terminated
        return other

    def get_valid_actions(self):
        if self.terminated:
            return []
        # An empty molecule has no attachment site, so its first action must be START.
        if not self.node_types:
            return [("START", atom) for atom in sorted(self.atom_symbols)]
        actions = []
        # Growth consumes bond capacity on BOTH the old atom and the new atom.
        # A hydrogen can therefore be added only with a single bond.
        if len(self.node_types) < self.max_atoms:
            for u, atom_u in enumerate(self.node_types):
                available = self.max_valency[atom_u] - self.current_bonds[u]
                for new_atom in sorted(self.atom_symbols):
                    for order in (1, 2, 3):
                        if order <= min(available, self.max_valency[new_atom]):
                            actions.append(("GROW", u, new_atom, order))
        # Visit u < v only: connecting (0,1) and (1,0) is the same physical bond.
        for u in range(len(self.node_types)):
            for v in range(u + 1, len(self.node_types)):
                current = self.bonds.get((u, v), 0)
                available = min(
                    self.max_valency[self.node_types[u]] - self.current_bonds[u],
                    self.max_valency[self.node_types[v]] - self.current_bonds[v],
                )
                # Upgrading a single bond to a triple consumes 3 - 1 = 2 extra units.
                for desired in (1, 2, 3):
                    if 0 < desired - current <= available:
                        actions.append(("CONNECT", u, v, desired))
        # Available valence does not imply that growth must continue: OH and CH3
        # are valid radical targets and must be allowed to stop before saturation.
        actions.append(("STOP",))
        return actions

    def get_valid_action_masks(self, action_space):
        # True means legal at this state. The requested composition is deliberately
        # absent here, so the mask does not supply the answer to the learning task.
        mask = torch.zeros(len(action_space.actions), dtype=torch.bool)
        for action in self.get_valid_actions():
            mask[action_space.index[action]] = True
        return mask

    def step(self, action):
        # Validate again at execution time, even though the network also masks actions.
        if action not in self.get_valid_actions():
            raise ValueError(f"Invalid action {action} in {self.describe()}")
        if action[0] == "START":
            self.node_types.append(action[1])
            self.current_bonds.append(0)
        elif action[0] == "GROW":
            _, u, atom, order = action
            # Append a new atom and update the bond-order sums at both endpoints.
            v = len(self.node_types)
            self.node_types.append(atom)
            self.current_bonds.append(order)
            self.current_bonds[u] += order
            self.bonds[(u, v)] = order
        elif action[0] == "CONNECT":
            _, u, v, desired = action
            # CONNECT specifies the FINAL order; add only the increase to valence sums.
            increase = desired - self.bonds.get((u, v), 0)
            self.bonds[(u, v)] = desired
            self.current_bonds[u] += increase
            self.current_bonds[v] += increase
        else:
            self.terminated = True

    # Convert mutable environment lists/dicts to the same atom/order graph format
    # used by the reference JSON and version 4's structure-matching logic.
    def to_graph(self):
        graph = nx.Graph()
        graph.add_nodes_from((i, {"atom": atom}) for i, atom in enumerate(self.node_types))
        graph.add_edges_from((u, v, {"order": order})
                             for (u, v), order in self.bonds.items())
        return graph

    def composition(self):
        # The condition order is always C, H, O, independently of numeric atom IDs.
        counts = Counter(self.atom_symbols[atom] for atom in self.node_types)
        return tuple(counts[symbol] for symbol in ("C", "H", "O"))

    def describe(self):
        atoms = ", ".join(f"{i}:{self.atom_symbols[a]}"
                          for i, a in enumerate(self.node_types)) or "empty"
        bonds = ", ".join(f"({u},{v},{order})"
                          for (u, v), order in sorted(self.bonds.items())) or "none"
        return f"atoms=[{atoms}] bonds=[{bonds}]"

    # Produce JSON-compatible data with explicit H nodes and each undirected bond once.
    def record(self):
        return {
            "nodes": [{"id": i, "atom": atom, "symbol": self.atom_symbols[atom]}
                      for i, atom in enumerate(self.node_types)],
            "edges": [{"source": u, "target": v, "order": order}
                      for (u, v), order in sorted(self.bonds.items())],
        }


class ActionSpace:
    """Fixed logit positions shared by labels, masking, sampling and log_prob."""

    def __init__(self, max_atoms, atom_symbols):
        self.max_atoms = max_atoms
        self.atom_types = sorted(atom_symbols)
        # This ordering must match GrowthGNN.forward exactly:
        # START elements, GROW node/element/order, CONNECT pair/order, then STOP.
        # Slots for nonexistent nodes remain present but are masked out.
        self.actions = [("START", atom) for atom in self.atom_types]
        self.actions += [
            ("GROW", u, atom, order)
            for u in range(max_atoms) for atom in self.atom_types for order in (1, 2, 3)
        ]
        self.pairs = [(u, v) for u in range(max_atoms) for v in range(u + 1, max_atoms)]
        self.actions += [("CONNECT", u, v, order)
                         for u, v in self.pairs for order in (1, 2, 3)]
        self.actions.append(("STOP",))
        # Translate an action tuple into the corresponding output-logit column.
        self.index = {action: index for index, action in enumerate(self.actions)}

    def describe(self, action, atom_symbols):
        if action[0] == "START":
            return f"START {atom_symbols[action[1]]}"
        if action[0] == "GROW":
            return f"ADD {atom_symbols[action[2]]} to node {action[1]}, bond order {action[3]}"
        if action[0] == "CONNECT":
            return f"CONNECT nodes {action[1]}-{action[2]}, final bond order {action[3]}"
        return "STOP"


class ReferenceSpecies:
    """Load JSON without importing species_graphs.py (which directly exports)."""

    def __init__(self, path, excluded_species, atom_symbols, max_valency, max_atoms):
        self.atom_symbols = dict(atom_symbols)
        self.max_valency = dict(max_valency)
        self.max_atoms = max_atoms
        # Isomorphism ignores node numbering but must preserve element and bond order.
        # Formula matching alone would confuse CH3O with CH2OH.
        self.node_match = nx.algorithms.isomorphism.categorical_node_match("atom", None)
        self.edge_match = nx.algorithms.isomorphism.categorical_edge_match("order", None)
        # Record which exact reference file produced this run for later reproducibility.
        self.source_hash = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.graphs = {}
        for name, record in data.items():
            if name in excluded_species:
                continue
            graph = nx.Graph()
            # Reject inconsistent input before making any demonstrations or updating weights.
            for node in record["nodes"]:
                node_id, atom = node["id"], node["atom"]
                if type(node_id) is not int or node_id < 0 or node_id in graph:
                    raise ValueError(f"{name}: invalid/duplicate node ID {node_id}")
                if atom not in atom_symbols or node.get("symbol") != atom_symbols[atom]:
                    raise ValueError(f"{name}: inconsistent atom encoding")
                graph.add_node(node_id, atom=atom)
            for edge in record["edges"]:
                u, v, order = edge["source"], edge["target"], edge["order"]
                if u not in graph or v not in graph or u == v or graph.has_edge(u, v):
                    raise ValueError(f"{name}: invalid/duplicate bond")
                if type(order) is not int or order not in (1, 2, 3):
                    raise ValueError(f"{name}: invalid bond order")
                graph.add_edge(u, v, order=order)
            if not graph or not nx.is_connected(graph) or len(graph) > max_atoms:
                raise ValueError(f"{name}: empty, disconnected or exceeds MAX_ATOMS")
            for node, attributes in graph.nodes(data=True):
                valence = sum(graph[node][v]["order"] for v in graph.neighbors(node))
                # Only excess valence is rejected; lower valence is allowed for radicals.
                if valence > max_valency[attributes["atom"]]:
                    raise ValueError(f"{name}: node {node} exceeds the configured valence")
            self.graphs[name] = graph
        if not self.graphs:
            raise ValueError("No active species")
        # Group names by (C,H,O) counts. Some conditions contain two reference isomers,
        # so 31 active species correspond to 29 distinct composition conditions.
        self.conditions = {}
        for name, graph in self.graphs.items():
            condition = self.composition(graph)
            self.conditions.setdefault(condition, []).append(name)
        self.pool = {}

    def composition(self, graph):
        counts = Counter(self.atom_symbols[d["atom"]] for _, d in graph.nodes(data=True))
        return tuple(counts[symbol] for symbol in ("C", "H", "O"))

    def isomorphic(self, left, right):
        return nx.is_isomorphic(left, right, node_match=self.node_match, edge_match=self.edge_match)

    # First narrow the candidates by composition, then compare their actual topology.
    def match(self, graph):
        for name in self.conditions.get(self.composition(graph), []):
            if self.isomorphic(graph, self.graphs[name]):
                return name
        return None

    def equivalent_actions(self, env, action):
        """Sum probability over node-symmetry-equivalent actions, not arbitrary targets."""
        if action[0] in ("START", "STOP"):
            return [action]
        graph = env.to_graph()
        # Map the partial graph onto itself to find interchangeable attachment sites.
        # For a symmetric C-C fragment, adding H to either end is equivalent.
        matcher = nx.algorithms.isomorphism.GraphMatcher(
            graph, graph, node_match=self.node_match, edge_match=self.edge_match)
        equivalent = set()
        for mapping in matcher.isomorphisms_iter():
            if action[0] == "GROW":
                equivalent.add(("GROW", mapping[action[1]], action[2], action[3]))
            else:
                u, v = sorted((mapping[action[1]], mapping[action[2]]))
                equivalent.add(("CONNECT", u, v, action[3]))
        return sorted(equivalent)

    # A demonstration is a list of (partial graph, condition, next-action label) items.
    # The reference graph is used to construct labels, never as an input to the GNN.
    def make_demonstration(self, name, rng, action_space):
        reference = self.graphs[name]
        env = AdvancedMoleculeEnv(self.atom_symbols, self.max_valency, self.max_atoms)
        mapping = {}  # Reference node -> current environment node, never a model input.
        items = []
        while True:
            if not mapping:
                # Random roots teach the same molecule under different construction histories.
                original = rng.choice(list(reference.nodes))
                action = ("START", reference.nodes[original]["atom"])
                new_original = original
            else:
                choices = []
                # A reference bond crossing the built/unbuilt boundary supplies a GROW
                # action. A missing bond between two built nodes supplies CONNECT.
                for u, v, edge in reference.edges(data=True):
                    if u in mapping and v not in mapping:
                        choices.append((("GROW", mapping[u], reference.nodes[v]["atom"], edge["order"]), v))
                    elif v in mapping and u not in mapping:
                        choices.append((("GROW", mapping[v], reference.nodes[u]["atom"], edge["order"]), u))
                    elif u in mapping and v in mapping:
                        a, b = sorted((mapping[u], mapping[v]))
                        if env.bonds.get((a, b), 0) != edge["order"]:
                            choices.append((("CONNECT", a, b, edge["order"]), None))
                if choices:
                    action, new_original = rng.choice(choices)
                else:
                    if len(mapping) != len(reference):
                        raise ValueError(f"{name}: incomplete construction")
                    # Completion is defined by the reference, not by exhausted valence.
                    action, new_original = ("STOP",), None
            alternatives = self.equivalent_actions(env, action)
            legal = env.get_valid_actions()
            if any(candidate not in legal for candidate in alternatives):
                raise ValueError(f"{name}: demonstration violates action masks")
            # Save the state BEFORE applying the label. copy() prevents all training
            # examples from accidentally pointing to the final completed molecule.
            items.append({
                "env": env.copy(),
                "condition": self.composition(reference),
                "action": action,
                "target_indices": [action_space.index[a] for a in alternatives],
            })
            new_id = len(env.node_types)
            env.step(action)
            if new_original is not None:
                mapping[new_original] = new_id
            if env.terminated:
                break
        # Replay verification catches missing atoms/bonds and incorrect bond orders.
        if not self.isomorphic(env.to_graph(), reference):
            raise ValueError(f"{name}: demonstration does not reconstruct the reference")
        return items

    def enumerate_sequences(self, name):
        """Enumerate unique action sequences, keeping structurally different routes.

        Equal atom types may have different future neighbours. Explore every
        reference-node mapping and deduplicate COMPLETE action sequences, so
        interchangeable labels disappear without losing valid construction routes.
        Bond orders are introduced directly, as in make_demonstration.
        """
        graph = self.graphs[name]
        sequences = set()

        def visit(mapping, built_bonds, actions):
            choices = []
            if not mapping:
                choices = [(("START", graph.nodes[u]["atom"]), u, None) for u in graph]
            else:
                for u, v, edge in graph.edges(data=True):
                    bond = frozenset((u, v))
                    if u in mapping and v not in mapping:
                        choices.append((("GROW", mapping[u], graph.nodes[v]["atom"],
                                         edge["order"]), v, bond))
                    elif v in mapping and u not in mapping:
                        choices.append((("GROW", mapping[v], graph.nodes[u]["atom"],
                                         edge["order"]), u, bond))
                    elif u in mapping and v in mapping and bond not in built_bonds:
                        a, b = sorted((mapping[u], mapping[v]))
                        choices.append((("CONNECT", a, b, edge["order"]), None, bond))
            if not choices:
                if len(mapping) != len(graph):
                    raise ValueError(f"{name}: incomplete enumeration")
                sequences.add(actions + (("STOP",),))
                return
            for action, new_node, bond in choices:
                next_mapping = dict(mapping)
                if new_node is not None:
                    next_mapping[new_node] = len(mapping)
                next_bonds = built_bonds | {bond} if bond is not None else built_bonds
                visit(next_mapping, next_bonds, actions + (action,))

        visit({}, set(), ())
        # Stable ordering makes seeded sampling reproducible across runs.
        return sorted(sequences)

    def build_demonstrations(self, action_space):
        # Replay every unique sequence once and retain the state BEFORE each action.
        for name in self.graphs:
            self.pool[name] = []
            for sequence in self.enumerate_sequences(name):
                env = AdvancedMoleculeEnv(self.atom_symbols, self.max_valency, self.max_atoms)
                items = []
                for action in sequence:
                    alternatives = self.equivalent_actions(env, action)
                    legal = env.get_valid_actions()
                    if any(a not in legal for a in alternatives):
                        raise ValueError(f"{name}: illegal enumerated target")
                    items.append({"env": env.copy(), "condition": self.composition(self.graphs[name]),
                                  "action": action,
                                  "target_indices": [action_space.index[a] for a in alternatives]})
                    env.step(action)
                if not env.terminated or not self.isomorphic(env.to_graph(), self.graphs[name]):
                    raise ValueError(f"{name}: enumerated sequence does not reconstruct reference")
                self.pool[name].append(items)

    @staticmethod
    def sample_count(total, fraction):
        # Reject zero, >100%, NaN and infinity instead of silently changing the request.
        if isinstance(fraction, bool) or not 0 < fraction <= 1:
            raise ValueError("trajectory_sample_fraction must be greater than 0 and at most 1")
        return math.ceil(total * fraction)

    def save_sequences(self, path, fraction):
        # Save every route as compact actions; training_examples has detailed graph states.
        CompactJSON.write(path, {
            name: {"total_sequences": len(orders),
                   "sampled_per_epoch": self.sample_count(len(orders), fraction),
                   "sequences": [[list(item["action"]) for item in items] for items in orders]}
            for name, orders in self.pool.items()
        })

    def save_demonstrations(self, path, action_space):
        # One readable example per species; all enumerated orders are validated in memory.
        data = {}
        for name, trajectories in self.pool.items():
            data[name] = {
                "condition_C_H_O": list(self.composition(self.graphs[name])),
                "steps": [
                    {"step": i + 1, "graph": item["env"].record(),
                     "next_action": list(item["action"]),
                     "description": action_space.describe(item["action"], self.atom_symbols),
                     "equivalent_actions": [list(action_space.actions[j])
                                            for j in item["target_indices"]]}
                    for i, item in enumerate(trajectories[0])
                ],
            }
        CompactJSON.write(path, data)


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

    def forward(self, nodes, adjacency, node_mask, contexts, valid_mask):
        h = nodes
        for conv in self.convs:
            h = conv(h, adjacency, node_mask)
        # Average real-node embeddings into a graph summary. clamp_min(1) also handles
        # the empty graph, whose zero summary is used to predict the START element.
        pooled = h.sum(dim=1) / node_mask.sum(dim=1, keepdim=True).clamp_min(1)
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


class Stage1Trainer:
    """Balanced imitation, readable saved logs, and independent free rollouts."""

    def __init__(self, config):
        self.config = dict(config)
        self.output = Path(config["output_folder"])
        # Refuse accidental replacement of a previous run; choose another relative path.
        self.output.mkdir(parents=True, exist_ok=False)
        self.log = TrainingLogger(self.output / "training.log")
        self.atom_symbols = config["atom_symbols"]
        self.max_valency = config["max_valency"]
        self.seed = config["seed"]

    def condition_text(self, condition):
        return ", ".join(f"{symbol}={count}" for symbol, count in zip(("C", "H", "O"), condition))

    def evaluate(self, model, references, action_space, epoch, samples, mode, save_trace=False):
        # Free generation uses the model's own actions, unlike teacher-forced training.
        # No reference next-action labels are consulted while constructing these molecules.
        model.eval()
        # Evaluate each unique composition once per sample, rather than giving an
        # isomer pair twice as many requests merely because it has two reference names.
        conditions = [condition for condition in references.conditions for _ in range(samples)]
        environments = [AdvancedMoleculeEnv(self.atom_symbols, self.max_valency,
                                            self.config["max_atoms"]) for _ in conditions]
        traces = [[] for _ in conditions]
        # Hitting the step cap is not successful termination, even if the final graph matches.
        reasons = ["step_limit"] * len(conditions)
        # Use a separate fixed evaluation RNG so sampling does not alter training choices.
        generator = torch.Generator().manual_seed(self.seed + 10000)
        # Evaluation does not update weights or need the gradient history.
        with torch.no_grad():
            for step in range(self.config["max_steps"]):
                # Finished molecules leave the active batch; the others keep growing.
                active = [i for i, env in enumerate(environments) if not env.terminated]
                if not active:
                    break
                inputs = model.encode([environments[i] for i in active], [conditions[i] for i in active])
                logits = model(*inputs)
                probabilities = logits.softmax(-1)
                # Greedy chooses the largest probability; sampled draws from the same
                # distribution and can reveal multiple isomers for one condition.
                if mode == "greedy":
                    choices = logits.argmax(-1)
                else:
                    choices = torch.multinomial(probabilities, 1, generator=generator).squeeze(1)
                for row, index in enumerate(active):
                    action = action_space.actions[int(choices[row])]
                    env = environments[index]
                    if save_trace:
                        traces[index].append({
                            "step": step + 1, "before": env.describe(), "action": list(action),
                            "description": action_space.describe(action, self.atom_symbols),
                            "probability": float(probabilities[row, choices[row]]),
                        })
                    env.step(action)
                    if env.terminated:
                        reasons[index] = "STOP"
        counts = Counter()
        summaries = {condition: {"correct": 0, "composition": 0, "stopped": 0, "outputs": Counter()}
                     for condition in references.conditions}
        records = []
        for i, (condition, env) in enumerate(zip(conditions, environments)):
            match = references.match(env.to_graph())
            composition_ok = env.composition() == condition
            # A formula match is insufficient: require an exact reference structure,
            # the requested composition, and an explicit STOP action.
            correct = reasons[i] == "STOP" and composition_ok and match in references.conditions[condition]
            summary = summaries[condition]
            summary["correct"] += int(correct)
            summary["composition"] += int(composition_ok)
            summary["stopped"] += int(reasons[i] == "STOP")
            summary["outputs"][match or "unknown"] += 1
            if correct:
                counts[match] += 1
            graph = env.to_graph()
            # Export compact base atoms/bonds and H counts alongside the complete graph.
            # For H or H2, retain H as a base atom because no heavy atom exists.
            base = [u for u in graph if self.atom_symbols[graph.nodes[u]["atom"]] != "H"]
            pure_h = not base
            if pure_h:
                base = list(graph)
            mapping = {u: j for j, u in enumerate(base)}
            record = {
                "epoch": epoch, "mode": mode, "sample": i + 1,
                "requested_C_H_O": list(condition), "generated_C_H_O": list(env.composition()),
                "species_key": match or "Unknown",
                "base_atom_indices": {j: self.atom_symbols[graph.nodes[u]["atom"]]
                                      for u, j in mapping.items()},
                "base_bonds": [(mapping[u], mapping[v], d["order"])
                               for u, v, d in graph.edges(data=True) if u in mapping and v in mapping],
                "attached_h_per_base_atom": [] if pure_h else [
                    sum(self.atom_symbols[graph.nodes[v]["atom"]] == "H" for v in graph.neighbors(u))
                    for u in base],
                "in_species_training_reference": match is not None,
                "correct_for_condition": correct, "termination": reasons[i],
                "graph": env.record(),
            }
            if save_trace:
                record["trace"] = traces[i]
            records.append(record)
        # Reconstruction rate counts successful samples. Coverage counts distinct species.
        # A run can cover every species while still producing many incorrect samples.
        rate = sum(s["correct"] for s in summaries.values()) / len(conditions)
        self.log.info(f"EVALUATION epoch={epoch} mode={mode} | correct structure+composition="
                      f"{rate:.1%} | coverage={len(counts)}/{len(references.graphs)}"
                      f" | STOP={sum(r == 'STOP' for r in reasons)}/{len(reasons)}")
        for condition, summary in summaries.items():
            self.log.info(
                f"  [{self.condition_text(condition)}] expected={','.join(references.conditions[condition])}"
                f" | correct={summary['correct']}/{samples}"
                f" | composition={summary['composition']}/{samples}"
                f" | generated={dict(summary['outputs'])}")
        for name in references.graphs:
            self.log.info(f"  SPECIES {name:<7} | correctly generated={counts[name]}")
        # Keep the console readable by showing detailed steps for OH, H2O and CH4.
        # Every final greedy trace is still included in the saved evaluation JSON.
        if save_trace:
            for record in records:
                if record["requested_C_H_O"] in ([0, 1, 1], [0, 2, 1], [1, 4, 0]):
                    self.log.info(f"ROLLOUT [{self.condition_text(record['requested_C_H_O'])}] "
                                  f"-> {record['species_key']} ({record['termination']})")
                    for item in record["trace"]:
                        self.log.info(f"  step {item['step']:02d} | {item['before']} | "
                                      f"{item['description']} | p={item['probability']:.4f}")
        return {"rate": rate, "coverage": len(counts), "counts": dict(counts), "records": records}

    def save_evaluation(self, evaluation, filename):
        records = evaluation["records"]
        CompactJSON.write(self.output / f"{filename}.json", evaluation)
        # JSON retains graphs and optional traces; CSV provides the compact Excel-readable table.
        columns = ["epoch", "mode", "sample", "requested_C_H_O", "generated_C_H_O", "species_key",
                   "base_atom_indices", "base_bonds", "attached_h_per_base_atom",
                   "in_species_training_reference", "correct_for_condition", "termination"]
        with (self.output / f"{filename}.csv").open("w", newline="", encoding="utf-8-sig") as output:
            writer = csv.DictWriter(output, fieldnames=columns)
            writer.writeheader()
            for record in records:
                writer.writerow({key: record[key] for key in columns})

    def run(self):
        try:
            self.train()
        except Exception:
            self.log.exception("TRAINING FAILED: see traceback; existing logs are retained.")
            raise
        # Close file handlers on success or failure so the saved log is complete.
        finally:
            self.log.close()

    @staticmethod
    def select_trajectories(cached, fraction, rng):
        """Sample ceil(N_s * fraction) routes and give each species total weight 1/S."""
        selected, step_weights = [], []
        for entries in cached.values():
            count = ReferenceSpecies.sample_count(len(entries), fraction)
            # rng persists across epochs; random.sample selects without replacement.
            for entry in rng.sample(entries, count):
                selected.append(entry)
                steps = len(entry[1])
                # s = species, k = sampled trajectory, t = action step:
                # each step receives 1 / (S * K_s * T_s,k).
                step_weights.append(torch.full((steps,), 1 / (len(cached) * count * steps)))
        return selected, torch.cat(step_weights)

    def train(self):
        cfg = self.config
        for key in ("epochs", "eval_every",
                    "eval_samples", "max_atoms", "max_steps", "threads"):
            if type(cfg[key]) is not int or cfg[key] < 1:
                raise ValueError(f"{key} must be a positive integer")
        ReferenceSpecies.sample_count(1, cfg["trajectory_sample_fraction"])
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.set_num_threads(cfg["threads"])
        torch.use_deterministic_algorithms(True)
        # This RNG selects training trajectories independently of demonstration generation.
        rng = random.Random(self.seed + 1)
        references = ReferenceSpecies(cfg["reference_path"], cfg["excluded_species"],
                                      self.atom_symbols, self.max_valency, cfg["max_atoms"])
        action_space = ActionSpace(cfg["max_atoms"], self.atom_symbols)
        self.log.info("STAGE 1 | shared composition-conditioned GNN | supervised imitation")
        self.log.info(f"Reference={cfg['reference_path']} | SHA256={references.source_hash}")
        self.log.info(f"Active species={len(references.graphs)} | distinct compositions="
                      f"{len(references.conditions)} | excluded={cfg['excluded_species']}")
        self.log.info("One molecule per episode. Condition order: C,H,O. "
                      "Masks use chemistry only; STOP is allowed before valence saturation.")
        self.log.info("Training and evaluation use supplied reference structures. "
                      "These are reconstruction metrics, not held-out molecular generalization.")
        self.log.info(f"Config: {json.dumps(cfg, sort_keys=True)}")
        CompactJSON.write(self.output / "config.json", cfg)
        references.build_demonstrations(action_space)
        references.save_sequences(self.output / "formation_sequences.json", cfg["trajectory_sample_fraction"])
        references.save_demonstrations(self.output / "training_examples.json", action_space)
        for name, trajectories in references.pool.items():
            example = trajectories[0]
            self.log.info(f"REFERENCE {name:<7} | {self.condition_text(example[0]['condition'])}"
                          f" | atoms={len(references.graphs[name])}"
                          f" | unique sequences={len(trajectories)}"
                          f" | sampled/epoch={references.sample_count(len(trajectories), cfg['trajectory_sample_fraction'])}"
                          f" | steps={len(example)}")
            self.log.info("  " + " -> ".join(action_space.describe(s["action"], self.atom_symbols)
                                           for s in example))
        model = GrowthGNN(cfg["hidden_dims"], action_space, self.atom_symbols, self.max_valency)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
        self.log.info(f"MODEL node_features={model.feature_dim} | hidden_dims={cfg['hidden_dims']}"
                      f" | trainable_parameters={sum(p.numel() for p in model.parameters())}")
        # Save an untrained baseline to distinguish learning from success due to masks alone.
        baseline = self.evaluate(model, references, action_space, 0, 1, "greedy")
        self.save_evaluation(baseline, "baseline_greedy")
        # Cache tensors per reference order; no target graph is fed into the network.
        cached = {}
        for name, trajectories in references.pool.items():
            cached[name] = []
            for items in trajectories:
                inputs = model.encode([item["env"] for item in items],
                                      [item["condition"] for item in items])
                # A target row may contain several True entries for symmetry-equivalent
                # actions. It is separate from the mask of all chemically legal actions.
                targets = torch.zeros(len(items), len(action_space.actions), dtype=torch.bool)
                for i, item in enumerate(items):
                    targets[i, item["target_indices"]] = True
                if not torch.all((targets & inputs[-1]).any(dim=1)):
                    raise ValueError("Demonstration target was masked out")
                cached[name].append((inputs, targets))
        metrics_path = self.output / "metrics.csv"
        with metrics_path.open("w", newline="", encoding="utf-8") as metrics_file:
            writer = csv.writer(metrics_file)
            writer.writerow(["epoch", "mean_species_nll", "equivalent_action_accuracy",
                             "mean_target_probability", "gradient_norm", "seconds"])
            started = time.monotonic()
            for epoch in range(1, cfg["epochs"] + 1):
                epoch_start = time.monotonic()
                # A fresh subset per species, with no duplicates within this epoch.
                # Cached tensors describe teacher-forced states, not model rollouts.
                selected, weights = self.select_trajectories(
                    cached, cfg["trajectory_sample_fraction"], rng)
                inputs = tuple(torch.cat([entry[0][i] for entry in selected], dim=0)
                               for i in range(5))
                targets = torch.cat([entry[1] for entry in selected])
                model.train()
                # Clear previous gradients before the single optimizer update for this epoch.
                optimizer.zero_grad()
                logits = model(*inputs)
                log_probs = F.log_softmax(logits, dim=-1)
                # Numerically stable log(sum of probabilities of acceptable actions).
                # With one target, this is ordinary cross-entropy; equivalent targets
                # share credit instead of penalizing an interchangeable attachment site.
                target_log_prob = torch.logsumexp(log_probs.masked_fill(~targets, -torch.inf), dim=1)
                loss = -(weights * target_log_prob).sum()
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite training loss")
                # Backpropagate through the GNN and all action heads. Gradient clipping
                # limits the combined gradient norm when an update would be unusually large.
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"])
                optimizer.step()
                # Teacher-action accuracy is measured on these demonstrated states BEFORE
                # the update; it is not the complete-molecule free-generation success rate.
                action_correct = targets.gather(1, logits.argmax(1, keepdim=True)).squeeze(1)
                accuracy = float((weights * action_correct).sum())
                probability = float((weights * target_log_prob.detach().exp()).sum())
                elapsed = time.monotonic() - epoch_start
                writer.writerow([epoch, float(loss.detach()), accuracy, probability,
                                 float(gradient_norm), elapsed])
                metrics_file.flush()
                self.log.info(f"EPOCH {epoch:04d}/{cfg['epochs']} | loss={float(loss.detach()):.5f}"
                              f" | sampled sequences={len(selected)}"
                              f" | teacher-action accuracy={accuracy:.1%}"
                              f" | mean correct-action probability={probability:.4f}"
                              f" | grad_norm={float(gradient_norm):.3f} | {elapsed:.2f}s")
                if epoch % cfg["eval_every"] == 0 or epoch == cfg["epochs"]:
                    result = self.evaluate(model, references, action_space, epoch, 1, "greedy",
                                           save_trace=epoch == cfg["epochs"])
                    self.save_evaluation(result, f"greedy_epoch_{epoch:04d}")
                    # Save the latest evaluated weights plus the settings needed to rebuild
                    # the same model/action space. This code does not automatically resume runs.
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(), "epoch": epoch,
                        "config": cfg, "source_sha256": references.source_hash,
                        "action_space": action_space.actions,
                    }, self.output / "trained_growth_gnn.pt")
            # Final stochastic evaluation measures isomer diversity beyond one greedy output.
            sampled = self.evaluate(model, references, action_space, cfg["epochs"],
                                    cfg["eval_samples"], "sampled")
            self.save_evaluation(sampled, "generated_species_inventory")
            self.log.info(f"FINISHED | elapsed={time.monotonic() - started:.1f}s"
                          f" | sampled reconstruction={sampled['rate']:.1%}"
                          f" | sampled coverage={sampled['coverage']}/{len(references.graphs)}")
            self.log.info(f"Saved log: {self.output / 'training.log'}")
            self.log.info(f"Saved model: {self.output / 'trained_growth_gnn.pt'}")
            self.log.info(f"Saved generated inventory: {self.output / 'generated_species_inventory.csv'}")


# ==========================================
# PARAMETERS — edit these before running
# ==========================================
ATOM_H = 0
ATOM_O = 1
ATOM_C = 2
ATOM_SYMBOL = {ATOM_H: "H", ATOM_O: "O", ATOM_C: "C"}
MAX_VALENCY = {ATOM_H: 1, ATOM_O: 2, ATOM_C: 4}
EXCLUDED_SPECIES = ["CO"]

REFERENCE_JSON_PATH = "species_graphs.json"
# Relative paths stay relative to your working directory.
OUTPUT_FOLDER = os.environ.get(
    "STAGE1_OUTPUT_FOLDER", "output_stage1/" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
# One width per message-passing layer. All species share these same learned layers.
GNN_HIDDEN_DIMS = [32, 64, 32, 32, 16]
SEED = 12345
# Generic safety bounds, not molecule-specific targets; cap termination is a failure.
MAX_ATOMS = 10
MAX_STEPS_PER_MOLECULE = 24
# Adam step size. TOTAL_EPOCHS counts balanced supervised updates, not molecule actions.
LEARNING_RATE = 0.003
TOTAL_EPOCHS = int(os.environ.get("STAGE1_EPOCHS", "300"))
# Enumerate all unique sequences once; sample ceil(total * fraction) per species.
# 0.20 means 20%, without replacement within an epoch. Valid range: 0 < fraction <= 1.
TRAJECTORY_SAMPLE_FRACTION = 0.20
# Run greedy generation periodically; final sampled evaluation uses the next parameter.
EVALUATE_EVERY = int(os.environ.get("STAGE1_EVAL_EVERY", "50"))
# 100 samples x 29 distinct compositions = 2,900 final generation attempts.
EVALUATION_SAMPLES_PER_COMPOSITION = int(os.environ.get("STAGE1_EVAL_SAMPLES", "100"))
# Maximum gradient norm per update; CPU_THREADS controls PyTorch CPU parallelism.
GRADIENT_CLIP = 5.0
CPU_THREADS = 1

# ==========================================
# RUN TRAINING — direct calls, no main guard
# ==========================================
# Pass the editable settings into the trainer and record them in config.json.
# Direct execution is intentional: running or importing this file starts training.
training_config = {
    "atom_symbols": ATOM_SYMBOL, "max_valency": MAX_VALENCY,
    "reference_path": REFERENCE_JSON_PATH, "excluded_species": EXCLUDED_SPECIES,
    "output_folder": OUTPUT_FOLDER, "hidden_dims": GNN_HIDDEN_DIMS, "seed": SEED,
    "max_atoms": MAX_ATOMS, "max_steps": MAX_STEPS_PER_MOLECULE,
    "learning_rate": LEARNING_RATE, "epochs": TOTAL_EPOCHS,
    "trajectory_sample_fraction": TRAJECTORY_SAMPLE_FRACTION,
    "eval_every": EVALUATE_EVERY, "eval_samples": EVALUATION_SAMPLES_PER_COMPOSITION,
    "gradient_clip": GRADIENT_CLIP, "threads": CPU_THREADS,
}
trainer = Stage1Trainer(training_config)
trainer.run()
