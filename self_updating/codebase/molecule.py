"""Molecular construction and reference graphs, extracted from Stage I."""
import json
import math
import csv
from collections import Counter
from pathlib import Path
import networkx as nx
import torch
from .storage import CompactJSON


class SpeciesInput:
    """Read a user-supplied species table and return the internal graph dictionary."""

    @staticmethod
    def read(path, atom_codes=None):
        path = Path(path)
        if path.suffix.lower() == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
        elif path.suffix.lower() == ".csv":
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                rows = csv.DictReader(handle)
                required = {"species_key", "graph_json"}
                if rows.fieldnames is None or not required <= set(rows.fieldnames):
                    raise ValueError(f"{path}: CSV needs columns {sorted(required)}")
                data = {}
                for row_number, row in enumerate(rows, start=2):
                    name = (row.get("species_key") or "").strip()
                    if not name or name in data:
                        raise ValueError(f"{path}:{row_number}: missing or duplicate species_key")
                    try:
                        record = json.loads(row.get("graph_json") or "")
                    except json.JSONDecodeError as error:
                        raise ValueError(f"{path}:{row_number}: invalid graph_json") from error
                    if not isinstance(record, dict):
                        raise ValueError(f"{path}:{row_number}: graph_json must be an object")
                    data[name] = {**record, **{key: row[key] for key in ("chemical_name", "state_notes") if row.get(key)}}
        elif path.suffix.lower() == ".xlsx":
            # Reuse the reference workbook reader so training and validation see
            # the same explicit nodes and bond orders. The current GNN's action
            # encoding is H=0, O=1, C=2; callers may override it if needed.
            from .validators.reference import WorkbookGraphs
            codes = {"H": 0, "O": 1, "C": 2} if atom_codes is None else dict(atom_codes)
            workbook = WorkbookGraphs(path, "Graph definitions")
            data = {}
            for item in workbook.records:
                graph = item["graph"]
                nodes = []
                for index, attributes in sorted(graph.nodes(data=True)):
                    symbol = attributes["symbol"]
                    if symbol not in codes:
                        raise ValueError(f"{path}: unsupported atom symbol {symbol!r}")
                    node = {"id": int(index), "atom": codes[symbol], "symbol": symbol}
                    if attributes.get("formal_charge") is not None:
                        node["formal_charge"] = attributes["formal_charge"]
                    nodes.append(node)
                data[item["species_key"]] = {
                    "nodes": nodes,
                    "edges": [{"source": int(u), "target": int(v), "order": attributes["order"]}
                              for u, v, attributes in sorted(graph.edges(data=True))],
                }
        else:
            raise ValueError(f"Species input must be .csv, .json, or .xlsx: {path}")
        if not isinstance(data, dict) or not data:
            raise ValueError(f"{path}: expected a nonempty species-name to graph dictionary")
        return data

    @staticmethod
    def read_negative(path):
        """Negative datasets use records with structure_id and graph, including []."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("Negative input must be a JSON list of species records")
        seen = set()
        for record in data:
            if not isinstance(record, dict):
                raise ValueError("Each negative record must be an object")
            key = record.get("structure_id")
            if not isinstance(key, str) or not key or key in seen or not isinstance(record.get("graph"), dict):
                raise ValueError("Negative records need unique structure_id values and graph objects")
            Molecule.graph(record["graph"])
            seen.add(key)
        return data

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
        self._valid_mask_cache = {}

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
        if action_space in self._valid_mask_cache:
            return self._valid_mask_cache[action_space]
        mask = torch.zeros(len(action_space.actions), dtype=torch.bool)
        for action in self.get_valid_actions():
            mask[action_space.index[action]] = True
        self._valid_mask_cache[action_space] = mask
        return mask

    def step(self, action):
        self._valid_mask_cache.clear()
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
    """Validate file-backed or in-memory species records for the growth model."""

    def __init__(self, path, excluded_species, atom_symbols, max_valency, max_atoms, records=None):
        self.atom_symbols = dict(atom_symbols)
        self.max_valency = dict(max_valency)
        self.max_atoms = max_atoms
        # Isomorphism ignores node numbering but must preserve element and bond order.
        # Formula matching alone would confuse CH3O with CH2OH.
        self.node_match = nx.algorithms.isomorphism.categorical_node_match("atom", None)
        self.edge_match = nx.algorithms.isomorphism.categorical_edge_match("order", None)
        # Record which exact reference file produced this run for later reproducibility.
        from .storage import Storage
        self.source_hash = Storage.digest(path) if records is None else Storage.fingerprint(records)
        codes = {symbol: atom for atom, symbol in atom_symbols.items()}
        data = SpeciesInput.read(path, atom_codes=codes) if records is None else records
        self.graphs = {}
        for name, record in data.items():
            if name in excluded_species:
                continue
            Molecule.require_supported_training(record)
            graph = Molecule.graph(record)
            for node in record["nodes"]:
                atom = node["atom"]
                if type(atom) is not int or atom not in atom_symbols or node.get("symbol") != atom_symbols[atom]:
                    raise ValueError(f"{name}: inconsistent atom encoding")
                graph.nodes[node["id"]]["atom"] = atom
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
    def enumerate_sequences(self, name):
        """Use the common symmetry-pruned enumeration for this molecule."""
        from .trajectories import RouteEnumerator
        from .config import Settings
        graph = self.graphs[name]
        return RouteEnumerator.enumerate(self, name, Settings.ROUTES, graph.number_of_edges()+2)[0]

    def build_demonstrations(self, action_space, route_cache=None, config=None):
        from .trajectories import ReplayCache
        from .config import Settings
        config = config or Settings.INITIAL
        replay = ReplayCache(config["replay_cache_size"])
        total = 0
        for name in self.graphs:
            routes = route_cache.get(self, name) if route_cache else self.enumerate_sequences(name)
            total += len(routes)
            if total > config["max_total_routes"]:
                raise ValueError("Initial training exceeded max_total_routes")
            self.pool[name] = []
            for sequence in routes:
                states, targets = replay.replay(self, name, sequence, action_space, True)
                self.pool[name].append([
                    {"env": env, "condition": self.composition(self.graphs[name]),
                     "action": action, "target_indices": ids}
                    for env, action, ids in zip(states, sequence, targets)])

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


class Molecule:
    """Graph identity with optional electronic metadata; coordinates are not identity.

    Unlike reference matching, deduplication treats unspecified formal charge as
    unknown, not automatically equal to a specified zero charge.
    """

    @staticmethod
    def make_graph(nodes, edges):
        graph = nx.Graph()
        for node in nodes:
            index, symbol = node['id'], node['symbol']
            if type(index) is not int or index < 0 or index in graph or not isinstance(symbol, str):
                raise ValueError('Invalid or duplicate atom')
            charge = node.get('formal_charge')
            if charge is not None and type(charge) is not int:
                raise ValueError('Formal charge must be an integer or None')
            graph.add_node(index, symbol=symbol, formal_charge=charge)
        for edge in edges:
            u,v,order = edge['source'],edge['target'],edge['order']
            if u not in graph or v not in graph or u == v or graph.has_edge(u,v) or type(order) is not int or order not in (1,2,3):
                raise ValueError('Invalid or duplicate bond')
            graph.add_edge(u,v,order=order)
        if not graph or not nx.is_connected(graph):
            raise ValueError('Molecule must be nonempty and connected')
        return graph

    @classmethod
    def graph(cls, record):
        graph = cls.make_graph(record['nodes'], record['edges'])
        charge,spin = record.get('total_charge'),record.get('spin_multiplicity')
        if charge is not None and type(charge) is not int:
            raise ValueError('Total charge must be an integer or None')
        if spin is not None and (type(spin) is not int or spin < 1):
            raise ValueError('Spin multiplicity must be a positive integer or None')
        return graph

    @classmethod
    def same(cls, left, right):
        if any(left.get(k) != right.get(k) for k in ('total_charge','spin_multiplicity')):
            return False
        return nx.is_isomorphic(cls.graph(left), cls.graph(right),
            node_match=nx.algorithms.isomorphism.categorical_node_match(['symbol','formal_charge'],[None,None]),
            edge_match=nx.algorithms.isomorphism.categorical_edge_match('order',None))

    @classmethod
    def key(cls, record):
        graph=cls.graph(record)
        return (tuple(sorted(Counter(d['symbol'] for _,d in graph.nodes(data=True)).items())),
                graph.number_of_edges(), record.get('total_charge'),record.get('spin_multiplicity'))

    @staticmethod
    def identity_payload(record):
        # Ignore coordinates/provenance for graph-only validation cache identity.
        return {'nodes': [{k:n[k] for k in ('id','atom','symbol','formal_charge') if k in n}
                          for n in record['nodes']], 'edges':record['edges'],
                'total_charge':record.get('total_charge'), 'spin_multiplicity':record.get('spin_multiplicity')}

    @classmethod
    def require_supported_training(cls, record):
        cls.graph(record)
        if record.get('total_charge') not in (None,0) or record.get('spin_multiplicity') is not None:
            raise ValueError('Current GNN does not model charge/spin; metadata is preserved but this graph cannot enter training')
        codes={'H':0,'O':1,'C':2}
        if any(n['symbol'] not in codes or n.get('atom') != codes.get(n['symbol']) or n.get('formal_charge') not in (None,0) for n in record['nodes']):
            raise ValueError('Current GNN supports neutral CHO atom encodings only')
