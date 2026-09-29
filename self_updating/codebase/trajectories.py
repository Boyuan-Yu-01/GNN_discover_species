"""Exhaustive route enumeration, replay and per-species sampling."""
import gzip
import hashlib
import json
import math
from pathlib import Path
from collections import OrderedDict
import networkx as nx
from .molecule import ReferenceSpecies, AdvancedMoleculeEnv, SpeciesInput
from .storage import Storage

class RoutePool:
    """Keep exhaustive routes on disk; sample indices uniformly without replacement."""

    def __init__(self, path, count):
        self.path, self.count = Path(path), count

    def __len__(self):
        return self.count

    def __iter__(self):
        with gzip.open(self.path, "rt", encoding="utf-8") as handle:
            for line in handle:
                yield tuple(tuple(action) for action in json.loads(line))

    def sample(self, rng, count):
        selected = set(rng.sample(range(self.count), count))
        with gzip.open(self.path, "rt", encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if index in selected:
                    yield tuple(tuple(action) for action in json.loads(line))


class RouteEnumerator:
    @staticmethod
    def enumerate(reference, name, config, max_steps):
        """Stage I's finite route family: introduce each final bond directly.

        This excludes optional intermediate bond-order upgrades, as Stage I does.
        Full-reference automorphisms remove interchangeable reference-node choices
        with identical futures, while retaining every distinct action sequence.
        Limits raise an error; the route pool is never silently truncated.
        """
        graph = reference.graphs[name]
        if graph.number_of_edges() + 2 > max_steps:
            raise ValueError(f"{name}: complete routes exceed the checkpoint action limit")
        matcher = nx.algorithms.isomorphism.GraphMatcher(
            graph, graph, node_match=reference.node_match, edge_match=reference.edge_match)
        automorphisms = []
        for mapping in matcher.isomorphisms_iter():
            automorphisms.append(mapping)
            if len(automorphisms) > config["max_enumeration_states_per_graph"]:
                raise ValueError(f"{name}: too many graph symmetries for the enumeration budget")
        sequences, visits = set(), 0

        def visit(mapping, built, actions, symmetries):
            nonlocal visits
            visits += 1
            if visits > config["max_enumeration_states_per_graph"]:
                raise ValueError(f"{name}: enumeration exceeded MAX_ENUMERATION_STATES_PER_GRAPH; increase it explicitly")
            choices = []
            if not mapping:
                choices = [(("START", graph.nodes[u]["atom"]), u, None) for u in graph]
            else:
                for u, v, edge in graph.edges(data=True):
                    bond = frozenset((u, v))
                    if u in mapping and v not in mapping:
                        choices.append((("GROW", mapping[u], graph.nodes[v]["atom"], edge["order"]), v, bond))
                    elif v in mapping and u not in mapping:
                        choices.append((("GROW", mapping[v], graph.nodes[u]["atom"], edge["order"]), u, bond))
                    elif u in mapping and v in mapping and bond not in built:
                        a, b = sorted((mapping[u], mapping[v]))
                        choices.append((("CONNECT", a, b, edge["order"]), None, bond))
            if not choices:
                if len(mapping) != len(graph):
                    raise ValueError(f"{name}: incomplete route")
                sequences.add(actions + (("STOP",),))
                if len(sequences) > config["max_routes_per_graph"]:
                    raise ValueError(f"{name}: enumeration exceeded MAX_ROUTES_PER_GRAPH; increase it explicitly")
                return
            seen = set()
            for action, new_node, bond in choices:
                orbit = min(m[new_node] for m in symmetries) if new_node is not None else None
                key = action, orbit
                if key in seen:
                    continue
                seen.add(key)
                next_mapping = dict(mapping)
                stabilizer = symmetries
                if new_node is not None:
                    next_mapping[new_node] = len(mapping)
                    stabilizer = [m for m in symmetries if m[new_node] == new_node]
                visit(next_mapping, built | {bond} if bond is not None else built,
                      actions + (action,), stabilizer)

        visit({}, set(), (), automorphisms)
        return sorted(sequences), {'visits': visits, 'automorphisms': len(automorphisms)}


class RouteCache:
    """Reuse routes by exact graph/action configuration, without caching learned features."""
    VERSION = 1

    def __init__(self, folder, config, trained):
        self.folder = Path(folder)
        self.config, self.trained = config, trained
        self.folder.mkdir(parents=True, exist_ok=True)

    def get(self, reference, name):
        graph = reference.graphs[name]
        payload = {"version": self.VERSION,
                   "nodes": [(u, d["atom"]) for u, d in graph.nodes(data=True)],
                   "edges": [(u, v, d["order"]) for u, v, d in graph.edges(data=True)],
                   "atom_symbols": reference.atom_symbols, "max_valency": reference.max_valency,
                   "max_atoms": reference.max_atoms, "max_steps": self.trained["max_steps"]}
        key = Storage.fingerprint(payload)
        path, metadata = self.folder/(key+".jsonl.gz"), self.folder/(key+".json")
        if path.exists() and metadata.exists():
            info = Storage.read(metadata)
            if info["sha256"] != Storage.digest(path):
                raise ValueError(f"Route cache changed: {path}")
            if (info["count"] > self.config["max_routes_per_graph"]
                    or max(info["visits"], info["automorphisms"]) > self.config["max_enumeration_states_per_graph"]):
                raise ValueError(f"{name}: cached routes exceed configured enumeration limits")
            return RoutePool(path, info["count"])
        routes, info = RouteEnumerator.enumerate(reference, name, self.config, self.trained["max_steps"])
        # The manifest is written only after gzip closes; interrupted writes are rebuilt.
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            for route in routes:
                handle.write(json.dumps(route, separators=(",", ":")) + "\n")
        Storage.write(metadata, {**info, "count": len(routes), "sha256": Storage.digest(path)})
        return RoutePool(path, len(routes))


class ReplayCache:
    """Bounded cache of immutable CPU graph states and targets, never model outputs."""

    def __init__(self, size=2048):
        self.size = size
        self.entries = OrderedDict()

    def replay(self, reference, name, sequence, space, equivalent):
        key = (id(reference), name, tuple(sequence), id(space), equivalent)
        if key in self.entries:
            self.entries.move_to_end(key)
            return self.entries[key]
        env = AdvancedMoleculeEnv(reference.atom_symbols, reference.max_valency, reference.max_atoms)
        states, targets = [], []
        for action in sequence:
            options = reference.equivalent_actions(env, action) if equivalent else [action]
            legal = env.get_valid_actions()
            if any(a not in legal for a in options):
                raise ValueError(f"{name}: illegal route action")
            states.append(env.copy())
            targets.append([space.index[a] for a in options])
            env.step(action)
        if not env.terminated or not reference.isomorphic(env.to_graph(), reference.graphs[name]):
            raise ValueError(f"{name}: route does not reconstruct its graph")
        self.entries[key] = (states, targets)
        if len(self.entries) > self.size:
            self.entries.popitem(last=False)
        return states, targets


class TrajectoryDataset:
    """Validate both sets; preserve one graph per topology and let positives win."""

    def __init__(self, config, trained):
        self.config, self.trained = config, trained
        self.hashes = {key: hashlib.sha256(Path(config[key]).read_bytes()).hexdigest()
                       for key in ("positive_path", "negative_path")}
        raw_positive = json.loads(Path(config["positive_path"]).read_text())
        raw_negative = SpeciesInput.read_negative(config["negative_path"])
        self.records = {"positive": raw_positive, "negative": raw_negative}
        self.replay_cache = ReplayCache(config["replay_cache_size"])
        if not isinstance(raw_positive, dict) or not raw_positive or not isinstance(raw_negative, list):
            raise ValueError("Expected positive species-key -> graph dictionary and negative record list")
        negatives = {}
        for record in raw_negative:
            key = record["structure_id"]
            if not isinstance(key, str) or not key or key in negatives:
                raise ValueError(f"Invalid/duplicate negative structure ID: {key}")
            negatives[key] = record["graph"]
        # The current GNN has no charge feature. Do not silently train charged graphs.
        for label, records in (("positive", raw_positive), ("negative", negatives)):
            for key, record in records.items():
                if label == "positive" and key in trained["excluded_species"]:
                    continue
                for node in record["nodes"]:
                    if type(node["atom"]) is not int:
                        raise ValueError(f"{label} {key}: atom type must be an integer")
                    charge = node.get("formal_charge")
                    if charge is not None and (type(charge) is not int or charge != 0):
                        raise ValueError(f"{label} {key}: charged graphs are unsupported by the parent model")
        self.positive = self.references(raw_positive, trained["excluded_species"])
        self.negative = self.references(negatives, []) if negatives else None
        self.removed = {"positive_duplicates": {}, "negative_duplicates": {}, "positive_negative_overlap": []}
        self.deduplicate(self.positive, "positive_duplicates")
        if self.negative:
            self.deduplicate(self.negative, "negative_duplicates")
            for name, graph in list(self.negative.graphs.items()):
                if self.positive.match(graph) is not None:
                    self.removed["positive_negative_overlap"].append(name)
                    del self.negative.graphs[name]
            self.reindex(self.negative)
            if not self.negative.graphs:
                self.negative = None
        self.positive.source_hash = self.hashes["positive_path"]
        self.sequences = {"positive": {}, "negative": {}}

    def references(self, records, excluded):
        return ReferenceSpecies(
            None, excluded, self.trained["atom_symbols"], self.trained["max_valency"],
            self.trained["max_atoms"], records=records)

    @staticmethod
    def reindex(reference):
        reference.conditions = {}
        for name, graph in reference.graphs.items():
            reference.conditions.setdefault(reference.composition(graph), []).append(name)

    def deduplicate(self, reference, label):
        kept = []
        for name, graph in list(reference.graphs.items()):
            duplicate = next((key for key, other in kept if reference.isomorphic(graph, other)), None)
            if duplicate is None:
                kept.append((name, graph))
            else:
                self.removed[label][name] = duplicate
                del reference.graphs[name]
        self.reindex(reference)

    def enumerate_routes(self, reference, name):
        return RouteEnumerator.enumerate(reference, name, self.config, self.trained["max_steps"])[0]

    def prepare(self, log, include_negative=True):
        total = 0
        cache = RouteCache(self.config.get("route_cache_folder") or
                           Path(self.config["output_folder"]) / "route_cache", self.config, self.trained)
        for label, reference in (("positive", self.positive), ("negative", self.negative)):
            if reference is None or (label == "negative" and not include_negative):
                continue
            fraction = self.config[label + "_trajectory_fraction"]
            names=list(reference.graphs)
            if label == 'negative' and self.config.get('hard_negative_ids') is not None:
                names=self.config['hard_negative_ids']
                missing=set(names)-set(reference.graphs)
                if missing:
                    raise ValueError(f'Hard-negative IDs are absent from the negative dataset: {sorted(missing)}')
            for index, name in enumerate(names):
                routes = cache.get(reference, name)
                total += len(routes)
                if total > self.config["max_total_routes"]:
                    raise ValueError(f"{label} {name}: exceeded MAX_TOTAL_ROUTES; increase the limit explicitly")
                self.sequences[label][name] = routes
                log.info(f"ROUTES {label} {name} | total={len(routes)}"
                         f" | sampled when species selected={math.ceil(len(routes) * fraction)}")

    def replay(self, reference, name, sequence, space, equivalent):
        return self.replay_cache.replay(reference, name, sequence, space, equivalent)

    def summary(self):
        return {"source_sha256": self.hashes, "positive_graphs": len(self.positive.graphs),
                "negative_graphs": len(self.negative.graphs) if self.negative else 0,
                "removed": self.removed,
                "enumeration_scope": "Stage I direct-final-bond action sequences, including STOP",
                "negative_label": "Invalid for this training round; reference absence is not chemical proof",
                "hard_negative_ids": self.config.get("hard_negative_ids"),
                "negative_sampling": "All pipeline-selected hard negatives; routes sampled without replacement each epoch",
                "routes": {label: {name: {"total": len(routes), "sampled_when_species_selected":
                    math.ceil(len(routes) * self.config[label + "_trajectory_fraction"])}
                    for name, routes in pool.items()} for label, pool in self.sequences.items()}}
