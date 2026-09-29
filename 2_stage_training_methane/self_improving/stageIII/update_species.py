"""Expand known graphs from Stage III and export unvalidated candidates separately.

Run from stageIII: python update_species.py
The expanded file uses Stage I's species-key -> nodes/edges format. This exports
data only; it does not start training or change Stage I's selected input file.
"""

import importlib.util
import json
from pathlib import Path

import networkx as nx


class SpeciesUpdater:
    """Combine existing knowledge with reference-matched discoveries."""

    def __init__(self, known_path, matches_path, output_folder, compact_json_source):
        self.known_path = Path(known_path)
        self.matches_path = Path(matches_path)
        self.output = Path(output_folder)
        # Load only the shared writer, without running the training script.
        spec = importlib.util.spec_from_file_location("species_update_json", compact_json_source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.writer = module.CompactJSON

    @staticmethod
    def make_graph(record):
        """Validate a CHO graph and prepare it for numbering-independent comparison."""
        symbols = {0: "H", 1: "O", 2: "C"}
        graph = nx.Graph()
        for node in record["nodes"]:
            identity, atom = node["id"], node["atom"]
            if type(identity) is not int or identity < 0 or identity in graph:
                raise ValueError(f"Invalid or duplicate atom ID: {identity}")
            if type(atom) is not int or atom not in symbols or node["symbol"] != symbols[atom]:
                raise ValueError(f"Inconsistent CHO atom encoding: {node}")
            graph.add_node(identity, atom=atom, formal_charge=node.get("formal_charge"))
        for edge in record["edges"]:
            u, v, order = edge["source"], edge["target"], edge["order"]
            if u not in graph or v not in graph or u == v or graph.has_edge(u, v):
                raise ValueError(f"Invalid or duplicate bond: {edge}")
            if type(order) is not int or order not in (1, 2, 3):
                raise ValueError(f"Invalid bond order: {edge}")
            graph.add_edge(u, v, order=order)
        if not graph or not nx.is_connected(graph):
            raise ValueError("Species graphs must be nonempty and connected")
        return graph

    @staticmethod
    def same_graph(left, right):
        # Atom numbering and list order do not define different molecules.
        # Missing charge is kept distinct from an explicitly assigned charge.
        return nx.is_isomorphic(
            left, right,
            node_match=nx.algorithms.isomorphism.categorical_node_match(
                ["atom", "formal_charge"], [None, None]),
            edge_match=nx.algorithms.isomorphism.categorical_edge_match("order", None),
        )

    def expand_known(self, known, validated):
        """Keep every original entry; add each distinct validated graph once."""
        expanded = dict(known)
        graphs = {key: self.make_graph(record) for key, record in known.items()}
        messages = []
        added = 0
        for record in validated:
            identity = record["structure_id"]
            aliases = record["ffcmii_species_keys"]
            if not record["in_ffcmii_reference"] or not aliases:
                raise ValueError(f"Validated structure {identity} has no reference match")
            if any(not isinstance(name, str) or not name for name in aliases):
                raise ValueError(f"Invalid reference names for {identity}")
            graph = self.make_graph(record["graph"])
            existing = next((key for key, value in graphs.items()
                             if self.same_graph(graph, value)), None)
            if existing is not None:
                messages.append(f"Already known: {identity} matches {existing}; skipped duplicate graph.")
                continue

            # One reference graph may have multiple electronic-state aliases.
            # Store one training graph and retain all aliases as metadata.
            key = next((name for name in aliases if name not in expanded), None)
            if key is None:
                # Preserve an existing name's graph when the workbook uses a
                # different bond representation (for example HCCO resonance).
                base = f"{aliases[0]}__{identity}"
                key = base
                suffix = 2
                while key in expanded:
                    key = f"{base}_{suffix}"
                    suffix += 1
                messages.append(f"Name conflict: {aliases[0]} kept unchanged; alternative graph saved as {key}.")
            expanded[key] = {
                **record["graph"],
                "reference_species_keys": list(aliases),
                "source_structure_id": identity,
                "electronic_state_resolved": record.get("electronic_state_resolved", False),
                "reference_state_notes": record.get("ffcmii_state_notes", {}),
            }
            graphs[key] = graph
            added += 1
            messages.append(f"Added: {key} from {identity}; reference names={aliases}")
        return expanded, added, messages

    def run(self):
        known = json.loads(self.known_path.read_text(encoding="utf-8"))
        groups = json.loads(self.matches_path.read_text(encoding="utf-8"))
        if not isinstance(known, dict) or not known:
            raise ValueError("Known species must be a nonempty species-key -> graph dictionary")
        if not isinstance(groups, dict) or any(
                not isinstance(groups.get(key), list) for key in ("validated", "unvalidated", "given")):
            raise ValueError("Expected Stage III validated, unvalidated and given lists")
        ids = [record["structure_id"] for key in ("validated", "unvalidated", "given")
               for record in groups[key]]
        if len(ids) != len(set(ids)):
            raise ValueError("A structure ID occurs more than once in Stage III results")

        expanded, added, messages = self.expand_known(known, groups["validated"])
        # Stage III preserves the membership used during the saved Stage II run.
        # A corrected training reference may no longer contain an old blue graph.
        # Report it without reintroducing an obsolete graph into the expanded set
        # or silently relabeling the historical exploration as a new negative.
        known_graphs = [self.make_graph(record) for record in known.values()]
        for record in groups["given"]:
            graph = self.make_graph(record["graph"])
            if not any(self.same_graph(graph, known_graph) for known_graph in known_graphs):
                messages.append(
                    f"Historical given graph {record['structure_id']} ({record['species_keys']}) "
                    "differs from the current known set; not imported or added to unmatched.")
        # These are generated red candidates, not reference species that Stage II
        # failed to recover. Keep their full Stage III records for later review.
        unmatched = groups["unvalidated"]
        for record in unmatched:
            if record["in_ffcmii_reference"]:
                raise ValueError(f"Unvalidated structure {record['structure_id']} has a reference match")
            self.make_graph(record["graph"])

        known_output = self.output / "expanded_species_graphs.json"
        unmatched_output = self.output / "unmatched_species.json"
        inputs = {self.known_path.resolve(), self.matches_path.resolve()}
        if any(path.resolve() in inputs for path in (known_output, unmatched_output)):
            raise ValueError("Output files must be separate from the input files")
        self.output.mkdir(parents=True, exist_ok=True)
        # Use Stage III's readable layout; numeric arrays still stay on one line.
        self.writer.write(known_output, expanded, pack=False)
        self.writer.write(unmatched_output, unmatched, pack=False)
        messages = [
            f"Known input: {self.known_path} ({len(known)} entries)",
            f"Comparison input: {self.matches_path}",
            *messages,
            f"Expanded known set: {len(known)} existing + {added} added = {len(expanded)} graph entries.",
            f"Unmatched candidates: {len(unmatched)}.",
            f"Saved: {known_output}",
            f"Saved: {unmatched_output}",
        ]
        with (self.output / "update_species.log").open("w", encoding="utf-8") as log:
            for message in messages:
                print(message, flush=True)
                log.write(message + "\n")
        return expanded, unmatched


# ==========================================
# PARAMETERS — paths relative to the working directory
# ==========================================
KNOWN_SPECIES_PATH = "../stageI/species_graphs.json"
MATCHES_PATH = "output/ffcmii_matches.json"
OUTPUT_FOLDER = "species_update"
COMPACT_JSON_SOURCE = "../stageI/compact_json.py"

# ==========================================
# RUN — direct calls, no main guard
# ==========================================
updater = SpeciesUpdater(KNOWN_SPECIES_PATH, MATCHES_PATH, OUTPUT_FOLDER, COMPACT_JSON_SOURCE)
updater.run()
