"""Build the active version-4 reference graphs and export them to JSON.

Run this script to write species_graphs.json beside it. CO is excluded for now.
Each species stores all atoms (including H) and undirected bonds once.
The atom codes match version 4: H=0, O=1, C=2.
"""

from pathlib import Path

import networkx as nx

from compact_json import CompactJSON


class SpeciesGraphBuilder:
    """Build molecular reference graphs and save them as JSON."""

    def __init__(self, atom_h, atom_symbols):
        self.atom_h = atom_h
        self.atom_symbols = dict(atom_symbols)

    def make_species_graph(self, heavy_atoms, heavy_bonds=(), hydrogen_counts=()):
        """Create a reference graph from heavy atoms, bonds, and attached H counts."""
        # Each graph represents one molecule: atoms are nodes and bonds are edges.
        graph = nx.Graph()  # creates an empty undirected graph to represent the molecule

        # Add the heavy atoms (carbon or oxygen) and store their atom type.
        for index, atom_type in enumerate(heavy_atoms):
            graph.add_node(index, atom=atom_type)

        # Add bonds between heavy atoms. Each tuple is (atom 1, atom 2, bond order).
        for u, v, order in heavy_bonds:
            graph.add_edge(u, v, order=order)

        # Number hydrogen nodes after all the heavy-atom nodes.
        next_index = len(heavy_atoms)

        # Attach the specified number of hydrogens to each heavy atom.
        for heavy_index, count in enumerate(hydrogen_counts):
            for _ in range(count):
                graph.add_node(next_index, atom=self.atom_h)
                # Hydrogen is always attached with a single bond.
                graph.add_edge(heavy_index, next_index, order=1)
                next_index += 1

        # Return the completed molecular reference graph.
        return graph

    def save_species_graphs_json(self, species_graphs, output_path):
        """Save every SPECIES_GRAPHS entry with its node IDs, atom types and bonds."""
        species_data = {}
        for species_key, graph in species_graphs.items():
            species_data[species_key] = {
                "nodes": [
                    {
                        "id": node_id,
                        "atom": attributes["atom"],
                        "symbol": self.atom_symbols[attributes["atom"]],
                    }
                    for node_id, attributes in sorted(graph.nodes(data=True))
                ],
                "edges": [
                    {"source": u, "target": v, "order": order}
                    for u, v, order in sorted(
                        (min(u, v), max(u, v), attributes["order"])
                        for u, v, attributes in graph.edges(data=True)
                    )
                ],
            }

        # Share the same compact format as training examples and generated inventories.
        return CompactJSON.write(output_path, species_data)

# ==========================================
# PARAMETERS
# ==========================================
# Atom encoding matches grow_train_animation_v4_1.py.
ATOM_H = 0
ATOM_O = 1
ATOM_C = 2
ATOM_SYMBOL = {ATOM_H: "H", ATOM_O: "O", ATOM_C: "C"}   # equivalent to ATOM_SYMBOL = {0: "H", 1: "O", 2: "C"}

SPECIES_JSON_PATH = "species_graphs.json"

# ==========================================
# BUILD REFERENCE GRAPHS
# ==========================================
graph_builder = SpeciesGraphBuilder(atom_h=ATOM_H, atom_symbols=ATOM_SYMBOL)
# Radicals are represented by atoms and bonds; unpaired electrons are implied
# by unsatisfied valency. CO is excluded for now.
SPECIES_GRAPHS = {
    # H2/O2 chemistry
    "H2": graph_builder.make_species_graph([ATOM_H, ATOM_H], [(0, 1, 1)]),
    "O2": graph_builder.make_species_graph([ATOM_O, ATOM_O], [(0, 1, 2)]),
    "H": graph_builder.make_species_graph([ATOM_H]),
    "O": graph_builder.make_species_graph([ATOM_O]),
    "OH": graph_builder.make_species_graph([ATOM_O], hydrogen_counts=[1]),
    "H2O": graph_builder.make_species_graph([ATOM_O], hydrogen_counts=[2]),
    "HO2": graph_builder.make_species_graph([ATOM_O, ATOM_O], [(0, 1, 1)], [1, 0]),
    "H2O2": graph_builder.make_species_graph([ATOM_O, ATOM_O], [(0, 1, 1)], [1, 1]),

    # CO2 chemistry (CO is excluded for now.)
    "CO2": graph_builder.make_species_graph([ATOM_C, ATOM_O, ATOM_O],
                              [(0, 1, 2), (0, 2, 2)], [0, 0, 0]),
    "HCO": graph_builder.make_species_graph([ATOM_C, ATOM_O], [(0, 1, 2)], [1, 0]),
    "CH2O": graph_builder.make_species_graph([ATOM_C, ATOM_O], [(0, 1, 2)], [2, 0]),

    # C1 chemistry
    "CH4": graph_builder.make_species_graph([ATOM_C], hydrogen_counts=[4]),
    "CH3": graph_builder.make_species_graph([ATOM_C], hydrogen_counts=[3]),
    "CH3O2": graph_builder.make_species_graph([ATOM_C, ATOM_O, ATOM_O],
                                [(0, 1, 1), (1, 2, 1)], [3, 0, 0]),
    "CH2": graph_builder.make_species_graph([ATOM_C], hydrogen_counts=[2]),
    "CH": graph_builder.make_species_graph([ATOM_C], hydrogen_counts=[1]),
    "C": graph_builder.make_species_graph([ATOM_C]),
    "CH3O": graph_builder.make_species_graph([ATOM_C, ATOM_O], [(0, 1, 1)], [3, 0]),
    "CH2OH": graph_builder.make_species_graph([ATOM_C, ATOM_O], [(0, 1, 1)], [2, 1]),
    "CH3OH": graph_builder.make_species_graph([ATOM_C, ATOM_O], [(0, 1, 1)], [3, 1]),

    # C2 chemistry
    "C2H6": graph_builder.make_species_graph([ATOM_C, ATOM_C], [(0, 1, 1)], [3, 3]),
    "C2H5": graph_builder.make_species_graph([ATOM_C, ATOM_C], [(0, 1, 1)], [3, 2]),
    "C2H4": graph_builder.make_species_graph([ATOM_C, ATOM_C], [(0, 1, 2)], [2, 2]),
    "C2H3": graph_builder.make_species_graph([ATOM_C, ATOM_C], [(0, 1, 2)], [2, 1]),
    "C2H2": graph_builder.make_species_graph([ATOM_C, ATOM_C], [(0, 1, 3)], [1, 1]),
    "C2H": graph_builder.make_species_graph([ATOM_C, ATOM_C], [(0, 1, 3)], [1, 0]),
    "HCCO": graph_builder.make_species_graph([ATOM_C, ATOM_C, ATOM_O],
                               [(0, 1, 2), (1, 2, 2)], [1, 0, 0]),
    "CH2CO": graph_builder.make_species_graph([ATOM_C, ATOM_C, ATOM_O],
                                [(0, 1, 2), (1, 2, 2)], [2, 0, 0]),
    "CH2CHO": graph_builder.make_species_graph([ATOM_C, ATOM_C, ATOM_O],
                                 [(0, 1, 1), (1, 2, 2)], [2, 1, 0]),
    "CH3CHO": graph_builder.make_species_graph([ATOM_C, ATOM_C, ATOM_O],
                                 [(0, 1, 1), (1, 2, 2)], [3, 1, 0]),
    "CH3CO": graph_builder.make_species_graph([ATOM_C, ATOM_C, ATOM_O],
                                [(0, 1, 1), (1, 2, 2)], [3, 0, 0]),
}

saved_path = graph_builder.save_species_graphs_json(
    species_graphs=SPECIES_GRAPHS,
    output_path=SPECIES_JSON_PATH,
)
print(f"Saved {len(SPECIES_GRAPHS)} species graphs to {saved_path}")
