import csv
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx   # python library for creating, analyzing, and comparing graphs
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# Universe Configuration
ATOM_H = 0
ATOM_O = 1
ATOM_C = 2
ATOM_SYMBOL = {ATOM_H: "H", ATOM_O: "O", ATOM_C: "C"}
MAX_VALENCY = {ATOM_H: 1, ATOM_O: 2, ATOM_C: 4}
RECOGNIZED_MOLECULE_REWARD = 1.0
DUPLICATE_MOLECULE_REWARD = -0.5
UNRECOGNIZED_MOLECULE_REWARD = -0.5
ISOLATED_ATOM_REWARD = -0.5
SEED = 12345
MAX_STEPS_PER_EPOCH = 500
# INITIAL_ATOMS = [ATOM_C, ATOM_H, ATOM_H, ATOM_H, ATOM_H, ATOM_O, ATOM_O]
INITIAL_ATOMS = [
    ATOM_C, ATOM_C, ATOM_C, ATOM_C, ATOM_C, ATOM_C, ATOM_C, ATOM_C, ATOM_C,
    ATOM_C, ATOM_C, ATOM_C, ATOM_C, ATOM_C, ATOM_C, ATOM_C, ATOM_C, ATOM_C,
    ATOM_H, ATOM_H, ATOM_H, ATOM_H, ATOM_H, ATOM_H,ATOM_H, ATOM_H, ATOM_H,
    ATOM_H, ATOM_H, ATOM_H, ATOM_H, ATOM_H, ATOM_H, ATOM_H, ATOM_H, ATOM_H,
    ATOM_H, ATOM_H, ATOM_H, ATOM_H, ATOM_H, ATOM_H, ATOM_H, ATOM_H, ATOM_H,
    ATOM_H, ATOM_H, ATOM_H, ATOM_H, ATOM_H, ATOM_H, ATOM_H, ATOM_H, ATOM_H,
    ATOM_H, ATOM_H, ATOM_H, ATOM_H, ATOM_H, ATOM_H, ATOM_H, ATOM_H, ATOM_H,
    ATOM_H, ATOM_H, ATOM_H, ATOM_H, ATOM_H, ATOM_H, ATOM_H, ATOM_H, ATOM_H,
    ATOM_H, ATOM_H, ATOM_H, ATOM_H, ATOM_H, ATOM_H, ATOM_H, ATOM_H, ATOM_H,
    ATOM_H, ATOM_H, ATOM_H, ATOM_H, ATOM_H, ATOM_H, ATOM_H, ATOM_H, ATOM_H,
    ATOM_O, ATOM_O, ATOM_O, ATOM_O, ATOM_O, ATOM_O, ATOM_O, ATOM_O, ATOM_O,
]

# Neural Network Configuration
GNN_HIDDEN_DIMS = [32, 64, 32, 32, 16]

# Save every generated plot beside this script, regardless of the launch folder.
SCRIPT_FOLDER = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FOLDER = os.path.join(SCRIPT_FOLDER, "output_v4_1")
INDIVIDUAL_PLOTS_FOLDER = os.path.join(OUTPUT_FOLDER, "individual_plots")
GIF_PATH = os.path.join(OUTPUT_FOLDER, "species_growth.gif")
LOG_PATH = os.path.join(SCRIPT_FOLDER, "log_v4_1.txt")
RESTART_FOLDER = os.path.join(OUTPUT_FOLDER, "restart_files")
MODEL_PATH = os.path.join(OUTPUT_FOLDER, "trained_growth_gnn.pt")
# CSV opens in Excel without extra packages; .xlsx requires openpyxl.
SPECIES_INVENTORY_PATH = os.path.join(OUTPUT_FOLDER, "generated_species_inventory.csv")


class Tee:
    # Send output to both the terminal and the log file.
    def __init__(self, terminal, log):
        self.terminal = terminal
        self.log = log

    # print() calls this method to write text.
    def write(self, message):
        # write sends text to both terminal output and the log file
        self.terminal.write(message)
        self.log.write(message)

    # Make sure buffered output is written immediately.
    def flush(self):
        # flush ensures that both the terminal and log file are updated immediately
        # Without flushing, Python may hold output in memory briefly before displaying or saving it.
        self.terminal.flush()
        self.log.flush()


def save_restart_file(epoch, model, optimizer, reward, config):
    """Save everything needed to continue training after a completed epoch."""
    os.makedirs(RESTART_FOLDER, exist_ok=True)
    restart_path = os.path.join(RESTART_FOLDER, f"restart_epoch_{epoch:03d}.pt")
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "reward": reward,
        "config": config,
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
    }, restart_path)
    return restart_path


def save_neural_network(model, config):
    """Save the final trained GNN weights and architecture settings."""
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config,
    }, MODEL_PATH)
    return MODEL_PATH


def delete_restart_files():
    """Delete per-epoch restart files after the full run succeeds."""
    deleted_count = 0
    if os.path.isdir(RESTART_FOLDER):
        for entry in os.scandir(RESTART_FOLDER):
            if (entry.is_file() and entry.name.startswith("restart_epoch_")
                    and entry.name.endswith(".pt")):
                os.remove(entry.path)
                deleted_count += 1
    return deleted_count


def calculate_discounted_returns(terminal_reward, num_actions, discount_factor):
    """Return G_t = gamma^(T-1-t) * R_terminal for every action."""
    return torch.tensor([
        terminal_reward * discount_factor ** (num_actions - 1 - t)
        for t in range(num_actions)
    ], dtype=torch.float)


def make_species_graph(heavy_atoms, heavy_bonds=(), hydrogen_counts=()):
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
            graph.add_node(next_index, atom=ATOM_H)
            # Hydrogen is always attached with a single bond.
            graph.add_edge(heavy_index, next_index, order=1)
            next_index += 1

    # Return the completed molecular reference graph.
    return graph


# Reference structures used during training. Radicals are represented by their
# atoms and bonds; unpaired electrons are implied by the unsatisfied valency.
SPECIES_GRAPHS = {
    # H2/O2 chemistry
    "H2": make_species_graph([ATOM_H, ATOM_H], [(0, 1, 1)]),
    "O2": make_species_graph([ATOM_O, ATOM_O], [(0, 1, 2)]),
    "H": make_species_graph([ATOM_H]),
    "O": make_species_graph([ATOM_O]),
    "OH": make_species_graph([ATOM_O], hydrogen_counts=[1]),
    "H2O": make_species_graph([ATOM_O], hydrogen_counts=[2]),
    "HO2": make_species_graph([ATOM_O, ATOM_O], [(0, 1, 1)], [1, 0]),
    "H2O2": make_species_graph([ATOM_O, ATOM_O], [(0, 1, 1)], [1, 1]),

    # CO/CO2 chemistry
    "CO": make_species_graph([ATOM_C, ATOM_O], [(0, 1, 3)], [0, 0]),
    "CO2": make_species_graph([ATOM_C, ATOM_O, ATOM_O],
                              [(0, 1, 2), (0, 2, 2)], [0, 0, 0]),
    "HCO": make_species_graph([ATOM_C, ATOM_O], [(0, 1, 2)], [1, 0]),
    "CH2O": make_species_graph([ATOM_C, ATOM_O], [(0, 1, 2)], [2, 0]),

    # C1 chemistry
    "CH4": make_species_graph([ATOM_C], hydrogen_counts=[4]),
    "CH3": make_species_graph([ATOM_C], hydrogen_counts=[3]),
    "CH3O2": make_species_graph([ATOM_C, ATOM_O, ATOM_O],
                                [(0, 1, 1), (1, 2, 1)], [3, 0, 0]),
    "CH2": make_species_graph([ATOM_C], hydrogen_counts=[2]),
    "CH": make_species_graph([ATOM_C], hydrogen_counts=[1]),
    "C": make_species_graph([ATOM_C]),
    "CH3O": make_species_graph([ATOM_C, ATOM_O], [(0, 1, 1)], [3, 0]),
    "CH2OH": make_species_graph([ATOM_C, ATOM_O], [(0, 1, 1)], [2, 1]),
    "CH3OH": make_species_graph([ATOM_C, ATOM_O], [(0, 1, 1)], [3, 1]),

    # C2 chemistry
    "C2H6": make_species_graph([ATOM_C, ATOM_C], [(0, 1, 1)], [3, 3]),
    "C2H5": make_species_graph([ATOM_C, ATOM_C], [(0, 1, 1)], [3, 2]),
    "C2H4": make_species_graph([ATOM_C, ATOM_C], [(0, 1, 2)], [2, 2]),
    "C2H3": make_species_graph([ATOM_C, ATOM_C], [(0, 1, 2)], [2, 1]),
    "C2H2": make_species_graph([ATOM_C, ATOM_C], [(0, 1, 3)], [1, 1]),
    "C2H": make_species_graph([ATOM_C, ATOM_C], [(0, 1, 3)], [1, 0]),
    "HCCO": make_species_graph([ATOM_C, ATOM_C, ATOM_O],
                               [(0, 1, 2), (1, 2, 2)], [1, 0, 0]),
    "CH2CO": make_species_graph([ATOM_C, ATOM_C, ATOM_O],
                                [(0, 1, 2), (1, 2, 2)], [2, 0, 0]),
    "CH2CHO": make_species_graph([ATOM_C, ATOM_C, ATOM_O],
                                 [(0, 1, 1), (1, 2, 2)], [2, 1, 0]),
    "CH3CHO": make_species_graph([ATOM_C, ATOM_C, ATOM_O],
                                 [(0, 1, 1), (1, 2, 2)], [3, 1, 0]),
    "CH3CO": make_species_graph([ATOM_C, ATOM_C, ATOM_O],
                                [(0, 1, 1), (1, 2, 2)], [3, 0, 0]),
}

VALID_SPECIES = set(SPECIES_GRAPHS)
NODE_MATCH = nx.algorithms.isomorphism.categorical_node_match("atom", None)     # compare two graph nodes using their "atom" attribute
EDGE_MATCH = nx.algorithms.isomorphism.categorical_edge_match("order", None)    # compare two graph edges using their "order" attribute


def get_species_inventory(env, epoch):
    """Describe each final connected molecule using the reference-table fields.

    Atom indices are local to each molecule. Recognized structures use the
    reference graph's ordering; unknown structures use sorted environment IDs.
    Pure-hydrogen species retain H as base atoms, as in SPECIES_GRAPHS.
    """
    inventory_graph = nx.Graph()
    for local_index, atom_type in enumerate(env.node_types):
        inventory_graph.add_node(env.offset + local_index, atom=atom_type)
    for (u, v), order in env.bonds.items():
        inventory_graph.add_edge(u, v, order=order)

    rows = []
    components = sorted(nx.connected_components(inventory_graph), key=min)
    for molecule_number, nodes in enumerate(components, start=1):
        structure = inventory_graph.subgraph(nodes)
        species_key = None
        for name, reference in SPECIES_GRAPHS.items():
            if nx.is_isomorphic(structure, reference,
                                node_match=NODE_MATCH, edge_match=EDGE_MATCH):
                species_key = name
                structure = reference
                break
        in_reference = species_key is not None

        if species_key is None:
            counts = {
                symbol: sum(data["atom"] == atom_type
                            for _, data in structure.nodes(data=True))
                for atom_type, symbol in ATOM_SYMBOL.items()
            }
            formula = "".join(
                symbol + (str(counts[symbol]) if counts[symbol] > 1 else "")
                for symbol in ("C", "H", "O") if counts[symbol]
            )
            species_key = f"Unknown: {formula}"

        base_nodes = sorted(
            node for node, data in structure.nodes(data=True)
            if data["atom"] != ATOM_H
        )
        hydrogen_only = not base_nodes
        if hydrogen_only:
            base_nodes = sorted(structure.nodes)
        base_index = {node: index for index, node in enumerate(base_nodes)}
        base_atoms = {
            index: ATOM_SYMBOL[structure.nodes[node]["atom"]]
            for node, index in base_index.items()
        }
        base_bonds = sorted(
            (min(base_index[u], base_index[v]),
             max(base_index[u], base_index[v]), data["order"])
            for u, v, data in structure.edges(data=True)
            if u in base_index and v in base_index
        )
        attached_h = [] if hydrogen_only else [
            sum(structure.nodes[neighbor]["atom"] == ATOM_H
                for neighbor in structure.neighbors(node))
            for node in base_nodes
        ]
        rows.append({
            "epoch": epoch,
            "molecule_number": molecule_number,
            "species_key": species_key,
            "base_atom_indices": base_atoms,
            "base_bonds": base_bonds,
            "attached_h_per_base_atom": attached_h,
            "in_species_training_reference": in_reference,
        })
    return rows


def save_species_inventory(env, epoch, output_path=None, append=False):
    """Export final molecules to CSV (default) or XLSX, one row per molecule.

    Set append=True to add another epoch to an existing inventory. Unknown
    molecules can share a formula; epoch and molecule number identify each row.
    Reference membership is checked against SPECIES_GRAPHS, the source of the
    species training reference workbook, rather than the workbook on Desktop.
    XLSX export requires openpyxl; CSV uses only Python's standard library.
    """
    output_path = os.fspath(
        SPECIES_INVENTORY_PATH if output_path is None else output_path
    )
    extension = os.path.splitext(output_path)[1].lower()
    if extension not in (".csv", ".xlsx"):
        raise ValueError("Species inventory output must end in .csv or .xlsx")
    headers = [
        "Epoch", "Molecule number", "Species key", "Base atom indices",
        "Base bonds (u, v, order)", "Attached H per base atom",
        "In species training reference",
    ]
    records = get_species_inventory(env, epoch)
    rows = [
        [
            record["epoch"], record["molecule_number"], record["species_key"],
            repr(record["base_atom_indices"]), repr(record["base_bonds"]),
            repr(record["attached_h_per_base_atom"]),
            "Yes" if record["in_species_training_reference"] else "No",
        ]
        for record in records
    ]
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    existing = append and os.path.isfile(output_path) and os.path.getsize(output_path) > 0

    if extension == ".csv":
        if existing:
            with open(output_path, newline="", encoding="utf-8-sig") as source:
                if next(csv.reader(source), None) != headers:
                    raise ValueError("Existing species inventory has different columns")
        with open(output_path, "a" if existing else "w",
                  newline="", encoding="utf-8" if existing else "utf-8-sig") as target:
            writer = csv.writer(target)
            if not existing:
                writer.writerow(headers)
            writer.writerows(rows)
    else:
        try:
            from openpyxl import Workbook, load_workbook
            from openpyxl.styles import Alignment, Font, PatternFill
        except ImportError as exc:
            raise ImportError(
                "XLSX export requires openpyxl in your training environment. "
                "Install it with python -m pip install openpyxl, or use a .csv path."
            ) from exc
        workbook = load_workbook(output_path) if existing else Workbook()
        try:
            sheet = workbook.active
            if existing:
                if [cell.value for cell in sheet[1]] != headers:
                    raise ValueError("Existing species inventory has different columns")
            else:
                sheet.title = "Generated species"
                sheet.append(headers)
                sheet.freeze_panes = "D2"
                sheet.sheet_view.showGridLines = False
                for column, width in zip("ABCDEFG", (10, 18, 26, 40, 55, 32, 32)):
                    sheet.column_dimensions[column].width = width
                for cell in sheet[1]:
                    cell.font = Font(name="Arial", bold=True, color="FFFFFF")
                    cell.fill = PatternFill("solid", fgColor="203B56")
                    cell.alignment = Alignment(wrap_text=True, vertical="center")
                sheet.row_dimensions[1].height = 32
            for row in rows:
                sheet.append(row)
                for cell in sheet[sheet.max_row]:
                    cell.alignment = Alignment(wrap_text=True, vertical="top")
            sheet.auto_filter.ref = sheet.dimensions
            workbook.save(output_path)
        finally:
            workbook.close()
    return output_path

# ==========================================
# MOLECULE ENVIRONMENT (UNION-FIND TRACKING)
# ==========================================
class AdvancedMoleculeEnv:
    def __init__(self, node_offset=0):  # node_offset allows multiple environments to share a global graph without index collisions
        self.node_types = INITIAL_ATOMS.copy()  # creates a separate copy of the initial atom list
        self.current_bonds = [0] * len(self.node_types)
        # Changed from list to dict: {(u, v): bond_order}
        self.bonds = {}             # stores bonds in a dictionary: {(0,1): 1, (0,5):2, }
        self.success = False        # record whether the environment considers its current operation successful
        self.terminated = False     # record whether the environment has reached a terminal state (no further actions possible)
        self.offset = node_offset
        self.node_to_species = {
            self.offset + i: {self.offset + i} for i in range(len(self.node_types))
        }
        # Initially, every atom is separate. With an offset of 20, it produces:
        # {
        #     20: {20},
        #     21: {21},
        #     22: {22},
        #     23: {23},
        #     24: {24},
        #     25: {25},
        #     26: {26},
        # }
        # If nodes 20 and 21 are bonded, both entries are updated to reference the same component:
        # {
        #     20: {20, 21},
        #     21: {20, 21},
        #     22: {22},
        #     23: {23},
        #     24: {24},
        #     25: {25},
        #     26: {26},
        # }

    def get_pyg_data(self):
        # returns a Data object containing:
        # - node features (x)
        # - edge indices (edge_index)
        x = [[1.0 if t == atom_type else 0.0 for atom_type in (ATOM_H, ATOM_O, ATOM_C)]
             for t in self.node_types]  # each atom is encoded using a one-hot vector with the order
        x = torch.tensor(x, dtype=torch.float)
        # Mirror bonds based on order for GNN input
        edge_list = []
        for (u, v), order in self.bonds.items():
            for _ in range(order):
                edge_list.append([u - self.offset, v - self.offset])
                edge_list.append([v - self.offset, u - self.offset])
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous() if edge_list else torch.empty((2, 0), dtype=torch.long)
        return Data(x=x, edge_index=edge_index) # returning the PyG graph data object with node features and edge indices

    def get_valid_action_masks(self):
        # returns two masks:
        # - identifies atoms that can accept new bonds (grow_mask)
        # - identifies valid bonds and bond orders between existing atoms
        #
        # Example: for [C, O, H] with one existing single C-O bond:
        # current_bonds = [1, 1, 0] and bonds = {(0, 1): 1}
        # node_grow_mask = [1, 1, 1]
        # edge_connect_mask[C, O] = [0, 1, 0]  # C-O can become double
        # edge_connect_mask[C, H] = [1, 0, 0]  # only single is valid
        # edge_connect_mask[O, H] = [1, 0, 0]  # only single is valid
        # Each edge entry is [single, double, triple]. Self-bond entries are
        # [0, 0, 0] because an atom cannot connect to itself.
        num_nodes = len(self.node_types)
        node_grow_mask = [1.0 if self.current_bonds[i] < MAX_VALENCY[self.node_types[i]] else 0.0 for i in range(num_nodes)]
        
        # Output shape: (num_nodes, num_nodes, 3) -> (pair_i_j, [single, double, triple])
        edge_connect_mask = torch.zeros((num_nodes, num_nodes, 3), dtype=torch.float)
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i == j: continue # 1. Prevent self-loops
                u_global, v_global = i + self.offset, j + self.offset
                pair = tuple(sorted((u_global, v_global)))
                current_order = self.bonds.get(pair, 0)

                # TREST EACH DESIRED BOND ORDER
                # Each channel represents the desired final order: 1 for a single, 2 for a double, and 3 for a triple bond. Enable it only when the required
                # increase fits within both atoms' remaining valency.
                for bond_type, desired_order in enumerate((1, 2, 3)):
                    order_increment = desired_order - current_order
                    i_remaining = MAX_VALENCY[self.node_types[i]] - self.current_bonds[i]
                    j_remaining = MAX_VALENCY[self.node_types[j]] - self.current_bonds[j]
                    if (order_increment > 0 and
                        order_increment <= i_remaining and
                        order_increment <= j_remaining):
                        edge_connect_mask[i, j, bond_type] = 1.0
        return torch.tensor(node_grow_mask, dtype=torch.float), edge_connect_mask

    def step(self, action_class, u_global, v_global=None, new_atom_type=None, bond_order_inc=0):
        # three possible action classes:
        # 0: add a new atom and connect to existing atom u_global with a single bond
        # 1: create or upgrade a bond
        # 2: terminate the episode
        if action_class == 2:
            self.terminated = True
            self.success = True
            print("Action: STOP")
            print("Result: molecule generation stopped")
            return self.success, self.terminated

        u_local = u_global - self.offset

        if action_class == 0:
            new_global_idx = len(self.node_types) + self.offset
            self.node_types.append(new_atom_type)
            self.current_bonds.append(1)
            self.current_bonds[u_local] += 1
            self.bonds[(u_global, new_global_idx)] = 1

            existing_symbol = ATOM_SYMBOL[self.node_types[u_local]]
            new_symbol = ATOM_SYMBOL[new_atom_type]
            print("Action: ADD ATOM")
            print(f"Created: {new_symbol}({new_global_idx})")
            print(f"Bond: {existing_symbol}({u_global}) - {new_symbol}({new_global_idx}) [single]")
            print("Result: accepted")
            
            u_species_set = self.node_to_species[u_global]
            u_species_set.add(new_global_idx)
            self.node_to_species[new_global_idx] = u_species_set
            return True, False # (Success=True, Terminated=False)

        elif action_class == 1:
            pair = tuple(sorted((u_global, v_global)))
            current_order = self.bonds.get(pair, 0)
            order_increment = bond_order_inc  # This is passed in as 1 or 2 depending on the action

            # STRICT CHECK
            if current_order + order_increment > 3:
                print("Action: ADD OR UPGRADE BOND")
                print(f"Atoms: {u_global} and {v_global}")
                print("Result: rejected; bond order would exceed 3")
                return False, False # (Success=False, Terminated=False)

            # Only proceed if we are actually adding capacity
            if order_increment > 0:
                # Calculate remaining capacity for both atoms
                u_local = u_global - self.offset
                v_local = v_global - self.offset

                u_remaining = MAX_VALENCY[self.node_types[u_local]] - self.current_bonds[u_local]
                v_remaining = MAX_VALENCY[self.node_types[v_local]] - self.current_bonds[v_local]
                
                # Check valency capacity
                if order_increment <= u_remaining and order_increment <= v_remaining:
                    # Update bond order state
                    self.bonds[pair] = current_order + order_increment
                    
                    # Update valency count: 
                    # cost is exactly the new units added to the bond order
                    self.current_bonds[u_local] += order_increment
                    self.current_bonds[v_local] += order_increment

                    if self.node_to_species[u_global] is not self.node_to_species[v_global]:
                        merged_set = self.node_to_species[u_global].union(self.node_to_species[v_global])
                        for idx in merged_set: self.node_to_species[idx] = merged_set
                    
                    u_symbol = ATOM_SYMBOL[self.node_types[u_local]]
                    v_symbol = ATOM_SYMBOL[self.node_types[v_local]]
                    print("Action: ADD OR UPGRADE BOND")
                    print(f"Atoms: {u_symbol}({u_global}) and {v_symbol}({v_global})")
                    print(f"Bond order: {current_order} -> {self.bonds[pair]}")
                    print("Result: accepted")
                    return True, False # (Success=True, Terminated=False)
                else:
                    print("Action: ADD OR UPGRADE BOND")
                    print(f"Atoms: {u_global} and {v_global}")
                    print("Result: rejected; insufficient remaining valency")
                    return False, False # (Success=False, Terminated=False)
                
        return False, self.terminated

    def evaluate_inventory(self):
        # Reward each recognized multi-atom species only once per epoch.
        # Repeated copies score 0 but still dilute the average reward.
        unique_sets = []
        for s in self.node_to_species.values():
            if s not in unique_sets: unique_sets.append(s)
        formulas, invalid_formula, all_legal = [], [], True
        molecule_scores = []
        score_details = []
        rewarded_species = set()
        for group in unique_sets:
            num_H = sum(1 for idx in group if self.node_types[idx - self.offset] == ATOM_H)
            num_O = sum(1 for idx in group if self.node_types[idx - self.offset] == ATOM_O)
            num_C = sum(1 for idx in group if self.node_types[idx - self.offset] == ATOM_C)

            # Build the actual connected structure, including atom types and bond orders.
            structure = nx.Graph()
            for idx in group:
                structure.add_node(idx, atom=self.node_types[idx - self.offset])
            for (u, v), order in self.bonds.items():
                if u in group and v in group:
                    structure.add_edge(u, v, order=order)

            # Compare the structure with every known species. This distinguishes
            # structural isomers such as CH3O/CH2OH and CH2CHO/CH3CO.
            species_name = None
            for name, reference in SPECIES_GRAPHS.items():
                if nx.is_isomorphic(structure, reference,
                                     node_match=NODE_MATCH, edge_match=EDGE_MATCH):
                    species_name = name
                    break

            if species_name is not None:
                formulas.append(species_name)
                if len(group) == 1:
                    score = ISOLATED_ATOM_REWARD
                    detail = f"{species_name}={score:+.1f}"
                elif species_name in rewarded_species:
                    score = DUPLICATE_MOLECULE_REWARD
                    detail = f"{species_name}={score:+.1f} (duplicate)"
                else:
                    score = RECOGNIZED_MOLECULE_REWARD
                    rewarded_species.add(species_name)
                    detail = f"{species_name}={score:+.1f}"
                molecule_scores.append(score)
                score_details.append(detail)
                continue

            # Unknown structures are displayed by empirical formula CxHyOn.
            H_part = f"H{num_H}" if num_H > 1 else ("H" if num_H == 1 else "")
            O_part = f"O{num_O}" if num_O > 1 else ("O" if num_O == 1 else "")
            C_part = f"C{num_C}" if num_C > 1 else ("C" if num_C == 1 else "")
            formula = f"{C_part}{H_part}{O_part}"
            print(f"Unrecognized molecule: {formula} (C={num_C}, H={num_H}, O={num_O})")
            invalid_formula.append(formula)
            # An unrecognized multi-atom molecule scores -0.5; an isolated atom scores 0.
            score = ISOLATED_ATOM_REWARD if len(group) == 1 else UNRECOGNIZED_MOLECULE_REWARD
            molecule_scores.append(score)
            score_details.append(f"{formula}={score:+.1f}")
            all_legal = False

        # R_epoch = (r_1 + r_2 + ... + r_N) / N molecular components.
        average_score = sum(molecule_scores) / len(molecule_scores)
        print("Molecule scores: " + ", ".join(score_details))
        print(f"Average molecule reward: {average_score:+.3f}")
        return formulas, invalid_formula, all_legal, average_score

# ==========================================
# POLICY GNN MODEL
# ==========================================
class GrowthGNN(nn.Module):
    def __init__(self, hidden_dims=None):
        super().__init__()
        hidden_dims = list(GNN_HIDDEN_DIMS if hidden_dims is None else hidden_dims)
        if not hidden_dims or any(type(width) is not int or width <= 0 for width in hidden_dims):
            raise ValueError("hidden_dims must be a nonempty list of positive integers")
        layer_dims = [3] + hidden_dims
        self.convs = nn.ModuleList([
            GCNConv(in_dim, out_dim)
            for in_dim, out_dim in zip(layer_dims[:-1], layer_dims[1:])
        ])
        final_dim = hidden_dims[-1]
        self.grow_head = nn.Linear(final_dim, 3)
        # Output 3 values per pair (Single=0, Double=1, Triple=2)
        self.connect_head = nn.Linear(final_dim * 2, 3)
        self.termination_layer = nn.Linear(final_dim, 1)

    def forward(self, data, grow_mask, edge_mask):
        h = data.x
        for conv in self.convs:
            h = F.leaky_relu(conv(h, data.edge_index), 0.1)
        grow_logits = self.grow_head(h) + (grow_mask.unsqueeze(1) - 1.0) * 1e9
        
        num_nodes = h.size(0)
        h_i = h.unsqueeze(1).expand(-1, num_nodes, -1)
        h_j = h.unsqueeze(0).expand(num_nodes, -1, -1)
        pair_features = torch.cat([h_i, h_j], dim=-1)
        # connect_logits shape: (N, N, 3)
        connect_logits = self.connect_head(pair_features) + (edge_mask - 1.0) * 1e9
        
        term_logit = self.termination_layer(global_mean_pool(h, batch=None))
        return grow_logits, connect_logits, term_logit

# ==========================================
# 2D SPIRAL MAPPER 
# ==========================================
def get_spiral_coordinates(index, spacing=35.0):
    if index == 0: return 0.0, 0.0
    x = y = 0
    dx = 0; dy = -1
    step_limit = max_steps = 1
    turns = 0
    for _ in range(index):
        x += dx; y += dy
        step_limit -= 1
        if step_limit == 0:
            dx, dy = -dy, dx
            turns += 1
            if turns % 2 == 0: max_steps += 1
            step_limit = max_steps
    return float(x * spacing), float(y * spacing)

# ==========================================
# MAIN TRAINING WORKSPACE
# ==========================================
if __name__ == "__main__":
    log_file = open(LOG_PATH, "w")
    sys.stdout = Tee(sys.stdout, log_file)

    # Make model initialization and exploration noise reproducible.
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)

    pretrained_model_path = os.environ.get("GNN_PRETRAINED_MODEL_PATH")
    pretrained_data = None
    if pretrained_model_path:
        pretrained_data = torch.load(pretrained_model_path, map_location="cpu", weights_only=False)

    saved_config = pretrained_data["config"] if pretrained_data else {}
    if "hidden_dims" in saved_config:
        hidden_dims = list(saved_config["hidden_dims"])
    elif pretrained_data:
        # Older checkpoints used the same width for every GCN layer.
        hidden_dims = [saved_config.get("hidden_dim", 32)] * saved_config.get("num_layers", 3)
    else:
        hidden_dims = list(GNN_HIDDEN_DIMS)
    num_layers = len(hidden_dims)
    learning_rate = 0.01
    reward_weight = 0.5
    discount_factor = 0.9
    total_epochs = int(os.environ.get("GNN_ADDITIONAL_EPOCHS", "100"))

    gnn_model = GrowthGNN(hidden_dims=hidden_dims)
    if pretrained_data:
        gnn_model.load_state_dict(pretrained_data["model_state_dict"])
    optimizer = torch.optim.Adam(gnn_model.parameters(), lr=learning_rate)

    training_config = {
        "model_class": gnn_model.__class__.__name__,
        "node_feature_dim": 3,
        "hidden_dims": hidden_dims,
        "num_layers": num_layers,
        "trainable_parameters": sum(p.numel() for p in gnn_model.parameters() if p.requires_grad),
        "learning_rate": learning_rate,
        "reward_weight": reward_weight,
        "discount_factor": discount_factor,
        "total_epochs": total_epochs,
        "max_steps_per_epoch": MAX_STEPS_PER_EPOCH,
        "initial_atoms": INITIAL_ATOMS,
        "recognized_molecule_reward": RECOGNIZED_MOLECULE_REWARD,
        "duplicate_molecule_reward": DUPLICATE_MOLECULE_REWARD,
        "unrecognized_molecule_reward": UNRECOGNIZED_MOLECULE_REWARD,
        "isolated_atom_reward": ISOLATED_ATOM_REWARD,
        "seed": SEED,
        "pretrained_model_path": pretrained_model_path,
        "termination_mode": "max_steps_or_no_valid_actions",
    }
    
    global_G = nx.Graph()
    global_pos = {}
    epoch_snapshots = [] 
    
    global_discovery_registry = {sp: 0 for sp in VALID_SPECIES}
    
    global_node_counter = 0
    
    # Dynamic styling rules to support large epoch counts cleanly
    if total_epochs <= 10:
        o_size, h_size, c_size, font_sz, padding_margin = 800, 700, 750, 6.0, 8.5
    elif total_epochs <= 25:
        o_size, h_size, c_size, font_sz, padding_margin = 450, 380, 415, 4.5, 12.0
    else:
        o_size, h_size, c_size, font_sz, padding_margin = 250, 200, 225, 3.5, 18.0
        
    print(f"Starting training: {total_epochs} epochs, up to {MAX_STEPS_PER_EPOCH} actions per epoch, seed={SEED}")
    if pretrained_model_path:
        print(f"Reusing trained neural network: {pretrained_model_path}")
    print(f"Neural network: {training_config['model_class']}, layers={num_layers}, "
          f"hidden_dims={hidden_dims}, trainable_parameters={training_config['trainable_parameters']}")
    print(f"Reward settings: recognized={RECOGNIZED_MOLECULE_REWARD:+.1f}, "
          f"duplicate={DUPLICATE_MOLECULE_REWARD:+.1f}, "
          f"unrecognized={UNRECOGNIZED_MOLECULE_REWARD:+.1f}, "
          f"isolated_atom={ISOLATED_ATOM_REWARD:+.1f}, reward_weight={reward_weight:.2f}, "
          f"discount_factor={discount_factor:.2f}")
    print(f"Termination: {MAX_STEPS_PER_EPOCH} actions or no valid actions remain")
    
    for epoch in range(1, total_epochs + 1):
        print(f"\n{'=' * 20} EPOCH {epoch:02d}/{total_epochs} {'=' * 20}")
        optimizer.zero_grad()
        env = AdvancedMoleculeEnv(node_offset=global_node_counter)
        
        # Keep neighboring epoch clusters far enough apart on the full canvas.
        x_center, y_center = get_spiral_coordinates(epoch - 1, spacing=80.0)
        cluster_center = np.array([x_center, y_center])
        
        epoch_initial_nodes = [
            (global_node_counter + i, atom_type)
            for i, atom_type in enumerate(INITIAL_ATOMS)
        ]
        
        for idx, a_type in epoch_initial_nodes:
            global_G.add_node(idx, element=ATOM_SYMBOL[a_type])
            
        global_node_counter += len(INITIAL_ATOMS)
        
        steps = 0
        action_log_probs = [] # Track log probabilities of chosen structural steps
        epoch_step_data = [] # List to store graph state at each step
        termination_reason = None
        
        while steps < MAX_STEPS_PER_EPOCH:
            print(f"\n---------- STEP {steps + 1:02d}/{MAX_STEPS_PER_EPOCH} ----------")
            pyg_data = env.get_pyg_data()
            grow_mask, edge_mask = env.get_valid_action_masks()

            no_valid_actions = (
                torch.sum(grow_mask) == 0 and torch.sum(edge_mask) == 0
            )
            if no_valid_actions:
                termination_reason = "no valid growth or connection actions remain"
                break

            # grow logit: scores for adding H, O, or C to each atom
            # connect_logits: scores for connecting pairs with single, double, or triple bonds
            grow_logits, connect_logits, _ = gnn_model(pyg_data, grow_mask, edge_mask)
            
            # Combine structural options into a single distribution for action sampling
            flat_grow = grow_logits.flatten()
            flat_conn = connect_logits.flatten()
            combined_logits = torch.cat([flat_grow, flat_conn])
            
            # Apply a softmax categorical distribution over valid structural moves
            probs = F.softmax(combined_logits, dim=-1)
            
            # (To retain exploration while updating, we sample or take argmax)
            noise_grow = torch.randn_like(grow_logits) * 0.4
            noise_conn = torch.randn_like(connect_logits) * 0.4
            max_grow_val = torch.max(grow_logits + noise_grow).item()
            max_conn_val = torch.max(connect_logits + noise_conn).item()

            # Determines which type of structural action the model takes at each time step.
            # Grow a new node or connect between existing nodes based on the highest logit value, 
            # with added noise for exploration.    
            if max_grow_val >= max_conn_val:    # GROW ACTION
                flat_idx = torch.argmax(grow_logits + noise_grow).item()
                # Track the log probability of choosing this growth path
                action_log_probs.append(torch.log(F.softmax(grow_logits.flatten(), dim=-1)[flat_idx] + 1e-8))
                
                # Decode the node and atom type from the flat index
                # Use integer division (//) to find the node index (every node has 3 possible atom types)
                # Use modulo (%) to determine the atom type (0 for H, 1 for O, 2 for C)
                # new atom could be H, O, C, and therefore integer division by 3 gives the node index, and modulo 3 gives the (grow) atom type.
                u_local = flat_idx // 3
                chosen_atom = flat_idx % 3

                # Convert local index to global index for environment consistency
                u_global = u_local + env.offset
                new_global_idx = len(env.node_types) + env.offset
                
                # Update graph and environment state
                global_G.add_node(new_global_idx, element=ATOM_SYMBOL[chosen_atom])
                env.step(action_class=0, u_global=u_global, new_atom_type=chosen_atom)
                global_G.add_edge(u_global, new_global_idx)
            else:   # CONNECTING BETWEEN EXISTING NODES
                flat_idx = torch.argmax(connect_logits + noise_conn).item()
                # Track the log probability of choosing this bonding connection
                action_log_probs.append(torch.log(F.softmax(connect_logits.flatten(), dim=-1)[flat_idx] + 1e-8))
                
                # Decode the flat index back into two distinct node indices (u, v)
                # The matrix is (num_nodes, num_nodes), so use floor division and modulo
                num_nodes = connect_logits.size(0)
                u_local = (flat_idx // 3) // num_nodes
                v_local = (flat_idx // 3) % num_nodes
                bond_type = flat_idx % 3 # 0 -> Single, 1 -> Double, 2 -> Triple
                u_global = u_local + env.offset
                v_global = v_local + env.offset
                
                # Update the environment and visual graph state
                # This checks valency and updates the edge list
                desired_order = 1 + bond_type # Map 0->1, 1->2, 2->3
                pair = tuple(sorted((u_global, v_global)))
                current_order = env.bonds.get(pair, 0)
                order_increment = desired_order - current_order
                success, terminated = env.step(action_class=1, u_global=u_global, v_global=v_global, bond_order_inc=order_increment)

                # Update the NetworkX graph so the visualization reflects the new bond
                if success:
                    global_G.add_edge(u_global, v_global)
                
            current_state = {
                'step': steps,
                'edges': list(global_G.edges()),
                'nodes': list(global_G.nodes(data=True))
            }
            epoch_step_data.append(current_state)
            steps += 1

        if termination_reason is None:
            termination_reason = f"maximum of {MAX_STEPS_PER_EPOCH} actions reached"
        print(f"\nTermination: {termination_reason}")
        
        # Evaluate all final molecules and use their average score as the epoch reward.
        formulas_list, invalid_formulas_list, success, reward = env.evaluate_inventory()
        inventory_path = save_species_inventory(env, epoch, append=epoch > 1)
        print(f"Species inventory saved: {inventory_path}")
        
        # Policy Loss: Penalize choices that lead to chemical errors, reward valid configurations
        policy_loss = torch.tensor(0.0)
        if len(action_log_probs) > 0:
            # Give action t the discounted return G_t = gamma^(T-1-t) * R_terminal.
            discounted_returns = calculate_discounted_returns(
                reward, len(action_log_probs), discount_factor
            )
            policy_loss = -(
                torch.stack(action_log_probs) * discounted_returns
            ).mean()
            
        # Only the structural policy is trained at this stage.
        total_loss = reward_weight * policy_loss
        if len(action_log_probs) > 0:
            total_loss.backward()
            optimizer.step()
        
        # ADDED: Clear loss formatting printed to console log
        mixture_str = " + ".join(formulas_list)
        if invalid_formulas_list:
            mixture_str += " | Unknown: " + ", ".join(invalid_formulas_list)
        # Report the averaged molecule reward alongside the total training loss.
        print(f"Epoch {epoch:02d}/{total_epochs} | System Output: {mixture_str:<22} "
              f"| Reward: {reward:+.3f} | Training Loss: {total_loss.item():.5f}")

        restart_path = save_restart_file(
            epoch, gnn_model, optimizer, reward, training_config
        )
        print(f"Restart file saved: {restart_path}")
        
        # ====================================================
        # COMPACT GEOMETRY POSITION GENERATOR
        # ====================================================
        unique_molecular_groups = []
        for s in env.node_to_species.values():
            if s not in unique_molecular_groups:
                unique_molecular_groups.append(list(s))
                
        num_molecules = len(unique_molecular_groups)

        # Put separate molecules into grid cells so their nodes cannot share the
        # same center. This is more reliable than placing differently sized
        # molecules around one circle.
        grid_columns = int(np.ceil(np.sqrt(num_molecules)))
        grid_rows = int(np.ceil(num_molecules / grid_columns))
        molecule_spacing = 22.0

        for mol_idx, atom_group in enumerate(unique_molecular_groups):
            num_atoms_in_mol = len(atom_group)

            row = mol_idx // grid_columns
            column = mol_idx % grid_columns
            x_offset = (column - (grid_columns - 1) / 2) * molecule_spacing
            y_offset = ((grid_rows - 1) / 2 - row) * molecule_spacing
            mol_center = cluster_center + np.array([x_offset, y_offset])

            if num_atoms_in_mol == 1:
                global_pos[atom_group[0]] = mol_center
            else:
                # Spring layout uses the actual bonds, keeping bonded atoms close
                # and greatly reducing crossed or intertwined edges.
                molecule_graph = global_G.subgraph(atom_group)
                molecule_positions = nx.spring_layout(
                    molecule_graph,
                    seed=epoch * 1000 + mol_idx,
                    center=mol_center,
                    scale=7.0,
                    weight=None,
                )
                global_pos.update(molecule_positions)

        epoch_snapshots.append({
            'epoch_num': epoch,
            'node_offset': env.offset,
            'max_node_idx': len(global_G.nodes()),
            'edges': list(global_G.edges()),
            'text': f"Epoch {epoch}: {mixture_str} | Reward: {reward:+.3f}"
        })
        global_node_counter = len(global_G.nodes())

    # ========================================================
    # OUTPUT 1: STANDALONE INDIVIDUAL PLOTS
    # ========================================================
    print("\nSaving standalone image files for each epoch layout...")
    output_dir = INDIVIDUAL_PLOTS_FOLDER
    os.makedirs(output_dir, exist_ok=True)
    
    for snap in epoch_snapshots:
        fig_ind, ax_ind = plt.subplots(figsize=(8, 8))
        ax_ind.set_facecolor('#fcfcfc')
        ax_ind.set_title(snap['text'], fontsize=11, fontweight='bold', color='#1e272c', pad=10)
        
        start_node = snap['node_offset']
        end_node = snap['max_node_idx']
        
        epoch_h = [n for n in range(start_node, end_node) if global_G.nodes[n]['element'] == "H"]
        epoch_o = [n for n in range(start_node, end_node) if global_G.nodes[n]['element'] == "O"]
        epoch_c = [n for n in range(start_node, end_node) if global_G.nodes[n]['element'] == "C"]
        epoch_edges = [(u, v) for u, v in snap['edges'] if start_node <= u < end_node and start_node <= v < end_node]
        
        if epoch_edges:
            nx.draw_networkx_edges(global_G, global_pos, edgelist=epoch_edges, width=1.5, edge_color="#7f8c8d", ax=ax_ind)
        if epoch_o:
            # Standalone plots keep structural high detail scaling
            nx.draw_networkx_nodes(global_G, global_pos, nodelist=epoch_o, node_color="#ff4d4d", 
                                   node_size=500, edgecolors="black", linewidths=0.8, ax=ax_ind)
        if epoch_h:
            nx.draw_networkx_nodes(global_G, global_pos, nodelist=epoch_h, node_color="#a6c8ff", 
                                   node_size=450, edgecolors="black", linewidths=0.8, ax=ax_ind)
        if epoch_c:
            nx.draw_networkx_nodes(global_G, global_pos, nodelist=epoch_c, node_color="#606060",
                                   node_size=475, edgecolors="black", linewidths=0.8, ax=ax_ind)
            
        # Index-only formatting
        epoch_nodes = epoch_h + epoch_o + epoch_c
        clean_labels = {n: f"({n})" for n in epoch_nodes}
        nx.draw_networkx_labels(global_G, global_pos, labels=clean_labels, font_size=6.5, 
                                font_weight="bold", font_family="sans-serif", ax=ax_ind)
        
        coords = np.array([global_pos[n] for n in epoch_nodes])
        x_min, y_min = np.min(coords, axis=0) - 10.0
        x_max, y_max = np.max(coords, axis=0) + 10.0
        
        ax_ind.set_xlim(x_min, x_max)
        ax_ind.set_ylim(y_min, y_max)
        ax_ind.axis('off')
        
        fig_ind.savefig(f"{output_dir}/epoch_{snap['epoch_num']}_final.png", bbox_inches='tight', dpi=150)
        plt.close(fig_ind)

    # ========================================================
    # OUTPUT 2: PROGRESSIVE MESH CANVAS (GIF GENERATION)
    # ========================================================
    print("\nCompiling full-grid composite plot map...")
    fig, ax = plt.subplots(figsize=(14, 14))

    def update_canvas(frame_idx):
        ax.clear()
        snapshot = epoch_snapshots[frame_idx]
        max_visible_node = snapshot['max_node_idx']
        
        ax.set_title(snapshot['text'], fontsize=14, fontweight='bold', color='#1e272c', pad=15)
        ax.set_facecolor('#fcfcfc')
        
        h_nodes = [n for n, attr in global_G.nodes(data=True) if attr['element'] == "H" and n < max_visible_node]
        o_nodes = [n for n, attr in global_G.nodes(data=True) if attr['element'] == "O" and n < max_visible_node]
        c_nodes = [n for n, attr in global_G.nodes(data=True) if attr['element'] == "C" and n < max_visible_node]
        
        visible_edges = [(u, v) for u, v in snapshot['edges'] if u < max_visible_node and v < max_visible_node]
        if visible_edges:
            nx.draw_networkx_edges(global_G, global_pos, edgelist=visible_edges, width=1.2, edge_color="#7f8c8d", ax=ax)
            
        if o_nodes:
            nx.draw_networkx_nodes(global_G, global_pos, nodelist=o_nodes, node_color="#ff4d4d", 
                                   node_size=o_size, edgecolors="black", linewidths=0.8, ax=ax)
        if h_nodes:
            nx.draw_networkx_nodes(global_G, global_pos, nodelist=h_nodes, node_color="#a6c8ff", 
                                   node_size=h_size, edgecolors="black", linewidths=0.8, ax=ax)
        if c_nodes:
            nx.draw_networkx_nodes(global_G, global_pos, nodelist=c_nodes, node_color="#606060",
                                   node_size=c_size, edgecolors="black", linewidths=0.8, ax=ax)
            
        visible_nodes = h_nodes + o_nodes + c_nodes
        global_clean_labels = {n: f"({n})" for n in visible_nodes}
        nx.draw_networkx_labels(global_G, global_pos, labels=global_clean_labels, font_size=font_sz, 
                                font_weight="bold", font_family="sans-serif", ax=ax)

        # Fit each animation frame to the epochs currently visible. Previously,
        # frame 1 used the bounds of all 50 epochs, making its atoms tiny and
        # visually crowded into a single point.
        visible_coords = np.array([global_pos[n] for n in visible_nodes])
        visible_x_min, visible_y_min = np.min(visible_coords, axis=0) - padding_margin
        visible_x_max, visible_y_max = np.max(visible_coords, axis=0) + padding_margin
        ax.set_xlim(visible_x_min, visible_x_max)
        ax.set_ylim(visible_y_min, visible_y_max)
        ax.axis('off')

    ani = animation.FuncAnimation(fig, update_canvas, frames=len(epoch_snapshots), interval=1600, repeat=False)
    
    output_filename = GIF_PATH
    print(f"Saving single-plot grid animation sequence: {output_filename} ...")
    ani.save(output_filename, writer='pillow', fps=0.62)
    plt.close()

    # Keep restart files if any earlier stage fails. Remove them only after the
    # complete run succeeds and the reusable trained model has been saved.
    model_path = save_neural_network(gnn_model, training_config)
    deleted_count = delete_restart_files()
    print(f"Trained neural network saved: {model_path}")
    print(f"Deleted {deleted_count} restart files after successful completion.")
    print("Everything processed successfully!")
