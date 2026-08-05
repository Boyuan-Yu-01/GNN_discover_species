"""Reaction environment for learning bond-formation and bond-breakage actions."""

from itertools import combinations
from numbers import Integral

import torch

from obj_edge import Bond
from obj_node import atom
from obj_subgraph import molecule


class MoleculeEnv:
    """Manage the atoms, bonds, and molecule subgraphs in one reaction episode.

    The environment is responsible for all graph-changing actions. The
    ``molecule`` objects are connected subgraph snapshots whose features can be
    supplied to the GNN. One environment instance represents one episode, so a
    new instance is constructed instead of calling reset.

    Four kinds of information remain separate:

    * node information for the ``N`` atoms;
    * existing-edge information for the ``E`` current bonds;
    * candidate-pair information for the ``P = N * (N - 1) / 2`` unordered
      atom pairs; and
    * molecule/subgraph information for the ``M`` connected molecules.

    Candidate pairs with current bond order zero are possible actions but are
    not GNN message-passing edges.
    """

    FORMATION = "formation"
    BREAKAGE = "breakage"
    MAX_BOND_ORDER = max(Bond.VALID_ORDERS)

    def __init__(self, initial_molecules, max_steps=100):
        """
        Initialize an oven from a collection of connected molecules.

        Register the initial atoms and bonds, build the atom-to-molecule map,
        initialize episode counters, and validate that no atom or bond is
        registered more than once. A singleton atom is represented by a
        one-atom molecule. The atom inventory and unordered pair rows remain
        fixed throughout the episode.
        """

        if not isinstance(max_steps, Integral) or isinstance(max_steps, bool):
            raise TypeError("max_steps must be an integer.")
        if max_steps <= 0:
            raise ValueError("max_steps must be positive.")

        self.molecules = list(initial_molecules)
        if not self.molecules:
            raise ValueError("MoleculeEnv requires at least one initial molecule.")
        if any(not isinstance(species, molecule) for species in self.molecules):
            raise TypeError("initial_molecules must contain only molecule objects.")

        # Stable component ordering makes molecule-feature rows reproducible.
        self.molecules.sort(key=self._minimum_atom_index)
        self.atoms = {}
        self.bonds = {}

        for species in self.molecules:
            for member in species.atoms:
                if not isinstance(member.index, Integral) or isinstance(
                    member.index, bool
                ):
                    raise TypeError("Every atom index must be an integer.")
                if member.index in self.atoms:
                    raise ValueError(
                        f"Atom index {member.index} appears in multiple molecules."
                    )
                self.atoms[member.index] = member

            for existing_bond in species.bonds:
                key = self._bond_key(*existing_bond.node_indices)
                if key in self.bonds:
                    raise ValueError(
                        f"Bond between atoms {key[0]} and {key[1]} is duplicated."
                    )
                self.bonds[key] = existing_bond

        self.max_steps = int(max_steps)
        self.step_count = 0
        self.terminated = False
        self.truncated = False

        self.atom_to_molecule = {}
        self.pair_index = ()
        self.pair_to_row = {}
        self.current_BO = []
        self.maximum_BO = []
        self.valid_actions = {}

        self._rebuild_atom_to_molecule()
        self._build_pair_tables()
        self.valid_actions = self.get_valid_actions()
        self.terminated = not bool(self.valid_actions)
        self.validate_state()

    def step(self, action):
        """Apply one model-selected action and return the next environment state.

        ``action`` has the form
        ``(atom1_index, atom2_index, action_type, bond_change)``. This method
        decodes and dispatches the action; the mutation method performs the local
        chemical safety checks. The return format is
        ``(observation, reward, terminated, truncated, info)``. Reward is kept at
        zero until a project-specific reward function is introduced.
        """

        if self.terminated or self.truncated:
            raise RuntimeError("Cannot step an environment whose episode has ended.")
        if not isinstance(action, (tuple, list)) or len(action) != 4:
            raise ValueError(
                "action must be (atom1_index, atom2_index, action_type, "
                "bond_change)."
            )

        atom1_index, atom2_index, action_type, bond_change = action
        if atom1_index not in self.atoms or atom2_index not in self.atoms:
            raise ValueError("Both action atom indices must belong to the oven.")

        atom1 = self.atoms[atom1_index]
        atom2 = self.atoms[atom2_index]
        old_order = self._current_bond_order(atom1, atom2)
        old_molecule_count = len(self.molecules)

        if action_type == self.FORMATION:
            self.bond_formation(atom1, atom2, bond_change)
        elif action_type == self.BREAKAGE:
            self.bond_breakage(atom1, atom2, bond_change)
        else:
            raise ValueError(
                f"action_type must be {self.FORMATION!r} or {self.BREAKAGE!r}."
            )

        self.step_count += 1
        self.truncated = self.step_count >= self.max_steps
        self.terminated = not bool(self.valid_actions)

        new_order = self._current_bond_order(atom1, atom2)
        new_molecule_count = len(self.molecules)
        info = {
            "action_type": action_type,
            "bond_change": int(bond_change),
            "old_bond_order": old_order,
            "new_bond_order": new_order,
            "molecule_count_before": old_molecule_count,
            "molecule_count_after": new_molecule_count,
            "merged": new_molecule_count < old_molecule_count,
            "split": new_molecule_count > old_molecule_count,
        }
        observation = self.get_gnn_observation()
        reward = 0.0
        return observation, reward, self.terminated, self.truncated, info

    def bond_formation(self, atom1, atom2, bond_change=1):
        """Increase the bond order between two atoms by a selected amount.

        Create a new bond when none exists, or strengthen an existing bond.
        Formation within one molecule replaces one snapshot; formation between
        two molecules merges their snapshots. The selected pair table row and
        all maximum-order rows affected by endpoint remaining valence are
        updated.
        """

        self._require_registered_atom(atom1)
        self._require_registered_atom(atom2)
        self._validate_bond_change(bond_change)
        if atom1 is atom2:
            raise ValueError("An atom cannot form a bond with itself.")

        key = self._bond_key(atom1.index, atom2.index)
        existing_bond = self.bonds.get(key)
        current_order = 0 if existing_bond is None else existing_bond.order
        maximum_order = self._maximum_reachable_order(
            atom1, atom2, current_order
        )
        maximum_change = maximum_order - current_order
        if bond_change > maximum_change:
            raise ValueError(
                f"Bond order can increase by at most {maximum_change}; "
                f"received {bond_change}."
            )

        molecule1 = self.molecule_containing(atom1)
        molecule2 = self.molecule_containing(atom2)
        new_order = current_order + bond_change
        new_bond = Bond(atom1, atom2, new_order)

        # All conditions are checked before either endpoint is mutated.
        atom1.form_bond(bond_change)
        atom2.form_bond(bond_change)
        self.bonds[key] = new_bond

        if existing_bond is None and molecule1 is not molecule2:
            self._merge_molecules(molecule1, molecule2, new_bond)
        else:
            updated_bonds = []
            replaced_existing_bond = False
            for member_bond in molecule1.bonds:
                if self._bond_key(*member_bond.node_indices) == key:
                    updated_bonds.append(new_bond)
                    replaced_existing_bond = True
                else:
                    updated_bonds.append(member_bond)

            if not replaced_existing_bond:
                updated_bonds.append(new_bond)

            updated_molecule = molecule(
                atoms=sorted(molecule1.atoms, key=lambda item: item.index),
                bonds=sorted(updated_bonds, key=self._bond_sort_key),
            )
            self._replace_molecules((molecule1,), (updated_molecule,))

        self._update_pair_tables(atom1, atom2)
        self.valid_actions = self.get_valid_actions()
        return new_bond

    def bond_breakage(self, atom1, atom2, bond_change=1):
        """Decrease the bond order between two atoms by a selected amount.

        Restore remaining valence at both endpoints. A remaining positive bond
        order replaces one molecule snapshot without changing connectivity.
        Complete removal replaces the affected molecule with one or two
        snapshots, depending on whether the removed bond was a bridge.
        """

        self._require_registered_atom(atom1)
        self._require_registered_atom(atom2)
        self._validate_bond_change(bond_change)
        if atom1 is atom2:
            raise ValueError("An atom cannot break a bond with itself.")

        key = self._bond_key(atom1.index, atom2.index)
        existing_bond = self.bonds.get(key)
        if existing_bond is None:
            raise ValueError(
                f"Atoms {key[0]} and {key[1]} do not have a bond to break."
            )
        if bond_change > existing_bond.order:
            raise ValueError(
                f"Bond order can decrease by at most {existing_bond.order}; "
                f"received {bond_change}."
            )

        old_molecule = self.molecule_containing(atom1)
        if self.molecule_containing(atom2) is not old_molecule:
            raise RuntimeError("Bond endpoints are assigned to different molecules.")

        new_order = existing_bond.order - bond_change
        atom1.break_bond(bond_change)
        atom2.break_bond(bond_change)

        if new_order > 0:
            new_bond = Bond(atom1, atom2, new_order)
            self.bonds[key] = new_bond
            updated_bonds = [
                new_bond
                if self._bond_key(*member_bond.node_indices) == key
                else member_bond
                for member_bond in old_molecule.bonds
            ]
            updated_molecule = molecule(
                atoms=sorted(old_molecule.atoms, key=lambda item: item.index),
                bonds=sorted(updated_bonds, key=self._bond_sort_key),
            )
            self._replace_molecules((old_molecule,), (updated_molecule,))
        else:
            new_bond = None
            del self.bonds[key]
            self._split_molecule(old_molecule, existing_bond)

        self._update_pair_tables(atom1, atom2)
        self.valid_actions = self.get_valid_actions()
        return new_bond

    def get_valid_actions(self):
        """Return maximum legal bond changes for every actionable pair.

        The returned dictionary maps
        ``(atom1_index, atom2_index, action_type)`` to the maximum positive
        ``bond_change``. Maximum formation is ``maximum_BO - current_BO`` and
        maximum breakage is ``current_BO``.
        """

        actions = {}
        for row, (atom1_index, atom2_index) in enumerate(self.pair_index):
            current_order = self.current_BO[row]
            maximum_order = self.maximum_BO[row]
            maximum_formation = maximum_order - current_order

            if maximum_formation > 0:
                actions[
                    (atom1_index, atom2_index, self.FORMATION)
                ] = maximum_formation
            if current_order > 0:
                actions[(atom1_index, atom2_index, self.BREAKAGE)] = current_order
        return actions

    def get_valid_action_masks(self):
        """Return formation and breakage masks with shape ``(P, 3)``."""

        pair_count = len(self.pair_index)
        formation_mask = torch.zeros(
            (pair_count, self.MAX_BOND_ORDER), dtype=torch.bool
        )
        breakage_mask = torch.zeros_like(formation_mask)

        for action_key, maximum_change in self.valid_actions.items():
            atom1_index, atom2_index, action_type = action_key
            row = self.pair_to_row[(atom1_index, atom2_index)]
            if action_type == self.FORMATION:
                formation_mask[row, :maximum_change] = True
            else:
                breakage_mask[row, :maximum_change] = True
        return formation_mask, breakage_mask

    def get_gnn_observation(self):
        """Convert the current oven state into GNN-ready tensor information.

        Existing bonds form ``edge_index`` and ``edge_attr``. All unordered atom
        pairs form ``pair_index`` and the bond-order/action tables. Molecule
        features and ``component_index`` remain separate subgraph information.
        """

        ordered_atoms = sorted(self.atoms.values(), key=lambda item: item.index)
        atom_to_row = {member.index: row for row, member in enumerate(ordered_atoms)}
        atom_indices = torch.tensor(
            [member.index for member in ordered_atoms], dtype=torch.long
        )
        node_features = torch.tensor(
            [member.features for member in ordered_atoms], dtype=torch.float
        )

        directed_edges = []
        directed_edge_features = []
        for key in sorted(self.bonds):
            existing_bond = self.bonds[key]
            row_i = atom_to_row[existing_bond.node_i.index]
            row_j = atom_to_row[existing_bond.node_j.index]
            directed_edges.extend(((row_i, row_j), (row_j, row_i)))
            directed_edge_features.extend(
                (existing_bond.features, existing_bond.features)
            )

        if directed_edges:
            edge_index = torch.tensor(directed_edges, dtype=torch.long).t()
            edge_attr = torch.tensor(
                directed_edge_features, dtype=torch.float
            )
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty(
                (0, self.MAX_BOND_ORDER), dtype=torch.float
            )

        if self.pair_index:
            pair_index = torch.tensor(
                [
                    [atom_to_row[index_i] for index_i, _ in self.pair_index],
                    [atom_to_row[index_j] for _, index_j in self.pair_index],
                ],
                dtype=torch.long,
            )
        else:
            pair_index = torch.empty((2, 0), dtype=torch.long)

        molecule_features = torch.tensor(
            [species.features for species in self.molecules], dtype=torch.float
        )
        molecule_to_row = {
            id(species): row for row, species in enumerate(self.molecules)
        }
        component_index = torch.tensor(
            [
                molecule_to_row[id(self.atom_to_molecule[member.index])]
                for member in ordered_atoms
            ],
            dtype=torch.long,
        )
        formation_mask, breakage_mask = self.get_valid_action_masks()

        return {
            "atom_indices": atom_indices,
            "x": node_features,
            "edge_index": edge_index.contiguous(),
            "edge_attr": edge_attr,
            "pair_index": pair_index.contiguous(),
            "current_bond_orders": torch.tensor(
                self.current_BO, dtype=torch.long
            ),
            "maximum_bond_orders": torch.tensor(
                self.maximum_BO, dtype=torch.long
            ),
            "formation_mask": formation_mask,
            "breakage_mask": breakage_mask,
            "molecule_features": molecule_features,
            "component_index": component_index,
        }

    def molecule_containing(self, selected_atom):
        """Return the current molecule snapshot containing an atom."""

        self._require_registered_atom(selected_atom)
        return self.atom_to_molecule[selected_atom.index]

    def validate_state(self):
        """Check that the complete environment state is internally consistent."""

        if not self.molecules:
            raise RuntimeError("The environment must contain at least one molecule.")

        molecule_atom_counts = {index: 0 for index in self.atoms}
        molecule_bonds = {}
        for species in self.molecules:
            if not isinstance(species, molecule) or not species.is_connected():
                raise RuntimeError("Every molecule snapshot must be connected.")
            for member in species.atoms:
                if self.atoms.get(member.index) is not member:
                    raise RuntimeError("Molecule atom does not match environment atom.")
                molecule_atom_counts[member.index] += 1
            for existing_bond in species.bonds:
                key = self._bond_key(*existing_bond.node_indices)
                if key in molecule_bonds:
                    raise RuntimeError("A bond appears in multiple molecules.")
                molecule_bonds[key] = existing_bond

        if any(count != 1 for count in molecule_atom_counts.values()):
            raise RuntimeError("Every atom must belong to exactly one molecule.")
        if set(molecule_bonds) != set(self.bonds):
            raise RuntimeError("Molecule bonds do not match environment bonds.")
        for key, existing_bond in self.bonds.items():
            if key != self._bond_key(*existing_bond.node_indices):
                raise RuntimeError("A bond dictionary key is not canonical.")
            if molecule_bonds[key] is not existing_bond:
                raise RuntimeError("Molecule and environment Bond objects disagree.")
            if existing_bond.order not in Bond.VALID_ORDERS:
                raise RuntimeError("A bond has an unsupported order.")
            if self.atoms.get(existing_bond.node_i.index) is not existing_bond.node_i:
                raise RuntimeError("A bond endpoint is not registered.")
            if self.atoms.get(existing_bond.node_j.index) is not existing_bond.node_j:
                raise RuntimeError("A bond endpoint is not registered.")

        used_valence = {index: 0 for index in self.atoms}
        for existing_bond in self.bonds.values():
            used_valence[existing_bond.node_i.index] += existing_bond.order
            used_valence[existing_bond.node_j.index] += existing_bond.order
        for index, member in self.atoms.items():
            expected_remaining_valence = (
                member.MAX_VALENCE[member.element] - used_valence[index]
            )
            if (
                expected_remaining_valence < 0
                or member.remaining_valence != expected_remaining_valence
            ):
                raise RuntimeError(
                    f"Atom {index} remaining valence does not agree with "
                    "incident bonds."
                )
            if self.atom_to_molecule.get(index) is not self.molecule_containing(
                member
            ):
                raise RuntimeError("atom_to_molecule is inconsistent.")

        expected_pairs = tuple(combinations(sorted(self.atoms), 2))
        if self.pair_index != expected_pairs:
            raise RuntimeError("pair_index does not contain every unordered pair.")
        if len(self.pair_to_row) != len(expected_pairs):
            raise RuntimeError("pair_to_row has an incorrect size.")
        if len(self.current_BO) != len(expected_pairs) or len(
            self.maximum_BO
        ) != len(expected_pairs):
            raise RuntimeError("Bond-order tables have incorrect lengths.")

        for row, pair in enumerate(self.pair_index):
            if self.pair_to_row.get(pair) != row:
                raise RuntimeError("pair_to_row does not match pair_index.")
            atom1 = self.atoms[pair[0]]
            atom2 = self.atoms[pair[1]]
            expected_current = self._current_bond_order(atom1, atom2)
            expected_maximum = self._maximum_reachable_order(
                atom1, atom2, expected_current
            )
            if self.current_BO[row] != expected_current:
                raise RuntimeError("A current bond-order entry is stale.")
            if self.maximum_BO[row] != expected_maximum:
                raise RuntimeError("A maximum bond-order entry is stale.")

        expected_actions = self.get_valid_actions()
        if self.valid_actions != expected_actions:
            raise RuntimeError("The valid-action cache is stale.")
        return True

    def _replace_molecules(self, old_molecules, new_molecules):
        """Replace affected molecule snapshots after a graph edit."""

        old_molecules = tuple(old_molecules)
        new_molecules = tuple(new_molecules)
        if not old_molecules:
            raise ValueError("At least one old molecule must be replaced.")
        if any(not isinstance(species, molecule) for species in new_molecules):
            raise TypeError("Replacement objects must be molecules.")

        current_ids = {id(species) for species in self.molecules}
        old_ids = {id(species) for species in old_molecules}
        if len(old_ids) != len(old_molecules) or not old_ids <= current_ids:
            raise ValueError("An old molecule is missing or listed more than once.")

        old_atom_indices = {
            member.index for species in old_molecules for member in species.atoms
        }
        new_atom_indices = {
            member.index for species in new_molecules for member in species.atoms
        }
        if old_atom_indices != new_atom_indices:
            raise RuntimeError("A molecule replacement must preserve its atoms.")

        self.molecules = [
            species for species in self.molecules if id(species) not in old_ids
        ]
        self.molecules.extend(new_molecules)
        self.molecules.sort(key=self._minimum_atom_index)
        self._rebuild_atom_to_molecule()

    def _merge_molecules(self, molecule1, molecule2, new_bond):
        """Construct one connected molecule from two molecules and a new bond."""

        if molecule1 is molecule2:
            raise ValueError("Cannot merge a molecule with itself.")
        merged_molecule = molecule(
            atoms=sorted(
                molecule1.atoms + molecule2.atoms,
                key=lambda item: item.index,
            ),
            bonds=sorted(
                molecule1.bonds + molecule2.bonds + (new_bond,),
                key=self._bond_sort_key,
            ),
        )
        self._replace_molecules(
            (molecule1, molecule2), (merged_molecule,)
        )
        return merged_molecule

    def _split_molecule(self, old_molecule, removed_bond):
        """Replace a molecule after removing a bond, splitting it if necessary."""

        removed_key = self._bond_key(*removed_bond.node_indices)
        remaining_bonds = tuple(
            member_bond
            for member_bond in old_molecule.bonds
            if self._bond_key(*member_bond.node_indices) != removed_key
        )
        if len(remaining_bonds) != len(old_molecule.bonds) - 1:
            raise RuntimeError("The removed bond was not present exactly once.")

        adjacency = {member.index: set() for member in old_molecule.atoms}
        for member_bond in remaining_bonds:
            index_i, index_j = member_bond.node_indices
            adjacency[index_i].add(index_j)
            adjacency[index_j].add(index_i)

        components = []
        unvisited = set(adjacency)
        while unvisited:
            pending = [min(unvisited)]
            component = set()
            while pending:
                current = pending.pop()
                if current in component:
                    continue
                component.add(current)
                pending.extend(adjacency[current] - component)
            unvisited -= component
            components.append(component)

        if len(components) not in (1, 2):
            raise RuntimeError("Removing one bond produced an invalid component count.")

        replacement_molecules = []
        for component in sorted(components, key=min):
            component_atoms = sorted(
                (
                    member
                    for member in old_molecule.atoms
                    if member.index in component
                ),
                key=lambda item: item.index,
            )
            component_bonds = sorted(
                (
                    member_bond
                    for member_bond in remaining_bonds
                    if member_bond.node_i.index in component
                    and member_bond.node_j.index in component
                ),
                key=self._bond_sort_key,
            )
            replacement_molecules.append(
                molecule(component_atoms, component_bonds)
            )

        self._replace_molecules((old_molecule,), replacement_molecules)
        return tuple(replacement_molecules)

    def _rebuild_atom_to_molecule(self):
        """Recreate the lookup from every atom index to its molecule snapshot."""

        mapping = {}
        for species in self.molecules:
            for member in species.atoms:
                if member.index in mapping:
                    raise RuntimeError(
                        f"Atom {member.index} belongs to multiple molecules."
                    )
                mapping[member.index] = species
        if set(mapping) != set(self.atoms):
            raise RuntimeError("Molecule membership does not cover every atom.")
        self.atom_to_molecule = mapping

    def _build_pair_tables(self):
        """Build fixed candidate-pair indices and initial bond-order tables."""

        self.pair_index = tuple(combinations(sorted(self.atoms), 2))
        self.pair_to_row = {
            pair: row for row, pair in enumerate(self.pair_index)
        }
        self.current_BO = []
        self.maximum_BO = []
        for index_i, index_j in self.pair_index:
            atom1 = self.atoms[index_i]
            atom2 = self.atoms[index_j]
            current_order = self._current_bond_order(atom1, atom2)
            self.current_BO.append(current_order)
            self.maximum_BO.append(
                self._maximum_reachable_order(atom1, atom2, current_order)
            )

    def _update_pair_tables(self, atom1, atom2):
        """Incrementally update pair information after one bond action."""

        selected_pair = self._bond_key(atom1.index, atom2.index)
        selected_row = self.pair_to_row[selected_pair]
        self.current_BO[selected_row] = self._current_bond_order(atom1, atom2)

        affected_indices = {atom1.index, atom2.index}
        for row, (index_i, index_j) in enumerate(self.pair_index):
            if index_i not in affected_indices and index_j not in affected_indices:
                continue
            member_i = self.atoms[index_i]
            member_j = self.atoms[index_j]
            current_order = self.current_BO[row]
            self.maximum_BO[row] = self._maximum_reachable_order(
                member_i, member_j, current_order
            )

    def _current_bond_order(self, atom1, atom2):
        """Return current bond order, using zero when no bond exists."""

        existing_bond = self.bonds.get(
            self._bond_key(atom1.index, atom2.index)
        )
        return 0 if existing_bond is None else existing_bond.order

    def _maximum_reachable_order(self, atom1, atom2, current_order):
        """Return the largest bond order reachable in the current state."""

        additional_capacity = min(
            atom1.remaining_valence,
            atom2.remaining_valence,
            self.MAX_BOND_ORDER - current_order,
        )
        return current_order + max(0, additional_capacity)

    def _require_registered_atom(self, selected_atom):
        """Require the exact atom object registered under its index."""

        if not isinstance(selected_atom, atom):
            raise TypeError("Bond actions require atom objects.")
        if self.atoms.get(selected_atom.index) is not selected_atom:
            raise ValueError(
                f"Atom {selected_atom.index} is not registered in this oven."
            )

    @staticmethod
    def _validate_bond_change(bond_change):
        if not isinstance(bond_change, Integral) or isinstance(bond_change, bool):
            raise TypeError("bond_change must be an integer.")
        if bond_change <= 0:
            raise ValueError("bond_change must be positive.")

    @staticmethod
    def _bond_key(index_i, index_j):
        if index_i == index_j:
            raise ValueError("A bond cannot connect an atom to itself.")
        return tuple(sorted((index_i, index_j)))

    @staticmethod
    def _bond_sort_key(existing_bond):
        return tuple(sorted(existing_bond.node_indices))

    @staticmethod
    def _minimum_atom_index(species):
        return min(member.index for member in species.atoms)
