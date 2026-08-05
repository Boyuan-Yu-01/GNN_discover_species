"""Molecule subgraph for the simplified C/H/O reaction system."""

from obj_edge import Bond
from obj_node import atom


class molecule:
    """A connected, read-only-style collection of atoms and bonds."""

    ELEMENT_ORDER = ("C", "H", "O")
    BOND_TYPE_ORDER = (
        ("C", "C", 1),
        ("C", "C", 2),
        ("C", "C", 3),
        ("C", "H", 1),
        ("C", "O", 1),
        ("C", "O", 2),
        ("C", "O", 3),
        ("O", "O", 1),
        ("O", "O", 2),
        ("H", "O", 1),
        ("H", "H", 1),
    )

    FEATURE_NAMES = (
        "n_C",
        "n_H",
        "n_O",
        "total_remaining_valence",
        "n_C-C",
        "n_C=C",
        "n_C#C",
        "n_C-H",
        "n_C-O",
        "n_C=O",
        "n_C#O",
        "n_O-O",
        "n_O=O",
        "n_O-H",
        "n_H-H",
    )

    def __init__(self, atoms, bonds=None):
        self.atoms = tuple(atoms)
        self.bonds = tuple(bonds or ())
        self._validate()

    @property
    def element_counts(self):
        """Return the number of C, H, and O atoms."""

        counts = {element: 0 for element in self.ELEMENT_ORDER}
        for atom in self.atoms:
            counts[atom.element] += 1
        return counts

    @property
    def bond_type_counts(self):
        """Return counts for every supported element-pair and bond order."""

        counts = {bond_type: 0 for bond_type in self.BOND_TYPE_ORDER}
        for bond in self.bonds:
            elements = tuple(sorted((bond.node_i.element, bond.node_j.element)))
            bond_type = (elements[0], elements[1], bond.order)
            if bond_type not in counts:
                raise ValueError(f"Unsupported bond type: {bond_type}.")
            counts[bond_type] += 1
        return counts

    @property
    def features(self):
        """Return atom counts, free valence, and bond-type counts."""

        element_counts = self.element_counts
        bond_counts = self.bond_type_counts

        values = [element_counts[element] for element in self.ELEMENT_ORDER]
        values.append(sum(atom.remaining_valence for atom in self.atoms))
        values.extend(bond_counts[bond_type] for bond_type in self.BOND_TYPE_ORDER)
        return [float(value) for value in values]

    def has_atom(self, atom):
        """Return whether this exact atom object belongs to the molecule.
        It checks whether the exact atom 'atom' object belongs to the molecule.
        """

        return any(member is atom for member in self.atoms)

    def has_bond(self, node_i, node_j):
        """Return the bond order, or 0 when the atoms are not bonded."""

        target = self._bond_key(node_i.index, node_j.index)
        for bond in self.bonds:
            if self._bond_key(*bond.node_indices) == target:
                return bond.order
        return 0

    def neighbors(self, atom):
        """Return all atoms directly bonded to the given atom."""

        if not self.has_atom(atom):
            raise ValueError(f"Atom {atom.index} does not belong to this molecule.")

        result = []
        for bond in self.bonds:
            if bond.node_i is atom:
                result.append(bond.node_j)
            elif bond.node_j is atom:
                result.append(bond.node_i)
        return result

    def is_connected(self):
        """Return whether every atom is reachable through the stored bonds."""

        adjacency = {atom.index: set() for atom in self.atoms}
        for bond in self.bonds:
            index_i, index_j = bond.node_indices
            adjacency[index_i].add(index_j)
            adjacency[index_j].add(index_i)

        visited = set()
        pending = [self.atoms[0].index]
        while pending:
            current = pending.pop()
            if current not in visited:
                visited.add(current)
                pending.extend(adjacency[current] - visited)
        return len(visited) == len(self.atoms)

    def _validate(self):
        if not self.atoms:
            raise ValueError("A molecule must contain at least one atom.")
        if any(not isinstance(item, atom) for item in self.atoms):
            raise TypeError("A molecule can only contain atom objects.")

        atom_by_index = {atom.index: atom for atom in self.atoms}
        if len(atom_by_index) != len(self.atoms):
            raise ValueError("Atom indices must be unique within a molecule.")

        bond_keys = set()
        for bond in self.bonds:
            if not isinstance(bond, Bond):
                raise TypeError("A molecule can only contain Bond objects.")
            if atom_by_index.get(bond.node_i.index) is not bond.node_i:
                raise ValueError("A bond endpoint is not contained in the molecule.")
            if atom_by_index.get(bond.node_j.index) is not bond.node_j:
                raise ValueError("A bond endpoint is not contained in the molecule.")

            key = self._bond_key(*bond.node_indices)
            if key in bond_keys:
                raise ValueError(f"Duplicate bond between atoms {key[0]} and {key[1]}.")
            bond_keys.add(key)

        if not self.is_connected():
            raise ValueError("Atoms in a molecule must form one connected subgraph.")

    @staticmethod
    def _bond_key(index_i, index_j):
        if index_i == index_j:
            raise ValueError("A bond cannot connect an atom to itself.")
        return tuple(sorted((index_i, index_j)))
