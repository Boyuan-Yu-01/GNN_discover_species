"""Node definition for the simplified C/H/O reaction system."""


class atom:
    """Represent one atom and its remaining valence."""

    MAX_VALENCE = {"C": 4, "H": 1, "O": 2}
    ELEMENT_ORDER = ("C", "H", "O")

    def __init__(self, index, element):
        element = element.strip().upper()
        if element not in self.MAX_VALENCE:
            raise ValueError("Only C, H, and O atoms are supported.")

        self.index = index
        self.element = element
        self.remaining_valence = self.MAX_VALENCE[element]

    @property
    def features(self):
        """Return [is_C, is_H, is_O, remaining_valence]."""

        element_features = [
            float(self.element == element) for element in self.ELEMENT_ORDER
        ]
        return element_features + [float(self.remaining_valence)]

    def has_sufficient_valence(self, bond_order=1):
        """Check whether the atom has enough remaining valence for a new bond."""

        return 0 < bond_order <= self.remaining_valence

    def form_bond(self, bond_order=1):
        """Reduce the remaining valence when a bond forms."""

        if not self.has_sufficient_valence(bond_order):
            raise ValueError(
                f"{self.element} atom {self.index} does not have enough "
                "remaining valence."
            )
        self.remaining_valence -= bond_order

    def break_bond(self, bond_order=1):
        """Restore remaining valence when a bond is broken."""

        new_remaining_valence = self.remaining_valence + bond_order
        if (
            bond_order <= 0
            or new_remaining_valence > self.MAX_VALENCE[self.element]
        ):
            raise ValueError(f"Invalid bond removal for atom {self.index}.")
        self.remaining_valence = new_remaining_valence
