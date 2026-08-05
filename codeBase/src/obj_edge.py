"""Bond definition for the simplified C/H/O reaction system."""

from obj_node import atom


class Bond:
    """Represent a bond connecting two atomic nodes."""

    VALID_ORDERS = (1, 2, 3)

    def __init__(self, node_i, node_j, order=1):
        if not isinstance(node_i, atom) or not isinstance(node_j, atom):
            raise TypeError("A bond must connect two atom objects.")
        if node_i.index == node_j.index:
            raise ValueError("A bond cannot connect a node to itself.")
        if order not in self.VALID_ORDERS:
            raise ValueError("Bond order must be 1, 2, or 3.")

        self.node_i = node_i
        self.node_j = node_j
        self.order = order

    @property
    def features(self):
        """Return [is_single, is_double, is_triple]."""

        return [float(self.order == order) for order in self.VALID_ORDERS]

    @property
    def node_indices(self):
        """Return the indices of the two connected nodes."""

        return self.node_i.index, self.node_j.index
