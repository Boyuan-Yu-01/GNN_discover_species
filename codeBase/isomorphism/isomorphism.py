"""Check whether two matrices are equal under allowed row/column swaps."""

from collections import Counter
from itertools import permutations, product


def isomorphism_0(mtx1, mtx2, ranges):
    """Original method: try every allowed row/column ordering."""
    n = len(mtx1)

    # Both inputs must be square matrices of the same size.
    if (
        len(mtx2) != n
        or any(len(row) != n for row in mtx1)
        or any(len(row) != n for row in mtx2)
    ):
        return False

    groups = []                 # creates an empty list for the parsed groups
    for item in ranges:         # for one-element itme , it treats the index as fixed and adds it to the groups list. For two-element item, it creates a list of indices from the first to the second element (inclusive) and adds that list to the groups list. If an item has more than two elements, it raises a ValueError.
        if len(item) == 1:
            groups.append([item[0]])
        elif len(item) == 2:    # creates a list of indices from the first to the second element (inclusive) and adds that list to the groups list.
            groups.append(list(range(item[0], item[1] + 1)))
        else:
            raise ValueError("Each range must contain one index or two endpoints")

    # CHECK: The ranges must describe every matrix index exactly once.
    indices = [index for group in groups for index in group]
    if sorted(indices) != list(range(n)):
        raise ValueError("Ranges must cover every matrix index exactly once")

    # Try every allowed ordering. The same ordering is used for rows and columns.
    for group_orderings in product(*(permutations(group) for group in groups)):
        order = [index for group in group_orderings for index in group]
        if all(mtx1[order[i]][order[j]] == mtx2[i][j]
               for i in range(n) for j in range(n)):
            return True

    return False


def isomorphism_1(mtx1, mtx2, ranges):
    """Return True if allowed simultaneous row/column swaps make the matrices equal.

    Each item in ``ranges`` is an inclusive range. For example, ``[0, 3]``
    allows indices 0, 1, 2, and 3 to be rearranged, while ``[4]`` fixes
    index 4 in place.
    """
    n = len(mtx1)

    # Both inputs must be square matrices of the same size.
    if (
        len(mtx2) != n
        or any(len(row) != n for row in mtx1)
        or any(len(row) != n for row in mtx2)
    ):
        return False

    groups = []                 # creates an empty list for the parsed groups
    for item in ranges:         # parses every allowed range so that the function knows which indices can be exchanged with each other.
        if len(item) == 1:
            groups.append([item[0]])
        elif len(item) == 2:    # creates a list of indices from the first endpoint to the second endpoint, including both endpoints.
            groups.append(list(range(item[0], item[1] + 1)))
        else:
            raise ValueError("Each range must contain one index or two endpoints")

    # CHECK: The ranges must describe every matrix index exactly once.
    indices = [index for group in groups for index in group]    
    if sorted(indices) != list(range(n)):
        raise ValueError("Ranges must cover every matrix index exactly once")

    # COLOR: Give each index the number of its allowed group. Indices with different colors cannot be matched because they are not allowed to swap with each other.
    group_of = [0] * n
    for group_number, group in enumerate(groups):
        for index in group:
            group_of[index] = group_number

    colors1 = group_of[:]
    colors2 = group_of[:]

    # REFINE: Repeatedly update each color by looking at the index's row, column, diagonal value, and the current colors of its neighbors. This separates indices that have different matrix properties.
    while True:
        def signatures(matrix, colors):
            result = []
            for i in range(n):
                outgoing = sorted((colors[j], matrix[i][j]) for j in range(n))
                incoming = sorted((colors[j], matrix[j][i]) for j in range(n))
                result.append((colors[i], matrix[i][i],
                               tuple(outgoing), tuple(incoming)))
            return result

        signatures1 = signatures(mtx1, colors1)
        signatures2 = signatures(mtx2, colors2)

        # Give identical signatures the same new integer color. The signatures from both matrices are processed together so that their color numbers are directly comparable.
        color_numbers = {}
        for signature in signatures1 + signatures2:
            if signature not in color_numbers:
                color_numbers[signature] = len(color_numbers)

        new_colors1 = [color_numbers[s] for s in signatures1]
        new_colors2 = [color_numbers[s] for s in signatures2]

        # CHECK: Different color counts prove immediately that no allowed mapping exists, so there is no reason to begin the slower search.
        if Counter(new_colors1) != Counter(new_colors2):
            return False

        # STOP: Refinement only splits color classes. If no class was split during this pass, repeating the procedure cannot provide more information.
        old_count = len(set(colors1 + colors2))
        new_count = len(set(new_colors1 + new_colors2))
        colors1, colors2 = new_colors1, new_colors2
        if new_count == old_count:
            break

    # SEARCH: Match only indices with the same final color. Unlike isomorphism_0, this method rejects a partial ordering as soon as one matrix entry disagrees.
    mapping = {}       # index in mtx1 -> index in mtx2
    used = set()

    def compatible(i, j):
        # CHECK: The new pair must have equal diagonal entries and equal entries in both directions to every pair that has already been matched.
        if mtx1[i][i] != mtx2[j][j]:
            return False
        return all(
            mtx1[i][old_i] == mtx2[j][old_j]
            and mtx1[old_i][i] == mtx2[old_j][j]
            for old_i, old_j in mapping.items()
        )

    def search():
        if len(mapping) == n:
            return True

        # CHOOSE: Find the unmatched index with the fewest compatible choices. Trying the most restricted index first usually makes an incorrect branch fail quickly.
        best_i = None
        best_candidates = None
        for i in range(n):
            if i in mapping:
                continue
            candidates = [
                j for j in range(n)
                if j not in used
                and colors1[i] == colors2[j]
                and compatible(i, j)
            ]
            if not candidates:
                return False
            if best_candidates is None or len(candidates) < len(best_candidates):
                best_i, best_candidates = i, candidates

        for j in best_candidates:
            mapping[best_i] = j
            used.add(j)
            if search():
                return True
            used.remove(j)
            del mapping[best_i]

        return False

    return search()


if __name__ == "__main__":
    mtx1 = [
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
        [1, 1, 1, 0, 0, 1],
        [0, 0, 0, 1, 1, 0],
    ]
    mtx2 = [
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 1],
        [0, 0, 1, 0, 1, 0],
    ]

    print(isomorphism_0(mtx1, mtx2, [[0, 3], [4], [5]]))  # True
    print(isomorphism_1(mtx1, mtx2, [[0, 3], [4], [5]]))  # True
