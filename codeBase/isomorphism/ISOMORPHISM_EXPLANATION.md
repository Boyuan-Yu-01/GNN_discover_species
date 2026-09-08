# How the matrix-isomorphism functions work

## Problem definition

The functions determine whether `mtx1` can be changed into `mtx2` by applying the same allowed permutation to its rows and columns. For an adjacency matrix, this is the same as relabeling vertices without changing the graph.

For example:

```python
ranges = [[0, 6], [7, 9]]
```

means that indices `0` through `6` may exchange labels with one another, and indices `7` through `9` may exchange labels with one another. An index from the first group may not exchange labels with an index from the second group.

Both functions are exact: they return `True` only when such a permutation exists.

## Shared input checks

Both functions first verify that the matrices are square and have the same size. They then expand every inclusive range. Thus, `[0, 3]` becomes `[0, 1, 2, 3]`, while `[4]` remains `[4]`.

The expanded groups must contain every matrix index exactly once. Missing or repeated indices cause a `ValueError`, because the set of allowed permutations would otherwise be ambiguous.

## `isomorphism_0`: complete permutation enumeration

`isomorphism_0` is the original method.

1. It generates every permutation inside each allowed group.
2. It combines one permutation from each group into a complete ordering.
3. It applies that ordering simultaneously to the rows and columns of `mtx1`.
4. It returns `True` when every resulting entry equals the corresponding entry of `mtx2`.
5. It returns `False` after all allowed orderings have failed.

For a candidate ordering called `order`, the comparison is:

```python
mtx1[order[i]][order[j]] == mtx2[i][j]
```

Using `order` for both subscripts is important: a row and its corresponding column represent the same vertex and must always move together.

If the group sizes are `k1`, `k2`, and so on, this method may test

```text
k1! * k2! * ...
```

complete orderings. This is easy to understand but becomes slow when a group is large.

## `isomorphism_1`: refinement and constrained search

`isomorphism_1` avoids generating all complete orderings in advance. It has two main stages.

### 1. Color refinement

Each matrix index initially receives a color representing its allowed group. Different initial colors can never be matched.

The function then constructs a signature for each index from:

- Its current color
- Its diagonal matrix entry
- The values in its row, paired with the colors of their column indices
- The values in its column, paired with the colors of their row indices

Indices with identical signatures receive the same new color. This process repeats until it cannot divide the indices into any smaller classes.

For a graph adjacency matrix, these signatures capture properties such as vertex degree, permitted atom type, and increasingly detailed information about neighboring vertices. If the two matrices have different numbers of indices of any color, they cannot be isomorphic and the function immediately returns `False`.

Color refinement is a rejection test and a search reduction. Matching signatures alone do not always prove isomorphism, because highly symmetric non-isomorphic graphs can have the same signatures.

### 2. Exact backtracking search

After refinement, an index in `mtx1` is considered only for unused indices in `mtx2` with the same color.

Whenever the search proposes a match `i -> j`, it checks:

```python
mtx1[i][i] == mtx2[j][j]
mtx1[i][old_i] == mtx2[j][old_j]
mtx1[old_i][i] == mtx2[old_j][j]
```

for every previously established match `old_i -> old_j`. Checking both directions also supports matrices that are not symmetric.

If any entry differs, that partial mapping is abandoned immediately. The search also chooses the unmatched index with the fewest currently compatible candidates. This heuristic tends to expose contradictions early.

When all indices have been matched consistently, every matrix entry is preserved under that mapping, so the function returns `True`. If every possible branch fails, it returns `False`.

## Why the new method is usually faster

The original function constructs every allowed complete permutation, even when a mismatch could have been detected from simple matrix properties. The new function uses those properties first and only explores mappings that remain possible.

For chemical adjacency matrices, atom groups, degree differences, and local bonding environments commonly divide the vertices into small color classes. The remaining search can therefore be much smaller than the factorial search performed by `isomorphism_0`.

The worst case can still be exponential. Matrices representing highly symmetric graphs may leave many indistinguishable candidates. This is inherent to the general graph-isomorphism problem, but `isomorphism_1` should be substantially quicker for typical molecular structures.

## Usage

```python
from isomorphism import isomorphism_0, isomorphism_1

ranges = [[0, 6], [7, 9]]

print(isomorphism_0(mtx1, mtx2, ranges))  # Original method
print(isomorphism_1(mtx1, mtx2, ranges))  # Optimized method
```

`isomorphism_0` is useful as a simple reference implementation. `isomorphism_1` is the recommended function for larger inputs.
