# Pseudo-Negative Species Generator

`pseudo_negative_generator.py` creates synthetic molecular samples for training
or evaluating species-survival models when true negative species are not
available.

The generator builds an explicit atom-and-bond graph, checks its valence and
connectivity, and then converts it to the same aggregate bond-count feature
format used by this project's ChemKin datasets. Generated samples are rejected
if their formula, bond counts, and ring flag already occur in the configured
positive reference data.

> A pseudo-negative is not proof that a molecule is impossible. It is a
> valence-valid synthetic structure intended to represent an unobserved or
> potentially less-survivable molecular pattern.

## Requirements
- The build script explicitly uses these positive-reference datasets:

```text
output/c8-c16_n-alkanes_LLNL_subgraph_info.json
output/nHeptane_LLNL_subgraph_info.json
```

`build_pseudo_dataset.py` constructs these paths relative to its own location.
Direct library users must explicitly pass their intended reference paths.

## Quick Start

For the standard recipe, run the companion build script:

```bash
python build_pseudo_dataset.py
```

It invokes all four generator families and writes
`output/pseudo_negative_dataset.json` and
`output/log_pseudo_negative_dataset.txt`. The default parameters are
intentionally kept directly in the script so the dataset recipe is visible and
easy to edit.

For direct library use, import the class and call the required generators:

```python
from pseudo_negative_generator import PseudoNegativeGenerator


generator = PseudoNegativeGenerator(
    n=200,
    reference_paths=[
        "output/c8-c16_n-alkanes_LLNL_subgraph_info.json",
        "output/nHeptane_LLNL_subgraph_info.json",
    ],
    random_seed=42,
    max_attempts=500,
)

generator.long_carbon_chain(
    chain_length=18,
    double_bond_seed=3,
    triple_bond_seed=1,
    O_seed=3,
)
generator.carbon_ring(
    ring_size=10,
    double_bond_seed=3,
    triple_bond_seed=1,
    O_seed=2,
)
generator.highly_branched_carbon(
    branch_to_main_chain_ratio=0.5,
    double_bond_seed=2,
    triple_bond_seed=1,
    O_seed=2,
    main_chain_length=10,
)
generator.long_oxygen_chain(
    chain_length=6,
    double_bond_seed=2,
)

output_path = generator.export_to_json(
    "output/pseudo_negative_samples.json"
)

print(output_path)
print(generator.summary())
```

The preferred PEP 8 class alias is `PseudoNegativeGenerator`. The underlying
class uses the requested snake-case name:

```python
from pseudo_negative_generator import pseudo_negative_generator
```

## Important Behavior

`n` is a global limit for one generator instance. Results from every generation
method accumulate in `generator.generated_samples`, but the combined list never
contains more than `n` samples.

Each generation method returns only the samples added by that particular call:

```python
new_samples = generator.long_carbon_chain(18, 3, 2)
all_samples = generator.generated_samples
```

If the global limit has already been reached, or if every candidate is invalid,
duplicated, or observed in the reference data, the method returns an empty
list.

The parameters named `double_bond_seed`, `triple_bond_seed`, and `O_seed` are
upper bounds. They are not random-number seeds:

- `double_bond_seed=3` explores structures containing zero through three
  C=C bonds, subject to topology and valence constraints.
- `triple_bond_seed=1` explores structures containing zero or one C#C bond.
- `O_seed=2` explores structures containing zero through two added oxygen
  atoms.
- `random_seed` in the constructor controls reproducibility.

## Constructor

```python
PseudoNegativeGenerator(
    n,
    reference_paths,
    random_seed=None,
    max_attempts=None,
)
```

### `n`

A positive integer giving the maximum number of unique samples retained across
all method calls.

### `random_seed`

An optional integer used by a private random-number generator. The same seed,
constructor arguments, and method-call sequence produce identical sample
content and ordering.

### `reference_paths`

The required path or iterable of JSON paths containing positive species data.
The generator reads records from the `species` and
`formula_degeneracies` sections.

The generator does not pre-specify positive datasets. Omitting this required
argument raises `TypeError`; explicitly passing `None` raises `ValueError`.

For isolated experiments or tests, positive-reference rejection can be
disabled explicitly:

```python
generator = PseudoNegativeGenerator(
    n=100,
    random_seed=42,
    reference_paths=[],
)
```

When reference loading is disabled, only paths explicitly configured on that
generator instance receive overwrite protection.

### `max_attempts`

A positive integer limiting the number of candidate count configurations
examined by each public generation call. The default is:

```python
max(100, n * 50)
```

Branched-carbon generation also uses a bounded dynamic-programming search. Its
state limit is derived from `max_attempts`. This prevents unusually large or
impossible requests from searching forever.

## Generation Methods

### Long carbon chains

```python
long_carbon_chain(
    chain_length,
    double_bond_seed,
    O_seed,
    triple_bond_seed=0,
)
```

- `chain_length` is the number of carbon atoms in the connected main chain and
  must be at least 2.
- The structure is acyclic.
- Up to `double_bond_seed` C-C bonds are promoted to C=C.
- Up to `triple_bond_seed` C-C bonds are promoted to C#C.
- Triple bonds cannot be placed next to another multiple bond when that would
  make a carbon exceed valence 4.
- Up to `O_seed` oxygen atoms are added as hydroxyl (`C-O-H`) or carbonyl
  (`C=O`) motifs.
- Remaining valences are filled with hydrogen.

### Carbon rings

```python
carbon_ring(ring_size, double_bond_seed, O_seed, triple_bond_seed=0)
```

- `ring_size` is the number of carbon atoms in the primary ring and must be at
  least 3.
- The output has `is_ring: true`.
- Ring C=C and C#C bonds are non-conflicting: promoted multiple bonds cannot
  share a carbon atom.
- Oxygen is attached to ring carbon as hydroxyl or carbonyl motifs; ring carbon
  is not silently replaced with oxygen.

### Highly branched carbon structures

```python
highly_branched_carbon(
    branch_to_main_chain_ratio,
    double_bond_seed,
    O_seed,
    main_chain_length=6,
    triple_bond_seed=0,
)
```

- `main_chain_length` must be at least 2.
- `branch_to_main_chain_ratio` must be in `(0, 1]`.
- Branch count uses positive half-up rounding:

```text
branch_count = max(1, floor(main_chain_length * ratio + 0.5))
```

- Branches are one-carbon branches connected to the main chain.
- Every accepted topology contains a real branch point: at least one carbon
  has three or more carbon neighbors.
- Branch placement and double-bond placement are selected together so a random
  early choice does not accidentally discard an otherwise valid aggregate
  signature.
- C=C and C#C placement are solved together with branch placement so carbon
  valence never exceeds 4.
- The structure is acyclic and has `is_ring: false`.

The original three-positional-argument call remains valid and uses a six-carbon
main chain:

```python
generator.highly_branched_carbon(0.5, 2, 2)
```

### Long oxygen chains

```python
long_oxygen_chain(chain_length, double_bond_seed)
```

- `chain_length` is the number of consecutive oxygen atoms and must be at
  least 2.
- Remaining terminal oxygen valence is filled with hydrogen.
- Oxygen valence is limited to 2.
- An O=O bond is possible only for a two-oxygen chain. In a longer connected
  chain, an O=O promotion would give at least one oxygen valence 3 and is
  therefore rejected.

Long O-O chains are intentionally allowed as pseudo-negative structures.
Valence validity is required, but experimental stability is not implied.

## Chemical and Graph Rules

Every accepted sample obeys these invariants:

- H has maximum valence 1 and forms only single bonds.
- O has maximum total bond order 2.
- C has maximum total bond order 4.
- Bond orders are 1, 2, or 3.
- The graph has no self-bonds or duplicate edges.
- The graph is connected.
- All remaining valences are filled with hydrogen.
- Formula and bond counts are derived from the final graph rather than updated
  independently.
- The stored `is_ring` flag must agree with graph cycle detection.

The generator currently creates only carbon, oxygen, and hydrogen atoms.
Nitrogen-related bond keys remain in every feature vector with value zero so
the output schema stays compatible with the positive datasets.

## Duplicate and Positive-Sample Rejection

The canonical identity of a sample is:

```text
formula + ordered bond_counts + is_ring
```

Before accepting a candidate, the generator rejects it if that identity:

1. is already present in `generated_samples`; or
2. occurs in any configured reference dataset.

Atom numbering, randomized placement, notes, and generator method are not part
of the identity. Two different atom-level placements that reduce to identical
model features are one degeneracy because the downstream model cannot
distinguish them.

## Generated Sample Schema

Each item in `generated_samples` has this structure:

```json
{
  "sample_id": "pseudo_C18H32O2_000001",
  "formula": "C18H32O2",
  "bond_counts": {
    "C-C": 15,
    "C=C": 2,
    "C=O": 1,
    "C-O": 1,
    "C-H": 31,
    "O-H": 1
  },
  "is_ring": false,
  "generator": "long_carbon_chain",
  "parameters": {
    "chain_length": 18,
    "double_bond_seed": 3,
    "triple_bond_seed": 1,
    "O_seed": 2
  },
  "is_pseudo_negative": true,
  "confidence": "synthetic",
  "notes": [
    "acyclic main chain with 18 carbon atoms",
    "description of generated motifs and positions"
  ],
  "topology": {
    "atoms": ["C", "C", "O", "H"],
    "bonds": [[0, 1, 1], [1, 2, 2], [0, 3, 1]]
  }
}
```

The example is abbreviated for readability. Actual `bond_counts` contains the
complete project bond-key vocabulary, including zero-valued keys, and actual
`topology` contains every atom and bond.

Topology bond entries use:

```text
[left_atom_index, right_atom_index, bond_order]
```

Atom indices are zero-based. Bond order is `1` for single, `2` for double, and
`3` for triple.

Formulae use Hill order: carbon, hydrogen, and then other elements
alphabetically.

## Exporting JSON

```python
path = generator.export_to_json(
    "output/pseudo_negative_samples.json",
    indent=2,
)
```

The method:

- creates missing parent directories;
- writes deterministic, human-readable JSON;
- writes exactly the same schema as `output/master_dataset.json`;
- assigns stable degeneracy IDs separately within each formula;
- stores the internal synthetic `sample_id` in the required `species` list;
- refuses to overwrite configured positive-reference datasets;
- returns the resolved `pathlib.Path` of the output file.

The exported file contains no generator metadata, topology, notes, or
parameters because those fields are not part of the master-dataset schema. Use
`summary()` before or after export when generation diagnostics are needed.

The exported structure is:

```json
{
  "formula_degeneracies": {
    "C18H32O2_001": {
      "formula": "C18H32O2",
      "degeneracy_id": "C18H32O2_001",
      "bond_counts": {
        "C-C": 15,
        "C=C": 2,
        "C=O": 1,
        "C-O": 1,
        "C-H": 31,
        "O-H": 1
      },
      "is_ring": false,
      "species": ["pseudo_C18H32O2_000001"]
    }
  }
}
```

The displayed `bond_counts` is abbreviated. Actual exports include every bond
key, including keys whose value is zero.

## Writing the Generation Log

```python
log_path = generator.write_generation_log(
    json_output_path="output/pseudo_negative_dataset.json",
    log_file="output/log_pseudo_negative_dataset.txt",
)
```

The log summarizes:

- formula-degeneracy, formula, backbone, and ring counts;
- accepted groups for each generator family;
- maximum and nonzero counts for every non-H backbone bond;
- attempted, accepted, invalid, duplicate, and observed-positive counts;
- random seed, sample limit, remaining capacity, and reference files; and
- the formulae with the most generated degeneracies.

The method creates the log directory if needed and returns the resolved
`pathlib.Path` of the log file.

## Diagnostics

Use `summary()` to inspect cumulative generation results:

```python
print(generator.summary())
```

Example:

```python
{
    "attempted": 245,
    "rejected_invalid": 15,
    "rejected_duplicate": 12,
    "rejected_observed": 18,
    "accepted": 200,
    "generated_sample_count": 200,
    "remaining_capacity": 0,
}
```

The counters accumulate across calls on the same generator instance.

## Exceptions

`ValueError` is raised for invalid public arguments, including:

- `n <= 0`;
- negative `double_bond_seed`, `triple_bond_seed`, or `O_seed`;
- carbon/oxygen chain length below 2;
- ring size below 3;
- branch ratio outside `(0, 1]`;
- non-integer count parameters, including Boolean values;
- `max_attempts <= 0`;
- malformed reference-record fields; or
- an attempt to export over a reference dataset.

`FileNotFoundError` is raised when a configured reference path does not exist.

Invalid or impossible generated candidates are rejected internally and do not
raise exceptions during normal generation.

