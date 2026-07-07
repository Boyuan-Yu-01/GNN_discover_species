# ChemKin Bond Dataset Builder

This directory converts ChemKin mechanism species labels into approximate
molecular bond-count datasets. The workflow has three parts:

1. `identify_bonds.py` reads one ChemKin mechanism file and estimates species
   formulae, bond counts, ring flags, and formula degeneracies.
2. `master_dataset_merger.py` merges the formula degeneracies from multiple
   mechanism outputs into one clash-safe master dataset.
3. `build_bond_dataset_from_chemkin.py` is the run script. It builds every
   per-mechanism JSON file first, then builds the master JSON file.

## Quick Start

Put ChemKin mechanism files in:

```text
mechanism_file_chemkin/
```

Then run:

```bash
python3 build_bond_dataset_from_chemkin.py
```

The output files are written to:

```text
output/
```

The run script generates:

```text
<mechanism_name>_subgraph_info.json
log_<mechanism_name>.txt
master_dataset.json
log_master_dataset.txt
```

## `identify_bonds.py`

`identify_bonds.py` contains the class:

```python
identify_bonds_from_chemikin_mechanism
```

The class reads species from a ChemKin mechanism and estimates a subgraph-like
description for each species. In this project, the atom graph is not fully
known because the mechanism files do not contain SMILES strings, adjacency
lists, or connection tables. Therefore, the code combines direct name parsing,
exact lookup rules, and chemically motivated heuristics.

### Main Ideas

`read_chemkin_species()` extracts the species labels from the ChemKin `SPECIES`
block. It ignores comments and stops at the end of the species section.

`parse_formula()` converts species labels such as `nc12h26`, `c7h15-1`, or
`ch3cho` into formula strings such as `C12H26`, `C7H15`, or `C2H4O`.

`read_species_backbone()` is the main method. It returns a dictionary with:

```text
species
formula_degeneracies
raw_species_count
multi_species_labels
```

`species` contains one entry per unique ChemKin species label.

`formula_degeneracies` groups species that have the same formula, bond counts,
and ring flag. A formula can have multiple degeneracies because the same
formula may describe different structures. For example, one `C7H14O`
degeneracy may be a carbonyl species, while another may be a cyclic ether.

`raw_species_count` records how many species labels were read from the ChemKin
file before unique-label processing.

`multi_species_labels` records labels that may represent more than one real
molecule. This can happen when a mechanism uses one lumped label for several
structural isomers.

`export_species_backbone_to_json()` writes the full per-mechanism result to a
JSON file.

`write_species_backbone_log()` writes a readable log with species counts,
formula-degeneracy counts, ambiguous formulae, and sanity-check information.

### Bond Keys

The generated bond-count dictionaries use a fixed set of bond keys:

```text
C-C, C=C, C#C
C-N, C=N, C#N
C-O, C=O, C#O
N-N, N=N, N#N
O-O, O=O
O-N, N=O
C-H, N-H, O-H, H-H
```

Missing bonds are written as zero so downstream scripts can safely compare
entries.

### Important Limitation

The result is a best-effort structural estimate, not a guaranteed molecular
graph. Without SMILES or adjacency lists, some species labels require
heuristics. Exact species rules are included for labels where the name carries
clear structural information or where a previous check identified a better
interpretation.

## `master_dataset_merger.py`

`master_dataset_merger.py` contains the class:

```python
master_dataset_merger
```

This class merges several per-mechanism JSON files into one master dataset.
It only keeps:

```text
formula_degeneracies
```

Everything else from the per-mechanism JSON files is discarded.

### Why the Merge Needs Care

Degeneracy IDs such as `C5H10O_001` are local to each input JSON file. The same
ID in two files does not always mean the same structure. The merger therefore
does not trust the source ID.

Instead, it compares entries by the signature:

```text
formula + normalized bond_counts + is_ring
```

If two input entries have the same signature, they are merged into one master
entry and their species lists are combined.

If two input entries have the same formula but different bond counts or ring
flags, they become separate master degeneracies with fresh IDs such as:

```text
C5H10O_001
C5H10O_002
C5H10O_003
```

### Master Output

The master JSON has this shape:

```json
{
  "formula_degeneracies": {
    "C5H10O_001": {
      "formula": "C5H10O",
      "degeneracy_id": "C5H10O_001",
      "bond_counts": {},
      "is_ring": false,
      "species": []
    }
  }
}
```

The master log records:

```text
input files
input formula-degeneracy count
master formula-degeneracy count
merged duplicate signatures
ambiguous formulae
invalid entries skipped
source degeneracy ID clashes
species signature clashes
```

`source degeneracy ID clashes` are not automatically errors. They usually mean
two files reused the same local ID for different signatures. The merger handles
this by assigning fresh master IDs.

`species signature clashes` are more serious. They mean the same species label
was mapped to different signatures in different input files.

## `build_bond_dataset_from_chemkin.py`

`build_bond_dataset_from_chemkin.py` is the main run script. It is intentionally
minimal and runs the whole pipeline from top to bottom.

It performs two stages:

1. Build one subgraph-info JSON file for each ChemKin `.txt` file in
   `mechanism_file_chemkin/`.
2. Merge those per-mechanism JSON files into `output/master_dataset.json`.

First, it loops over every mechanism file and generates:

```text
output/<mechanism_name>_subgraph_info.json
output/log_<mechanism_name>.txt
```

Then it calls `master_dataset_merger` to generate:

```text
output/master_dataset.json
output/log_master_dataset.txt
```

This is the recommended entry point when you add, remove, or update mechanism
files.

## Typical Workflow

1. Add or update mechanism files in `mechanism_file_chemkin/`.
2. Run:

```bash
python3 build_bond_dataset_from_chemkin.py
```

3. Check the per-mechanism logs for parsing issues.
4. Check `output/log_master_dataset.txt` for merge clashes.
5. Use `output/master_dataset.json` as the combined dataset.

## Interpreting Formula Degeneracies

A formula degeneracy is one structural interpretation of a molecular formula.
For example, `C5H10O_001` and `C5H10O_002` have the same formula but different
bond-count signatures or ring states.

This is useful for GNN species-discovery work because the same formula can
represent multiple chemically different subgraphs. The master dataset keeps
those structural alternatives separate instead of collapsing everything by
formula alone.
