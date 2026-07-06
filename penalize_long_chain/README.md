# Species Backbone Extraction From ChemKin Mechanisms

This folder contains `identify_species_backbone.py`, a small heuristic parser for
ChemKin mechanism files. The script reads the `SPECIES` block from each mechanism
file, estimates simple molecular backbone features for every species name, and
returns the result as a Python dictionary.

The main goal is to create species-level features that can later be used in a GNN
or reinforcement-learning workflow, for example to penalize unrealistic long
hydrocarbon chains or identify species with multiple-bond, heteroatom, or ring
patterns.

## Main Script

```text
identify_species_backbone.py
```

The main class is:

```python
identify_backbone_from_chemkin_mechanism
```

Example:

```python
from identify_species_backbone import identify_backbone_from_chemkin_mechanism

reader = identify_backbone_from_chemkin_mechanism(
    "mechanism_file_chemkin/Mech_JetSurF2.0.txt"
)

species = reader.read_chemkin_species()
backbone = reader.read_species_backbone()

print(len(species))
print(backbone["species"]["H2"])
```

## What The Script Extracts

For each species, the script estimates:

- formula, such as `C12H26`, `H2O`, or `CH3O`
- counts of selected bond types
- whether the species is likely a ring
- confidence level, either exact rule or name heuristic
- notes explaining which heuristic was used
- formula-variant ID, such as `C4H6_001`
- whether a generic species label may represent multiple molecular structures

The tracked bond keys are:

```python
(
    "C-C", "C=C", "C#C",
    "C-N", "C=N", "C#N",
    "C-O", "C=O", "C#O",
    "N-N", "N=N", "N#N",
    "O-O", "O=O", "O-N", "N=O",
    "C-H", "N-H", "O-H", "H-H",
)
```

Here `-` means single bond, `=` means double bond, and `#` means triple bond.

## Why `EXACT_SPECIES` Exists

ChemKin mechanism files usually list species names, not full molecular graphs.
That means the parser often sees a label such as `CH3O`, `CH2OH`, `CO`, or `N2`,
but does not receive explicit bond connectivity.

`EXACT_SPECIES` is a small chemistry lookup table for common species where the
bonding should be defined directly instead of guessed from formula alone.

Examples:

```python
"h2": {"formula": "H2", "bonds": {"H-H": 1}, "is_ring": False}
"n2": {"formula": "N2", "bonds": {"N#N": 1}, "is_ring": False}
"co": {"formula": "CO", "bonds": {"C#O": 1}, "is_ring": False}
"ch2oh": {"formula": "CH3O", "bonds": {"C-O": 1, "C-H": 2, "O-H": 1}, "is_ring": False}
"ch3o": {"formula": "CH3O", "bonds": {"C-O": 1, "C-H": 3}, "is_ring": False}
```

This is important because two species can have the same formula but different
bonding. `CH2OH` and `CH3O` both have formula `CH3O`, but they should not have
the same bond-count features.

## Main Workflow

1. `read_chemkin_species()`

   Reads the `SPECIES` block from a ChemKin file. It removes `!` comments,
   stops at `END`, `THERMO`, or `REACTIONS`, and stores only unique species
   labels. It also records `raw_species_count` before duplicate filtering.

2. `read_species_backbone()`

   Estimates formula and bond-count information for every parsed species.
   It groups species by formula and bond signature, then assigns separate
   formula-variant IDs such as `C3H4_001`, `C3H4_002`, and `C3H4_003`.

3. Exact species lookup

   If a species name appears in `EXACT_SPECIES`, the script uses the stored
   formula, bond counts, and ring flag directly.

4. Formula and heuristic parsing

   If no exact rule exists, the script infers the formula from the species name
   and applies name-based heuristics for carbon skeletons, rings, heteroatoms,
   peroxide groups, hydroperoxide groups, carbonyl groups, and hydrogen bonds.

5. Ambiguity reporting

   If the same formula appears with multiple bond patterns, the script stores
   all variants separately and reports the formula as ambiguous.

## Worked Example: `C12H26`

`C12H26` is useful as a simple long-chain hydrocarbon example. It is not listed
in `EXACT_SPECIES`, so the class interprets it using the name/formula heuristics.

If the mechanism contains `C12H26` directly, or a normal-chain label such as
`nc12h26`, the result is the same:

```python
{
    "formula": "C12H26",
    "bond_counts": {
        "C-C": 11,
        "C-H": 26,
        # all other BOND_KEYS are zero
    },
    "is_ring": False,
    "confidence": "estimated_from_name",
    "notes": [],
}
```

The class reaches this result as follows:

1. `_clean_species_name("C12H26")`

   Removes whitespace and `*` markers. The name remains `C12H26`.

2. Exact lookup

   The lowercase name `c12h26` is not in `EXACT_SPECIES`, so the script does not
   use a manually stored molecular graph.

3. `_infer_formula_counts("C12H26")`

   Reads the formula from the species label:

   ```python
   {"C": 12, "H": 26}
   ```

   For `nc12h26`, `_strip_non_formula_prefix()` removes the leading `n` because
   it means normal-chain, not nitrogen. The parsed formula is still `C12H26`.

4. `_formula_string({"C": 12, "H": 26})`

   Converts the element-count dictionary back into:

   ```python
   "C12H26"
   ```

5. `_looks_like_ring("C12H26")`

   Returns `False`, because there is no cyclic naming marker such as `c-`,
   `cC`, or a cyclic ether pattern.

6. `_estimate_carbon_skeleton_bonds(...)`

   The molecule has 12 carbon atoms, is not a ring, and does not look like an
   alkene or alkyne. Therefore the script treats it as a saturated open-chain
   hydrocarbon:

   ```python
   C-C = carbon_count - 1 = 12 - 1 = 11
   ```

7. `_add_heteroatom_bond_heuristics(...)`

   No O or N pattern is present, so this step adds nothing.

8. `_add_hydrogen_bond_heuristics(...)`

   Since carbon is present, all 26 hydrogens are assigned as C-H bonds:

   ```python
   C-H = 26
   ```

So chemically, the class interprets `C12H26` as an open-chain saturated alkane
with 11 C-C single bonds and 26 C-H single bonds. This is a good coarse feature
for penalizing long saturated hydrocarbons, but it does not distinguish exact
connectivity such as straight-chain dodecane versus branched C12 isomers unless
the species name gives enough structural information.

## Function Reference

### Class: `identify_backbone_from_chemkin_mechanism`

This class owns the full workflow for one ChemKin mechanism file. It stores the
mechanism path, extracts species names, estimates bond-count features, and builds
the final dictionary used by downstream analysis.

### Public Methods

`__init__(chemkin_file)`

Initializes the reader with one ChemKin mechanism file path. It also creates
empty containers for `species`, `species_backbone`, and `raw_species_count`.

`read_chemkin_species()`

Reads only the `SPECIES` block of the ChemKin file. It removes comments after
`!`, ignores blank lines, stops when it reaches `END`, `THERMO`, or `REACTIONS`,
and returns a unique species-name list. The variable `raw_species_count` records
how many species labels appeared before duplicate filtering.

`read_species_backbone()`

Builds the full species-backbone dictionary. For each species, it estimates the
formula, bond counts, ring flag, confidence level, and notes. It then groups
species by formula and bond signature, assigns formula-variant IDs such as
`C4H6_001`, records ambiguous formulas, and marks generic labels that may
represent multiple molecular structures.

`export_species_backbone_to_json(output_file, indent=2)`

Exports the full species-backbone dictionary to a JSON file. If
`read_species_backbone()` has not been called yet, this method calls it
automatically before writing. It creates parent directories if needed and returns
the resolved output path.

### Signature And Variant Helpers

`_bond_signature(info)`

Turns one species result into a hashable signature. The signature contains all
bond counts in the fixed `BOND_KEYS` order plus the `is_ring` flag. This lets the
script decide whether two species with the same formula have the same estimated
backbone.

`_signature_to_bond_counts(signature)`

Converts a stored signature back into a normal `bond_counts` dictionary and
`is_ring` value. This is used when building the `formula_variants` output.

`_formula_variant_id(formula, variant_number)`

Creates stable formula-variant IDs. For example, the first estimated structure
for `C4H6` becomes `C4H6_001`, the second becomes `C4H6_002`, and so on.

`_is_generic_species_label(species_name, formula)`

Checks whether a species name is just a bare formula label, such as `C6H12`.
Bare labels can hide structural ambiguity when the same formula appears with
multiple bond patterns. More explicit names such as `aC3H4`, `pC3H4`, or
`cC3H4` are treated as named species rather than generic formula buckets.

### Bond-Count Dictionary Helpers

`_empty_bond_counts()`

Creates a zero-filled dictionary containing every tracked bond key. This ensures
all species have the same feature dimensions even when most bond counts are zero.

`_with_empty_bond_counts(partial_counts)`

Starts from `_empty_bond_counts()` and inserts known nonzero counts. This is used
for `EXACT_SPECIES`, where only the nonzero bonds are written manually.

`_clean_species_name(species_name)`

Normalizes a species label before parsing. Currently it strips whitespace and
removes `*`, which sometimes appears in mechanism labels.

`_formula_string(formula_counts)`

Converts an element-count dictionary into a compact formula string. For example,
`{"C": 2, "H": 6, "O": 1}` becomes `C2H6O`.

### Species Parsing And Bond Estimation

`_estimate_species_bond_counts(species_name)`

Estimates all information for one species. It first checks `EXACT_SPECIES`. If
the species is not in the lookup table, it infers the formula from the name,
estimates carbon skeleton bonds, adds heteroatom heuristics, adds hydrogen-bond
heuristics, and returns the final species dictionary.

`_infer_formula_counts(species_name)`

Extracts approximate element counts from the ChemKin species name. It handles
common prefixes such as `n`, `i`, `p`, `s`, `t`, `nc`, `ic`, and `neo`, and it
also calls `_infer_ooh_formula()` for hydroperoxide-style names.

`_strip_non_formula_prefix(lower_name)`

Removes structural prefixes that are not part of the elemental formula. For
example, `nc7h16` is interpreted as `c7h16`, because the leading `n` means
normal-chain, not nitrogen.

`_infer_ooh_formula(lower_name)`

Handles species names containing `ooh`, such as hydroperoxides. For example, a
name like `c8h17ooh` is interpreted as containing one extra H and two O atoms
from the hydroperoxide group.

`_looks_like_ring(species_name)`

Detects common naming patterns for cyclic species. Examples include names that
start with `c-`, contain `cC`, or match cyclic ether-style labels such as
`c4h8o1-2`.

`_carbon_count_from_name(species_name)`

Extracts the number after `c` in a species name. For example, `c12h26` gives
`12`. This is used as a backup when the full formula parser is incomplete.

`_estimate_carbon_skeleton_bonds(species_name, formula_counts, bond_counts, is_ring, notes)`

Estimates the carbon-carbon backbone. For saturated non-ring hydrocarbons, it
uses approximately `C-C = carbon_count - 1`. For alkenes, alkynes, and rings, it
adds `C=C`, `C#C`, or cyclic carbon skeleton estimates.

`_estimate_ring_carbon_bonds(carbon_count, hydrogen_count, bond_counts)`

Adds approximate C-C and C=C counts for cyclic species. Benzene-like species are
treated specially with three `C-C` and three `C=C` bonds.

`_looks_like_alkyne(lower_name, carbon_count, hydrogen_count)`

Checks whether the name or formula suggests a carbon-carbon triple bond. This is
mainly used for small or clearly alkyne-like labels such as `C2H2`.

`_looks_like_alkene(lower_name, carbon_count, hydrogen_count)`

Checks whether the name or formula suggests a carbon-carbon double bond. This is
used for labels such as `C4H8-1` or formulas with alkene-like hydrogen counts.

`_add_heteroatom_bond_heuristics(species_name, formula_counts, bond_counts, is_ring, notes)`

Adds approximate heteroatom bonds based on name patterns. It handles cyclic
ethers, ketohydroperoxides, hydroperoxides, peroxy radicals, carbonyl-like names,
and alcohol or alkoxy-like names.

`_add_hydrogen_bond_heuristics(species_name, formula_counts, bond_counts)`

Assigns hydrogen atoms to approximate `H-H`, `C-H`, `N-H`, or `O-H` bonds. It
puts hydrogen on oxygen for oxygen-only species, on nitrogen for nitrogen-only
species, and on carbon for remaining hydrogens in carbon-containing species.

`_contains_carbonyl_name(lower_name)`

Checks whether a species name contains carbonyl or aldehyde markers such as
`cho` or `co`. It excludes standalone `co` and `co2`, because those are handled
by `EXACT_SPECIES`.

`_contains_alcohol_or_alkoxy_name(lower_name)`

Checks whether a species name looks like an alcohol or alkoxy species, usually
through an `oh` ending or an `o-` position marker. It avoids double-counting
peroxide and hydroperoxide names.

### Top-Level Helper Functions

`_nonzero_bond_counts(bond_counts)`

Removes zero-count bonds from a bond dictionary. This is only used for cleaner
printing in summaries; the stored output still keeps the full zero-filled
`bond_counts` dictionary.

`print_formula_ambiguity_summary(backbone, max_formulas=8, max_species=5)`

Prints a readable summary of ambiguous formulas and possible multi-species
labels. This is useful as a quick sanity check after parsing a new mechanism
file.

## Output Dictionary Structure

`read_species_backbone()` returns:

```python
{
    "species": {
        "H2": {
            "formula": "H2",
            "bond_counts": {"H-H": 1, ...},
            "is_ring": False,
            "confidence": "exact_name_rule",
            "notes": ["matched exact small-species rule"],
            "formula_variant_id": "H2_001",
            "may_represent_multiple_species": False,
            "related_formula_variant_ids": [],
            "multi_species_reason": None,
        },
        ...
    },
    "formula_variants": {
        "C3H4_001": {
            "formula": "C3H4",
            "variant_id": "C3H4_001",
            "bond_counts": {...},
            "is_ring": False,
            "species": ["aC3H4"],
        },
        ...
    },
    "ambiguous_formulas": {
        "C3H4": ["C3H4_001", "C3H4_002", "C3H4_003"],
        ...
    },
    "multi_species_labels": {
        "C6H12": {
            "formula": "C6H12",
            "formula_variant_id": "C6H12_001",
            "related_formula_variant_ids": ["C6H12_001", "C6H12_002"],
            "reason": "...",
        },
        ...
    },
    "sanity_checks": {
        "raw_species_count": 348,
        "variant_species_reference_count": 348,
        "variant_species_reference_matches_species_count": True,
    },
}
```

## Exporting To JSON

Use `export_species_backbone_to_json()` when you want to save the parsed species
features for later analysis or for a GNN data-processing pipeline.

```python
from identify_species_backbone import identify_backbone_from_chemkin_mechanism

reader = identify_backbone_from_chemkin_mechanism(
    "mechanism_file_chemkin/c8-c16_n-alkanes_LLNL.txt"
)

output_path = reader.export_species_backbone_to_json(
    "output/c8_c16_species_backbone.json"
)

print(output_path)
```

The JSON file contains the same top-level keys returned by
`read_species_backbone()`:

```text
species
formula_variants
ambiguous_formulas
multi_species_labels
sanity_checks
```
