# Plan for Tuning the Bond-Breaking Probability Parameters

This document describes how to write a script that tunes the three parameters in the bond-breaking probability mapping using:

```text
output/master_dataset.json
```

The dataset contains `formula_degeneracies`. Each entry is a group of species with the same formula, ring flag, and bond-count signature.

## Goal

Use the backbone bond counts

```python
BACKBONE_BONDS = ["C#C", "C-C", "C-O", "C=C", "C=O"]
```

to estimate a species-group survival probability:

$$
P_{\mathrm{exist}}(g)
$$

where `g` is one formula-degeneracy group.

The current proposed mapping is:

$$
\mathrm{Arr}_b = A \exp \left(-\frac{E_b}{E_{\mathrm{ref}}}\right)
$$

$$
P_{\mathrm{break}, b}
=
\frac{1}{1 + k \exp(-\mathrm{Arr}_b)}
$$

where:

- `b` is a bond type, for example `C-C` or `C=O`;
- `E_b` is the bond dissociation energy of that bond type;
- `A` is an Arrhenius-like scale factor;
- `E_ref` controls how quickly the Arrhenius-like term decays with BDE;
- `k` controls the probability floor and sigmoid shift.

For a formula-degeneracy group `g`, estimate survival as:

$$
P_{\mathrm{exist}}(g)
=
\prod_{b \in \mathrm{backbone}}
\left(1 - P_{\mathrm{break}, b}\right)^{N_{g,b}}
$$

where `N_{g,b}` is the number of bonds of type `b` in group `g`.

For numerical stability, the script should compute this in log space:

$$
\log P_{\mathrm{exist}}(g)
=
\sum_{b \in \mathrm{backbone}}
N_{g,b}
\log \left(1 - P_{\mathrm{break}, b}\right)
$$

## Important Limitation

This dataset does not contain direct probability labels. It also does not contain true negative examples.

Therefore, the script cannot strictly learn calibrated physical probabilities from the dataset alone.

Instead, the script should tune the parameters as a weakly supervised scoring model. The output should be interpreted as a relative survival or existence score unless additional probability labels are added later.

## Current Dataset Shape

From the current `master_dataset.json`:

```text
formula-degeneracy groups: 273
total species labels:      2115
groups with backbone:      263
ring groups:               15
```

Backbone bond summary:

```text
  C-C: max 15, nonzero groups 237
  C=C: max 2, nonzero groups 58
  C#C: max 1, nonzero groups 3
  C-O: max 2, nonzero groups 142
  C=O: max 2, nonzero groups 75
  C#O: max 1, nonzero groups 1
  O-O: max 2, nonzero groups 94
  O=O: max 1, nonzero groups 1
```

Because `C-C` dominates the dataset, the tuning objective should avoid letting only `C-C` determine all parameters.

## Recommended Strategy

Use three weak training signals together:

1. Positive-only survival pressure.
2. Pseudo-negative ranking.
3. Degeneracy-support consistency.

### 1. Positive-Only Survival Pressure

Every group in `formula_degeneracies` is an observed group, so treat it as a positive example.

However, do not force all positives to have probability exactly 1. Instead, require observed groups to have reasonably high survival compared with generated pseudo-negatives.

Use the species-list length as a confidence weight:

```python
support_g = len(entry["species"])
weight_g = log1p(support_g)
```

This should not be interpreted as a probability. It is only a weak confidence measure.

### 2. Pseudo-Negative Ranking

Since there are no true negatives, generate pseudo-negative bond-count vectors.

Possible negative-generation methods:

#### Method A: Random Count Vectors

Sample random backbone count vectors within slightly expanded observed ranges.

For example:

```python
C-C: sample from 0 to max_observed_C_C + 3
C-O: sample from 0 to max_observed_C_O + 2
C=C: sample from 0 to max_observed_C_eq_C + 2
C=O: sample from 0 to max_observed_C_eq_O + 2
C#C: sample from 0 to max_observed_C_triple_C + 1
```

Reject samples that exactly match an observed backbone signature.

#### Method B: Corrupted Observed Positives

Start from an observed group and perturb it:

- add one or more weak backbone bonds;
- increase the chain length;
- replace one stronger bond with a weaker bond;
- generate a bond-count vector not present in the observed dataset.

These are not guaranteed real negatives. They are weak negatives and should be given lower weight.

#### Method C: Hard Negatives

Create pseudo-negatives that look close to observed groups but have extra fragile structure. These help the model learn the boundary between plausible and unlikely long-chain structures.

Example:

```text
observed:       C-C = 9, C=O = 1
pseudo-negative C-C = 14, C=O = 2, C-O = 2
```

### 3. Degeneracy-Support Consistency

Groups with larger `len(species)` are not necessarily more probable, but they do represent more mechanism support.

Use this only as a soft ranking signal:

```text
groups with larger support should not systematically receive lower survival scores
```

A simple diagnostic is Spearman correlation between:

```python
log_survival_score
```

and:

```python
log1p(len(species))
```

Do not make this the only objective, because mechanism naming degeneracy is not the same as physical probability.

## Proposed Objective Function

For a first implementation, use a pairwise ranking objective:

$$
L_{\mathrm{rank}}
=
\frac{1}{M}
\sum_{(g^+, g^-)}
\max
\left(
0,
m - [s(g^+) - s(g^-)]
\right)
$$

where:

$$
s(g) = \log P_{\mathrm{exist}}(g)
$$

`g+` is an observed formula-degeneracy group, `g-` is a pseudo-negative, and `m` is a fixed margin.

Then add a weak support consistency term:

$$
L
=
L_{\mathrm{rank}}
-
\lambda_{\mathrm{support}}
\rho_{\mathrm{Spearman}}
\lambda_{\mathrm{reg}} L_{\mathrm{reg}}
$$

where:

- `rho_Spearman` is the Spearman correlation between survival score and support;
- `lambda_support` should be small;
- `L_reg` prevents extreme parameter values.

A simpler first version can skip the Spearman term and use only ranking:

$$
L = L_{\mathrm{rank}} + \lambda_{\mathrm{reg}} L_{\mathrm{reg}}
$$

## Parameter Search

Tune the three parameters:

```text
A
E_ref
k
```

Use log-space parameters so all values remain positive:

```python
log_A
log_E_ref
log_k
```

Suggested search ranges:

```python
A      in [1, 100]
E_ref  in [50, 1000]     # same energy unit as BDE, likely kJ/mol
k      in [1e1, 1e6]
```

Use either:

1. random search over log-space;
2. grid search for the first version;
3. `scipy.optimize.differential_evolution` if SciPy is available.

Recommended first version:

```text
random search with 5,000 to 20,000 parameter samples
```

This is simple, reproducible, and enough for only three parameters.

## Bond Dissociation Energy Table

Put the BDE values in one dictionary near the top of the script:

```python
BDE_BY_BOND = {
    "C-C": 346.0,
    "C-O": 358.0,
    "C=C": 602.0,
    "C=O": 732.0,
    "C#C": 835.0,
}
```

These values should be treated as configurable approximate values, not universal constants. Real BDE values depend on molecular context.

## Script Structure

Suggested script name:

```text
tune_bond_breaking_parameters.py
```

Suggested location:

```text
penalize_long_chain/tune_bond_breaking_parameters.py
```

### Main Functions

```python
def load_master_dataset(json_path):
    """Load formula_degeneracies from master_dataset.json."""
```

```python
def build_positive_table(formula_degeneracies):
    """Return one row per observed formula-degeneracy group."""
```

Each row should contain:

```text
degeneracy_id
formula
is_ring
species_count
C#C
C-C
C-O
C=C
C=O
```

```python
def make_pseudo_negatives(positive_table, n_negatives, seed):
    """Generate weak negative backbone count vectors."""
```

```python
def bond_break_probability(E_b, A, E_ref, k):
    arr = A * np.exp(-E_b / E_ref)
    return 1.0 / (1.0 + k * np.exp(-arr))
```

```python
def log_survival(row, params, bde_by_bond):
    """Compute log P_exist for one group."""
```

```python
def ranking_loss(positive_rows, negative_rows, params):
    """Compute pairwise ranking loss."""
```

```python
def random_search(objective, parameter_ranges, n_trials, seed):
    """Return best parameters and diagnostics."""
```

```python
def score_all_groups(positive_table, best_params):
    """Add P_exist and log_P_exist to every observed group."""
```

```python
def save_outputs(best_params, scored_table, diagnostics):
    """Write JSON, CSV, and plots."""
```

## Output Files

The tuning script should write:

```text
output/tuned_bond_breaking_parameters.json
output/scored_formula_degeneracies.csv
output/tuning_diagnostics.csv
output/tuning_plots/
```

The parameter JSON should contain:

```json
{
  "A": 10.0,
  "E_ref": 300.0,
  "k": 1000.0,
  "objective_value": 0.123,
  "bde_by_bond": {
    "C-C": 346.0,
    "C-O": 358.0,
    "C=C": 602.0,
    "C=O": 732.0,
    "C#C": 835.0
  }
}
```

The scored CSV should contain:

```text
degeneracy_id, formula, is_ring, species_count, log_P_exist, P_exist, C#C, C-C, C-O, C=C, C=O
```

## Validation Checks

After tuning, inspect the following:

1. Bond-level probabilities:

```text
P_break(C-C)
P_break(C-O)
P_break(C=C)
P_break(C=O)
P_break(C#C)
```

The expected trend is:

```text
higher BDE -> lower bond-breaking probability
```

2. Group-level survival:

Check the top 20 and bottom 20 groups by `P_exist`.

3. Chain-length behavior:

For otherwise similar groups, longer backbones should generally have lower survival.

4. Pseudo-negative separation:

Observed groups should rank above pseudo-negative samples on average.

5. Sensitivity to pseudo-negative generation:

Run the script with several random seeds. The chosen parameters should not change drastically.

## Recommended First Implementation

Start with this minimal workflow:

1. Load observed groups from `master_dataset.json`.
2. Keep only groups with at least one of the five backbone bonds.
3. Generate pseudo-negative count vectors by random count sampling.
4. Tune `A`, `E_ref`, and `k` by random search using pairwise ranking loss.
5. Save the best parameters.
6. Score all observed groups.
7. Plot:
   - `P_break` versus BDE for each bond type;
   - distribution of `P_exist` for observed groups;
   - observed versus pseudo-negative score distributions;
   - top and bottom formula-degeneracy groups.

## Main Caveat

Because the dataset has no measured probabilities and no true negative examples, the tuned parameters are not uniquely identifiable as physical constants.

The first useful target is therefore not a perfectly calibrated probability model. The first useful target is a chemically reasonable ranking model:

```text
stable backbone groups should receive high survival scores,
fragile or over-extended backbone groups should receive low survival scores.
```

Later, if measured species probabilities, mole fractions, simulation frequencies, or reliable absent-species candidates become available, the same script can be upgraded from weak ranking to supervised probability calibration.

