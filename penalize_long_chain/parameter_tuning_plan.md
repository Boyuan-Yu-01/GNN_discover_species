# Plan: Tune Bond-Breaking Parameters

## Goal

Find one physically reasonable parameter triple:

```text
A
E_ref
k
```

for the existing bond-breaking score. This is a three-parameter numerical
search, not machine learning and not a neural-network training procedure.

Use observed formula-degeneracy groups as positive evidence and fabricated
pseudo-negative groups as weak contrast evidence. The result is a relative
survival score, not a calibrated physical probability.

## Data

Use the filtered datasets already in the project:

```text
training/org_dataset/filtered_master.json
training/org_dataset/filtered_pseudo_negative.json
```

They contain:

| Source | Formula-degeneracy groups |
| --- | ---: |
| Observed positives | 267 |
| Fabricated pseudo-negatives | 569 |

The pseudo-negative groups are not confirmed chemical negatives. They should
have lower influence than the observed groups in the objective.

## Bond-Breaking Model

Use the backbone bond types:

```python
BACKBONE_BONDS = ["C#C", "C-C", "C-O", "C=C", "C=O"]

BDE_BY_BOND = {
    "C-C": 346.0,
    "C-O": 358.0,
    "C=C": 602.0,
    "C=O": 732.0,
    "C#C": 835.0,
}
```

For each backbone bond (b):

$$
\mathrm{Arr}_b
=
A\exp\left(-\frac{E_b}{E_{\mathrm{ref}}}\right),
$$

$$
P_{\mathrm{break},b}
=
\frac{1}{1+k\exp(-\mathrm{Arr}_b)}.
$$

For formula-degeneracy group (g), calculate the log survival score:

$$
s_\theta(g)
=
\log P_{\mathrm{exist}}(g)
=
\sum_{b\in\mathrm{backbone}}
N_{g,b}\log(1-P_{\mathrm{break},b}),
$$

where:

$$
\theta=\{A,E_{\mathrm{ref}},k\}.
$$

Scores satisfy (s_\theta(g)\leq0). A score closer to zero indicates higher
predicted survival.

Calculate in log space and clamp each `P_break` away from 0 and 1 before taking
the logarithm.

## Objective

For each tested parameter triple, score every group independently and calculate separate
distribution statistics:

$$
\bar{s}_+
=
\frac{1}{|P|}\sum_{g\in P}s_\theta(g),
$$

$$
\bar{s}_-
=
\frac{1}{|N|}\sum_{g\in N}s_\theta(g),
$$

where (P) is the observed-positive set and (N) is the pseudo-negative set.

Reward high scores for observed groups:

$$
L_{\mathrm{positive}}=-\bar{s}_+.
$$

Require the positive distribution to have a higher mean score than the
pseudo-negative distribution by a fixed margin (m):

$$
L_{\mathrm{separation}}
=
\max\left(0,\,m-[\bar{s}_+-\bar{s}_-]\right).
$$

Use the final objective:

$$
\boxed{
L(\theta)
=
\alpha L_{\mathrm{positive}}
+
\beta L_{\mathrm{separation}}
+
L_{\mathrm{reg}}(\theta)
}.
$$

Recommended starting values:

```text
alpha = 1.00      # observed positives are the main evidence
beta  = 0.75      # lower than alpha, but large enough to enforce separation
m     = 0.10      # fixed score margin; test sensitivity later
```

The objective compares whole score distributions. It contains no positive-
negative pairing and no pair list.

The loss components have different numerical scales, so a very small `beta`
can make the all-high-survival solution preferable. Test `beta` values `0.50`,
`0.75`, and `1.00`; use the smallest value that satisfies the fixed margin. On
the current filtered datasets, `0.75` is the first of these values to reach the
`0.10` mean-score gap while remaining below `alpha`.

### Optional high-score pseudo-negative check

The mean pseudo-negative score can hide a few highly scored pseudo-negatives.
After the first run, optionally replace $\bar{s}_-$ in the separation term with
the 90th percentile of pseudo-negative scores:

$$
Q_{0.9}(s_-).
$$

This is valid because the search uses random search or differential evolution,
not gradient descent. Keep the mean-based objective as the first version.

## Parameter Regularization

Search in log space to guarantee positive parameters:

```python
log_A
log_E_ref
log_k
```

Use a weak prior around physically reasonable reference values:

```python
A_0 = 10.0
E_ref_0 = 300.0
k_0 = 1000.0
```

$$
L_{\mathrm{reg}}
=
\lambda_A(\log A-\log A_0)^2
+
\lambda_E(\log E_{\mathrm{ref}}-\log E_{\mathrm{ref},0})^2
+
\lambda_k(\log k-\log k_0)^2.
$$

Start with:

```text
lambda_A = lambda_E = lambda_k = 1e-3
```

The regularizer prevents extreme parameter values from winning merely by
driving every score toward a boundary.

## Parameter Search

Use these bounds:

```python
A      in [1.0, 100.0]
E_ref  in [50.0, 1000.0]   # same energy unit as BDE
k      in [10.0, 1_000_000.0]
```

Recommended procedure:

1. Draw 10,000 to 20,000 log-uniform parameter triples with a fixed seed.
2. Calculate the non-paired objective for each triple.
3. Retain the best 20 triples.
4. Refine the best candidates with `scipy.optimize.differential_evolution` or
   bounded local optimization.
5. Select the best stable candidate that passes the checks below.

The objective does not require a validation split. To test stability, repeat
the search with several pseudo-negative subsampling seeds and compare the
selected parameter ranges and score distributions.

## Required Functions

Suggested script:

```text
tune_bond_breaking_parameters.py
```

Suggested functions:

```python
def load_formula_degeneracies(path):
    """Return the formula_degeneracies mapping from one filtered JSON file."""


def bond_break_probability(energy, A, E_ref, k):
    """Return P_break for one backbone bond type."""


def log_survival(entry, params):
    """Return s_theta(g) from the five configured backbone bond counts."""


def distribution_objective(positive_entries, pseudo_entries, params):
    """Return total, positive, separation, and regularization losses."""


def random_search(positive_entries, pseudo_entries, seed, n_trials):
    """Search log-space parameters and return the best candidates."""


def save_outputs(best_params, diagnostics, scored_entries):
    """Write parameters, scores, diagnostics, and a readable log."""
```

## Checks and Diagnostics

For the final parameters, verify:

1. Every bond-breaking probability is finite and lies strictly between 0 and 1.
2. Higher BDE gives lower predicted `P_break` for the configured bond table.
3. Observed groups have a higher mean log survival score than pseudo-negatives.
4. The separation margin and each loss component are reported separately.
5. For otherwise similar bond inventories, longer backbones generally have
   lower survival scores.
6. The best parameters are not at a search boundary.
7. Results remain similar across pseudo-negative subsampling seeds.
8. Inspect the top 20 and bottom 20 observed groups by score.

Report pseudo-negative separation as a diagnostic, not as confirmed-negative
accuracy, precision, or calibrated probability.

## Output Files

Write:

```text
output/tuned_bond_breaking_parameters.json
output/scored_formula_degeneracies.csv
output/tuning_diagnostics.csv
output/log_tuned_bond_breaking_parameters.txt
```

The parameter JSON should include:

```json
{
  "A": 0.0,
  "E_ref": 0.0,
  "k": 0.0,
  "objective_value": 0.0,
  "positive_loss": 0.0,
  "separation_loss": 0.0,
  "regularization_loss": 0.0,
  "alpha": 1.0,
  "beta": 0.75,
  "margin": 0.1,
  "bde_by_bond": {},
  "search_seed": 0
}
```

The scored CSV should include:

```text
source,degeneracy_id,formula,is_ring,log_P_exist,P_exist,C#C,C-C,C-O,C=C,C=O
```

For filtered records, derive `formula` by removing the final `_###` suffix from
`degeneracy_id`.

## Interpretation

The output parameters are the values that best satisfy the chosen weak,
distribution-level constraints. They are not uniquely identified physical
constants and do not provide calibrated existence probabilities.

The useful outcome is a chemically interpretable survival ranking in which
observed groups receive high scores and fabricated long-chain or fragile groups
generally receive lower scores.
