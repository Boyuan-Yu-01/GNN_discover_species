# Plan: Tune A, E_ref, and k

## Goal

Find one physically reasonable parameter triple:

```text
A
E_ref
k
```

Use one broad numerical search. The earlier `E_ref = R*T` values below
25 kJ/mol are no longer used because they make the Arrhenius response nearly
zero for the configured bond dissociation energies.

This is parameter fitting, not machine learning. Observed degeneracies are
reliable positive evidence. Fabricated pseudo-negatives provide weak contrast
evidence and are not treated as confirmed negatives.

## Data

Use:

```text
training/org_dataset/filtered_master.json
training/org_dataset/filtered_pseudo_negative.json
```

| Source | Total groups | Used in objective |
| --- | ---: | ---: |
| Observed positives | 267 | 258 |
| Fabricated pseudo-negatives | 569 | 564 |

The parameters cannot affect groups whose configured backbone-bond counts are
all zero. Exclude the 9 positive and 5 pseudo-negative zero-backbone groups
from the objective, but retain them in scored output and report the counts.

## Bond-Breaking Model

Use:

```python
BACKBONE_BONDS = ("C#C", "C-C", "C-O", "C=C", "C=O")

BDE_BY_BOND = {
    "C#C": 835.0,
    "C-C": 346.0,
    "C-O": 358.0,
    "C=C": 602.0,
    "C=O": 732.0,
}
```

All energies use `kJ/mol`. For bond type $b$:

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

For degeneracy $g$:

$$
s_\theta(g)
=
\log P_{\mathrm{exist}}(g)
=
\sum_b N_{g,b}\log\left(1-P_{\mathrm{break},b}\right),
$$

$$
P_{\mathrm{exist}}(g;\theta)=\exp(s_\theta(g)),
$$

where $\theta=(A,E_{\mathrm{ref}},k)$. Calculate survival in log space and
clamp breaking probabilities away from exactly zero and one.

## Individual-Sample Objective

Every usable degeneracy contributes an independent nonlinear loss. There is no
positive-negative pairing, ranking margin, or mean feature vector.

For each observed positive:

$$
\ell_+(g;\theta)
=
\left[1-P_{\mathrm{exist}}(g;\theta)\right]^2.
$$

For each fabricated pseudo-negative:

$$
\ell_-(g;\theta)
=
P_{\mathrm{exist}}(g;\theta)^2.
$$

Aggregate only after calculating every individual loss:

$$
L_{\mathrm{positive}}
=
\frac{1}{|P^*|}\sum_{g\in P^*}\ell_+(g;\theta),
$$

$$
L_{\mathrm{pseudo}}
=
\frac{1}{|N^*|}\sum_{g\in N^*}\ell_-(g;\theta),
$$

$$
L(\theta)
=
\alpha L_{\mathrm{positive}}
+\beta L_{\mathrm{pseudo}}
+L_{\mathrm{reg}}(\theta).
$$

Use:

```text
alpha = 1.00
beta  = 0.50
```

The pseudo-negative term is weaker because the samples are fabricated. The
averages normalize unequal source sizes only; losses are calculated per record
before averaging.

## Search Range and Initial Guess

Search the physical parameters in natural-log space:

```text
A      in [1, 100]
E_ref  in [50, 1000] kJ/mol
k      in [10, 1_000_000]
```

Use this exact initial/reference candidate:

```text
A_0     = 10
E_ref_0 = 300 kJ/mol
k_0     = 1000
```

The 300 kJ/mol `E_ref` value is close to the approximately 310 kJ/mol region
identified during the earlier exploratory pointwise calculation. It is a
starting guess, not a fixed value.

The optimizer coordinates are:

```text
log(A)
log(E_ref)
log(k)
```

This guarantees positive physical parameters and allows orders-of-magnitude
coverage without the low-temperature transformation previously required.

## Regularization

Keep the accepted rule that `E_ref` is not pulled toward its initial value:

$$
L_{\mathrm{reg}}
=
\lambda_A(\log A-\log A_0)^2
+\lambda_k(\log k-\log k_0)^2.
$$

Use:

```text
lambda_A     = 1e-3
lambda_E_ref = 0
lambda_k     = 1e-3
```

Therefore, `E_ref_0=300` initializes and documents the search but does not act
as a prior. `E_ref` may move freely anywhere in `[50, 1000]`.

## Search Procedure

1. Include the exact initial candidate `(10, 300, 1000)`.
2. Draw 19,999 additional log-uniform candidates over the full bounds with a
   fixed random seed.
3. Calculate the pointwise objective for all 258 usable positives and 564
   usable pseudo-negatives.
4. Sort candidates by total loss.
5. Refine the best candidates with bounded Powell optimization.
6. Keep the better of each local result and its starting candidate.
7. Select the configured number of distinct lowest-loss triples that pass the
   numerical checks.

The full-range random stage reduces dependence on the initial guess, while the
exact reference candidate makes its objective directly inspectable.

## Checks

For the final result:

1. Verify all parameters and intermediate values are finite.
2. Verify every `P_break` lies strictly between zero and one.
3. Verify higher BDE produces no larger `P_break`.
4. Report `max(P_break) - min(P_break)` and flag a spread below `1e-6` as
   weakly identifiable.
5. Report whether a fitted parameter lies near a search boundary.
6. Report positive, pseudo-negative, and regularization losses separately.
7. Report mean, median, and selected quantiles of individual survival
   probabilities for both sources.
8. Report the 9 positive and 5 pseudo-negative zero-backbone exclusions.

Mean scores are descriptive diagnostics only. They are not the fitting target.
Pseudo-negative separation must not be described as confirmed-negative
accuracy or a calibrated physical probability.

## Implementation

Use:

```text
training/find_three_parameters/parameter_finder.py
training/find_three_parameters/parameter_utils.py
training/find_three_parameters/parameter_animation.py
training/find_three_parameters/main.py
```

`ParameterFinder` evaluates the three logged physical parameters and can refine
them in either log or physical coordinates. `main.py` exposes the bounds,
initial guess, objective weights, numerical controls, and output-set count.
Utilities load the datasets and write reproducible reports. The
animation writer displays the bounded 3D log-parameter space, all 20,000 random
candidates, the configured retained Powell starts, their completed Powell
iteration paths, and the final selected result. The camera remains fixed so
apparent motion comes only from parameter changes.

## Outputs

When the search is approved and run, write:

```text
output/parameter_set_01/tuned_bond_breaking_parameters.json
output/parameter_set_01/scored_formula_degeneracies.csv
output/parameter_set_01/log_tuned_bond_breaking_parameters.txt
output/parameter_set_01/positive_pseudo_negative_scores.png
output/parameter_set_02/...
output/parameter_search_animation.mp4
```

`NUMBER_OF_PARAMETER_SETS` controls how many ranked, distinct, identifiable
parameter triples are written. Each folder contains its own JSON, CSV, text
summary, and a plot of observed positives (blue) and pseudo-negatives (grey)
on the same existence-probability axis.

## Interpretation

The selected triple is the best result under this model, pointwise objective,
weak pseudo-negative weight, bounds, and regularization. It is not a uniquely
identified set of physical constants or a calibrated existence probability.
