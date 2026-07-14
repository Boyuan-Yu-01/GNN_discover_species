# Finding A, E_ref, and k

This directory numerically fits the bond-breaking parameters:

```text
A
E_ref
k
```

The search uses individual observed and fabricated degeneracy records. It is
not machine learning and does not pair positive and pseudo-negative samples.

> The restored broad-range search has not been run yet. Do not interpret older
> files under `output/` as results from the implementation described here.

## Files

```text
parameter_finder.py  Physical model, pointwise objective, and optimizer
parameter_utils.py   Dataset loading and output writers
parameter_animation.py  Staged 3D MP4 renderer for the complete search
main.py              Tunable settings and search orchestration
output/              Created when main.py is approved and run
```

The complete design is in
[`../../parameter_tuning_plan.md`](../../parameter_tuning_plan.md).

## Data

`main.py` loads:

```text
../org_dataset/filtered_master.json
../org_dataset/filtered_pseudo_negative.json
```

| Source | Total | Used in objective |
| --- | ---: | ---: |
| Observed positives | 267 | 258 |
| Fabricated pseudo-negatives | 569 | 564 |

Nine positive and five pseudo-negative records contain none of the configured
backbone bonds. Their scores cannot change with the parameters, so they are
excluded from the objective but retained in scored output.

## Why E_ref Returned to a Larger Range

The rejected `E_ref=R*T` values ranged from 0.8314 to 24.942 kJ/mol. Compared
with BDE values of 346–835 kJ/mol, these values make:

$$
\exp\left(-\frac{E_b}{E_{\mathrm{ref}}}\right)
$$

extremely small. The model then loses sensitivity to bond type.

The restored search uses:

```text
E_ref in [50, 1000] kJ/mol
initial E_ref = 300 kJ/mol
```

The 300 kJ/mol initial value is near the approximately 310 kJ/mol region found
in the earlier exploratory pointwise calculation.

## Bond Model

For bond type `b` with BDE `E_b`:

$$
\mathrm{Arr}_b=A\exp\left(-\frac{E_b}{E_{\mathrm{ref}}}\right),
$$

$$
P_{\mathrm{break},b}
=
\frac{1}{1+k\exp(-\mathrm{Arr}_b)}.
$$

For degeneracy `g`:

$$
\log P_{\mathrm{exist}}(g)
=
\sum_b N_{g,b}\log(1-P_{\mathrm{break},b}).
$$

The implementation evaluates Arrhenius and survival terms in log space.

## Individual-Sample Objective

Every usable degeneracy is scored separately.

For an observed positive:

$$
\ell_+(g)=\left[1-P_{\mathrm{exist}}(g)\right]^2.
$$

For a fabricated pseudo-negative:

$$
\ell_-(g)=P_{\mathrm{exist}}(g)^2.
$$

The total objective is:

$$
L
=
\alpha\frac{1}{|P^*|}\sum_{g\in P^*}\ell_+(g)
+\beta\frac{1}{|N^*|}\sum_{g\in N^*}\ell_-(g)
+L_{\mathrm{reg}}.
$$

Defaults:

```text
alpha = 1.00
beta  = 0.50
```

Pseudo-negatives receive less influence because they are fabricated rather
than confirmed negatives. The averages normalize unequal dataset sizes; the
nonlinear loss is calculated before averaging.

## Search Bounds and Initial Candidate

The optimizer searches:

```text
A      in [1, 100]
E_ref  in [50, 1000] kJ/mol
k      in [10, 1_000_000]
```

The exact initial/reference candidate is:

```text
A      = 10
E_ref  = 300 kJ/mol
k      = 1000
```

The optimizer coordinates are:

```text
log(A)
log(E_ref)
log(k)
```

This keeps all physical parameters positive. The previous low-temperature `u`
coordinate is no longer necessary.

## Regularization

The implementation uses:

$$
L_{\mathrm{reg}}
=
10^{-3}(\log A-\log 10)^2
+10^{-3}(\log k-\log 1000)^2.
$$

The `E_ref` regularization weight remains zero. Therefore, 300 kJ/mol is an
initial candidate, not a prior that pulls the final result toward 300.

In code:

```python
REGULARIZATION_WEIGHTS = (1e-3, 0.0, 1e-3)
```

The order is `log(A)`, `log(E_ref)`, `log(k)`.

## Parameter-Finding Algorithm

1. Add the exact candidate `(10, 300, 1000)`.
2. Generate 19,999 reproducible log-uniform candidates across the full bounds.
3. Calculate the independent pointwise losses for every candidate.
4. Sort candidates by total loss.
5. Send the best candidates to bounded Powell optimization.
6. Keep a local result only if it improves its starting candidate.
7. Select the overall lowest loss.
8. Reject the result if all bond-breaking probabilities are numerically
   indistinguishable.

The full-range random stage reduces dependence on the initial guess.

## ParameterFinder Walkthrough

`ParameterFinder.__init__`
: Validates matrices, energies, bounds, and objective settings. It excludes
  zero-backbone rows from fitting only.

`encode` / `decode`
: Convert between physical parameters and their natural logarithms.

`arrhenius_terms`
: Calculates `A * exp(-E_b/E_ref)` safely in log space.

`bond_break_probabilities`
: Calculates one breaking probability per configured bond.

`score_counts`
: Produces one log-existence score per individual degeneracy.

`evaluate`
: Calculates individual positive and pseudo-negative losses before averaging.

`random_search`
: Evaluates the exact reference plus full-range log-uniform candidates.

`refine`
: Runs bounded, derivative-free Powell optimization from the best candidates.

`probability_spread` / `is_at_boundary`
: Diagnose loss of bond sensitivity and boundary solutions.

## main.py Walkthrough

The settings block near the top is the intended control surface.

`make_finder`
: Constructs the shared physical model and pointwise objective.

`run_search`
: Performs the global candidate stage and local refinement stage.

`search_settings`
: Records bounds, references, weights, seeds, and tolerances in output JSON.

`main`
: Loads data, runs the search, verifies identifiability, writes numerical
  output, and renders the optional animation.

## Important Tunable Settings

```python
PARAMETER_BOUNDS = ParameterBounds(
    A=(1.0, 100.0),
    E_ref=(50.0, 1000.0),
    k=(10.0, 1_000_000.0),
)

REFERENCE_PARAMETERS = Parameters(A=10.0, E_ref=300.0, k=1000.0)
REGULARIZATION_WEIGHTS = (1e-3, 0.0, 1e-3)

POSITIVE_WEIGHT = 1.00
PSEUDO_NEGATIVE_WEIGHT = 0.50

RANDOM_TRIALS = 20_000
KEEP_BEST_RANDOM = 20
LOCAL_STARTS = 20
LOCAL_MAX_ITERATIONS = 1000

GENERATE_SEARCH_ANIMATION = True
ANIMATION_FILENAME = "parameter_search_animation.mp4"
```

## Search Animation

When enabled, `main.py` records every objective evaluation requested by Powell
and renders a staged 3D animation in logarithmic parameter coordinates.
Physical tick labels and axis limits still correspond exactly to
`PARAMETER_BOUNDS`. The camera remains fixed throughout the MP4.

The phases are:

1. Display the empty bounded `(A, E_ref, k)` space.
2. Fade in all 20,000 random-stage candidates.
3. Fade out discarded candidates, leaving the configured Powell starts.
4. Draw all Powell objective probes simultaneously, one evaluation per frame.
5. Mark successful and unsuccessful SciPy terminations, then hold the selected
   result.

`evaluations_per_frame=1` displays every evaluated point. These points include
Powell's internal line-search probes; they are not all accepted outer-iteration
updates. Increase this value only when a shorter video is more important than
showing every probe.

Different final points do not by themselves mean that Powell failed. A run is
marked successful only from SciPy's termination status; several successful runs
may end in different local minima or different points on a flat valley.

## Outputs Created When Run

```text
output/tuned_bond_breaking_parameters.json
output/scored_formula_degeneracies.csv
output/log_tuned_bond_breaking_parameters.txt
output/parameter_search_animation.mp4
```

- The JSON contains the selected triple, objective components, distribution
  quantiles, physical diagnostics, and reproducibility settings.
- The scored CSV preserves every sample and its individual loss.
- The log is a compact human-readable summary.
- The MP4 visualizes global coverage, candidate selection, and local paths.

## Running Later

After reviewing the restored settings:

```bash
python main.py
```

Required packages:

```text
numpy
scipy
matplotlib
```

MP4 encoding also requires the system `ffmpeg` executable. The current machine
uses the Homebrew installation at `/opt/homebrew/bin/ffmpeg`.

The resulting `P_exist` values are model scores, not calibrated physical
probabilities.
