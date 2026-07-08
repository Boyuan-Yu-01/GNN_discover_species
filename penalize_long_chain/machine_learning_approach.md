# Machine Learning Approach for Tuning the Bond-Breaking Probability Model

This document explains how to convert the bond-breaking probability idea into a machine learning problem using:

```text
output/master_dataset.json
```

The key difficulty is that the dataset contains observed formula-degeneracy groups, but it does not contain measured species probabilities and does not contain true negative examples. Therefore, the problem should not be treated as ordinary supervised regression or ordinary supervised binary classification.

The recommended formulation is:

```text
weakly supervised positive-unlabeled ranking
```

or, more simply:

```text
learn a chemically reasonable survival score that ranks observed species groups above generated pseudo-negative groups
```

## 1. Dataset

The dataset contains:

```text
formula_degeneracies
```

Each formula-degeneracy group contains:

```text
formula
degeneracy_id
bond_counts
is_ring
species
```

For example, one group may contain:

```json
{
  "formula": "C10H20O",
  "degeneracy_id": "C10H20O_001",
  "bond_counts": {
    "C-C": 9,
    "C=O": 1
  },
  "is_ring": false,
  "species": ["nc9h19cho"]
}
```

Each group should be treated as one machine learning sample.

## 2. ML Sample Definition

Let one formula-degeneracy group be called:

$$
g
$$

For each group, define an input feature vector:

$$
x_g =
\left[
N_{g,\mathrm{C-C}},
N_{g,\mathrm{C=C}},
N_{g,\mathrm{C\#C}},
N_{g,\mathrm{C-O}},
N_{g,\mathrm{C=O}},
\ldots,
I_{\mathrm{ring}}
\right]
$$

where:

- `N_{g,b}` is the number of bonds of type `b` in group `g`;
- `I_ring` is 1 if the group is a ring and 0 otherwise.

For the first version, focus on the selected backbone bonds:

```python
BACKBONE_BONDS = ["C#C", "C-C", "C-O", "C=C", "C=O"]
```

Later, this can be expanded to all non-H bonds:

```python
NON_H_BONDS = [
    "C-C", "C=C", "C#C",
    "C-N", "C=N", "C#N",
    "C-O", "C=O", "C#O",
    "N-N", "N=N", "N#N",
    "O-O", "O=O",
    "O-N", "N=O",
]
```

## 3. Why This Is Not Standard Supervised Learning

In ordinary supervised binary classification, every sample has a label:

$$
y_g \in \{0, 1\}
$$

where:

- `y = 1` means the species exists;
- `y = 0` means the species does not exist.

But in this dataset:

1. Observed species groups are known.
2. Unobserved species groups are not explicitly listed.
3. There is no measured probability for each observed group.
4. There are no true negative examples.

Therefore, we cannot directly train:

```text
features -> true probability
```

Instead, we train:

```text
features -> relative survival score
```

The survival score can later be interpreted as a probability-like quantity, but it should not be considered a calibrated physical probability unless additional labels are added.

## 4. Positive and Pseudo-Negative Samples

### Positive Samples

Every observed formula-degeneracy group is treated as a positive example:

$$
g^+ \in \mathcal{D}_{\mathrm{obs}}
$$

This does not mean every observed group should have probability exactly 1. It only means observed groups should generally receive higher survival scores than chemically implausible generated groups.

### Pseudo-Negative Samples

Because there are no true negatives, generate pseudo-negative groups:

$$
g^- \in \mathcal{D}_{\mathrm{pseudo}}
$$

These are artificial bond-count vectors that are not present in the observed dataset.

Pseudo-negatives should be treated carefully. They are weak labels, not guaranteed physical non-existing species.

## 5. Pseudo-Negative Generation

Use several pseudo-negative generation methods.

### Method 1: Random Bond-Count Sampling

Sample random bond counts within expanded observed ranges.

Example:

```python
negative["C-C"] = random integer from 0 to max_observed_C_C + 3
negative["C-O"] = random integer from 0 to max_observed_C_O + 2
negative["C=C"] = random integer from 0 to max_observed_C_eq_C + 2
negative["C=O"] = random integer from 0 to max_observed_C_eq_O + 2
negative["C#C"] = random integer from 0 to max_observed_C_triple_C + 1
```

Reject a pseudo-negative if its backbone signature exactly matches an observed group.

### Method 2: Corrupted Positives

Start from an observed positive group and perturb it.

Examples:

```text
add extra weak bonds
increase C-C chain length
replace a strong bond with a weaker bond
add additional O-O or C-O structure
```

Example:

```text
observed:        C-C = 9, C=O = 1
pseudo-negative: C-C = 14, C=O = 2, C-O = 2
```

### Method 3: Hard Negatives

Hard negatives should look similar to observed groups but contain extra fragile structure.

These are useful because easy random negatives may be too different from observed data.

Example:

```text
observed:        C-C = 9,  C-O = 1, C=O = 1
hard negative:   C-C = 12, C-O = 2, C=O = 2
```

### Method 4: Long-Chain Penalty Negatives

Because the project goal involves penalizing unlikely long chains, generate pseudo-negatives with long backbones:

```text
C-C count above observed common range
many repeated fragile motifs
large total non-H bond count
```

These examples help the model learn that long fragile structures should have lower survival scores.

## 6. Chemistry-Informed ML Model

Instead of using a generic neural network first, use the chemistry-inspired model as the machine learning model.

For each bond type `b`, define:

$$
\mathrm{Arr}_b
=
A \exp \left(-\frac{E_b}{E_{\mathrm{ref}}}\right)
$$

where:

- `E_b` is the bond dissociation energy of bond type `b`;
- `A` is a positive scale factor;
- `E_ref` is a positive energy scale.

Then map the Arrhenius-like term to bond-breaking probability:

$$
P_{\mathrm{break},b}
=
\frac{1}
{1 + k \exp(-\mathrm{Arr}_b)}
$$

where `k` is another positive parameter.

The learnable parameters are:

$$
\theta =
\left\{
A,
E_{\mathrm{ref}},
k
\right\}
$$

Use log-space parameters during training:

$$
\phi =
\left\{
\log A,
\log E_{\mathrm{ref}},
\log k
\right\}
$$

This guarantees:

$$
A > 0,\quad E_{\mathrm{ref}} > 0,\quad k > 0
$$

## 7. Group Survival Score

For a formula-degeneracy group `g`, estimate survival probability as:

$$
P_{\mathrm{exist}}(g)
=
\prod_{b \in \mathcal{B}}
\left(
1 - P_{\mathrm{break},b}
\right)^{N_{g,b}}
$$

where:

- `B` is the selected bond set;
- `N_{g,b}` is the count of bond type `b` in group `g`.

Compute the score in log space:

$$
s_{\theta}(g)
=
\log P_{\mathrm{exist}}(g)
=
\sum_{b \in \mathcal{B}}
N_{g,b}
\log
\left(
1 - P_{\mathrm{break},b}
\right)
$$

The model score is:

$$
s_{\theta}(g)
$$

Higher score means higher predicted survival or existence tendency.

Because:

$$
0 \leq P_{\mathrm{exist}}(g) \leq 1
$$

the log score satisfies:

$$
s_{\theta}(g) \leq 0
$$

Scores closer to zero mean more stable or more likely to exist.

## 8. Ranking Loss

The most natural training objective is pairwise ranking.

For an observed group `g+` and a pseudo-negative group `g-`, require:

$$
s_{\theta}(g^+) > s_{\theta}(g^-)
$$

Use hinge ranking loss:

$$
L_{\mathrm{rank}}
=
\frac{1}{M}
\sum_{(g^+,g^-)}
\max
\left(
0,
m -
\left[
s_{\theta}(g^+)
-
s_{\theta}(g^-)
\right]
\right)
$$

where:

- `m` is the ranking margin;
- `M` is the number of positive and pseudo-negative pairs.

If the positive score is already larger than the pseudo-negative score by at least `m`, the loss is zero.

## 9. Support-Weighted Ranking

Some formula-degeneracy groups contain many species labels, while others contain only one.

Use:

```python
support_g = len(entry["species"])
weight_g = log1p(support_g)
```

This support value is not a true probability. It is only a weak confidence weight.

A weighted ranking loss can be:

$$
L_{\mathrm{rank}}
=
\frac{1}{M}
\sum_{(g^+,g^-)}
w(g^+)
\max
\left(
0,
m -
\left[
s_{\theta}(g^+)
-
s_{\theta}(g^-)
\right]
\right)
$$

where:

$$
w(g^+) = \log(1 + \mathrm{support}_{g^+})
$$

## 10. Regularization

Because there are only weak labels, the parameters may become extreme.

Add a regularization term:

$$
L_{\mathrm{reg}}
=
\lambda_A (\log A - \log A_0)^2
+
\lambda_E (\log E_{\mathrm{ref}} - \log E_0)^2
+
\lambda_k (\log k - \log k_0)^2
$$

The total loss becomes:

$$
L
=
L_{\mathrm{rank}}
+
L_{\mathrm{reg}}
$$

This keeps the learned parameters near reasonable initial guesses.

Example default values:

```python
A_0 = 10.0
E_0 = 300.0
k_0 = 1000.0
```

## 11. Alternative PU Learning View

This problem can also be described as positive-unlabeled learning.

Observed groups are positive:

$$
y = 1
$$

Unobserved groups are unlabeled:

$$
y = ?
$$

Pseudo-negatives are sampled from the unlabeled or generated space and treated as weak negatives:

$$
\tilde{y} = 0
$$

This is not the same as true negative labeling.

Therefore, ranking is safer than strict binary cross-entropy in the first version.

## 12. Why Not Direct Binary Cross-Entropy First?

Binary cross-entropy would require labels:

$$
y_g \in \{0,1\}
$$

and would use:

$$
L_{\mathrm{BCE}}
=
-
y_g \log p_g
-
(1-y_g)\log(1-p_g)
$$

But the negative labels are artificial. If the pseudo-negatives contain some chemically valid but unobserved species, BCE would punish the model too strongly.

Ranking loss is more forgiving because it only says:

```text
observed groups should score higher than generated pseudo-negatives
```

It does not require pseudo-negatives to have absolute probability zero.

## 13. Training Procedure

### Step 1: Load Dataset

Read:

```text
output/master_dataset.json
```

Extract:

```python
formula_degeneracies = data["formula_degeneracies"]
```

### Step 2: Build Positive Table

Create one row per formula-degeneracy group.

Each row should contain:

```text
degeneracy_id
formula
is_ring
species_count
bond count features
```

For the first version:

```text
C#C
C-C
C-O
C=C
C=O
```

### Step 3: Generate Pseudo-Negatives

Generate pseudo-negative groups using:

```text
random count sampling
corrupted positives
hard negatives
long-chain negatives
```

Use multiple random seeds to test stability.

### Step 4: Split Data

Split observed groups into:

```text
train positives
validation positives
test positives
```

Recommended:

```text
70 percent train
15 percent validation
15 percent test
```

Pseudo-negatives should be generated separately for each split to avoid leakage.

### Step 5: Tune Parameters

For only three parameters, start with random search.

Sample:

```python
log_A     ~ uniform(log(1), log(100))
log_E_ref ~ uniform(log(50), log(1000))
log_k     ~ uniform(log(10), log(1e6))
```

Convert back:

```python
A = exp(log_A)
E_ref = exp(log_E_ref)
k = exp(log_k)
```

Evaluate the ranking loss on training data.

Choose the parameter set with lowest validation loss.

### Step 6: Score All Groups

For each observed formula-degeneracy group:

$$
s_{\theta}(g)
=
\log P_{\mathrm{exist}}(g)
$$

and:

$$
P_{\mathrm{exist}}(g)
=
\exp(s_{\theta}(g))
$$

Save these values.

## 14. Evaluation Metrics

Because there are no true negative examples, evaluation should focus on ranking and diagnostics.

### Pairwise Ranking Accuracy

Compute the fraction of positive and pseudo-negative pairs where:

$$
s_{\theta}(g^+) > s_{\theta}(g^-)
$$

### Pseudo-AUC

Treat positives as 1 and pseudo-negatives as 0 only for diagnostic purposes.

Compute ROC-AUC or PR-AUC if available.

This is not a true physical AUC because the negatives are generated.

### Score Distribution Plot

Plot:

```text
observed log survival scores
pseudo-negative log survival scores
```

Good behavior:

```text
observed distribution is shifted higher than pseudo-negative distribution
```

### Bond Probability Check

For each bond type, compute:

$$
P_{\mathrm{break},b}
$$

Expected trend:

```text
higher BDE -> lower P_break
```

### Sensitivity to Random Seed

Repeat pseudo-negative generation and tuning with different seeds.

Good behavior:

```text
best parameters remain in the same rough range
rankings do not change drastically
```

## 15. Recommended Output Files

The ML tuning script should generate:

```text
output/tuned_bond_breaking_parameters.json
output/scored_formula_degeneracies.csv
output/pseudo_negative_samples.csv
output/tuning_diagnostics.csv
output/tuning_plots/
```

### Parameter JSON

```json
{
  "A": 10.0,
  "E_ref": 300.0,
  "k": 1000.0,
  "train_loss": 0.123,
  "validation_loss": 0.145,
  "bond_set": ["C#C", "C-C", "C-O", "C=C", "C=O"],
  "bde_by_bond": {
    "C-C": 346.0,
    "C-O": 358.0,
    "C=C": 602.0,
    "C=O": 732.0,
    "C#C": 835.0
  }
}
```

### Scored CSV

```text
degeneracy_id
formula
is_ring
species_count
log_P_exist
P_exist
C#C
C-C
C-O
C=C
C=O
```

### Diagnostic Plots

Recommended plots:

```text
P_break versus BDE
observed versus pseudo-negative score distributions
top 20 highest-scoring groups
bottom 20 lowest-scoring groups
parameter sensitivity across random seeds
```

## 16. Suggested Script Structure

Suggested script:

```text
tune_parameters_ml.py
```

Suggested location:

```text
penalize_long_chain/tune_parameters_ml.py
```

Suggested functions:

```python
def load_formula_degeneracies(json_path):
    pass
```

```python
def build_positive_rows(formula_degeneracies, bond_keys):
    pass
```

```python
def generate_pseudo_negatives(positive_rows, bond_keys, n_negatives, seed):
    pass
```

```python
def calculate_p_break_from_energy(E_b, A, E_ref, k):
    arr = A * np.exp(-E_b / E_ref)
    return 1.0 / (1.0 + k * np.exp(-arr))
```

```python
def calculate_log_survival(row, bond_keys, bde_by_bond, A, E_ref, k):
    pass
```

```python
def ranking_loss(positive_rows, negative_rows, params):
    pass
```

```python
def random_search(train_data, validation_data, parameter_ranges, n_trials, seed):
    pass
```

```python
def score_observed_groups(rows, best_params):
    pass
```

```python
def write_outputs(best_params, scored_rows, diagnostics, output_dir):
    pass
```

## 17. Minimal Training Pseudocode

```python
data = load_formula_degeneracies("output/master_dataset.json")
positive_rows = build_positive_rows(data, BACKBONE_BONDS)

train_pos, val_pos, test_pos = split_positive_rows(positive_rows)

train_neg = generate_pseudo_negatives(train_pos, BACKBONE_BONDS, n_negatives=5000, seed=1)
val_neg = generate_pseudo_negatives(val_pos, BACKBONE_BONDS, n_negatives=1000, seed=2)
test_neg = generate_pseudo_negatives(test_pos, BACKBONE_BONDS, n_negatives=1000, seed=3)

best_params = random_search(
    train_data=(train_pos, train_neg),
    validation_data=(val_pos, val_neg),
    parameter_ranges={
        "A": (1.0, 100.0),
        "E_ref": (50.0, 1000.0),
        "k": (10.0, 1e6),
    },
    n_trials=10000,
    seed=0,
)

scored_rows = score_observed_groups(positive_rows, best_params)
write_outputs(best_params, scored_rows, diagnostics, output_dir="output")
```

## 18. Baseline Models

Before trusting the chemistry-informed model, compare it against simple baselines.

### Baseline 1: Total Bond Count Penalty

$$
s(g)
=
-
\alpha
\sum_b
N_{g,b}
$$

This tests whether the model is doing more than simply penalizing large molecules.

### Baseline 2: Weighted Bond Count

$$
s(g)
=
-
\sum_b
w_b N_{g,b}
$$

where weaker bonds get larger weights.

### Baseline 3: Logistic Regression

Use pseudo-labels:

```text
observed = 1
pseudo-negative = 0
```

Then train logistic regression on bond-count features.

This is only a diagnostic baseline, not the preferred final model.

## 19. Interpretation of Learned Parameters

The learned parameters should be interpreted carefully.

### `A`

Controls the scale of the Arrhenius-like term.

Larger `A` generally increases `Arr_b`, which increases `P_break`.

### `E_ref`

Controls how sensitive the mapping is to bond dissociation energy.

Small `E_ref` makes `Arr_b` decay quickly with `E_b`.

Large `E_ref` makes `Arr_b` decay slowly with `E_b`.

### `k`

Controls the sigmoid shift and lower-limit behavior.

When `E_b` is very large:

$$
\mathrm{Arr}_b \rightarrow 0
$$

so:

$$
P_{\mathrm{break},b}
\rightarrow
\frac{1}{1+k}
$$

Therefore, large `k` makes the high-BDE bond-breaking probability closer to zero.

## 20. Main Caveat

This ML problem is not fully identifiable from the current dataset.

The dataset tells us:

```text
which formula-degeneracy groups are observed
```

but it does not tell us:

```text
the true probability of each group
which unobserved groups are truly impossible
```

Therefore, the learned model should be described as:

```text
a weakly supervised, chemistry-informed ranking model
```

not as:

```text
a calibrated probability model
```

The model becomes a true supervised probability model only if future data provide measured quantities such as:

```text
species occurrence frequency
mole fraction
active-species probability
simulation survival frequency
confirmed absent species
```

## 21. Recommended First Version

The first useful implementation should do only this:

1. Load `master_dataset.json`.
2. Build observed positive rows from backbone bond counts.
3. Generate pseudo-negatives by random sampling and corrupted positives.
4. Tune `A`, `E_ref`, and `k` using ranking loss.
5. Save best parameters.
6. Score all observed formula-degeneracy groups.
7. Plot score distributions and bond-level probabilities.

This keeps the first ML version simple, interpretable, and close to the chemistry idea.

