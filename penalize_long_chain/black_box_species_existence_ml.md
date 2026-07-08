# Black-Box Machine Learning Model for Species Existence

This document reformulates the problem. Instead of using machine learning only to tune the parameters `A`, `E_ref`, and `k` in a fixed bond-breaking probability formula, we use machine learning to directly predict whether a formula-degeneracy group or molecule-like bond structure may exist.

The central idea is:

```text
bond-count information + bond-strength information + structural descriptors
    -> machine learning model
    -> existence score
```

The model should learn from known species groups, but it should also be able to score an unknown species if we provide its bond counts and bond-strength descriptors.

## 1. Goal

Given a candidate species or formula-degeneracy group `g`, predict:

$$
P_{\mathrm{exist}}(g)
$$

or a probability-like score:

$$
s(g) \in [0, 1]
$$

where:

- high score means the structure is likely to be chemically plausible or observed;
- low score means the structure is unlikely, unstable, or over-penalized by weak/long-chain bonds.

This is no longer a pure parameter-fitting problem. It is a representation-learning problem:

```text
Can the model learn a mapping from bond inventory and bond strength to species existence?
```

## 2. Why Not Just Tune `A`, `E_ref`, and `k`?

The previous plan used a fixed formula:

$$
\mathrm{Arr}_b
=
A \exp \left(-\frac{E_b}{E_{\mathrm{ref}}}\right)
$$

$$
P_{\mathrm{break},b}
=
\frac{1}{1 + k\exp(-\mathrm{Arr}_b)}
$$

and then tuned:

```text
A
E_ref
k
```

That approach is interpretable, but limited. It assumes the functional form is already correct.

The new approach lets the model learn a more flexible relationship:

$$
P_{\mathrm{exist}}(g)
=
f_{\theta}
\left(
\mathrm{bond\ counts},
\mathrm{bond\ strengths},
\mathrm{bond\ descriptors},
\mathrm{global\ structure}
\right)
$$

where `f_theta` is a learned machine learning model.

## 3. Key Design Requirement

The model should be able to handle an unknown species or a newly introduced atom type as long as we can provide its bond-level descriptors.

That means we should avoid a fixed input vector like:

```text
[C-C count, C-O count, C=O count, C=C count, C#C count]
```

because this fixed vector cannot naturally accept a new bond type without changing the input dimension.

Instead, use a set-based representation:

```text
one row per bond type present in the molecule
```

Each row contains numerical descriptors of that bond type.

Example:

| bond | count | BDE | bond_order | atom_1_Z | atom_2_Z | atom_1_en | atom_2_en |
| ---- | ----: | --: | ---------: | -------: | -------: | --------: | --------: |
| C-C  | 9     | 346 | 1          | 6        | 6        | 2.55      | 2.55      |
| C=O  | 1     | 732 | 2          | 6        | 8        | 2.55      | 3.44      |

Then the model sees a collection of bond rows rather than a fixed set of named columns.

## 4. Recommended Model Type: Bond-Set Neural Network

Use a DeepSets-style model.

For each bond type `b`, construct a feature vector:

$$
x_b
=
\left[
N_b,
E_b,
o_b,
Z_1,
Z_2,
\chi_1,
\chi_2,
r_1,
r_2,
\ldots
\right]
$$

where:

- `N_b` is the number of bonds of that type;
- `E_b` is the bond dissociation energy;
- `o_b` is the bond order: 1, 2, or 3;
- `Z_1`, `Z_2` are atomic numbers;
- `chi_1`, `chi_2` are electronegativities;
- `r_1`, `r_2` are atomic radii or covalent radii.

Then apply the same neural network to every bond row:

$$
h_b = \phi_{\theta}(x_b)
$$

Aggregate all bond embeddings:

$$
h_g
=
\sum_{b \in g}
h_b
$$

Finally predict existence:

$$
P_{\mathrm{exist}}(g)
=
\sigma
\left(
\rho_{\theta}(h_g)
\right)
$$

where:

- `phi_theta` is a bond encoder MLP;
- `sum` makes the model independent of bond-row ordering;
- `rho_theta` is a final predictor MLP;
- `sigma` maps the output to `[0, 1]`.

## 5. Why This Helps With Unknown Species

This representation can handle a new species because the model does not require a fixed species label.

It only needs:

```text
bond counts
bond strengths
bond descriptors
optional atom descriptors
```

If a new atom type is added, the model can still accept it if the same descriptor fields are available.

For example, if sulfur is added later, a new bond row could look like:

| bond | count | BDE | bond_order | atom_1_Z | atom_2_Z | atom_1_en | atom_2_en |
| ---- | ----: | --: | ---------: | -------: | -------: | --------: | --------: |
| C-S  | 1     | 272 | 1          | 6        | 16       | 2.55      | 2.58      |

The network architecture does not need to change because this is still the same input feature shape:

```text
[count, BDE, bond_order, atom descriptors]
```

Important caveat:

```text
The model can accept new atom/bond descriptors without changing architecture,
but prediction reliability depends on whether the new chemistry is close to
the training distribution.
```

So this approach reduces the need for retraining, but it does not guarantee perfect extrapolation.

## 6. Input Features

### Bond-Level Features

Recommended bond-level features:

```text
bond_count
bond_dissociation_energy
bond_order
minimum_atomic_number
maximum_atomic_number
mean_atomic_number
absolute_atomic_number_difference
minimum_electronegativity
maximum_electronegativity
mean_electronegativity
electronegativity_difference
minimum_covalent_radius
maximum_covalent_radius
mean_covalent_radius
radius_difference
is_hydrogen_bond
is_carbon_bond
is_oxygen_bond
is_nitrogen_bond
```

For the current dataset, you may start with:

```text
bond_count
bond_dissociation_energy
bond_order
atom_1_atomic_number
atom_2_atomic_number
atom_1_electronegativity
atom_2_electronegativity
```

### Global Group Features

Add global features after aggregating bond embeddings:

```text
total_non_h_bond_count
total_h_bond_count
total_bond_count
is_ring
number_of_distinct_bond_types
estimated_heavy_atom_count
estimated_formula_size
```

Then the model becomes:

$$
h_g
=
\left[
\sum_b \phi_{\theta}(x_b),
x_{\mathrm{global}}
\right]
$$

$$
P_{\mathrm{exist}}(g)
=
\sigma
\left(
\rho_{\theta}(h_g)
\right)
$$

## 7. Labels

The dataset still has no measured probability labels and no true negative examples.

Therefore, observed formula-degeneracy groups are positive examples:

$$
y = 1
$$

Pseudo-negative groups are generated artificial examples:

$$
\tilde{y} = 0
$$

This is a weakly supervised learning problem.

The model should be described as learning:

```text
existence tendency
```

not perfectly calibrated physical probability.

## 8. Positive Samples

Each group from:

```text
formula_degeneracies
```

is an observed positive sample.

For each positive group, build:

```text
bond-row table
global feature vector
label = 1
support_weight = log1p(len(species))
```

The species-list length can be used as a weak confidence weight:

$$
w_g = \log(1 + \mathrm{species\_count}_g)
$$

This should not be interpreted as the true probability of existence.

## 9. Pseudo-Negative Samples

Since there are no true negatives, generate pseudo-negatives.

### Random Count Negatives

Sample bond counts from expanded observed ranges and reject any exact observed signature.

### Corrupted Positive Negatives

Start from an observed group and perturb it:

```text
increase weak-bond counts
increase chain length
add fragile motifs
replace stronger bonds with weaker bonds
```

### Long-Chain Negatives

Generate examples with too many repeated backbone bonds.

This directly targets the long-chain penalty problem.

### Valence-Violation Negatives

If approximate valence rules are available, generate chemically impossible or very unlikely structures.

Example:

```text
too many heavy-atom bonds for a formula size
impossible ring/bond-count combinations
too many multiple bonds for a saturated formula
```

These can be stronger negatives, but they should be used carefully because the current dataset has only approximate bond-count information.

## 10. Model Architecture

A simple PyTorch architecture could be:

```text
Bond rows -> shared bond encoder MLP -> sum pooling -> concatenate global features -> predictor MLP -> existence score
```

### Bond Encoder

$$
h_b = \phi_{\theta}(x_b)
$$

Example:

```text
input dimension: number of bond descriptor features
hidden dimension: 32 or 64
output dimension: 32 or 64
```

### Pooling

Use:

$$
h_{\mathrm{bond\ set}} = \sum_b h_b
$$

or:

$$
h_{\mathrm{bond\ set}} = \frac{1}{|\mathcal{B}_g|}\sum_b h_b
$$

Sum pooling usually preserves molecule-size information better. Mean pooling removes some size information, so if mean pooling is used, total bond count should be included as a global feature.

### Predictor

$$
z_g = \rho_{\theta}
\left(
\left[
h_{\mathrm{bond\ set}},
x_{\mathrm{global}}
\right]
\right)
$$

$$
P_{\mathrm{exist}}(g) = \sigma(z_g)
$$

## 11. Loss Function

Use a combination of binary classification and ranking.

### Weighted Binary Cross-Entropy

For observed positives and pseudo-negatives:

$$
L_{\mathrm{BCE}}
=
-
w_g
\left[
y_g \log p_g
+
(1-y_g)\log(1-p_g)
\right]
$$

where:

- `y_g = 1` for observed groups;
- `y_g = 0` for pseudo-negatives;
- `w_g` can be larger for high-support observed groups.

### Pairwise Ranking Loss

Also require observed groups to score above pseudo-negatives:

$$
L_{\mathrm{rank}}
=
\max
\left(
0,
m - [s(g^+) - s(g^-)]
\right)
$$

where:

$$
s(g) = \mathrm{logit}(P_{\mathrm{exist}}(g))
$$

or simply the model output before sigmoid.

### Total Loss

$$
L
=
L_{\mathrm{BCE}}
+
\lambda_{\mathrm{rank}} L_{\mathrm{rank}}
+
\lambda_{\mathrm{reg}} L_{\mathrm{reg}}
$$

Recommended first version:

```text
Start with BCE only.
Then add ranking loss if pseudo-negative separation is weak.
```

## 12. Training Strategy

### Step 1: Build Positive Dataset

For every formula-degeneracy group:

1. Read `bond_counts`.
2. Convert nonzero bond counts into bond rows.
3. Attach bond energy and atom descriptors to each row.
4. Build global descriptors.
5. Assign label 1.

### Step 2: Generate Pseudo-Negatives

Generate multiple pseudo-negative samples per positive group.

Recommended starting ratio:

```text
3 pseudo-negatives per observed positive
```

Then test:

```text
1:1
3:1
5:1
10:1
```

### Step 3: Split Dataset

Do not split only at the row level if pseudo-negatives are generated from positives.

Better split:

```text
split observed positives first
generate pseudo-negatives separately inside each split
```

Suggested:

```text
70 percent train
15 percent validation
15 percent test
```

### Step 4: Train the Model

Train with:

```text
Adam optimizer
small MLP
early stopping on validation loss
```

Suggested first hyperparameters:

```text
bond encoder hidden dimension: 64
pooled embedding dimension: 64
predictor hidden dimension: 64
learning rate: 1e-3
batch size: 32
epochs: 200
early stopping patience: 20
```

## 13. How to Represent Variable Number of Bond Rows

Each species group can contain a different number of bond types.

Use one of these options:

### Option A: Process One Molecule at a Time

For each sample:

```python
bond_features = tensor[num_bond_types, feature_dim]
```

Apply the bond encoder to all rows and sum over rows.

This is easiest for a first implementation.

### Option B: Padding and Masking

For minibatch training:

```python
bond_features = tensor[batch_size, max_bond_types, feature_dim]
mask = tensor[batch_size, max_bond_types]
```

Apply the encoder and mask padded rows before pooling.

### Option C: Use PyTorch Geometric-Style Batching

Treat bond rows like nodes in a set graph and use a batch index for pooling.

This is elegant but not necessary for the first version.

## 14. Prediction for an Unknown Species

To predict a new species:

1. Estimate its bond counts.
2. For each bond type, provide bond energy and bond descriptors.
3. Build the bond-row table.
4. Build global features.
5. Pass into the model.
6. Output `P_exist`.

Example input:

```json
{
  "candidate_id": "unknown_species_001",
  "is_ring": false,
  "bond_rows": [
    {
      "bond": "C-C",
      "count": 7,
      "BDE": 346,
      "bond_order": 1,
      "atom_1_Z": 6,
      "atom_2_Z": 6,
      "atom_1_en": 2.55,
      "atom_2_en": 2.55
    },
    {
      "bond": "C-S",
      "count": 1,
      "BDE": 272,
      "bond_order": 1,
      "atom_1_Z": 6,
      "atom_2_Z": 16,
      "atom_1_en": 2.55,
      "atom_2_en": 2.58
    }
  ]
}
```

The model can accept the `C-S` row if the feature schema is the same.

## 15. Why This Can Add New Atoms Without Changing the Model

The model does not need a hard-coded `C-S` column.

It only needs numerical descriptors:

```text
count
BDE
bond order
atomic numbers
electronegativity
radius
other atom/bond descriptors
```

Therefore, adding a new atom means adding descriptor values, not changing the neural network input dimension.

This is the main advantage over a fixed bond-count vector.

## 16. Important Extrapolation Warning

The model can technically accept a new atom, but prediction quality depends on the training distribution.

For example, if the model has only seen C/H/O/N chemistry, then a sulfur-containing candidate may be outside the training distribution.

The model may still produce a score, but it should be treated as:

```text
extrapolated prediction
```

not a highly reliable calibrated probability.

To improve this, include atom descriptors that help generalization:

```text
atomic number
period
group
electronegativity
covalent radius
typical valence
bond dissociation energy
bond order
```

## 17. Evaluation

Because there are no true negative examples, evaluate the model in several ways.

### Pseudo-Negative Accuracy

Check whether observed positives score higher than generated pseudo-negatives.

### Ranking Accuracy

For pairs:

$$
(g^+, g^-)
$$

compute:

$$
\mathbb{1}
\left[
s(g^+) > s(g^-)
\right]
$$

### Score Distribution

Plot:

```text
observed positive scores
pseudo-negative scores
```

Good behavior:

```text
positive distribution shifted toward higher score
pseudo-negative distribution shifted toward lower score
```

### Holdout by Formula Size

Train on smaller species and test on larger species.

This tests whether the model understands long-chain penalties.

### Holdout by Bond Type

Hold out groups containing certain bond types and test whether bond-strength descriptors help generalization.

This is important for the unknown-species goal.

## 18. Output Files

Suggested model outputs:

```text
output/species_existence_model.pt
output/species_existence_feature_scaler.json
output/species_existence_feature_schema.json
output/scored_formula_degeneracies_ml.csv
output/pseudo_negative_samples_ml.csv
output/species_existence_training_log.csv
output/species_existence_plots/
```

The feature schema is especially important. It records the required input fields:

```json
{
  "bond_feature_columns": [
    "count",
    "BDE",
    "bond_order",
    "atom_1_Z",
    "atom_2_Z",
    "atom_1_en",
    "atom_2_en"
  ],
  "global_feature_columns": [
    "is_ring",
    "total_bond_count",
    "total_non_h_bond_count",
    "number_of_distinct_bond_types"
  ]
}
```

## 19. Suggested Script Structure

Suggested script:

```text
train_species_existence_model.py
```

Suggested functions:

```python
def load_master_dataset(json_path):
    pass
```

```python
def build_bond_rows(entry, bond_energy_table, atom_descriptor_table):
    pass
```

```python
def build_global_features(entry):
    pass
```

```python
def generate_pseudo_negatives(positive_entries, seed):
    pass
```

```python
def train_model(train_data, validation_data):
    pass
```

```python
def score_candidate(candidate, model, scaler, schema):
    pass
```

```python
def save_outputs(model, scaler, schema, scored_rows, output_dir):
    pass
```

## 20. Minimal PyTorch Model Sketch

```python
class BondSetExistenceModel(nn.Module):
    def __init__(self, bond_feature_dim, global_feature_dim, hidden_dim=64):
        super().__init__()
        self.bond_encoder = nn.Sequential(
            nn.Linear(bond_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim + global_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, bond_features, global_features, mask=None):
        h = self.bond_encoder(bond_features)
        if mask is not None:
            h = h * mask.unsqueeze(-1)
        h = h.sum(dim=1)
        z = torch.cat([h, global_features], dim=-1)
        return self.predictor(z).squeeze(-1)
```

Use:

```python
probability = torch.sigmoid(logit)
```

for interpretation.

## 21. Recommended First Implementation

Start simple:

1. Use observed formula-degeneracy groups as positives.
2. Use random and corrupted pseudo-negatives.
3. Use bond-row features:

```text
count
BDE
bond_order
atom_1_Z
atom_2_Z
atom_1_electronegativity
atom_2_electronegativity
```

4. Use global features:

```text
is_ring
total_bond_count
total_non_h_bond_count
number_of_distinct_bond_types
```

5. Train the bond-set neural network.
6. Score all observed groups.
7. Test manually whether known stable groups score higher than fragile or long-chain pseudo-negatives.

## 22. Final Interpretation

The model should be described as:

```text
a descriptor-based species-existence classifier
```

or:

```text
a bond-strength-informed neural scoring model
```

Its main advantage is extensibility:

```text
new species can be scored if their bond inventory and bond descriptors are provided
```

Its main limitation is extrapolation:

```text
new atom types far outside the training chemistry may still require new data for reliable predictions
```

