# Bond-Count Species-Existence Model

This document defines a machine-learning model that scores whether a
formula-degeneracy group may exist. Each sample is represented only by its bond
counts and ring flag. Bond dissociation energies (BDEs) are retained as fixed
physical knowledge used inside the model; they are not duplicated in every
sample record.

## 1. Goal

For a formula-degeneracy group `g`, predict an existence score:

$$
P_{\mathrm{exist}}(g) \in [0,1].
$$

A larger score means that the group resembles observed degeneracies more than
the generated candidates used during training. Without measured probabilities
or confirmed negatives, this score is not a calibrated physical probability.

## 2. Training Data

Use:

```text
training/dataSets/train_validation_1.json
training/dataSets/train_validation_2.json
training/dataSets/train_validation_3.json
```

Every file has this nested hierarchy:

```text
training_set
    positive_samples
        <degeneracy_id>: <sample record>
    pseudo_negative_samples
        <degeneracy_id>: <sample record>
validation_set
    positive_samples
        <degeneracy_id>: <sample record>
    pseudo_negative_samples
        <degeneracy_id>: <sample record>
```

Each degeneracy contains:

```text
degeneracy_id
formula
bond_counts
is_ring
label
class_name
```

The counts in every file are:

| Partition | Positive samples | Pseudo-negative samples | Total |
| --- | ---: | ---: | ---: |
| `training_set` | 214 | 214 | 428 |
| `validation_set` | 53 | 114 | 167 |

The three files contain the same 214 training positives and the same validation
set. They differ only in the 214 pseudo-negative training samples. No molecular
formula is shared between training and validation.

These collection names describe **where the samples came from**, not their PU
training roles:

```text
positive_samples         -> observed-positive source pool
pseudo_negative_samples  -> fabricated source pool
```

Do not directly equate `positive_samples` with `P`, and do not directly equate
`pseudo_negative_samples` with `U`. A PU unlabeled pool must be a mixture that
contains some real positives. Construct the training roles as:

```text
P = 171 retained observed positives
U = 43 hidden observed positives + 214 fabricated samples = 257 samples
```

Thus 20% of the 214 observed training positives are deliberately hidden inside
`U`. Select the 43 hidden positives by complete formula groups with a fixed seed
so closely related degeneracies are not divided between `P` and the known-
positive portion of `U`.

The internal `label` and `class_name` fields are used only to construct and
audit this experiment. They must be removed or masked after samples enter `U`;
the model and loss must not be told which 43 unlabeled records are known
positives. The fabricated samples are also not confirmed negatives.

Minimal loading pattern:

```python
def load_partition(dataset, partition_name):
    partition = dataset[partition_name]
    observed = list(partition["positive_samples"].values())
    fabricated = list(partition["pseudo_negative_samples"].values())
    return observed, fabricated
```

Construct validation roles in the same way: retain 42 observed records as
`P_validation`, then mix 11 hidden observed records with the 114 fabricated
records to form `U_validation` (125 records total). Keep the original source
identities in an evaluation-only copy so all 53 observed positives can be used
to calculate recall. Never expose those hidden identities to the model or loss.

## 3. Degeneracy Feature Vector

Use the bond order stored in the JSON files:

```python
BOND_TYPES = [
    "C-C",
    "C=C",
    "C#C",
    "O-O",
    "O=O",
    "C-O",
    "C=O",
    "C-H",
    "O-H",
]
```

For degeneracy `g`, construct the fixed feature vector:

$$
x_g = [
N_{\mathrm{C-C}},
N_{\mathrm{C=C}},
N_{\mathrm{C\#C}},
N_{\mathrm{O-O}},
N_{\mathrm{O=O}},
N_{\mathrm{C-O}},
N_{\mathrm{C=O}},
N_{\mathrm{C-H}},
N_{\mathrm{O-H}},
I_{\mathrm{ring}}
].
$$

Here:

- `N_b` is the integer count of bond type `b`;
- `I_ring = 1` for a ring group and `0` otherwise.

The input dimension is therefore exactly 10.

Example:

```json
{
  "bond_counts": {
    "C-C": 9,
    "C=C": 0,
    "C#C": 0,
    "O-O": 0,
    "O=O": 0,
    "C-O": 0,
    "C=O": 1,
    "C-H": 19,
    "O-H": 0
  },
  "is_ring": false
}
```

becomes:

```text
[9, 0, 0, 0, 0, 0, 1, 19, 0, 0]
```

Do not include formula, degeneracy ID, class name, or label in the feature
vector. They are identifiers or training targets, not predictors.

## 4. Retaining Bond Dissociation Energy

BDE remains part of the model as a fixed lookup aligned with `BOND_TYPES`:

```python
BDE_BY_BOND = {
    "C-C": 346.0,
    "C=C": 602.0,
    "C#C": 835.0,
    "O-O": 146.0,
    "O=O": 498.0,
    "C-O": 358.0,
    "C=O": 732.0,
    "C-H": 413.0,
    "O-H": 463.0,
}
```

The values are approximate and must remain configurable. BDE depends on the
molecular environment, so these numbers are physical descriptors rather than
exact universal constants.

The important separation is:

```text
sample-specific information = bond counts + is_ring
shared physical information  = BDE lookup by bond type
```

This prevents the same BDE constants from being stored repeatedly in every
degeneracy while ensuring they still influence prediction.

## 5. BDE-Informed Physical Descriptor

Convert each BDE into a fixed fragility weight:

$$
q_b = \exp\left(-\frac{E_b}{E_{\mathrm{scale}}}\right),
$$

where `E_b` is the BDE of bond type `b`. Begin with:

```python
E_SCALE = 400.0  # same energy unit as the BDE table
```

Higher-BDE bonds receive smaller fragility weights. Calculate the total
BDE-informed fragility of degeneracy `g`:

$$
F_{\mathrm{BDE}}(g)
=
\sum_{b} N_{g,b}q_b.
$$

This value is derived from the bond-count vector and the shared BDE table. It is
not an additional field required in the JSON dataset.

Because hydrogen-bond counts can be much larger than backbone-bond counts,
calculate a second backbone-only descriptor:

$$
F_{\mathrm{backbone}}(g)
=
\sum_{b \in \mathcal{B}_{\mathrm{backbone}}} N_{g,b}q_b,
$$

with:

```python
BACKBONE_BONDS = ["C-C", "C=C", "C#C", "O-O", "O=O", "C-O", "C=O"]
```

This prevents `C-H` and `O-H` counts from hiding the heavy-atom structure.

## 6. Recommended Model

Use a small multilayer perceptron for the 10-value feature vector and combine it
with the two fixed BDE-informed descriptors:

$$
h_g = \operatorname{MLP}_{\mathrm{count}}(\operatorname{scale}(x_g)),
$$

$$
z_g
=
\operatorname{MLP}_{\mathrm{out}}
\left([
h_g,
\widetilde{F}_{\mathrm{BDE}}(g),
\widetilde{F}_{\mathrm{backbone}}(g)
]\right),
$$

$$
P_{\mathrm{exist}}(g)=\sigma(z_g).
$$

The tildes indicate descriptors standardized using training-set statistics.
Never fit a scaler on the validation set.

Suggested first architecture:

```text
10 input values
32 hidden units + ReLU
16 hidden units + ReLU
concatenate 2 BDE-informed descriptors
16 hidden units + ReLU
1 output logit
```

This is still a bond-count model: all sample-specific inputs originate from
`bond_counts` and `is_ring`. BDE contributes through a deterministic physical
transformation.

## 7. Minimal PyTorch Model Sketch

```python
class SpeciesExistenceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.count_encoder = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
        )
        self.predictor = nn.Sequential(
            nn.Linear(16 + 2, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, feature_vector, bde_descriptors):
        count_embedding = self.count_encoder(feature_vector)
        combined = torch.cat([count_embedding, bde_descriptors], dim=-1)
        return self.predictor(combined).squeeze(-1)
```

The model returns a logit. Apply `torch.sigmoid(logit)` only when a probability-
like score is needed for reporting.

## 8. Feature Construction

```python
def build_feature_vector(entry):
    bond_counts = entry["bond_counts"]
    return [
        *[float(bond_counts.get(bond, 0)) for bond in BOND_TYPES],
        float(entry["is_ring"]),
    ]
```

Construct BDE descriptors without changing the stored sample schema:

```python
import math


def build_bde_descriptors(entry, energy_scale=400.0):
    counts = entry["bond_counts"]
    fragility = {
        bond: math.exp(-energy / energy_scale)
        for bond, energy in BDE_BY_BOND.items()
    }
    total = sum(counts.get(bond, 0) * fragility[bond] for bond in BOND_TYPES)
    backbone = sum(
        counts.get(bond, 0) * fragility[bond]
        for bond in BACKBONE_BONDS
    )
    return [total, backbone]
```

## 9. Positive-Focused PU Training Objective

After the training mixture is constructed, `P` contains 171 retained observed
degeneracies. `U` contains 43 hidden observed degeneracies mixed with 214
fabricated samples. The loss receives only membership in `P` or `U`; it does not
receive the hidden source labels inside `U`.

For logits from a positive batch and an unlabeled batch, calculate:

$$
R_P^+ = \mathbb{E}_{g\sim P}[\ell(f(g),1)],
$$

$$
R_P^- = \mathbb{E}_{g\sim P}[\ell(f(g),0)],
$$

$$
R_U^- = \mathbb{E}_{g\sim U}[\ell(f(g),0)].
$$

### Positive-only warm-up

For the first 5 epochs, train only on the retained set `P`:

$$
L_{\mathrm{warmup}}=R_P^+.
$$

This first teaches the model that observed degeneracies should receive high
scores. Keep the warm-up short; otherwise the model may learn to label every
sample positive.

### Cost-sensitive nnPU loss

The standard non-negative PU risk is:

$$
L_{\mathrm{nnPU}}
=
\pi_p R_P^+
+
\max\left(0,R_U^- - \pi_p R_P^-\right),
$$

where `pi_p` is an assumed or separately estimated positive-class prior. Do not
derive `pi_p` from the selected positive-to-unlabeled sample ratio.

Because membership in `P` is reliable while membership in `U` is uncertain, use
a cost-sensitive version:

$$
L_{\mathrm{PU}}
=
\alpha\pi_pR_P^+
+
\beta\max\left(0,R_U^- - \pi_pR_P^-\right),
$$

with:

```text
alpha = 1.00
beta  = 0.25
```

`alpha` controls the reliable positive risk. `beta` reduces the effect of the
uncertain unlabeled risk. Test `beta` values `0.10`, `0.25`, `0.50`, and `1.00`;
use `0.25` for the first run.

Minimal loss implementation:

```python
def nnpu_loss(
    positive_logits,
    unlabeled_logits,
    positive_prior,
    positive_weight=1.0,
    unlabeled_weight=0.25,
):
    positive_risk = F.binary_cross_entropy_with_logits(
        positive_logits, torch.ones_like(positive_logits)
    )
    positive_as_negative = F.binary_cross_entropy_with_logits(
        positive_logits, torch.zeros_like(positive_logits)
    )
    unlabeled_as_negative = F.binary_cross_entropy_with_logits(
        unlabeled_logits, torch.zeros_like(unlabeled_logits)
    )
    negative_risk = unlabeled_as_negative - positive_prior * positive_as_negative
    return (
        positive_weight * positive_prior * positive_risk
        + unlabeled_weight * torch.clamp(negative_risk, min=0.0)
    )
```

The constructed unlabeled pool contains at least:

$$
\frac{43}{43+214}\approx0.167
$$

known positive content, and some fabricated samples might also be real
positives. Therefore, begin with `pi_p = 0.20` and test `0.17`, `0.20`, `0.25`,
and `0.30`. Do not use `273 / (273 + 569)`; that ratio reflects dataset
construction rather than the positive prevalence inside `U`.

### Optional weak ranking loss

If nnPU alone does not rank held-out positives sufficiently above unlabeled
candidates, add:

$$
L_{\mathrm{rank}}
=
\frac{1}{M}\sum_{(g^+,g^u)}
\max\left(0,m-[s(g^+)-s(g^u)]\right).
$$

This is a weak preference, not a claim that every `g^u` is negative. Use:

```text
margin = 0.5
lambda_rank = 0.05
```

The complete optional objective is:

$$
L
=
L_{\mathrm{PU}}
+
\lambda_{\mathrm{rank}}L_{\mathrm{rank}}
+
\lambda_{\mathrm{reg}}\lVert\theta\rVert_2^2.
$$

For the first implementation, set `lambda_rank = 0` and apply L2 regularization
through optimizer weight decay.

## 10. Training Procedure

For each of the three dataset files:

1. Read `training_set["positive_samples"]` as `observed_train`.
2. Read `training_set["pseudo_negative_samples"]` as `fabricated_train`.
3. Select complete formula groups containing exactly 43 records from
   `observed_train`, using a fixed seed.
4. Put the remaining 171 observed records in `P_train`.
5. Mix the 43 hidden observed records with all 214 fabricated records to form
   `U_train`; discard their source labels from the training view.
6. Read the validation collections as `observed_validation` and
   `fabricated_validation`.
7. Retain 42 validation observations as `P_validation`; hide the other 11 among
   the 114 fabricated records to form `U_validation` with 125 records. Keep a
   separate evaluation-only copy of the source identities.
8. Assert the raw counts: `214/214` for training and `53/114` for
   validation.
9. Assert the PU counts: training `171/257` and validation `42/125` for `P/U`.
10. Build the 10-value vector and two BDE descriptors for every degeneracy.
11. Fit feature scalers using `P_train + U_train` only.
12. Build independent `P_train` and `U_train` loaders.
13. Warm up for 5 epochs using `P_train` loss only.
14. Train with balanced positive/unlabeled minibatches and cost-sensitive nnPU.
15. Use early stopping based on a positive-focused validation criterion.
16. Repeat with random initialization seeds `11`, `22`, and `33`.

Suggested first settings:

```text
optimizer: Adam
learning rate: 1e-3
weight decay: 1e-4
positive batch size: 32
unlabeled batch size: 32
positive-only warm-up: 5 epochs
maximum epochs: 300
early-stopping patience: 30
positive prior: 0.20
positive-risk weight: 1.00
unlabeled-risk weight: 0.25
ranking-loss weight: 0.00 initially
```

Use all three training variants as robustness experiments. Because their
validation mappings are identical, their validation results are directly
comparable. Three datasets and three initialization seeds produce nine runs.
Report their mean and standard deviation. The final prediction may be the mean
score from the retained models:

$$
s_{\mathrm{ensemble}}(g)
=
\frac{1}{K}\sum_{k=1}^{K}s_k(g).
$$

## 11. Validation

The validation loss sees 42 labeled positives and an unlabeled mixture of 11
hidden positives plus 114 fabricated records. The evaluation code may use a
separate, protected copy of the original source identities to audit behavior,
but those identities must never affect gradients, feature scaling, or model
selection inputs.

The fabricated records are not confirmed negatives. Therefore, do not describe
ordinary accuracy, specificity, or precision as ground-truth measurements.

Report:

- recall of the 53 held-out known positives at score thresholds;
- positive recall among the top 25, 50, and 100 validation scores;
- observed-versus-fabricated source-ranking AUC, clearly labeled as a diagnostic;
- score distributions for observed and fabricated source groups;
- mean and standard deviation across the three training variants and random
  initialization seeds.

Use a positive-focused early-stopping score such as:

$$
S_{\mathrm{validation}}
=
0.7\,\mathrm{Recall}_P
+
0.3\,\mathrm{SourceRankingAUC}_{\mathrm{observed,fabricated}}.
$$

The source-ranking AUC is only a diagnostic because fabricated samples are not
confirmed negatives and the actual PU validation pool contains hidden
positives. Select a reporting threshold that retains at least 90% or 95% of the
53 observed validation records rather than maximizing ordinary accuracy.

Also compare the learned model with two ablations:

```text
count only:       bond counts + is_ring
BDE informed:     bond counts + is_ring + derived BDE branch
```

The BDE-informed design is useful only if it improves held-out ranking or
stability without harming positive recall.

## 12. Feature and Physical Checks

Before training, verify:

1. every sample produces exactly 10 primary features;
2. bond counts are non-negative integers;
3. `is_ring` converts only to `0.0` or `1.0`;
4. every `BOND_TYPES` entry has a BDE value;
5. the training and validation formula sets do not overlap;
6. scaler statistics are calculated from training data only;
7. all BDE descriptors and logits are finite.

Test the BDE transformation separately:

```text
higher BDE -> smaller fragility weight
adding a bond -> never decreases its fragility descriptor
zero bond count -> zero contribution from that bond
```

## 13. Prediction for a New Degeneracy

A new candidate must provide exactly the same fields:

```json
{
  "bond_counts": {
    "C-C": 7,
    "C=C": 1,
    "C#C": 0,
    "O-O": 0,
    "O=O": 0,
    "C-O": 1,
    "C=O": 0,
    "C-H": 15,
    "O-H": 1
  },
  "is_ring": false
}
```

The prediction code must:

1. build the 10-value feature vector in the saved bond order;
2. derive the BDE descriptors from the same saved BDE table;
3. apply the training scalers;
4. run the model;
5. return the score and an out-of-distribution warning when appropriate.

This fixed representation cannot accept a new bond type automatically. Adding
new chemistry such as `C-S` requires updating `BOND_TYPES`, extending the BDE
table, changing the input dimension, and retraining the model. This limitation
is preferable to silently ignoring an unknown bond.

## 14. Saved Outputs

Save:

```text
output/species_existence_model.pt
output/species_existence_scaler.json
output/species_existence_schema.json
output/species_existence_validation_scores.csv
output/species_existence_training_log.csv
```

The schema must record the exact order and physical metadata:

```json
{
  "feature_columns": [
    "C-C", "C=C", "C#C", "O-O", "O=O",
    "C-O", "C=O", "C-H", "O-H", "is_ring"
  ],
  "derived_bde_features": [
    "total_bde_fragility",
    "backbone_bde_fragility"
  ],
  "bde_by_bond": {
    "C-C": 346.0,
    "C=C": 602.0,
    "C#C": 835.0,
    "O-O": 146.0,
    "O=O": 498.0,
    "C-O": 358.0,
    "C=O": 732.0,
    "C-H": 413.0,
    "O-H": 463.0
  },
  "bde_energy_scale": 400.0
}
```

Saving the BDE table with the model prevents training and inference from using
different physical constants.

## 15. Recommended First Implementation

1. Use the fixed 10-value feature vector.
2. Preserve BDE as a shared lookup and derive the two physical descriptors.
3. Hide 43 of the 214 observed training positives inside an unlabeled mixture
   with all 214 fabricated samples.
4. Retain the other 171 observed positives as the labeled-positive set.
5. Warm up on the retained positive set for 5 epochs.
6. Train with cost-sensitive nnPU using `pi_p = 0.20`, positive weight `1.00`,
   and unlabeled weight `0.25`.
7. Apply `1e-4` optimizer weight decay and initially omit ranking loss.
8. Compare the count-only and BDE-informed versions.
9. Repeat across all three training variants and seeds `11`, `22`, and `33`.
10. Test positive priors `0.17`, `0.20`, `0.25`, and `0.30` and unlabeled weights
   `0.10`, `0.25`, `0.50`, and `1.00`.
11. Select a model based primarily on held-out positive recall, ranking, and
   stability—not simply the smallest training loss.

The recommended initial loss is:

$$
\boxed{
L
=
\pi_pR_P^+
+
0.25\max\left(0,R_U^- - \pi_pR_P^-\right)
+
10^{-4}\lVert\theta\rVert_2^2
}.
$$

The final model should be described as:

```text
a BDE-informed bond-count existence scoring model trained with positive and
unlabeled formula-degeneracy groups
```
