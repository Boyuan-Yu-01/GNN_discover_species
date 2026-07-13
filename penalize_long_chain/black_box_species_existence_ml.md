# BDE-Informed Species-Existence Model
<span style="color:red">Help us to identify any missing species in a combustion temperature(system)?</span>  

## 1. Goal

Train a positive-unlabeled (PU) model that assigns an existence score to each
formula-degeneracy group:

$$
\widehat P_{\mathrm{exist}}(g)=\sigma(z_g).
$$

Observed degeneracies have no measured existence probabilities, and fabricated
samples are not confirmed negatives. Therefore, interpret the output as a
relative existence score rather than a calibrated physical probability.

## 2. Data and PU Construction

Use:

```text
training/org_dataset/filtered_master.json
training/org_dataset/filtered_pseudo_negative.json
```

Both files store records under `formula_degeneracies`:

| Source | Records |
| --- | ---: |
| Observed degeneracies | 267 |
| Fabricated degeneracies | 569 |

Construct one transductive PU dataset:

```text
P = 200 known observed degeneracies

U =  67 hidden observed degeneracies
   + 380 fabricated degeneracies
   = 447 unlabeled degeneracies
```

The known-positive fraction in `U` is:

$$
\frac{67}{447}\approx0.15.
$$

Select the 67 hidden observations by complete molecular-formula groups with a
fixed seed. Do not divide degeneracies of the same formula between `P` and the
hidden-positive part of `U`. Sample the 380 fabricated records reproducibly.

Remove source labels from the model-facing `U` records. Store the identities of
the 67 hidden positives separately for evaluation only. Preserve the remaining
189 fabricated records for later external scoring or sensitivity analysis.

## 3. Primary Feature Vector

Use this fixed bond order:

```python
BOND_TYPES = [
    "C-C", "C=C", "C#C",
    "O-O", "O=O",
    "C-O", "C=O",
    "C-H", "O-H",
]
```

For degeneracy `g`:

$$
x_g=[
N_{\mathrm{C-C}},N_{\mathrm{C=C}},N_{\mathrm{C\#C}},
N_{\mathrm{O-O}},N_{\mathrm{O=O}},
N_{\mathrm{C-O}},N_{\mathrm{C=O}},
N_{\mathrm{C-H}},N_{\mathrm{O-H}},I_{\mathrm{ring}}
].
$$

The input dimension is 10. Formula, degeneracy ID, source class, and protected
labels are not model features.

## 4. BDE-Informed Descriptors

Keep BDE as shared model metadata:

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

BACKBONE_BONDS = ["C-C", "C=C", "C#C", "O-O", "O=O", "C-O", "C=O"]
E_SCALE = 400.0
```

These BDE values are approximate and must remain configurable. Define a
dimensionless relative fragility weight:

$$
q_b=\exp\left(-\frac{E_b}{E_{\mathrm{scale}}}\right).
$$

`q_b` is not a calibrated bond-breaking probability. Calculate:

$$
F_{\mathrm{BDE}}(g)=\sum_{b\in\mathcal B_{\mathrm{all}}}N_{g,b}q_b,
$$

$$
F_{\mathrm{backbone}}(g)
=\sum_{b\in\mathcal B_{\mathrm{backbone}}}N_{g,b}q_b.
$$

`F_BDE` represents total BDE-weighted molecular exposure. `F_backbone` isolates
heavy-atom structural fragility from the usually larger `C-H` and `O-H` counts.

The backbone descriptor is related to the earlier survival model:

$$
P_{\mathrm{exist}}(g)
=\prod_b(1-P_{\mathrm{break},b})^{N_{g,b}}.
$$

Taking the negative logarithm gives:

$$
-\log P_{\mathrm{exist}}(g)
=\sum_bN_{g,b}[-\log(1-P_{\mathrm{break},b})].
$$

For small breaking probabilities, `-log(1-p)` is approximately `p`. If breaking
probability is approximately proportional to `q_b`, `F_backbone` is a first-
order proxy for negative log survival without claiming a calibrated physical
probability.

## 5. Preprocessing

For each of the nine bond counts:

$$
u_{g,b}=\log(1+N_{g,b}),
$$

$$
\widetilde{x}_{g,b}
=\frac{u_{g,b}-\mu_b}{\sigma_b+\epsilon}.
$$

Use the following rules:

```text
Bond counts:       log1p -> feature-wise standardization -> clip to [-5, 5]
is_ring:           keep as 0.0 or 1.0
BDE descriptors:   log1p -> separate standardization -> clip to [-5, 5]
```

Fit the 9-column count scaler and 2-column BDE scaler using model-facing `P + U`
only. Freeze and save both scalers. Do not normalize a degeneracy by total bond
count or vector norm because that would remove absolute chain-length
information.

## 6. Model Architecture

The selected model uses the scaled 10-value vector and both BDE descriptors:

$$
h_g=\operatorname{MLP}_{\mathrm{count}}(\widetilde{x}_g),
$$

$$
z_g=\operatorname{MLP}_{\mathrm{out}}
\left([h_g,\widetilde F_{\mathrm{BDE}}(g),
\widetilde F_{\mathrm{backbone}}(g)]\right).
$$

Architecture:

```text
10 primary inputs
-> Linear(10, 32) + ReLU
-> Linear(32, 16) + ReLU
-> concatenate 2 scaled BDE descriptors (18 values total)
-> Linear(18, 16) + ReLU
-> Linear(16, 1)
-> output logit
```

Minimal PyTorch definition:

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
            nn.Linear(18, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, feature_vector, bde_descriptors):
        count_embedding = self.count_encoder(feature_vector)
        combined = torch.cat([count_embedding, bde_descriptors], dim=-1)
        return self.predictor(combined).squeeze(-1)
```

Use logits directly in the loss. Apply sigmoid only when reporting scores.

## 7. Positive-Focused nnPU Loss

For positive logits from `P` and unlabeled logits from `U`, calculate:

$$
R_P^+=\mathbb E_{g\sim P}[\ell(f(g),1)],
$$

$$
R_P^-=\mathbb E_{g\sim P}[\ell(f(g),0)],
$$

$$
R_U^-=\mathbb E_{g\sim U}[\ell(f(g),0)].
$$

Use the cost-sensitive non-negative PU objective:

$$
L_{\mathrm{PU}}
=\alpha\pi_pR_P^+
+\beta\max(0,R_U^- - \pi_pR_P^-).
$$

Initial settings:

```text
positive prior pi_p:       0.15
positive weight alpha:     1.00
unlabeled weight beta:     0.25
weight decay:              1e-4
ranking-loss weight:       0.00
```

The `max` prevents the estimated negative risk from becoming negative. The
reduced `beta` reflects lower trust in `U`. This weighting is a deliberate
cost-sensitive heuristic, so test:

```text
pi_p:  0.15, 0.18, 0.20, 0.25
beta:  0.10, 0.25, 0.50, 1.00
```

Do not use a prior below `0.15`, because 67 of the 447 records in `U` are known
internally to be positive.

## 8. Training Procedure

1. Load all 267 observed and 569 fabricated records.
2. Construct `P` with 200 observations and `U` with 67 hidden observations plus
   380 fabricated records.
3. Mask all source information inside `U`.
4. Build the 10 primary features and two BDE descriptors.
5. Fit and apply the two scalers.
6. Create independent `P` and `U` loaders.
7. Warm up for 5 epochs using only positive loss `R_P+`.
8. Train with balanced minibatches: 32 from `P` and 32 from `U`.
9. Use Adam with learning rate `1e-3`, maximum 300 epochs, and patience 30 on
   the smoothed nnPU training loss.
10. Repeat with initialization seeds `11`, `22`, and `33`.
11. Average retained model scores if an ensemble is desired.

Keep the positive-only warm-up short; otherwise the model may learn to score
everything as positive.

## 9. Hidden-Positive Recovery

After training, rank all 447 members of `U`. Reveal the 67 protected identities
only for evaluation:

$$
\operatorname{Recall@K}
=\frac{\text{hidden positives among the top K}}{67}.
$$

Recovery cannot exceed 100%. A review budget may exceed 100%:

$$
\operatorname{ReviewBudget}(K)=\frac{K}{67}\times100\%.
$$

Report:

| Ranked candidates | Review budget | Metric    |
| ----------------: | ------------: | --------- |
|            Top 67 |        100.0% | Recall@67 |
|            Top 74 |        110.4% | Recall@74 |
|            Top 80 |        119.4% | Recall@80 |

Also report Recall@25, Recall@50, the largest hidden-positive rank, and mean and
standard deviation across initialization seeds. Finding all 67 positives among
the top 74 means:

$$
\operatorname{Recall@74}=100\%,\qquad
\operatorname{Precision@74}=67/74\approx90.5\%.
$$

This is transductive recovery because the hidden feature vectors participated
in training as unlabeled records. Fabricated records are not confirmed
negatives, so ordinary accuracy and specificity are not physical ground-truth
metrics.

Repeatedly selecting hyperparameters using all 67 protected identities would
produce optimistic results. Reserve some identities for one-time final audit or
use a repeated formula-grouped hidden-positive protocol.

## 10. Required Checks

Before training, verify:

1. `len(P) == 200` and `len(U) == 447`;
2. `U` contains exactly 67 protected hidden positives and 380 fabricated records;
3. no molecular formula is divided between `P` and hidden positives in `U`;
4. every sample produces 9 bond counts plus one binary ring feature;
5. all bond counts are non-negative integers;
6. every configured bond has a BDE value;
7. `F_BDE` uses all nine bonds;
8. `F_backbone` excludes `C-H` and `O-H`;
9. the count scaler has 9 columns and the BDE scaler has 2 columns;
10. transformed features, descriptors, logits, and losses are finite;
11. a fixed seed reproduces the same PU construction;
12. protected hidden identities never enter model inputs or the PU loss.

## 11. Saved Outputs

Save:

```text
output/species_existence_model.pt
output/species_existence_count_scaler.json
output/species_existence_bde_scaler.json
output/species_existence_schema.json
output/species_existence_recovery_scores.csv
output/species_existence_training_log.csv
```

The schema must record:

```json
{
  "model": "BDE-informed bond-count MLP",
  "feature_columns": [
    "C-C", "C=C", "C#C", "O-O", "O=O",
    "C-O", "C=O", "C-H", "O-H", "is_ring"
  ],
  "derived_bde_features": [
    "total_bde_fragility",
    "backbone_bde_fragility"
  ],
  "preprocessing": {
    "bond_counts": "log1p_then_standardize",
    "is_ring": "unchanged_binary",
    "bde_descriptors": "log1p_then_standardize",
    "clip": [-5.0, 5.0]
  },
  "bde_by_bond": {},
  "bde_energy_scale": 400.0,
  "positive_prior": 0.15,
  "positive_weight": 1.0,
  "unlabeled_weight": 0.25,
  "split_seed": 0,
  "model_seed": 0
}
```

Save the complete BDE table and actual seeds rather than the empty or zero
placeholders shown above. Reuse the saved schema and scalers unchanged when
scoring the remaining 189 fabricated records or future degeneracies.
