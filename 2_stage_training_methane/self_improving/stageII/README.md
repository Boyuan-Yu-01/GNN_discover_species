# PUCT exploration in self_improving

This folder contains the new PUCT (Monte Carlo tree search) implementation.
The original sibling `../../stageI/` and `../../stageII/` Python files are unchanged.

## Run

From `2_stage_training_methane/self_improving/stageII`:

```sh
conda activate GNN
python test_stage2.py
python explore_stage2.py
```

The default mode lets PUCT choose how many atoms to add and when to STOP:

```python
GENERATION_MODE = "free"
STOP_WHEN_ALL_SPECIES_RECOVERED = False
EXPLORATION_RUNS = int(os.environ.get("STAGE2_RUNS", "10"))
ATTEMPTS_PER_RUN = int(os.environ.get("STAGE2_ATTEMPTS", "1000"))
```

This produces **10,000 construction attempts** (10 runs × 1,000 attempts).
The number of attempts is separate from the size of each generated molecule.

For a shorter run:

```sh
STAGE2_RUNS=1 STAGE2_ATTEMPTS=100 python explore_stage2.py
```

To request exact formulas again, set `GENERATION_MODE = "formula"`.
Then `COMPOSITIONS` and `SAMPLES_PER_COMPOSITION` apply; the existing defaults
give 10 × 29 × 100 = 29,000 attempts.

Full-recovery stopping remains optional: set
`STOP_WHEN_ALL_SPECIES_RECOVERED = True` to enable it for PUCT.

## Free growth and automatic STOP

Every attempt starts with an empty graph and **no requested C,H,O counts**.
START chooses the first element. After that, STOP competes with GROW and CONNECT.
Every successful attempt must choose STOP; reaching the action limit is a failure.

The current stage-I checkpoint was trained with a formula input. The
`MarginalGrowthPolicy` adapter evaluates that checkpoint under every distinct
active reference composition and averages its action probabilities for the
current graph. This averaged policy guides the search without assigning any
formula to an attempt. It can generate compositions absent from the reference
set. Averaging is performed at each visited state; no formula is secretly
selected and enforced. Temperature and uniform exploration are then applied.

This is an inference adapter, not a retrained unconditional network. Its STOP
preference still comes from the conditional training data. Unknown stopped
graphs receive the existing novelty reward; their chemical stability and
importance are not established by stopping or by the reward.

The existing H/O/C encoding, connected growth, valence rules, **10-atom cap** and
**24-action cap** remain. These are upper bounds, not target atom counts.
CONNECT can still modify bonds at the atom cap, and STOP is available.
No minimum molecule size is imposed beyond requiring a first atom.

In free mode:
- `COMPOSITIONS`, `SAMPLES_PER_COMPOSITION`, and exact-formula masks are inactive.
- There is one persistent tree per run, shared across generated compositions.
- A stopped graph is matched against all compatible reference graphs.
- `requested_C_H_O` is null; `generated_C_H_O` records the actual composition.
- Deduplication uses actual composition plus exact graph isomorphism.
- `frequency_overall` divides occurrences by all attempts in the run/campaign.
  The formula-specific frequency is null. PNG/MP4 labels use actual counts and
  overall frequencies.
- Summaries use `per_generated_composition`. No attempt fails merely because
  its formula differs from a target.

To use the previous sampling algorithm in this copy:

```sh
STAGE2_METHOD=sampling STAGE2_RUNS=1 python explore_stage2.py
```

Classes appear first, followed by editable parameters and direct calls.
There is no main guard. Paths are configurable and relative to the working
directory. Importing the exploration script directly runs its campaign.

The default checkpoint and active reference JSON are read from the local copy in
`../stageI` (inside `self_improving`).
The current checkpoint is the local run `20260921_221023_510656`.
After training again, edit `CHECKPOINT_PATH` in `explore_stage2.py`.
Tests read its literal `STAGE1_DIRECTORY`, `CHECKPOINT_PATH`, and
`REFERENCE_JSON_PATH` settings without starting exploration, so these paths
only need to be configured in one file. Run the tests from this `stageII` folder.
The FFCMII YAML and Excel workbook in the parent folder are **not** automatically
converted into a new neural-network checkpoint or active reference set.
“Unknown” means absent from the configured `REFERENCE_JSON_PATH`, currently
the 31 active stage-I reference graphs.

## Why ordinary sampling misses unknown structures

Stage I learned to imitate reference formation sequences. Ordinary sampling
therefore spends most attempts on actions that recreate those references.
A low network probability does not establish that a different structure is
unimportant or chemically impossible.

PUCT adds feedback to the **search tree**. Completed trajectories receive a
reward; the tree uses those observations to choose subsequent branches.
The network remains frozen. This is neither PPO nor neural-network retraining.

## Selection rule

For a partial graph s and action a:

```text
score(s,a) = Q(s,a) + c_puct * P_search(s,a) * sqrt(max(1, N(s))) / (1 + N(s,a))
```

- `Q(s,a)`: average terminal reward observed through this edge.
- `N(s)`: simulations that visited the partial graph.
- `N(s,a)`: simulations that selected this edge.
- `P_search(s,a)`: the exploration prior for the action.
- `c_puct`: how strongly the search favors insufficiently visited actions.

The `max(1, N(s))` convention uses the prior to select the first action at an
unvisited node. This implementation uses deterministic tie breaking.

PUCT follows the policy-guided tree-search principle described by
[Silver et al., Mastering the game of Go with deep neural networks and tree search](https://www.nature.com/articles/nature16961).
Here, completed molecule rollouts supply rewards because our stage-I model
has no trained value head.

### Protect low-probability actions

PUCT alone can still neglect actions with extremely small priors. This
implementation retains every admissible action and mixes the policy with a
uniform distribution:

```text
P_search(a|s) = (1 - epsilon) * P_constrained(a|s) + epsilon / K
```

Here `K` is the number of admissible actions and the default `epsilon = 0.25`.
The policy is temperature-adjusted and renormalized over admissible actions
before the uniform mixture. The same mixture is used for rollout sampling.
There is no top-k pruning. A positive prior improves access to rare actions,
but finite search budgets do not guarantee every branch will be visited.

## One simulation

1. Start at the empty-graph root for the requested C,H,O composition.
2. Select tree edges using the PUCT score.
3. On reaching a new child, finish that partial graph with a policy rollout.
4. Evaluate the completed result using the reward below.
5. Add that same reward to every selected tree edge and increment visits.
6. Save the entire trajectory, including rollout actions, as one attempt.

### Stop when the reference set is recovered

With full recovery enabled in formula mode, a persistent tree is kept for each requested
composition. The scheduler takes one simulation at a time, cycling through
formulas that still have missing target species. A formula is skipped after all
its target isomers are recovered. Search ends immediately after the last target
is recovered, without finishing a batch or starting another independent run.

In free mode, the single tree continues until all target graphs are recovered.

A species counts as recovered only after STOP, with any requested composition satisfied,
and an atom- and bond-order-preserving graph-isomorphism match. Formula equality
alone does not recover both CH3O and CH2OH. Duplicate observations do not increase
coverage. Unknown candidates found along the way remain in the outputs.

`RECOVERY_SPECIES = None` targets all active keys from the configured reference
JSON: currently 31 species after the checkpoint's CO exclusion. Excluded species
are explicitly listed in logs and recovery metadata. To choose a smaller set,
provide keys, for example `RECOVERY_SPECIES = ["CH3O", "CH2OH"]`.

Requested targets must be active, their compositions must be included in
`COMPOSITIONS`, and their minimum construction length must fit the checkpoint's
step limit. Otherwise the script raises an explanatory error. Positive uniform
prior mixing is required so admissible actions have nonzero search priors.
These checks do not turn the current model into an all-96-species FFCMII model.

`SAMPLES_PER_COMPOSITION`, `EXPLORATION_RUNS`, and `BATCH_SIZE` do not terminate
full-recovery search. The per-molecule step limit still ends an individual
failed attempt; search then continues. There is no global time or attempt cap.
Manual interruption or an error is not reported as successful recovery.
If interrupted, the latest `recovery_progress.json` and log retain the recorded
coverage; full inventories and media are written after normal completion.

In bounded mode (`STOP_WHEN_ALL_SPECIES_RECOVERED = False`), the previous
behavior remains: one tree per composition per independent run, with
`SAMPLES_PER_COMPOSITION` simulations. Every simulation is recorded as an
attempt. In either mode, frequencies use the actual attempt count for each
requested composition.

The tree indexes construction histories. Equivalent partial graphs reached by
different histories are not merged in the tree. Final outputs are deduplicated
using atom- and bond-aware graph isomorphism, as in the original Stage II.

## Reward and constraints

| Outcome | Default reward |
|---|---:|
| STOP, exact requested composition, absent from active references | 1.0 |
| STOP, exact requested composition, matches an active reference | 0.1 |
| Wrong composition, dead end, or step limit | 0.0 |

Rewards remain fixed throughout a run. Repeated observations of a candidate
still receive 1.0; this first implementation optimizes candidate completion
rather than counting only first discoveries. Report **unique candidate
structures** separately from candidate trajectories.

This is a novelty proxy. It does not measure stability, reaction relevance,
kinetic importance, or physical abundance. A useful chemistry/property evaluator
can later replace or extend `PUCTSearch.reward`.

In formula mode, search additionally forbids adding an element beyond the requested
count and forbids STOP until all requested counts are present. CONNECT remains
available when chemically legal, even after all atoms are present. A branch with
no admissible actions is recorded as `failed_dead_end`; it is not forced to STOP.

Stage I training masks are unchanged. The checkpoint's existing H/O/C encoding,
valence rules and size/step limits still apply. PUCT cannot bypass those rules;
it does not add formal-charge support for CO or support for all 96 FFCMII species.

## Editable parameters

All parameters are below the classes in `explore_stage2.py`.

| Parameter | Default | Meaning |
|---|---:|---|
| SEARCH_METHOD | "puct" | Choose "puct" or original "sampling" |
| GENERATION_MODE | "free" | Free growth or formula-constrained growth |
| ATTEMPTS_PER_RUN | 1000 | Construction attempts per run in free mode |
| STOP_WHEN_ALL_SPECIES_RECOVERED | False | Use configured counts; True enables full-recovery stopping |
| RECOVERY_SPECIES | None | All active species keys, or an explicit list |
| PUCT_C | 2.0 | Exploration strength |
| PUCT_UNIFORM_FRACTION | 0.25 | Prior mass reserved for uniform exploration |
| PUCT_REQUIRE_COMPOSITION | True only in formula mode | Apply exact-formula constraints |
| PUCT_REFERENCE_REWARD | 0.1 | Known, complete graph reward |
| PUCT_CANDIDATE_REWARD | 1.0 | Unknown, complete graph reward |
| SAMPLES_PER_COMPOSITION | 100 | Per-composition budget in bounded formula mode only |
| EXPLORATION_RUNS | 10 | Independent runs in bounded mode only |
| TEMPERATURE | 1.0 | Temperature before prior mixing |
| COMPOSITIONS | None | All unique active reference compositions |
| SAVE_MEDIA | True | Save final PNG and growth MP4 |

Free growth can explore new compositions without listing them. In formula mode, explicitly list
them, e.g. `COMPOSITIONS = [(2, 4, 1), (2, 6, 1)]`. Counts always use C,H,O order
and must fit the checkpoint's atom limit.

In bounded mode, increasing simulations per run lets each tree accumulate more evidence.
Increasing the number of runs restarts the tree more often.
`BATCH_SIZE` controls batching for ordinary sampling; PUCT is sequential so each
simulation can use the preceding simulation's reward.

For backward compatibility, programmatic configurations default to sampling and
`stop_when_all_species_recovered=False`. The script's editable parameters enable
PUCT with count-controlled stopping by default.

## Outputs

Results are saved directly under `output/`, relative to the working directory
(normally `self_improving/stageII/output`). Every execution replaces the previous
exploration results, including the old logs and `runs/` folder. Stale PNG/MP4
and PUCT statistics are removed even when the new run disables those outputs.
There is no timestamped parent folder. `STAGE2_OUTPUT_FOLDER` can override the path.

The campaign saves the original compact JSON files, console/file logs,
`final_structures.png`, and `exploration_growth.mp4`. Full-recovery files are written directly to `output/`.
Bounded campaigns additionally have per-run files under `runs/run_001/`, etc. Orange panels are unvalidated unknown candidates.

Additional PUCT diagnostics:

- `recovery_progress.json`: target, recovered, missing, and excluded species;
  updated on every new recovery and every 25 attempts. Final `summary.json`
  records `all_species_recovered` and `termination_reason`.

- `search_statistics.json`: simulations, tree nodes, model evaluations, and root
  edge priors, visit counts and mean rewards. Campaign records retain run IDs.
- Each attempt's `search`: terminal reward, selected tree depth, simulation number,
  and whether the search reached a dead end.
- Each action: original temperature-adjusted `policy_probability`, mixed
  `search_prior`, selection method, and tree visits/Q/U before selection.
- `probability` is 1 for deterministic PUCT choices, and the mixed sampling
  probability for rollout choices. It is not the network probability for PUCT.
- Progress is logged every 25 simulations, with per-composition outcome totals.

Occurrence frequencies describe this search procedure. They are not model
probabilities or molecule abundances. Weights are not updated, so a higher
discovery frequency does not imply a higher probability in the saved network.

## Verification and initial comparison

All 16 tests in `test_stage2.py` pass. They check rare-prior exploration and
reward backup, composition constraints, dead ends, parameter validation,
repeatability, trace replay, unchanged checkpoint bytes, campaign merging,
full-coverage stopping beyond nominal budgets, isomer-specific recovery, and
rejection of unreachable target settings, probability averaging, free STOP and
reward behavior, and reproducible free-growth campaign outputs. A separate real-checkpoint validation
recovered all 31 active species with nominal sample/run budgets both set to 1.

The earlier bounded-mode comparison used seed 24680, one run, and 100 attempts for each of
C2H4O, C2H6O and CH4O (300 attempts per method):

| Method | Reference attempts | Unknown candidate attempts | Failed attempts | Unique unknown structures |
|---|---:|---:|---:|---:|
| Original sampling | 128 | 8 | 164 | 2 |
| PUCT + mixed prior + composition constraints | 110 | 60 | 130 | 6 |

For C2H4O alone, candidate attempts increased from 1 to 35 per 100 attempts.
For C2H6O, they increased from 7 to 25. CH4O produced no unknown candidates in
either method. C2H6O is absent from the active training compositions.

The comparison is stored under
`output_stage2/comparison_20260921_220002/`. The `puct/` subfolder includes PNG and MP4.
This small comparison measures the combined PUCT configuration, including its
broader prior and composition constraints; it does not isolate the contribution
of PUCT alone or establish performance across every composition.
