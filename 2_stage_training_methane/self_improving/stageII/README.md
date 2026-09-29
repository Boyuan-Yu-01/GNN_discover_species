# PUCT exploration in self_improving

This folder contains the new PUCT (Monte Carlo tree search) implementation.
The original sibling `../../stageI/` and `../../stageII/` Python files are unchanged.

## Run

From `2_stage_training_methane/self_improving/stageII`:

```sh
conda activate GNN
python explore_stage2.py
```

The default mode lets PUCT choose how many atoms to add and when to STOP:

```python
GENERATION_MODE = "free"
EXPLORATION_RUNS = int(os.environ.get("STAGE2_RUNS", "100"))
ATTEMPTS_PER_RUN = int(os.environ.get("STAGE2_ATTEMPTS", "1000"))
```

Exploration stops after the configured number of attempts:

- Free mode: `EXPLORATION_RUNS × ATTEMPTS_PER_RUN` (default **100,000**).
- Formula mode: `EXPLORATION_RUNS × SAMPLES_PER_COMPOSITION × number of compositions`.

Recovery of the given species does not change the stopping point. There is no
full-recovery setting. Each run uses a different seed and independent PUCT trees.
The per-molecule action limit still applies to each construction attempt.

For a shorter exploration:

```sh
STAGE2_RUNS=1 STAGE2_ATTEMPTS=100 python explore_stage2.py
```

To request exact formulas, set `GENERATION_MODE = "formula"`.
Then `COMPOSITIONS` and `SAMPLES_PER_COMPOSITION` apply; the existing defaults
produce 100 × 29 × 100 = 290,000 attempts. The composition count comes from
the configured reference file unless `COMPOSITIONS` is explicitly supplied.

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
- Each run uses one tree, which persists across its attempts. The next run
  starts an independent tree with a different seed.
- A stopped graph is matched against all compatible reference graphs.
- `requested_C_H_O` is null; `generated_C_H_O` records the actual composition.
- Deduplication uses actual composition plus exact graph isomorphism.
- `frequency_overall` divides occurrences by all attempts in the run/campaign.
  The formula-specific frequency is null. PNG/MP4 labels use actual counts and
  overall frequencies.
- Summaries use `per_generated_composition`. No attempt fails merely because
  its formula differs from a target.

To use the previous sampling algorithm, run:

```sh
STAGE2_METHOD=sampling STAGE2_RUNS=1 python explore_stage2.py
```

Classes appear first, followed by editable parameters and direct calls.
There is no main guard. Paths are configurable and relative to the working
directory. Importing the exploration script directly runs its campaign.

The default checkpoint and active reference JSON are read from the local copy in
`../stageI` (inside `self_improving`).
The checkpoint path is `../stageI/output_stage1/trained_growth_gnn.pt`.
Stage I replaces this file on each run, so no path change or file transfer is
needed after retraining with the default output folder. If you use a custom
Stage I output folder, update `CHECKPOINT_PATH` in `explore_stage2.py`.
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

### Exploration count

Each run completes exactly `ATTEMPTS_PER_RUN` attempts in free mode, or
`SAMPLES_PER_COMPOSITION` attempts for each requested composition in formula
mode. The campaign completes `EXPLORATION_RUNS` independent runs. Failed
constructions and repeated structures also count as attempts.

Given species and new candidates are classified and recorded as before.
Recovering all given species neither stops exploration early nor extends it.
The network stays frozen throughout exploration.

The tree indexes construction histories. Equivalent partial graphs reached by
different histories are not merged in the tree. Final outputs are deduplicated
using atom- and bond-aware graph isomorphism, as in the original Stage II.

## Reward and constraints

| Outcome | Default reward |
|---|---:|
| STOP, exact requested composition, absent from active references | 1.0 |
| STOP, exact requested composition, matches an active reference | 0.5 |
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
| PUCT_C | 2.0 | Exploration strength |
| PUCT_UNIFORM_FRACTION | 0.25 | Prior mass reserved for uniform exploration |
| PUCT_REQUIRE_COMPOSITION | True only in formula mode | Apply exact-formula constraints |
| PUCT_REFERENCE_REWARD | 0.5 | Known, complete graph reward |
| PUCT_CANDIDATE_REWARD | 1.0 | Unknown, complete graph reward |
| SAMPLES_PER_COMPOSITION | 100 | Attempts per composition per run in formula mode |
| EXPLORATION_RUNS | 100 | Number of independent runs |
| TEMPERATURE | 1.0 | Temperature before prior mixing |
| COMPOSITIONS | None | All unique active reference compositions |
| SAVE_MEDIA | True | Save final PNG and growth MP4 |

Free growth can explore new compositions without listing them. In formula mode, explicitly list
them, e.g. `COMPOSITIONS = [(2, 4, 1), (2, 6, 1)]`. Counts always use C,H,O order
and must fit the checkpoint's atom limit.

Increasing simulations per run lets each tree accumulate more evidence.
Increasing the number of runs restarts the tree more often.
`BATCH_SIZE` controls batching for ordinary sampling; PUCT is sequential so each
simulation can use the preceding simulation's reward.

Programmatic configurations default to sampling. The script's editable
parameters select PUCT. Both methods stop at the configured attempt count.

## Outputs

Results are saved directly under `output/`, relative to the working directory
(normally `self_improving/stageII/output`). Every execution replaces the previous
exploration results, including the old logs and `runs/` folder. Stale PNG/MP4
and PUCT statistics are removed even when the new run disables those outputs.
There is no timestamped parent folder. `STAGE2_OUTPUT_FOLDER` can override the path.

The campaign saves the original compact JSON files, console/file logs,
`final_structures.png`, and `exploration_growth.mp4` directly to `output/`.
Per-run files are saved under `runs/run_001/`, etc. Orange panels are unvalidated unknown candidates.

JSON outputs use Stage I's shared `CompactJSON` writer with its packed layout:
short fields and atom/bond records share lines up to a target width of 120
characters, numeric arrays stay on one line, and larger records remain indented.
This applies to campaign and per-run reports without changing their stored data.

Additional PUCT diagnostics:

- `summary.json`: actual attempt counts, reference and candidate counts,
  per-run summaries, and `termination_reason = "attempt_budget_complete"`.
  The former `recovery_progress.json` is no longer produced and is cleared
  when starting a new exploration.

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

The earlier test suite checked rare-prior exploration and
reward backup, composition constraints, dead ends, parameter validation,
repeatability, trace replay, unchanged checkpoint bytes, campaign merging,
full-coverage stopping beyond nominal budgets, isomer-specific recovery, and
rejection of unreachable target settings, probability averaging, free STOP and
reward behavior, and reproducible free-growth campaign outputs. That test script
has since been removed. A separate earlier real-checkpoint validation
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

## Direct network generation

- Run `python stageII_direct_generate.py` from this directory.
- The separate class-based runner reuses this Stage II implementation with
  `search_method="sampling"`, temperature 1.0 and no uniform exploration mixture.
  It does not run PUCT or update network weights.
- Default budget: 100 runs × 1,000 attempts, using the same Stage I checkpoint,
  reference and seed schedule as the PUCT runner. Parameters are below the class.
- In free mode, probabilities are averaged over the reference composition contexts
  at every step. The network samples legal actions and STOP without a fixed formula.
- The direct runner reuses graph features across contexts, as in Stage IV. Its
  probabilities and legal masks were checked against the original Stage II
  implementation on 100 partial states; differences stayed within numerical tolerance.
- Outputs go to `output_direct/`: compact JSON inventories and traces, per-run
  results, `final_structures.png`, and `exploration_growth.mp4`. The existing PUCT
  `output/` is separate. Rerunning replaces generated files in `output_direct/`.
- `DIRECT_RUNS`, `DIRECT_ATTEMPTS` and `DIRECT_OUTPUT_FOLDER` optionally override
  those settings through environment variables.
- To use Stage IV, change both the checkpoint path to its selected model and the
  reference path to the expanded species JSON. Stage III can process this output
  by pointing its prediction input at `output_direct/unique_structures.json`.
- Unknown candidates still need Stage III classification; direct sampling alone
  does not label them validated or invalid. Inference uses CPU, matching Stage II.

### Direct-generation result (100,000 attempts)

- The executed campaign recovered all 31 reference structures and produced 233
  distinct unknown candidates (264 distinct structures in total).
- Attempts: 74,773 reference matches and 25,227 unknown candidates; no failed
  attempts were recorded. Repeated molecules count toward these attempt totals.
- These results use the Stage I checkpoint and original reference set. Unknown
  candidates have not yet been classified against FFCM2 by Stage III.
