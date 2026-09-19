# Stage II: explore the trained network

Stage II currently **samples a fixed stage I network**. It does not retrain the
network or run reinforcement learning. CO remains excluded.

## Run

From this folder:

```sh
conda activate GNN
python test_stage2.py
python explore_stage2.py
```

All classes precede editable parameters and direct execution calls. There is no
main guard. Paths are configurable and relative to the current working directory.
Importing explore_stage2.py directly starts exploration.

## Parameters

- `CHECKPOINT_PATH`: stage I checkpoint to explore. Defaults to the existing
  300-epoch run `../stageI/output_stage1/20260916_105526_866421/trained_growth_gnn.pt`.
- `REFERENCE_JSON_PATH`: reference graphs used for matching.
- `COMPOSITIONS = None`: all 29 distinct reference compositions. Alternatively,
  use a list such as `[(2, 4, 1), (2, 6, 1)]`, always in **C,H,O** order.
  Nearby compositions may be supplied, but their total atom count must fit the
  trained limit. Their performance is unmeasured until explored.
- `EXPLORATION_RUNS = 10`: number of explorations executed by one script call.
- `SAMPLES_PER_COMPOSITION = 100`: attempts for each formula **per run**.
- `TEMPERATURE = 1.0`: samples the learned action distribution. Smaller positive
  values concentrate probability; larger values spread it across more actions.
- `SEED`: makes an unchanged configuration reproducible. Batch size also affects
  the order of random draws, so keep it fixed when reproducing a run.
- `OUTPUT_FOLDER`: new output folder; existing folders are never overwritten.
  Optional overrides: `STAGE2_OUTPUT_FOLDER`, `STAGE2_SAMPLES`, and `STAGE2_RUNS`.

## How it works

1. Load only imports and class definitions from the configured stage I source,
   using the same AST approach as its tests. No stage I training calls execute.
2. Restore architecture, atom encoding, valence limits, action ordering and weights
   from the checkpoint. Check action ordering and load weights strictly.
3. Freeze the network, request a composition, and start from an empty graph.
4. Sample START/GROW/CONNECT/STOP actions using the stage I chemistry masks.
   The requested composition is an input, not an extra hard action constraint.
5. Stop on STOP or the checkpoint's step limit, and record every action and its
   sampling probability.
6. Classify each attempt:
   - `failed_step_limit`: no STOP, even if the graph happens to match.
   - `failed_composition`: stopped, but not the requested C,H,O counts.
   - `reference_match`: stopped with the requested composition and known structure.
   - `new_candidate`: stopped with the requested composition, absent from the
     active reference set.
7. Deduplicate successful requests using full atom- and bond-aware graph
   isomorphism. Atom numbering and construction order do not create new species;
   distinct isomers remain separate.

An unknown graph is a **candidate for inspection**, not a demonstrated stable,
important, or globally novel molecule. Masks limit bond-order sums but do not
determine electronic states or chemical stability. Unused valence is recorded as
a diagnostic; it is not a confirmed radical assignment. Frequency is model sampling
frequency, not predicted abundance. No physical reaction pathway is inferred.

## Saved outputs

Each timestamped folder under `output_stage2/` contains compact, readable JSON:

| File | Contents |
|---|---|
| final_structures.png | All unique successful structures, with candidate/reference colors and frequencies |
| exploration_growth.mp4 | Growth replay of the first successful trace for each distinct structure |
| exploration.log | Console messages saved to disk |
| config.json | Settings, resolved compositions, checkpoint/reference/code hashes |
| summary.json | Counts per requested composition and overall |
| attempts.json | Every attempt, including failures, graphs and action traces |
| unique_structures.json | Distinct successful structures and occurrence counts |
| reference_matches.json | Distinct reference matches |
| new_candidates.json | Distinct structures absent from the active reference set |

Structures include species key, base atom indices, base bonds, attached H per
base atom, unused valence, and reference membership. Pure H species retain H as
base atoms, following stage I. The first successful trace represents each unique
structure; all other traces remain in attempts.json. Frequencies use all requested
attempts for that composition as the denominator, including failed attempts.

## Tests

Seven tests cover loading without training, termination/formula classification,
composition validation, isomer-preserving deduplication, attached H reporting,
and a repeated end-to-end run. The integration test checks saved counts, replays
every action trace, verifies reproducibility, and confirms the checkpoint file is
unchanged.

## First exploration run

Saved under `output_stage2/exploration_II/`: 2,900 attempts across 29 reference compositions, 100 per composition, temperature 1.0, seed 24680. Recovered all 31 reference structures and found 9 distinct candidate structures absent from the active references. Candidates need chemical assessment.

Attempt counts: failed_composition=1313, reference_match=1576, new_candidate=11.

## Plots and animation

`SAVE_MEDIA = True` saves both PNG and MP4 alongside the JSON outputs.
The PNG shows all distinct successful structures, with new candidates first.
Each panel includes its JSON structure ID, composition, occurrence count and
sampling frequency. C is grey, H is blue and O is red; multiple strokes show
double/triple bonds. Orange panels indicate unvalidated candidates; teal panels
indicate reference matches.

The MP4 shows six structures per page and replays their saved action sequences
from empty graph to STOP. Positions are fixed using each final graph layout, so
atoms do not move as other atoms are added. It represents one successful growth
trace per distinct structure; failed attempts remain available in attempts.json.
`VIDEO_FPS = 2` controls playback speed and `PLOT_DPI = 150` controls PNG resolution.
MP4 encoding requires ffmpeg; rendering reports an error if it is unavailable.

The requested earlier first exploration folder was deleted. Its replacement,
including JSON, MP4 and PNG, is `output_stage2/exploration_II/`.

## Multiple explorations in one script

Running `python explore_stage2.py` now calls `ExplorationCampaign`.
It runs `EXPLORATION_RUNS` explorations with seeds
`SEED + run_index * 1000` (zero-based run index).
The default 10 runs × 29 compositions × 100 samples gives **29,000 attempts**.
Network weights, temperature, and requested compositions stay the same.

Individual run logs and JSON files are saved under `runs/run_001/`, etc.
The campaign folder contains the combined JSON inventory, one final PNG and one
MP4. Deduplication spans every run, so an identical structure appears only once
in the combined plots. Candidate/reference colors and growth-trace rendering
follow the single-run format.

Combined occurrence counts include all runs. Frequencies use all attempts for
that requested composition across the campaign, including failed attempts.
`runs_observed` lists the runs in which each structure appeared. Combined
attempts contain their run ID, original attempt number, global attempt number,
and global structure ID. The animation replays the first observed successful
trace for each unique structure, not every repeated attempt.

The campaign test verifies cross-run deduplication, accumulated counts, global
IDs, distinct seeds, and combined frequency denominators.

## Ten-run campaign results

Saved in `output_stage2/exploration_III/`: 29,000 attempts, 31 distinct reference structures and 19 distinct unvalidated candidates. The combined PNG shows 50 structures; the MP4 is 55 seconds long (1440 × 900). All seven tests pass.
