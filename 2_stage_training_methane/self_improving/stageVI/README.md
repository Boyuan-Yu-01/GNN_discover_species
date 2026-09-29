# Stage VI: compare both Stage V predictions with FFCM2

- Run after Stage V, from `self_improving/stageVI/`: `python compare_stage6.py`.
- Reads `../stageV/output/puct/` and `../stageV/output/direct/`.
- Checks that both campaigns recorded the same checkpoint and known-reference hashes.
- Reuses Stage III graph matching against `../stageIII/FFCM2_CHO_reference.xlsx`.
- Matches atom types and bond orders. Charges are checked only when predicted;
  connectivity matching does not resolve electronic states or prove chemical stability.
- Saves separate results in `output/puct/` and `output/direct/`:
  - `ffcmii_matches.json`: validated, unvalidated and given sets.
  - `summary.json` and `config.json`.
  - `comparison.csv`, `comparison.log` and one full `final_structures.png`.
- Plot order and colors:
  - Green: FFCM2 matches outside the expanded known set.
  - Red: unvalidated structures outside FFCM2 and the known set.
  - Blue: recovered members of the expanded known set used in Stage V.
- Thus “given” now refers to Stage IV's expanded reference, not just Stage I's
  original 31 species. Do not compare green counts across rounds without accounting
  for that changed reference set.
- No page images or extra classification JSON files are produced.
- Settings are below the class; paths stay relative to the working directory.
  Reruns replace these outputs, preserving Stage III's earlier comparisons.
- This stage classifies results only; it does not update training datasets or weights.
- Verification: both temporary campaigns classified successfully; mismatched
  checkpoint hashes were rejected and the input model remained unchanged.

## Molecular drawing

- The shared Stage III renderer uses RDKit 2D coordinates to make bond endpoints
  clear. This fixes the misleading NC4H5 drawing without changing its graph or
  classification. RDKit is required for rendering; radicals are not inferred.
