# Read species from InChI

`species_reader.py` defines the `SpeciesReadOut` class. Importing it does not
create files. `run_species_readout.py` defines the editable parameters, creates
one object, and calls the two export methods directly, without a main guard.

## Run

From this directory, in your chosen Python environment:

```sh
python -m pip install -r requirements.txt
python run_species_readout.py
```

Paths in the runner are relative to the working directory. Edit them freely.
Rerunning replaces the two named output files.
The converter reads local files only; it does not require a URL or internet access.

1. **`FFCM2_with_SMILES.xlsx`**: all 96 input rows, in source order, with exactly
   four columns: `Species`, `Chemical Name`, `InChI`, and `SMILES`.
2. **`FFCM2_species_reference.xlsx`**: a workbook similar to the existing FFCMII
   species reference. `Species reference` contains a diagram for each species,
   indexed base atoms, base bonds, attached H counts, chemical names, and notes.
   `Graph definitions` contains
   compact JSON arrays, charges, SMILES, source and normalized InChI, electronic
   state annotations.

The CSV is the only input and is read only. The final requested
outputs are Excel workbooks; no additional CSV exports are produced.

## Conversion details

- `Chem.MolFromInchi` reads the structure; `Chem.MolToSmiles` writes canonical
  isomeric SMILES. Every input row must convert successfully before either
  workbook is written. Invalid rows and duplicate species keys raise errors.
- The original InChI column is preserved exactly. Recognized `EXCITED_` and
  `SINGLET_` annotations are removed only from the string passed to RDKit.
- Electronic states are retained separately. OH/OH*, CH/CH*, and CH2/CH2(S)
  can share SMILES and connectivity while remaining distinct species rows.
  RDKit's radical-electron assignments do **not** establish their physical
  electronic state or spin multiplicity. This also matters for O2 and carbenes.
- `Chem.AddHs` creates explicit H atoms. Non-H base atoms come first, then H.
  For H and H2, the H atoms themselves are base atoms and attached-H counts are
  `[]`, matching the existing reference's convention.
- Diagrams and JSON arrays share zero-based atom indices. Atom numbering may
  differ from the old workbook while describing the same connectivity.
- Every atom, including H, has a labeled circle. Light fills distinguish
  elements, and parallel lines indicate double and triple bonds.
- Formal charges are preserved, including `[C-]#[O+]` for CO. The graph sheet
  also records RDKit's assigned radical-electron counts, labeled as such.

RDKit API: https://www.rdkit.org/docs/source/rdkit.Chem.inchi.html
