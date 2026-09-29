"""Run from species_readOut: python run_species_readout.py."""
from species_reader import SpeciesReadOut


# PARAMETERS — paths are relative to your working directory and freely editable.
INPUT_CSV = "FFCM2_InChI.csv"
SMILES_WORKBOOK = "FFCM2_with_SMILES.xlsx"
SPECIES_REFERENCE_WORKBOOK = "FFCM2_species_reference.xlsx"


# 1. Create one reader object; parse and convert each species once.
species = SpeciesReadOut(INPUT_CSV)

# 2. Save the original three columns with one additional SMILES column.
species.write_smiles_workbook(SMILES_WORKBOOK)

# 3. Save diagrams, base atoms/bonds, attached H, charges, and graph arrays.
species.write_species_reference(SPECIES_REFERENCE_WORKBOOK)
