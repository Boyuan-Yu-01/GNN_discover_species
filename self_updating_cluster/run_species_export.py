"""Export the FFCM2 species table using SpeciesExporter.

Run from self_updating in the GNN environment:
    python run_species_export.py

All 96 source entries are defined below, so this example runs offline.
The source table is retained as published, including alternative names for the
same connectivity and the CH2(S), OH*, and CH* electronic-state annotations.
"""

from codebase.species_exporter import SpeciesExporter


# Source: https://web.stanford.edu/group/haiwanglab/FFCM2/docs/TrialModel/Species/
# Table checked on 2026-09-27. Edit the entries or output path for your own data.
# This full reference includes HE, AR and N2; it is not restricted to CHO.
OUTPUT_PATH = "test_generator.xlsx"

# Each entry needs a species key and either an InChI or a SMILES string.
# Chemical names are optional. Here all structures come from the table's InChI.
SPECIES = [
    {"species_key":"H","chemical_name":"Hydrogen atom","inchi":"InChI=1S/H"},
    {"species_key":"H2","chemical_name":"Hydrogen","inchi":"InChI=1S/H2/h1H"},
    {"species_key":"O","chemical_name":"Oxygen atom","inchi":"InChI=1S/O"},
    {"species_key":"O2","chemical_name":"Oxygen","inchi":"InChI=1S/O2/c1-2"},
    {"species_key":"OH","chemical_name":"Hydroxyl radical","inchi":"InChI=1S/HO/h1H"},
    {"species_key":"H2O","chemical_name":"Water","inchi":"InChI=1S/H2O/h1H2"},
    {"species_key":"HO2","chemical_name":"Hydroperoxy radical","inchi":"InChI=1S/HO2/c1-2/h1H"},
    {"species_key":"H2O2","chemical_name":"Hydrogen peroxide","inchi":"InChI=1S/H2O2/c1-2/h1-2H"},
    {"species_key":"HE","chemical_name":"Helium","inchi":"InChI=1S/He"},
    {"species_key":"AR","chemical_name":"Argon","inchi":"InChI=1S/Ar"},
    {"species_key":"N2","chemical_name":"Nitrogen","inchi":"InChI=1S/N2/c1-2"},
    {"species_key":"C","chemical_name":"Carbon atom","inchi":"InChI=1S/C"},
    {"species_key":"CH","chemical_name":"Methylidyne","inchi":"InChI=1S/CH/h1H"},
    {"species_key":"CH2","chemical_name":"Methylene radical triplet","inchi":"InChI=1S/CH2/h1H2"},
    {"species_key":"CH2(S)","chemical_name":"Methylene radical singlet","inchi":"SINGLET_InChI=1S/CH2/h1H2"},
    {"species_key":"CH3","chemical_name":"Methyl radical","inchi":"InChI=1S/CH3/h1H3"},
    {"species_key":"CH4","chemical_name":"Methane","inchi":"InChI=1S/CH4/h1H4"},
    {"species_key":"CO","chemical_name":"Carbon monoxide","inchi":"InChI=1S/CO/c1-2"},
    {"species_key":"CO2","chemical_name":"Carbon dioxide","inchi":"InChI=1S/CO2/c2-1-3"},
    {"species_key":"HCO","chemical_name":"Formyl radical","inchi":"InChI=1S/CHO/c1-2/h1H"},
    {"species_key":"CH2O","chemical_name":"Formaldehyde","inchi":"InChI=1S/CH2O/c1-2/h1H2"},
    {"species_key":"CH2OH","chemical_name":"Hydroxymethyl radical","inchi":"InChI=1S/CH3O/c1-2/h2H,1H2"},
    {"species_key":"CH3O","chemical_name":"Methoxy radical","inchi":"InChI=1S/CH3O/c1-2/h1H3"},
    {"species_key":"CH3OH","chemical_name":"Methanol","inchi":"InChI=1S/CH4O/c1-2/h2H,1H3"},
    {"species_key":"C2H","chemical_name":"Ethynyl radical","inchi":"InChI=1S/C2H/c1-2/h1H"},
    {"species_key":"C2H2","chemical_name":"Acetylene","inchi":"InChI=1S/C2H2/c1-2/h1-2H"},
    {"species_key":"C2H3","chemical_name":"Vinyl radical","inchi":"InChI=1S/C2H3/c1-2/h1H,2H2"},
    {"species_key":"C2H4","chemical_name":"Ethylene","inchi":"InChI=1S/C2H4/c1-2/h1-2H2"},
    {"species_key":"C2H5","chemical_name":"Ethyl radical","inchi":"InChI=1S/C2H5/c1-2/h1H2,2H3"},
    {"species_key":"C2H6","chemical_name":"Ethane","inchi":"InChI=1S/C2H6/c1-2/h1-2H3"},
    {"species_key":"HCCO","chemical_name":"Ketenyl radical","inchi":"InChI=1S/C2HO/c1-2-3/h1H"},
    {"species_key":"CH2CO","chemical_name":"Ketene","inchi":"InChI=1S/C2H2O/c1-2-3/h1H2"},
    {"species_key":"CH2CHO","chemical_name":"Vinoxy radical","inchi":"InChI=1S/C2H3O/c1-2-3/h2H,1H2"},
    {"species_key":"CH3CHO","chemical_name":"Acetaldehyde","inchi":"InChI=1S/C2H4O/c1-2-3/h2H,1H3"},
    {"species_key":"CH3CO","chemical_name":"Acetyl radical","inchi":"InChI=1S/C2H3O/c1-2-3/h1H3"},
    {"species_key":"H2CC","chemical_name":"Vinylidene","inchi":"InChI=1S/C2H2/c1-2/h1H2"},
    {"species_key":"CH3O2","chemical_name":"Methyldioxy radical","inchi":"InChI=1S/CH3O2/c1-3-2/h1H3"},
    {"species_key":"CH3OOH","chemical_name":"Methyl hydroperoxide","inchi":"InChI=1S/CH4O2/c1-3-2/h2H,1H3"},
    {"species_key":"C2H5O2","chemical_name":"Ethyldioxy radical","inchi":"InChI=1S/C2H5O2/c1-2-4-3/h2H2,1H3"},
    {"species_key":"C2H5OOH","chemical_name":"Ethyl hydroperoxide","inchi":"InChI=1S/C2H6O2/c1-2-4-3/h3H,2H2,1H3"},
    {"species_key":"C2H2OH","chemical_name":"2-Hydroxyvinyl radical","inchi":"InChI=1S/C2H3O/c1-2-3/h1-3H"},
    {"species_key":"C2H3OH","chemical_name":"Ethenol","inchi":"InChI=1S/C2H4O/c1-2-3/h2-3H,1H2"},
    {"species_key":"C2H4O","chemical_name":"Acetaldehyde","inchi":"InChI=1S/C2H4O/c1-2-3/h2H,1H3"},
    {"species_key":"C2H5OH","chemical_name":"Ethanol","inchi":"InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3"},
    {"species_key":"C2H4OH","chemical_name":"CH2CH2OH","inchi":"InChI=1S/C2H5O/c1-2-3/h3H,1-2H2"},
    {"species_key":"CH3CHOH","chemical_name":"1-Hydroxyethyl radical","inchi":"InChI=1S/C2H5O/c1-2-3/h2-3H,1H3"},
    {"species_key":"C3H8","chemical_name":"Propane","inchi":"InChI=1S/C3H8/c1-3-2/h3H2,1-2H3"},
    {"species_key":"NC3H7","chemical_name":"Propyl radical","inchi":"InChI=1S/C3H7/c1-3-2/h1,3H2,2H3"},
    {"species_key":"IC3H7","chemical_name":"Isopropyl radical","inchi":"InChI=1S/C3H7/c1-3-2/h3H,1-2H3"},
    {"species_key":"C3H6","chemical_name":"Propene","inchi":"InChI=1S/C3H6/c1-3-2/h3H,1H2,2H3"},
    {"species_key":"C3H5","chemical_name":"Allyl radical","inchi":"InChI=1S/C3H5/c1-3-2/h3H,1-2H2"},
    {"species_key":"CH3CCH2","chemical_name":"Isopropenyl radical","inchi":"InChI=1S/C3H5/c1-3-2/h1H2,2H3"},
    {"species_key":"AC3H4","chemical_name":"Allene","inchi":"InChI=1S/C3H4/c1-3-2/h1-2H2"},
    {"species_key":"PC3H4","chemical_name":"Propyne","inchi":"InChI=1S/C3H4/c1-3-2/h1H,2H3"},
    {"species_key":"C3H3","chemical_name":"Propargyl radical","inchi":"InChI=1S/C3H3/c1-3-2/h1H,2H2"},
    {"species_key":"C2H5CHO","chemical_name":"Propanal","inchi":"InChI=1S/C3H6O/c1-2-3-4/h3H,2H2,1H3"},
    {"species_key":"CH3COCH3","chemical_name":"Acetone","inchi":"InChI=1S/C3H6O/c1-3(2)4/h1-2H3"},
    {"species_key":"CH3COCH2","chemical_name":"2-Oxopropyl","inchi":"InChI=1S/C3H5O/c1-3(2)4/h1H2,2H3"},
    {"species_key":"C2H3CHO","chemical_name":"2-Propenal","inchi":"InChI=1S/C3H4O/c1-2-3-4/h2-3H,1H2"},
    {"species_key":"C3H5OH","chemical_name":"2-Propen-1-ol","inchi":"InChI=1S/C3H6O/c1-2-3-4/h2,4H,1,3H2"},
    {"species_key":"NC3H7O2","chemical_name":"Propyldioxy","inchi":"InChI=1S/C3H7O2/c1-2-3-5-4/h2-3H2,1H3"},
    {"species_key":"NC3H7OOH","chemical_name":"Propylhydroperoxide","inchi":"InChI=1S/C3H8O2/c1-2-3-5-4/h4H,2-3H2,1H3"},
    {"species_key":"IC3H7O2","chemical_name":"Isopropylperoxy","inchi":"InChI=1S/C3H7O2/c1-3(2)5-4/h3H,1-2H3"},
    {"species_key":"IC3H7OOH","chemical_name":"2-Hydroperoxypropane","inchi":"InChI=1S/C3H8O2/c1-3(2)5-4/h3-4H,1-2H3"},
    {"species_key":"C4H2","chemical_name":"1,3-Butadiyne","inchi":"InChI=1S/C4H2/c1-3-4-2/h1-2H"},
    {"species_key":"NC4H3","chemical_name":"1-Butene-3-yne-4-yl radical","inchi":"InChI=1S/C4H3/c1-3-4-2/h3H,1H2"},
    {"species_key":"IC4H3","chemical_name":"Butatrienyl radical","inchi":"InChI=1S/C4H3/c1-3-4-2/h1H,2H2"},
    {"species_key":"C4H4","chemical_name":"1-Buten-3-yne","inchi":"InChI=1S/C4H4/c1-3-4-2/h1,4H,2H2"},
    {"species_key":"NC4H5","chemical_name":"1-Butynylradical","inchi":"InChI=1S/C4H5/c1-3-4-2/h3H2,1H3"},
    {"species_key":"IC4H5","chemical_name":"1-Butyn-3-yl radical","inchi":"InChI=1S/C4H5/c1-3-4-2/h1,4H,2H3"},
    {"species_key":"C4H5-2","chemical_name":"But-2-yn-1-yl radical","inchi":"InChI=1S/C4H5/c1-3-4-2/h1H2,2H3"},
    {"species_key":"C4H6","chemical_name":"1,3-Butadiene","inchi":"InChI=1S/C4H6/c1-3-4-2/h3-4H,1-2H2"},
    {"species_key":"C4H612","chemical_name":"1,2-Butadiene","inchi":"InChI=1S/C4H6/c1-3-4-2/h4H,1H2,2H3"},
    {"species_key":"C4H6-2","chemical_name":"2-Butyne","inchi":"InChI=1S/C4H6/c1-3-4-2/h1-2H3"},
    {"species_key":"C4H7","chemical_name":"2-Butenyl radical","inchi":"InChI=1S/C4H7/c1-3-4-2/h3-4H,1H2,2H3"},
    {"species_key":"IC4H7","chemical_name":"2-Methylallyl radical","inchi":"InChI=1S/C4H7/c1-4(2)3/h1-2H2,3H3"},
    {"species_key":"IC4H7-1","chemical_name":"iso-Butyl vinylic radical","inchi":"InChI=1S/C4H7/c1-4(2)3/h1H,2-3H3"},
    {"species_key":"C4H81","chemical_name":"1-Butene","inchi":"InChI=1S/C4H8/c1-3-4-2/h3H,1,4H2,2H3"},
    {"species_key":"C4H82","chemical_name":"2-Butene","inchi":"InChI=1S/C4H8/c1-3-4-2/h3-4H,1-2H3"},
    {"species_key":"IC4H8","chemical_name":"iso-Butene","inchi":"InChI=1S/C4H8/c1-4(2)3/h1H2,2-3H3"},
    {"species_key":"NC4H9","chemical_name":"1-Butyl radical","inchi":"InChI=1S/C4H9/c1-3-4-2/h1,3-4H2,2H3"},
    {"species_key":"SC4H9","chemical_name":"2-Butyl radical","inchi":"InChI=1S/C4H9/c1-3-4-2/h3H,4H2,1-2H3"},
    {"species_key":"IC4H9","chemical_name":"iso-Butyl radical","inchi":"InChI=1S/C4H9/c1-4(2)3/h4H,1H2,2-3H3"},
    {"species_key":"TC4H9","chemical_name":"tert-Butyl radical","inchi":"InChI=1S/C4H9/c1-4(2)3/h1-3H3"},
    {"species_key":"C4H10","chemical_name":"Butane","inchi":"InChI=1S/C4H10/c1-3-4-2/h3-4H2,1-2H3"},
    {"species_key":"IC4H10","chemical_name":"iso-Butane","inchi":"InChI=1S/C4H10/c1-4(2)3/h4H,1-3H3"},
    {"species_key":"H2C4O","chemical_name":"Butatrienone","inchi":"InChI=1S/C4H2O/c1-2-3-4-5/h1H2"},
    {"species_key":"CH2CHCHCHO","chemical_name":"N/A","inchi":"InChI=1S/C4H5O/c1-2-3-4-5/h2-4H,1H2"},
    {"species_key":"CH3CHCHCO","chemical_name":"1-Methyl-3-oxoallyl radical","inchi":"InChI=1S/C4H5O/c1-2-3-4-5/h2-3H,1H3"},
    {"species_key":"CH3CHCHCHO","chemical_name":"2-Butenal","inchi":"InChI=1S/C4H6O/c1-2-3-4-5/h2-4H,1H3"},
    {"species_key":"C3H7CHO","chemical_name":"Butanal","inchi":"InChI=1S/C4H8O/c1-2-3-4-5/h4H,2-3H2,1H3"},
    {"species_key":"IC3H7CHO","chemical_name":"2-Methylpropanal","inchi":"InChI=1S/C4H8O/c1-4(2)3-5/h3-4H,1-2H3"},
    {"species_key":"C2H5COCH3","chemical_name":"Ethyl methyl ketone","inchi":"InChI=1S/C4H8O/c1-3-4(2)5/h3H2,1-2H3"},
    {"species_key":"C2H3COCH3","chemical_name":"Methyl vinyl ketone","inchi":"InChI=1S/C4H6O/c1-3-4(2)5/h3H,1H2,2H3"},
    {"species_key":"OH*","chemical_name":"Hydroxyl radical excited (${\\rm A}^2\\Sigma^+$)","inchi":"EXCITED_InChI=1S/HO/h1H"},
    {"species_key":"CH*","chemical_name":"Methylidyne excited (${\\rm A}^2\\Delta$)","inchi":"EXCITED_InChI=1S/CH/h1H"},
]

# Create the object, then write the workbook. No training is started.
exporter = SpeciesExporter(SPECIES)
exporter.write_xlsx(OUTPUT_PATH)

