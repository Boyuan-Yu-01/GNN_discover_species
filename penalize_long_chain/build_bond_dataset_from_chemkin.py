from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from identify_bonds import identify_bonds_from_chemikin_mechanism
from master_dataset_merger import master_dataset_merger


script_dir = Path(__file__).resolve().parent
mechanism_dir = script_dir / "mechanism_file_chemkin"
output_dir = script_dir / "output"
output_dir.mkdir(exist_ok=True)

json_paths = []
for mechanism_file in sorted(mechanism_dir.glob("*.txt")):
    reader = identify_bonds_from_chemikin_mechanism(mechanism_file)
    backbone = reader.read_species_backbone()

    json_path = output_dir / f"{mechanism_file.stem}_subgraph_info.json"
    log_path = output_dir / f"log_{mechanism_file.stem}.txt"
    json_path = reader.export_species_backbone_to_json(json_path)

    with redirect_stdout(StringIO()):
        reader.write_species_backbone_log(json_path, log_path)

    json_paths.append(json_path)
    print(
        f"{mechanism_file.name}: "
        f"{len(backbone['species'])} species, "
        f"{len(backbone['formula_degeneracies'])} formula degeneracies"
    )

merger = master_dataset_merger(json_paths)
with redirect_stdout(StringIO()):
    master_json_path, master_log_path = merger.merge_and_export(
        output_file=output_dir / "master_dataset.json",
        log_file=output_dir / "log_master_dataset.txt",
    )
    master_summary_path = merger.export_formula_degeneracy_summary_to_csv(
        output_dir / "master_dataset_summary.csv"
    )

print(
    "master_dataset.json: "
    f"{len(merger.master_dataset['formula_degeneracies'])} formula degeneracies"
)
print(f"log_master_dataset.txt: {master_log_path}")
print(f"master_dataset_summary.csv: {master_summary_path}")
