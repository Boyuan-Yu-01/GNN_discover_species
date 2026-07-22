"""Create a cleaned JSON dataset from the two JSON files in this folder.

The script keeps only these bond types:
    C-C, C=C, C#C, O-O, O=O, C-O, C=O, C#O, C-H, O-H, H-H

It also removes any degeneracy whose retained bond counts are all zero.
"""

from __future__ import annotations

import json
from pathlib import Path


TARGET_BONDS = [
    "C-C",
    "C=C",
    "C#C",
    "O-O",
    "O=O",
    "C-O",
    "C=O",
    "C#O",
    "C-H",
    "O-H",
    "H-H"
]


def load_json(path: Path) -> dict:
    """Load one JSON file."""
    with path.open("r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def clean_dataset(data: dict) -> dict:
    """Keep only the fields needed for each degeneracy."""
    cleaned = {}

    for degeneracy_id, entry in data.get("formula_degeneracies", {}).items():
        bond_counts = entry.get("bond_counts", {})

        # Skip degeneracies that do not have any of the target bonds.
        if not any(bond_counts.get(bond, 0) > 0 for bond in TARGET_BONDS):
            continue

        # Keep only the requested bond counts.
        filtered_bond_counts = {
            bond: bond_counts.get(bond, 0)
            for bond in TARGET_BONDS
        }

        cleaned[degeneracy_id] = {
            "bond_counts": filtered_bond_counts,
            "is_ring": entry.get("is_ring", False),
        }

    return {"formula_degeneracies": cleaned}


base_dir = Path(__file__).resolve().parent
input_outputs = [
    (base_dir / "master_dataset.json", base_dir / "filtered_master.json"),
    (base_dir / "pseudo_negative_dataset.json", base_dir / "filtered_pseudo_negative.json"),
]

for input_file, output_file in input_outputs:
    data = load_json(input_file)
    cleaned_data = clean_dataset(data)

    with output_file.open("w", encoding="utf-8") as file_handle:
        json.dump(cleaned_data, file_handle, indent=2)
        file_handle.write("\n")

    print(f"Wrote {output_file}")