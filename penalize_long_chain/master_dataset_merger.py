from collections import defaultdict
import json
from pathlib import Path
import re

from identify_bonds import identify_bonds_from_chemikin_mechanism


class master_dataset_merger:
    """Merge formula-degeneracy dictionaries from multiple species JSON files."""

    BOND_KEYS = identify_bonds_from_chemikin_mechanism.BOND_KEYS

    def __init__(self, json_files=None):
        """Store input JSON paths and initialize merge/log containers."""
        self.json_files = [Path(path).expanduser().resolve() for path in (json_files or [])]
        self.master_dataset = {}
        self.merge_log = {}

    def merge_formula_degeneracies(self):
        """Merge input formula_degeneracies into one clash-safe master dataset.

        Source degeneracy IDs such as ``C5H10O_001`` are local to each input
        file. This method therefore rebuilds master IDs from each entry's true
        signature: formula, bond counts, and ring flag.
        """
        signature_records = defaultdict(
            lambda: {
                "formula": None,
                "bond_counts": None,
                "is_ring": None,
                "species": set(),
                "source_entries": [],
            }
        )
        input_summaries = []
        invalid_entries = []
        source_id_signatures = defaultdict(set)
        species_signatures = defaultdict(set)

        for json_file in self.json_files:
            data = self._read_json(json_file)
            formula_degeneracies = data.get("formula_degeneracies", {})
            input_summaries.append(
                {
                    "file": str(json_file),
                    "formula_degeneracy_count": len(formula_degeneracies),
                }
            )

            for source_id, entry in formula_degeneracies.items():
                validation_error = self._validate_formula_degeneracy_entry(
                    source_id,
                    entry,
                    json_file,
                )
                if validation_error:
                    invalid_entries.append(validation_error)
                    continue

                formula = entry["formula"]
                bond_counts = self._normalized_bond_counts(entry["bond_counts"])
                is_ring = bool(entry["is_ring"])
                signature = self._signature(formula, bond_counts, is_ring)
                species = sorted(set(entry.get("species", [])), key=str.lower)

                record = signature_records[signature]
                record["formula"] = formula
                record["bond_counts"] = bond_counts
                record["is_ring"] = is_ring
                record["species"].update(species)
                record["source_entries"].append(
                    {
                        "file": str(json_file),
                        "source_degeneracy_id": source_id,
                        "species_count": len(species),
                    }
                )

                source_id_signatures[source_id].add(signature)
                for species_name in species:
                    species_signatures[species_name].add(signature)

        self.master_dataset = {
            "formula_degeneracies": self._build_master_formula_degeneracies(
                signature_records
            )
        }
        self.merge_log = self._build_merge_log(
            input_summaries,
            signature_records,
            invalid_entries,
            source_id_signatures,
            species_signatures,
        )
        return self.master_dataset

    def export_master_dataset_to_json(self, output_file, indent=2):
        """Write the merged master dataset to JSON."""
        if not self.master_dataset:
            self.merge_formula_degeneracies()

        output_path = Path(output_file).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as file_handle:
            json.dump(
                self.master_dataset,
                file_handle,
                indent=indent,
                sort_keys=True,
            )
            file_handle.write("\n")
        return output_path

    def write_log(self, log_file, master_json_path=None):
        """Write a human-readable log describing the merge and any clashes."""
        if not self.merge_log:
            self.merge_formula_degeneracies()

        log_path = Path(log_file).expanduser().resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)

        lines = self._log_lines(master_json_path, log_path)
        log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("\n".join(lines))
        return log_path

    def merge_and_export(self, output_file, log_file, indent=2):
        """Run the full merge, write the JSON, and write the log."""
        self.merge_formula_degeneracies()
        output_path = self.export_master_dataset_to_json(output_file, indent=indent)
        log_path = self.write_log(log_file, master_json_path=output_path)
        return output_path, log_path

    @staticmethod
    def _read_json(json_file):
        """Read one JSON file and return its parsed dictionary."""
        with Path(json_file).open("r", encoding="utf-8") as file_handle:
            return json.load(file_handle)

    @classmethod
    def _normalized_bond_counts(cls, bond_counts):
        """Return a full, stable bond-count dictionary."""
        normalized = {bond: 0 for bond in cls.BOND_KEYS}
        for bond, count in bond_counts.items():
            normalized[bond] = count
        return normalized

    @staticmethod
    def _signature(formula, bond_counts, is_ring):
        """Create a hashable chemistry signature for one formula degeneracy."""
        return (
            formula,
            tuple(sorted(bond_counts.items())),
            is_ring,
        )

    @staticmethod
    def _validate_formula_degeneracy_entry(source_id, entry, json_file):
        """Return an error dictionary if a source entry is not mergeable."""
        required_keys = {"formula", "bond_counts", "is_ring", "species"}
        missing_keys = sorted(required_keys - set(entry))
        if missing_keys:
            return {
                "file": str(json_file),
                "source_degeneracy_id": source_id,
                "reason": f"missing required keys: {missing_keys}",
            }
        if not isinstance(entry["bond_counts"], dict):
            return {
                "file": str(json_file),
                "source_degeneracy_id": source_id,
                "reason": "bond_counts is not a dictionary",
            }
        if not isinstance(entry["species"], list):
            return {
                "file": str(json_file),
                "source_degeneracy_id": source_id,
                "reason": "species is not a list",
            }
        return None

    def _build_master_formula_degeneracies(self, signature_records):
        """Build new master degeneracy IDs after all signatures are known."""
        records_by_formula = defaultdict(list)
        for signature, record in signature_records.items():
            records_by_formula[record["formula"]].append((signature, record))

        master_formula_degeneracies = {}
        for formula in sorted(records_by_formula):
            records = sorted(
                records_by_formula[formula],
                key=lambda item: (
                    item[1]["is_ring"],
                    tuple(item[1]["bond_counts"][bond] for bond in self.BOND_KEYS),
                    item[0],
                ),
            )
            for degeneracy_number, (_, record) in enumerate(records, start=1):
                degeneracy_id = self._formula_degeneracy_id(
                    formula,
                    degeneracy_number,
                )
                master_formula_degeneracies[degeneracy_id] = {
                    "formula": formula,
                    "degeneracy_id": degeneracy_id,
                    "bond_counts": record["bond_counts"],
                    "is_ring": record["is_ring"],
                    "species": sorted(record["species"], key=str.lower),
                }

        return master_formula_degeneracies

    @staticmethod
    def _formula_degeneracy_id(formula, degeneracy_number):
        """Create stable master IDs such as C5H10O_001."""
        formula_label = re.sub(r"[^A-Za-z0-9]+", "_", formula or "unknown").strip("_")
        formula_label = formula_label or "unknown"
        return f"{formula_label}_{degeneracy_number:03d}"

    def _build_merge_log(
        self,
        input_summaries,
        signature_records,
        invalid_entries,
        source_id_signatures,
        species_signatures,
    ):
        """Collect merge statistics and clash summaries for the text log."""
        input_degeneracy_count = sum(
            summary["formula_degeneracy_count"] for summary in input_summaries
        )
        formula_to_signature_count = defaultdict(int)
        for signature, record in signature_records.items():
            formula_to_signature_count[record["formula"]] += 1

        source_id_clashes = {
            source_id: len(signatures)
            for source_id, signatures in source_id_signatures.items()
            if len(signatures) > 1
        }
        species_signature_clashes = {
            species: len(signatures)
            for species, signatures in species_signatures.items()
            if len(signatures) > 1
        }

        return {
            "input_files": input_summaries,
            "input_formula_degeneracy_count": input_degeneracy_count,
            "master_formula_degeneracy_count": len(signature_records),
            "merged_duplicate_signature_count": (
                input_degeneracy_count - len(signature_records)
            ),
            "formula_count": len(formula_to_signature_count),
            "ambiguous_formula_count": sum(
                count > 1 for count in formula_to_signature_count.values()
            ),
            "source_id_clashes": source_id_clashes,
            "species_signature_clashes": species_signature_clashes,
            "invalid_entries": invalid_entries,
            "top_ambiguous_formulae": sorted(
                (
                    (formula, count)
                    for formula, count in formula_to_signature_count.items()
                    if count > 1
                ),
                key=lambda item: (-item[1], item[0]),
            )[:20],
        }

    def _log_lines(self, master_json_path, log_path):
        """Format merge statistics and clash summaries as readable lines."""
        lines = [
            "Master dataset merge log",
            f"Master JSON output: {master_json_path}",
            f"Log output: {log_path}",
            f"Input files: {len(self.merge_log['input_files'])}",
        ]
        for summary in self.merge_log["input_files"]:
            lines.append(
                "  "
                f"{summary['file']}: "
                f"{summary['formula_degeneracy_count']} formula degeneracies"
            )

        lines.extend(
            [
                f"Input formula degeneracies: {self.merge_log['input_formula_degeneracy_count']}",
                f"Master formula degeneracies: {self.merge_log['master_formula_degeneracy_count']}",
                f"Merged duplicate signatures: {self.merge_log['merged_duplicate_signature_count']}",
                f"Formulae represented: {self.merge_log['formula_count']}",
                f"Ambiguous formulae in master: {self.merge_log['ambiguous_formula_count']}",
                f"Invalid entries skipped: {len(self.merge_log['invalid_entries'])}",
                f"Source degeneracy ID clashes: {len(self.merge_log['source_id_clashes'])}",
                f"Species signature clashes: {len(self.merge_log['species_signature_clashes'])}",
                "Top ambiguous formulae:",
            ]
        )

        for formula, count in self.merge_log["top_ambiguous_formulae"]:
            lines.append(f"  {formula}: {count} master degeneracies")

        if self.merge_log["source_id_clashes"]:
            lines.append("Source degeneracy ID clash examples:")
            for source_id, count in list(self.merge_log["source_id_clashes"].items())[:20]:
                lines.append(f"  {source_id}: {count} different signatures")

        if self.merge_log["species_signature_clashes"]:
            lines.append("Species signature clash examples:")
            for species, count in list(self.merge_log["species_signature_clashes"].items())[:20]:
                lines.append(f"  {species}: {count} different signatures")

        if self.merge_log["invalid_entries"]:
            lines.append("Invalid entry examples:")
            for entry in self.merge_log["invalid_entries"][:20]:
                lines.append(
                    "  "
                    f"{entry['file']}::{entry['source_degeneracy_id']}: "
                    f"{entry['reason']}"
                )

        return lines


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    output_dir = script_dir / "output"
    input_json_files = sorted(output_dir.glob("*_subgraph_info.json"))

    merger = master_dataset_merger(input_json_files)
    merger.merge_and_export(
        output_file=output_dir / "master_dataset.json",
        log_file=output_dir / "log_master_dataset.txt",
    )
