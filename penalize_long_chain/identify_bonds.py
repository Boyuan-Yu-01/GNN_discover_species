from collections import defaultdict
import json
from pathlib import Path
import re


# This programme serves to:
#   1. Read a chemical kinetic file to extract the species.
#   2. Estimate bond-count features for each species.
#   3. Store backbone/bond information for further analysis.


class identify_bonds_from_chemikin_mechanism():
    """Read a ChemKin mechanism file and collect species-level information."""

    # Types of bond keys to track
    BOND_KEYS = (   # '-' single bond, '=' double bond, '#' triple bond
        "C-C",
        "C=C",
        "C#C",
        "C-N",
        "C=N",
        "C#N",
        "C-O",
        "C=O",
        "C#O",
        "N-N",
        "N=N",
        "N#N",
        "O-O",
        "O=O",
        "O-N",
        "N=O",
        "C-H",
        "N-H",
        "O-H",
        "H-H",
    )

    # Exact entries for small/common species whose formula alone is ambiguous
    # or whose bond order is important for later feature construction.
    EXACT_SPECIES = {
        "ar": {"formula": "Ar", "bonds": {}, "is_ring": False},
        "he": {"formula": "He", "bonds": {}, "is_ring": False},
        "h": {"formula": "H", "bonds": {}, "is_ring": False},
        "h2": {"formula": "H2", "bonds": {"H-H": 1}, "is_ring": False},
        "o": {"formula": "O", "bonds": {}, "is_ring": False},
        "oh": {"formula": "HO", "bonds": {"O-H": 1}, "is_ring": False},
        "h2o": {"formula": "H2O", "bonds": {"O-H": 2}, "is_ring": False},
        "n2": {"formula": "N2", "bonds": {"N#N": 1}, "is_ring": False},
        "nh": {"formula": "HN", "bonds": {"N-H": 1}, "is_ring": False},
        "nh2": {"formula": "H2N", "bonds": {"N-H": 2}, "is_ring": False},
        "nh3": {"formula": "H3N", "bonds": {"N-H": 3}, "is_ring": False},
        "hno": {"formula": "HNO", "bonds": {"N=O": 1, "N-H": 1}, "is_ring": False},
        "o2": {"formula": "O2", "bonds": {"O=O": 1}, "is_ring": False},
        "ho2": {"formula": "HO2", "bonds": {"O-O": 1, "O-H": 1}, "is_ring": False},
        "h2o2": {"formula": "H2O2", "bonds": {"O-O": 1, "O-H": 2}, "is_ring": False},
        "co": {"formula": "CO", "bonds": {"C#O": 1}, "is_ring": False},
        "co2": {"formula": "CO2", "bonds": {"C=O": 2}, "is_ring": False},
        "hco": {"formula": "CHO", "bonds": {"C=O": 1, "C-H": 1}, "is_ring": False},
        "ch2o": {"formula": "CH2O", "bonds": {"C=O": 1, "C-H": 2}, "is_ring": False},
        "ch3o": {"formula": "CH3O", "bonds": {"C-O": 1, "C-H": 3}, "is_ring": False},
        "ch2oh": {"formula": "CH3O", "bonds": {"C-O": 1, "C-H": 2, "O-H": 1}, "is_ring": False},
        "ch3oh": {"formula": "CH4O", "bonds": {"C-O": 1, "C-H": 3, "O-H": 1}, "is_ring": False},
        "hoch2o": {"formula": "CH3O2", "bonds": {"C-O": 2, "C-H": 2, "O-H": 1}, "is_ring": False},
        "ch2co": {"formula": "C2H2O", "bonds": {"C=C": 1, "C=O": 1, "C-H": 2}, "is_ring": False},
        "hcco": {"formula": "C2HO", "bonds": {"C=C": 1, "C=O": 1, "C-H": 1}, "is_ring": False},
        "hccoh": {"formula": "C2H2O", "bonds": {"C#C": 1, "C-O": 1, "C-H": 1, "O-H": 1}, "is_ring": False},
        "ch3co": {"formula": "C2H3O", "bonds": {"C-C": 1, "C=O": 1, "C-H": 3}, "is_ring": False},
        "ch3co2": {"formula": "C2H3O2", "bonds": {"C-C": 1, "C-O": 1, "C=O": 1, "C-H": 3}, "is_ring": False},
        "ch3cho": {"formula": "C2H4O", "bonds": {"C-C": 1, "C=O": 1, "C-H": 4}, "is_ring": False},
        "c2h5oh": {"formula": "C2H6O", "bonds": {"C-C": 1, "C-O": 1, "C-H": 5, "O-H": 1}, "is_ring": False},
        "c2h5o": {"formula": "C2H5O", "bonds": {"C-C": 1, "C-O": 1, "C-H": 5}, "is_ring": False},
        "pc2h4oh": {"formula": "C2H5O", "bonds": {"C-C": 1, "C-O": 1, "C-H": 4, "O-H": 1}, "is_ring": False},
        "sc2h4oh": {"formula": "C2H5O", "bonds": {"C-C": 1, "C-O": 1, "C-H": 4, "O-H": 1}, "is_ring": False},
        "ch3coch3": {"formula": "C3H6O", "bonds": {"C-C": 2, "C=O": 1, "C-H": 6}, "is_ring": False},
        "ch3coch2": {"formula": "C3H5O", "bonds": {"C-C": 2, "C=O": 1, "C-H": 5}, "is_ring": False},
        "c2h5cho": {"formula": "C3H6O", "bonds": {"C-C": 2, "C=O": 1, "C-H": 6}, "is_ring": False},
        "c2h5co": {"formula": "C3H5O", "bonds": {"C-C": 2, "C=O": 1, "C-H": 5}, "is_ring": False},
        "ch2cch2oh": {"formula": "C3H5O", "bonds": {"C-C": 1, "C=C": 1, "C-O": 1, "C-H": 4, "O-H": 1}, "is_ring": False},
        "neoc5h9q2": {"formula": "C5H11O4", "bonds": {"C-C": 4, "C-O": 2, "O-O": 2, "C-H": 10, "O-H": 1}, "is_ring": False},
        "neoc5h9q2-n": {"formula": "C5H11O4", "bonds": {"C-C": 4, "C-O": 2, "O-O": 2, "C-H": 10, "O-H": 1}, "is_ring": False},
        "neoc5ketox": {"formula": "C5H9O2", "bonds": {"C-C": 4, "C-O": 1, "C=O": 1, "C-H": 9}, "is_ring": False},
        "neoc5kejol": {"formula": "C5H9O2", "bonds": {"C-C": 4, "C-O": 1, "C=O": 1, "C-H": 8, "O-H": 1}, "is_ring": False},
        "neo-c5h10o": {"formula": "C5H10O", "bonds": {"C-C": 4, "C-O": 2, "C-H": 10}, "is_ring": True},
        "p12oohx2": {"formula": "C12H25O2", "bonds": {"C-C": 11, "C-O": 1, "O-O": 1, "C-H": 24, "O-H": 1}, "is_ring": False},
        "oc12ooh": {"formula": "C12H24O3", "bonds": {"C-C": 11, "C-O": 1, "C=O": 1, "O-O": 1, "C-H": 23, "O-H": 1}, "is_ring": False},
        "ch3o2": {"formula": "CH3O2", "bonds": {"C-O": 1, "O-O": 1, "C-H": 3}, "is_ring": False},
        "ch3o2h": {"formula": "CH4O2", "bonds": {"C-O": 1, "O-O": 1, "C-H": 3, "O-H": 1}, "is_ring": False},
        "c2h5o2": {"formula": "C2H5O2", "bonds": {"C-C": 1, "C-O": 1, "O-O": 1, "C-H": 5}, "is_ring": False},
        "c2h5o2h": {"formula": "C2H6O2", "bonds": {"C-C": 1, "C-O": 1, "O-O": 1, "C-H": 5, "O-H": 1}, "is_ring": False},
        "c2h": {"formula": "C2H", "bonds": {"C#C": 1, "C-H": 1}, "is_ring": False},
        "c2h2": {"formula": "C2H2", "bonds": {"C#C": 1, "C-H": 2}, "is_ring": False},
        "c2h3": {"formula": "C2H3", "bonds": {"C=C": 1, "C-H": 3}, "is_ring": False},
        "c2h4": {"formula": "C2H4", "bonds": {"C=C": 1, "C-H": 4}, "is_ring": False},
        "c2h5": {"formula": "C2H5", "bonds": {"C-C": 1, "C-H": 5}, "is_ring": False},
        "c2h6": {"formula": "C2H6", "bonds": {"C-C": 1, "C-H": 6}, "is_ring": False},
        "ic4h6q2-ii": {"formula": "C4H8O4", "bonds": {"C-C": 2, "C=C": 1, "C-O": 2, "O-O": 2, "C-H": 6, "O-H": 2}, "is_ring": False},
        "c4h71-4": {"formula": "C4H7", "bonds": {"C-C": 2, "C=C": 1, "C-H": 7}, "is_ring": False},
        "c3h4-a": {"formula": "C3H4", "bonds": {"C=C": 2, "C-H": 4}, "is_ring": False},
        "ac3h4": {"formula": "C3H4", "bonds": {"C=C": 2, "C-H": 4}, "is_ring": False},
        "c3h4-p": {"formula": "C3H4", "bonds": {"C-C": 1, "C#C": 1, "C-H": 4}, "is_ring": False},
        "pc3h4": {"formula": "C3H4", "bonds": {"C-C": 1, "C#C": 1, "C-H": 4}, "is_ring": False},
        "cc3h4": {"formula": "C3H4", "bonds": {"C-C": 2, "C=C": 1, "C-H": 4}, "is_ring": True},
        "c3h6": {"formula": "C3H6", "bonds": {"C-C": 1, "C=C": 1, "C-H": 6}, "is_ring": False},
        "c3h8": {"formula": "C3H8", "bonds": {"C-C": 2, "C-H": 8}, "is_ring": False},
        "nc3h7": {"formula": "C3H7", "bonds": {"C-C": 2, "C-H": 7}, "is_ring": False},
        "ic3h7": {"formula": "C3H7", "bonds": {"C-C": 2, "C-H": 7}, "is_ring": False},
        "c4h6": {"formula": "C4H6", "bonds": {"C-C": 1, "C=C": 2, "C-H": 6}, "is_ring": False},
        "c4h612": {"formula": "C4H6", "bonds": {"C-C": 1, "C=C": 2, "C-H": 6}, "is_ring": False},
        "c4h6-2": {"formula": "C4H6", "bonds": {"C-C": 2, "C#C": 1, "C-H": 6}, "is_ring": False},
        "c4h8-1": {"formula": "C4H8", "bonds": {"C-C": 2, "C=C": 1, "C-H": 8}, "is_ring": False},
        "c4h8-2": {"formula": "C4H8", "bonds": {"C-C": 2, "C=C": 1, "C-H": 8}, "is_ring": False},
        "c4h81": {"formula": "C4H8", "bonds": {"C-C": 2, "C=C": 1, "C-H": 8}, "is_ring": False},
        "c4h82": {"formula": "C4H8", "bonds": {"C-C": 2, "C=C": 1, "C-H": 8}, "is_ring": False},
        "ic4h8o": {"formula": "C4H8O", "bonds": {"C-C": 3, "C-O": 2, "C-H": 8}, "is_ring": True},
        "c4h4o": {"formula": "C4H4O", "bonds": {"C-C": 1, "C=C": 2, "C-O": 2, "C-H": 4}, "is_ring": True},
        "c2h3choch2": {"formula": "C4H6O", "bonds": {"C-C": 2, "C=C": 1, "C-O": 2, "C-H": 6}, "is_ring": True},
        "c4h6o23": {"formula": "C4H6O", "bonds": {"C-C": 2, "C=C": 1, "C-O": 2, "C-H": 6}, "is_ring": True},
        "c4h6o25": {"formula": "C4H6O", "bonds": {"C-C": 2, "C=C": 1, "C-O": 2, "C-H": 6}, "is_ring": True},
        "c2h5-2-c4h513": {"formula": "C6H10", "bonds": {"C-C": 3, "C=C": 2, "C-H": 10}, "is_ring": False},
        "c6h10-12": {"formula": "C6H10", "bonds": {"C-C": 3, "C=C": 2, "C-H": 10}, "is_ring": False},
        "c6h10-15": {"formula": "C6H10", "bonds": {"C-C": 3, "C=C": 2, "C-H": 10}, "is_ring": False},
        "l-c6h4": {"formula": "C6H4", "bonds": {"C-C": 2, "C=C": 1, "C#C": 2, "C-H": 4}, "is_ring": False},
        "o-c6h4": {"formula": "C6H4", "bonds": {"C-C": 3, "C=C": 2, "C#C": 1, "C-H": 4}, "is_ring": True},
        "c4h10": {"formula": "C4H10", "bonds": {"C-C": 3, "C-H": 10}, "is_ring": False},
        "ic4h10": {"formula": "C4H10", "bonds": {"C-C": 3, "C-H": 10}, "is_ring": False},
        "c6h6": {"formula": "C6H6", "bonds": {"C-C": 3, "C=C": 3, "C-H": 6}, "is_ring": True},
    }

    def __init__(self, chemkin_file):
        """Store the mechanism path and initialize parsed-species containers."""
        # Keep the original path so the same reader can parse species first,
        # then build the backbone dictionary without reopening a new object.
        self.chemkin_file = chemkin_file
        self.species = []
        self.species_backbone = {}
        self.raw_species_count = 0

    def read_chemkin_species(self):
        """Read the ChemKin species block and return the species names.

        ChemKin mechanism files usually contain a block such as:

            species
            h h2 o2 ...
            end

        Some files end the species block implicitly when the next section begins,
        for example with ``thermo`` or ``reactions``. This method supports both
        styles and stores the extracted names in ``self.species``.
        """
        species = []
        seen_species = set()
        in_species_block = False
        raw_species_count = 0

        for line in Path(self.chemkin_file).read_text(errors="ignore").splitlines():
            # ChemKin comments start with "!"; anything after it is metadata.
            line = line.split("!")[0].strip()
            if not line:
                continue

            lower_line = line.lower()

            # The species block starts when ChemKin declares "species".
            if lower_line.startswith("species"):
                in_species_block = True
                # Species can appear on the same line as the section header.
                rest_of_line = line[len("species"):].strip()
                if rest_of_line:
                    for species_name in rest_of_line.split():
                        raw_species_count += 1
                        if species_name in seen_species:
                            continue
                        seen_species.add(species_name)
                        species.append(species_name)
                continue

            if in_species_block:
                # Stop before reading thermo or reaction lines as species names.
                if (
                    lower_line == "end"
                    or lower_line.startswith("thermo")
                    or lower_line.startswith("reactions")
                ):
                    break

                for species_name in line.split():
                    raw_species_count += 1
                    # Keep the first occurrence only, but retain raw count as
                    # a sanity check for repeated labels in the mechanism file.
                    if species_name in seen_species:
                        continue
                    seen_species.add(species_name)
                    species.append(species_name)

        self.species = species
        self.raw_species_count = raw_species_count
        return self.species

    def read_species_backbone(self):
        """Estimate bond-count features for every species in the mechanism.

        Returns
        -------
        dict
            Dictionary with five main top-level keys:

            ``species``
                Maps each ChemKin species name to formula, bond counts, ring
                flag, formula-degeneracy ID, confidence, and notes.

            ``formula_degeneracies``
                Stores each formula/bond-pattern degeneracy under its own ID, for
                example ``C3H4_001``. This avoids storing different
                molecules under one formula bucket.

            ``ambiguous_formulae``
                Lists formulae that map to more than one formula degeneracy.

            ``multi_species_labels``
                Records species labels that may represent multiple structural
                species. This is a conservative name-based flag for bare/generic
                ChemKin names whose formula has multiple bond-pattern variants.

            ``sanity_checks``
                Reports raw species count and whether formula-degeneracy species
                references match the unique species list.

        Notes
        -----
        ChemKin files generally do not store molecular graphs. This method uses
        exact rules for common species and name heuristics for mechanism
        shorthand. For quantitative chemistry, replace heuristic entries with
        SMILES or RMG adjacency lists when available.
        """
        if not self.species:
            self.read_chemkin_species()

        species_info = {}
        signatures_by_formula = defaultdict(lambda: defaultdict(list))

        # Estimate each species independently, then group by formula/signature
        # so isomers or ring variants do not overwrite each other.
        for species_name in self.species:
            info = self._estimate_species_bond_counts(species_name)
            species_info[species_name] = info

            signature = self._bond_signature(info)
            formula_key = info["formula"] or "unknown"
            signatures_by_formula[formula_key][signature].append(species_name)

        formula_degeneracies = {}
        ambiguous_formulae = {}
        degeneracy_id_by_formula_signature = {}

        # Build one formula-degeneracy entry per unique bond signature.
        for formula_key in sorted(signatures_by_formula):
            signatures = sorted(signatures_by_formula[formula_key])
            if len(signatures) > 1:
                ambiguous_formulae[formula_key] = []

            for degeneracy_number, signature in enumerate(signatures, start=1):
                degeneracy_id = self._formula_degeneracy_id(
                    formula_key,
                    degeneracy_number,
                )
                bond_counts, is_ring = self._signature_to_bond_counts(signature)

                formula_degeneracies[degeneracy_id] = {
                    "formula": formula_key,
                    "degeneracy_id": degeneracy_id,
                    "bond_counts": bond_counts,
                    "is_ring": is_ring,
                    "species": signatures_by_formula[formula_key][signature],
                }
                degeneracy_id_by_formula_signature[
                    (formula_key, signature)
                ] = degeneracy_id

                if len(signatures) > 1:
                    ambiguous_formulae[formula_key].append(degeneracy_id)

        # Attach the degeneracy ID back to each individual species record.
        for species_name, info in species_info.items():
            formula_key = info["formula"] or "unknown"
            signature = self._bond_signature(info)
            info["formula_degeneracy_id"] = degeneracy_id_by_formula_signature[
                (formula_key, signature)
            ]

        multi_species_labels = {}
        # Bare formula labels can hide structural ambiguity, so flag them when
        # the same formula has multiple estimated bond patterns.
        for species_name, info in species_info.items():
            formula_key = info["formula"] or "unknown"
            degeneracy_ids = ambiguous_formulae.get(formula_key, [])
            may_represent_multiple = bool(
                degeneracy_ids
                and self._is_generic_species_label(species_name, formula_key)
            )

            info["may_represent_multiple_species"] = may_represent_multiple
            info["related_formula_degeneracy_ids"] = degeneracy_ids
            if may_represent_multiple:
                reason = (
                    "generic/bare species label and same formula has multiple "
                    "bond-count or ring degeneracies in this mechanism"
                )
                info["multi_species_reason"] = reason
                multi_species_labels[species_name] = {
                    "formula": formula_key,
                    "formula_degeneracy_id": info["formula_degeneracy_id"],
                    "related_formula_degeneracy_ids": degeneracy_ids,
                    "reason": reason,
                }
            else:
                info["multi_species_reason"] = None

        degeneracy_species_reference_count = sum(
            len(degeneracy["species"])
            for degeneracy in formula_degeneracies.values()
        )

        # This final object is intentionally plain Python data so it can be
        # serialized, inspected, or converted into model features later.
        self.species_backbone = {
            "species": species_info,
            "formula_degeneracies": formula_degeneracies,
            "ambiguous_formulae": ambiguous_formulae,
            "multi_species_labels": multi_species_labels,
            "sanity_checks": {
                "raw_species_count": self.raw_species_count,
                "degeneracy_species_reference_count": degeneracy_species_reference_count,
                "degeneracy_species_reference_matches_species_count": (
                    degeneracy_species_reference_count == len(self.species)
                ),
            },
        }
        return self.species_backbone

    def export_species_backbone_to_json(self, output_file, indent=2):
        """Write the species-backbone dictionary to a JSON file.

        Parameters
        ----------
        output_file : str or pathlib.Path
            Path to the JSON file that should be written.
        indent : int or None, optional
            Indentation passed to ``json.dump``. Use ``None`` for compact JSON.

        Returns
        -------
        pathlib.Path
            The resolved output path.
        """
        # Build the backbone on demand so callers do not have to remember the
        # exact read/export sequence.
        if not self.species_backbone:
            self.read_species_backbone()

        output_path = Path(output_file).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # The backbone is plain Python data, so it can be written directly.
        # sort_keys keeps diffs stable if the JSON is tracked or compared later.
        with output_path.open("w", encoding="utf-8") as file_handle:
            json.dump(
                self.species_backbone,
                file_handle,
                indent=indent,
                sort_keys=True,
            )
            file_handle.write("\n")

        return output_path

    def write_species_backbone_log(self, json_output_path, log_file):
        """Write and print a summary log for the current parsed mechanism."""
        if not self.species_backbone:
            self.read_species_backbone()

        log_output_path = Path(log_file).expanduser().resolve()
        sanity_checks = self.species_backbone["sanity_checks"]

        output_lines = [
            f"Mechanism file: {self.chemkin_file}",
            f"JSON output: {json_output_path}",
            f"Log output: {log_output_path}",
            f"Unique species found: {len(self.species)}",
            f"Raw species labels read: {sanity_checks['raw_species_count']}",
            f"Formula degeneracies: {len(self.species_backbone['formula_degeneracies'])}",
            f"Ambiguous formulae: {len(self.species_backbone['ambiguous_formulae'])}",
            f"Possible multi-species labels: {len(self.species_backbone['multi_species_labels'])}",
            "Degeneracy species references match unique species count: "
            f"{sanity_checks['degeneracy_species_reference_matches_species_count']}",
            "Top formula ambiguity summary:",
        ]
        output_lines.extend(
            formula_ambiguity_summary_lines(
                self.species_backbone,
                max_formulae=len(self.species_backbone["ambiguous_formulae"]),
            )
        )

        log_output_path.parent.mkdir(parents=True, exist_ok=True)
        log_output_path.write_text("\n".join(output_lines) + "\n", encoding="utf-8")

        print("\n".join(output_lines))
        return log_output_path

    def _bond_signature(self, info):
        """Create a hashable formula-degeneracy signature from bond counts and ring flag."""
        # Use BOND_KEYS order so two dictionaries with the same chemistry make
        # exactly the same tuple, independent of dictionary insertion details.
        return (
            tuple((key, info["bond_counts"][key]) for key in self.BOND_KEYS),
            info["is_ring"],
        )

    def _signature_to_bond_counts(self, signature):
        """Convert a stored formula-degeneracy signature back into dictionary form."""
        bond_count_items, is_ring = signature
        return dict(bond_count_items), is_ring

    @staticmethod
    def _formula_degeneracy_id(formula, degeneracy_number):
        """Create stable IDs such as C4H6_001 for formula-level degeneracies."""
        # Remove punctuation so the ID is convenient as a dictionary key or
        # column name in a later feature table.
        formula_label = re.sub(r"[^A-Za-z0-9]+", "_", formula or "unknown").strip("_")
        formula_label = formula_label or "unknown"
        return f"{formula_label}_{degeneracy_number:03d}"

    def _is_generic_species_label(self, species_name, formula):
        """Return True for labels that may hide structural ambiguity.

        Bare formula labels such as ``C4H6`` can hide structural ambiguity if
        the same formula has several bond/ring variants elsewhere in the same
        mechanism. Labels with explicit structural hints such as ``aC3H4``,
        ``pC3H4``, ``cC3H4``, or ``c12h24-4`` are treated as single named
        mechanism species.
        """
        clean_name = species_name.replace("*", "")
        # Exact rules are already explicit species, even if their formula also
        # appears in another form elsewhere.
        if clean_name.lower() in self.EXACT_SPECIES:
            return False
        return clean_name.lower() == (formula or "").lower()

    @classmethod
    def _empty_bond_counts(cls):
        """Return a zero-filled bond-count dictionary for all tracked bond keys."""
        # Every species gets the same feature dimensions, which is useful for ML.
        return {key: 0 for key in cls.BOND_KEYS}

    @classmethod
    def _with_empty_bond_counts(cls, partial_counts):
        """Merge known nonzero bond counts into a full zero-filled bond dictionary."""
        # Exact species are stored compactly, then expanded before returning.
        counts = cls._empty_bond_counts()
        counts.update(partial_counts)
        return counts

    @staticmethod
    def _clean_species_name(species_name):
        """Normalize a ChemKin species label before formula and bond parsing."""
        return species_name.strip().replace("*", "")

    @staticmethod
    def _formula_string(formula_counts):
        """Convert an element-count dictionary into a compact formula string."""
        if not formula_counts:
            return None

        pieces = []
        # Keep a stable element order instead of depending on dictionary order.
        for element in ("C", "H", "O", "N", "Ar", "He"):
            count = formula_counts.get(element, 0)
            if count:
                pieces.append(element if count == 1 else f"{element}{count}")
        return "".join(pieces) if pieces else None

    def _estimate_species_bond_counts(self, species_name):
        """Estimate formula, bond counts, ring status, and confidence for one species."""
        clean_name = self._clean_species_name(species_name)
        lower_name = clean_name.lower()
        notes = []

        # Exact species avoid ambiguous formula-only guesses, for example CO,
        # N2, CH3O, CH2OH, and H2.
        if lower_name in self.EXACT_SPECIES:
            exact = self.EXACT_SPECIES[lower_name]
            bond_counts = self._with_empty_bond_counts(exact["bonds"])
            return {
                "formula": exact["formula"],
                "bond_counts": bond_counts,
                "is_ring": exact["is_ring"],
                "confidence": "exact_name_rule",
                "notes": ["matched exact species rule"],
            }

        # If no exact entry exists, fall back to ChemKin-name heuristics.
        formula_counts = self._infer_formula_counts(clean_name)
        formula = self._formula_string(formula_counts)
        bond_counts = self._empty_bond_counts()
        is_ring = self._looks_like_ring(clean_name)

        if formula is None:
            notes.append("formula could not be inferred exactly from species name")

        # Add features in layers: carbon backbone first, then heteroatoms, then H.
        self._estimate_carbon_skeleton_bonds(
            clean_name,
            formula_counts,
            bond_counts,
            is_ring,
            notes,
        )
        is_ring = self._add_heteroatom_bond_heuristics(
            clean_name,
            formula_counts,
            bond_counts,
            is_ring,
            notes,
        )
        self._add_hydrogen_bond_heuristics(clean_name, formula_counts, bond_counts)

        confidence = "estimated_from_name"
        if notes:
            confidence = "name_heuristic"

        return {
            "formula": formula,
            "bond_counts": bond_counts,
            "is_ring": is_ring,
            "confidence": confidence,
            "notes": notes,
        }

    def _infer_formula_counts(self, species_name):
        """Infer element counts from ChemKin shorthand such as nc7h16 or c4h8o1-2."""
        lower_name = species_name.lower()
        no_star = lower_name.replace("*", "")

        # Inert bath gases are valid species but have no chemical backbone bonds.
        if no_star in {"ar", "he"}:
            return {"Ar" if no_star == "ar" else "He": 1}

        # Hydroperoxide shorthand needs special handling because "ooh" encodes
        # both two oxygen atoms and one additional hydrogen atom.
        if "ooh" in no_star:
            formula_counts = self._infer_ooh_formula(no_star)
            if formula_counts:
                return formula_counts

        if "ket" in no_star:
            formula_counts = self._infer_ketohydroperoxide_formula(no_star)
            if formula_counts:
                return formula_counts

        if self._looks_like_cyclic_ether_name(no_star):
            formula_counts = self._infer_cyclic_ether_formula(no_star)
            if formula_counts:
                return formula_counts

        formula_counts = self._infer_acyl_fragment_formula(no_star)
        if formula_counts:
            return formula_counts

        # Remove structural prefixes/suffixes before reading elemental tokens.
        text = self._strip_non_formula_prefix(no_star)
        text = re.sub(r"-\d+", "", text)
        text = re.sub(r"-[a-z]+$", "", text)
        text = self._strip_position_suffix_digits(text)
        text = re.sub(r"cc(\d+)", r"c\1", text)

        counts = defaultdict(int)
        matched = False
        # Parse repeated element tokens such as c12, h26, o2, n1.
        for match in re.finditer(r"(ar|he|[chon])(\d*)", text):
            element, amount = match.groups()
            start = match.start()
            end = match.end()
            next_char = text[end] if end < len(text) else ""

            # Skip structural prefixes such as nc7h16 or cC6H12.
            if start == 0 and element == "n" and next_char == "c":
                continue
            if start == 0 and element == "c" and next_char == "c":
                continue

            element = {"c": "C", "h": "H", "o": "O", "n": "N"}.get(
                element, element.capitalize()
            )
            counts[element] += int(amount) if amount else 1
            matched = True

        return dict(counts) if matched else {}

    @staticmethod
    def _strip_non_formula_prefix(lower_name):
        """Remove structural prefixes that are not part of the elemental formula."""
        prefixes = (
            "neo",
            "iso",
            "nc",
            "ic",
            "pc",
            "sc",
            "tc",
            "px",
            "sx",
            "tx",
            "s2x",
            "s3x",
            "s4x",
            "sax",
            "a",
            "i",
            "n",
            "p",
            "s",
            "t",
        )
        if lower_name.startswith(("o-c", "m-c")) and len(lower_name) > 3:
            return lower_name[2:]
        # Longer prefixes appear first so "neo" is removed before "n".
        for prefix in prefixes:
            if (
                lower_name.startswith(prefix + "-c")
                and len(lower_name) > len(prefix) + 2
            ):
                return lower_name[len(prefix) + 1:]
            if lower_name.startswith(prefix + "c") and len(lower_name) > len(prefix) + 1:
                return lower_name[len(prefix):]
        if lower_name.startswith("c-c"):
            return lower_name[2:]
        return lower_name

    @staticmethod
    def _strip_position_suffix_digits(lower_name):
        """Remove position labels that are attached directly to formula tokens.

        Some mechanisms use labels such as C4H81 for C4H8-1,
        C4H612 for C4H6-1,2, and C4H6O25 for C4H6O-2,5. The trailing
        position digits are not atom counts, so strip only the impossible
        count tail while keeping legitimate labels such as C16H33.
        """

        def replace_hydrogen_count(match):
            carbon_count = int(match.group(1))
            hydrogen_digits = match.group(2)
            maximum_hydrogen_count = 2 * carbon_count + 2

            if int(hydrogen_digits) <= maximum_hydrogen_count:
                return match.group(0)

            usable_count = None
            # Keep the longest chemically plausible hydrogen-count prefix and
            # treat the remaining digits as positional labels.
            for end_index in range(1, len(hydrogen_digits)):
                candidate = int(hydrogen_digits[:end_index])
                if candidate <= maximum_hydrogen_count:
                    usable_count = hydrogen_digits[:end_index]

            if usable_count is None:
                return match.group(0)
            return f"c{carbon_count}h{usable_count}"

        lower_name = re.sub(r"c(\d+)h(\d{2,})(?=$|[-a-z])", replace_hydrogen_count, lower_name)
        lower_name = re.sub(r"o(\d{2,})(?=$|[-a-z])", "o", lower_name)
        return lower_name

    @staticmethod
    def _infer_ooh_formula(lower_name):
        """Infer formulae for names containing hydroperoxide shorthand, such as c8h17ooh."""
        # Match the hydrocarbon part before "ooh"; the group contributes +HO2.
        match = re.search(r"c(\d+)h(\d+)ooh", lower_name)
        if match:
            carbon_count = int(match.group(1))
            hydrogen_count = int(match.group(2)) + 1
            oxygen_count = 2
            suffix_after_ooh = lower_name[match.end():]
        else:
            # LLNL C8-C16 mechanisms also use compact labels such as
            # c16ooh1-2 and c16ooh1-2o2, where the alkyl H count is omitted.
            match = re.search(r"c(\d+)ooh", lower_name)
            if not match:
                return {}

            carbon_count = int(match.group(1))
            hydrogen_count = 2 * carbon_count + 1
            oxygen_count = 2
            suffix_after_ooh = lower_name[match.end():]

        # Some names contain an additional peroxy group after the OOH group.
        if "o2" in suffix_after_ooh:
            oxygen_count += 2

        return {"C": carbon_count, "H": hydrogen_count, "O": oxygen_count}

    @staticmethod
    def _infer_ketohydroperoxide_formula(lower_name):
        """Infer formulae for KET shorthand labels such as nc7ket12."""
        if "ketox" in lower_name:
            return {}

        match = re.search(r"c(\d+)ket", lower_name)
        if not match:
            return {}

        carbon_count = int(match.group(1))
        return {"C": carbon_count, "H": 2 * carbon_count, "O": 3}

    @staticmethod
    def _infer_cyclic_ether_formula(lower_name):
        """Infer formulae for cyclic-ether position labels such as c5h10o1-2."""
        match = re.search(r"c(\d+)h(\d+)o", lower_name)
        if match:
            return {"C": int(match.group(1)), "H": int(match.group(2)), "O": 1}

        compact_match = re.match(r"^c(\d+)o\d+-\d+$", lower_name)
        if compact_match:
            carbon_count = int(compact_match.group(1))
            return {"C": carbon_count, "H": 2 * carbon_count, "O": 1}

        return {}

    @staticmethod
    def _infer_acyl_fragment_formula(lower_name):
        """Infer formulae for labels such as c12coc2h4p.

        These LLNL labels describe an alkyl-acyl fragment:
        c12coc2h4p is C12H25-CO-C2H4 radical, so its formula is C15H29O.
        """
        match = re.match(r"^c(\d+)coc(\d+)h(\d+)p$", lower_name)
        if not match:
            return {}

        alkyl_carbon_count = int(match.group(1))
        tail_carbon_count = int(match.group(2))
        tail_hydrogen_count = int(match.group(3))
        return {
            "C": alkyl_carbon_count + 1 + tail_carbon_count,
            "H": 2 * alkyl_carbon_count + 1 + tail_hydrogen_count,
            "O": 1,
        }

    @staticmethod
    def _looks_like_ring(species_name):
        """Detect common ChemKin naming patterns that indicate a cyclic species."""
        lower_name = species_name.lower()

        # ChemKin mechanisms often use c- or cC to mark cyclic species.
        if species_name.startswith("cC") or "cC" in species_name or lower_name.startswith("c-"):
            return True
        if lower_name in {"c5h5", "c6h5", "c6h6"}:  # known small rings
            return True
        if identify_bonds_from_chemikin_mechanism._looks_like_cyclic_ether_name(
            lower_name
        ):
            return True
        return False

    @staticmethod
    def _looks_like_cyclic_ether_name(lower_name):
        """Return True for LLNL-style cyclic ether labels."""
        # Some cyclic ethers store the oxygen bridge position as a suffix, e.g.
        # c5h10o1-2. Require H = 2*C so c5h11o2-1 stays a peroxy radical.
        position_match = re.match(r"^c(\d+)h(\d+)o\d+-\d+$", lower_name)
        if position_match and int(position_match.group(2)) == 2 * int(position_match.group(1)):
            return True
        compact_position_match = re.match(r"^c(\d+)o\d+-\d+$", lower_name)
        if compact_position_match:
            return True
        cc_match = re.match(r"^cc(\d+)h(\d+)o$", lower_name)
        if cc_match and int(cc_match.group(2)) == 2 * int(cc_match.group(1)):
            return True
        # Low-temperature alkane mechanisms also use labels like a-ac5h10o.
        branch_match = re.match(r"^[a-z]-[a-z]c(\d+)h(\d+)o$", lower_name)
        if branch_match and int(branch_match.group(2)) == 2 * int(branch_match.group(1)):
            return True
        return False

    @staticmethod
    def _carbon_count_from_name(species_name):
        """Extract the carbon count from a species name when formula parsing is incomplete."""
        # This backup is useful for partially parsed or unusual species labels.
        match = re.search(r"c(\d+)", species_name.lower())
        if match:
            return int(match.group(1))
        if "c" in species_name.lower():
            return 1
        return 0

    def _estimate_carbon_skeleton_bonds(
        self,
        species_name,
        formula_counts,
        bond_counts,
        is_ring,
        notes,
    ):
        """Estimate C-C, C=C, and C#C counts from formula and name patterns."""
        lower_name = species_name.lower()
        carbon_count = formula_counts.get("C", 0) or self._carbon_count_from_name(species_name)
        hydrogen_count = formula_counts.get("H")
        oxygen_count = formula_counts.get("O", 0)

        # Single-carbon species have no carbon-carbon skeleton.
        if carbon_count <= 1:
            return

        # Ring species need a different carbon-bond count from open chains.
        if is_ring:
            if oxygen_count:
                # Monocyclic ethers such as c5h10o1-2 and a-ac5h10o contain
                # two C-O ring bonds, so only C-1 carbon-carbon ring bonds remain.
                bond_counts["C-C"] += max(carbon_count - 1, 0)
                notes.append("oxygen-containing ring carbon skeleton estimated")
                return
            self._estimate_ring_carbon_bonds(carbon_count, hydrogen_count, bond_counts)
            notes.append("ring carbon skeleton estimated from name/formula")
            return

        # Acyclic aldehydes/ketones already use one degree of unsaturation for
        # C=O, so CnH2nO should not be treated as also containing C=C.
        if "ket" in lower_name or self._contains_carbonyl_name(lower_name):
            if hydrogen_count is not None and hydrogen_count <= 2 * carbon_count - 2:
                bond_counts["C=C"] += 1
                bond_counts["C-C"] += max(carbon_count - 2, 0)
            else:
                bond_counts["C-C"] += max(carbon_count - 1, 0)
            return

        # A single triple or double bond replaces one C-C single bond.
        if self._looks_like_alkyne(lower_name, carbon_count, hydrogen_count):
            bond_counts["C#C"] += 1
            bond_counts["C-C"] += max(carbon_count - 2, 0)
            return

        if self._looks_like_alkene(lower_name, carbon_count, hydrogen_count):
            bond_counts["C=C"] += 1
            bond_counts["C-C"] += max(carbon_count - 2, 0)
            return

        # Default open-chain alkane/radical skeleton.
        bond_counts["C-C"] += max(carbon_count - 1, 0)

    @staticmethod
    def _estimate_ring_carbon_bonds(carbon_count, hydrogen_count, bond_counts):
        """Add approximate carbon-skeleton bonds for cyclic hydrocarbon species."""
        # Benzene/phenyl-like labels are represented as alternating single/double bonds.
        if carbon_count == 6 and hydrogen_count in {5, 6}:
            bond_counts["C-C"] += 3
            bond_counts["C=C"] += 3
            return

        # Unsaturated rings get one double bond and the rest single bonds.
        if hydrogen_count is not None and hydrogen_count <= 2 * carbon_count - 2:
            bond_counts["C=C"] += 1
            bond_counts["C-C"] += max(carbon_count - 1, 0)
            return

        # Saturated ring approximation: one C-C bond per carbon atom.
        bond_counts["C-C"] += carbon_count

    @staticmethod
    def _looks_like_alkyne(lower_name, carbon_count, hydrogen_count):
        """Return True when a name/formula pattern suggests one C#C bond."""
        # Small acetylene/propargyl-style names are common in combustion mechanisms.
        if lower_name in {"c2h", "c2h2"}:
            return True
        if "cch" in lower_name or "c2h2" in lower_name:
            return True
        if hydrogen_count is None:
            return False
        return hydrogen_count <= 2 * carbon_count - 2 and carbon_count == 2

    @staticmethod
    def _looks_like_alkene(lower_name, carbon_count, hydrogen_count):
        """Return True when a name/formula pattern suggests one C=C bond."""
        # Position suffixes such as c4h8-1 usually denote alkene isomers.
        if (
            re.match(r"^c\d+h\d+-\d+$", lower_name)
            and hydrogen_count == 2 * carbon_count
        ):
            return True
        if hydrogen_count is None:
            return False
        return hydrogen_count in {2 * carbon_count, 2 * carbon_count - 1}

    def _add_heteroatom_bond_heuristics(
        self,
        species_name,
        formula_counts,
        bond_counts,
        is_ring,
        notes,
    ):
        """Add approximate C-O, C=O, O-O, and ring features from heteroatom names."""
        lower_name = species_name.lower()
        carbon_count = formula_counts.get("C", 0) or self._carbon_count_from_name(species_name)
        oxygen_count = formula_counts.get("O", 0)
        is_cyclic_ether = self._looks_like_cyclic_ether_name(lower_name)

        # o1-2 and a-ac5h10o style labels are treated as cyclic ethers with two C-O bonds.
        if is_cyclic_ether and carbon_count:
            bond_counts["C-O"] += 2
            is_ring = True
            notes.append("cyclic ether C-O bonds inferred from name pattern")

        # Ketohydroperoxide shorthand combines carbonyl, C-O, and O-O features.
        if "ket" in lower_name:
            bond_counts["C=O"] += 1
            bond_counts["C-O"] += 1
            bond_counts["O-O"] += 1
            notes.append("ketohydroperoxide shorthand inferred from species name")

        # OOH and O2 patterns are common in low-temperature oxidation chemistry.
        if "ooh" in lower_name:
            bond_counts["C-O"] += 1 if carbon_count else 0
            bond_counts["O-O"] += 1
            if "o2" in lower_name.split("ooh", 1)[-1]:
                bond_counts["C-O"] += 1 if carbon_count else 0
                bond_counts["O-O"] += 1
            notes.append("hydroperoxide/peroxy bonds inferred from ooh/o2 pattern")

        elif (
            not is_cyclic_ether
            and ("o2h" in lower_name or re.search(r"o2-\d+", lower_name))
        ):
            bond_counts["C-O"] += 1 if carbon_count else 0
            bond_counts["O-O"] += 1
            notes.append("peroxide bonds inferred from o2/o2h pattern")

        elif carbon_count and lower_name.endswith("o2"):
            bond_counts["C-O"] += 1
            bond_counts["O-O"] += 1
            notes.append("peroxy radical bonds inferred from o2 suffix")

        # Acyl-peroxy/peroxy-carbonyl labels can place the o2/co3/oo marker in
        # the middle or beginning of the name, for example ch3co3, ho2cho,
        # o2cho, o2c4h8cho, and ch3choococh3.
        if (
            carbon_count
            and not is_cyclic_ether
            and bond_counts["O-O"] == 0
            and self._contains_peroxy_or_acylperoxy_name(lower_name)
        ):
            bond_counts["C-O"] += 1
            bond_counts["O-O"] += 1
            notes.append("peroxy/acyl-peroxy bonds inferred from name pattern")

        # Carbonyl and alcohol/alkoxy labels add additional C=O or C-O features.
        if self._contains_carbonyl_name(lower_name):
            bond_counts["C=O"] += 1

        if self._contains_alcohol_or_alkoxy_name(lower_name):
            bond_counts["C-O"] += 1

        # If the formula contains C and O but no oxygen bond was inferred, add
        # one conservative C-O bond. This catches alkoxy/acyl shorthand such as
        # ac5h11o, pc4h9o, and c3h5o without changing carbonyl/peroxy cases.
        if carbon_count and oxygen_count and not self._has_oxygen_bond(bond_counts):
            bond_counts["C-O"] += 1
            notes.append("fallback C-O bond inferred for oxygen-containing species")

        return is_ring

    @staticmethod
    def _has_oxygen_bond(bond_counts):
        """Return True when any tracked bond includes oxygen."""
        oxygen_bonds = (
            "C-O",
            "C=O",
            "C#O",
            "O-O",
            "O=O",
            "O-N",
            "N=O",
            "O-H",
        )
        return any(bond_counts[bond] > 0 for bond in oxygen_bonds)

    @staticmethod
    def _contains_peroxy_or_acylperoxy_name(lower_name):
        """Check for peroxy markers not covered by the simple suffix rules."""
        if "ooh" in lower_name or "o2h" in lower_name:
            return False
        return (
            "co3" in lower_name
            or "o2" in lower_name
            or "oo" in lower_name
        )

    @staticmethod
    def _add_hydrogen_bond_heuristics(species_name, formula_counts, bond_counts):
        """Infer H-H, C-H, N-H, and O-H counts from formula/name information.

        ChemKin names do not explicitly list all H-atom attachments. This rule
        keeps the estimate conservative:

        - hydrogen-only species are counted as H-H pairs where possible;
        - explicit ``ooh``, ``o2h``, and ``oh`` name markers contribute O-H;
        - hydrogen-only oxygen species such as H2O and H2O2 put H on oxygen;
        - hydrogen-only nitrogen species put H on nitrogen;
        - remaining H atoms in carbon-containing species are counted as C-H.
        """
        hydrogen_count = formula_counts.get("H", 0)
        if hydrogen_count <= 0:
            return

        lower_name = species_name.lower()
        carbon_count = formula_counts.get("C", 0)
        nitrogen_count = formula_counts.get("N", 0)
        oxygen_count = formula_counts.get("O", 0)

        hydrogen_hydrogen_count = 0
        oxygen_hydrogen_count = 0
        nitrogen_hydrogen_count = 0

        # If there is no carbon, assign hydrogens to the present heteroatom
        # or to H-H pairs for hydrogen-only species.
        if carbon_count == 0:
            if oxygen_count:
                oxygen_hydrogen_count = hydrogen_count
            elif nitrogen_count:
                nitrogen_hydrogen_count = hydrogen_count
            else:
                hydrogen_hydrogen_count = hydrogen_count // 2
        else:
            oxygen_hydrogen_count += lower_name.count("ooh")
            remaining_name = lower_name.replace("ooh", "")
            oxygen_hydrogen_count += remaining_name.count("ho2")
            remaining_name = remaining_name.replace("ho2", "")
            oxygen_hydrogen_count += remaining_name.count("o2h")
            remaining_name = remaining_name.replace("o2h", "")
            oxygen_hydrogen_count += remaining_name.count("co3h")
            remaining_name = remaining_name.replace("co3h", "co3")
            if "ket" in lower_name and "ketox" not in lower_name:
                oxygen_hydrogen_count += 1
            oxygen_hydrogen_count += remaining_name.count("oh")

        # Clamp inferred H attachments so the estimate cannot exceed formula H.
        hydrogen_hydrogen_count = min(hydrogen_hydrogen_count, hydrogen_count // 2) # '//' divides and rounds down to the nearest whole number
        oxygen_hydrogen_count = min(oxygen_hydrogen_count, hydrogen_count)
        nitrogen_hydrogen_count = min(
            nitrogen_hydrogen_count,
            hydrogen_count - 2 * hydrogen_hydrogen_count - oxygen_hydrogen_count,
        )

        remaining_hydrogen_count = (
            hydrogen_count
            - 2 * hydrogen_hydrogen_count
            - oxygen_hydrogen_count
            - nitrogen_hydrogen_count
        )

        bond_counts["H-H"] += hydrogen_hydrogen_count
        bond_counts["O-H"] += oxygen_hydrogen_count
        bond_counts["N-H"] += nitrogen_hydrogen_count

        # In carbon-containing species, any remaining hydrogens are approximated
        # as C-H because ChemKin names rarely identify every H attachment.
        if carbon_count:
            bond_counts["C-H"] += max(remaining_hydrogen_count, 0)
        elif nitrogen_count and remaining_hydrogen_count:
            bond_counts["N-H"] += remaining_hydrogen_count
        elif oxygen_count and remaining_hydrogen_count:
            bond_counts["O-H"] += remaining_hydrogen_count

    @staticmethod
    def _contains_carbonyl_name(lower_name):
        """Check whether the species name contains aldehyde/carbonyl-like markers."""
        if lower_name in {"co", "co2"}:
            return False
        if lower_name.endswith("-co2"):
            return False
        return "cho" in lower_name or "co" in lower_name

    @staticmethod
    def _contains_alcohol_or_alkoxy_name(lower_name):
        """Check whether the species name contains alcohol or alkoxy-like markers."""
        if (
            "ooh" in lower_name
            or "o2" in lower_name
            or identify_bonds_from_chemikin_mechanism._contains_carbonyl_name(lower_name)
        ):
            return False
        return lower_name.endswith("oh") or re.search(r"o-\d+$", lower_name) is not None


# Backward-compatible alias for older notebooks/scripts.
identify_backbone_from_chemkin_mechanism = identify_bonds_from_chemikin_mechanism


def _nonzero_bond_counts(bond_counts):
    """Drop zero-count bond entries for compact printing."""
    # The stored data keeps zeros; this helper only shortens terminal output.
    return {bond: count for bond, count in bond_counts.items() if count}


def formula_ambiguity_summary_lines(backbone, max_formulae=8, max_species=5):
    """Return summary lines for formulae with multiple bond-count/ring variants."""
    sanity_checks = backbone["sanity_checks"]
    ambiguous_formulae = backbone["ambiguous_formulae"]
    formula_degeneracies = backbone["formula_degeneracies"]
    multi_species_labels = backbone["multi_species_labels"]
    lines = []

    lines.append(
        "  sanity: "
        f"raw={sanity_checks['raw_species_count']}, "
        "degeneracy_refs_match_species="
        f"{sanity_checks['degeneracy_species_reference_matches_species_count']}"
    )

    lines.append(
        "  species labels that may represent multiple structures: "
        f"{len(multi_species_labels)}"
    )
    for species_name, label_info in list(multi_species_labels.items())[:max_species]:
        lines.append(
            f"    {species_name}: formula={label_info['formula']}, "
            f"degeneracies={label_info['related_formula_degeneracy_ids']}"
        )

    lines.append(
        "  formulae with multiple possible molecules: "
        f"{len(ambiguous_formulae)}"
    )

    sorted_formulae = sorted(
        ambiguous_formulae,
        key=lambda formula: (-len(ambiguous_formulae[formula]), formula),
    )

    for formula in sorted_formulae[:max_formulae]:
        degeneracy_ids = ambiguous_formulae[formula]
        lines.append(f"    {formula}: {len(degeneracy_ids)} degeneracies")
        for degeneracy_id in degeneracy_ids:
            degeneracy = formula_degeneracies[degeneracy_id]
            species_preview = ", ".join(degeneracy["species"][:max_species])
            if len(degeneracy["species"]) > max_species:
                species_preview += ", ..."
            lines.append(
                f"      {degeneracy_id}: "
                f"ring={degeneracy['is_ring']}, "
                f"bonds={_nonzero_bond_counts(degeneracy['bond_counts'])}, "
                f"species=[{species_preview}]"
            )

    return lines


def print_formula_ambiguity_summary(backbone, max_formulae=8, max_species=5):
    """Print formulae that have multiple bond-count/ring variants."""
    # This summary is meant for quick inspection, not for machine parsing.
    for line in formula_ambiguity_summary_lines(backbone, max_formulae, max_species):
        print(line)


if __name__ == "__main__":
    # When run as a script, parse the n-heptane mechanism and export the full
    # species-backbone dictionary for downstream inspection or GNN preprocessing.
    script_dir = Path(__file__).resolve().parent
    mechanism_dir = script_dir / "mechanism_file_chemkin"
    mechanism_file = mechanism_dir / "nHeptane_LLNL.txt"
    output_file = script_dir / "output" / "nHeptane_LLNL_species_backbone.json"
    log_file = script_dir / "output" / f"log_{mechanism_file.stem}.txt"

    reader = identify_bonds_from_chemikin_mechanism(mechanism_file)
    reader.read_chemkin_species()
    reader.read_species_backbone()
    json_output_path = reader.export_species_backbone_to_json(output_file)
    reader.write_species_backbone_log(json_output_path, log_file)
