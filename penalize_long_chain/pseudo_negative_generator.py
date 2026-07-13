"""Generate valence-valid pseudo-negative molecular feature vectors.

The positive data only stores aggregate bond counts, so that is also the unit
of uniqueness here.  Atom graphs are built first, validated, and then reduced
to those counts.  This keeps formula and bond bookkeeping honest while still
producing records that match the existing dataset.
"""

from __future__ import annotations

import json
import math
import random
import re
from collections import Counter
from pathlib import Path


# Keep this fallback in sync with the feature vocabulary used by identify_bonds.py.
# Reference files may add more keys; those are collected when they are loaded.
DEFAULT_BOND_KEYS = (
    "C#C",
    "C#N",
    "C#O",
    "C-C",
    "C-H",
    "C-N",
    "C-O",
    "C=C",
    "C=N",
    "C=O",
    "H-H",
    "N#N",
    "N-H",
    "N-N",
    "N=N",
    "N=O",
    "O-H",
    "O-N",
    "O-O",
    "O=O",
)

MAX_VALENCE = {"C": 4, "H": 1, "O": 2}
BOND_ORDER = {"-": 1, "=": 2, "#": 3}
BOND_KEY_RE = re.compile(r"^([A-Z][a-z]?)([-=#])([A-Z][a-z]?)$")


class pseudo_negative_generator:
    """Build synthetic molecular degeneracies without violating valence.

    ``n`` is a global cap: samples accumulate across generator calls, but the
    instance never retains more than ``n`` unique aggregate signatures.
    Randomness belongs to the instance so a fixed seed reproduces both sample
    order and atom placement.
    """

    def __init__(
        self,
        n,
        reference_paths,
        random_seed=None,
        max_attempts=None,
    ):
        self.n = self._require_int("n", n, minimum=1)
        if random_seed is not None:
            random_seed = self._require_int("random_seed", random_seed)
        self.random_seed = random_seed
        self._rng = random.Random(random_seed)

        if max_attempts is None:
            # This is a per-method limit.  It prevents impossible or mostly
            # duplicate parameter spaces from turning into unbounded searches.
            max_attempts = max(100, self.n * 50)
        self.max_attempts = self._require_int(
            "max_attempts", max_attempts, minimum=1
        )
        # Branched placement uses dynamic programming rather than randomized
        # retries.  Cap its total state transitions for one public call so one
        # very large topology cannot hide unbounded work inside one attempt.
        self._branch_search_state_limit = min(
            2_000_000, max(10_000, self.max_attempts * 2_000)
        )

        self.reference_paths = self._resolve_reference_paths(reference_paths)
        reference_records = self._load_reference_records()

        bond_keys = set(DEFAULT_BOND_KEYS)
        for _, _, _, record in reference_records:
            bond_keys.update(record["bond_counts"])
        self.bond_keys = tuple(sorted(bond_keys))
        self._bond_key_lookup = self._build_bond_key_lookup()

        self._observed_signatures = {
            self._canonical_signature(
                record["formula"], record["bond_counts"], record["is_ring"]
            )
            for _, _, _, record in reference_records
        }
        self._generated_signatures = set()
        self.generated_samples = []
        self._statistics = {
            "attempted": 0,
            "rejected_invalid": 0,
            "rejected_duplicate": 0,
            "rejected_observed": 0,
            "accepted": 0,
        }

    def long_carbon_chain(
        self,
        chain_length,
        double_bond_seed,
        O_seed,
        triple_bond_seed=0,
    ):
        """Generate connected acyclic carbon chains.

        The bond/O seed arguments are inclusive upper bounds, not random-number
        seeds. Carbon bonds may be promoted to C=C or C#C. Oxygen is added as
        hydroxyl (C-O-H) or carbonyl (C=O). Every feasible count mix is explored
        up to max_attempts; atom positions are randomized.
        """
        chain_length = self._require_int(
            "chain_length", chain_length, minimum=2
        )
        double_bond_seed = self._require_int(
            "double_bond_seed", double_bond_seed, minimum=0
        )
        triple_bond_seed = self._require_int(
            "triple_bond_seed", triple_bond_seed, minimum=0
        )
        O_seed = self._require_int("O_seed", O_seed, minimum=0)

        if self._is_full():
            return []

        # A single-bonded C_n chain has 2n+2 free carbon valences before H is
        # added.  No oxygen-substituent count above that can be valid.
        max_double = min(double_bond_seed, chain_length - 1)
        max_triple = min(triple_bond_seed, chain_length - 1)
        max_oxygen = min(O_seed, 2 * chain_length + 2)
        parameters = {
            "chain_length": chain_length,
            "double_bond_seed": double_bond_seed,
            "triple_bond_seed": triple_bond_seed,
            "O_seed": O_seed,
        }

        def build(double_count, triple_count, oxygen_count, carbonyl_count):
            graph = self._carbon_chain_graph(chain_length)
            hydroxyl_count = oxygen_count - carbonyl_count
            if not self._promote_chain_bonds(
                graph,
                double_count,
                triple_count,
                carbonyl_count,
                hydroxyl_count,
            ):
                return None
            placements = self._attach_oxygen_motifs(
                graph, carbonyl_count, hydroxyl_count
            )
            if placements is None:
                return None
            self._saturate_with_hydrogen(graph)
            notes = [
                f"acyclic main chain with {chain_length} carbon atoms",
                self._motif_note(
                    double_count,
                    triple_count,
                    carbonyl_count,
                    hydroxyl_count,
                    placements,
                ),
            ]
            return graph, notes

        return self._run_carbon_configurations(
            "long_carbon_chain",
            parameters,
            is_ring=False,
            max_double=max_double,
            max_triple=max_triple,
            max_oxygen=max_oxygen,
            builder=build,
        )

    def carbon_ring(
        self,
        ring_size,
        double_bond_seed,
        O_seed,
        triple_bond_seed=0,
    ):
        """Generate one carbon ring with hydroxyl/carbonyl substituents.

        Promoted ring edges form a matching: C=C and C#C bonds may not share a
        carbon. This implements the task's "non-conflicting" ring-bond rule.
        """
        ring_size = self._require_int("ring_size", ring_size, minimum=3)
        double_bond_seed = self._require_int(
            "double_bond_seed", double_bond_seed, minimum=0
        )
        triple_bond_seed = self._require_int(
            "triple_bond_seed", triple_bond_seed, minimum=0
        )
        O_seed = self._require_int("O_seed", O_seed, minimum=0)

        if self._is_full():
            return []

        max_double = min(double_bond_seed, ring_size // 2)
        max_triple = min(triple_bond_seed, ring_size // 2)
        max_oxygen = min(O_seed, 2 * ring_size)
        parameters = {
            "ring_size": ring_size,
            "double_bond_seed": double_bond_seed,
            "triple_bond_seed": triple_bond_seed,
            "O_seed": O_seed,
        }

        def build(double_count, triple_count, oxygen_count, carbonyl_count):
            graph = self._carbon_ring_graph(ring_size)
            hydroxyl_count = oxygen_count - carbonyl_count
            if not self._promote_ring_bonds(
                graph,
                double_count,
                triple_count,
                carbonyl_count,
                hydroxyl_count,
            ):
                return None
            placements = self._attach_oxygen_motifs(
                graph, carbonyl_count, hydroxyl_count
            )
            if placements is None:
                return None
            self._saturate_with_hydrogen(graph)
            notes = [
                f"single ring with {ring_size} carbon atoms",
                self._motif_note(
                    double_count,
                    triple_count,
                    carbonyl_count,
                    hydroxyl_count,
                    placements,
                ),
            ]
            return graph, notes

        return self._run_carbon_configurations(
            "carbon_ring",
            parameters,
            is_ring=True,
            max_double=max_double,
            max_triple=max_triple,
            max_oxygen=max_oxygen,
            builder=build,
        )

    def highly_branched_carbon(
        self,
        branch_to_main_chain_ratio,
        double_bond_seed,
        O_seed,
        main_chain_length=6,
        triple_bond_seed=0,
    ):
        """Generate an acyclic main chain with one-carbon branches.

        The original three positional arguments remain valid.
        ``main_chain_length`` defaults to six because the old API did not
        specify a carbon count.  Branch count uses half-up rounding:
        ``floor(main_chain_length * ratio + 0.5)``, with a minimum of one.
        """
        ratio = self._require_ratio(branch_to_main_chain_ratio)
        double_bond_seed = self._require_int(
            "double_bond_seed", double_bond_seed, minimum=0
        )
        triple_bond_seed = self._require_int(
            "triple_bond_seed", triple_bond_seed, minimum=0
        )
        O_seed = self._require_int("O_seed", O_seed, minimum=0)
        main_chain_length = self._require_int(
            "main_chain_length", main_chain_length, minimum=2
        )

        if self._is_full():
            return []

        branch_count = max(1, math.floor(main_chain_length * ratio + 0.5))
        carbon_count = main_chain_length + branch_count
        max_double = min(double_bond_seed, carbon_count - 1)
        max_triple = min(triple_bond_seed, carbon_count - 1)
        max_oxygen = min(O_seed, 2 * carbon_count + 2)
        parameters = {
            "branch_to_main_chain_ratio": ratio,
            "double_bond_seed": double_bond_seed,
            "triple_bond_seed": triple_bond_seed,
            "O_seed": O_seed,
            "main_chain_length": main_chain_length,
            "branch_count": branch_count,
        }
        skeletons = {}
        branch_search_budget = {"remaining": self._branch_search_state_limit}

        def build(double_count, triple_count, oxygen_count, carbonyl_count):
            skeleton_key = (double_count, triple_count)
            if skeleton_key not in skeletons:
                skeletons[skeleton_key] = self._branched_carbon_graph(
                    main_chain_length,
                    branch_count,
                    double_count,
                    triple_count,
                    branch_search_budget,
                )
            skeleton = skeletons[skeleton_key]
            if skeleton is None:
                return None
            source_graph, branch_positions = skeleton
            graph = self._copy_graph(source_graph)
            hydroxyl_count = oxygen_count - carbonyl_count
            placements = self._attach_oxygen_motifs(
                graph, carbonyl_count, hydroxyl_count
            )
            if placements is None:
                return None
            self._saturate_with_hydrogen(graph)
            notes = [
                (
                    f"{main_chain_length}-carbon main chain with "
                    f"{branch_count} one-carbon branches at "
                    f"positions {branch_positions}"
                ),
                self._motif_note(
                    double_count,
                    triple_count,
                    carbonyl_count,
                    hydroxyl_count,
                    placements,
                ),
            ]
            return graph, notes

        return self._run_carbon_configurations(
            "highly_branched_carbon",
            parameters,
            is_ring=False,
            max_double=max_double,
            max_triple=max_triple,
            max_oxygen=max_oxygen,
            builder=build,
        )

    def long_oxygen_chain(self, chain_length, double_bond_seed):
        """Generate a connected O-O chain and fill terminal valence with H.

        In a chain longer than two atoms, promoting any O-O edge to O=O would
        give at least one oxygen valence three.  Such configurations are tried
        and rejected normally rather than special-cased as valid chemistry.
        """
        chain_length = self._require_int(
            "chain_length", chain_length, minimum=2
        )
        double_bond_seed = self._require_int(
            "double_bond_seed", double_bond_seed, minimum=0
        )

        if self._is_full():
            return []

        double_counts = list(
            range(min(double_bond_seed, chain_length - 1) + 1)
        )
        self._rng.shuffle(double_counts)
        parameters = {
            "chain_length": chain_length,
            "double_bond_seed": double_bond_seed,
        }
        new_samples = []

        for double_count in double_counts[: self.max_attempts]:
            if self._is_full():
                break
            graph = self._oxygen_chain_graph(chain_length)
            if not self._promote_oxygen_chain_bonds(graph, double_count):
                self._record_invalid_attempt()
                continue
            self._saturate_with_hydrogen(graph)
            sample = self._try_add_sample(
                graph,
                generator="long_oxygen_chain",
                parameters=parameters,
                is_ring=False,
                notes=[
                    f"acyclic chain with {chain_length} consecutive oxygen atoms",
                    f"{double_count} O=O bond(s)",
                ],
            )
            if sample is not None:
                new_samples.append(sample)

        return new_samples

    def export_to_json(self, output_path, indent=2):
        """Write samples using the same schema as master_dataset.json."""
        indent = self._require_int("indent", indent, minimum=0)
        output_path = Path(output_path).expanduser().resolve()
        if output_path in self.reference_paths:
            raise ValueError("refusing to overwrite a reference dataset")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = self._master_dataset_payload()
        with output_path.open("w", encoding="utf-8") as output_file:
            json.dump(payload, output_file, indent=indent, sort_keys=True)
            output_file.write("\n")
        return output_path

    def write_generation_log(self, json_output_path, log_file):
        """Write a readable summary of the generated pseudo dataset."""
        json_output_path = Path(json_output_path).expanduser().resolve()
        log_file = Path(log_file).expanduser().resolve()
        log_file.parent.mkdir(parents=True, exist_ok=True)

        statistics = self._generation_statistics()
        lines = self._generation_log_lines(
            statistics,
            json_output_path,
            log_file,
        )
        with log_file.open("w", encoding="utf-8") as output_file:
            output_file.write("\n".join(lines))
            output_file.write("\n")
        return log_file

    def _master_dataset_payload(self):
        """Convert internal samples to master formula-degeneracy entries."""
        samples_by_formula = {}
        for sample in self.generated_samples:
            samples_by_formula.setdefault(sample["formula"], []).append(sample)

        formula_degeneracies = {}
        for formula in sorted(samples_by_formula):
            samples = sorted(
                samples_by_formula[formula],
                key=lambda sample: (
                    sample["is_ring"],
                    tuple(sample["bond_counts"][key] for key in self.bond_keys),
                    sample["sample_id"],
                ),
            )
            for number, sample in enumerate(samples, start=1):
                degeneracy_id = self._formula_degeneracy_id(formula, number)
                formula_degeneracies[degeneracy_id] = {
                    "formula": formula,
                    "degeneracy_id": degeneracy_id,
                    "bond_counts": sample["bond_counts"],
                    "is_ring": sample["is_ring"],
                    "species": [sample["sample_id"]],
                }

        return {"formula_degeneracies": formula_degeneracies}

    @staticmethod
    def _formula_degeneracy_id(formula, number):
        """Create stable IDs such as C20H38O_001."""
        formula_label = re.sub(r"[^A-Za-z0-9]+", "_", formula or "unknown")
        formula_label = formula_label.strip("_") or "unknown"
        return f"{formula_label}_{number:03d}"

    def _generation_statistics(self):
        """Summarize accepted samples and their non-H backbone bonds."""
        formula_counts = Counter(
            sample["formula"] for sample in self.generated_samples
        )
        generator_counts = Counter(
            sample["generator"] for sample in self.generated_samples
        )
        non_h_bond_keys = [key for key in self.bond_keys if "H" not in key]
        bond_summary = {}
        for key in non_h_bond_keys:
            counts = [
                sample["bond_counts"][key] for sample in self.generated_samples
            ]
            bond_summary[key] = {
                "max": max(counts, default=0),
                "nonzero_groups": sum(count > 0 for count in counts),
            }

        return {
            "formula_degeneracy_groups": len(self.generated_samples),
            "formula_count": len(formula_counts),
            "ambiguous_formula_count": sum(
                count > 1 for count in formula_counts.values()
            ),
            "ring_groups": sum(
                sample["is_ring"] for sample in self.generated_samples
            ),
            "groups_with_backbone": sum(
                any(sample["bond_counts"][key] for key in non_h_bond_keys)
                for sample in self.generated_samples
            ),
            "generator_counts": dict(sorted(generator_counts.items())),
            "bond_summary": bond_summary,
            "top_formulae": sorted(
                formula_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )[:20],
        }

    def _generation_log_lines(self, statistics, json_output_path, log_file):
        """Format generation and dataset statistics as readable log lines."""
        lines = [
            "Pseudo-negative dataset generation log",
            "",
            "========================================",
            "PSEUDO-NEGATIVE DATASET STATISTICS",
            "========================================",
            (
                "Formula-degeneracy groups: "
                f"{statistics['formula_degeneracy_groups']}"
            ),
            f"Formulae represented: {statistics['formula_count']}",
            f"Ambiguous formulae: {statistics['ambiguous_formula_count']}",
            f"Groups with backbone: {statistics['groups_with_backbone']}",
            f"Ring groups: {statistics['ring_groups']}",
            "Generated groups by family:",
        ]
        for generator, count in statistics["generator_counts"].items():
            lines.append(f"  {generator}: {count}")

        lines.append("Backbone bond summary, excluding bonds that contain H:")
        for bond, summary in statistics["bond_summary"].items():
            lines.append(
                f"  {bond}: max {summary['max']}, "
                f"nonzero groups {summary['nonzero_groups']}"
            )

        generation_summary = self.summary()
        lines.extend(
            [
                "========================================",
                "",
                f"Pseudo JSON output: {json_output_path}",
                f"Log output: {log_file}",
                f"Random seed: {self.random_seed}",
                f"Maximum retained samples: {self.n}",
                f"Candidate attempts: {generation_summary['attempted']}",
                f"Accepted: {generation_summary['accepted']}",
                (
                    "Rejected as invalid: "
                    f"{generation_summary['rejected_invalid']}"
                ),
                (
                    "Rejected as duplicate: "
                    f"{generation_summary['rejected_duplicate']}"
                ),
                (
                    "Rejected as observed positive: "
                    f"{generation_summary['rejected_observed']}"
                ),
                f"Remaining capacity: {generation_summary['remaining_capacity']}",
                f"Positive reference files: {len(self.reference_paths)}",
            ]
        )
        for path in self.reference_paths:
            lines.append(f"  {path}")

        lines.append("Top formulae by degeneracy count:")
        for formula, count in statistics["top_formulae"]:
            lines.append(f"  {formula}: {count}")
        return lines

    def summary(self):
        """Return generation counters without exposing mutable internal state."""
        result = dict(self._statistics)
        result["generated_sample_count"] = len(self.generated_samples)
        result["remaining_capacity"] = self.n - len(self.generated_samples)
        return result

    # ------------------------------------------------------------------
    # Input and reference-data handling

    @staticmethod
    def _require_int(name, value, minimum=None):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
        if minimum is not None and value < minimum:
            raise ValueError(f"{name} must be >= {minimum}")
        return value

    @staticmethod
    def _require_ratio(value):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("branch_to_main_chain_ratio must be a number")
        value = float(value)
        if not math.isfinite(value) or not 0.0 < value <= 1.0:
            raise ValueError(
                "branch_to_main_chain_ratio must be finite and in (0, 1]"
            )
        return value

    def _resolve_reference_paths(self, reference_paths):
        if reference_paths is None:
            raise ValueError("reference_paths must be explicitly provided")
        if isinstance(reference_paths, (str, Path)):
            reference_paths = [reference_paths]

        try:
            paths = [Path(path).expanduser().resolve() for path in reference_paths]
        except TypeError as exc:
            raise ValueError("reference_paths must be an iterable of paths") from exc

        unique_paths = []
        seen = set()
        for path in paths:
            if path in seen:
                continue
            if not path.is_file():
                raise FileNotFoundError(f"reference file not found: {path}")
            unique_paths.append(path)
            seen.add(path)
        return tuple(unique_paths)

    def _load_reference_records(self):
        records = []
        for path in self.reference_paths:
            try:
                with path.open(encoding="utf-8") as input_file:
                    data = json.load(input_file)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in reference file {path}") from exc

            if not isinstance(data, dict):
                raise ValueError(f"reference file must contain an object: {path}")

            found_section = False
            for section in ("species", "formula_degeneracies"):
                if section not in data:
                    continue
                found_section = True
                entries = data[section]
                if not isinstance(entries, dict):
                    raise ValueError(f"{path}: {section} must be an object")
                for record_id, record in entries.items():
                    self._validate_reference_record(path, section, record_id, record)
                    records.append((path, section, record_id, record))

            if not found_section:
                raise ValueError(
                    f"{path}: expected species or formula_degeneracies section"
                )
        return records

    @staticmethod
    def _validate_reference_record(path, section, record_id, record):
        label = f"{path}:{section}:{record_id}"
        if not isinstance(record, dict):
            raise ValueError(f"{label}: record must be an object")
        if not isinstance(record.get("formula"), str) or not record["formula"]:
            raise ValueError(f"{label}: formula must be a non-empty string")
        if not isinstance(record.get("is_ring"), bool):
            raise ValueError(f"{label}: is_ring must be a boolean")

        bond_counts = record.get("bond_counts")
        if not isinstance(bond_counts, dict):
            raise ValueError(f"{label}: bond_counts must be an object")
        for key, count in bond_counts.items():
            if not isinstance(key, str):
                raise ValueError(f"{label}: bond-count keys must be strings")
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError(
                    f"{label}: bond count {key!r} must be a non-negative integer"
                )

    def _build_bond_key_lookup(self):
        lookup = {}
        for key in self.bond_keys:
            match = BOND_KEY_RE.fullmatch(key)
            if match is None:
                continue
            left, symbol, right = match.groups()
            identity = (tuple(sorted((left, right))), BOND_ORDER[symbol])
            previous = lookup.get(identity)
            if previous is not None and previous != key:
                raise ValueError(
                    f"ambiguous feature keys {previous!r} and {key!r}"
                )
            lookup[identity] = key

        required = (
            (("C", "C"), 1),
            (("C", "C"), 2),
            (("C", "C"), 3),
            (("C", "H"), 1),
            (("C", "O"), 1),
            (("C", "O"), 2),
            (("H", "O"), 1),
            (("O", "O"), 1),
            (("O", "O"), 2),
        )
        missing = [identity for identity in required if identity not in lookup]
        if missing:
            raise ValueError(f"bond feature vocabulary is missing {missing}")
        return lookup

    # ------------------------------------------------------------------
    # Candidate-space traversal

    def _configuration_counts(self, max_double, max_triple, max_oxygen):
        """Return randomized multiple-bond and oxygen-count combinations.

        There are ``oxygen_count + 1`` carbonyl/hydroxyl splits for each oxygen
        count.  Random rank sampling avoids materializing a huge Cartesian
        product when max_attempts is the tighter bound.
        """
        oxygen_splits = (max_oxygen + 1) * (max_oxygen + 2) // 2
        total = (max_double + 1) * (max_triple + 1) * oxygen_splits
        count = min(total, self.max_attempts)

        if count == total:
            ranks = list(range(total))
            self._rng.shuffle(ranks)
        else:
            ranks = self._rng.sample(range(total), count)

        result = []
        for rank in ranks:
            multiple_bond_rank, oxygen_rank = divmod(rank, oxygen_splits)
            double_count, triple_count = divmod(
                multiple_bond_rank,
                max_triple + 1,
            )
            oxygen_count = (math.isqrt(8 * oxygen_rank + 1) - 1) // 2
            first_rank = oxygen_count * (oxygen_count + 1) // 2
            carbonyl_count = oxygen_rank - first_rank
            result.append(
                (double_count, triple_count, oxygen_count, carbonyl_count)
            )
        return result

    def _run_carbon_configurations(
        self,
        generator,
        parameters,
        is_ring,
        max_double,
        max_triple,
        max_oxygen,
        builder,
    ):
        new_samples = []
        for double_count, triple_count, oxygen_count, carbonyl_count in (
            self._configuration_counts(max_double, max_triple, max_oxygen)
        ):
            if self._is_full():
                break
            candidate = builder(
                double_count,
                triple_count,
                oxygen_count,
                carbonyl_count,
            )
            if candidate is None:
                self._record_invalid_attempt()
                continue
            graph, notes = candidate
            sample = self._try_add_sample(
                graph,
                generator=generator,
                parameters=parameters,
                is_ring=is_ring,
                notes=notes,
            )
            if sample is not None:
                new_samples.append(sample)
        return new_samples

    def _is_full(self):
        return len(self.generated_samples) >= self.n

    def _record_invalid_attempt(self):
        self._statistics["attempted"] += 1
        self._statistics["rejected_invalid"] += 1

    # ------------------------------------------------------------------
    # Small graph builders.  Graphs use parallel lists to stay dependency-free:
    # atoms are element strings; bonds are mutable [left, right, order] triples.

    @staticmethod
    def _new_graph(elements=()):
        return {"atoms": list(elements), "bonds": []}

    @staticmethod
    def _add_atom(graph, element):
        graph["atoms"].append(element)
        return len(graph["atoms"]) - 1

    def _add_bond(self, graph, left, right, order=1):
        if isinstance(order, bool) or order not in (1, 2, 3):
            raise ValueError("bond order must be 1, 2, or 3")
        atom_count = len(graph["atoms"])
        if not 0 <= left < atom_count or not 0 <= right < atom_count:
            raise ValueError("bond endpoint is outside the atom list")
        if left == right:
            raise ValueError("self-bonds are not allowed")
        if any({left, right} == {a, b} for a, b, _ in graph["bonds"]):
            raise ValueError("duplicate bond is not allowed")
        if self._remaining_valence(graph, left) < order:
            raise ValueError(f"atom {left} has insufficient valence")
        if self._remaining_valence(graph, right) < order:
            raise ValueError(f"atom {right} has insufficient valence")
        graph["bonds"].append([left, right, order])

    def _remaining_valence(self, graph, atom_id):
        element = graph["atoms"][atom_id]
        if element not in MAX_VALENCE:
            raise ValueError(f"unsupported generated element: {element}")
        used = sum(
            order
            for left, right, order in graph["bonds"]
            if left == atom_id or right == atom_id
        )
        return MAX_VALENCE[element] - used

    def _carbon_chain_graph(self, chain_length):
        graph = self._new_graph(["C"] * chain_length)
        for atom_id in range(chain_length - 1):
            self._add_bond(graph, atom_id, atom_id + 1)
        return graph

    def _carbon_ring_graph(self, ring_size):
        graph = self._carbon_chain_graph(ring_size)
        self._add_bond(graph, ring_size - 1, 0)
        return graph

    def _oxygen_chain_graph(self, chain_length):
        graph = self._new_graph(["O"] * chain_length)
        for atom_id in range(chain_length - 1):
            self._add_bond(graph, atom_id, atom_id + 1)
        return graph

    def _branched_carbon_graph(
        self,
        main_chain_length,
        branch_count,
        double_count,
        triple_count,
        search_budget,
    ):
        """Return the most oxygen-capable valid branched skeleton.

        Branch positions and multiple bonds affect how many carbonyls fit even
        when their aggregate counts are the same. A dynamic program over the main
        chain chooses both together.  Its score first maximizes available
        carbonyl sites and then spreads branches across as many sites as
        possible.  Equal solutions are selected with the instance RNG.
        """
        # State: branches, doubles, triples, left promotion, is branched.
        # Value: ((carbonyl slots, occupied branch sites), decisions so far)
        states = {(0, 0, 0, 0, False): ((0, 0), ())}

        for atom_id in range(main_chain_length):
            chain_degree = 1 if atom_id in (0, main_chain_length - 1) else 2
            right_choices = (
                (0,) if atom_id == main_chain_length - 1 else (0, 1, 2)
            )
            next_states = {}

            for state, (score, decisions) in states.items():
                (
                    branches_used,
                    doubles_used,
                    triples_used,
                    left_promotion,
                    already_branched,
                ) = state
                branch_limit = min(
                    4 - chain_degree,
                    branch_count - branches_used,
                )
                for branches_here in range(branch_limit + 1):
                    free_valence = 4 - chain_degree - branches_here
                    is_branched = already_branched or (
                        chain_degree + branches_here >= 3
                    )
                    for right_promotion in right_choices:
                        for branch_doubles in range(branches_here + 1):
                            remaining_branches = branches_here - branch_doubles
                            for branch_triples in range(remaining_branches + 1):
                                search_budget["remaining"] -= 1
                                if search_budget["remaining"] < 0:
                                    return None
                                promoted_at_atom = (
                                    left_promotion
                                    + right_promotion
                                    + branch_doubles
                                    + 2 * branch_triples
                                )
                                if promoted_at_atom > free_valence:
                                    continue

                                new_branch_count = branches_used + branches_here
                                new_double_count = doubles_used + (
                                    right_promotion == 1
                                ) + branch_doubles
                                new_triple_count = triples_used + (
                                    right_promotion == 2
                                ) + branch_triples
                                if new_double_count > double_count:
                                    continue
                                if new_triple_count > triple_count:
                                    continue

                                # A double-bonded branch retains one carbonyl
                                # slot; a triple-bonded branch retains none.
                                carbonyl_slots = (
                                    free_valence - promoted_at_atom
                                ) // 2 + branches_here - branch_triples
                                new_score = (
                                    score[0] + carbonyl_slots,
                                    score[1] + bool(branches_here),
                                )
                                new_state = (
                                    new_branch_count,
                                    new_double_count,
                                    new_triple_count,
                                    right_promotion,
                                    is_branched,
                                )
                                new_decisions = decisions + (
                                    (
                                        branches_here,
                                        right_promotion,
                                        branch_doubles,
                                        branch_triples,
                                    ),
                                )

                                old = next_states.get(new_state)
                                if old is None or new_score > old[0]:
                                    next_states[new_state] = (
                                        new_score,
                                        new_decisions,
                                    )
                                elif (
                                    new_score == old[0] and self._rng.randrange(2)
                                ):
                                    next_states[new_state] = (
                                        new_score,
                                        new_decisions,
                                    )
            states = next_states

        final = states.get(
            (branch_count, double_count, triple_count, 0, True)
        )
        if final is None:
            return None

        _, decisions = final
        graph = self._carbon_chain_graph(main_chain_length)
        branch_edges = [[] for _ in range(main_chain_length)]
        positions = []

        for atom_id, (branches_here, _, _, _) in enumerate(decisions):
            for _ in range(branches_here):
                branch_id = self._add_atom(graph, "C")
                self._add_bond(graph, atom_id, branch_id)
                branch_edges[atom_id].append(len(graph["bonds"]) - 1)
                positions.append(atom_id)

        for atom_id, decision in enumerate(decisions):
            _, right_promotion, branch_doubles, branch_triples = decision
            if right_promotion:
                self._set_bond_orders(
                    graph,
                    [atom_id],
                    order=right_promotion + 1,
                )
            if branch_doubles:
                candidates = list(branch_edges[atom_id])
                self._rng.shuffle(candidates)
                double_edges = candidates[:branch_doubles]
                self._set_bond_orders(graph, double_edges, order=2)
            else:
                candidates = list(branch_edges[atom_id])
                self._rng.shuffle(candidates)
            if branch_triples:
                triple_start = branch_doubles
                triple_edges = candidates[
                    triple_start : triple_start + branch_triples
                ]
                self._set_bond_orders(graph, triple_edges, order=3)

        return graph, positions

    @staticmethod
    def _copy_graph(graph):
        return {
            "atoms": list(graph["atoms"]),
            "bonds": [list(bond) for bond in graph["bonds"]],
        }

    def _set_bond_orders(self, graph, bond_indices, order):
        """Promote known single bonds to the requested multiple-bond order."""
        if order not in (2, 3):
            raise ValueError("promoted bond order must be 2 or 3")
        extra_valence = order - 1
        for bond_index in bond_indices:
            left, right, current_order = graph["bonds"][bond_index]
            if current_order != 1:
                raise ValueError("only single bonds can be promoted")
            if self._remaining_valence(graph, left) < extra_valence:
                raise ValueError(f"atom {left} has insufficient valence")
            if self._remaining_valence(graph, right) < extra_valence:
                raise ValueError(f"atom {right} has insufficient valence")
            graph["bonds"][bond_index][2] = order

    def _oxygen_capacity_is_sufficient(
        self, graph, carbonyl_count, hydroxyl_count
    ):
        capacities = [
            self._remaining_valence(graph, atom_id)
            for atom_id, element in enumerate(graph["atoms"])
            if element == "C"
        ]
        return (
            sum(capacity // 2 for capacity in capacities) >= carbonyl_count
            and sum(capacities) >= 2 * carbonyl_count + hydroxyl_count
        )

    def _promote_chain_bonds(
        self,
        graph,
        double_count,
        triple_count,
        carbonyl_count=0,
        hydroxyl_count=0,
    ):
        """Place exact C=C/C#C counts while maximizing oxygen capacity."""
        atom_count = len(graph["atoms"])
        # State: doubles used, triples used, promotion from the left edge.
        states = {(0, 0, 0): (0, ())}
        for atom_id in range(atom_count):
            free_valence = 3 if atom_id in (0, atom_count - 1) else 2
            right_choices = (0,) if atom_id == atom_count - 1 else (0, 1, 2)
            next_states = {}
            for state, (score, decisions) in states.items():
                doubles_used, triples_used, left_promotion = state
                for right_promotion in right_choices:
                    if left_promotion + right_promotion > free_valence:
                        continue
                    new_doubles = doubles_used + (right_promotion == 1)
                    new_triples = triples_used + (right_promotion == 2)
                    if new_doubles > double_count or new_triples > triple_count:
                        continue
                    new_state = (new_doubles, new_triples, right_promotion)
                    new_score = score + (
                        free_valence - left_promotion - right_promotion
                    ) // 2
                    new_decisions = decisions + (right_promotion,)
                    old = next_states.get(new_state)
                    if old is None or new_score > old[0]:
                        next_states[new_state] = (new_score, new_decisions)
                    elif new_score == old[0] and self._rng.randrange(2):
                        next_states[new_state] = (new_score, new_decisions)
            states = next_states

        final = states.get((double_count, triple_count, 0))
        if final is None:
            return False
        _, decisions = final
        for bond_index, promotion in enumerate(decisions[:-1]):
            if promotion:
                self._set_bond_orders(
                    graph,
                    [bond_index],
                    order=promotion + 1,
                )
        return self._oxygen_capacity_is_sufficient(
            graph, carbonyl_count, hydroxyl_count
        )

    def _promote_ring_bonds(
        self,
        graph,
        double_count,
        triple_count,
        carbonyl_count=0,
        hydroxyl_count=0,
    ):
        """Place pairwise non-adjacent ring C=C and C#C bonds."""
        ring_size = len(graph["atoms"])
        multiple_bond_count = double_count + triple_count
        if multiple_bond_count > ring_size // 2:
            return False
        offset = self._rng.randrange(ring_size)
        bond_indices = [
            (offset + 2 * index) % ring_size
            for index in range(multiple_bond_count)
        ]
        orders = [2] * double_count + [3] * triple_count
        self._rng.shuffle(orders)
        for bond_index, order in zip(bond_indices, orders):
            self._set_bond_orders(graph, [bond_index], order=order)
        return self._oxygen_capacity_is_sufficient(
            graph, carbonyl_count, hydroxyl_count
        )

    def _promote_oxygen_chain_bonds(self, graph, count):
        """Only a two-oxygen chain can contain a valence-valid O=O bond."""
        if count == 0:
            return True
        if len(graph["atoms"]) != 2 or count != 1:
            return False
        self._set_bond_orders(graph, [0], order=2)
        return True

    def _attach_oxygen_motifs(
        self, graph, carbonyl_count, hydroxyl_count
    ):
        """Attach C=O first, then C-O; return the chosen carbon positions."""
        carbon_ids = [
            atom_id
            for atom_id, element in enumerate(graph["atoms"])
            if element == "C"
        ]
        capacities = {
            atom_id: self._remaining_valence(graph, atom_id)
            for atom_id in carbon_ids
        }

        if sum(capacity // 2 for capacity in capacities.values()) < carbonyl_count:
            return None
        if sum(capacities.values()) < 2 * carbonyl_count + hydroxyl_count:
            return None

        carbonyl_positions = []
        for _ in range(carbonyl_count):
            candidates = [
                atom_id for atom_id, capacity in capacities.items() if capacity >= 2
            ]
            position = self._rng.choice(candidates)
            capacities[position] -= 2
            carbonyl_positions.append(position)

        hydroxyl_positions = []
        for _ in range(hydroxyl_count):
            candidates = [
                atom_id for atom_id, capacity in capacities.items() if capacity >= 1
            ]
            if not candidates:
                return None
            position = self._rng.choice(candidates)
            capacities[position] -= 1
            hydroxyl_positions.append(position)

        for position in carbonyl_positions:
            oxygen_id = self._add_atom(graph, "O")
            self._add_bond(graph, position, oxygen_id, order=2)
        for position in hydroxyl_positions:
            oxygen_id = self._add_atom(graph, "O")
            self._add_bond(graph, position, oxygen_id, order=1)

        return {
            "carbonyl_positions": carbonyl_positions,
            "hydroxyl_positions": hydroxyl_positions,
        }

    def _saturate_with_hydrogen(self, graph):
        # Iterate only over heavy atoms.  Newly appended H atoms are already full.
        heavy_atom_count = len(graph["atoms"])
        for atom_id in range(heavy_atom_count):
            for _ in range(self._remaining_valence(graph, atom_id)):
                hydrogen_id = self._add_atom(graph, "H")
                self._add_bond(graph, atom_id, hydrogen_id)

    # ------------------------------------------------------------------
    # Validation, reduction, and insertion

    def _validate_graph(self, graph, require_saturated=True):
        if not isinstance(graph, dict):
            raise ValueError("graph must be an object")
        atoms = graph.get("atoms")
        bonds = graph.get("bonds")
        if not isinstance(atoms, list) or not atoms:
            raise ValueError("graph must contain at least one atom")
        if not isinstance(bonds, list):
            raise ValueError("graph bonds must be a list")
        if any(element not in MAX_VALENCE for element in atoms):
            raise ValueError("graph contains an unsupported element")

        valence = [0] * len(atoms)
        adjacency = [set() for _ in atoms]
        seen_edges = set()
        for bond in bonds:
            if not isinstance(bond, (list, tuple)) or len(bond) != 3:
                raise ValueError("each bond must be [left, right, order]")
            left, right, order = bond
            if (
                isinstance(left, bool)
                or not isinstance(left, int)
                or isinstance(right, bool)
                or not isinstance(right, int)
            ):
                raise ValueError("bond endpoints must be integers")
            if not 0 <= left < len(atoms) or not 0 <= right < len(atoms):
                raise ValueError("bond endpoint is outside the atom list")
            if left == right:
                raise ValueError("self-bonds are not allowed")
            if isinstance(order, bool) or order not in (1, 2, 3):
                raise ValueError("bond order must be 1, 2, or 3")

            edge = tuple(sorted((left, right)))
            if edge in seen_edges:
                raise ValueError("duplicate bond is not allowed")
            seen_edges.add(edge)
            if "H" in (atoms[left], atoms[right]) and order != 1:
                raise ValueError("hydrogen may only form single bonds")

            valence[left] += order
            valence[right] += order
            adjacency[left].add(right)
            adjacency[right].add(left)

        for atom_id, element in enumerate(atoms):
            if valence[atom_id] > MAX_VALENCE[element]:
                raise ValueError(f"atom {atom_id} exceeds {element} valence")
            if require_saturated and valence[atom_id] != MAX_VALENCE[element]:
                raise ValueError(f"atom {atom_id} has unfilled {element} valence")

        visited = {0}
        pending = [0]
        while pending:
            atom_id = pending.pop()
            for neighbor in adjacency[atom_id] - visited:
                visited.add(neighbor)
                pending.append(neighbor)
        if len(visited) != len(atoms):
            raise ValueError("graph must be connected")
        return True

    @staticmethod
    def _graph_has_cycle(graph):
        adjacency = [set() for _ in graph["atoms"]]
        for left, right, _ in graph["bonds"]:
            adjacency[left].add(right)
            adjacency[right].add(left)

        visited = set()
        for start in range(len(adjacency)):
            if start in visited:
                continue
            visited.add(start)
            pending = [(start, -1)]
            while pending:
                atom_id, parent = pending.pop()
                for neighbor in adjacency[atom_id]:
                    if neighbor == parent:
                        continue
                    if neighbor in visited:
                        return True
                    visited.add(neighbor)
                    pending.append((neighbor, atom_id))
        return False

    @staticmethod
    def _formula_from_graph(graph):
        counts = Counter(graph["atoms"])
        elements = []
        if "C" in counts:
            elements.append("C")
        if "H" in counts:
            elements.append("H")
        elements.extend(
            sorted(element for element in counts if element not in {"C", "H"})
        )
        return "".join(
            element + (str(counts[element]) if counts[element] != 1 else "")
            for element in elements
        )

    def _bond_counts_from_graph(self, graph):
        counts = {key: 0 for key in self.bond_keys}
        for left, right, order in graph["bonds"]:
            elements = tuple(sorted((graph["atoms"][left], graph["atoms"][right])))
            key = self._bond_key_lookup.get((elements, order))
            if key is None:
                raise ValueError(
                    f"no bond feature for {elements[0]}-{elements[1]} order {order}"
                )
            counts[key] += 1
        return counts

    def _canonical_signature(self, formula, bond_counts, is_ring):
        return (
            formula,
            tuple(bond_counts.get(key, 0) for key in self.bond_keys),
            is_ring,
        )

    @staticmethod
    def _serialize_graph(graph):
        bonds = [
            [min(left, right), max(left, right), order]
            for left, right, order in graph["bonds"]
        ]
        bonds.sort()
        return {"atoms": list(graph["atoms"]), "bonds": bonds}

    def _try_add_sample(self, graph, generator, parameters, is_ring, notes):
        self._statistics["attempted"] += 1
        try:
            self._validate_graph(graph)
            actual_is_ring = self._graph_has_cycle(graph)
            if actual_is_ring != is_ring:
                raise ValueError("declared ring flag does not match graph topology")
            formula = self._formula_from_graph(graph)
            bond_counts = self._bond_counts_from_graph(graph)
        except ValueError:
            self._statistics["rejected_invalid"] += 1
            return None

        signature = self._canonical_signature(formula, bond_counts, is_ring)
        if signature in self._generated_signatures:
            self._statistics["rejected_duplicate"] += 1
            return None
        if signature in self._observed_signatures:
            self._statistics["rejected_observed"] += 1
            return None

        sample_number = len(self.generated_samples) + 1
        sample = {
            "sample_id": f"pseudo_{formula}_{sample_number:06d}",
            "formula": formula,
            "bond_counts": bond_counts,
            "is_ring": is_ring,
            "generator": generator,
            "parameters": dict(parameters),
            "is_pseudo_negative": True,
            "confidence": "synthetic",
            "notes": list(notes),
            # The model consumes aggregate counts.  Topology is retained so the
            # valence claims can be audited and placements can be reproduced.
            "topology": self._serialize_graph(graph),
        }
        self.generated_samples.append(sample)
        self._generated_signatures.add(signature)
        self._statistics["accepted"] += 1
        return sample

    @staticmethod
    def _motif_note(
        double_count,
        triple_count,
        carbonyl_count,
        hydroxyl_count,
        placements,
    ):
        return (
            f"{double_count} C=C, {triple_count} C#C, "
            f"{carbonyl_count} carbonyl, and {hydroxyl_count} hydroxyl "
            "motif(s); "
            f"carbonyl positions {placements['carbonyl_positions']}, "
            f"hydroxyl positions {placements['hydroxyl_positions']}"
        )


# Conventional alias for callers that prefer PEP 8 class naming.
PseudoNegativeGenerator = pseudo_negative_generator


__all__ = ["PseudoNegativeGenerator", "pseudo_negative_generator"]
