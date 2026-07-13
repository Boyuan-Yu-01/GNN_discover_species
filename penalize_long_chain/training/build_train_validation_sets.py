"""Create three balanced training sets and one shared validation set."""

import json
import random
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).parent
DATA = HERE / "org_dataset"
OUTPUT = HERE / "dataSets"
POSITIVE_FILE = DATA / "filtered_master.json"
NEGATIVE_FILE = DATA / "filtered_pseudo_negative.json"

TRAIN_COUNT = 214
SPLIT_SEED = 20260712
TRAIN_SEEDS = (101, 202, 303)


def load(path):
    """Load the formula-degeneracy records from one dataset."""
    with path.open(encoding="utf-8") as file:
        return json.load(file)["formula_degeneracies"]


def formula(sample_id):
    """Convert an ID such as C10H20_002 to C10H20."""
    return sample_id.rsplit("_", 1)[0]


def group_by_formula(samples):
    """Keep degeneracies of the same formula in one group."""
    # This lets the split operate at the formula level instead of the record level.
    groups = defaultdict(list)
    for sample_id in samples:
        groups[formula(sample_id)].append(sample_id)
    return groups


def select_formulae(groups, target, seed):
    """Select complete formula groups containing exactly target samples."""
    items = list(groups.items())
    random.Random(seed).shuffle(items)
    choices = {0: set()}

    # Subset-sum selection prevents a formula from being split across datasets.
    # Each group is either fully included or fully excluded.
    for name, sample_ids in items:
        for total in sorted(choices, reverse=True):
            new_total = total + len(sample_ids)
            if new_total <= target and new_total not in choices:
                choices[new_total] = choices[total] | {name}

    if target not in choices:
        raise ValueError(f"Cannot select exactly {target} grouped samples")
    return choices[target]


def records(samples, sample_ids, label, class_name):
    """Add explicit labels to selected samples."""
    return {
        sample_id: {
            "degeneracy_id": sample_id,
            "formula": formula(sample_id),
            **samples[sample_id],
            "label": label,
            "class_name": class_name,
        }
        for sample_id in sorted(sample_ids)
    }


def main():
    positive = load(POSITIVE_FILE)
    negative = load(NEGATIVE_FILE)
    positive_groups = group_by_formula(positive)
    negative_groups = group_by_formula(negative)

    # Hold out 53 of 267 positives, leaving exactly 214 for training.
    # The seed keeps the split reproducible.
    validation_formulae = select_formulae(
        positive_groups, len(positive) - TRAIN_COUNT, SPLIT_SEED
    )

    # Hold out 20% of negatives. Shared positive/negative formulae stay together.
    # First account for negatives already reserved by the positive validation formulas.
    forced_negatives = sum(
        len(negative_groups.get(name, [])) for name in validation_formulae
    )
    # Only formulas unique to the negative set are eligible for the remaining validation quota.
    negative_only = {
        name: ids
        for name, ids in negative_groups.items()
        if name not in positive_groups
    }
    validation_formulae |= select_formulae(
        negative_only, round(0.2 * len(negative)) - forced_negatives, SPLIT_SEED + 1
    )

    positive_train = [x for x in positive if formula(x) not in validation_formulae]
    # Sample training negatives only from formulas not reserved for validation.
    negative_pool = [x for x in negative if formula(x) not in validation_formulae]
    positive_valid = [x for x in positive if formula(x) in validation_formulae]
    negative_valid = [x for x in negative if formula(x) in validation_formulae]

    # All three files use the same validation set for a fair comparison.
    validation = {
        "positive_samples": records(positive, positive_valid, 1, "positive"),
        "pseudo_negative_samples": records(
            negative, negative_valid, 0, "pseudo_negative"
        ),
    }

    OUTPUT.mkdir(exist_ok=True)
    for number, seed in enumerate(TRAIN_SEEDS, 1):
        # Draw a different negative training sample for each of the three runs.
        negative_train = random.Random(seed).sample(negative_pool, TRAIN_COUNT)
        training = {
            "positive_samples": records(positive, positive_train, 1, "positive"),
            "pseudo_negative_samples": records(
                negative, negative_train, 0, "pseudo_negative"
            ),
        }

        # Write one self-contained JSON file per train/validation split.
        path = OUTPUT / f"train_validation_{number}.json"
        with path.open("w", encoding="utf-8") as file:
            json.dump({"training_set": training, "validation_set": validation}, file, indent=2)
            file.write("\n")

        print(
            f"Wrote {path}: "
            f"training={len(training['positive_samples'])} positive + "
            f"{len(training['pseudo_negative_samples'])} pseudo-negative; "
            f"validation={len(validation['positive_samples'])} positive + "
            f"{len(validation['pseudo_negative_samples'])} pseudo-negative"
        )


if __name__ == "__main__":
    main()
