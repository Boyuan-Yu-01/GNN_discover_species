"""Generate the default pseudo-negative dataset."""

from pathlib import Path

from pseudo_negative_generator import pseudo_negative_generator

script_dir = Path(__file__).resolve().parent
output_path = script_dir / "output" / "pseudo_negative_dataset.json"
log_path = script_dir / "output" / "log_pseudo_negative_dataset.txt"
reference_paths = [
    script_dir / "output" / "c8-c16_n-alkanes_LLNL_subgraph_info.json",
    script_dir / "output" / "nHeptane_LLNL_subgraph_info.json",
]

generator = pseudo_negative_generator(
    n=100000,   # unrealistically high number to ensure we get enough samples
    reference_paths=reference_paths,
    random_seed=42,
)

generated_counts = {
    "long carbon chains 1": len(
        generator.long_carbon_chain(21, 3, 2, triple_bond_seed=1)
    ),
    "long carbon chains 2": len(
        generator.long_carbon_chain(22, 3, 2, triple_bond_seed=1)
    ),
    "long carbon chains 3": len(
        generator.long_carbon_chain(23, 3, 2, triple_bond_seed=1)
    ),
    "long carbon chains 4": len(
        generator.long_carbon_chain(24, 3, 2, triple_bond_seed=1)
    ),
    "long carbon chains 5": len(
        generator.long_carbon_chain(25, 3, 2, triple_bond_seed=1)
    ),
    "long carbon chains 6": len(
        generator.long_carbon_chain(30, 3, 2, triple_bond_seed=1)
    ),
    "carbon rings 1": len(
        generator.carbon_ring(10, 3, 2, triple_bond_seed=1)
    ),
    "carbon rings 2": len(
        generator.carbon_ring(12, 3, 2, triple_bond_seed=1)
    ),
    "branched carbons 1": len(
        generator.highly_branched_carbon(
            0.1, 2, 2, main_chain_length=15, triple_bond_seed=1
        )
    ),
    "branched carbons 2": len(
        generator.highly_branched_carbon(
            0.2, 2, 2, main_chain_length=15, triple_bond_seed=1
        )
    ),
    "branched carbons 3": len(
        generator.highly_branched_carbon(
            0.3, 2, 2, main_chain_length=15, triple_bond_seed=1
        )
    ),
    "branched carbons 4": len(
        generator.highly_branched_carbon(
            0.4, 2, 2, main_chain_length=15, triple_bond_seed=1
        )
    ),
    "branched carbons 5": len(
        generator.highly_branched_carbon(
            0.5, 2, 2, main_chain_length=15, triple_bond_seed=1
        )
    ),
    "branched carbons 6": len(
        generator.highly_branched_carbon(
            0.1, 2, 2, main_chain_length=20, triple_bond_seed=1
        )
    ),
    "branched carbons 7": len(
        generator.highly_branched_carbon(
            0.2, 2, 2, main_chain_length=20, triple_bond_seed=1
        )
    ),
    "branched carbons 8": len(
        generator.highly_branched_carbon(
            0.3, 2, 2, main_chain_length=20, triple_bond_seed=1
        )
    ),
    "branched carbons 9": len(
        generator.highly_branched_carbon(
            0.4, 2, 2, main_chain_length=20, triple_bond_seed=1
        )
    ),
    "branched carbons 10": len(
        generator.highly_branched_carbon(
            0.5, 2, 2, main_chain_length=20, triple_bond_seed=1
        )
    ),
    "long oxygen chains 1": len(generator.long_oxygen_chain(3, 2)),
    "long oxygen chains 2": len(generator.long_oxygen_chain(5, 2)),
    "long oxygen chains 3": len(generator.long_oxygen_chain(6, 2)),
    "long oxygen chains 4": len(generator.long_oxygen_chain(8, 2)),
    "long oxygen chains 5": len(generator.long_oxygen_chain(10, 2)),
}

output_path = generator.export_to_json(output_path)
log_path = generator.write_generation_log(output_path, log_path)

for family, count in generated_counts.items():
    print(f"{family}: {count}")
print(f"total: {len(generator.generated_samples)}")
print(f"output: {output_path}")
print(f"log: {log_path}")
