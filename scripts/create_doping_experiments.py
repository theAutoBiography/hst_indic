#!/usr/bin/env python3
"""
Create mixed datasets for transfer learning "doping" experiments

Tests how much Malayalam/Telugu data improves Sanskrit sandhi splitting
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Tuple


def load_dataset(file_path: str) -> List[Dict]:
    """Load dataset from JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_mixed_dataset(
    sanskrit_data: List[Dict],
    external_data: List[Dict],
    external_ratio: float,
    external_lang: str
) -> List[Dict]:
    """
    Create mixed dataset with specified ratio of external data

    Args:
        sanskrit_data: Sanskrit sandhi examples
        external_data: Malayalam or Telugu sandhi examples
        external_ratio: Ratio of external data (0.0 to 1.0)
        external_lang: Language name for metadata

    Returns:
        Mixed dataset
    """
    total_size = len(sanskrit_data)
    n_external = int(total_size * external_ratio / (1 - external_ratio))
    n_sanskrit = total_size

    print(f"  Sanskrit: {n_sanskrit:,} examples")
    print(f"  {external_lang}: {n_external:,} examples")
    print(f"  Ratio: {external_ratio*100:.1f}% {external_lang}")

    # Sample external data
    if n_external > len(external_data):
        print(f"  Warning: Requested {n_external:,} but only {len(external_data):,} available")
        n_external = len(external_data)

    external_sample = random.sample(external_data, n_external)

    # Add language tags
    for ex in external_sample:
        ex['language'] = external_lang

    for ex in sanskrit_data:
        ex['language'] = 'sanskrit'

    # Combine and shuffle
    mixed_data = sanskrit_data + external_sample
    random.shuffle(mixed_data)

    return mixed_data


def split_dataset(data: List[Dict], train_ratio=0.85, val_ratio=0.10) -> Tuple[List, List, List]:
    """Split dataset into train/val/test"""
    n = len(data)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train = data[:n_train]
    val = data[n_train:n_train + n_val]
    test = data[n_train + n_val:]

    return train, val, test


def save_dataset(data: List[Dict], output_file: str):
    """Save dataset to JSON"""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"  ✓ Saved to {output_file}")


def create_experiment(
    sanskrit_data: List[Dict],
    external_data: List[Dict],
    external_lang: str,
    external_ratio: float,
    output_dir: Path
):
    """Create one doping experiment with specified ratio"""

    print(f"\nCreating experiment: {external_ratio*100:.0f}% {external_lang}")
    print("-" * 60)

    # Create mixed dataset
    mixed_data = create_mixed_dataset(
        sanskrit_data.copy(),
        external_data,
        external_ratio,
        external_lang
    )

    # Split into train/val/test
    train, val, test = split_dataset(mixed_data)

    print(f"  Train: {len(train):,}")
    print(f"  Val: {len(val):,}")
    print(f"  Test: {len(test):,}")

    # Save
    exp_dir = output_dir / f"{external_lang}_doping_{int(external_ratio*100):02d}pct"
    exp_dir.mkdir(parents=True, exist_ok=True)

    save_dataset(train, str(exp_dir / "train.json"))
    save_dataset(val, str(exp_dir / "val.json"))
    save_dataset(test, str(exp_dir / "test.json"))

    # Save metadata
    metadata = {
        'external_language': external_lang,
        'external_ratio': external_ratio,
        'total_examples': len(mixed_data),
        'train_examples': len(train),
        'val_examples': len(val),
        'test_examples': len(test),
        'sanskrit_examples': len([ex for ex in mixed_data if ex['language'] == 'sanskrit']),
        'external_examples': len([ex for ex in mixed_data if ex['language'] == external_lang]),
    }

    with open(exp_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)


def main():
    """Create all doping experiments"""

    print("="*60)
    print("CREATING TRANSFER LEARNING DOPING EXPERIMENTS")
    print("="*60)

    random.seed(42)

    # Load datasets
    print("\nLoading datasets...")
    sanskrit_data = load_dataset("data/processed/dcs_sandhi_splits.json")
    print(f"  Sanskrit: {len(sanskrit_data):,} examples")

    malayalam_path = Path("data/processed/malayalam_sandhi_iithlp.json")
    telugu_path = Path("data/processed/telugu_sandhi_iithlp.json")

    malayalam_data = None
    telugu_data = None

    if malayalam_path.exists():
        malayalam_data = load_dataset(str(malayalam_path))
        print(f"  Malayalam: {len(malayalam_data):,} examples")

    if telugu_path.exists():
        telugu_data = load_dataset(str(telugu_path))
        print(f"  Telugu: {len(telugu_data):,} examples")

    # Doping ratios to test
    doping_ratios = [0.05, 0.10, 0.20, 0.30, 0.50]

    output_dir = Path("data/experiments/doping")

    # Create baseline (0% external)
    print("\n" + "="*60)
    print("BASELINE: 0% External Data (Pure Sanskrit)")
    print("="*60)

    baseline_dir = output_dir / "baseline_00pct"
    baseline_dir.mkdir(parents=True, exist_ok=True)

    # Add language tags
    for ex in sanskrit_data:
        ex['language'] = 'sanskrit'

    train, val, test = split_dataset(sanskrit_data.copy())
    save_dataset(train, str(baseline_dir / "train.json"))
    save_dataset(val, str(baseline_dir / "val.json"))
    save_dataset(test, str(baseline_dir / "test.json"))

    # Malayalam experiments
    if malayalam_data:
        print("\n" + "="*60)
        print("MALAYALAM DOPING EXPERIMENTS")
        print("="*60)

        for ratio in doping_ratios:
            create_experiment(
                sanskrit_data,
                malayalam_data,
                'malayalam',
                ratio,
                output_dir
            )

    # Telugu experiments
    if telugu_data:
        print("\n" + "="*60)
        print("TELUGU DOPING EXPERIMENTS")
        print("="*60)

        for ratio in doping_ratios:
            create_experiment(
                sanskrit_data,
                telugu_data,
                'telugu',
                ratio,
                output_dir
            )

    # Create summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)

    experiments = sorted(output_dir.glob("*/metadata.json"))

    print(f"\nCreated {len(experiments)} experiments:")
    print("\n{:<30} {:<10} {:<15} {:<15}".format(
        "Experiment", "Ratio", "Total", "External"))
    print("-" * 70)

    for exp_meta_path in experiments:
        with open(exp_meta_path) as f:
            meta = json.load(f)

        exp_name = exp_meta_path.parent.name
        ratio = meta.get('external_ratio', 0.0)
        total = meta.get('total_examples', 0)
        external = meta.get('external_examples', 0)

        print(f"{exp_name:<30} {ratio*100:>5.0f}%     {total:>10,}     {external:>10,}")

    print("\n" + "="*60)
    print("EXPERIMENTS READY!")
    print("="*60)
    print(f"\nLocation: {output_dir}")
    print("\nNext steps:")
    print("1. Run training on each experiment")
    print("2. Compare accuracy vs. doping ratio")
    print("3. Find optimal transfer learning ratio")


if __name__ == "__main__":
    main()
