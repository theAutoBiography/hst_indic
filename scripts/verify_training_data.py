#!/usr/bin/env python3
"""
Verify training data is ready for ByT5 training
"""

import json
from pathlib import Path


def check_dcs_data():
    """Check DCS sandhi splits"""
    data_path = Path("data/processed/dcs_sandhi_splits.json")

    if not data_path.exists():
        print(f"✗ {data_path} not found")
        print()
        print("Generate it with:")
        print("  python scripts/extract_from_dcs_conllu.py")
        print()
        return False

    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not isinstance(data, list):
            print(f"✗ DCS data is not a list")
            return False

        if len(data) == 0:
            print(f"✗ DCS data is empty")
            return False

        # Check format
        sample = data[0]
        if 'surface_form' not in sample or 'split_words' not in sample:
            print(f"✗ DCS data has wrong format")
            print(f"Expected: {{'surface_form': str, 'split_words': [str]}}")
            print(f"Got: {sample.keys()}")
            return False

        print(f"✓ DCS data looks good!")
        print(f"  Total: {len(data):,} examples")
        print()

        return True

    except Exception as e:
        print(f"✗ Error reading DCS data: {e}")
        return False


def check_sandhiset_data():
    """Check sandhiset annotations"""
    data_path = Path("data/annotations/sandhiset.txt")

    if not data_path.exists():
        print(f"⚠ {data_path} not found (optional)")
        print()
        return True  # Optional data

    try:
        count = 0
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and '=>' in line:
                    count += 1

        if count == 0:
            print(f"⚠ Sandhiset data is empty")
            print()
            return True  # Optional

        print(f"✓ Sandhiset data looks good!")
        print(f"  Total: {count:,} examples")
        print()

        return True

    except Exception as e:
        print(f"⚠ Error reading sandhiset data: {e}")
        print()
        return True  # Optional


def main():
    print("=" * 60)
    print("VERIFY TRAINING DATA")
    print("=" * 60)
    print()

    dcs_ok = check_dcs_data()
    sandhiset_ok = check_sandhiset_data()

    print("=" * 60)

    if dcs_ok and sandhiset_ok:
        print("✅ All data checks passed!")
        print()
        print("Ready to train:")
        print("  python scripts/train_byt5_sandhi.py")
        print()
        return 0
    else:
        print("❌ Data checks failed!")
        print()
        print("Fix the issues above before training")
        print()
        return 1


if __name__ == "__main__":
    exit(main())
