#!/usr/bin/env python3
"""
Extract sandhi splits from DCS CoNLL-U corpus
Produces ~625k gold-standard training examples
"""

import json
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple, Dict
from tqdm import tqdm


def parse_conllu_file(file_path: Path) -> List[Tuple[str, List[str]]]:
    """
    Parse CoNLL-U file and extract sandhi splits from multi-word tokens

    CoNLL-U format for multi-word tokens:
    1-2    compound_form    _    _    ...
    1      word1            _    _    ...
    2      word2            _    _    ...
    """
    examples = []

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Skip comments and empty lines
        if not line or line.startswith('#'):
            i += 1
            continue

        # Check for multi-word token (e.g., "1-2")
        if '-' in line.split('\t')[0]:
            parts = line.split('\t')
            if len(parts) < 2:
                i += 1
                continue

            # Get range (e.g., "1-2" => start=1, end=2)
            range_str = parts[0]
            try:
                start, end = map(int, range_str.split('-'))
            except:
                i += 1
                continue

            # Surface form (compound)
            surface_form = parts[1]

            # Collect the individual words
            split_words = []
            i += 1

            for j in range(start, end + 1):
                if i >= len(lines):
                    break

                word_line = lines[i].strip()
                if not word_line or word_line.startswith('#'):
                    break

                word_parts = word_line.split('\t')
                if len(word_parts) < 2:
                    break

                # Check if this is the expected token number
                try:
                    token_id = int(word_parts[0])
                    if token_id != j:
                        break
                except:
                    break

                word = word_parts[1]
                split_words.append(word)
                i += 1

            # Validate
            if len(split_words) == (end - start + 1) and len(split_words) >= 2:
                # Filter out very short or very long compounds
                if 3 <= len(surface_form) <= 50:
                    examples.append((surface_form, split_words))
        else:
            i += 1

    return examples


def extract_all_dcs_data(dcs_path: Path) -> List[Dict]:
    """Extract all sandhi examples from DCS corpus"""
    conllu_dir = dcs_path / "conllu" / "files"

    if not conllu_dir.exists():
        print(f"ERROR: {conllu_dir} not found")
        print("Make sure DCS corpus is downloaded to data/corpora/dcs/")
        return []

    # Find all .conllu files
    conllu_files = list(conllu_dir.rglob("*.conllu"))
    print(f"Found {len(conllu_files)} CoNLL-U files")

    if not conllu_files:
        print("ERROR: No .conllu files found")
        return []

    # Extract from all files
    all_examples = defaultdict(lambda: {"split_words": [], "count": 0, "sources": []})

    for file_path in tqdm(conllu_files, desc="Processing files"):
        try:
            examples = parse_conllu_file(file_path)

            for surface, split_words in examples:
                key = surface.lower()
                split_key = tuple(split_words)

                # Track this example
                all_examples[key]["split_words"].append(split_key)
                all_examples[key]["count"] += 1
                all_examples[key]["sources"].append(file_path.stem)

        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")
            continue

    # Convert to list format
    output_data = []

    for surface, data in all_examples.items():
        # Use most common split
        split_counter = defaultdict(int)
        for split_words in data["split_words"]:
            split_counter[split_words] += 1

        most_common_split = max(split_counter.items(), key=lambda x: x[1])[0]

        output_data.append({
            "surface_form": surface,
            "split_words": list(most_common_split),
            "count": data["count"],
            "sources": list(set(data["sources"]))[:5]  # Keep max 5 sources
        })

    # Sort by frequency
    output_data.sort(key=lambda x: x["count"], reverse=True)

    return output_data


def main():
    print("=" * 60)
    print("EXTRACT DCS SANDHI SPLITS")
    print("=" * 60)
    print()

    dcs_path = Path("data/corpora/dcs")
    output_path = Path("data/processed/dcs_sandhi_splits.json")

    # Extract data
    print("Extracting sandhi splits from DCS corpus...")
    print("This may take 10-30 minutes...")
    print()

    data = extract_all_dcs_data(dcs_path)

    if not data:
        print("ERROR: No data extracted")
        return 1

    print()
    print(f"Extracted {len(data):,} unique sandhi examples")
    print()

    # Statistics
    total_occurrences = sum(item["count"] for item in data)
    print(f"Total occurrences: {total_occurrences:,}")
    print(f"Average occurrences per example: {total_occurrences / len(data):.1f}")
    print()

    # Sample
    print("Sample examples:")
    for item in data[:10]:
        surface = item["surface_form"]
        split = ' + '.join(item["split_words"])
        count = item["count"]
        print(f"  {surface:20s} => {split:30s} ({count:3d}x)")
    print()

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # File size
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"File size: {size_mb:.1f} MB")
    print()

    print("=" * 60)
    print("✅ EXTRACTION COMPLETE")
    print("=" * 60)
    print()
    print("Next step:")
    print("  python scripts/verify_training_data.py")
    print()

    return 0


if __name__ == "__main__":
    exit(main())
