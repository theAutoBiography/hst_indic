#!/usr/bin/env python3
"""
Convert external sandhi datasets (Malayalam, Telugu) to unified format with IITHLP transliteration
"""

import json
import re
from pathlib import Path
from typing import List, Tuple, Optional

try:
    from iithlp import to_roman, to_script
    IITHLP_AVAILABLE = True
except ImportError:
    print("WARNING: iithlp not installed. Skipping transliteration.")
    IITHLP_AVAILABLE = False


def parse_telugu_dataset(file_path: str) -> List[Tuple[str, str]]:
    """
    Parse Telugu sandhi dataset from WX format
    Format: surface=word1+word2 | split_position
    Example: praXAnAMSamani=praXAna+aMSamani | 6
    """
    examples = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or '=' not in line:
                continue

            # Split on pipe to remove position info
            if '|' in line:
                line = line.split('|')[0].strip()

            # Parse: surface=word1+word2
            parts = line.split('=')
            if len(parts) != 2:
                continue

            surface = parts[0].strip()
            split_words = parts[1].strip()

            # Convert + separator to space +
            split_words = split_words.replace('+', ' + ')

            examples.append((surface, split_words))

    return examples


def parse_malayalam_dataset(file_path: str, max_examples: Optional[int] = None) -> List[Tuple[str, str]]:
    """
    Parse Malayalam sandhi dataset from character-level format
    Format: char1 char2 ... \t NSP NSP SP NSP ...
    Where SP = split point, NSP = non-split point
    """
    examples = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_examples and i >= max_examples:
                break

            line = line.strip()
            if not line or '\t' not in line:
                continue

            parts = line.split('\t')
            if len(parts) != 2:
                continue

            chars = parts[0].split()
            labels = parts[1].split()

            if len(chars) != len(labels):
                continue

            # Reconstruct surface form and split form
            surface = ''.join(chars)

            # Build split form by inserting ' + ' at split points
            split_words = []
            current_word = []

            for char, label in zip(chars, labels):
                if label == 'SP' and current_word:
                    split_words.append(''.join(current_word))
                    current_word = [char]
                else:
                    current_word.append(char)

            if current_word:
                split_words.append(''.join(current_word))

            if len(split_words) >= 2:  # Only keep if it has at least 2 words
                split_form = ' + '.join(split_words)
                examples.append((surface, split_form))

    return examples


def transliterate_to_iithlp(text: str, source_script: str) -> str:
    """Convert text to IITHLP Roman script"""
    if not IITHLP_AVAILABLE:
        return text

    try:
        return to_roman(text)
    except Exception as e:
        print(f"Warning: Failed to transliterate '{text[:50]}...': {e}")
        return text


def convert_dataset(
    input_file: str,
    output_file: str,
    dataset_type: str,
    source_script: str,
    max_examples: Optional[int] = None
):
    """Convert dataset to unified format with IITHLP transliteration"""

    print(f"\n{'='*60}")
    print(f"Converting {dataset_type} dataset")
    print(f"{'='*60}")
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    print(f"Source script: {source_script}")

    # Parse dataset
    if dataset_type == 'telugu':
        examples = parse_telugu_dataset(input_file)
    elif dataset_type == 'malayalam':
        examples = parse_malayalam_dataset(input_file, max_examples)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    print(f"Parsed {len(examples):,} examples")

    # Convert to IITHLP if available
    if IITHLP_AVAILABLE:
        print("Converting to IITHLP Roman script...")
        converted_examples = []

        for i, (surface, split) in enumerate(examples):
            if i % 1000 == 0:
                print(f"  Converted {i:,}/{len(examples):,} examples", end='\r')

            surface_iithlp = transliterate_to_iithlp(surface, source_script)
            split_iithlp = transliterate_to_iithlp(split, source_script)

            converted_examples.append({
                'surface_form': surface_iithlp,
                'split_words': split_iithlp.split(' + '),
                'source_language': source_script,
                'original_surface': surface,
                'original_split': split
            })

        print(f"  Converted {len(converted_examples):,}/{len(examples):,} examples")
    else:
        print("IITHLP not available, keeping original script")
        converted_examples = [
            {
                'surface_form': surface,
                'split_words': split.split(' + '),
                'source_language': source_script,
                'original_surface': surface,
                'original_split': split
            }
            for surface, split in examples
        ]

    # Save to JSON
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted_examples, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Saved {len(converted_examples):,} examples to {output_file}")

    # Show samples
    print("\nSample conversions:")
    for i, ex in enumerate(converted_examples[:3]):
        print(f"\n  Example {i+1}:")
        print(f"    Original: {ex['original_surface']} => {ex['original_split']}")
        print(f"    IITHLP:   {ex['surface_form']} => {' + '.join(ex['split_words'])}")


def main():
    """Convert all external datasets"""

    print("="*60)
    print("EXTERNAL DATASET CONVERSION TO IITHLP")
    print("="*60)

    base_dir = Path("data/external_datasets")
    output_dir = Path("data/processed")

    # Telugu dataset
    telugu_input = base_dir / "telugu_sandhi/Train/sandhiwords_wx_train.txt"
    telugu_output = output_dir / "telugu_sandhi_iithlp.json"

    if telugu_input.exists():
        convert_dataset(
            str(telugu_input),
            str(telugu_output),
            'telugu',
            'telugu'
        )
    else:
        print(f"\nWarning: Telugu dataset not found at {telugu_input}")

    # Malayalam dataset (use the largest one)
    malayalam_input = base_dir / "malayalam_sandhi/data/40k_gold+wiki_sandhi_char_level_training_data"
    malayalam_output = output_dir / "malayalam_sandhi_iithlp.json"

    if malayalam_input.exists():
        convert_dataset(
            str(malayalam_input),
            str(malayalam_output),
            'malayalam',
            'malayalam',
            max_examples=50000  # Limit to 50k for manageable size
        )
    else:
        print(f"\nWarning: Malayalam dataset not found at {malayalam_input}")

    print("\n" + "="*60)
    print("CONVERSION COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()
