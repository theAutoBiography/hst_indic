"""
Extract sandhi examples from SandhiKosh dataset
Converts Excel format to usable test cases
"""
import pandas as pd
import json
from pathlib import Path

def extract_sandhikosh_data(excel_path, output_path, limit=100):
    """
    Extract sandhi examples from SandhiKosh Excel file

    Args:
        excel_path: Path to Excel file
        output_path: Path to save JSON output
        limit: Maximum number of examples to extract
    """
    print(f"Reading {excel_path}...")

    try:
        # Try reading as .xls (old Excel format)
        df = pd.read_excel(excel_path, engine='xlrd')
    except Exception as e:
        print(f"Error reading with xlrd: {e}")
        try:
            # Try openpyxl for .xlsx format
            df = pd.read_excel(excel_path, engine='openpyxl')
        except Exception as e2:
            print(f"Error reading with openpyxl: {e2}")
            return None

    print(f"Loaded {len(df)} rows")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nFirst few rows:")
    print(df.head())

    # Extract relevant columns
    # SandhiKosh has: 'Word' (joined), 'Split' (split with +), 'Type' (sandhi type)

    examples = []

    for idx, row in df.iterrows():
        if idx >= limit:
            break

        # Skip rows with missing data
        if pd.isna(row.get('Word')) or pd.isna(row.get('Split')):
            continue

        word = str(row['Word']).strip()
        split = str(row['Split']).strip()
        sandhi_type = str(row.get('Type', 'unknown')).strip()

        # Skip header rows or empty rows
        if not word or not split or word == 'nan':
            continue

        # Parse split (format: "word1+word2" or "word1+word2+word3")
        split_words = [w.strip() for w in split.split('+')]

        example = {
            'id': idx,
            'surface_form': word,
            'split': split,
            'split_words': split_words,
            'sandhi_type': sandhi_type,
            'num_words': len(split_words)
        }
        examples.append(example)

    # Save to JSON
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'source': str(excel_path),
            'total_rows': len(df),
            'extracted': len(examples),
            'columns': df.columns.tolist(),
            'examples': examples
        }, f, indent=2, ensure_ascii=False)

    print(f"\nExtracted {len(examples)} examples to {output_path}")
    return examples


if __name__ == "__main__":
    # Extract from Bhagavad Gita corpus
    extract_sandhikosh_data(
        excel_path="data/raw/sandhikosh_bhagavad_gita.xls",
        output_path="data/processed/sandhikosh_test_100.json",
        limit=100
    )
