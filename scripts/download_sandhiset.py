#!/usr/bin/env python3
"""
Download sandhiset.txt from GitHub
"""

import urllib.request
from pathlib import Path


def download_sandhiset():
    """Download sandhiset.txt from GitHub"""
    url = "https://raw.githubusercontent.com/SushantDave/Sandhi_Prakarana/master/Data/sandhiset.txt"
    output_path = Path("data/annotations/sandhiset.txt")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading sandhiset.txt from GitHub...")
    print(f"URL: {url}")
    print(f"Output: {output_path}")
    print()

    try:
        urllib.request.urlretrieve(url, output_path)

        # Check file
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"✓ Downloaded successfully ({size_mb:.1f} MB)")

        # Count lines
        with open(output_path, 'r', encoding='utf-8') as f:
            count = sum(1 for line in f if line.strip() and '=>' in line)

        print(f"✓ Found {count:,} sandhi examples")
        print()

        return True

    except Exception as e:
        print(f"✗ Download failed: {e}")
        return False


if __name__ == "__main__":
    success = download_sandhiset()
    exit(0 if success else 1)
