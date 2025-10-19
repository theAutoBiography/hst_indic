"""
Parse Monier-Williams dictionary into usable format
Extract headwords for fast lookup
"""
import re
import pickle
from pathlib import Path
from collections import defaultdict

def parse_mw_dictionary(input_file, output_file):
    """
    Parse MW dictionary and extract headwords

    Format: <k1>headword</k1>
    """
    print(f"Parsing {input_file}...")

    headwords = set()
    word_variants = defaultdict(list)

    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            if line_num % 10000 == 0:
                print(f"  Processed {line_num} lines, found {len(headwords)} headwords...")

            # Extract headword from <k1>word<k2> or <k1>word<h> format
            k1_match = re.search(r'<k1>([^<]+)<', line)
            if k1_match:
                word = k1_match.group(1)
                headwords.add(word)

                # Also extract k2 variant if different
                k2_match = re.search(r'<k2>([^<]+)<', line)
                if k2_match:
                    variant = k2_match.group(1)
                    if variant != word:
                        headwords.add(variant)
                        word_variants[word].append(variant)

    print(f"\nTotal headwords: {len(headwords)}")
    print(f"Total variants: {sum(len(v) for v in word_variants.values())}")

    # Save as pickle for fast loading
    output_path = Path(output_file)
    with open(output_path, 'wb') as f:
        pickle.dump({
            'headwords': headwords,
            'variants': dict(word_variants)
        }, f)

    print(f"Saved to {output_path}")

    # Also save as text for inspection
    text_file = output_path.with_suffix('.txt')
    with open(text_file, 'w', encoding='utf-8') as f:
        for word in sorted(headwords)[:1000]:  # First 1000 for inspection
            f.write(f"{word}\n")
    print(f"Sample saved to {text_file}")

    return headwords, word_variants


if __name__ == "__main__":
    headwords, variants = parse_mw_dictionary(
        "data/dictionaries/mw_full.txt",
        "data/dictionaries/mw_headwords.pkl"
    )

    print(f"\nSample headwords:")
    for word in list(headwords)[:20]:
        print(f"  {word}")
