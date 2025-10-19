"""
Test sandhi detection on real Sanskrit texts
"""
import logging

# Suppress verbose logging
logging.basicConfig(level=logging.WARNING)
logging.getLogger('sanskrit_parser').setLevel(logging.ERROR)
logging.getLogger('sanskrit_util').setLevel(logging.ERROR)

from sanskrit_parser import Parser

def test_sanskrit_parser_directly():
    """
    Test sanskrit_parser directly to understand its output format
    """
    print("="*70)
    print("TESTING sanskrit_parser DIRECTLY")
    print("="*70)

    parser = Parser()

    # Test cases with known sandhi
    test_cases = [
        ("rAmaH", "rāmaḥ - Single word (should have no/few splits)"),
        ("rAmo'sti", "rāmo'sti - Visarga sandhi: rāmaḥ + asti"),
        ("tawA'pi", "tathā'pi - Vowel sandhi: tathā + api"),
        ("namaste", "namaste - namaḥ + te"),
    ]

    for slp1_input, description in test_cases:
        print(f"\n{'-'*70}")
        print(f"Input (SLP1): {slp1_input}")
        print(f"Description: {description}")
        print(f"{'-'*70}")

        try:
            # Get splits
            splits = parser.split(slp1_input, limit=5)

            if splits:
                print(f"✓ Found {len(splits)} possible analysis/analyses")
                print(f"\nRaw output from parser:")
                print(f"  {splits}")

                # Try to extract information
                for i, split in enumerate(splits[:3], 1):
                    print(f"\nAnalysis {i}:")
                    print(f"  Type: {type(split)}")
                    print(f"  String repr: {split}")

                    # Try different ways to extract words
                    if hasattr(split, '__iter__'):
                        print(f"  Is iterable: Yes")
                        try:
                            for j, item in enumerate(split):
                                print(f"    Item {j}: {item} (type: {type(item).__name__})")
                        except:
                            print(f"    Could not iterate")

                    # Check for common attributes
                    attrs = ['words', 'splits', 'graph', 'nodes', 'getPairs']
                    for attr in attrs:
                        if hasattr(split, attr):
                            print(f"  Has attribute '{attr}': Yes")
                            try:
                                value = getattr(split, attr)
                                if callable(value):
                                    result = value()
                                    print(f"    {attr}() = {result}")
                                else:
                                    print(f"    {attr} = {value}")
                            except Exception as e:
                                print(f"    Error accessing {attr}: {e}")

            else:
                print("✗ No splits found (single valid word)")

        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*70)
    print("Test complete!")
    print("="*70)


if __name__ == "__main__":
    test_sanskrit_parser_directly()
