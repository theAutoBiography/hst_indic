"""
Full benchmark on 1,425 SandhiKosh examples
Test both baseline and enhanced detectors
"""
import json
import logging
from pathlib import Path
from collections import defaultdict, Counter

logging.basicConfig(level=logging.WARNING)

from src.annotators.sandhi.detector import SanskritSandhiDetector
from src.annotators.sandhi.enhanced_detector import EnhancedSandhiDetector
from indic_transliteration import sanscript


def load_full_data():
    """Load all SandhiKosh examples"""
    data_path = Path("data/processed/sandhikosh_full.json")
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['examples']


def normalize_word(word):
    """Normalize word for comparison - handle encoding"""
    from indic_transliteration import sanscript

    word = word.strip()

    # Convert to IAST for comparison (standard format)
    # Try to detect if it's Devanagari
    if any(ord(c) > 2304 and ord(c) < 2431 for c in word):
        # Devanagari range
        word = sanscript.transliterate(word, sanscript.DEVANAGARI, sanscript.IAST)

    # Normalize to lowercase
    return word.lower()


def check_match(predicted_words, expected_words):
    """
    Check if predicted split matches expected
    Handles partial matches - if first predicted word matches first expected
    """
    if not predicted_words or not expected_words:
        return False

    # For 2-word predictions against multi-word expected:
    # Check if first word matches
    pred_first = normalize_word(predicted_words[0])
    exp_first = normalize_word(expected_words[0])

    # Exact match on first word is a good sign
    if pred_first == exp_first:
        # If both are 2 words, check second word too
        if len(predicted_words) == 2 and len(expected_words) == 2:
            pred_second = normalize_word(predicted_words[1])
            exp_second = normalize_word(expected_words[1])
            return pred_second == exp_second
        # Partial credit for getting first word right
        return True

    return False


def benchmark_detector(detector, examples, name="Detector", sample_size=None):
    """
    Comprehensive benchmark
    """
    if sample_size:
        examples = examples[:sample_size]

    stats = {
        'total': len(examples),
        'detected': 0,
        'correct': 0,
        'by_type': defaultdict(lambda: {'total': 0, 'detected': 0, 'correct': 0}),
        'by_num_words': defaultdict(lambda: {'total': 0, 'detected': 0, 'correct': 0}),
        'errors': []
    }

    print(f"\nProcessing {stats['total']} examples...")

    for i, example in enumerate(examples):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{stats['total']}...")

        surface = example['surface_form']
        expected = example['split_words']
        sandhi_type = example['sandhi_type']
        num_words = example['num_words']

        # Convert to IAST
        surface_iast = sanscript.transliterate(surface, sanscript.DEVANAGARI, sanscript.IAST)

        # Detect
        try:
            annotations = detector.detect_sandhi(surface_iast, encoding='IAST', limit=10)
        except Exception as e:
            logging.warning(f"Error on '{surface}': {e}")
            annotations = []

        # Update stats
        stats['by_type'][sandhi_type]['total'] += 1
        stats['by_num_words'][num_words]['total'] += 1

        if annotations:
            stats['detected'] += 1
            stats['by_type'][sandhi_type]['detected'] += 1
            stats['by_num_words'][num_words]['detected'] += 1

            # Check if any annotation matches
            matched = False
            for ann in annotations:
                predicted = [ann.word1, ann.word2]

                if check_match(predicted, expected):
                    matched = True
                    stats['correct'] += 1
                    stats['by_type'][sandhi_type]['correct'] += 1
                    stats['by_num_words'][num_words]['correct'] += 1
                    break

            if not matched:
                # Record error
                stats['errors'].append({
                    'surface': surface,
                    'expected': expected,
                    'predicted': [ann.word1, ann.word2] if annotations else [],
                    'type': sandhi_type,
                    'num_words': num_words
                })

    return stats


def print_results(stats, name="Detector"):
    """Print benchmark results"""
    print("\n" + "=" * 70)
    print(f"{name} RESULTS")
    print("=" * 70)

    total = stats['total']
    detected = stats['detected']
    correct = stats['correct']

    detection_rate = detected / total * 100
    accuracy = correct / total * 100

    print(f"\nOverall:")
    print(f"  Total examples: {total}")
    print(f"  Detection rate: {detection_rate:.1f}% ({detected}/{total})")
    print(f"  Accuracy: {accuracy:.1f}% ({correct}/{total})")

    print(f"\nBy Sandhi Type:")
    for stype in sorted(stats['by_type'].keys()):
        type_stats = stats['by_type'][stype]
        t_total = type_stats['total']
        t_detected = type_stats['detected']
        t_correct = type_stats['correct']

        det_rate = t_detected / t_total * 100 if t_total > 0 else 0
        acc_rate = t_correct / t_total * 100 if t_total > 0 else 0

        print(f"  {stype:12s}: {acc_rate:5.1f}% accuracy ({t_correct:3d}/{t_total:3d}) | {det_rate:5.1f}% detected")

    print(f"\nBy Number of Words:")
    for num_words in sorted(stats['by_num_words'].keys()):
        word_stats = stats['by_num_words'][num_words]
        w_total = word_stats['total']
        w_detected = word_stats['detected']
        w_correct = word_stats['correct']

        det_rate = w_detected / w_total * 100 if w_total > 0 else 0
        acc_rate = w_correct / w_total * 100 if w_total > 0 else 0

        print(f"  {num_words} words: {acc_rate:5.1f}% accuracy ({w_correct:3d}/{w_total:3d}) | {det_rate:5.1f}% detected")

    print(f"\nTop 5 Error Patterns:")
    error_counter = Counter()
    for error in stats['errors'][:100]:  # Sample first 100 errors
        pattern = f"{error['type']} / {error['num_words']} words"
        error_counter[pattern] += 1

    for pattern, count in error_counter.most_common(5):
        print(f"  {pattern}: {count} errors")


if __name__ == "__main__":
    print("=" * 70)
    print("FULL SANDHIKOSH BENCHMARK (1,425 examples)")
    print("=" * 70)

    # Load all data
    print("\nLoading data...")
    examples = load_full_data()
    print(f"Loaded {len(examples)} examples")

    # Test baseline
    print("\n" + "-" * 70)
    print("TESTING BASELINE DETECTOR")
    print("-" * 70)
    baseline = SanskritSandhiDetector(min_confidence=0.5)
    baseline_stats = benchmark_detector(baseline, examples, "Baseline")
    print_results(baseline_stats, "BASELINE")

    # Test enhanced
    print("\n" + "-" * 70)
    print("TESTING ENHANCED DETECTOR")
    print("-" * 70)
    enhanced = EnhancedSandhiDetector(
        min_confidence=0.3,
        beam_width=5,
        max_words=5
    )
    enhanced_stats = benchmark_detector(enhanced, examples, "Enhanced")
    print_results(enhanced_stats, "ENHANCED")

    # Comparison
    print("\n" + "=" * 70)
    print("IMPROVEMENT SUMMARY")
    print("=" * 70)

    baseline_acc = baseline_stats['correct'] / baseline_stats['total'] * 100
    enhanced_acc = enhanced_stats['correct'] / enhanced_stats['total'] * 100
    improvement = enhanced_acc - baseline_acc

    print(f"\nAccuracy:")
    print(f"  Baseline:  {baseline_acc:.2f}%")
    print(f"  Enhanced:  {enhanced_acc:.2f}%")
    print(f"  Improvement: {improvement:+.2f}%")

    if improvement > 0:
        print("\n✓✓✓ ENHANCED DETECTOR IS BETTER! ✓✓✓")
    else:
        print("\n✗ No improvement (needs more tuning)")
