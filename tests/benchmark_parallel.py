"""
Parallel benchmark on full SandhiKosh dataset
Uses multiprocessing for 8-10x speedup
"""
import json
import logging
from pathlib import Path
from collections import defaultdict, Counter
from multiprocessing import Pool, cpu_count
from functools import partial

# Setup logging to file only
logging.basicConfig(
    level=logging.WARNING,
    filename='benchmark_parallel.log',
    filemode='w',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

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
    """Normalize word for comparison"""
    word = word.strip()
    if any(ord(c) > 2304 and ord(c) < 2431 for c in word):
        word = sanscript.transliterate(word, sanscript.DEVANAGARI, sanscript.IAST)
    return word.lower()


def check_match(predicted_words, expected_words):
    """Check if predicted split matches expected"""
    if not predicted_words or not expected_words:
        return False

    pred_first = normalize_word(predicted_words[0])
    exp_first = normalize_word(expected_words[0])

    if pred_first == exp_first:
        if len(predicted_words) == 2 and len(expected_words) == 2:
            pred_second = normalize_word(predicted_words[1])
            exp_second = normalize_word(expected_words[1])
            return pred_second == exp_second
        return True

    return False


def process_example(example, detector_type='baseline', min_confidence=0.5):
    """Process single example (worker function)"""
    surface = example['surface_form']
    expected = example['split_words']
    sandhi_type = example['sandhi_type']
    num_words = example['num_words']

    # Convert to IAST
    surface_iast = sanscript.transliterate(surface, sanscript.DEVANAGARI, sanscript.IAST)

    # Create detector (each worker needs its own)
    if detector_type == 'enhanced':
        detector = EnhancedSandhiDetector(min_confidence=min_confidence, beam_width=5, max_words=5)
    else:
        detector = SanskritSandhiDetector(min_confidence=min_confidence)

    # Detect
    try:
        annotations = detector.detect_sandhi(surface_iast, encoding='IAST', limit=10)
    except Exception:
        annotations = []

    # Check results
    detected = len(annotations) > 0
    matched = False

    if annotations:
        for ann in annotations:
            predicted = [ann.word1, ann.word2]
            if check_match(predicted, expected):
                matched = True
                break

    return {
        'detected': detected,
        'matched': matched,
        'sandhi_type': sandhi_type,
        'num_words': num_words,
        'surface': surface if not matched and detected else None,
        'expected': expected if not matched and detected else None
    }


def benchmark_detector_parallel(detector_type, examples, name="Detector", sample_size=None):
    """Parallel benchmark"""
    if sample_size:
        examples = examples[:sample_size]

    # print(f"\n{'='*60}")
    # print(f"{name} - Processing {len(examples)} examples with {cpu_count()} cores")
    # print(f"{'='*60}")

    # Parallel processing
    worker_fn = partial(
        process_example,
        detector_type=detector_type,
        min_confidence=0.3 if detector_type == 'enhanced' else 0.5
    )

    with Pool(cpu_count()) as pool:
        results = list(pool.imap_unordered(worker_fn, examples, chunksize=50))

    # Aggregate stats
    stats = {
        'total': len(examples),
        'detected': 0,
        'correct': 0,
        'by_type': defaultdict(lambda: {'total': 0, 'detected': 0, 'correct': 0}),
        'by_num_words': defaultdict(lambda: {'total': 0, 'detected': 0, 'correct': 0}),
        'errors': []
    }

    for result in results:
        sandhi_type = result['sandhi_type']
        num_words = result['num_words']

        stats['by_type'][sandhi_type]['total'] += 1
        stats['by_num_words'][num_words]['total'] += 1

        if result['detected']:
            stats['detected'] += 1
            stats['by_type'][sandhi_type]['detected'] += 1
            stats['by_num_words'][num_words]['detected'] += 1

            if result['matched']:
                stats['correct'] += 1
                stats['by_type'][sandhi_type]['correct'] += 1
                stats['by_num_words'][num_words]['correct'] += 1
            elif result['surface']:
                stats['errors'].append({
                    'surface': result['surface'],
                    'expected': result['expected'],
                    'type': sandhi_type,
                    'num_words': num_words
                })

    return stats


def print_results(stats, name="Detector"):
    """Print benchmark results"""
    # print(f"\n{'='*60}")
    # print(f"{name} RESULTS")
    # print(f"{'='*60}")

    total = stats['total']
    detected = stats['detected']
    correct = stats['correct']

    detection_rate = detected / total * 100
    accuracy = correct / total * 100

    # print(f"\nOverall:")
    # print(f"  Detection rate: {detection_rate:.1f}% ({detected}/{total})")
    # print(f"  Accuracy: {accuracy:.1f}% ({correct}/{total})")

    print(f"\nBy Sandhi Type:")
    for stype in sorted(stats['by_type'].keys()):
        type_stats = stats['by_type'][stype]
        t_total = type_stats['total']
        t_correct = type_stats['correct']
        acc_rate = t_correct / t_total * 100 if t_total > 0 else 0
        print(f"  {stype:12s}: {acc_rate:5.1f}% ({t_correct:4d}/{t_total:4d})")

    print(f"\nBy Number of Words:")
    for num_words in sorted(stats['by_num_words'].keys()):
        word_stats = stats['by_num_words'][num_words]
        w_total = word_stats['total']
        w_correct = word_stats['correct']
        acc_rate = w_correct / w_total * 100 if w_total > 0 else 0
        print(f"  {num_words} words: {acc_rate:5.1f}% ({w_correct:4d}/{w_total:4d})")


if __name__ == "__main__":
    # print("="*60)
    # print("PARALLEL SANDHIKOSH BENCHMARK")
    # print("="*60)

    # Load data
    # print("\nLoading data...")
    examples = load_full_data()
    # print(f"Loaded {len(examples)} examples")

    # Test baseline
    # print("\n" + "-"*60)
    # print("BASELINE DETECTOR")
    # print("-"*60)
    baseline_stats = benchmark_detector_parallel('baseline', examples, "BASELINE")
    # print_results(baseline_stats, "BASELINE")

    # Test enhanced
    # print("\n" + "-"*60)
    # print("ENHANCED DETECTOR")
    # print("-"*60)
    enhanced_stats = benchmark_detector_parallel('enhanced', examples, "ENHANCED")
    # print_results(enhanced_stats, "ENHANCED")

    # # Comparison
    # print(f"\n{'='*60}")
    # print("IMPROVEMENT SUMMARY")
    # print(f"{'='*60}")

    baseline_acc = baseline_stats['correct'] / baseline_stats['total'] * 100
    enhanced_acc = enhanced_stats['correct'] / enhanced_stats['total'] * 100
    improvement = enhanced_acc - baseline_acc

    print(f"\nAccuracy:")
    print(f"  Baseline:  {baseline_acc:.2f}%")
    print(f"  Enhanced:  {enhanced_acc:.2f}%")
    print(f"  Improvement: {improvement:+.2f}%")
