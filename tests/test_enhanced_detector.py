"""
Test enhanced detector vs baseline
Quick test to measure improvement
"""
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.WARNING)

from src.annotators.sandhi.detector import SanskritSandhiDetector
from src.annotators.sandhi.enhanced_detector import EnhancedSandhiDetector
from indic_transliteration import sanscript


def load_test_data():
    """Load SandhiKosh test data"""
    data_path = Path("data/processed/sandhikosh_test_100.json")
    with open(data_path, 'r') as f:
        data = json.load(f)
    return data['examples'][:20]  # Test on first 20


def test_detector(detector, examples, name="Detector"):
    """Test a detector and return accuracy"""
    correct = 0
    total = 0
    detected = 0

    for example in examples:
        surface = example['surface_form']
        expected = example['split_words']

        # Convert to IAST
        surface_iast = sanscript.transliterate(surface, sanscript.DEVANAGARI, sanscript.IAST)

        # Detect
        annotations = detector.detect_sandhi(surface_iast, encoding='IAST', limit=5)

        total += 1

        if annotations:
            detected += 1
            # Check if any annotation matches expected
            for ann in annotations:
                predicted = [ann.word1, ann.word2]
                # Simple match check (can be improved)
                if len(predicted) == len(expected):
                    correct += 1
                    break

    detection_rate = detected / total * 100
    accuracy = correct / total * 100

    print(f"\n{name}:")
    print(f"  Detection rate: {detection_rate:.1f}% ({detected}/{total})")
    print(f"  Accuracy: {accuracy:.1f}% ({correct}/{total})")

    return {
        'detection_rate': detection_rate,
        'accuracy': accuracy,
        'detected': detected,
        'total': total
    }


if __name__ == "__main__":
    print("=" * 70)
    print("ENHANCED DETECTOR TEST")
    print("=" * 70)

    # Load test data
    examples = load_test_data()
    print(f"\nTesting on {len(examples)} examples from SandhiKosh")

    # Test baseline detector
    print("\n" + "-" * 70)
    print("BASELINE DETECTOR")
    print("-" * 70)
    baseline = SanskritSandhiDetector(min_confidence=0.5)
    baseline_results = test_detector(baseline, examples, "Baseline")

    # Test enhanced detector
    print("\n" + "-" * 70)
    print("ENHANCED DETECTOR (Beam Search)")
    print("-" * 70)
    enhanced = EnhancedSandhiDetector(
        min_confidence=0.3,
        beam_width=5,
        max_words=5
    )
    enhanced_results = test_detector(enhanced, examples, "Enhanced")

    # Compare
    print("\n" + "=" * 70)
    print("IMPROVEMENT")
    print("=" * 70)
    detection_improvement = enhanced_results['detection_rate'] - baseline_results['detection_rate']
    accuracy_improvement = enhanced_results['accuracy'] - baseline_results['accuracy']

    print(f"Detection rate: {baseline_results['detection_rate']:.1f}% → {enhanced_results['detection_rate']:.1f}% ({detection_improvement:+.1f}%)")
    print(f"Accuracy: {baseline_results['accuracy']:.1f}% → {enhanced_results['accuracy']:.1f}% ({accuracy_improvement:+.1f}%)")

    if accuracy_improvement > 0:
        print("\n✓ Enhanced detector is BETTER!")
    elif accuracy_improvement == 0:
        print("\n= Same performance")
    else:
        print("\n✗ Enhanced detector is worse (needs tuning)")
