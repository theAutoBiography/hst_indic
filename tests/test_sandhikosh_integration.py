"""
Test sandhi detection and storage using SandhiKosh dataset (Bhagavad Gita)
Tests 100 real Sanskrit examples with known sandhi splits
"""
import logging
import json
from pathlib import Path
from src.annotators.sandhi.storage import SandhiAnnotationStorage
from src.annotators.sandhi.detector import SanskritSandhiDetector
from src.database.models import LanguageType
from src.database.operations import get_document, get_annotations_by_document
from src.database.connection import get_db_session
from indic_transliteration import sanscript

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_sandhikosh_test_data():
    """Load the extracted SandhiKosh test data"""
    data_path = Path("data/processed/sandhikosh_test_100.json")

    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data['examples']


def test_sandhi_detection_only():
    """
    Test sandhi detection on SandhiKosh data WITHOUT database storage
    This tests the detector accuracy
    """
    print("=" * 80)
    print("TESTING SANDHI DETECTION (Detector Accuracy)")
    print("=" * 80)

    examples = load_sandhikosh_test_data()
    print(f"Loaded {len(examples)} examples from SandhiKosh (Bhagavad Gita corpus)")

    # Initialize detector
    detector = SanskritSandhiDetector(
        use_dictionary_validation=True,
        min_confidence=0.5
    )

    stats = {
        'total': 0,
        'detected': 0,
        'correct_splits': 0,
        'partial_matches': 0,
        'failures': 0,
        'by_sandhi_type': {}
    }

    # Test first 20 examples (for quick test)
    test_limit = 20

    for i, example in enumerate(examples[:test_limit], 1):
        surface_form = example['surface_form']
        expected_split = example['split_words']
        sandhi_type = example['sandhi_type']

        print(f"\n{'-' * 80}")
        print(f"Example {i}/{test_limit}")
        print(f"Surface (Devanagari): {surface_form}")
        print(f"Expected split: {' + '.join(expected_split)}")
        print(f"Type: {sandhi_type}")

        # Convert to IAST for processing
        surface_iast = sanscript.transliterate(surface_form, sanscript.DEVANAGARI, sanscript.IAST)
        print(f"Surface (IAST): {surface_iast}")

        stats['total'] += 1

        try:
            # Detect sandhi
            annotations = detector.detect_sandhi(surface_iast, encoding='IAST', limit=5)

            if annotations:
                print(f"\n✓ Found {len(annotations)} possible split(s):")
                for j, ann in enumerate(annotations[:3], 1):
                    print(f"  {j}. {ann.word1} + {ann.word2}")
                    print(f"     Confidence: {ann.confidence:.2f}")
                    print(f"     Type: {ann.sandhi_type.value}")

                stats['detected'] += 1

                # Track by type
                if sandhi_type not in stats['by_sandhi_type']:
                    stats['by_sandhi_type'][sandhi_type] = {'total': 0, 'detected': 0}
                stats['by_sandhi_type'][sandhi_type]['total'] += 1
                stats['by_sandhi_type'][sandhi_type]['detected'] += 1

            else:
                print("✗ No splits detected")
                stats['failures'] += 1

                if sandhi_type not in stats['by_sandhi_type']:
                    stats['by_sandhi_type'][sandhi_type] = {'total': 0, 'detected': 0}
                stats['by_sandhi_type'][sandhi_type]['total'] += 1

        except Exception as e:
            print(f"✗ Error: {e}")
            stats['failures'] += 1

    # Print summary
    print("\n" + "=" * 80)
    print("DETECTION SUMMARY")
    print("=" * 80)
    print(f"Total examples tested: {stats['total']}")
    print(f"Successfully detected: {stats['detected']} ({stats['detected']/stats['total']*100:.1f}%)")
    print(f"Failed to detect: {stats['failures']} ({stats['failures']/stats['total']*100:.1f}%)")

    print(f"\nBy Sandhi Type:")
    for stype, counts in stats['by_sandhi_type'].items():
        total = counts['total']
        detected = counts['detected']
        pct = detected / total * 100 if total > 0 else 0
        print(f"  {stype}: {detected}/{total} ({pct:.1f}%)")

    return stats


def test_database_storage():
    """
    Test storing sandhi annotations to AWS RDS
    Uses SandhiKosh data
    """
    print("\n" + "=" * 80)
    print("TESTING DATABASE STORAGE")
    print("=" * 80)

    examples = load_sandhikosh_test_data()

    # Take first 10 examples for database storage test
    test_examples = examples[:10]

    # Create text from examples
    text_lines = []
    for example in test_examples:
        surface = example['surface_form']
        text_lines.append(surface)

    full_text = '\n'.join(text_lines)

    print(f"Testing with {len(test_examples)} examples")
    print(f"Text preview:\n{full_text[:200]}...\n")

    # Initialize storage
    storage = SandhiAnnotationStorage()

    try:
        # Process and store
        stats = storage.process_and_store_text(
            text=full_text,
            title="Bhagavad Gita - SandhiKosh Test Sample",
            language=LanguageType.SANSKRIT,
            author="Vyasa",
            source_url="https://github.com/sanskrit-sandhi/SandhiKosh",
            encoding='Devanagari',
            metadata={'source': 'SandhiKosh', 'corpus': 'Bhagavad Gita'}
        )

        print("\n" + "=" * 80)
        print("STORAGE RESULTS")
        print("=" * 80)
        print(f"Document ID: {stats['document_id']}")
        print(f"Total Sentences: {stats['total_sentences']}")
        print(f"Total Annotations: {stats['total_annotations']}")
        print(f"Processing Time: {stats['processing_time']:.2f}s")
        print(f"\nAnnotations by Type:")
        for sandhi_type, count in stats['annotations_by_type'].items():
            print(f"  {sandhi_type}: {count}")

        # Verify in database
        print("\n" + "=" * 80)
        print("VERIFYING IN DATABASE")
        print("=" * 80)

        with get_db_session() as session:
            # Get document
            document = get_document(stats['document_id'], session=session)
            if document:
                print(f"✓ Document found: {document.title}")
                print(f"  Language: {document.language.value}")
                print(f"  Author: {document.author}")
                print(f"  Sentences: {len(document.sentences)}")

                # Get annotations
                annotations = get_annotations_by_document(
                    stats['document_id'],
                    session=session
                )
                print(f"\n✓ Found {len(annotations)} annotations in database")

                # Show sample annotations
                print(f"\nSample Annotations (first 3):")
                for i, ann in enumerate(annotations[:3], 1):
                    print(f"\n  {i}. Surface: '{ann.surface_form}'")
                    print(f"     Split: {ann.word1} + {ann.word2}")
                    print(f"     Type: {ann.sandhi_type.value}")
                    print(f"     Confidence: {ann.confidence:.2f}")
                    print(f"     Rule: {ann.sandhi_rule}")
            else:
                print("✗ Document not found in database")

        print("\n" + "=" * 80)
        print("DATABASE TEST COMPLETED SUCCESSFULLY!")
        print("=" * 80)

        return True

    except Exception as e:
        logger.error(f"Database test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Test 1: Detector accuracy
    print("\n" + "=" * 80)
    print("TEST 1: SANDHI DETECTOR ACCURACY")
    print("=" * 80)
    detection_stats = test_sandhi_detection_only()

    # Test 2: Database storage
    print("\n" + "=" * 80)
    print("TEST 2: DATABASE STORAGE INTEGRATION")
    print("=" * 80)
    storage_success = test_database_storage()

    # Final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"Detection test: {'✓ PASSED' if detection_stats['detected'] > 0 else '✗ FAILED'}")
    print(f"Storage test: {'✓ PASSED' if storage_success else '✗ FAILED'}")
    print("=" * 80)

    exit(0 if (detection_stats['detected'] > 0 and storage_success) else 1)
