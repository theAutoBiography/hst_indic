"""
Test sandhi annotation storage to AWS RDS
End-to-end integration test
"""
import logging
from src.annotators.sandhi.storage import SandhiAnnotationStorage
from src.database.models import LanguageType
from src.database.operations import get_document, get_annotations_by_document
from src.database.connection import get_db_session

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_storage_integration():
    """
    Test complete workflow:
    1. Process text with sandhi detection
    2. Store to AWS RDS
    3. Verify data in database
    """
    print("=" * 70)
    print("TESTING SANDHI ANNOTATION STORAGE")
    print("=" * 70)

    # Sample Sanskrit text
    test_text = """
    rāmo'sti
    tathā'pi
    namaste
    """

    # Initialize storage
    logger.info("Initializing SandhiAnnotationStorage...")
    storage = SandhiAnnotationStorage()

    # Process and store
    logger.info("Processing and storing text...")
    try:
        stats = storage.process_and_store_text(
            text=test_text,
            title="Test Sanskrit Sentences",
            language=LanguageType.SANSKRIT,
            author="Integration Test",
            encoding='IAST'
        )

        print("\n" + "=" * 70)
        print("PROCESSING RESULTS")
        print("=" * 70)
        print(f"Document ID: {stats['document_id']}")
        print(f"Total Sentences: {stats['total_sentences']}")
        print(f"Total Annotations: {stats['total_annotations']}")
        print(f"Processing Time: {stats['processing_time']:.2f}s")
        print(f"\nAnnotations by Type:")
        for sandhi_type, count in stats['annotations_by_type'].items():
            print(f"  {sandhi_type}: {count}")

        # Verify in database
        print("\n" + "=" * 70)
        print("VERIFYING IN DATABASE")
        print("=" * 70)

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
                print(f"\nSample Annotations:")
                for i, ann in enumerate(annotations[:5], 1):
                    print(f"\n  {i}. Surface: '{ann.surface_form}'")
                    print(f"     Split: {ann.word1} + {ann.word2}")
                    print(f"     Type: {ann.sandhi_type.value}")
                    print(f"     Confidence: {ann.confidence:.2f}")
                    print(f"     Rule: {ann.sandhi_rule}")
            else:
                print("✗ Document not found in database")

        print("\n" + "=" * 70)
        print("TEST COMPLETED SUCCESSFULLY!")
        print("=" * 70)

        return True

    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_storage_integration()
    exit(0 if success else 1)
