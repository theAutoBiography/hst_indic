#!/usr/bin/env python3
"""
Command-line interface for processing Sanskrit texts
Detects sandhi and stores annotations to AWS RDS
"""
import argparse
import sys
import logging
from pathlib import Path
from src.annotators.sandhi.storage import SandhiAnnotationStorage
from src.database.models import LanguageType
from src.database.connection import db_connection

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def process_text_file(
    input_file: Path,
    title: str,
    author: str = None,
    source_url: str = None,
    encoding: str = 'IAST',
    min_confidence: float = 0.5
):
    """
    Process a Sanskrit text file and store annotations to database

    Args:
        input_file: Path to input text file
        title: Document title
        author: Author name (optional)
        source_url: Source URL (optional)
        encoding: Text encoding (IAST, Devanagari, SLP1, etc.)
        min_confidence: Minimum confidence threshold for annotations
    """
    # Read input file
    logger.info(f"Reading file: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()

    # Initialize storage
    storage = SandhiAnnotationStorage()
    storage.detector.min_confidence = min_confidence

    # Process and store
    logger.info(f"Processing text with encoding: {encoding}")
    try:
        stats = storage.process_and_store_text(
            text=text,
            title=title,
            language=LanguageType.SANSKRIT,
            author=author,
            source_url=source_url,
            encoding=encoding
        )

        # Print results
        print("\n" + "=" * 70)
        print("PROCESSING COMPLETE")
        print("=" * 70)
        print(f"Document ID: {stats['document_id']}")
        print(f"Total Sentences: {stats['total_sentences']}")
        print(f"Total Annotations: {stats['total_annotations']}")
        print(f"Processing Time: {stats['processing_time']:.2f}s")
        print(f"\nAnnotations by Type:")
        for sandhi_type, count in stats['annotations_by_type'].items():
            print(f"  {sandhi_type}: {count}")
        print("=" * 70)

        return True

    except Exception as e:
        logger.error(f"Error processing file: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_connection():
    """Test database connection"""
    print("Testing database connection...")
    if db_connection.test_connection():
        print("✓ Database connection successful!")
        return True
    else:
        print("✗ Database connection failed!")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Process Sanskrit texts and store sandhi annotations to AWS RDS',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a Sanskrit text file
  python scripts/process_sanskrit_text.py input.txt "Bhagavad Gita" --author "Vyasa" --encoding IAST

  # Process Devanagari text
  python scripts/process_sanskrit_text.py input.txt "Sample Text" --encoding Devanagari

  # Test database connection
  python scripts/process_sanskrit_text.py --test-connection
        """
    )

    parser.add_argument(
        'input_file',
        type=str,
        nargs='?',
        help='Path to input Sanskrit text file'
    )

    parser.add_argument(
        'title',
        type=str,
        nargs='?',
        help='Document title'
    )

    parser.add_argument(
        '--author',
        type=str,
        help='Author name'
    )

    parser.add_argument(
        '--source-url',
        type=str,
        help='Source URL'
    )

    parser.add_argument(
        '--encoding',
        type=str,
        default='IAST',
        choices=['IAST', 'Devanagari', 'SLP1', 'HK', 'ITRANS'],
        help='Text encoding (default: IAST)'
    )

    parser.add_argument(
        '--min-confidence',
        type=float,
        default=0.5,
        help='Minimum confidence threshold (0.0-1.0, default: 0.5)'
    )

    parser.add_argument(
        '--test-connection',
        action='store_true',
        help='Test database connection and exit'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Test connection mode
    if args.test_connection:
        sys.exit(0 if test_connection() else 1)

    # Validate required arguments
    if not args.input_file or not args.title:
        parser.error("input_file and title are required (unless using --test-connection)")

    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        sys.exit(1)

    # Process the file
    success = process_text_file(
        input_file=input_path,
        title=args.title,
        author=args.author,
        source_url=args.source_url,
        encoding=args.encoding,
        min_confidence=args.min_confidence
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
