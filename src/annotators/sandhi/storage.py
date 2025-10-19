"""
Database storage for sandhi annotations
Connects sandhi detector with AWS RDS PostgreSQL
"""
import logging
from typing import List, Optional, Dict, Any
from uuid import UUID
from datetime import datetime

from src.database.connection import get_db_session
from src.database.operations import (
    create_document,
    create_sentence,
    create_sandhi_annotation,
    get_document
)
from src.database.models import (
    Document,
    Sentence,
    SandhiAnnotation as DBSandhiAnnotation,
    LanguageType,
    AnnotationMethod,
    SandhiType as DBSandhiType
)
from src.annotators.sandhi.detector import (
    SanskritSandhiDetector,
    SandhiAnnotation,
    SandhiType
)

logger = logging.getLogger(__name__)


class SandhiAnnotationStorage:
    """
    Handles storing sandhi annotations to AWS RDS database

    Workflow:
    1. Create/get document
    2. Create sentences
    3. Detect sandhi for each sentence
    4. Store annotations
    """

    def __init__(self, detector: Optional[SanskritSandhiDetector] = None):
        """
        Initialize storage handler

        Args:
            detector: SandhiDetector instance (creates new one if not provided)
        """
        self.detector = detector or SanskritSandhiDetector()
        logger.info("SandhiAnnotationStorage initialized")

    def process_and_store_text(
        self,
        text: str,
        title: str,
        language: LanguageType = LanguageType.SANSKRIT,
        author: Optional[str] = None,
        source_url: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        encoding: str = 'IAST'
    ) -> Dict[str, Any]:
        """
        Process a text and store all annotations to database

        Args:
            text: Full text to process
            title: Document title
            language: Language (default: Sanskrit)
            author: Author name
            source_url: Source URL
            metadata: Additional metadata
            encoding: Text encoding (IAST, Devanagari, etc.)

        Returns:
            Dictionary with processing results and statistics
        """
        logger.info(f"Processing text: '{title}'")

        # Split text into sentences
        sentences = self._split_into_sentences(text)
        logger.info(f"Split into {len(sentences)} sentences")

        stats = {
            'document_id': None,
            'total_sentences': len(sentences),
            'total_annotations': 0,
            'annotations_by_type': {},
            'processing_time': None
        }

        start_time = datetime.now()

        try:
            with get_db_session() as session:
                # Create document
                document = create_document(
                    title=title,
                    language=language,
                    author=author,
                    source_url=source_url,
                    metadata=metadata,
                    session=session
                )
                stats['document_id'] = str(document.doc_id)
                logger.info(f"Created document: {document.doc_id}")

                # Process each sentence
                for position, sentence_text in enumerate(sentences):
                    if not sentence_text.strip():
                        continue

                    # Create sentence in database
                    sentence = create_sentence(
                        doc_id=document.doc_id,
                        position=position,
                        original_text=sentence_text,
                        normalized_text=sentence_text,  # TODO: Add normalization
                        session=session
                    )

                    # Detect sandhi
                    annotations = self.detector.detect_sandhi(
                        sentence_text,
                        encoding=encoding
                    )

                    # Store annotations
                    for annotation in annotations:
                        self._store_annotation(
                            annotation,
                            sentence.sent_id,
                            session
                        )
                        stats['total_annotations'] += 1

                        # Track by type
                        sandhi_type = annotation.sandhi_type.value
                        stats['annotations_by_type'][sandhi_type] = \
                            stats['annotations_by_type'].get(sandhi_type, 0) + 1

                session.commit()
                logger.info(f"Successfully stored {stats['total_annotations']} annotations")

        except Exception as e:
            logger.error(f"Error processing text: {e}")
            raise

        stats['processing_time'] = (datetime.now() - start_time).total_seconds()
        return stats

    def process_and_store_sentences(
        self,
        sentences: List[str],
        document_title: str,
        language: LanguageType = LanguageType.SANSKRIT,
        author: Optional[str] = None,
        encoding: str = 'IAST'
    ) -> UUID:
        """
        Process a list of sentences and store to database

        Args:
            sentences: List of sentence strings
            document_title: Title for the document
            language: Language
            author: Author name
            encoding: Text encoding

        Returns:
            Document UUID
        """
        logger.info(f"Processing {len(sentences)} sentences for '{document_title}'")

        with get_db_session() as session:
            # Create document
            document = create_document(
                title=document_title,
                language=language,
                author=author,
                session=session
            )

            # Process each sentence
            for position, sentence_text in enumerate(sentences):
                if not sentence_text.strip():
                    continue

                # Create sentence
                sentence = create_sentence(
                    doc_id=document.doc_id,
                    position=position,
                    original_text=sentence_text,
                    session=session
                )

                # Detect and store sandhi
                annotations = self.detector.detect_sandhi(sentence_text, encoding=encoding)
                for annotation in annotations:
                    self._store_annotation(annotation, sentence.sent_id, session)

            session.commit()

        logger.info(f"Stored document: {document.doc_id}")
        return document.doc_id

    def _store_annotation(
        self,
        annotation: SandhiAnnotation,
        sent_id: UUID,
        session
    ):
        """
        Store a single sandhi annotation to database

        Args:
            annotation: SandhiAnnotation object
            sent_id: Sentence UUID
            session: Database session
        """
        # Convert sandhi type enum
        db_sandhi_type = self._convert_sandhi_type(annotation.sandhi_type)

        # Convert annotation method
        method_map = {
            'sanskrit_parser': AnnotationMethod.RULE_BASED,
            'heritage_api': AnnotationMethod.RULE_BASED,
            'hybrid': AnnotationMethod.HYBRID,
            'manual': AnnotationMethod.MANUAL
        }
        db_method = method_map.get(
            annotation.annotation_method,
            AnnotationMethod.RULE_BASED
        )

        # Create annotation
        create_sandhi_annotation(
            sent_id=sent_id,
            surface_form=annotation.surface_form,
            word1=annotation.word1,
            word2=annotation.word2,
            sandhi_type=db_sandhi_type,
            sandhi_rule=annotation.sandhi_rule,
            position_start=annotation.position_start,
            position_end=annotation.position_end,
            confidence=annotation.confidence,
            annotation_method=db_method,
            metadata=annotation.metadata,
            session=session
        )

    def _convert_sandhi_type(self, sandhi_type: SandhiType) -> DBSandhiType:
        """Convert detector SandhiType to database SandhiType"""
        mapping = {
            SandhiType.VOWEL: DBSandhiType.VOWEL,
            SandhiType.VISARGA: DBSandhiType.VISARGA,
            SandhiType.CONSONANT: DBSandhiType.CONSONANT,
            SandhiType.OTHER: DBSandhiType.OTHER,
            SandhiType.UNKNOWN: DBSandhiType.OTHER
        }
        return mapping.get(sandhi_type, DBSandhiType.OTHER)

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences

        Args:
            text: Full text

        Returns:
            List of sentences

        Note: Simple implementation - could be improved with
        better sentence boundary detection
        """
        # For Sanskrit, we can split on common delimiters
        # This is simplified - production would use proper tokenization

        # Split on pipe, newline, or double space
        sentences = []

        for line in text.split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            # Split on pipe (common in Sanskrit texts)
            parts = line.split('|')
            for part in parts:
                part = part.strip()
                if part:
                    sentences.append(part)

        return sentences


def store_sandhi_annotations(
    text: str,
    title: str,
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function to process and store sandhi annotations

    Args:
        text: Text to process
        title: Document title
        **kwargs: Additional arguments for process_and_store_text

    Returns:
        Processing statistics
    """
    storage = SandhiAnnotationStorage()
    return storage.process_and_store_text(text, title, **kwargs)
