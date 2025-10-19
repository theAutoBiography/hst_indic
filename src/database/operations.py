"""
CRUD operations for database models
Provides high-level functions for creating, reading, updating, and deleting records
"""
import logging
from typing import List, Optional, Dict, Any
from uuid import UUID
from datetime import datetime

from sqlalchemy import and_, or_, func
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from src.database.models import (
    Document, Sentence, SandhiAnnotation, SamasaAnnotation, TaddhitaAnnotation,
    LanguageType, AnnotationMethod, SandhiType, SamasaType
)
from src.database.connection import get_db_session

logger = logging.getLogger(__name__)


# =====================================================
# DOCUMENT OPERATIONS
# =====================================================

def create_document(
    title: str,
    language: LanguageType,
    author: Optional[str] = None,
    period: Optional[str] = None,
    genre: Optional[str] = None,
    source_url: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    session: Optional[Session] = None
) -> Document:
    """Create a new document"""
    doc = Document(
        title=title,
        author=author,
        language=language,
        period=period,
        genre=genre,
        source_url=source_url,
        metadata=metadata
    )

    if session:
        session.add(doc)
        session.flush()
        return doc
    else:
        with get_db_session() as sess:
            sess.add(doc)
            sess.commit()
            sess.refresh(doc)
            return doc


def get_document(doc_id: UUID, session: Optional[Session] = None) -> Optional[Document]:
    """Get a document by ID"""
    if session:
        return session.query(Document).filter(Document.doc_id == doc_id).first()
    else:
        with get_db_session() as sess:
            return sess.query(Document).filter(Document.doc_id == doc_id).first()


def get_documents_by_language(
    language: LanguageType,
    limit: int = 100,
    offset: int = 0,
    session: Optional[Session] = None
) -> List[Document]:
    """Get all documents for a specific language"""
    if session:
        return session.query(Document)\
            .filter(Document.language == language)\
            .offset(offset)\
            .limit(limit)\
            .all()
    else:
        with get_db_session() as sess:
            return sess.query(Document)\
                .filter(Document.language == language)\
                .offset(offset)\
                .limit(limit)\
                .all()


def update_document(
    doc_id: UUID,
    updates: Dict[str, Any],
    session: Optional[Session] = None
) -> Optional[Document]:
    """Update a document"""
    if session:
        doc = session.query(Document).filter(Document.doc_id == doc_id).first()
        if doc:
            for key, value in updates.items():
                setattr(doc, key, value)
            session.flush()
        return doc
    else:
        with get_db_session() as sess:
            doc = sess.query(Document).filter(Document.doc_id == doc_id).first()
            if doc:
                for key, value in updates.items():
                    setattr(doc, key, value)
                sess.commit()
                sess.refresh(doc)
            return doc


def delete_document(doc_id: UUID, session: Optional[Session] = None) -> bool:
    """Delete a document (cascades to sentences and annotations)"""
    try:
        if session:
            doc = session.query(Document).filter(Document.doc_id == doc_id).first()
            if doc:
                session.delete(doc)
                session.flush()
                return True
            return False
        else:
            with get_db_session() as sess:
                doc = sess.query(Document).filter(Document.doc_id == doc_id).first()
                if doc:
                    sess.delete(doc)
                    sess.commit()
                    return True
                return False
    except SQLAlchemyError as e:
        logger.error(f"Error deleting document {doc_id}: {e}")
        return False


# =====================================================
# SENTENCE OPERATIONS
# =====================================================

def create_sentence(
    doc_id: UUID,
    position: int,
    original_text: str,
    normalized_text: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    session: Optional[Session] = None
) -> Sentence:
    """Create a new sentence"""
    sentence = Sentence(
        doc_id=doc_id,
        position=position,
        original_text=original_text,
        normalized_text=normalized_text or original_text,
        word_count=len(original_text.split()),
        char_count=len(original_text),
        metadata=metadata
    )

    if session:
        session.add(sentence)
        session.flush()
        return sentence
    else:
        with get_db_session() as sess:
            sess.add(sentence)
            sess.commit()
            sess.refresh(sentence)
            return sentence


def bulk_create_sentences(
    sentences: List[Dict[str, Any]],
    session: Optional[Session] = None
) -> List[Sentence]:
    """Bulk create multiple sentences for better performance"""
    sentence_objs = []
    for sent_data in sentences:
        sent = Sentence(
            doc_id=sent_data['doc_id'],
            position=sent_data['position'],
            original_text=sent_data['original_text'],
            normalized_text=sent_data.get('normalized_text', sent_data['original_text']),
            word_count=len(sent_data['original_text'].split()),
            char_count=len(sent_data['original_text']),
            metadata=sent_data.get('metadata')
        )
        sentence_objs.append(sent)

    if session:
        session.bulk_save_objects(sentence_objs)
        session.flush()
        return sentence_objs
    else:
        with get_db_session() as sess:
            sess.bulk_save_objects(sentence_objs)
            sess.commit()
            return sentence_objs


def get_sentences_by_document(
    doc_id: UUID,
    session: Optional[Session] = None
) -> List[Sentence]:
    """Get all sentences for a document, ordered by position"""
    if session:
        return session.query(Sentence)\
            .filter(Sentence.doc_id == doc_id)\
            .order_by(Sentence.position)\
            .all()
    else:
        with get_db_session() as sess:
            return sess.query(Sentence)\
                .filter(Sentence.doc_id == doc_id)\
                .order_by(Sentence.position)\
                .all()


# =====================================================
# SANDHI ANNOTATION OPERATIONS
# =====================================================

def create_sandhi_annotation(
    sent_id: UUID,
    surface_form: str,
    word1: str,
    word2: str,
    sandhi_type: SandhiType,
    position_start: int,
    position_end: int,
    sandhi_rule: Optional[str] = None,
    confidence: Optional[float] = None,
    annotation_method: AnnotationMethod = AnnotationMethod.RULE_BASED,
    metadata: Optional[Dict[str, Any]] = None,
    session: Optional[Session] = None
) -> SandhiAnnotation:
    """Create a new sandhi annotation"""
    annotation = SandhiAnnotation(
        sent_id=sent_id,
        surface_form=surface_form,
        word1=word1,
        word2=word2,
        sandhi_type=sandhi_type,
        sandhi_rule=sandhi_rule,
        position_start=position_start,
        position_end=position_end,
        confidence=confidence,
        annotation_method=annotation_method,
        metadata=metadata
    )

    if session:
        session.add(annotation)
        session.flush()
        return annotation
    else:
        with get_db_session() as sess:
            sess.add(annotation)
            sess.commit()
            sess.refresh(annotation)
            return annotation


def get_sandhi_annotations_by_sentence(
    sent_id: UUID,
    session: Optional[Session] = None
) -> List[SandhiAnnotation]:
    """Get all sandhi annotations for a sentence"""
    if session:
        return session.query(SandhiAnnotation)\
            .filter(SandhiAnnotation.sent_id == sent_id)\
            .all()
    else:
        with get_db_session() as sess:
            return sess.query(SandhiAnnotation)\
                .filter(SandhiAnnotation.sent_id == sent_id)\
                .all()


def get_annotations_by_document(
    doc_id: UUID,
    session: Optional[Session] = None
) -> List[SandhiAnnotation]:
    """Get all sandhi annotations for a document (across all sentences)"""
    if session:
        return session.query(SandhiAnnotation)\
            .join(Sentence)\
            .filter(Sentence.doc_id == doc_id)\
            .order_by(Sentence.position, SandhiAnnotation.position_start)\
            .all()
    else:
        with get_db_session() as sess:
            return sess.query(SandhiAnnotation)\
                .join(Sentence)\
                .filter(Sentence.doc_id == doc_id)\
                .order_by(Sentence.position, SandhiAnnotation.position_start)\
                .all()


def get_unverified_sandhi_annotations(
    limit: int = 100,
    session: Optional[Session] = None
) -> List[SandhiAnnotation]:
    """Get unverified sandhi annotations for human review"""
    if session:
        return session.query(SandhiAnnotation)\
            .filter(SandhiAnnotation.is_verified == False)\
            .order_by(SandhiAnnotation.created_at)\
            .limit(limit)\
            .all()
    else:
        with get_db_session() as sess:
            return sess.query(SandhiAnnotation)\
                .filter(SandhiAnnotation.is_verified == False)\
                .order_by(SandhiAnnotation.created_at)\
                .limit(limit)\
                .all()


def verify_annotation(
    annotation_id: UUID,
    annotation_type: str,
    verified_by: str,
    session: Optional[Session] = None
) -> bool:
    """Verify an annotation (works for sandhi, samasa, taddhita)"""
    model_map = {
        'sandhi': SandhiAnnotation,
        'samasa': SamasaAnnotation,
        'taddhita': TaddhitaAnnotation
    }

    model = model_map.get(annotation_type)
    if not model:
        logger.error(f"Invalid annotation type: {annotation_type}")
        return False

    try:
        if session:
            annotation = session.query(model).filter(model.annotation_id == annotation_id).first()
            if annotation:
                annotation.is_verified = True
                annotation.verified_by = verified_by
                annotation.verified_at = datetime.utcnow()
                session.flush()
                return True
            return False
        else:
            with get_db_session() as sess:
                annotation = sess.query(model).filter(model.annotation_id == annotation_id).first()
                if annotation:
                    annotation.is_verified = True
                    annotation.verified_by = verified_by
                    annotation.verified_at = datetime.utcnow()
                    sess.commit()
                    return True
                return False
    except SQLAlchemyError as e:
        logger.error(f"Error verifying annotation {annotation_id}: {e}")
        return False


# =====================================================
# SAMASA ANNOTATION OPERATIONS
# =====================================================

def create_samasa_annotation(
    sent_id: UUID,
    compound: str,
    components: List[str],
    position_start: int,
    position_end: int,
    samasa_type: Optional[SamasaType] = None,
    confidence: Optional[float] = None,
    annotation_method: AnnotationMethod = AnnotationMethod.RULE_BASED,
    metadata: Optional[Dict[str, Any]] = None,
    session: Optional[Session] = None
) -> SamasaAnnotation:
    """Create a new samasa annotation"""
    annotation = SamasaAnnotation(
        sent_id=sent_id,
        compound=compound,
        components=components,
        samasa_type=samasa_type,
        position_start=position_start,
        position_end=position_end,
        confidence=confidence,
        annotation_method=annotation_method,
        metadata=metadata
    )

    if session:
        session.add(annotation)
        session.flush()
        return annotation
    else:
        with get_db_session() as sess:
            sess.add(annotation)
            sess.commit()
            sess.refresh(annotation)
            return annotation


# =====================================================
# TADDHITA ANNOTATION OPERATIONS
# =====================================================

def create_taddhita_annotation(
    sent_id: UUID,
    derived_word: str,
    root_word: str,
    suffix: str,
    position_start: int,
    position_end: int,
    derivation_type: Optional[str] = None,
    confidence: Optional[float] = None,
    annotation_method: AnnotationMethod = AnnotationMethod.RULE_BASED,
    metadata: Optional[Dict[str, Any]] = None,
    session: Optional[Session] = None
) -> TaddhitaAnnotation:
    """Create a new taddhita annotation"""
    annotation = TaddhitaAnnotation(
        sent_id=sent_id,
        derived_word=derived_word,
        root_word=root_word,
        suffix=suffix,
        derivation_type=derivation_type,
        position_start=position_start,
        position_end=position_end,
        confidence=confidence,
        annotation_method=annotation_method,
        metadata=metadata
    )

    if session:
        session.add(annotation)
        session.flush()
        return annotation
    else:
        with get_db_session() as sess:
            sess.add(annotation)
            sess.commit()
            sess.refresh(annotation)
            return annotation


# =====================================================
# STATISTICS AND ANALYTICS
# =====================================================

def get_annotation_statistics(
    doc_id: Optional[UUID] = None,
    language: Optional[LanguageType] = None,
    session: Optional[Session] = None
) -> Dict[str, Any]:
    """Get annotation statistics for a document or language"""
    if session:
        return _compute_stats(doc_id, language, session)
    else:
        with get_db_session() as sess:
            return _compute_stats(doc_id, language, sess)


def _compute_stats(
    doc_id: Optional[UUID],
    language: Optional[LanguageType],
    session: Session
) -> Dict[str, Any]:
    """Internal function to compute statistics"""
    query = session.query(Document)

    if doc_id:
        query = query.filter(Document.doc_id == doc_id)
    if language:
        query = query.filter(Document.language == language)

    documents = query.all()

    stats = {
        'total_documents': len(documents),
        'total_sentences': 0,
        'total_sandhi_annotations': 0,
        'total_samasa_annotations': 0,
        'total_taddhita_annotations': 0,
        'verified_sandhi': 0,
        'verified_samasa': 0,
        'verified_taddhita': 0,
    }

    for doc in documents:
        stats['total_sentences'] += len(doc.sentences)
        for sent in doc.sentences:
            stats['total_sandhi_annotations'] += len(sent.sandhi_annotations)
            stats['total_samasa_annotations'] += len(sent.samasa_annotations)
            stats['total_taddhita_annotations'] += len(sent.taddhita_annotations)

            stats['verified_sandhi'] += sum(1 for a in sent.sandhi_annotations if a.is_verified)
            stats['verified_samasa'] += sum(1 for a in sent.samasa_annotations if a.is_verified)
            stats['verified_taddhita'] += sum(1 for a in sent.taddhita_annotations if a.is_verified)

    return stats


# =====================================================
# BATCH OPERATIONS
# =====================================================

def bulk_create_annotations(
    annotations: List[Dict[str, Any]],
    annotation_type: str,
    session: Optional[Session] = None
) -> bool:
    """Bulk create annotations for better performance"""
    model_map = {
        'sandhi': SandhiAnnotation,
        'samasa': SamasaAnnotation,
        'taddhita': TaddhitaAnnotation
    }

    model = model_map.get(annotation_type)
    if not model:
        logger.error(f"Invalid annotation type: {annotation_type}")
        return False

    try:
        annotation_objs = [model(**ann) for ann in annotations]

        if session:
            session.bulk_save_objects(annotation_objs)
            session.flush()
            return True
        else:
            with get_db_session() as sess:
                sess.bulk_save_objects(annotation_objs)
                sess.commit()
                return True
    except SQLAlchemyError as e:
        logger.error(f"Error bulk creating annotations: {e}")
        return False
