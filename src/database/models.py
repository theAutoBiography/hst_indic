"""
SQLAlchemy ORM Models for Indic Annotation Pipeline
"""
from datetime import datetime
from typing import Optional, List
from uuid import uuid4

from sqlalchemy import (
    Column, String, Integer, Text, TIMESTAMP, Boolean, Numeric,
    ForeignKey, CheckConstraint, Index, Enum as SQLEnum, ARRAY
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.sql import func
import enum

# Create base class for all models
Base = declarative_base()


# =====================================================
# ENUMS
# =====================================================
class LanguageType(str, enum.Enum):
    """Supported languages"""
    SANSKRIT = "sanskrit"
    TAMIL = "tamil"
    GUJARATI = "gujarati"


class AnnotationMethod(str, enum.Enum):
    """Method used for annotation"""
    RULE_BASED = "rule_based"
    ML_MODEL = "ml_model"
    MANUAL = "manual"
    HYBRID = "hybrid"


class SandhiType(str, enum.Enum):
    """Types of sandhi"""
    VOWEL = "vowel"
    VISARGA = "visarga"
    CONSONANT = "consonant"
    OTHER = "other"


class SamasaType(str, enum.Enum):
    """Types of samasa (compounds)"""
    DVANDVA = "dvandva"
    TATPURUSHA = "tatpurusha"
    BAHUVRIHI = "bahuvrihi"
    AVYAYIBHAVA = "avyayibhava"
    KARMADHARAYA = "karmadharaya"
    DVIGU = "dvigu"


# =====================================================
# MODELS
# =====================================================
class Document(Base):
    """
    Represents a source document for annotation
    """
    __tablename__ = "documents"

    doc_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    title = Column(String(500), nullable=False)
    author = Column(String(255))
    language = Column(SQLEnum(LanguageType, name="language_type"), nullable=False)
    period = Column(String(100))
    genre = Column(String(100))
    source_url = Column(Text)
    meta_data = Column('metadata', JSONB)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    sentences = relationship("Sentence", back_populates="document", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Document(doc_id={self.doc_id}, title='{self.title}', language='{self.language}')>"

    def to_dict(self):
        """Convert model to dictionary"""
        return {
            "doc_id": str(self.doc_id),
            "title": self.title,
            "author": self.author,
            "language": self.language.value if self.language else None,
            "period": self.period,
            "genre": self.genre,
            "source_url": self.source_url,
            "metadata": self.meta_data,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class Sentence(Base):
    """
    Represents a sentence from a document
    """
    __tablename__ = "sentences"

    sent_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    doc_id = Column(UUID(as_uuid=True), ForeignKey("documents.doc_id", ondelete="CASCADE"), nullable=False)
    position = Column(Integer, nullable=False)
    original_text = Column(Text, nullable=False)
    normalized_text = Column(Text)
    word_count = Column(Integer)
    char_count = Column(Integer)
    meta_data = Column('metadata', JSONB)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

    # Relationships
    document = relationship("Document", back_populates="sentences")
    sandhi_annotations = relationship("SandhiAnnotation", back_populates="sentence", cascade="all, delete-orphan")
    samasa_annotations = relationship("SamasaAnnotation", back_populates="sentence", cascade="all, delete-orphan")
    taddhita_annotations = relationship("TaddhitaAnnotation", back_populates="sentence", cascade="all, delete-orphan")

    # Constraints
    __table_args__ = (
        CheckConstraint("position >= 0", name="positive_position"),
        Index("idx_sentences_doc_id", "doc_id"),
        Index("idx_sentences_position", "doc_id", "position"),
        Index("idx_sentences_word_count", "word_count"),
    )

    def __repr__(self):
        text_preview = self.original_text[:50] + "..." if len(self.original_text) > 50 else self.original_text
        return f"<Sentence(sent_id={self.sent_id}, position={self.position}, text='{text_preview}')>"

    def to_dict(self):
        """Convert model to dictionary"""
        return {
            "sent_id": str(self.sent_id),
            "doc_id": str(self.doc_id),
            "position": self.position,
            "original_text": self.original_text,
            "normalized_text": self.normalized_text,
            "word_count": self.word_count,
            "char_count": self.char_count,
            "metadata": self.meta_data,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class SandhiAnnotation(Base):
    """
    Represents a sandhi (phonetic combination) annotation
    """
    __tablename__ = "sandhi_annotations"

    annotation_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    sent_id = Column(UUID(as_uuid=True), ForeignKey("sentences.sent_id", ondelete="CASCADE"), nullable=False)
    surface_form = Column(Text, nullable=False)
    word1 = Column(Text, nullable=False)
    word2 = Column(Text, nullable=False)
    sandhi_type = Column(SQLEnum(SandhiType, name="sandhi_type"), nullable=False)
    sandhi_rule = Column(Text)
    position_start = Column(Integer, nullable=False)
    position_end = Column(Integer, nullable=False)
    confidence = Column(Numeric(3, 2))
    annotation_method = Column(SQLEnum(AnnotationMethod, name="annotation_method"),
                               nullable=False, default=AnnotationMethod.RULE_BASED)
    verified_by = Column(String(255))
    verified_at = Column(TIMESTAMP(timezone=True))
    is_verified = Column(Boolean, default=False)
    notes = Column(Text)
    meta_data = Column('metadata', JSONB)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    sentence = relationship("Sentence", back_populates="sandhi_annotations")

    # Constraints
    __table_args__ = (
        CheckConstraint("confidence >= 0 AND confidence <= 1", name="valid_confidence"),
        CheckConstraint("position_start >= 0 AND position_end > position_start", name="valid_position"),
        Index("idx_sandhi_sent_id", "sent_id"),
        Index("idx_sandhi_type", "sandhi_type"),
        Index("idx_sandhi_confidence", "confidence"),
        Index("idx_sandhi_method", "annotation_method"),
        Index("idx_sandhi_verified", "is_verified"),
        Index("idx_sandhi_surface_form", "surface_form"),
    )

    def __repr__(self):
        return f"<SandhiAnnotation(id={self.annotation_id}, surface='{self.surface_form}', " \
               f"split='{self.word1}+{self.word2}', type={self.sandhi_type})>"

    def to_dict(self):
        """Convert model to dictionary"""
        return {
            "annotation_id": str(self.annotation_id),
            "sent_id": str(self.sent_id),
            "surface_form": self.surface_form,
            "word1": self.word1,
            "word2": self.word2,
            "sandhi_type": self.sandhi_type.value if self.sandhi_type else None,
            "sandhi_rule": self.sandhi_rule,
            "position_start": self.position_start,
            "position_end": self.position_end,
            "confidence": float(self.confidence) if self.confidence else None,
            "annotation_method": self.annotation_method.value if self.annotation_method else None,
            "verified_by": self.verified_by,
            "verified_at": self.verified_at.isoformat() if self.verified_at else None,
            "is_verified": self.is_verified,
            "notes": self.notes,
            "metadata": self.meta_data,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class SamasaAnnotation(Base):
    """
    Represents a samasa (compound) annotation
    """
    __tablename__ = "samasa_annotations"

    annotation_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    sent_id = Column(UUID(as_uuid=True), ForeignKey("sentences.sent_id", ondelete="CASCADE"), nullable=False)
    compound = Column(Text, nullable=False)
    components = Column(ARRAY(Text), nullable=False)
    samasa_type = Column(SQLEnum(SamasaType, name="samasa_type"))
    position_start = Column(Integer, nullable=False)
    position_end = Column(Integer, nullable=False)
    confidence = Column(Numeric(3, 2))
    annotation_method = Column(SQLEnum(AnnotationMethod, name="annotation_method"),
                               nullable=False, default=AnnotationMethod.RULE_BASED)
    verified_by = Column(String(255))
    verified_at = Column(TIMESTAMP(timezone=True))
    is_verified = Column(Boolean, default=False)
    notes = Column(Text)
    meta_data = Column('metadata', JSONB)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    sentence = relationship("Sentence", back_populates="samasa_annotations")

    # Constraints
    __table_args__ = (
        CheckConstraint("confidence >= 0 AND confidence <= 1", name="valid_confidence"),
        CheckConstraint("position_start >= 0 AND position_end > position_start", name="valid_position"),
        Index("idx_samasa_sent_id", "sent_id"),
        Index("idx_samasa_type", "samasa_type"),
        Index("idx_samasa_confidence", "confidence"),
        Index("idx_samasa_method", "annotation_method"),
        Index("idx_samasa_verified", "is_verified"),
        Index("idx_samasa_compound", "compound"),
    )

    def __repr__(self):
        components_str = "+".join(self.components) if self.components else "[]"
        return f"<SamasaAnnotation(id={self.annotation_id}, compound='{self.compound}', " \
               f"components='{components_str}', type={self.samasa_type})>"

    def to_dict(self):
        """Convert model to dictionary"""
        return {
            "annotation_id": str(self.annotation_id),
            "sent_id": str(self.sent_id),
            "compound": self.compound,
            "components": self.components,
            "samasa_type": self.samasa_type.value if self.samasa_type else None,
            "position_start": self.position_start,
            "position_end": self.position_end,
            "confidence": float(self.confidence) if self.confidence else None,
            "annotation_method": self.annotation_method.value if self.annotation_method else None,
            "verified_by": self.verified_by,
            "verified_at": self.verified_at.isoformat() if self.verified_at else None,
            "is_verified": self.is_verified,
            "notes": self.notes,
            "metadata": self.meta_data,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class TaddhitaAnnotation(Base):
    """
    Represents a taddhita (derived word) annotation
    """
    __tablename__ = "taddhita_annotations"

    annotation_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    sent_id = Column(UUID(as_uuid=True), ForeignKey("sentences.sent_id", ondelete="CASCADE"), nullable=False)
    derived_word = Column(Text, nullable=False)
    root_word = Column(Text, nullable=False)
    suffix = Column(Text, nullable=False)
    derivation_type = Column(Text)
    position_start = Column(Integer, nullable=False)
    position_end = Column(Integer, nullable=False)
    confidence = Column(Numeric(3, 2))
    annotation_method = Column(SQLEnum(AnnotationMethod, name="annotation_method"),
                               nullable=False, default=AnnotationMethod.RULE_BASED)
    verified_by = Column(String(255))
    verified_at = Column(TIMESTAMP(timezone=True))
    is_verified = Column(Boolean, default=False)
    notes = Column(Text)
    meta_data = Column('metadata', JSONB)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    sentence = relationship("Sentence", back_populates="taddhita_annotations")

    # Constraints
    __table_args__ = (
        CheckConstraint("confidence >= 0 AND confidence <= 1", name="valid_confidence"),
        CheckConstraint("position_start >= 0 AND position_end > position_start", name="valid_position"),
        Index("idx_taddhita_sent_id", "sent_id"),
        Index("idx_taddhita_confidence", "confidence"),
        Index("idx_taddhita_method", "annotation_method"),
        Index("idx_taddhita_verified", "is_verified"),
        Index("idx_taddhita_derived", "derived_word"),
        Index("idx_taddhita_root", "root_word"),
        Index("idx_taddhita_suffix", "suffix"),
        Index("idx_taddhita_type", "derivation_type"),
    )

    def __repr__(self):
        return f"<TaddhitaAnnotation(id={self.annotation_id}, derived='{self.derived_word}', " \
               f"root='{self.root_word}', suffix='{self.suffix}')>"

    def to_dict(self):
        """Convert model to dictionary"""
        return {
            "annotation_id": str(self.annotation_id),
            "sent_id": str(self.sent_id),
            "derived_word": self.derived_word,
            "root_word": self.root_word,
            "suffix": self.suffix,
            "derivation_type": self.derivation_type,
            "position_start": self.position_start,
            "position_end": self.position_end,
            "confidence": float(self.confidence) if self.confidence else None,
            "annotation_method": self.annotation_method.value if self.annotation_method else None,
            "verified_by": self.verified_by,
            "verified_at": self.verified_at.isoformat() if self.verified_at else None,
            "is_verified": self.is_verified,
            "notes": self.notes,
            "metadata": self.meta_data,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
