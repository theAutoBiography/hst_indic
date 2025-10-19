-- Database Schema for Indic Annotation Pipeline
-- PostgreSQL 12+

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create ENUM types for better data integrity
CREATE TYPE language_type AS ENUM ('sanskrit', 'tamil', 'gujarati');
CREATE TYPE annotation_method AS ENUM ('rule_based', 'ml_model', 'manual', 'hybrid');
CREATE TYPE sandhi_type AS ENUM ('vowel', 'visarga', 'consonant', 'other');
CREATE TYPE samasa_type AS ENUM ('dvandva', 'tatpurusha', 'bahuvrihi', 'avyayibhava', 'karmadharaya', 'dvigu');

-- =====================================================
-- 1. DOCUMENTS TABLE
-- =====================================================
CREATE TABLE documents (
    doc_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title VARCHAR(500) NOT NULL,
    author VARCHAR(255),
    language language_type NOT NULL,
    period VARCHAR(100),  -- e.g., "Classical", "Vedic", "Modern"
    genre VARCHAR(100),   -- e.g., "Poetry", "Prose", "Scripture"
    source_url TEXT,
    metadata JSONB,       -- Additional flexible metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    -- Indexes
    CONSTRAINT valid_url CHECK (source_url IS NULL OR source_url ~ '^https?://')
);

CREATE INDEX idx_documents_language ON documents(language);
CREATE INDEX idx_documents_author ON documents(author);
CREATE INDEX idx_documents_period ON documents(period);
CREATE INDEX idx_documents_genre ON documents(genre);
CREATE INDEX idx_documents_created_at ON documents(created_at);
CREATE INDEX idx_documents_metadata ON documents USING GIN (metadata);

-- =====================================================
-- 2. SENTENCES TABLE
-- =====================================================
CREATE TABLE sentences (
    sent_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    doc_id UUID NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    position INTEGER NOT NULL,  -- Position within the document
    original_text TEXT NOT NULL,
    normalized_text TEXT,       -- Preprocessed/normalized version
    word_count INTEGER,
    char_count INTEGER,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    -- Constraints
    CONSTRAINT positive_position CHECK (position >= 0),
    CONSTRAINT unique_doc_position UNIQUE (doc_id, position)
);

CREATE INDEX idx_sentences_doc_id ON sentences(doc_id);
CREATE INDEX idx_sentences_position ON sentences(doc_id, position);
CREATE INDEX idx_sentences_word_count ON sentences(word_count);
CREATE INDEX idx_sentences_metadata ON sentences USING GIN (metadata);

-- =====================================================
-- 3. SANDHI ANNOTATIONS TABLE
-- =====================================================
CREATE TABLE sandhi_annotations (
    annotation_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    sent_id UUID NOT NULL REFERENCES sentences(sent_id) ON DELETE CASCADE,
    surface_form TEXT NOT NULL,         -- The combined form (e.g., "rāmo 'sti")
    word1 TEXT NOT NULL,                -- First word before sandhi (e.g., "rāmaḥ")
    word2 TEXT NOT NULL,                -- Second word before sandhi (e.g., "asti")
    sandhi_type sandhi_type NOT NULL,   -- Type of sandhi
    sandhi_rule TEXT,                   -- Specific rule applied (e.g., "aḥ + a → o 'a")
    position_start INTEGER NOT NULL,    -- Character position in sentence
    position_end INTEGER NOT NULL,
    confidence NUMERIC(3, 2) CHECK (confidence >= 0 AND confidence <= 1),
    annotation_method annotation_method NOT NULL DEFAULT 'rule_based',
    verified_by VARCHAR(255),           -- Username or ID of verifier
    verified_at TIMESTAMP WITH TIME ZONE,
    is_verified BOOLEAN DEFAULT FALSE,
    notes TEXT,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    -- Constraints
    CONSTRAINT valid_position CHECK (position_start >= 0 AND position_end > position_start)
);

CREATE INDEX idx_sandhi_sent_id ON sandhi_annotations(sent_id);
CREATE INDEX idx_sandhi_type ON sandhi_annotations(sandhi_type);
CREATE INDEX idx_sandhi_confidence ON sandhi_annotations(confidence);
CREATE INDEX idx_sandhi_method ON sandhi_annotations(annotation_method);
CREATE INDEX idx_sandhi_verified ON sandhi_annotations(is_verified);
CREATE INDEX idx_sandhi_surface_form ON sandhi_annotations(surface_form);
CREATE INDEX idx_sandhi_metadata ON sandhi_annotations USING GIN (metadata);

-- =====================================================
-- 4. SAMASA ANNOTATIONS TABLE
-- =====================================================
CREATE TABLE samasa_annotations (
    annotation_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    sent_id UUID NOT NULL REFERENCES sentences(sent_id) ON DELETE CASCADE,
    compound TEXT NOT NULL,             -- The compound word
    components TEXT[] NOT NULL,         -- Array of component words
    samasa_type samasa_type,            -- Type of compound
    position_start INTEGER NOT NULL,
    position_end INTEGER NOT NULL,
    confidence NUMERIC(3, 2) CHECK (confidence >= 0 AND confidence <= 1),
    annotation_method annotation_method NOT NULL DEFAULT 'rule_based',
    verified_by VARCHAR(255),
    verified_at TIMESTAMP WITH TIME ZONE,
    is_verified BOOLEAN DEFAULT FALSE,
    notes TEXT,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    -- Constraints
    CONSTRAINT valid_components CHECK (array_length(components, 1) >= 2),
    CONSTRAINT valid_position CHECK (position_start >= 0 AND position_end > position_start)
);

CREATE INDEX idx_samasa_sent_id ON samasa_annotations(sent_id);
CREATE INDEX idx_samasa_type ON samasa_annotations(samasa_type);
CREATE INDEX idx_samasa_confidence ON samasa_annotations(confidence);
CREATE INDEX idx_samasa_method ON samasa_annotations(annotation_method);
CREATE INDEX idx_samasa_verified ON samasa_annotations(is_verified);
CREATE INDEX idx_samasa_compound ON samasa_annotations(compound);
CREATE INDEX idx_samasa_components ON samasa_annotations USING GIN (components);
CREATE INDEX idx_samasa_metadata ON samasa_annotations USING GIN (metadata);

-- =====================================================
-- 5. TADDHITA ANNOTATIONS TABLE
-- =====================================================
CREATE TABLE taddhita_annotations (
    annotation_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    sent_id UUID NOT NULL REFERENCES sentences(sent_id) ON DELETE CASCADE,
    derived_word TEXT NOT NULL,         -- The word formed with suffix
    root_word TEXT NOT NULL,            -- Base word
    suffix TEXT NOT NULL,               -- The taddhita suffix
    derivation_type TEXT,               -- Type of derivation
    position_start INTEGER NOT NULL,
    position_end INTEGER NOT NULL,
    confidence NUMERIC(3, 2) CHECK (confidence >= 0 AND confidence <= 1),
    annotation_method annotation_method NOT NULL DEFAULT 'rule_based',
    verified_by VARCHAR(255),
    verified_at TIMESTAMP WITH TIME ZONE,
    is_verified BOOLEAN DEFAULT FALSE,
    notes TEXT,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    -- Constraints
    CONSTRAINT valid_position CHECK (position_start >= 0 AND position_end > position_start)
);

CREATE INDEX idx_taddhita_sent_id ON taddhita_annotations(sent_id);
CREATE INDEX idx_taddhita_confidence ON taddhita_annotations(confidence);
CREATE INDEX idx_taddhita_method ON taddhita_annotations(annotation_method);
CREATE INDEX idx_taddhita_verified ON taddhita_annotations(is_verified);
CREATE INDEX idx_taddhita_derived ON taddhita_annotations(derived_word);
CREATE INDEX idx_taddhita_root ON taddhita_annotations(root_word);
CREATE INDEX idx_taddhita_suffix ON taddhita_annotations(suffix);
CREATE INDEX idx_taddhita_type ON taddhita_annotations(derivation_type);
CREATE INDEX idx_taddhita_metadata ON taddhita_annotations USING GIN (metadata);

-- =====================================================
-- 6. TRIGGERS FOR UPDATED_AT
-- =====================================================
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_documents_updated_at BEFORE UPDATE ON documents
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_sandhi_updated_at BEFORE UPDATE ON sandhi_annotations
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_samasa_updated_at BEFORE UPDATE ON samasa_annotations
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_taddhita_updated_at BEFORE UPDATE ON taddhita_annotations
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- =====================================================
-- 7. VIEWS FOR COMMON QUERIES
-- =====================================================

-- View: All annotations with document context
CREATE VIEW all_annotations AS
SELECT
    'sandhi' as annotation_type,
    sa.annotation_id,
    d.doc_id,
    d.title as document_title,
    d.language,
    s.sent_id,
    s.original_text as sentence,
    sa.surface_form as text,
    sa.confidence,
    sa.annotation_method,
    sa.is_verified,
    sa.created_at
FROM sandhi_annotations sa
JOIN sentences s ON sa.sent_id = s.sent_id
JOIN documents d ON s.doc_id = d.doc_id

UNION ALL

SELECT
    'samasa' as annotation_type,
    sam.annotation_id,
    d.doc_id,
    d.title as document_title,
    d.language,
    s.sent_id,
    s.original_text as sentence,
    sam.compound as text,
    sam.confidence,
    sam.annotation_method,
    sam.is_verified,
    sam.created_at
FROM samasa_annotations sam
JOIN sentences s ON sam.sent_id = s.sent_id
JOIN documents d ON s.doc_id = d.doc_id

UNION ALL

SELECT
    'taddhita' as annotation_type,
    ta.annotation_id,
    d.doc_id,
    d.title as document_title,
    d.language,
    s.sent_id,
    s.original_text as sentence,
    ta.derived_word as text,
    ta.confidence,
    ta.annotation_method,
    ta.is_verified,
    ta.created_at
FROM taddhita_annotations ta
JOIN sentences s ON ta.sent_id = s.sent_id
JOIN documents d ON s.doc_id = d.doc_id;

-- View: Annotation statistics by document
CREATE VIEW document_annotation_stats AS
SELECT
    d.doc_id,
    d.title,
    d.language,
    COUNT(DISTINCT s.sent_id) as total_sentences,
    COUNT(DISTINCT sa.annotation_id) as sandhi_count,
    COUNT(DISTINCT sam.annotation_id) as samasa_count,
    COUNT(DISTINCT ta.annotation_id) as taddhita_count,
    AVG(sa.confidence) as avg_sandhi_confidence,
    AVG(sam.confidence) as avg_samasa_confidence,
    AVG(ta.confidence) as avg_taddhita_confidence
FROM documents d
LEFT JOIN sentences s ON d.doc_id = s.doc_id
LEFT JOIN sandhi_annotations sa ON s.sent_id = sa.sent_id
LEFT JOIN samasa_annotations sam ON s.sent_id = sam.sent_id
LEFT JOIN taddhita_annotations ta ON s.sent_id = ta.sent_id
GROUP BY d.doc_id, d.title, d.language;

-- =====================================================
-- 8. COMMENTS FOR DOCUMENTATION
-- =====================================================
COMMENT ON TABLE documents IS 'Stores source documents for annotation';
COMMENT ON TABLE sentences IS 'Individual sentences extracted from documents';
COMMENT ON TABLE sandhi_annotations IS 'Sandhi (phonetic combination) annotations';
COMMENT ON TABLE samasa_annotations IS 'Samasa (compound) annotations';
COMMENT ON TABLE taddhita_annotations IS 'Taddhita (derived word) annotations';

COMMENT ON COLUMN documents.metadata IS 'Flexible JSONB field for additional document properties';
COMMENT ON COLUMN sandhi_annotations.confidence IS 'Confidence score from 0.00 to 1.00';
COMMENT ON COLUMN samasa_annotations.components IS 'Array of component words forming the compound';
