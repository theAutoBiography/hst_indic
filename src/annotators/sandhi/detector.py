"""
Sanskrit Sandhi Detector
Wrapper around sanskrit_parser with dictionary validation and confidence scoring
"""
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# Suppress verbose logging from sanskrit_parser
logging.getLogger('sanskrit_parser').setLevel(logging.WARNING)
logging.getLogger('sanskrit_util').setLevel(logging.WARNING)

from sanskrit_parser import Parser
from indic_transliteration import sanscript

from src.preprocessing.dictionary import SanskritDictionary

logger = logging.getLogger(__name__)


class SandhiType(str, Enum):
    """Types of sandhi"""
    VOWEL = "vowel"
    VISARGA = "visarga"
    CONSONANT = "consonant"
    OTHER = "other"
    UNKNOWN = "unknown"


@dataclass
class SandhiAnnotation:
    """
    Represents a single sandhi annotation
    """
    surface_form: str  # Combined form (e.g., "rāmo 'sti")
    word1: str  # First word before sandhi (e.g., "rāmaḥ")
    word2: str  # Second word before sandhi (e.g., "asti")
    sandhi_type: SandhiType
    sandhi_rule: Optional[str]  # Rule description
    position_start: int
    position_end: int
    confidence: float  # 0.0 to 1.0
    annotation_method: str  # 'sanskrit_parser', 'heritage_api', 'hybrid'
    metadata: Dict

    def to_dict(self) -> Dict:
        """Convert to dictionary for database storage"""
        return {
            'surface_form': self.surface_form,
            'word1': self.word1,
            'word2': self.word2,
            'sandhi_type': self.sandhi_type.value,
            'sandhi_rule': self.sandhi_rule,
            'position_start': self.position_start,
            'position_end': self.position_end,
            'confidence': self.confidence,
            'annotation_method': self.annotation_method,
            'metadata': self.metadata
        }


class SanskritSandhiDetector:
    """
    Sanskrit Sandhi Detector using sanskrit_parser with dictionary validation

    Features:
    - Automatic sandhi splitting using sanskrit_parser
    - Dictionary validation for split words
    - Confidence scoring based on multiple factors
    - Support for batch processing
    """

    def __init__(
        self,
        use_dictionary_validation: bool = True,
        min_confidence: float = 0.5,
        dictionary_dir: Optional[str] = None
    ):
        """
        Initialize sandhi detector

        Args:
            use_dictionary_validation: Whether to validate splits with dictionary
            min_confidence: Minimum confidence threshold (0.0 to 1.0)
            dictionary_dir: Path to dictionary directory
        """
        self.use_dictionary_validation = use_dictionary_validation
        self.min_confidence = min_confidence

        # Initialize sanskrit_parser
        logger.info("Initializing sanskrit_parser...")
        self.parser = Parser()

        # Initialize dictionary
        if use_dictionary_validation:
            logger.info("Loading Sanskrit dictionary...")
            self.dictionary = SanskritDictionary(dictionary_dir=dictionary_dir)
        else:
            self.dictionary = None

        logger.info("SanskritSandhiDetector initialized successfully")

    def detect_sandhi(
        self,
        sentence: str,
        encoding: str = 'IAST',
        limit: int = 5
    ) -> List[SandhiAnnotation]:
        """
        Detect sandhi in a Sanskrit sentence

        Args:
            sentence: Sanskrit text to analyze
            encoding: Input encoding (IAST, Devanagari, SLP1, etc.)
            limit: Maximum number of split alternatives to consider

        Returns:
            List of SandhiAnnotation objects
        """
        logger.debug(f"Detecting sandhi in: {sentence}")

        # Convert to SLP1 (internal format for sanskrit_parser)
        if encoding != 'SLP1':
            sentence_slp1 = self._transliterate(sentence, encoding, 'SLP1')
        else:
            sentence_slp1 = sentence

        # Get splits from sanskrit_parser
        try:
            splits = self.parser.split(sentence_slp1, limit=limit)
        except Exception as e:
            logger.error(f"Error in sanskrit_parser: {e}")
            return []

        if not splits:
            logger.debug("No sandhi splits found (single valid word)")
            return []

        # Process each split
        annotations = []
        for split_graph in splits:
            try:
                # Extract split information
                annotation = self._process_split(
                    split_graph,
                    sentence,
                    sentence_slp1,
                    encoding
                )

                if annotation and annotation.confidence >= self.min_confidence:
                    annotations.append(annotation)

            except Exception as e:
                logger.warning(f"Error processing split: {e}")
                continue

        # Sort by confidence (highest first)
        annotations.sort(key=lambda x: x.confidence, reverse=True)

        logger.info(f"Found {len(annotations)} sandhi annotations above confidence threshold")
        return annotations

    def _process_split(
        self,
        split_graph,
        original_sentence: str,
        sentence_slp1: str,
        encoding: str
    ) -> Optional[SandhiAnnotation]:
        """
        Process a single split from sanskrit_parser

        Args:
            split_graph: Split graph from sanskrit_parser
            original_sentence: Original input sentence
            sentence_slp1: Sentence in SLP1 format
            encoding: Original encoding

        Returns:
            SandhiAnnotation or None
        """
        # Extract words from split graph
        # Note: sanskrit_parser returns complex graph objects
        # We need to extract the actual word forms

        try:
            # Get string representation of split
            split_str = str(split_graph)

            # Extract word components (simplified - may need refinement)
            # The split graph contains morphological analysis
            # For now, we'll extract basic word forms

            words = self._extract_words_from_split(split_graph)

            if len(words) < 2:
                return None

            word1_slp1 = words[0]
            word2_slp1 = words[1] if len(words) > 1 else ""

            # Convert back to original encoding
            word1 = self._transliterate(word1_slp1, 'SLP1', encoding)
            word2 = self._transliterate(word2_slp1, 'SLP1', encoding) if word2_slp1 else ""

            # Dictionary validation
            dict_confidence = 0.0
            if self.use_dictionary_validation and self.dictionary:
                dict_confidence = self._validate_with_dictionary(word1, word2)
            else:
                dict_confidence = 0.7  # Default if no dictionary

            # Determine sandhi type
            sandhi_type = self._determine_sandhi_type(word1, word2, original_sentence)

            # Calculate overall confidence
            confidence = self._calculate_confidence(
                dict_confidence,
                len(words),
                split_graph
            )

            # Create annotation
            annotation = SandhiAnnotation(
                surface_form=original_sentence,
                word1=word1,
                word2=word2,
                sandhi_type=sandhi_type,
                sandhi_rule=self._infer_sandhi_rule(word1, word2, original_sentence),
                position_start=0,
                position_end=len(original_sentence),
                confidence=confidence,
                annotation_method='sanskrit_parser',
                metadata={
                    'num_words': len(words),
                    'parser_confidence': 0.8,  # Placeholder
                    'dictionary_validated': self.use_dictionary_validation
                }
            )

            return annotation

        except Exception as e:
            logger.debug(f"Error extracting split details: {e}")
            return None

    def _extract_words_from_split(self, split_graph) -> List[str]:
        """
        Extract word forms from sanskrit_parser split graph

        Args:
            split_graph: Split graph object (sanskrit_parser.api.Split)

        Returns:
            List of word strings in SLP1
        """
        try:
            # The Split object's string representation is a list of words
            # e.g., "['rAmaH', 'asti']"

            split_str = str(split_graph)

            # Parse the list representation
            # Remove brackets and quotes, split by comma
            if split_str.startswith('[') and split_str.endswith(']'):
                # Remove outer brackets
                content = split_str[1:-1].strip()

                # Split by comma and clean quotes
                words = []
                for word in content.split(','):
                    word = word.strip().strip("'\"")
                    if word:
                        words.append(word)

                return words

            # Fallback: return as single word
            return [split_str]

        except Exception as e:
            logger.debug(f"Error extracting words: {e}")
            return []

    def _validate_with_dictionary(self, word1: str, word2: str) -> float:
        """
        Validate split words against dictionary

        Args:
            word1: First word
            word2: Second word

        Returns:
            Confidence score (0.0 to 1.0)
        """
        if not self.dictionary:
            return 0.5

        word1_exists = self.dictionary.exists(word1)
        word2_exists = self.dictionary.exists(word2)

        if word1_exists and word2_exists:
            return 1.0
        elif word1_exists or word2_exists:
            return 0.6
        else:
            return 0.3

    def _determine_sandhi_type(
        self,
        word1: str,
        word2: str,
        surface: str
    ) -> SandhiType:
        """
        Determine the type of sandhi based on word endings/beginnings

        Args:
            word1: First word
            word2: Second word
            surface: Surface form

        Returns:
            SandhiType enum
        """
        # Simplified heuristics
        # In production, would use more sophisticated analysis

        if not word1 or not word2:
            return SandhiType.UNKNOWN

        # Check last character of word1
        last_char = word1[-1].lower()
        first_char = word2[0].lower() if word2 else ''

        # Visarga sandhi: ends with ḥ or ः
        if last_char in ['ḥ', 'ः', 'H']:
            return SandhiType.VISARGA

        # Vowel sandhi: ends with vowel
        vowels = set('aāiīuūṛṝḷḹeaioau')
        if last_char in vowels:
            return SandhiType.VOWEL

        # Consonant sandhi
        return SandhiType.CONSONANT

    def _infer_sandhi_rule(
        self,
        word1: str,
        word2: str,
        surface: str
    ) -> Optional[str]:
        """
        Infer the sandhi rule that was applied

        Args:
            word1: First word
            word2: Second word
            surface: Surface form

        Returns:
            Description of sandhi rule or None
        """
        # Simplified rule inference
        # Would be expanded with actual sandhi rules

        if not word1 or not word2:
            return None

        last_char = word1[-1].lower()

        if last_char in ['ḥ', 'ः']:
            return "Visarga sandhi"
        elif last_char in 'aāiīuūṛṝḷḹeaioau':
            return "Vowel sandhi"
        else:
            return "Consonant sandhi"

    def _calculate_confidence(
        self,
        dict_confidence: float,
        num_words: int,
        split_graph
    ) -> float:
        """
        Calculate overall confidence score

        Args:
            dict_confidence: Confidence from dictionary validation
            num_words: Number of words in split
            split_graph: Split graph from parser

        Returns:
            Overall confidence (0.0 to 1.0)
        """
        # Combine multiple confidence signals

        # Base confidence from dictionary
        confidence = dict_confidence * 0.6

        # Bonus for 2-word splits (most common)
        if num_words == 2:
            confidence += 0.2
        elif num_words > 2:
            confidence += 0.1

        # Parser confidence (if available)
        # sanskrit_parser doesn't directly provide scores
        # but we can infer from graph structure
        parser_confidence = 0.2  # Placeholder
        confidence += parser_confidence

        # Normalize to 0-1 range
        return min(1.0, max(0.0, confidence))

    def _transliterate(self, text: str, from_scheme: str, to_scheme: str) -> str:
        """
        Transliterate between encoding schemes

        Args:
            text: Text to transliterate
            from_scheme: Source scheme
            to_scheme: Target scheme

        Returns:
            Transliterated text
        """
        if from_scheme == to_scheme:
            return text

        # Map scheme names to sanscript constants
        scheme_map = {
            'IAST': sanscript.IAST,
            'Devanagari': sanscript.DEVANAGARI,
            'SLP1': sanscript.SLP1,
            'HK': sanscript.HK,
            'ITRANS': sanscript.ITRANS
        }

        from_sc = scheme_map.get(from_scheme, sanscript.IAST)
        to_sc = scheme_map.get(to_scheme, sanscript.SLP1)

        return sanscript.transliterate(text, from_sc, to_sc)

    def detect_batch(
        self,
        sentences: List[str],
        encoding: str = 'IAST'
    ) -> Dict[str, List[SandhiAnnotation]]:
        """
        Detect sandhi in multiple sentences

        Args:
            sentences: List of Sanskrit sentences
            encoding: Input encoding

        Returns:
            Dictionary mapping sentences to their annotations
        """
        results = {}

        for sentence in sentences:
            try:
                annotations = self.detect_sandhi(sentence, encoding=encoding)
                results[sentence] = annotations
            except Exception as e:
                logger.error(f"Error processing sentence '{sentence}': {e}")
                results[sentence] = []

        return results


# Convenience function
def detect_sandhi(sentence: str, **kwargs) -> List[SandhiAnnotation]:
    """
    Convenience function to detect sandhi in a sentence

    Args:
        sentence: Sanskrit text
        **kwargs: Additional arguments for SanskritSandhiDetector

    Returns:
        List of SandhiAnnotation objects
    """
    detector = SanskritSandhiDetector(**kwargs)
    return detector.detect_sandhi(sentence)
