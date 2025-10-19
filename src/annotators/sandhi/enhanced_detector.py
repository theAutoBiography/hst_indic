"""
Enhanced Sandhi Detector with immediate improvements
Target: 60% → 75-80% accuracy in next 2 hours
"""
import logging
from typing import List, Optional, Tuple
from dataclasses import dataclass
from collections import Counter

from src.annotators.sandhi.detector import (
    SanskritSandhiDetector,
    SandhiAnnotation,
    SandhiType
)

logger = logging.getLogger(__name__)


@dataclass
class BeamHypothesis:
    """Hypothesis in beam search"""
    remaining_text: str
    split_words: List[str]
    score: float
    annotations: List[SandhiAnnotation]


class EnhancedSandhiDetector(SanskritSandhiDetector):
    """
    Enhanced detector with:
    1. Multi-word beam search
    2. Better confidence scoring
    3. Context validation
    """

    def __init__(
        self,
        use_dictionary_validation: bool = True,
        min_confidence: float = 0.3,  # Lower to catch more candidates
        beam_width: int = 5,
        max_words: int = 5,
        **kwargs
    ):
        super().__init__(
            use_dictionary_validation=use_dictionary_validation,
            min_confidence=min_confidence,
            **kwargs
        )
        self.beam_width = beam_width
        self.max_words = max_words

        # Frequency statistics (can be loaded from corpus)
        self.word_frequency = self._initialize_frequencies()

    def _initialize_frequencies(self) -> Counter:
        """
        Initialize word frequency counts
        In production, load from corpus statistics
        """
        # Common Sanskrit words (to be expanded)
        common_words = {
            'asti': 1000,
            'ca': 5000,
            'api': 3000,
            'tu': 2000,
            'eva': 4000,
            'hi': 2500,
            'च': 5000,
            'अपि': 3000,
        }
        return Counter(common_words)

    def detect_sandhi_enhanced(
        self,
        sentence: str,
        encoding: str = 'IAST',
        use_beam_search: bool = True
    ) -> List[SandhiAnnotation]:
        """
        Enhanced sandhi detection with beam search for multi-word splits

        Args:
            sentence: Sanskrit text
            encoding: Input encoding
            use_beam_search: Whether to use beam search for multi-word

        Returns:
            List of SandhiAnnotation objects with improved confidence
        """
        if not use_beam_search:
            # Fallback to basic detection
            return self.detect_sandhi(sentence, encoding=encoding)

        # Use beam search for better multi-word handling
        annotations = self._beam_search_split(sentence, encoding)

        # Re-score with enhanced features
        for ann in annotations:
            ann.confidence = self._calculate_enhanced_confidence(ann, sentence)

        # Sort by enhanced confidence
        annotations.sort(key=lambda x: x.confidence, reverse=True)

        return annotations

    def _beam_search_split(
        self,
        text: str,
        encoding: str
    ) -> List[SandhiAnnotation]:
        """
        Beam search for multi-word sandhi splitting

        Example:
            Input: "pāṇḍavāścaiva"
            Beam search tries:
            - pāṇḍavāḥ + ca + eva (3 words)
            - pāṇḍavāḥ + caiva (2 words)
            - pāṇḍavāścaiva (1 word, no split)
        """
        # Convert to SLP1 for processing
        if encoding != 'SLP1':
            text_slp1 = self._transliterate(text, encoding, 'SLP1')
        else:
            text_slp1 = text

        # Initialize beam with full text (no split)
        beam = [BeamHypothesis(
            remaining_text=text_slp1,
            split_words=[],
            score=1.0,
            annotations=[]
        )]

        all_annotations = []

        # Iteratively expand beam
        for depth in range(self.max_words):
            new_beam = []

            for hypothesis in beam:
                if not hypothesis.remaining_text:
                    # Complete hypothesis
                    if hypothesis.split_words:
                        # Convert to annotations
                        ann = self._hypothesis_to_annotation(hypothesis, text, encoding)
                        if ann:
                            all_annotations.append(ann)
                    continue

                # Try splitting remaining text
                try:
                    basic_splits = self.parser.split(hypothesis.remaining_text, limit=5)

                    if basic_splits:
                        for split in basic_splits[:3]:  # Top 3 splits
                            words = self._extract_words_from_split(split)

                            if len(words) >= 2:
                                # Split into first word + remaining
                                word1 = words[0]
                                remaining = words[1] if len(words) == 2 else ' '.join(words[1:])

                                # Calculate score increment
                                word_score = self._score_word(word1)
                                new_score = hypothesis.score * word_score

                                new_hypothesis = BeamHypothesis(
                                    remaining_text=remaining,
                                    split_words=hypothesis.split_words + [word1],
                                    score=new_score,
                                    annotations=hypothesis.annotations
                                )
                                new_beam.append(new_hypothesis)

                    # Also consider no more splits (accept remaining as is)
                    if hypothesis.remaining_text:
                        final_score = hypothesis.score * self._score_word(hypothesis.remaining_text)
                        new_beam.append(BeamHypothesis(
                            remaining_text='',
                            split_words=hypothesis.split_words + [hypothesis.remaining_text],
                            score=final_score * 0.9,  # Penalty for not splitting
                            annotations=hypothesis.annotations
                        ))

                except Exception as e:
                    logger.debug(f"Error in beam search: {e}")
                    continue

            # Keep top beam_width hypotheses
            beam = sorted(new_beam, key=lambda x: x.score, reverse=True)[:self.beam_width]

            if not beam:
                break

        return all_annotations[:10]  # Return top 10 annotations

    def _hypothesis_to_annotation(
        self,
        hypothesis: BeamHypothesis,
        original_text: str,
        encoding: str
    ) -> Optional[SandhiAnnotation]:
        """Convert beam hypothesis to SandhiAnnotation"""
        if len(hypothesis.split_words) < 2:
            return None

        # For now, create annotation for first split
        word1 = self._transliterate(hypothesis.split_words[0], 'SLP1', encoding)
        word2 = self._transliterate(' '.join(hypothesis.split_words[1:]), 'SLP1', encoding)

        sandhi_type = self._determine_sandhi_type(word1, word2, original_text)

        return SandhiAnnotation(
            surface_form=original_text,
            word1=word1,
            word2=word2,
            sandhi_type=sandhi_type,
            sandhi_rule=self._infer_sandhi_rule(word1, word2, original_text),
            position_start=0,
            position_end=len(original_text),
            confidence=hypothesis.score,
            annotation_method='beam_search',
            metadata={
                'num_words': len(hypothesis.split_words),
                'beam_score': hypothesis.score
            }
        )

    def _score_word(self, word: str) -> float:
        """
        Score a word based on multiple features:
        - Dictionary presence
        - Frequency in corpus
        - Length (prefer reasonable lengths)
        """
        score = 0.5  # Base score

        # Dictionary check
        if self.dictionary and self.dictionary.exists(word):
            score += 0.3

        # Frequency bonus
        if word in self.word_frequency:
            freq = self.word_frequency[word]
            score += min(0.2, freq / 10000)  # Cap at 0.2

        # Length penalty (very short or very long words less likely)
        word_len = len(word)
        if 2 <= word_len <= 15:
            score += 0.1
        elif word_len < 2:
            score -= 0.2

        return max(0.1, min(1.0, score))

    def _calculate_enhanced_confidence(
        self,
        annotation: SandhiAnnotation,
        sentence: str
    ) -> float:
        """
        Enhanced confidence calculation with multiple features

        Features:
        1. Dictionary validation (0-0.3)
        2. Word frequency (0-0.2)
        3. Rule compliance (0-0.2)
        4. Split likelihood (0-0.2)
        5. Context fit (0-0.1)
        """
        features = {}

        # Feature 1: Dictionary validation
        dict_conf = 0.0
        if self.dictionary:
            word1_exists = self.dictionary.exists(annotation.word1)
            word2_exists = self.dictionary.exists(annotation.word2)

            if word1_exists and word2_exists:
                dict_conf = 0.3
            elif word1_exists or word2_exists:
                dict_conf = 0.15

        features['dictionary'] = dict_conf

        # Feature 2: Word frequency
        freq_conf = 0.0
        for word in [annotation.word1, annotation.word2]:
            if word in self.word_frequency:
                freq_conf += min(0.1, self.word_frequency[word] / 10000)
        features['frequency'] = min(0.2, freq_conf)

        # Feature 3: Rule compliance
        rule_conf = 0.15  # Base assumption: rules are mostly followed
        # TODO: Implement actual rule checking
        features['rule'] = rule_conf

        # Feature 4: Split likelihood (based on number of words)
        num_words = annotation.metadata.get('num_words', 2)
        if num_words == 2:
            split_conf = 0.2  # Most common
        elif num_words == 1:
            split_conf = 0.05  # Rare (no split)
        else:
            split_conf = 0.1  # Multi-word
        features['split_likelihood'] = split_conf

        # Feature 5: Context fit (placeholder)
        features['context'] = 0.05

        # Total confidence
        total_confidence = sum(features.values())

        # Log feature breakdown for debugging
        logger.debug(f"Confidence for {annotation.word1} + {annotation.word2}: {total_confidence:.2f}")
        logger.debug(f"Features: {features}")

        return min(1.0, total_confidence)


# Convenience function
def detect_sandhi_enhanced(sentence: str, **kwargs) -> List[SandhiAnnotation]:
    """
    Quick access to enhanced detection

    Args:
        sentence: Sanskrit text
        **kwargs: Additional arguments for EnhancedSandhiDetector

    Returns:
        List of SandhiAnnotation objects
    """
    detector = EnhancedSandhiDetector(**kwargs)
    return detector.detect_sandhi_enhanced(sentence)
