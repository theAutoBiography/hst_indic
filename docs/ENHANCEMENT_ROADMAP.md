# Sandhi Annotation Enhancement Roadmap

## Current Performance Baseline

**Detection Accuracy (SandhiKosh Benchmark):**
- Overall: 60.0%
- Visarga sandhi: 71.4%
- Consonant sandhi: 53.8%
- Vowel sandhi: Not separately tracked

**Target for Next Week: 75%+ overall accuracy**

---

## Week 1 Goals: Core Algorithm Enhancement

### Priority 1: Improve Detection Accuracy (Target: 75%+)

#### 1.1 Enhanced Dictionary Validation
**Current Issue:** Limited dictionary coverage leads to low confidence scores

**Solutions:**
- [ ] **Expand dictionary sources**
  - Download full Monier-Williams dictionary (~200k entries)
  - Add Apte Sanskrit dictionary
  - Include Shabda-Sagara dictionary
  - Add verb conjugation tables

- [ ] **Morphological analysis**
  - Recognize declined forms (prathamā, dvitīyā, etc.)
  - Support verb conjugations (laṭ, loṭ, vidhiliṅ, etc.)
  - Handle sandhi-applied forms in dictionary lookup

- [ ] **Compound word handling**
  - Implement recursive compound splitting
  - Use samāsa patterns (tatpuruṣa, bahuvrīhi, etc.)
  - Validate compound components independently

**Implementation Plan:**
```python
# Week 1, Day 1-2
class EnhancedDictionary:
    def __init__(self):
        self.root_dictionary = load_monier_williams()
        self.morphology_analyzer = MorphologyAnalyzer()
        self.compound_splitter = CompoundSplitter()

    def validate_word(self, word):
        # Check root form
        if self.root_dictionary.exists(word):
            return 1.0

        # Try morphological analysis
        root = self.morphology_analyzer.get_root(word)
        if root and self.root_dictionary.exists(root):
            return 0.9

        # Try compound splitting
        components = self.compound_splitter.split(word)
        if all(self.validate_word(c) >= 0.7 for c in components):
            return 0.8

        return 0.0
```

**Expected Impact:** +10-15% accuracy

---

#### 1.2 Context-Aware Sandhi Detection
**Current Issue:** Single-sentence analysis misses contextual clues

**Solutions:**
- [ ] **Sentence-level context**
  - Analyze surrounding words
  - Use grammatical agreement (liṅga, vibhakti, vacana)
  - Validate semantic coherence

- [ ] **Document-level patterns**
  - Track recurring word pairs
  - Build corpus-specific vocabulary
  - Learn author's style patterns

**Implementation Plan:**
```python
# Week 1, Day 3
class ContextAwareSandhiDetector:
    def detect_with_context(self, sentence, prev_sentence=None, next_sentence=None):
        # Basic sandhi detection
        candidates = self.basic_detect(sentence)

        # Rerank using context
        scored_candidates = []
        for candidate in candidates:
            score = candidate.confidence

            # Grammatical agreement check
            if self.check_agreement(candidate, sentence):
                score += 0.1

            # Semantic coherence
            if self.check_semantic_fit(candidate, prev_sentence, next_sentence):
                score += 0.1

            scored_candidates.append((candidate, score))

        return sorted(scored_candidates, key=lambda x: x[1], reverse=True)
```

**Expected Impact:** +5-10% accuracy

---

#### 1.3 Rule-Based Sandhi Validation
**Current Issue:** sanskrit_parser doesn't validate sandhi rules explicitly

**Solutions:**
- [ ] **Implement Pāṇini sandhi rules**
  - Vowel sandhi (ac-sandhi): 6.1.77-6.1.127
  - Visarga sandhi: 8.2.66, 8.3.15-8.3.37
  - Consonant sandhi (hal-sandhi): 8.4.40-8.4.68

- [ ] **Rule verification layer**
  - Check if detected split follows valid sandhi rule
  - Prefer rule-compliant splits
  - Flag rule violations for manual review

**Implementation Plan:**
```python
# Week 1, Day 4
class SandhiRuleValidator:
    def __init__(self):
        self.vowel_rules = load_vowel_sandhi_rules()
        self.visarga_rules = load_visarga_sandhi_rules()
        self.consonant_rules = load_consonant_sandhi_rules()

    def validate_split(self, surface, word1, word2):
        """
        Validate if word1 + word2 -> surface follows valid sandhi rule
        """
        # Identify sandhi type
        last_char = word1[-1]
        first_char = word2[0]

        # Check applicable rules
        if self.is_vowel(last_char):
            return self.check_vowel_sandhi(word1, word2, surface)
        elif last_char in ['ḥ', 'ः']:
            return self.check_visarga_sandhi(word1, word2, surface)
        else:
            return self.check_consonant_sandhi(word1, word2, surface)

    def check_vowel_sandhi(self, word1, word2, surface):
        # Example: a + a -> ā
        if word1[-1] == 'a' and word2[0] == 'a':
            expected = word1[:-1] + 'ā' + word2[1:]
            return expected == surface, "6.1.101 (a + a = ā)"
        # ... more rules
```

**Expected Impact:** +5-8% accuracy

---

### Priority 2: Multi-Word Sandhi Detection

**Current Issue:** Fails on complex 3+ word splits

**Test Cases:**
```
पाण्डवाः + च + एव → पाण्डवाश्चैव (currently fails)
पाण्डव + आनीकम् + व्यूढम् + दुर्योधनः + तदा → पाण्डवानीकं व्यूढं दुर्योधनस्तदा
```

**Solutions:**
- [ ] **Recursive splitting algorithm**
  - Split into 2 parts first
  - Recursively split each part
  - Score based on total confidence

- [ ] **Beam search approach**
  - Keep top K split hypotheses
  - Expand each hypothesis
  - Select best overall split

**Implementation Plan:**
```python
# Week 1, Day 5
class MultiWordSandhiDetector:
    def detect_multi_word(self, text, max_words=5, beam_width=10):
        """
        Detect sandhi with up to max_words components
        """
        # Initialize with single word (no split)
        beam = [(text, [], 1.0)]  # (remaining_text, split_words, score)

        for _ in range(max_words - 1):
            new_beam = []

            for remaining, words, score in beam:
                # Try splitting remaining text
                splits = self.basic_detect(remaining, limit=5)

                for split in splits:
                    new_words = words + [split.word1]
                    new_remaining = split.word2
                    new_score = score * split.confidence

                    new_beam.append((new_remaining, new_words, new_score))

                # Also consider no more splits
                if remaining:
                    new_beam.append((None, words + [remaining], score * 0.5))

            # Keep top beam_width hypotheses
            beam = sorted(new_beam, key=lambda x: x[2], reverse=True)[:beam_width]

        return beam[0]  # Best split
```

**Expected Impact:** +10-15% accuracy on complex cases

---

### Priority 3: Confidence Score Calibration

**Current Issue:** All scores around 0.58, not discriminative

**Solutions:**
- [ ] **Feature-based scoring**
  - Dictionary match: 0-1
  - Rule compliance: 0-1
  - Context fit: 0-1
  - Frequency in corpus: 0-1
  - Weighted combination

- [ ] **Calibrate thresholds**
  - Analyze error patterns on SandhiKosh
  - Set type-specific thresholds
  - Optimize for precision-recall tradeoff

**Implementation Plan:**
```python
# Week 1, Day 6-7
class CalibratedConfidenceScorer:
    def score_annotation(self, annotation, context):
        features = {
            'dict_word1': self.dictionary.get_confidence(annotation.word1),
            'dict_word2': self.dictionary.get_confidence(annotation.word2),
            'rule_compliance': self.rule_validator.validate(annotation),
            'context_fit': self.context_scorer.score(annotation, context),
            'corpus_frequency': self.corpus_stats.get_frequency(annotation),
        }

        # Learned weights from training data
        weights = {
            'dict_word1': 0.30,
            'dict_word2': 0.30,
            'rule_compliance': 0.25,
            'context_fit': 0.10,
            'corpus_frequency': 0.05,
        }

        score = sum(features[k] * weights[k] for k in features)
        return score
```

**Expected Impact:** Better separation of correct vs incorrect splits

---

## Week 1 Implementation Schedule

### Day 1-2: Dictionary Enhancement
- Download and integrate Monier-Williams full dictionary
- Add Apte dictionary
- Implement morphological analyzer stub
- Test on SandhiKosh subset

### Day 3: Context-Aware Detection
- Implement grammatical agreement checker
- Add sentence-level context scoring
- Test with multi-sentence documents

### Day 4: Rule Validation
- Implement core Pāṇini sandhi rules
- Add rule validator to pipeline
- Measure rule compliance rate

### Day 5: Multi-Word Detection
- Implement beam search splitter
- Test on complex SandhiKosh examples
- Tune beam width parameter

### Day 6-7: Confidence Calibration & Testing
- Implement feature-based scoring
- Run full SandhiKosh benchmark
- Analyze error patterns
- Optimize thresholds

---

## Success Metrics

### Accuracy Targets (End of Week 1)
- [ ] Overall accuracy: 75%+ (from 60%)
- [ ] Visarga sandhi: 85%+ (from 71%)
- [ ] Consonant sandhi: 70%+ (from 54%)
- [ ] Multi-word (3+): 40%+ (from ~0%)

### Performance Targets
- [ ] Processing time: < 1s per sentence
- [ ] Confidence scores: Well-calibrated (0.3-0.95 range)
- [ ] False positive rate: < 20%

### Code Quality
- [ ] Unit tests for all new components
- [ ] Integration tests pass
- [ ] Documentation updated
- [ ] No performance regression

---

## Technical Debt to Address

### Priority Issues
1. **Encoding normalization**: Ensure consistent IAST/Devanagari handling
2. **Error handling**: Better error messages for invalid input
3. **Logging**: Add debug mode with detailed split reasoning
4. **Caching**: Cache dictionary lookups and rule validations

### Nice-to-Have
1. **Visualization**: Show split alternatives in web UI
2. **Batch processing**: Parallel processing for multiple documents
3. **API**: REST API for external integration

---

## Research Questions to Explore

1. **Can we use transformer models?**
   - Fine-tune IndicBERT or mBERT on sandhi data
   - Compare with rule-based approach
   - Hybrid: rules + ML

2. **Active learning strategy?**
   - Which examples should we annotate manually?
   - Prioritize high-impact, low-confidence cases

3. **Transfer learning from other languages?**
   - Tamil has similar sandhi patterns
   - Can we leverage Dravidian NLP tools?

---

## Resources Needed

### Data
- [ ] Monier-Williams full dictionary (~200MB)
- [ ] Apte Sanskrit dictionary
- [ ] DCS (Digital Corpus of Sanskrit) for statistics
- [ ] Manually annotated gold standard (100-200 examples)

### Tools
- [ ] Morphological analyzer (consider IndoWordNet)
- [ ] POS tagger for Sanskrit
- [ ] Dependency parser (if available)

### Compute
- Current AWS RDS sufficient
- Consider GPU for ML experiments (optional)

---

## Next Steps After Week 1

### Week 2+: ML Integration (If time permits)
1. Collect training data from corrected annotations
2. Train sequence-to-sequence model
3. Hybrid approach: rules + ML ensemble

### Week 3+: Samasa & Taddhita
1. Apply same enhancement strategy to compound detection
2. Integrate with sandhi for full morphological analysis

### Week 4+: Production Readiness
1. Web interface for verification
2. API for external tools
3. Performance optimization
4. Documentation and tutorials

---

## Open Questions for Discussion

1. **Priority**: Should we focus on accuracy or coverage?
   - High accuracy on common patterns vs broad coverage with lower accuracy?

2. **Dictionary**: Which dictionary sources are most reliable?
   - Trade-off between size and quality

3. **Validation**: How to get gold standard annotations?
   - Manual annotation? Expert review? Crowd-sourcing?

4. **ML vs Rules**: When to switch to ML approach?
   - At what accuracy threshold do rules plateau?

---

## Daily Check-ins (Week 1)

**Questions to answer each day:**
1. What accuracy improvement did we achieve?
2. What was the biggest challenge?
3. What should we prioritize next?
4. Any blockers or risks?

**End of week review:**
- Comprehensive accuracy report
- Error analysis
- Recommendations for Week 2
