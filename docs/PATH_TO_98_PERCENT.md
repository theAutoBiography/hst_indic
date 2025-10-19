# Path to 98%+ Sandhi Detection Accuracy

## Reality Check: What 98% Requires

Achieving 98%+ accuracy on Sanskrit sandhi detection is **extremely challenging** but possible. Here's why and how:

### Current State: 60% → Target: 98%

This is not a linear improvement. Going from 60% to 75% is relatively straightforward (better dictionaries, rules). Going from 90% to 98% requires fundamentally different approaches.

---

## The 98% Strategy: Multi-Layered Approach

### Layer 1: Perfect Rule-Based Foundation (Target: 85%)

**What we need:**
1. **Complete Pāṇini sandhi rule implementation**
   - All 47 vowel sandhi rules (6.1.77-6.1.127)
   - All visarga rules (8.2.66, 8.3.15-8.3.37)
   - All consonant rules (8.4.40-8.4.68)
   - Exception handling for irregular forms

2. **Comprehensive dictionary coverage**
   - Monier-Williams: ~200k entries
   - Apte: ~150k entries
   - Shabda-Sagara: ~100k entries
   - DCS (Digital Corpus of Sanskrit): actual usage data
   - **Total**: 400k+ unique word forms

3. **Full morphological analysis**
   - Declension tables for all nouns/pronouns (21 forms each)
   - Conjugation tables for all verbs (~90 forms per root)
   - **Estimated**: 2-3 million valid Sanskrit forms

**Implementation:**
```python
class CompleteSandhiRuleEngine:
    def __init__(self):
        # Load all Panini sutras
        self.vowel_rules = self._load_all_vowel_sandhi_rules()  # 47 rules
        self.visarga_rules = self._load_all_visarga_rules()     # ~30 rules
        self.consonant_rules = self._load_all_consonant_rules() # ~35 rules

    def _load_all_vowel_sandhi_rules(self):
        """
        Implement all Panini sutras:
        - 6.1.77: इको यणचि (iko yaṇaci)
        - 6.1.87: आद्गुणः (ād guṇaḥ)
        - 6.1.101: अकः सवर्णे दीर्घः (akaḥ savarṇe dīrghaḥ)
        - ... 44 more rules
        """
        return {
            '6.1.77': IkoYanaci(),
            '6.1.87': AdGunah(),
            '6.1.101': AkahSavarneDirghah(),
            # ... all others
        }
```

**Estimated accuracy:** 85% (with perfect implementation)

**Why not higher?**
- Ambiguity: Some splits are genuinely ambiguous
- Rare/archaic words not in dictionaries
- Errors in source texts
- Poetic license (छन्द-विच्छेद)

---

### Layer 2: Statistical Models (Target: 92%)

**The gap from 85% → 92%: Learn from data**

**Approach: Sequence-to-Sequence Learning**

1. **Prepare training data**
   - Use SandhiKosh (1,430 examples) ✓
   - DCS Sanskrit corpus (~7 million words)
   - Manually annotate 5,000-10,000 challenging examples
   - **Total training set: ~50k annotated sandhi instances**

2. **Model architecture options:**

   **Option A: Character-level Transformer**
   ```python
   # Input:  r ā m o ' s t i
   # Output: r ā m a ḥ | a s t i

   model = TransformerSeq2Seq(
       vocab_size=len(sanskrit_chars),  # ~100 characters
       d_model=512,
       num_heads=8,
       num_layers=6,
       max_length=100
   )
   ```

   **Option B: Fine-tuned IndicBERT**
   ```python
   from transformers import AutoModelForTokenClassification

   # Tag each character with sandhi boundary information
   model = AutoModelForTokenClassification.from_pretrained(
       "ai4bharat/indic-bert",
       num_labels=4  # [NO_SPLIT, SPLIT_HERE, WORD1_END, WORD2_START]
   )
   ```

   **Option C: Hybrid CNN-LSTM**
   ```python
   # Proven architecture for Sanskrit (research papers)
   model = HybridSandhiSplitter(
       char_embedding_dim=256,
       cnn_filters=256,
       lstm_units=512,
       num_layers=3
   )
   ```

3. **Training strategy**
   - **Step 1**: Train on SandhiKosh (perfect labels)
   - **Step 2**: Augment with DCS data (noisy labels from rule-based)
   - **Step 3**: Fine-tune on manually verified hard cases
   - **Step 4**: Active learning loop (label most uncertain examples)

**Expected accuracy:** 92% on SandhiKosh benchmark

**Why this works:**
- Models learn statistical patterns rules miss
- Captures corpus-specific vocabulary
- Handles ambiguity through context

---

### Layer 3: Ensemble + Expert System (Target: 95%)

**Combine multiple approaches:**

```python
class EnsembleSandhiDetector:
    def __init__(self):
        # Three independent systems
        self.rule_based = CompleteSandhiRuleEngine()
        self.transformer = TransformerSandhiModel()
        self.lstm = LSTMSandhiModel()

        # Meta-learner
        self.meta_classifier = XGBoostMetaClassifier()

    def detect(self, text):
        # Get predictions from all systems
        rule_preds = self.rule_based.detect(text)
        trans_preds = self.transformer.detect(text)
        lstm_preds = self.lstm.detect(text)

        # Extract features from each prediction
        features = self._extract_features(rule_preds, trans_preds, lstm_preds)

        # Meta-classifier decides final answer
        final_pred = self.meta_classifier.predict(features)

        return final_pred

    def _extract_features(self, rule_preds, trans_preds, lstm_preds):
        """
        Features for meta-classifier:
        - Agreement between systems
        - Individual confidence scores
        - Rule compliance flags
        - Dictionary match scores
        - Context features
        """
        return {
            'all_agree': rule_preds == trans_preds == lstm_preds,
            'rule_confidence': rule_preds.confidence,
            'transformer_confidence': trans_preds.confidence,
            'lstm_confidence': lstm_preds.confidence,
            'num_agreeing': sum([rule_preds == trans_preds,
                                 trans_preds == lstm_preds,
                                 lstm_preds == rule_preds]),
            # ... 50+ more features
        }
```

**Why ensembles work:**
- Different systems make different errors
- Combining reduces variance
- Meta-learner learns when to trust which system

**Expected accuracy:** 95%

---

### Layer 4: Human-in-the-Loop (Target: 98%+)

**The final 3%: Human verification on edge cases**

**Strategy:**

1. **Confidence-based routing**
   ```python
   class HumanInLoopDetector:
       def detect_with_verification(self, text):
           prediction = self.ensemble.detect(text)

           if prediction.confidence < 0.95:
               # Route to human verification
               return self.request_human_verification(text, prediction)
           else:
               # High confidence - accept automatically
               return prediction
   ```

2. **Active learning**
   - System identifies examples it's uncertain about
   - Human expert annotates these cases
   - Model is retrained with new data
   - Iteratively improves on hard cases

3. **Error pattern analysis**
   ```python
   # Analyze systematic errors
   errors = analyze_errors(predictions, ground_truth)

   # Common error patterns:
   # - Rare words not in dictionary
   # - Archaic sandhi forms
   # - Poetic variations
   # - Scribal errors in source

   # Create specialized rules/models for each pattern
   ```

4. **Expert verification UI**
   - Show top 3 predictions
   - Highlight rule violations
   - Display similar examples from corpus
   - Allow expert to add new rules on-the-fly

**With human verification:** 98%+

**Why this is necessary:**
- Some cases are genuinely ambiguous (even experts disagree)
- OCR/transcription errors in source texts
- Regional variations in sandhi application
- Poetic license overrides rules

---

## Detailed Implementation Roadmap

### Phase 1: Foundation (Weeks 1-4)
**Target: 85% accuracy**

**Week 1-2: Complete Rule Implementation**
- [ ] Day 1-3: Implement all 47 vowel sandhi rules
- [ ] Day 4-6: Implement all visarga rules
- [ ] Day 7-10: Implement all consonant rules
- [ ] Day 11-14: Exception handling, edge cases, testing

**Week 3-4: Dictionary & Morphology**
- [ ] Download and integrate 3-4 major dictionaries
- [ ] Build morphological analyzer
  - Noun declension generator (21 forms × 100k nouns = 2.1M forms)
  - Verb conjugation generator (90 forms × 10k roots = 900k forms)
- [ ] Test coverage on DCS corpus

**Deliverable:** Rule-based system with 85% accuracy

---

### Phase 2: Machine Learning (Weeks 5-8)
**Target: 92% accuracy**

**Week 5: Data Preparation**
- [ ] Extract sandhi examples from DCS corpus (target: 50k examples)
- [ ] Clean and validate SandhiKosh data
- [ ] Create train/dev/test splits (80/10/10)
- [ ] Manual annotation of 1,000 hard cases

**Week 6-7: Model Training**
- [ ] Train baseline transformer model
- [ ] Fine-tune IndicBERT
- [ ] Train hybrid CNN-LSTM
- [ ] Hyperparameter tuning
- [ ] Cross-validation

**Week 8: Model Optimization**
- [ ] Error analysis
- [ ] Data augmentation for rare cases
- [ ] Model ensemble experiments
- [ ] Benchmark on SandhiKosh

**Deliverable:** ML models achieving 90-92% accuracy

---

### Phase 3: Ensemble System (Weeks 9-10)
**Target: 95% accuracy**

**Week 9: Ensemble Architecture**
- [ ] Feature engineering for meta-classifier
- [ ] Train meta-learner (XGBoost, LightGBM)
- [ ] Stacking experiments
- [ ] Weighted voting schemes

**Week 10: Optimization & Testing**
- [ ] Error analysis of ensemble
- [ ] Calibrate confidence scores
- [ ] Optimize for precision-recall tradeoff
- [ ] Comprehensive testing

**Deliverable:** Ensemble system with 95% accuracy

---

### Phase 4: Human-in-Loop (Weeks 11-12)
**Target: 98%+ accuracy**

**Week 11: Verification System**
- [ ] Build verification UI
- [ ] Implement confidence-based routing
- [ ] Active learning pipeline
- [ ] Expert feedback integration

**Week 12: Production Polish**
- [ ] Error pattern analysis
- [ ] Create specialized rules for common errors
- [ ] Performance optimization
- [ ] Documentation

**Deliverable:** Production system with 98%+ accuracy (with human verification)

---

## Required Resources

### Data
| Resource | Size | Source | Status |
|----------|------|--------|--------|
| SandhiKosh | 1,430 examples | ✓ Available | Downloaded |
| DCS Corpus | 7M words | digitalsanskritlibrary.org | Need to acquire |
| Monier-Williams | 200k entries | cologne.edu | Need to download |
| Apte Dictionary | 150k entries | Sanskrit resources | Need to acquire |
| Manual annotations | 5-10k examples | Expert annotators | Need to create |

**Estimated cost for manual annotations:** $5,000-$10,000 (assuming $1-2 per example)

### Compute
| Task | GPU Needed | Time | Cost |
|------|------------|------|------|
| Rule development | None | 4 weeks | $0 |
| Dictionary building | None | 2 weeks | $0 |
| ML training | NVIDIA A100 | 2 weeks | $500-1000 |
| Inference | CPU (AWS) | Ongoing | $50/month |

**Total estimated compute cost:** $1,000-1,500

### Human Resources
| Role | Time Required | Purpose |
|------|---------------|---------|
| Sanskrit Expert | 100 hours | Rule validation, manual annotation |
| ML Engineer | 200 hours | Model development and training |
| Software Engineer | 100 hours | Infrastructure, UI, integration |

---

## Critical Success Factors

### 1. Data Quality > Data Quantity
- 1,000 perfect annotations > 10,000 noisy annotations
- Expert validation is essential
- Gold standard test set (500-1000 examples) with 100% verified labels

### 2. Systematic Error Analysis
```python
# After each improvement, analyze errors:
def analyze_errors(predictions, ground_truth):
    errors = []
    for pred, truth in zip(predictions, ground_truth):
        if pred != truth:
            error = {
                'example': pred.text,
                'predicted': pred.split,
                'correct': truth.split,
                'sandhi_type': classify_sandhi_type(pred, truth),
                'error_type': classify_error_type(pred, truth),
                'dict_coverage': check_dictionary(pred, truth),
                'rule_violation': check_rules(pred, truth)
            }
            errors.append(error)

    # Group by error type
    error_patterns = group_by(errors, 'error_type')

    # Prioritize fixes
    return sorted(error_patterns, key=lambda x: len(x), reverse=True)
```

### 3. Iterative Improvement
- Don't aim for 98% in one shot
- Target: 60% → 70% → 80% → 90% → 95% → 98%
- After each increment, analyze what's blocking further progress

### 4. Domain Expert Collaboration
- Work closely with Sanskrit scholars
- Validate rules and exceptions
- Get feedback on edge cases
- Build trust in the system

---

## Realistic Timeline

### Conservative Estimate: 6 months to 98%
- Month 1-2: Rule implementation (85%)
- Month 3: Data preparation (85%)
- Month 4: ML training (90%)
- Month 5: Ensemble building (95%)
- Month 6: Human-in-loop refinement (98%)

### Aggressive Estimate: 3 months to 95%
- Skip full rule implementation
- Focus on ML from day 1
- Use transfer learning aggressively
- Accept 95% without human verification

### Risk Factors
- **High risk**: Limited training data
- **Medium risk**: Ambiguous cases (may never reach 98% without human input)
- **Low risk**: Compute resources (manageable with cloud)

---

## Alternative: Hybrid Accuracy Targets

Instead of 98% overall, consider:

### Option A: Tiered Accuracy
- **High confidence (60% of cases):** 99% accuracy (auto-accept)
- **Medium confidence (30% of cases):** 95% accuracy (quick review)
- **Low confidence (10% of cases):** 85% accuracy (detailed review)
- **Overall effective accuracy:** 97%+

### Option B: Domain-Specific
- **Bhagavad Gita:** 98% (focused training)
- **Vedic texts:** 90% (different sandhi rules)
- **Modern Sanskrit:** 95% (simpler sandhi)

### Option C: Sandhi-Type Specific
- **Visarga sandhi:** 98% (most regular)
- **Vowel sandhi:** 96% (some ambiguity)
- **Consonant sandhi:** 92% (complex rules)

---

## Benchmark: How Good is "Good Enough"?

### Human Expert Performance
- **Sanskrit scholars:** ~98% agreement on standard texts
- **Experts disagree:** ~2-5% of cases (genuinely ambiguous)
- **OCR/text errors:** ~1-3% of source texts have errors

**Implication:** 98% may be near theoretical maximum without cleaning source texts

### Comparison to Other Systems
| System | Accuracy | Notes |
|--------|----------|-------|
| Sanskrit Heritage API | ~70% | Production system |
| sanskrit_parser | ~60% | What we're using |
| DCS sandhi splitter | ~75% | Research tool |
| **Our target** | **98%** | Would be SOTA |

**98% would be state-of-the-art for Sanskrit sandhi detection.**

---

## Recommended Approach: Pragmatic Path

### Immediate (Week 1-2): Low-Hanging Fruit → 75%
- Expand dictionary coverage
- Add top 20 sandhi rules
- Context-based filtering

### Short-term (Month 1-2): Solid Foundation → 85%
- Complete rule implementation
- Full morphological analysis
- Comprehensive testing

### Medium-term (Month 3-4): ML Integration → 92%
- Train transformer model
- Active learning on hard cases
- Ensemble with rules

### Long-term (Month 5-6): Production Polish → 95-98%
- Human verification workflow
- Specialized models for text types
- Continuous improvement loop

---

## Key Insight: The Last 2% is Hardest

Going from 96% → 98% may require as much effort as going from 60% → 90%.

**Why?**
- Remaining errors are edge cases
- May require domain expertise
- Source texts may have genuine errors
- Some ambiguity is irreducible

**Recommendation:**
- Aim for 95% fully automated
- 98%+ with human verification on low-confidence cases
- This is both achievable AND practical

---

## Next Steps

1. **Validate approach** with Sanskrit expert
2. **Acquire training data** (DCS corpus license)
3. **Budget approval** for compute and annotations
4. **Hire/contract** ML engineer (if needed)
5. **Start with Phase 1** (rule implementation)

## Questions for You

1. **Timeline:** 3 months aggressive or 6 months conservative?
2. **Budget:** Can we allocate $10-15k for data + compute?
3. **Expert access:** Do you have Sanskrit scholars to consult?
4. **Acceptable accuracy:** Is 95% auto + 98% with human OK?
5. **Text focus:** Which corpus matters most (BG, Vedas, etc.)?
