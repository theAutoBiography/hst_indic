I'm building an annotation pipeline for Sanskrit/Tamil/Gujarati grammatical datasets. 

CONTEXT:
- Goal: Create annotated datasets for sandhi, samasa, taddhita detection
- Approach: Rule-based auto-annotation + ML assistance + human verification
- Tech stack: Python, PostgreSQL, Flask

TASK 1 - Project Setup:
Create a Python project with this structure:

indic-annotation-pipeline/
├── src/
│   ├── data_acquisition/
│   ├── preprocessing/
│   ├── annotators/
│   │   ├── sandhi/
│   │   ├── samasa/
│   │   └── taddhita/
│   ├── models/
│   ├── verification/
│   └── database/
├── data/
│   ├── raw/
│   ├── processed/
│   ├── annotations/
│   └── dictionaries/
├── config/
├── tests/
└── docs/

Include:
- pyproject.toml with dependencies: indic-nlp-library, pandas, sqlalchemy, psycopg2-binary, requests, beautifulsoup4, pyyaml, pytest
- .env.example for configuration
- config/config.yaml with database settings, data paths
- Basic README.md with setup instructions
- .gitignore for Python projects

Set up proper Python package structure with __init__.py files where needed.
```

---

### **Step 2: Database Schema (Second Session)**

After reviewing the project structure:
```
TASK 2 - Database Schema:

Create PostgreSQL schema for storing texts and annotations:

Tables needed:
1. documents (doc_id, title, author, language, period, genre, source_url, metadata JSONB)
2. sentences (sent_id, doc_id, position, original_text, normalized_text)
3. sandhi_annotations (annotation_id, sent_id, surface_form, word1, word2, sandhi_type, sandhi_rule, confidence, annotation_method, verified_by)
4. samasa_annotations (compound, components as array, samasa_type, confidence)
5. taddhita_annotations (derived_word, root_word, suffix, derivation_type)

Create:
- src/database/schema.sql with full table definitions
- src/database/models.py with SQLAlchemy ORM models
- src/database/connection.py for database connection management
- Basic CRUD operations in src/database/operations.py

Add appropriate indexes for performance on language, confidence, doc_id, sent_id.
```

---

### **Step 3: Dictionary Integration (Third Session)**
```
TASK 3 - Sanskrit Dictionary Integration:

Build a fast dictionary lookup system.

Create src/preprocessing/dictionary.py with:

class SanskritDictionary:
    - Load Monier-Williams dictionary (I'll provide the file/URL)
    - Fast word lookup using hash table
    - exists(word) -> bool method
    - get_definitions(word) -> str method
    
Include:
- Download script for Cologne Digital Sanskrit Dictionaries
- Preprocessing to clean dictionary data
- Unit tests with known Sanskrit words
- CLI for testing: python -m src.preprocessing.dictionary --word rāma

Optimize for lookup speed < 1ms per query.
```

---

### **Step 4: Rule-Based Sandhi Detector (Fourth Session)**
```
TASK 4 - Sanskrit Sandhi Detector:

Implement rule-based sandhi detection in src/annotators/sandhi/sanskrit_rules.py

Create SanskritSandhiDetector class with:

1. detect_sandhi(sentence) method that:
   - Identifies potential sandhi locations
   - Applies sandhi rules
   - Returns splits with confidence scores

2. Implement these top sandhi rules (cover ~80% of cases):
   Vowel sandhi:
   - a + a → ā
   - a + i → e  
   - a + u → o
   - etc.
   
   Visarga sandhi:
   - aḥ + a → o 'a
   - aḥ + voiced → o
   
   Consonant sandhi:
   - t + c → c c
   - Common assimilation rules

3. verify_with_dictionary(word1, word2) - check if both splits exist

Input: "rāmo 'sti"
Output: {
    'surface_form': "rāmo 'sti",
    'word1': 'rāmaḥ',
    'word2': 'asti',
    'sandhi_type': 'visarga',
    'rule': 'visarga before a → o',
    'confidence': 0.95
}

Include unit tests with 20+ known examples.
```

---

## **Key Tips:**

### **DO:**
✅ Work **one task at a time**  
✅ Review each output before proceeding  
✅ Test each component as you build  
✅ Keep the full instruction document handy for reference  
✅ Adjust based on what Claude Code produces  

### **DON'T:**
❌ Paste the entire 5-phase plan at once  
❌ Move to next task if current one doesn't work  
❌ Accept code without understanding it  
❌ Skip testing  

---

## **When Things Go Wrong:**

If Claude Code produces something off-track:
```
"This isn't quite right. Let me clarify:

[Explain what's wrong]

Instead, please:
1. [Specific instruction]
2. [Another specific instruction]

Here's an example of what I expect:
[Show example output/structure]
"
```

---

## **Follow-up Questions to Ask:**

After each task:
- "Can you explain how [specific part] works?"
- "What should I test to verify this works?"
- "What's the next logical step?"
- "Are there any edge cases I should worry about?"

---

## **Session Management:**

Claude Code has memory within a session but not across sessions, so:

**Start each new session with:**
```
Continuing work on the indic-annotation-pipeline project.

So far we've built:
- [List what's complete]

Current project structure:
- [Brief overview]

Next task:
- [What you want to build now]