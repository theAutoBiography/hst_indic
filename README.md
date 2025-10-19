# Indic Annotation Pipeline

An automated annotation pipeline for creating grammatical datasets for Sanskrit, Tamil, and Gujarati languages. This project combines rule-based auto-annotation, ML assistance, and human verification to detect sandhi, samasa, and taddhita patterns.

## Features

- **Multi-language Support**: Sanskrit, Tamil, and Gujarati
- **Grammatical Analysis**:
  - Sandhi (euphonic combination) detection
  - Samasa (compound) identification
  - Taddhita (derived word) analysis
- **Rule-based Annotation**: Fast, accurate detection using linguistic rules
- **Database Storage**: PostgreSQL backend for scalable data management
- **Verification Workflow**: Human-in-the-loop validation

## Project Structure

```
indic-annotation-pipeline/
├── src/
│   ├── data_acquisition/    # Scripts to fetch texts from online sources
│   ├── preprocessing/        # Text normalization and dictionary integration
│   ├── annotators/           # Core annotation logic
│   │   ├── sandhi/          # Sandhi detection rules
│   │   ├── samasa/          # Compound identification
│   │   └── taddhita/        # Derivation analysis
│   ├── models/              # ML models (future)
│   ├── verification/        # Human verification interface
│   └── database/            # Database models and operations
├── data/
│   ├── raw/                 # Raw text files
│   ├── processed/           # Cleaned and normalized texts
│   ├── annotations/         # Generated annotations
│   └── dictionaries/        # Language dictionaries
├── config/                  # Configuration files
├── tests/                   # Unit and integration tests
└── docs/                    # Documentation
```

## Setup

### Prerequisites

- Python 3.9 or higher
- PostgreSQL 12 or higher (AWS RDS or local)
- pip or poetry for package management
- AWS account (for RDS database hosting)

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd indic-annotation-pipeline
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -e .
   ```

   For development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

4. **Configure environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your database credentials
   ```

5. **Set up PostgreSQL database**:

   **Option A - AWS RDS (Recommended for production)**:
   - See [AWS Setup Guide](docs/AWS_SETUP_GUIDE.md) for detailed instructions
   - Update `.env` with your RDS endpoint

   **Option B - Local PostgreSQL**:
   ```bash
   createdb indic_annotation
   psql -d indic_annotation -f src/database/schema.sql
   ```

6. **Initialize database schema**:
   ```bash
   python -c "from src.database.connection import init_db; init_db()"
   ```

7. **Test database connection**:
   ```bash
   python -c "from src.database.connection import test_db_connection; test_db_connection()"
   ```

8. **Create logs directory**:
   ```bash
   mkdir -p logs
   ```

## Configuration

The pipeline uses two configuration files:

- **`.env`**: Environment-specific settings (database credentials, API keys)
- **`config/config.yaml`**: Application configuration (paths, thresholds, processing settings)

Edit these files according to your setup before running the pipeline.

## Usage

### Basic Usage (Coming Soon)

```python
from src.annotators.sandhi import SanskritSandhiDetector

detector = SanskritSandhiDetector()
result = detector.detect_sandhi("rāmo 'sti")
print(result)
# Output: {'surface_form': "rāmo 'sti", 'word1': 'rāmaḥ', 'word2': 'asti', ...}
```

### Running Tests

```bash
pytest tests/
```

### Running with Coverage

```bash
pytest --cov=src tests/
```

## Development Roadmap

### Phase 1: Foundation ✅
- [x] Project setup
- [ ] Database schema design
- [ ] Dictionary integration

### Phase 2: Core Annotation
- [ ] Sandhi detection (Sanskrit)
- [ ] Samasa detection
- [ ] Taddhita detection

### Phase 3: Multi-language Support
- [ ] Tamil support
- [ ] Gujarati support

### Phase 4: ML Integration
- [ ] Training data preparation
- [ ] Model training
- [ ] Hybrid rule-ML pipeline

### Phase 5: Verification Interface
- [ ] Web-based annotation tool
- [ ] Quality metrics dashboard

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Create a new branch for your feature
2. Write tests for new functionality
3. Ensure all tests pass
4. Submit a pull request with a clear description

## License

MIT License - see LICENSE file for details

## Acknowledgments

- Indic NLP Library
- Sanskrit dictionaries from Cologne Digital Sanskrit Dictionaries
- Tamil and Gujarati linguistic resources

## Contact

For questions or feedback, please open an issue on GitHub.
