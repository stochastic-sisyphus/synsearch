# Copilot Instructions for SynSearch

## Project Overview
SynSearch is a sophisticated Python-based research paper analysis system that combines advanced NLP techniques, clustering algorithms, and scientific text processing. The project helps researchers effectively analyze and summarize large collections of scientific literature.

## Tech Stack
- **Python**: 3.8+
- **ML/DL Frameworks**: PyTorch (2.0+), Transformers (4.15+), Sentence-Transformers
- **NLP Libraries**: NLTK, spaCy, BERT-Score
- **Clustering**: HDBSCAN, UMAP, NetworkX, scipy
- **Visualization**: Plotly, Dash, Streamlit, Matplotlib, Seaborn
- **Data Processing**: pandas, numpy, datasets (Hugging Face)
- **Testing**: pytest, pytest-cov, pytest-asyncio
- **Code Quality**: black, isort, flake8, mypy

## Repository Structure
```
synsearch/
├── src/                      # Main source code
│   ├── api/                  # API integrations (ArXiv, etc.)
│   ├── preprocessing/        # Text preprocessing modules
│   ├── clustering/           # Clustering algorithms
│   ├── summarization/        # Summary generation
│   ├── utils/                # Utility functions
│   ├── visualization/        # Visualization tools
│   ├── dashboard/            # Dashboard components
│   ├── data_loader/          # Data loading utilities
│   ├── evaluation/           # Evaluation metrics
│   ├── training/             # Model training scripts
│   └── personalization/      # Personalization features
├── tests/                    # Test suite
├── config/                   # Configuration files
├── data/                     # Dataset storage
├── scripts/                  # Utility scripts
└── .github/                  # GitHub workflows and configs
```

## Development Guidelines

### Code Style
- **Follow PEP 8** style guidelines strictly
- **Line length**: 100 characters (configured in pyproject.toml)
- **Use type hints** for all function parameters and return values
- **Documentation**: Use Google docstring format for all modules, classes, and functions
- **Imports**: Organize with isort (profile: black)

### Code Quality Standards
1. All code must pass:
   - `black` formatting (100 char line length)
   - `isort` import sorting
   - `flake8` linting
   - `mypy` type checking (when applicable)
   - `pytest` tests with coverage

2. Write unit tests for all new functionality
3. Maintain or improve code coverage (currently using pytest-cov)
4. Document complex algorithms and business logic

### Testing
- **Framework**: pytest with coverage reporting
- **Command**: `make test` or `PYTHONPATH=. python -m pytest tests/ -v --cov=src`
- **Location**: Place tests in `tests/` directory
- **Naming**: Test files should be named `test_*.py`
- **Coverage**: Aim for high coverage on core modules

### Build & Run Commands
```bash
# Setup (first time)
make setup                    # Full setup including data download

# Installation
make install                  # Install package and dependencies
make install-deps             # Install dependencies only

# Development
make format                   # Format code with black and isort
make lint                     # Run flake8 and mypy
make test                     # Run tests with coverage

# Running the application
make run                      # Run optimized pipeline
make run-optimized            # Run with performance optimizations
make run-dashboard            # Launch dashboard

# Data operations
make download-data            # Download required datasets
make prepare-datasets         # Prepare datasets for processing

# Model operations
make train-models             # Train models
make compare-models           # Compare model performance
make test-models              # Test model training

# Cleanup
make clean                    # Clean cache and temporary files
```

### Key Modules

#### Data Processing
- `data_loader.py`: Handles dataset loading and validation (XL-Sum, ScisummNet)
- `data_preparation.py`: Prepares and preprocesses text data
- `data_validator.py`: Ensures data quality and format
- `preprocessing/`: Domain-specific text preprocessing

#### Analysis & Clustering
- `clustering/cluster_manager.py`: Manages document clustering with HDBSCAN
- `embedding_generator.py`: Generates text embeddings using transformers
- `data_exploration.py`: Exploratory data analysis tools

#### Summarization
- `summarization/`: Multi-style text summarization (technical, concise, detailed)
- Supports batch processing and GPU acceleration

#### API & Integration
- `api/arxiv_api.py`: ArXiv API integration for paper retrieval
- Support for aiohttp async operations

### Environment Variables
The project uses several performance optimization environment variables (defined in Makefile):
- `OMP_NUM_THREADS=8`: OpenMP parallelism
- `TOKENIZERS_PARALLELISM=true`: HuggingFace tokenizer parallelism
- `TORCH_NUM_THREADS=8`: PyTorch threading
- `CUDA_LAUNCH_BLOCKING=0`: Async CUDA operations

### Configuration
- Main config: `config/config.yaml`
- Project config: `pyproject.toml` (black, pytest, isort settings)
- Setup config: `setup.py` and `setup.cfg`

### Important Patterns
1. **Batch Processing**: Use batch processing for efficiency with large datasets
2. **Caching**: Utilize cachetools and joblib for performance optimization
3. **GPU Support**: Code should support both CPU and GPU execution
4. **Async Operations**: Use aiohttp for API calls when appropriate
5. **Progress Tracking**: Use tqdm for long-running operations
6. **Error Handling**: Implement robust error handling with meaningful messages
7. **Logging**: Use structured logging (logs stored in `logs/synsearch.log`)

### Dependencies
- Core dependencies in `requirements.txt` (DO NOT modify without discussion)
- Use existing libraries when possible
- Major libraries: torch, transformers, sentence-transformers, scikit-learn, hdbscan

### Git Workflow
- **Pre-commit hooks**: Configured in `.pre-commit-config.yaml`
- **Branches**: Work on feature branches
- **Commits**: Write clear, descriptive commit messages

### Common Tasks

#### Adding New Preprocessing
1. Create module in `src/preprocessing/`
2. Follow existing patterns (e.g., `domain_agnostic_preprocessor.py`)
3. Add corresponding tests in `tests/test_preprocessor.py`
4. Update configuration if needed

#### Adding New Clustering Algorithm
1. Implement in `src/clustering/`
2. Integrate with `cluster_manager.py`
3. Add tests in `tests/test_dynamic_clustering.py`
4. Document parameters and usage

#### Adding New Summarization Style
1. Extend `src/summarization/` modules
2. Update hybrid summarizer
3. Add evaluation tests
4. Document new style in README

### Performance Considerations
- Optimize for batch processing
- Use GPU when available (check CUDA availability)
- Implement caching for expensive operations
- Monitor memory usage (especially with large models)
- Use multiprocessing/threading appropriately

### Documentation
- README.md: Main project documentation
- info.md: Additional project information
- usage.md: Usage examples and guides
- Update relevant docs when adding features

### Known Patterns to Avoid
- Don't commit large model files (use model hubs)
- Don't hardcode file paths (use config)
- Don't skip type hints for public APIs
- Don't merge without passing tests

## Questions & Help
- Check existing tests for usage examples
- Review `src/main.py` for pipeline integration patterns
- See `scripts/` for utility examples
- Refer to README.md for high-level architecture
