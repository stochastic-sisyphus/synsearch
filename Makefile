.PHONY: setup download-data install test clean venv

VENV = .venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip

# Enhanced performance optimization env vars
export OMP_NUM_THREADS=8  # OpenMP threads 
export MKL_NUM_THREADS=8  # MKL threads
export NUMEXPR_NUM_THREADS=8  # NumExpr threads
export OPENBLAS_NUM_THREADS=8  # OpenBLAS threads
export TOKENIZERS_PARALLELISM=true  # Enable HuggingFace tokenizer parallelism
export TORCH_NUM_THREADS=8  # PyTorch threads
export CUDA_LAUNCH_BLOCKING=0  # Async CUDA operations
export PYTHONWARNINGS="ignore"  # Reduce overhead from warnings
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512  # Memory allocation strategy
export TRANSFORMERS_OFFLINE=1  # Avoid network checks
export HF_DATASETS_OFFLINE=1  # Avoid dataset downloads during processing

venv:
    python3 -m venv $(VENV)

setup: venv download-data install

install-deps: venv
    $(PIP) install requests tqdm datasets transformers torch numpy sentencepiece protobuf \
        nltk spacy scikit-learn pandas scipy \
        beautifulsoup4 lxml textacy

download-data: install-deps
    $(PYTHON) -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
    $(PYTHON) scripts/download_datasets.py

install: venv
    $(PIP) install -e .
    $(PIP) install spacy joblib pandas datasets transformers sentence-transformers tqdm torch numpy 
    $(PYTHON) -m spacy download en_core_web_sm

test: venv
    PYTHONPATH=. $(PYTHON) -m pytest tests/ -v --cov=src

format: venv
    $(PYTHON) -m black src/ tests/
    $(PYTHON) -m isort src/ tests/

lint: venv
    $(PYTHON) -m flake8 src/ tests/
    $(PYTHON) -m mypy src/ tests/

run-optimized: install-deps
    $(PYTHON) run_optimized.py --config config/config.yaml

run: venv
    $(PYTHON) run_optimized.py --config config/config.yaml

clean:
    rm -rf data/scisummnet.zip $(VENV)
    find . -type d -name "__pycache__" -exec rm -rf {} +
    find . -type d -name ".pytest_cache" -exec rm -rf {} +
    find . -type d -name ".mypy_cache" -exec rm -rf {} +
