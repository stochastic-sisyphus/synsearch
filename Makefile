.PHONY: setup download-data install test clean venv

VENV = .venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip

venv:
	python3 -m venv $(VENV)

setup: venv download-data install

download-data:
	$(PYTHON) scripts/download_datasets.py

install: venv
	$(PIP) install -e .
	$(PYTHON) -m spacy download en_core_web_sm

test: venv
	PYTHONPATH=. $(PYTHON) -m pytest tests/ -v --cov=src

format: venv
	$(PYTHON) -m black src/ tests/
	$(PYTHON) -m isort src/ tests/

lint: venv
	$(PYTHON) -m flake8 src/ tests/
	$(PYTHON) -m mypy src/ tests/

clean:
	rm -rf data/scisummnet.zip $(VENV)
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +