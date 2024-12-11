.PHONY: setup download-data install test clean

setup: download-data install

download-data:
	python scripts/download_datasets.py

install:
	pip install -e .
	python -m spacy download en_core_web_sm

test:
	PYTHONPATH=. pytest tests/ -v --cov=src

format:
	black src/ tests/
	isort src/ tests/

lint:
	flake8 src/ tests/
	mypy src/ tests/

clean:
	rm -rf data/scisummnet.zip
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +