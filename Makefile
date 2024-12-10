
.PHONY: install test lint format clean docker-build docker-run

install:
	pip install -r requirements.txt

test:
	python -m pytest tests/

lint:
	flake8 src/ tests/
	mypy src/
	black --check .

format:
	black .
	isort .

clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .coverage htmlcov/

docker-build:
	docker-compose build

docker-run:
	docker-compose up

.PHONY: docs
docs:
	sphinx-build -b html docs/source docs/build