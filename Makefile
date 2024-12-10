.PHONY: install test lint format clean docker-build docker-run dashboard-dash dashboard-streamlit

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

dashboard-dash:
	python -m src.dashboard.app --framework dash

dashboard-streamlit:
	streamlit run src/dashboard/app.py