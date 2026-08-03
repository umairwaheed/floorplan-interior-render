PYTHON ?= python3.12
PY := .venv/bin/python
PIP := .venv/bin/pip

.PHONY: help venv install run test lint fmt clean catalog docker check

help:
	@echo "make install   Install dependencies into .venv"
	@echo "make run       Start the API + web UI on :8000"
	@echo "make catalog   Build/rebuild the product index"
	@echo "make test      Run tests"
	@echo "make lint      Ruff check"
	@echo "make fmt       Ruff format + autofix"
	@echo "make docker    Build the container image"
	@echo "make check     lint + tests, as CI would"

venv:
	$(PYTHON) -m venv .venv

install: venv
	$(PIP) install -q --upgrade pip
	$(PIP) install -q -r requirements.txt

run:
	.venv/bin/uvicorn backend.main:app --reload --port 8000

catalog:
	$(PY) -m backend.catalog.build_index

test:
	.venv/bin/pytest

lint:
	.venv/bin/ruff check backend tests

fmt:
	.venv/bin/ruff check --fix backend tests
	.venv/bin/ruff format backend tests

clean:
	rm -rf data/outputs/* data/catalog.db
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

docker:
	docker build -t interior-render .

check: lint test
