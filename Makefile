PYTHON ?= python3

.PHONY: install format lint typecheck test check cli-help

install:
	$(PYTHON) -m pip install -e ".[dev]"

format:
	ruff format src/ tests/

lint:
	ruff check src/ tests/

typecheck:
	mypy src/amd/

test:
	pytest

check: format lint typecheck test

cli-help:
	amd --help