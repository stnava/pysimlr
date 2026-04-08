# Makefile for pysimlr

# Configuration
PYTHON := python3
PIP := pip
PYTEST := pytest
SRC_DIR := src
TEST_DIR := tests
VENV_DIR := venv

# Default target
.PHONY: all
all: help

.PHONY: help
help:
	@echo "Available targets:"
	@echo "  test       : Run all tests"
	@echo "  test-cov   : Run tests with coverage report"
	@echo "  clean      : Remove build artifacts and temporary files"
	@echo "  install    : Install the package in editable mode"
	@echo "  lint       : Check code style (requires ruff or flake8)"
	@echo "  venv       : Create a local virtual environment (Python 3.12)"

.PHONY: test
test:
	export PYTHONPATH=$(shell pwd)/$(SRC_DIR):$$PYTHONPATH; \
	$(PYTHON) -m pytest $(TEST_DIR)/test_pysimlr.py

.PHONY: test-cov
test-cov:
	export PYTHONPATH=$(shell pwd)/$(SRC_DIR):$$PYTHONPATH; \
	$(PYTHON) -m pytest --cov=$(SRC_DIR)/pysimlr $(TEST_DIR)/test_pysimlr.py --cov-report=term-missing

.PHONY: clean
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete

.PHONY: install
install:
	$(PIP) install -e .

.PHONY: venv
venv:
	$(PYTHON) -m venv $(VENV_DIR)
	@echo "Virtual environment created in $(VENV_DIR). Run 'source $(VENV_DIR)/bin/activate' to use it."

.PHONY: lint
lint:
	@if command -v ruff > /dev/null; then \
		ruff check $(SRC_DIR); \
	else \
		echo "ruff not found. Skipping lint."; \
	fi
