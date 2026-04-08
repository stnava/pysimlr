# Makefile for pysimlr

# Configuration - Default to 'python' to pick up active venv
PYTHON ?= python
PIP ?= pip
SRC_DIR := src
TEST_DIR := tests

# Get the absolute path of the current directory, properly quoted for shell
SRC_PATH := "$(shell pwd)/$(SRC_DIR)"

# Default target
.PHONY: all
all: help

.PHONY: help
help:
	@echo "Available targets:"
	@echo "  test       : Run all tests with code coverage (default)"
	@echo "  clean      : Remove build artifacts and temporary files"
	@echo "  install    : Install the package in editable mode"
	@echo "  lint       : Check code style (requires ruff or flake8)"
	@echo "  venv       : Create a local virtual environment"

.PHONY: test
test:
	export PYTHONPATH=$(SRC_PATH):$$PYTHONPATH; \
	$(PYTHON) -m pytest --cov=pysimlr $(TEST_DIR)/ --cov-report=term-missing --cov-report=html

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

.PHONY: install
install:
	$(PIP) install -e .

.PHONY: venv
venv:
	$(PYTHON) -m venv venv
	@echo "Virtual environment created. Run 'source venv/bin/activate' to use it."

.PHONY: lint
lint:
	@if command -v ruff > /dev/null; then \
		ruff check $(SRC_DIR); \
	else \
		echo "ruff not found. Skipping lint."; \
	fi
