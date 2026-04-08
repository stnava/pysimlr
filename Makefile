# Makefile for pysimlr

# Configuration - Default to 'python' to pick up active venv
PYTHON ?= python
PIP ?= pip
SRC_DIR := src
TEST_DIR := tests
DOCS_DIR := docs

# Get the absolute path of the current directory, properly quoted for shell
SRC_PATH := "$(shell pwd)/$(SRC_DIR)"

# Default target
.PHONY: all
all: help

.PHONY: help
help:
	@echo "Available targets:"
	@echo "  test       : Run all tests with code coverage (default)"
	@echo "  docs       : Render Quarto tutorials using the active virtual environment"
	@echo "  clean      : Remove build artifacts and temporary files"
	@echo "  install    : Install the package in editable mode"
	@echo "  lint       : Check code style (requires ruff or flake8)"
	@echo "  venv       : Create a local virtual environment"

.PHONY: test
test:
	export PYTHONPATH=$(SRC_PATH):$$PYTHONPATH; \
	$(PYTHON) -m pytest --cov=pysimlr $(TEST_DIR)/ --cov-report=term-missing --cov-report=html

.PHONY: docs
docs:
	@echo "Rendering Quarto docs using python: $(shell which $(PYTHON))"
	export QUARTO_PYTHON=$(shell which $(PYTHON)); \
	export PYTHONPATH=$(SRC_PATH):$$PYTHONPATH; \
	quarto render $(DOCS_DIR)

.PHONY: clean
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf $(DOCS_DIR)/*_files
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

.PHONY: install
install:
	$(PIP) install -e .[dev]

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
