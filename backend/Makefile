.PHONY: help install dev clean lint format test run index split server stats

help:
	@echo "Available commands:"
	@echo "  make install    - Install dependencies"
	@echo "  make dev        - Install dev dependencies"
	@echo "  make clean      - Clean cache and build files"
	@echo "  make lint       - Run linting"
	@echo "  make format     - Format code"
	@echo "  make test       - Run tests"
	@echo "  make run        - Run the server"
	@echo "  make index      - Create image index"
	@echo "  make split      - Split dataset"
	@echo "  make server     - Run server with reload"
	@echo "  make stats      - Show index statistics"

install:
	uv sync

dev:
	uv sync --dev

clean:
	@echo "Cleaning cache and build files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ .pytest_cache/ .mypy_cache/ .coverage htmlcov/
	@echo "Clean complete!"

lint:
	uv run ruff check src/

format:
	uv run ruff format src/

test:
	uv run pytest tests/ -v

run:
	uv run backend server

server:
	uv run backend server --host 0.0.0.0 --port 8000

index:
	uv run backend index

split:
	uv run backend split -r 0.01

stats:
	uv run backend stats

setup-dirs:
	@echo "Creating data directories..."
	mkdir -p data/archive data/train data/test data/index
	@echo "Directories created!"
