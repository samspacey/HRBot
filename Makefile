"#" Minimal POC Makefile
"#" Common development tasks

.DEFAULT_GOAL := help
.PHONY: help install install-dev clean lint format type-check setup-dev serve serve-dev docker-build docker-run info

# Variables
PYTHON := python3
PIP := pip
STREAMLIT := streamlit
DOCKER := docker
DOCKER_COMPOSE := docker-compose

# Help target
help: ## Show this help message
	@echo "HR Chatbot Development Commands"
	@echo "==============================="
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Installation targets
install: ## Install production dependencies
	$(PIP) install -r requirements.txt

install-dev: ## Install development dependencies
	$(PIP) install -r requirements-dev.txt
	pre-commit install


# Environment setup
setup-dev: install-dev ## Set up development environment
	@echo "Setting up development environment..."
	@if [ ! -f .env ]; then cp .env.example .env; echo "Created .env from .env.example - please update with your API keys"; fi
	@mkdir -p cache notebooks logs
	@echo "Development environment setup complete!"

# Cleaning targets
clean: ## Clean up temporary files and caches
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	rm -rf build/ dist/ .coverage htmlcov/

# Code quality targets
lint: ## Run linting (flake8)
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 . --count --max-complexity=10 --max-line-length=88 --statistics

format: ## Format code with black and isort
	black .
	isort .

format-check: ## Check code formatting without making changes
	black --check .
	isort --check-only .

type-check: ## Run type checking with mypy
	mypy . --ignore-missing-imports

serve: ## Start Streamlit web interface
	$(STREAMLIT) run streamlit_app.py

serve-dev: ## Start Streamlit in development mode with auto-reload
	$(STREAMLIT) run streamlit_app.py --server.runOnSave=true --server.fileWatcherType=poll

# Docker targets
docker-build: ## Build Docker image
	$(DOCKER) build -t hr-chatbot .

docker-build-dev: ## Build development Docker image
	$(DOCKER) build -f Dockerfile.dev -t hr-chatbot:dev .

docker-run: ## Run Docker container
	$(DOCKER) run -p 8501:8501 --env-file .env -v $(PWD)/policies:/app/policies:ro hr-chatbot

# Environment information
info: ## Show environment information
	@echo "Python version: $(shell $(PYTHON) --version)"
	@echo "Pip version: $(shell $(PIP) --version)"
	@echo "Current directory: $(shell pwd)"
	@echo "Git branch: $(shell git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'Not a git repository')"
	@echo "Git commit: $(shell git rev-parse HEAD 2>/dev/null || echo 'Not a git repository')"
	@echo "Virtual environment: $(shell echo $$VIRTUAL_ENV || echo 'None')"
