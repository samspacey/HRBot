# HR Chatbot Makefile
# Common development tasks and commands

.DEFAULT_GOAL := help
.PHONY: help install install-dev clean test test-unit test-integration test-coverage lint format type-check security-scan pre-commit docker-build docker-run docker-dev setup-dev index query serve

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

clean-cache: ## Clean application cache
	rm -rf cache/*
	rm -rf faiss_index_hr/*

# Testing targets
test: ## Run all tests
	pytest tests/ -v

test-unit: ## Run unit tests only
	pytest tests/unit/ -v

test-integration: ## Run integration tests only
	pytest tests/integration/ -v

test-coverage: ## Run tests with coverage report
	pytest tests/ --cov=. --cov-report=html --cov-report=term-missing

test-watch: ## Run tests in watch mode
	pytest-watch tests/ -- -v

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

security-scan: ## Run security scanning with bandit and safety
	bandit -r . -x tests/
	safety check --file requirements.txt

pre-commit: ## Run all pre-commit hooks
	pre-commit run --all-files

# HR Chatbot specific targets
index: ## Build search index from policy documents
	$(PYTHON) cli.py --index --folder ./policies

query: ## Interactive query mode (example: make query QUESTION="What is the vacation policy?")
	@if [ -z "$(QUESTION)" ]; then \
		echo "Usage: make query QUESTION=\"Your question here\""; \
		echo "Example: make query QUESTION=\"What is the vacation policy?\""; \
	else \
		$(PYTHON) cli.py --query "$(QUESTION)"; \
	fi

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

docker-dev: ## Start development environment with docker-compose
	$(DOCKER_COMPOSE) --profile dev up

docker-stop: ## Stop all Docker containers
	$(DOCKER_COMPOSE) down

docker-logs: ## View Docker container logs
	$(DOCKER_COMPOSE) logs -f

# Jupyter targets
jupyter: ## Start Jupyter Lab for development
	jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root

# Database/Index management
reset-index: clean-cache index ## Reset and rebuild the search index

backup-index: ## Backup the current search index
	@if [ -d "faiss_index_hr" ]; then \
		cp -r faiss_index_hr faiss_index_hr_backup_$(shell date +%Y%m%d_%H%M%S); \
		echo "Index backed up to faiss_index_hr_backup_$(shell date +%Y%m%d_%H%M%S)"; \
	else \
		echo "No index found to backup"; \
	fi

restore-index: ## Restore index from backup (specify BACKUP_DIR)
	@if [ -z "$(BACKUP_DIR)" ]; then \
		echo "Usage: make restore-index BACKUP_DIR=faiss_index_hr_backup_YYYYMMDD_HHMMSS"; \
		ls -la | grep faiss_index_hr_backup || echo "No backup directories found"; \
	elif [ -d "$(BACKUP_DIR)" ]; then \
		rm -rf faiss_index_hr; \
		cp -r $(BACKUP_DIR) faiss_index_hr; \
		echo "Index restored from $(BACKUP_DIR)"; \
	else \
		echo "Backup directory $(BACKUP_DIR) not found"; \
	fi

# Performance testing
benchmark: ## Run performance benchmarks
	@echo "Running performance benchmarks..."
	$(PYTHON) -m pytest tests/ -k "benchmark" -v || echo "No benchmark tests found"

load-test: ## Run load testing with locust
	@echo "Starting load test server..."
	@echo "Visit http://localhost:8089 to configure and run tests"
	locust -f tests/load_test.py --host=http://localhost:8501

# Release targets
build: clean ## Build distribution packages
	$(PYTHON) setup.py sdist bdist_wheel

release-check: ## Check if package is ready for release
	twine check dist/*

# Documentation targets
docs: ## Generate documentation
	@echo "Documentation generation not yet implemented"
	@echo "Will use Sphinx to generate docs from docstrings"

# Development utilities
requirements-update: ## Update requirements files with latest versions
	pip-compile requirements.in
	pip-compile requirements-dev.in

check-deps: ## Check for dependency issues
	pip check

list-outdated: ## List outdated packages
	pip list --outdated

# Quick development cycle
dev-cycle: format lint type-check test ## Run complete development cycle checks

# Production deployment preparation
pre-deploy: clean test-coverage security-scan ## Run all checks before deployment
	@echo "Pre-deployment checks completed successfully"

# Environment information
info: ## Show environment information
	@echo "Python version: $(shell $(PYTHON) --version)"
	@echo "Pip version: $(shell $(PIP) --version)"
	@echo "Current directory: $(shell pwd)"
	@echo "Git branch: $(shell git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'Not a git repository')"
	@echo "Git commit: $(shell git rev-parse HEAD 2>/dev/null || echo 'Not a git repository')"
	@echo "Virtual environment: $(shell echo $$VIRTUAL_ENV || echo 'None')"