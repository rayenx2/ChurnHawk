.PHONY: help install test lint format clean docker-up docker-down dvc-setup train evaluate deploy

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
RED := \033[0;31m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(BLUE)Churn Prediction Pipeline - Available Commands:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}'

install: ## Install dependencies
	@echo "$(BLUE)Installing dependencies...$(NC)"
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install -e .
	@echo "$(GREEN)✓ Dependencies installed$(NC)"

install-dev: ## Install development dependencies
	@echo "$(BLUE)Installing dev dependencies...$(NC)"
	pip install -r requirements-dev.txt
	pre-commit install
	@echo "$(GREEN)✓ Dev dependencies installed$(NC)"

test: ## Run all tests
	@echo "$(BLUE)Running tests...$(NC)"
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing
	@echo "$(GREEN)✓ Tests complete$(NC)"

test-unit: ## Run unit tests only
	@echo "$(BLUE)Running unit tests...$(NC)"
	pytest tests/unit/ -v
	@echo "$(GREEN)✓ Unit tests complete$(NC)"

test-integration: ## Run integration tests only
	@echo "$(BLUE)Running integration tests...$(NC)"
	pytest tests/integration/ -v
	@echo "$(GREEN)✓ Integration tests complete$(NC)"

lint: ## Run linting checks
	@echo "$(BLUE)Running linters...$(NC)"
	black --check src/ tests/
	isort --check-only src/ tests/
	flake8 src/ tests/ --max-line-length=100
	mypy src/ --ignore-missing-imports
	@echo "$(GREEN)✓ Linting complete$(NC)"

format: ## Format code with black and isort
	@echo "$(BLUE)Formatting code...$(NC)"
	black src/ tests/
	isort src/ tests/
	@echo "$(GREEN)✓ Code formatted$(NC)"

clean: ## Clean temporary files
	@echo "$(BLUE)Cleaning temporary files...$(NC)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ htmlcov/ .coverage
	@echo "$(GREEN)✓ Cleanup complete$(NC)"

docker-build: ## Build Docker images
	@echo "$(BLUE)Building Docker images...$(NC)"
	docker compose build
	@echo "$(GREEN)✓ Docker images built$(NC)"

docker-up: ## Start all services
	@echo "$(BLUE)Starting services...$(NC)"
	docker compose up -d
	@echo "$(GREEN)✓ Services started$(NC)"
	@echo "Access points:"
	@echo "  - Airflow:   http://localhost:8080"
	@echo "  - MLflow:    http://localhost:5000"
	@echo "  - MinIO:     http://localhost:9001"
	@echo "  - API:       http://localhost:8000"
	@echo "  - Grafana:   http://localhost:3000"

docker-down: ## Stop all services
	@echo "$(BLUE)Stopping services...$(NC)"
	docker compose down
	@echo "$(GREEN)✓ Services stopped$(NC)"

docker-down-volumes: ## Stop services and remove volumes
	@echo "$(RED)Stopping services and removing volumes...$(NC)"
	docker compose down -v
	@echo "$(GREEN)✓ Services stopped and volumes removed$(NC)"

docker-logs: ## View Docker logs
	docker compose logs -f

docker-ps: ## List running containers
	docker compose ps

dvc-setup: ## Setup DVC with MinIO
	@echo "$(BLUE)Setting up DVC...$(NC)"
	bash scripts/setup_dvc.sh
	@echo "$(GREEN)✓ DVC setup complete$(NC)"

dvc-pull: ## Pull data from DVC remote
	@echo "$(BLUE)Pulling data from DVC remote...$(NC)"
	dvc pull
	@echo "$(GREEN)✓ Data pulled$(NC)"

dvc-push: ## Push data to DVC remote
	@echo "$(BLUE)Pushing data to DVC remote...$(NC)"
	dvc push
	@echo "$(GREEN)✓ Data pushed$(NC)"

dvc-status: ## Check DVC status
	dvc status

train: ## Run training pipeline with DVC
	@echo "$(BLUE)Running training pipeline...$(NC)"
	dvc repro
	@echo "$(GREEN)✓ Training complete$(NC)"

train-params: ## Show current training parameters
	@echo "$(BLUE)Current training parameters:$(NC)"
	cat params.yaml

evaluate: ## Evaluate model on test set
	@echo "$(BLUE)Evaluating model...$(NC)"
	dvc repro evaluate_model
	@echo "$(GREEN)✓ Evaluation complete$(NC)"

metrics: ## Show model metrics
	@echo "$(BLUE)Model Metrics:$(NC)"
	dvc metrics show

plots: ## Show model plots
	@echo "$(BLUE)Generating plots...$(NC)"
	dvc plots show

compare: ## Compare experiments
	@echo "$(BLUE)Comparing experiments...$(NC)"
	dvc metrics diff
	dvc plots diff

mlflow-ui: ## Start MLflow UI
	@echo "$(BLUE)Starting MLflow UI...$(NC)"
	mlflow ui --host 0.0.0.0 --port 5000

api-run: ## Run inference API locally
	@echo "$(BLUE)Starting inference API...$(NC)"
	uvicorn src.inference:app --reload --host 0.0.0.0 --port 8000

api-test: ## Test API endpoint
	@echo "$(BLUE)Testing API...$(NC)"
	curl -X POST http://localhost:8000/predict \
	  -H "Content-Type: application/json" \
	  -d @tests/fixtures/sample_request.json

monitoring-up: ## Start monitoring stack
	@echo "$(BLUE)Starting monitoring stack...$(NC)"
	docker compose -f monitoring/docker-compose.monitoring.yml up -d
	@echo "$(GREEN)✓ Monitoring started$(NC)"
	@echo "Access Grafana: http://localhost:3000"

monitoring-down: ## Stop monitoring stack
	@echo "$(BLUE)Stopping monitoring stack...$(NC)"
	docker compose -f monitoring/docker-compose.monitoring.yml down
	@echo "$(GREEN)✓ Monitoring stopped$(NC)"

pre-commit-run: ## Run pre-commit hooks on all files
	@echo "$(BLUE)Running pre-commit hooks...$(NC)"
	pre-commit run --all-files
	@echo "$(GREEN)✓ Pre-commit complete$(NC)"

ci-local: ## Run CI checks locally
	@echo "$(BLUE)Running CI checks locally...$(NC)"
	make lint
	make test
	make docker-build
	@echo "$(GREEN)✓ All CI checks passed$(NC)"

deploy: ## Deploy to production (placeholder)
	@echo "$(RED)Deploy command not yet implemented$(NC)"
	@echo "Please configure your deployment strategy"

all: clean install test lint docker-build ## Run all setup and checks
