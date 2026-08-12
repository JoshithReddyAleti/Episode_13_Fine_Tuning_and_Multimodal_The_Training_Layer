.PHONY: help install test bench lint fmt clean docker serve

help:
	@echo "Episode 13 — Fine-Tuning + Multimodal"
	@echo ""
	@echo "Targets:"
	@echo "  make install       Install dependencies"
	@echo "  make test          Run test suite"
	@echo "  make bench         Run benchmarks"
	@echo "  make lint          Ruff + mypy"
	@echo "  make fmt           Auto-format with ruff"
	@echo "  make clean         Remove build artifacts"
	@echo "  make docker        Build container image"
	@echo "  make serve         Start multi-LoRA vLLM via docker-compose"

install:
	pip install --upgrade pip && pip install -r requirements-dev.txt

test:
	pytest tests/ -v

bench:
	python benchmarks/bench_dataset_validation.py
	python benchmarks/bench_lora_size_estimates.py
	python benchmarks/bench_multimodal_estimates.py

lint:
	ruff check src/ tests/ examples/ benchmarks/
	mypy src/utils/ --ignore-missing-imports

fmt:
	ruff format src/ tests/ examples/ benchmarks/
	ruff check --fix src/ tests/ examples/ benchmarks/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true

docker:
	docker build -t ep13:latest .

serve:
	docker compose up -d
