.DEFAULT_GOAL := help
.PHONY: help install test test-agentdojo lint format smoke plan quickstart reproduce build check

help:
	@printf '%s\n' \
		'install        Install the locked Python 3.12 environment' \
		'check          Run lint, formatting checks, and offline tests' \
		'smoke          Exercise all four scaffolds with fake models' \
		'plan           Preview the quickstart without running models' \
		'quickstart     Run the offline quickstart and generate a report' \
		'reproduce      Run the smaller 64-episode reproducible subset' \
		'test-agentdojo Test the optional pinned AgentDojo seed exporter' \
		'build          Build source and wheel distributions'

install:
	uv sync --frozen --python 3.12

test:
	uv run --frozen pytest -q

test-agentdojo:
	uv run --frozen --extra agentdojo pytest -q tests/test_agentdojo_upstream.py

lint:
	uv run --frozen ruff check .
	uv run --frozen ruff format --check .

format:
	uv run --frozen ruff format .

smoke:
	uv run --frozen transferbench smoke

plan:
	uv run --frozen transferbench matrix configs/quickstart.yaml --dry-run

quickstart:
	uv run --frozen transferbench matrix configs/quickstart.yaml --report

reproduce:
	uv run --frozen transferbench matrix configs/reproduce.yaml --report

build:
	uv build

check: lint test
