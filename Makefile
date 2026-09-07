.PHONY: install test lint format smoke quickstart check

install:
	uv sync --frozen --python 3.12

test:
	uv run --frozen pytest -q

lint:
	uv run --frozen ruff check .
	uv run --frozen ruff format --check .

format:
	uv run --frozen ruff format .

smoke:
	uv run --frozen transferbench smoke

quickstart:
	uv run --frozen transferbench matrix configs/quickstart.yaml --report

check: lint test
