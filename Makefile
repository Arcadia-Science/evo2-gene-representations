.PHONY: lint
lint:
	uv run --frozen ruff check .

.PHONY: test
test:
	uv run --frozen pytest -q

.PHONY: format
format:
	uv run --frozen ruff check --fix .
	uv run --frozen ruff format .

.PHONY: pre-commit
pre-commit:
	uv run --frozen pre-commit run --all-files
