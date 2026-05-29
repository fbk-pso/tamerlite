set shell := ["bash", "-cu"]

default:
    @just --list

# Sync the environment from uv.lock (project + dev group + rust extra + workspace)
install:
    uv sync --all-extras

# Build and install the Rust extension in-place via maturin develop
build-rust:
    uv run --no-sync maturin develop --release --manifest-path crates/rustamer/Cargo.toml

# Run pytest. Set PYTHONPATH externally for extra fixtures (e.g. up_test_cases).
test:
    uv run pytest tests/ -v

# Run lint + format checks (read-only)
lint:
    uv run ruff check src tests
    uv run ruff format --check src tests

# Apply formatter and auto-fix lint issues
format:
    uv run ruff format src tests
    uv run ruff check --fix src tests

# Static type checking
typecheck:
    uv run mypy

# Run all pre-commit hooks against the whole repo
precommit:
    uv run pre-commit run --all-files --show-diff-on-failure

# Build sdist + wheel for the tamerlite Python package
build:
    uv build

# Remove build, cache, and tooling artifacts
clean:
    rm -rf build/ dist/ target/ .mypy_cache/ .pytest_cache/ .ruff_cache/
