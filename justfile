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

# Run all lint + format checks (Python + Rust). Read-only.
lint:
    uv run ruff check src tests ci
    uv run ruff format --check src tests ci
    cargo fmt --all -- --check
    cargo clippy --workspace --all-targets -- -W warnings

# Apply formatters (Python + Rust) and auto-fix Python lint issues
format:
    uv run ruff format src tests ci
    uv run ruff check --fix src tests ci
    cargo fmt --all

# Static type checking
typecheck:
    uv run mypy

# Run all pre-commit hooks against the whole repo
precommit:
    uv run pre-commit run --all-files --show-diff-on-failure

# Build sdist + wheel for the tamerlite Python package
build:
    uv build

# Verify pyproject.toml and root Cargo.toml base versions agree
# Uses python3 directly (stdlib only) so it doesn't re-resolve uv during a bump.
check-versions:
    python3 ci/check_versions.py

# Bump version in pyproject.toml, root Cargo.toml, and the rustamer pin
bump version:
    sed -i 's/^version = ".*"/version = "{{version}}"/' pyproject.toml
    sed -i 's/^version = ".*"/version = "{{version}}"/' Cargo.toml
    sed -i 's/"rustamer==.*"/"rustamer=={{version}}"/' pyproject.toml
    just check-versions
    uv lock
    @echo "Now: git commit -am 'release: v{{version}}' && git tag v{{version}} && git push --follow-tags"

# Remove build, cache, and tooling artifacts
clean:
    rm -rf build/ dist/ target/ .mypy_cache/ .pytest_cache/ .ruff_cache/
