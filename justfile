set shell := ["bash", "-cu"]

default:
    @just --list

# Sync the environment from uv.lock (project + dev group + rust extra + workspace)
install:
    uv sync --all-extras
    just build-rust

# Build and install the Rust extension in-place via maturin develop
build-rust:
    uv run --no-sync maturin develop --release --manifest-path crates/rustamer/Cargo.toml

# NOTE: every recipe below uses `uv run --no-sync`. A bare `uv run` re-syncs the
# environment first and, when it decides the editable `rustamer` member needs
# rebuilding, reinstalls it through uv's own PEP 517 path -- silently replacing
# the wheel `just build-rust` put there, sometimes with a stale cached one. Use
# `just install` as the single syncing entry point.

# Run pytest in parallel (pytest-xdist). Set PYTHONPATH externally for extra fixtures (e.g. up_test_cases; see `just up-checkout`).
test:
    uv run --no-sync pytest tests/ -n auto

# Run all lint and formatting checks (Python + Rust). Fails if any issues are found.
lint:
    uv run --no-sync ruff check src tests ci crates
    uv run --no-sync ruff format --check src tests ci crates
    cargo fmt --all -- --check
    cargo clippy --workspace --all-targets -- -D warnings

# Apply Python and Rust formatters and auto-fix lint issues where possible
format:
    uv run --no-sync ruff format src tests ci crates
    uv run --no-sync ruff check --fix src tests ci crates
    cargo fmt --all
    cargo clippy --workspace --all-targets --fix --allow-dirty --allow-staged

# Static type checking
typecheck:
    uv run --no-sync mypy

# Run all pre-commit hooks against the whole repo
precommit:
    uv run --no-sync pre-commit run --all-files --show-diff-on-failure

# up_test_cases fixtures live in ./up-checkout; a mismatch against the installed
# unified-planning shows up as NameError collection failures or spurious
# regression diffs. Uses system python3 (stdlib tomllib) so it works on
# interpreters older than the one uv installed, matching the CI step in
# .github/workflows/test.yml.
# Clone or re-pin ./up-checkout to the unified-planning commit uv.lock resolves
up-checkout:
    #!/usr/bin/env bash
    set -euo pipefail
    sha=$(python3 -c "import tomllib; d = tomllib.load(open('uv.lock', 'rb')); print(next(p['source']['git'].rsplit('#', 1)[1] for p in d['package'] if p['name'] == 'unified-planning'))")
    if [ -d up-checkout ]; then
        git -C up-checkout fetch --filter=blob:none origin
    else
        git clone --filter=blob:none https://github.com/aiplan4eu/unified-planning.git up-checkout
    fi
    git -C up-checkout checkout --detach --quiet "$sha"
    echo "up-checkout pinned to $sha"
    echo "run the suite with: PYTHONPATH=up-checkout/up_test_cases just test"

# Build sdist + wheel for the tamerlite Python package (hatchling).
# Used by the CI tamerlite job; locally callable on its own.
build-python:
    uv build

# Build the rustamer wheel + sdist (current interpreter only) via maturin.
build-rust-wheel:
    uv run --no-sync maturin build --release --out dist \
        --manifest-path crates/rustamer/Cargo.toml
    uv run --no-sync maturin sdist --out dist \
        --manifest-path crates/rustamer/Cargo.toml

# Build BOTH wheels + sdists into ./dist/ (release-equivalent local mirror).
# Note: the rustamer wheel here covers only the current Python interpreter;
# CI's maturin-action handles the multi-Python matrix on tag.
build: clean-dist build-python build-rust-wheel
    @ls -1 dist/

# Helper: remove ./dist/ for a clean build state.
clean-dist:
    rm -rf dist

# Verify pyproject.toml and root Cargo.toml base versions agree
# Uses python3 directly (stdlib only) so it doesn't re-resolve uv during a bump.
check-versions:
    python3 ci/check_versions.py

# Print rustamer and tamerlite versions and verify they match
check-installed-versions:
    @uv run --no-sync python -c 'from importlib.metadata import version;\
    r=version("rustamer"); t=version("tamerlite"); print(f"rustamer:  {r}"); print(f"tamerlite: {t}");\
    assert r == t, f"Version mismatch: rustamer={r}, tamerlite={t}"; print("✓ Versions match")'

# Bump version everywhere (pyproject.toml, root Cargo.toml, rustamer crate pyproject, rustamer pin) and refresh both lockfiles
bump version:
    sed -i 's/^version = ".*"/version = "{{version}}"/' pyproject.toml
    sed -i 's/^version = ".*"/version = "{{version}}"/' Cargo.toml
    sed -i 's/^version = ".*"/version = "{{version}}"/' crates/rustamer/pyproject.toml
    sed -i 's/"rustamer==.*"/"rustamer=={{version}}"/' pyproject.toml
    just check-versions
    # re-sync the rustamer/rustamer-base version entries in Cargo.lock (bump only
    # edits Cargo.toml; without this the committed lock stays at the old version).
    cargo update --workspace
    uv lock
    @echo "Now: git commit -am 'release: v{{version}}' && git tag v{{version}} && git push --follow-tags"

# Remove build, cache, and tooling artifacts
clean:
    rm -rf build/ dist/ target/ .mypy_cache/ .pytest_cache/ .ruff_cache/
