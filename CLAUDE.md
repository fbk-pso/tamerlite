# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick orientation

TamerLite is a heuristic search-based temporal planner ([README.md](README.md)). It ships **two PyPI wheels** that are released in lockstep at the same `X.Y.Z`:

- `tamerlite` — pure-Python wrapper (hatchling), source under `src/tamerlite/`.
- `rustamer` — PyO3 wheel (maturin) providing a Rust acceleration backend; source under `crates/rustamer/`. Depends on the internal Rust crate `crates/rustamer-base/`.

At runtime [src/tamerlite/core/__init__.py](src/tamerlite/core/__init__.py) tries `import rustamer`; on `ImportError` or if `DISABLE_RUSTAMER=true` it falls back to the pure-Python implementations in `src/tamerlite/core/`. **The fallback is load-bearing and tested**: parametrized tests in [tests/test_engine.py](tests/test_engine.py) exercise both code paths.

## Repository layout

```
/
├── pyproject.toml         # tamerlite (hatchling); uv project root + workspace config
├── Cargo.toml             # Cargo workspace root; inherited package/version by members
├── Cargo.lock             # committed for reproducible Rust builds
├── uv.lock                # committed; pins all Python deps incl. unified-planning git rev
├── src/tamerlite/         # Python package (PEP-660 src layout)
├── crates/
│   ├── rustamer/          # PyO3 wheel (maturin); pyproject.toml + Cargo.toml
│   └── rustamer-base/     # core Rust crate (no Python wheel)
├── tests/                 # pytest suite, regression baselines, PDDL fixtures
├── ci/                    # Python helper scripts used by CI
│   ├── check_versions.py  # version-equality guard
│   └── stamp_dev_version.py  # dev-version stamper for main builds
├── justfile               # task runner — single source of truth for dev + CI commands
├── .pre-commit-config.yaml
└── .github/workflows/
    ├── test.yml           # reusable: lint + test matrix
    ├── ci-pr.yml          # PR trigger
    └── build-and-release.yml  # main + tag trigger; builds, publishes, releases
```

## Setup

```bash
uv sync --all-extras       # creates .venv, builds rustamer workspace member, installs dev deps
uv run pre-commit install  # one-time
```

`uv` is required. `just` is the task runner — install via `uv tool install rust-just` or your package manager.

## Common tasks (all via `just`)

| Recipe | What it does |
|---|---|
| `just install` | `uv sync --all-extras` |
| `just build-rust` | `maturin develop` — rebuild rustamer in-place into the venv (dev iteration) |
| `just build` | Produce both `tamerlite` and `rustamer` wheels + sdists into `./dist/` |
| `just build-python` | Only the `tamerlite` wheel (used by CI's `tamerlite` publish job) |
| `just build-rust-wheel` | Only the `rustamer` wheel + sdist for current interpreter |
| `just test` | `uv run pytest tests/ -v` — set `PYTHONPATH=up-checkout/up_test_cases` if you need the UP fixtures |
| `just lint` | Ruff (check + format --check) + cargo fmt --check + cargo clippy (informational, -W warnings) |
| `just format` | Ruff format + ruff --fix + cargo fmt --all |
| `just typecheck` | `uv run mypy` (config in `pyproject.toml`, scope `src/tamerlite`) |
| `just precommit` | `pre-commit run --all-files --show-diff-on-failure` — same as CI's `lint` job |
| `just check-versions` | Verify pyproject + Cargo agree on base `X.Y.Z` |
| `just bump VERSION` | Update version in pyproject + Cargo + rustamer pin; refresh `uv.lock` |
| `just clean` | Remove build/dist/target/cache dirs |

## Running the test suite (with UP fixtures)

Most tests need `unified-planning`'s `up_test_cases/` directory. `uv sync` installs `unified-planning` from a pinned git commit; the test fixtures live in the same repo and must be cloned and **checked out to the locked commit** (CI does this automatically in [test.yml](.github/workflows/test.yml)):

```bash
sha=$(python3 -c "import tomllib; d=tomllib.load(open('uv.lock','rb')); print(next(p['source']['git'].rsplit('#',1)[1] for p in d['package'] if p['name']=='unified-planning'))")
git clone --filter=blob:none https://github.com/aiplan4eu/unified-planning.git up-checkout
git -C up-checkout checkout "$sha"
PYTHONPATH=up-checkout/up_test_cases just test
```

Mismatched commits cause `NameError` collection failures from newer TAMP fixtures referencing symbols absent in the installed UP version.

## Architecture

### Python/Rust dual implementation ([src/tamerlite/core/](src/tamerlite/core/))

[src/tamerlite/core/__init__.py](src/tamerlite/core/__init__.py) is the dispatch point. The exposed interface is identical between backends:

- **Search algorithms**: `wastar_search`, `astar_search`, `gbfs_search`, `bfs_search`, `dfs_search`, `ehc_search`, `multiqueue_search` (and `*_memory_bounded` variants).
- **Heuristics**: `HFF`, `HAdd`, `HMax`, `HMaxExplicit`, `CustomHeuristic`.
- **Data structures**: `SearchSpace`, `State`, `Action`, `Event`, `Effect`, `Timing`, `Expression`.

Rust implementation lives in [crates/rustamer-base/src/](crates/rustamer-base/src/) (core library) and [crates/rustamer/src/](crates/rustamer/src/) (PyO3 bindings).

### Problem encoding ([src/tamerlite/encoder.py](src/tamerlite/encoder.py))

`Encoder` bridges Unified Planning and TamerLite's internal search space:

1. Accepts a grounded UP `Problem` and a lifted one (for map-back).
2. Converts UP fluents/actions/conditions/effects into internal `Expression`/`Event`/`Action` via [src/tamerlite/converter.py](src/tamerlite/converter.py) (a `DagWalker` over UP expression trees).
3. Builds the internal `SearchSpace`.
4. Optional preprocessing: symmetry breaking, compression-safe action identification, relevance analysis via HMax reachability.

### Engine ([src/tamerlite/engine.py](src/tamerlite/engine.py))

`TamerLite` implements both `OneshotPlannerMixin` and `AnytimePlannerMixin`.

**Solve pipeline** (`_solve` / `_solve_ground_problem`):
1. Compile the UP problem: remove undefined numeric initials → ground.
2. If all actions are *compression-safe*, further compile temporal → sequential via UP's `TimedToSequential`.
3. Build an `Encoder` from the grounded problem.
4. Run the selected search with the selected heuristic.
5. Reconstruct and map back the plan.

**Anytime** (`_get_solutions_with_params`): iteratively tightens the quality constraint and re-solves until UNSAT or timeout.

### Configuration

`SearchParams` (single queue) and `MultiqueueParams` (parallel queues) are frozen dataclasses passed via `params={"search": ...}` to the UP planner factory. Default: `wastar` + `hff` at weight `0.8`.

### Test infrastructure ([tests/](tests/))

- [tests/problems_generator.py](tests/problems_generator.py) — synthetic UP problems (logistics, numeric, satellite, temporal flight, hierarchical types).
- [tests/testing_utils.py](tests/testing_utils.py) — helpers for compiling problems, checking kind.
- [tests/test_engine.py](tests/test_engine.py) — parametric tests over all (search × heuristic × Rust/Python) combinations.
- `tests/pddl/` — PDDL files for additional cases.
- `tests/test_engine/` — pytest-regressions baselines.

## Versioning and release flow

Versions live **manually** in two places and CI enforces equality:

- `pyproject.toml` → `[project].version`
- `Cargo.toml` → `[workspace.package].version` (inherited by both crates)
- The `rustamer==X.Y.Z` pin in `pyproject.toml` → `[project.optional-dependencies].rust`

Pre-release / dev versions follow this scheme:

- Python (PEP 440): `<base>.dev<N>+g<sha>` (e.g. `0.2.0.dev42+gabc1234`)
- Cargo (SemVer): `<base>-dev.<N>` (e.g. `0.2.0-dev.42`)

`N` = `git rev-list --count HEAD`.

**Cut a release:**
```bash
just bump 0.2.0
git commit -am "release: v0.2.0"
git tag v0.2.0 && git push --follow-tags
```

The `v*` tag triggers [build-and-release.yml](.github/workflows/build-and-release.yml):
- `publish-rustamer` / `publish-tamerlite` → `pypa/gh-action-pypi-publish@release/v1` using **PyPI Trusted Publishing** (OIDC). Each job declares a GitHub environment (`pypi-rustamer` / `pypi-tamerlite`) that matches the corresponding pending publisher registered on PyPI; no API tokens are stored in the repo.
- `github-release` → `softprops/action-gh-release@v2` with auto-generated notes (from PR titles since the previous tag) and all wheels attached.

Both `github-release` and `dev-release` jobs authenticate with an **installation token from the `tamerlite-releaser` GitHub App** (`actions/create-github-app-token@v1`), not `GITHUB_TOKEN` — because the org policy locks workflow tokens to read-only. The App is installed only on this repo with `Contents: read/write`; its credentials live in two repo secrets: `RELEASER_APP_ID` and `RELEASER_APP_PRIVATE_KEY`.

**Every push to `main`:**
- `stamp-dev` stamps the dev version into pyproject + Cargo (artifact only, not committed)
- Build jobs produce dev-versioned wheels
- `dev-release` replaces a rolling GitHub pre-release tagged `dev` with the new wheels (`pip install --pre <url>` for testing)

## Tooling-related conventions

- All formatting via `ruff format` (config in `pyproject.toml` → `[tool.ruff.format]`).
- Mypy config in `pyproject.toml` → `[tool.mypy]`.
- **`tests/` is outside `src/`** (modern best practice).
- `Cargo.lock` is **committed** — uncommon for libraries but right for a workspace shipping a cdylib wheel.
- The justfile's `check-versions` recipe calls `python3` directly (not `uv run`) so it doesn't trigger a uv resolve mid-bump.
- Ruff is scoped to `src tests ci`.
- `clippy` is informational (`-W warnings`) until the ~27-warning backlog on `rustamer-base` is cleaned up; switch to `-D warnings` in [justfile](justfile) + [.pre-commit-config.yaml](.pre-commit-config.yaml) when ready.
