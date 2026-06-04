# Contributing to TamerLite

Thanks for wanting to hack on TamerLite. This file covers the practical
"how do I run X" side of the repo. For a deeper architectural reference
(dispatch points, file paths, design rationale) see [CLAUDE.md](CLAUDE.md).

The Rust acceleration (`rustamer`) is **optional** at runtime: when
`rustamer` cannot be imported (or `DISABLE_RUSTAMER=1` is set), TamerLite
falls back to the pure-Python implementations in
[`src/tamerlite/core/`](src/tamerlite/core/). This fallback is load-bearing
and tested.

## Setup

Prerequisites:

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) — Python project / environment manager
- [just](https://github.com/casey/just) — command runner (install with
  `uv tool install rust-just` or your package manager)
- Rust toolchain (for building the `rustamer` extension)

Clone and bootstrap:

```bash
git clone https://github.com/fbk-pso/tamerlite.git
cd tamerlite
uv sync --all-extras       # creates .venv, builds rustamer workspace member, installs dev deps
uv run pre-commit install  # one-time
```

This repo uses **uv** workspace (single root `uv.lock` covering both
Python and Rust), **ruff** for lint+format, **mypy** for type-checking,
**pre-commit** for hooks (enforced in CI), and **just** as the
unified task runner.

## Common tasks (via `just`)

| Recipe                  | What it does                                                                                          |
|-------------------------|-------------------------------------------------------------------------------------------------------|
| `just install`          | `uv sync --all-extras`                                                                                |
| `just build-rust`       | `maturin develop` — rebuild `rustamer` in-place into the venv (fast dev iteration)                    |
| `just build`            | Produce both `tamerlite` and `rustamer` wheels + sdists into `./dist/`                                |
| `just build-python`     | Only the `tamerlite` wheel (used by CI's tamerlite publish job)                                       |
| `just build-rust-wheel` | Only the `rustamer` wheel + sdist for the current interpreter                                         |
| `just test`             | `uv run pytest tests/ -v` (set `PYTHONPATH=up-checkout/up_test_cases` to use the UP fixtures)         |
| `just lint`             | ruff check + ruff format --check + cargo fmt --check + cargo clippy (informational, `-W warnings`)    |
| `just format`           | ruff format + ruff check --fix + cargo fmt --all                                                      |
| `just typecheck`        | `uv run mypy` (config in `pyproject.toml`, scope `src/tamerlite`)                                     |
| `just precommit`        | `pre-commit run --all-files --show-diff-on-failure` (same command CI's lint job runs)                 |
| `just check-versions`   | Assert that `pyproject.toml` and `Cargo.toml` agree on base `X.Y.Z`                                   |
| `just bump VERSION`     | Update version in `pyproject.toml` + root `Cargo.toml` + the `rustamer` pin; refresh `uv.lock`        |
| `just clean`            | Remove build / dist / target / cache directories                                                      |

Running `just precommit` locally reproduces exactly what CI's `lint` job
does, including the cargo fmt strict check and the (informational)
clippy run.

## Running the test suite with UP fixtures

Most tests need `unified-planning`'s `up_test_cases/` directory. `uv sync`
installs `unified-planning` from a pinned git commit; clone the same
commit's tree separately for the fixtures:

```bash
sha=$(python3 -c "import tomllib; d=tomllib.load(open('uv.lock','rb')); print(next(p['source']['git'].rsplit('#',1)[1] for p in d['package'] if p['name']=='unified-planning'))")
git clone --filter=blob:none https://github.com/aiplan4eu/unified-planning.git up-checkout
git -C up-checkout checkout "$sha"
PYTHONPATH=up-checkout/up_test_cases just test
```

CI does this automatically; locally it's a one-off.

## Development wheels

Every push to `main` builds platform wheels and publishes them as a
rolling pre-release tagged `dev` under
[Releases](https://github.com/fbk-pso/tamerlite/releases/tag/dev). Wheels
carry PEP 440 dev versions like `0.2.0.dev42+g<sha>`. To pick one up:

```bash
pip install --pre <url-of-wheel-on-dev-release>
```

Useful for smoke-testing main against downstream code without waiting for
a tagged release.

## Cutting a release

```bash
just bump 0.2.0
git commit -am "release: v0.2.0"
git tag v0.2.0 && git push --follow-tags
```

`just bump` updates `pyproject.toml`, root `Cargo.toml`, and the
`rustamer==X.Y.Z` pin in lockstep. CI's `lint` job enforces equality
between `pyproject.toml` and `Cargo.toml` on every push via
`just check-versions`.

The push of a `v*` tag triggers
[build-and-release.yml](.github/workflows/build-and-release.yml), which:

- Publishes `rustamer` and `tamerlite` to PyPI via **Trusted Publishing**
  (OIDC; no API tokens in the repo). Each publish job is gated by a
  GitHub environment (`pypi-rustamer` / `pypi-tamerlite`) matching the
  corresponding pending publisher on PyPI.
- Creates an immutable GitHub Release `vX.Y.Z` with auto-generated notes
  (from PR titles since the previous tag) and all built wheels attached.

The `dev-release` and `github-release` jobs authenticate with an
installation token from the `tamerlite-releaser` GitHub App (because the
org policy locks `GITHUB_TOKEN` to read-only). Credentials live in two
repo secrets: `RELEASER_APP_ID` and `RELEASER_APP_PRIVATE_KEY`.

## Deeper reference

For architecture (Python/Rust dual implementation, encoder, engine
pipeline), tooling-related conventions, and the rationale behind specific
choices, see [CLAUDE.md](CLAUDE.md). It's primarily written for Claude
Code agents but is useful for any contributor wanting context.
