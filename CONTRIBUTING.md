# Contributing to TamerLite

Thanks for wanting to contribute to TamerLite. This file covers both the
practical "how do I run X" side of the repo and the community process
(asking questions, reporting bugs, proposing changes, signing the CLA).

**Community process:** [Code of Conduct](#code-of-conduct) ·
[Asking questions](#asking-questions) ·
[Reporting bugs](#reporting-bugs) ·
[Suggesting enhancements](#suggesting-enhancements) ·
[Contribution workflow](#contribution-workflow) ·
[CLA](#contributor-license-agreement-cla)

**[Developer guide](#developer-guide):**
[Setup](#setup) ·
[Common tasks](#common-tasks-via-just) ·
[Test suite](#running-the-test-suite) ·
[Development wheels](#development-wheels)

**[For maintainers](#for-maintainers):**
[Cutting a release](#cutting-a-release) ·
[Dependencies](#keeping-dependencies-fresh)

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md).
By participating, you agree to uphold it. Report unacceptable behaviour
to <pso-tools@fbk.eu>; reports are handled confidentially.

## Asking questions

For usage questions, in order:

1. Check the [README](README.md) and the runnable
   [tutorial notebook](tamerlite_tutorial.ipynb).
2. Search [open and closed issues](https://github.com/fbk-pso/tamerlite/issues?q=is%3Aissue).
3. If still unanswered, open a [new issue](https://github.com/fbk-pso/tamerlite/issues/new)
   with the `question` label. Include your TamerLite version
   (`pip show tamerlite`), Python version, and OS.

## Reporting bugs

Before filing a bug:

- Confirm you're on the latest TamerLite version.
- Search [existing issues](https://github.com/fbk-pso/tamerlite/issues?q=is%3Aissue+label%3Abug)
  to avoid duplicates.

A useful bug report includes:

- A **minimal reproducible example**: the planning problem (PDDL or UP
  Python snippet), the solver parameters, expected vs. actual output.
- The full stack trace, if any.
- Versions: `pip show tamerlite rustamer unified-planning`,
  `python --version`, OS.
- Whether you're running with the Rust extension or with
  `DISABLE_RUSTAMER=1` (the pure-Python fallback).

Reports without a reproduction get labelled `needs-repro` and won't be
actively worked on until reproducible.

## Suggesting enhancements

For new features, search the issues for prior discussion and then open
one yourself, **before writing code**. Describe:

- The use case and *why it benefits the broader project*, not only your
  immediate need.
- A sketch of the proposed API or behaviour.
- Any alternatives you considered.

This lets us catch scope / design issues early and saves you from
writing a PR that needs to be redone.

## Contribution workflow

For non-trivial changes:

1. Open an issue first (or comment on an existing one) to align on
   scope and approach.
2. Fork the repo and create a feature branch off `main`:
   `git checkout -b feat/short-description`.
3. Make focused, well-described commits. Behaviour changes should come
   with new or updated tests.
4. Run `just precommit` locally — it must be green (this is the same
   command CI's lint job runs).
5. Run `just test` with the UP fixtures (see
   [Running the test suite](#running-the-test-suite)) —
   CI runs the full test matrix, so catch failures locally first. If you
   intentionally changed planner output, regenerate the
   pytest-regressions baselines in `tests/test_engine/` with
   `uv run pytest tests/ --force-regen` and commit them.
6. Push to your fork and open a pull request against `main`. Give the PR
   a clear, user-facing title: release notes are auto-generated from PR
   titles, so your title becomes a release-note line verbatim.
7. On your first PR the [CLA assistant](https://cla-assistant.io/) bot
   will ask you to sign the CLA — see the next section.
8. Address review comments.

Typo fixes and tiny doc tweaks can skip the issue step and go straight
to a PR.

## Contributor License Agreement (CLA)

Before your first contribution can be merged, you must sign the
**FBK PSO Unit Individual Contributor License Agreement**:

- [CLA text](https://gist.github.com/alvalentini/a8c5e371be4e7e43b79035c67dc2a1ac)

TamerLite itself is released under the [GPL-3.0](LICENSE); the CLA
defines the licence terms under which you grant FBK the right to use
your contributions across all `fbk-pso` open-source projects.
On your first PR, the [cla-assistant](https://cla-assistant.io/) bot
posts a comment with a sign-in link; you authenticate with GitHub
OAuth and click "I agree". The signature applies to every subsequent
contribution you make to any project under the
[fbk-pso](https://github.com/fbk-pso) organisation.

**Exemptions:** FBK PSO Unit staff (whose contributions are governed
by their employment contracts) and automated accounts (bots) are
whitelisted in the cla-assistant configuration and skip the prompt.

## Developer guide

### Setup

The Rust acceleration (`rustamer`) is **optional** at runtime: when
`rustamer` cannot be imported (or `DISABLE_RUSTAMER=1` is set), TamerLite
falls back to the pure-Python implementations in
[`src/tamerlite/core/`](src/tamerlite/core/). This fallback is load-bearing
and tested.

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

### Common tasks (via `just`)

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

### Running the test suite

Most tests need `unified-planning`'s `up_test_cases/` directory. `uv sync`
installs `unified-planning` from a pinned git commit; clone the same
commit's tree separately for the fixtures:

```bash
sha=$(python3 -c "import tomllib; d=tomllib.load(open('uv.lock','rb')); print(next(p['source']['git'].rsplit('#',1)[1] for p in d['package'] if p['name']=='unified-planning'))")
git clone --filter=blob:none https://github.com/aiplan4eu/unified-planning.git up-checkout
git -C up-checkout checkout "$sha"
PYTHONPATH=up-checkout/up_test_cases just test
```

CI does this automatically. Locally, re-run the `checkout "$sha"` step
whenever an update to `uv.lock` moves the `unified-planning` pin
(Dependabot does this regularly) — a mismatched checkout shows up as
`NameError` collection failures from fixtures referencing symbols absent
in the installed UP version.

Alongside the pytest suite, CI also runs `up_test_cases`' own `report.py`
against the `tamerlite` UP engine. To reproduce it, reuse the `up-checkout` clone
above, register the engine, then run the report:

```bash
echo -e "[engine tamerlite]\nmodule_name: tamerlite.engine\nclass_name: TamerLite" > .up.ini
uv run python up-checkout/up_test_cases/report.py tamerlite
```

See [test.yml](.github/workflows/test.yml) for the exact steps.

### Development wheels

CI builds platform wheels on every push to `main` and publishes them as a
rolling pre-release tagged `dev` under
[Releases](https://github.com/fbk-pso/tamerlite/releases/tag/dev). Wheels
carry PEP 440 dev versions like `0.2.0.dev42+g<sha>`. To pick one up:

```bash
pip install --pre <url-of-wheel-on-dev-release>
```

Useful for smoke-testing main against downstream code without waiting for
a tagged release — e.g. to verify your fix right after your PR merges,
without building locally.

## For maintainers

The rest of this file covers operations that need push or merge rights
on the repo.

### Cutting a release

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

### Keeping dependencies fresh

Dependabot is configured ([.github/dependabot.yml](.github/dependabot.yml))
to open weekly PRs against three ecosystems:

- **uv** (Python deps) — including the git-tracked `unified-planning`
  dependency: a PR lands whenever the upstream `main` moves.
- **cargo** (Rust deps) — patch/minor/major updates for both crates.
- **github-actions** — pinned action versions in workflows, grouped into a
  single PR per week to cut noise.

Just review the PR's diff, let CI run, and merge if green. For
`unified-planning` in particular this is how we stay synchronised with
upstream without manually running `uv lock --upgrade` ourselves.

## Deeper reference

For architecture (Python/Rust dual implementation, encoder, engine
pipeline), tooling-related conventions, and the rationale behind specific
choices, see [CLAUDE.md](CLAUDE.md). It's primarily written for Claude
Code agents but is useful for any contributor wanting context.
