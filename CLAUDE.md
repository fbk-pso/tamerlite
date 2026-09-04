# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick orientation

TamerLite is a heuristic search-based temporal planner ([README.md](README.md)). It ships **two PyPI wheels** that are released in lockstep at the same `X.Y.Z`:

- `tamerlite` — pure-Python wrapper (hatchling), source under `src/tamerlite/`.
- `rustamer` — PyO3 wheel (maturin) providing a Rust acceleration backend; source under `crates/rustamer/`. Depends on the internal Rust crate `crates/rustamer-base/`.

At runtime [src/tamerlite/core/__init__.py](src/tamerlite/core/__init__.py) tries `import rustamer`; on `ImportError` or if `DISABLE_RUSTAMER=true` it falls back to the pure-Python implementations in `src/tamerlite/core/`. **The fallback is load-bearing and tested**: parametrized tests in [tests/test_engine.py](tests/test_engine.py) exercise both code paths.

## Implementation guidance

- Favor efficient implementations: prefer precomputed lookups/O(1) structures over per-call
  scans on hot paths (e.g. state expansion, expression evaluation). Concrete precedent:
  `Converter`'s `int -> up.model.Object` reverse list, built once in `Encoder`, instead of
  `Problem.object(name)`'s linear scan per call.
- The Rust core (`crates/rustamer-base`) can safely assume it runs single-core,
  single-thread: search never spawns threads or releases the GIL (no
  `Python::detach`/`allow_threads` anywhere in the crate). Prefer `RefCell` +
  `thread_local!` over `Mutex`/`RwLock` for module-level mutable state -- a plain
  `static` requires its type to be `Sync`, which `RefCell` isn't, so `thread_local!`
  is what makes that legal without `unsafe`. Concrete precedent: `interpreted_functions.rs`'s
  `INTERPRETED_FUNCTIONS`/`IF_IDS_BY_PTR` interpreted-function callable registry, and its
  `IF_RESULTS` result cache (and `utils.rs`'s `FRACTION_TYPE` type cache) alongside it. If
  this assumption ever changes (e.g. parallel search), anything built on `thread_local!`
  would need revisiting -- it fails silently (each thread gets its own empty state)
  rather than refusing to compile.
- When a `thread_local!`-cached value is a callback into arbitrary Python (e.g.
  `IF_RESULTS` memoizing an interpreted-function call), never hold the `RefCell` borrow
  across the Python call itself. The callable can re-enter Rust (e.g. by calling
  `evaluate`/`simplify` on an expression that itself contains an interpreted-function
  call), and even argument marshalling can trigger arbitrary Python (allocating a
  `PyExpressionNode` can run a GC pass, which can run `__del__`). A borrow held across
  either panics with "already borrowed" the moment that happens. The same hazard applies
  in principle to *dropping* a cached `Py<PyAny>`, not just calling it (dropping can run
  `__del__` too) -- `clear_interpreted_function_cache` (`interpreted_functions.rs`) does drop every
  registered callable while holding `INTERPRETED_FUNCTIONS`'s borrow, and is documented
  there as assuming no registered callable's closure (nor anything it captures) defines a
  finalizer that re-enters this module. Not enforced by this crate.
- `INTERPRETED_FUNCTIONS`' `func_id`s are recycled, not monotonic: `clear_interpreted_function_cache`
  empties the registry back to nothing, and the next registration starts again from index
  0. Pointer dedup (`register_interpreted_function` keying on `Py::as_ptr`) is what makes
  that safe *within* one epoch (a still-registered pointer can't be recycled to a
  different callable, since the registry's strong ref keeps it alive), but a `func_id`
  from a *previous* epoch, if evaluated after the clear, resolves against whatever the new
  epoch registered under that same number -- silently, for an in-range id (only an
  out-of-range one raises, via `get_interpreted_function`'s `PyRuntimeError`). The correctness
  invariant this crate depends on is therefore enforced entirely on the Python side:
  `tamerlite.converter.interpreted_function_scope` reference-counts every solve currently
  in flight (anytime generators for their whole suspended lifetime, not just while
  executing, plus oneshot solves) and only calls `clear_interpreted_function_cache` when
  that count returns to zero -- i.e. when nothing still running could hold a stale
  `func_id`. Do not add a new call site for `clear_interpreted_function_cache` (or a new
  path that registers interpreted functions) without routing it through that scope.

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

### Known gotcha: `uv`'s own cache can serve a stale `rustamer` wheel

`rustamer` is a `uv` workspace member with `source = { editable = ... }` in `uv.lock`. Plain `uv run`/`uv sync` (no `--no-sync`) silently re-syncs the environment first and, when it decides the editable `rustamer` package needs rebuilding, builds it through **its own** PEP 517 path — not `maturin develop`. That path has been observed to serve a wheel from `uv`'s persistent build cache (`~/.cache/uv/archive-v0/...`) that predates the current Rust source, instead of rebuilding fresh. The symptom is a confusing runtime `TypeError`/`AttributeError` from a `rustamer`-exported function that looks like a genuine Rust bug but disappears on a from-scratch build — e.g. a PyO3 argument-extraction error mentioning an old parameter name/type that no longer exists in the current source (concretely hit after the `10ce304` refactor: `make_object_node(oid: usize)` calls raised `TypeError: 'int' object is not an instance of 'str', while processing 'name'`, because the cached wheel still had the pre-refactor `make_object_node(name: String)`).

Every dev-loop recipe (`test`, `lint`, `format`, `typecheck`, `precommit`, `check-installed-versions`) therefore passes `--no-sync`, so anything invoked through `just` leaves a `just build-rust` build alone. **`just install` is the single syncing entry point** — run it after changing dependencies, then `just build-rust`. The trap is still live for bare `uv run ...` typed by hand.

**Before concluding there's a Rust logic bug**, rule this out:
1. Rebuild explicitly with `just build-rust` (`maturin develop --release`) and retest — this bypasses `uv`'s own build path entirely.
2. If the bug persists after that but reappears whenever a bare `uv run`/`uv sync` executes, purge the cache: `uv cache clean rustamer` (or `uv cache clean` for everything) and rebuild.
3. Prefer `uv run --no-sync ...` for ad hoc commands during Rust dev iteration, right after `just build-rust`, to avoid uv silently reinstalling over your local build.
4. Symptoms are not limited to import/signature errors: a stale wheel can also produce *plausible but wrong results* on only one backend. A test that passes under `DISABLE_RUSTAMER=true` and fails without it — or vice versa — after you have edited `crates/` is a stale-wheel signal, not necessarily a real backend divergence.

## Common tasks (all via `just`)

| Recipe | What it does |
|---|---|
| `just install` | `uv sync --all-extras` + `just build-rust` — **the only recipe that syncs** |
| `just build-rust` | `maturin develop` — rebuild rustamer in-place into the venv (dev iteration) |
| `just up-checkout` | Clone or re-pin `./up-checkout` to the `unified-planning` commit `uv.lock` resolves |
| `just build` | Produce both `tamerlite` and `rustamer` wheels + sdists into `./dist/` |
| `just build-python` | Only the `tamerlite` wheel (used by CI's `tamerlite` publish job) |
| `just build-rust-wheel` | Only the `rustamer` wheel + sdist for current interpreter |
| `just test` | `uv run --no-sync pytest tests/ -n auto` — runs in parallel via pytest-xdist; set `PYTHONPATH=up-checkout/up_test_cases` if you need the UP fixtures |
| `just lint` | Ruff (check + format --check) + cargo fmt --check + cargo clippy (informational, -W warnings) |
| `just format` | Ruff format + ruff --fix + cargo fmt --all |
| `just typecheck` | `uv run mypy` (config in `pyproject.toml`, scope `src/tamerlite` + `tests`) |
| `just precommit` | `pre-commit run --all-files --show-diff-on-failure` — same as CI's `lint` job |
| `just check-versions` | Verify pyproject + Cargo agree on base `X.Y.Z` |
| `just bump VERSION` | Update version in pyproject + Cargo + rustamer pin; refresh `uv.lock` |
| `just clean` | Remove build/dist/target/cache dirs |

## Running the test suite (with UP fixtures)

Most tests need `unified-planning`'s `up_test_cases/` directory. `uv sync` installs `unified-planning` from a pinned git commit; the test fixtures live in the same repo and must be cloned and **checked out to the locked commit** (CI does this automatically in [test.yml](.github/workflows/test.yml)):

```bash
just up-checkout          # clones, or re-pins an existing checkout, to the locked commit
PYTHONPATH=up-checkout/up_test_cases just test
```

Re-run `just up-checkout` whenever `uv.lock` changes — an existing checkout does **not** move on its own. Mismatched commits cause `NameError` collection failures from newer TAMP fixtures referencing symbols absent in the installed UP version, and can also make otherwise-healthy regression tests (e.g. `test_heuristic_values`) fail with diffs that have nothing to do with your change.

### Cap memory when running tests

The full suite (and some individual heavy cases -- unbounded `timeout=None` searches, `-n auto` parallelism multiplying peak RSS across workers) can exhaust host RAM and take down the whole environment. Always run pytest under `runlim` with a memory cap, e.g.:

```bash
runlim --space-limit=4096 uv run --no-sync pytest tests/ -n auto
```

`runlim` does **not** accept a `--` separator before the wrapped command (`runlim: invalid option '--'`) -- pass the command directly after the options.

Adjust `--space-limit` (MB) down for a single test file/case, and prefer targeting specific tests (`-k`) over the full suite when iterating. `runlim` also accepts `--time-limit=<seconds>` (`-t`) if a run risks hanging instead of just ballooning memory.

**Known flakiness**: in some sandboxed dev-container setups, `runlim`'s own `execvp` of `uv` intermittently fails (`[runlim] status: execvp failed`, exits immediately with no child spawned) for no apparent reason tied to the command's arguments -- the identical invocation can fail several times in a row and then succeed. This is unrelated to the wrapped command itself (a bare, unwrapped run of the same command is reliable). If you hit this, retry the same `runlim` invocation a few times (a short sleep between attempts helps) rather than assuming the test setup is broken.

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

**Load-bearing invariant: every durative action owns an event at delay 0 from start.**
`SearchSpace._open_action` opens an action when its *first* event fires and evaluates the duration bounds against that state, so `events[0]` must sit exactly at the action's start. UP problems don't guarantee this — an action may only have `at end` conditions/effects, may start with an intermediate `start + delay` event, or (degenerately) may have no conditions and no effects at all and therefore no events. `Encoder._build_events` synthesizes a trivially-true, effect-less event at delay 0 whenever the action doesn't provide one; an action whose timings resolve *before* its own start is rejected outright.

Relatedly, `SearchSpace.__init__` records the duration expression's fluents in the **start event's read set**, so `MutexChecker` orders that event against whatever writes them. Without it, `build_plan` — which, unlike `_expand_event`, only adds ordering edges for mutex/precedence pairs — is free to reschedule the action away from the state its duration was read from, producing plans that UP's own `PlanValidator` rejects. Both halves are needed: the invariant alone gives the ordering nothing to attach to.

**Relevance-restricted duplicate-state detection.** `Encoder._compute_dedup_relevant_fluents` computes the set of fluents that can actually affect search outcome as the least fixpoint of a backward slice: seeded from everything read by a precondition/effect-condition/goal/duration bound, then closed under "an effect's RHS matters only if the fluent it writes matters" (for every effect `f := expr`, once `f` is relevant every fluent `expr` reads becomes relevant too). The closure is what makes the old special-cased exclusion of an effect's own target fluent from its own right-hand side unnecessary: increase/decrease effects desugar into a self-referencing assignment (`cost := cost + 1`), so a bookkeeping fluent like an accumulated cost counter only gets pulled in when it's already relevant — never from a bare self-reference, and never transitively through another bookkeeping fluent that is itself never seeded (e.g. `log := cost` where neither `log` nor `cost` is read by anything else). The set is installed on `SearchSpace.dedup_relevant_fluents`. Every search algorithm in both cores keys its visited-state set/bloom filter on this reduced set instead of every fluent, restricting only the *dedup key*: the full state used for heuristic evaluation, effect application, and goal checking is untouched, so this can only make dedup recognize more states as duplicates, never change what an action does or which actions are applicable. It feeds both the classical (`not is_temporal`) dedup path and the temporal `weak_equality` dedup path through a single wrapper type (`WeakEqState` in both cores, plus `DedupKey` in Rust for the memory-bounded bloom-filter path) — one type suffices for the cases that need it because `WeakEqState`'s extra `todo` (durative actions in progress) comparison is a no-op on the classical path: `todo` is provably always empty there (the initial state seeds it empty, and the only insertion is gated on the problem being temporal), so it doesn't need a *separate* `todo`-blind type. Rust always keys its visited-state sets on `WeakEqState` (`fluents=None` in the common case); a measured pass over `State`'s actual layout showed the wrapper's extra `fluents: Option<&[usize]>` field costs a few bytes per hash-set bucket against the ~3 KB a retained state already costs (dominated by `im::Vector`'s whole-chunk copy-on-write), so a second `Rc<State>`-keyed set to avoid it wasn't worth the duplicated branches across three search algorithms. Python keeps a bare-`State` fallback in `state_representation` for the `fluents=None, weak_equality=False` case instead, since there a `WeakEqState` is a real heap-allocated `dataclass` object (not a zero-cost Rust struct), so skipping the allocation is a genuine win, not a rounding error. `is_temporal and not weak_equality` has no dedup at all (by design — that regime exists to stay complete) and is unaffected. Because `tests/test_engine.py::check_metrics_equality` asserts identical `expanded_states`/`goal_depth` between the Python and Rust backends, any divergence between the two dedup implementations here fails that check — keep both cores in sync when touching this. Gated by `SearchParams.relevant_equality`/`MultiqueueParams.relevant_equality` (default `True`); turning it off leaves `Encoder._dedup_relevant_fluents`/`SearchSpace.dedup_relevant_fluents` at `None`, falling back to the pre-existing full-assignments dedup key. `Encoder` also skips computing the reduction whenever it's known nothing will consult it -- `is_temporal and not weak_equality` (`Encoder`'s own `weak_equality` constructor param, threaded from `SearchParams.weak_equality`/`MultiqueueParams.weak_equality`) -- since that's exactly the regime with no dedup at all described above; this is a pure perf optimization, not a third on/off switch, and produces the same `None` result `relevant_equality=True` would have reached anyway once search ran, just without doing the (cheap, but pointless there) analysis.

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
- Ruff is scoped to `src tests ci crates`.
- `clippy` runs with `-D warnings` in [justfile](justfile) (enforced via `.pre-commit-config.yaml`'s `just lint` hook) — the backlog that once kept it informational-only has been cleared.
