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

- **Search algorithms**: `wastar_search`, `astar_search`, `gbfs_search`, `bfs_search`, `dfs_search`, `ehc_search`, `multiqueue_search`, `novbfs_search` (and `*_memory_bounded` variants of `wastar_search`, `astar_search`, `gbfs_search` and `novbfs_search`).
- **Heuristics**: `HFF`, `HAdd`, `HMax`, `HMaxExplicit`, `CustomHeuristic`.
- **Data structures**: `SearchSpace`, `State`, `Action`, `Event`, `Effect`, `Timing`, `Expression`, `NumericNovelty`.
- **Id types**: `Fluent`, `Object`, `Action` (see below).

**The three id types are swapped per backend, and their `__hash__` is written
out by hand on both sides on purpose.** A fluent id, an object id and an action
index used to be bare `int`s, mutually substitutable and indistinguishable to a
reader and to mypy alike -- a `dict[int, set[int]]` in `Encoder` could be
action-to-fluents or fluent-to-actions with nothing in the type to say which.
`Fluent`/`Object`/`Action` are frozen dataclasses in
[src/tamerlite/core/search_space.py](src/tamerlite/core/search_space.py) and
`#[pyclass]`es in
[crates/rustamer-base/src/structures.rs](crates/rustamer-base/src/structures.rs);
`core/__init__.py` binds each name to whichever backend is live, so unlike
`FluentDomain` (shared, backend-agnostic data) these follow `IfReturnType`'s
per-backend-swap pattern and the two classes never coexist.

Two invariants hold them together, and both fail silently if broken:

- **`__hash__` must be the bare index on both sides.** `#[pyclass(hash)]`
  derives `__hash__` from `DefaultHasher` (SipHash) over the Rust `Hash` impl,
  which would not match the Python dataclass -- and then a `set[Fluent]` built
  on the Python side of `Encoder` would iterate in a *different order*
  depending on which backend is live. (Only `set`/`frozenset` are exposed:
  `dict` iteration is insertion-ordered, not hash-ordered.) Nothing reads such a
  set in an order-sensitive way today -- the fixpoints in
  `_compute_relevant_fluents`/`_compute_relevant_actions` are closures -- which
  is exactly why this is dangerous: `check_metrics_equality` would stay green
  while the two cores drifted. So the Rust side declares `eq, ord` but *not*
  `hash`, and writes `fn __hash__(&self) -> u64 { self.idx as u64 }` explicitly.
  The hand-written method is mandatory, not an optimization: `eq` without `hash`
  and without it leaves `object`'s identity hash in the slot, i.e. equal values
  hashing differently.
- **`ord` on the pyclass and `order=True` on the dataclass must stay aligned.**
  mypy resolves these names to the *Python* classes (via `core/__init__.pyi`),
  so a missing `ord` lets `sorted(fluents)` type-check and raise only under the
  Rust backend.

`tests/test_engine.py::test_id_types_hash_and_order_agree_across_backends` pins
both, per type, on both backends.

**Known cost.** Wrapping the ids costs the pure-Python core roughly 5% against
bare `int`s, and it is a deliberate trade for the runtime enforcement the
`#[pyclass]` gives at the PyO3 boundary (a bare `int` handed to
`make_fluent_node` raises `TypeError` there). The cost is *not* allocation --
hoisting the per-evaluation `FluentNode` construction was measured and recovers
~1% of it. It is hashing: the keys of `DeleteRelaxationHeuristic`'s
`costs`/`precondition_of` are `Expression` tuples containing
`FluentNode`/`ObjectNode`, so `hash()` on them now recurses into a Python-level
`__hash__` instead of `int`'s C slot -- 3.85M calls on a mid-size logistics
instance, 0.268s of cumulative time against 0.444s. Before optimizing anything
here, profile first and check that number.

**`novbfs_search`/`NumericNovelty`** (partitioned numeric-novelty search,
`search="novbfs_hg"`/`"novbfs_lg"`) is mirrored between
[novelty.py](src/tamerlite/core/novelty.py)/[search.py](src/tamerlite/core/search.py)
and [novelty.rs](crates/rustamer-base/src/novelty.rs)/[search.rs](crates/rustamer-base/src/search.rs),
same invariant as the object-equality rewrite below: the returned novelty
class and hence `expanded_states`/`goal_depth` must agree exactly
(`tests/test_novbfs.py::test_novbfs_cross_backend_parity`, plus
`test_novbfs_metrics_regression`'s pinned YAMLs, run on both backends).
novbfs takes its heuristic like `wastar` does (`TamerLite._get_search`:
the configured `heuristic`, default `hff`, or a custom callable), and uses
its raw value both as the novelty partition (`floor(h)`) and as the
tie-break after novelty; `weight` is ignored with a warning. The heuristic
must never return a negative value (`NumericNovelty.partition_of` asserts it).
Unlike most of this file's cross-backend invariants, the two cores' internal
leaf **numbering** does *not* need to match -- `novelty.rs`'s module
docstring works through why the algorithm's outcome is invariant to leaf-id
permutation. Both cores additionally cache a parent state's own features
(satisfied? how close?) once per *expansion* rather than recomputing per
child -- `NumericNovelty.begin_expansion()`, called once per popped state
before scoring its successors; a state's own `eval()` call still runs once
per generated child, in generation order, matching every other search's
dedup/heuristic-evaluation contract. `NumericNovelty` classifies an `"=="`
subgoal as numeric-vs-object the same way the heuristics below do --
statically, from `FluentDomain`, via the same
`search_space.is_object_typed_operand`/`is_object_typed` (below) rather than
a second, runtime-probe-based classifier.

Rust implementation lives in [crates/rustamer-base/src/](crates/rustamer-base/src/) (core library) and [crates/rustamer/src/](crates/rustamer/src/) (PyO3 bindings).

**`wastar_search`, `wastar_search_memory_bounded`, `novbfs_search` and
`novbfs_search_memory_bounded` share one priority-search driver per core** -- `_priority_search` (`search.py`) /
`priority_search` (`search.rs`) -- rather than four hand-mirrored copies of
the same loop. Named generically rather than "best-first": `wastar_search`
(parameterized by `weight`) already generalizes classical best-first
search/GBFS (`weight=1`) and A* (`weight=0.5`), and `novbfs_search`'s
priority isn't a best-first evaluation function at all, so "best-first"
would misname the thing the driver is meant to generalize over. Each of the
four is a thin wrapper supplying the open list, the dedup store, and
open-list-item construction (plus, for the two novbfs variants, the
`NumericNovelty.begin_expansion` hook); the Rust side expresses this via
small traits (`StatePayload`/`OpenList`/`DedupStore`/`SearchStrategy`), the
idiom `multiqueue.rs`'s `MQSwitchPolicy` already uses. This makes the
cross-backend parity invariant above structural rather than something each
edit has to re-preserve by hand across eight copies -- but two pre-existing,
deliberately-*un*unified quirks are worth knowing before touching the driver:
an `early_termination` successor's goal check sits *before* dedup in Python
and *after* dedup+heuristic-eval in Rust (a pre-existing cross-language
divergence, consistent across all four searches in each language); and
`ehc_search`/`multiqueue_search` are excluded on purpose (`ehc` closes at
*expansion* time and restarts on improvement; multiqueue has its own queue
type and switch-policy abstraction) -- don't fold either into the shared
driver.

**`"=="` covers both numeric and object equality; classification comes from
`FluentDomain`, not from the operands' shape and not from a type name.** UP's
`EQUALS` covers both numeric equality and user-type (object) equality, and both
compile down to the same `"=="` / `ExpressionNode::Equals` node -- there is no
separate operator kind for the two. `DeleteRelaxationHeuristic._is_numeric_leaf_expression`
(`src/tamerlite/core/heuristics.py`) / `is_numeric_leaf_expression`
(`crates/rustamer-base/src/heuristics.rs`) therefore decide per-operand, via
`search_space.is_object_typed_operand`/`is_object_typed`: an operand is
object-typed if it's a literal object, or a fluent whose `FluentDomain` is
the object variant. `is_object_typed_operand` (`src/tamerlite/core/search_space.py`)
is a free function, not a method on `DeleteRelaxationHeuristic` -- it's shared
verbatim with `NumericNovelty` (`src/tamerlite/core/novelty.py`, see above),
which is also why it additionally classifies an interpreted-function operand
(from its declared `return_type`), a case `DeleteRelaxationHeuristic` never
reaches (`_simplify_leaf` bails out on `has_interpreted_function` first).
`is_object_typed` (`crates/rustamer-base/src/heuristics.rs`) is `pub(crate)`
for the identical reason: `novelty.rs` reuses it verbatim, including its own
`InterpretedFunction` arm that `is_numeric_leaf_expression` never exercises.
This replaced an earlier version that only checked for a literal `ObjectNode`
operand -- a fluent compared to *another* fluent of the same object type
(`loc_a == loc_b`) has no such literal, so it was misclassified as numeric,
rewritten into `<=`/`<` pairs, and crashed both backends.

**Why `FluentDomain` and not the type name.** The heuristics used to receive a
`fluent_types: list[str]` plus an `objects: dict[str, list[int]]` keyed by type
name, and re-derive from the name both the fluent's kind and its object domain.
That is not decidable: `Encoder` put builtin type names (`"bool"`/`"int"`/`"real"`)
and user-type names in one string namespace, and `UserType("int")` is legal in
UP. An intermediate fix that classified by name got a *numeric* `n1 == n2` leaf
wrong in exactly that case -- expanding it over the user type's objects and
reporting a solvable problem `UNSOLVABLE` -- while the other core's spelling of
the same predicate got it right, so the backends also disagreed. The same
ambiguity corrupted the effect encoding (an object-valued effect on such a
fluent took the numeric branch). `FluentDomain`
(`src/tamerlite/core/search_space.py`, mirrored as `enum FluentDomain` in
`crates/rustamer-base/src/heuristics.rs`) carries the kind and, for object
fluents, the domain itself; `Encoder.__init__` builds it where it still holds
the UP `Type`, and the name never leaves the encoder. Both cores then resolve a
fluent through one oracle -- `_object_domain`/`object_domain` -- which is also
what the rewrites below use, so "is it object-typed?" and "what does it range
over?" cannot disagree. The four heuristic constructors take `fluent_domains`
in place of the old pair, across the PyO3 boundary too (extracted via
`extract_fluent_domains` in `crates/rustamer-base/src/heuristics.rs`;
`FluentKind`'s member *values* are the wire tag and must match the Rust
discriminants). Unlike `IfReturnType`,
`FluentKind` is never swapped in for a Rust-native type: `FluentDomain` is
shared, backend-agnostic data (`Encoder` builds it once and hands it to
whichever backend is active), and its own `kind`-identity checks
(`__post_init__`, `_object_domain`) always compare against
`search_space.py`'s own `FluentKind` -- a per-backend swap would make those
checks fail for any `FluentDomain` the Rust backend's data reaches, since a
swapped-in Rust value would never be identical to that module's own enum
member. Each core extends the list with a
`Bool` domain per bookkeeping fluent it allocates, so the lookup is total and
no caller needs to know where the real fluents end.

The delete relaxation's cost table only ever holds `fluent == object` facts
(seeded from the state and from operator effects), so a fluent-vs-fluent
object equality has nothing to match once it's correctly classified as
non-numeric -- it must be expanded before it can reach the cost table, exactly,
in both polarities:

- `fluent1 == fluent2` into `OR` over `o` in the intersection of both
  fluents' domains of `(fluent1 == o AND fluent2 == o)`.
- `not(fluent1 == fluent2)` into `OR` over every ordered pair `(o1, o2)` with
  `o1 != o2`, one from each fluent's domain, of `(fluent1 == o1 AND fluent2 == o2)`.

(`_simplify_object_equality`/`simplify_object_equality` in the two
`heuristics.py`/`.rs` files, sibling to the pre-existing `fluent != object`
rewrite.) Domain iteration order (first fluent's domain outer, second's inner,
first fluent's atom before the second's in each conjunct) must match
**exactly** between the two cores -- `hff`'s relaxed-plan extraction breaks
ties in an `OR` by operand order, so a different order changes
`expanded_states`, which `check_metrics_equality` asserts identical between
backends.

Because there's no separate node kind marking object equality, the shape
`fluent1 == fluent2` (or its negation) that `_simplify_object_equality` matches
would equally match a *numeric* fluent-vs-fluent equality. What prevents that:
`_simplify_leaf`/`simplify_leaf`'s dispatch tries the numeric rewrite first and
never falls through when it applies, so a numeric leaf never reaches the
object-equality rule -- this ordering is load-bearing, not an optimization.
The invariant is exercised by `problems_generator.get_problem_object_equality_fluents`
(both polarities, hierarchical types, an always-false sibling-type equality for
the empty-domain-intersection case) via `tests/test_engine.py`'s
cross-backend/cross-heuristic parametrization.

### Problem encoding ([src/tamerlite/encoder.py](src/tamerlite/encoder.py))

`Encoder` bridges Unified Planning and TamerLite's internal search space:

1. Accepts a grounded UP `Problem` and a lifted one (for map-back).
2. Converts UP fluents/actions/conditions/effects into internal `Expression`/`Event`/`Action` via [src/tamerlite/converter.py](src/tamerlite/converter.py) (a `DagWalker` over UP expression trees).
3. Builds the internal `SearchSpace`.
4. Optional preprocessing: symmetry breaking, compression-safe action identification, relevance analysis via HMax reachability.

**Load-bearing invariant: every durative action owns an event at delay 0 from start.**
`SearchSpace._open_action` opens an action when its *first* event fires and evaluates the duration bounds against that state, so `events[0]` must sit exactly at the action's start. UP problems don't guarantee this — an action may only have `at end` conditions/effects, may start with an intermediate `start + delay` event, or (degenerately) may have no conditions and no effects at all and therefore no events. `Encoder._build_events` synthesizes a trivially-true, effect-less event at delay 0 whenever the action doesn't provide one; an action whose timings resolve *before* its own start is rejected outright.

Relatedly, `SearchSpace.__init__` records the duration expression's fluents in the **start event's read set**, so `MutexChecker` orders that event against whatever writes them. Without it, `build_plan` — which, unlike `_expand_event`, only adds ordering edges for mutex/precedence pairs — is free to reschedule the action away from the state its duration was read from, producing plans that UP's own `PlanValidator` rejects. Both halves are needed: the invariant alone gives the ordering nothing to attach to.

**Load-bearing invariant: an `Event`'s `effects` never target the same fluent twice.** `_convert_effects` buckets a UP action's effects per fluent into increase/decrease/assign and folds each bucket into one `Effect` before any `Event` exists, so this always holds for anything the encoder produces — see the `Effect`/`Event` docs in `search_space.py`/`structures.rs`.

**Relevance analysis: compaction, not a filter.** `Encoder._compute_relevant_fluents(actions)` computes the set of fluents that can actually affect search outcome as the least fixpoint of a backward slice: seeded from everything read by a precondition/effect-condition/goal/duration bound of an `actions` action, then closed under "an effect's RHS matters only if the fluent it writes matters" (for every effect `f := expr`, once `f` is relevant every fluent `expr` reads becomes relevant too). The closure is what makes exclusion of an effect's own target fluent from its own right-hand side fall out for free: increase/decrease effects desugar into a self-referencing assignment (`cost := cost + 1`), so a bookkeeping fluent like an accumulated cost counter only gets pulled in when it's already relevant — never from a bare self-reference, and never transitively through another bookkeeping fluent that is itself never seeded (e.g. `log := cost` where neither `log` nor `cost` is read by anything else). `__init__` seeds this with `Encoder.considered_actions` (`relevant_actions` if relevance analysis narrowed it, else `applicable_actions`), which is also the set `_encode`'s pass 2 (below) restricts its *own* conversion to, via `_build_events`/`_build_actions_duration`. **The seed set and the conversion set have to be the same set**, and that coupling is the whole subtlety here: seeding from anything narrower than what pass 2 converts drops a fluent a still-converted precondition reads, and conversion then fails to resolve it — a real bug hit and fixed during development of this section, not a hypothetical; seeding from anything broader (e.g. every action in the grounded problem) keeps fluents nothing converted ever reads, which is just a missed compaction. Restricting pass 2's conversion is what makes the narrow seed sound, and that restriction is itself safe only because a non-considered action is never expanded: the encoder sets `SearchSpace.relevant_actions` to exactly `considered_actions`' content, and both cores reach an action's events only through it (plus `_open_action`/`build_plan`, which see only actions already expanded). `SearchSpace.__init__` checks that invariant in both cores and refuses an encoding that violates it, because violating it is *silent* on the Rust side: `get_successor_state`/`build_plan` look events up with `events.get(...)`, so a missing entry reports the action as inapplicable and drops it from a reconstructed plan instead of raising the way Python's `events[action]` does. Two consequences worth knowing before touching this: after a compacting pass 2, `Encoder.events` no longer holds an entry for every action in the problem, and `Encoder._actions_duration` — the list handed straight to `SearchSpace`, which indexes it by action id — still holds pass-1-numbered, i.e. stale, duration expressions for a non-considered durative action (they can't be reconverted: their bounds may read a dropped fluent, and they can't be nulled either, since `None` there means "not durative" and would flip `is_temporal`). Both are correct only under that same never-expanded argument; evaluating a stale duration would be silently wrong rather than raise, so keep new consumers of either away from non-considered actions.

Gated by `SearchParams.relevant_equality`/`MultiqueueParams.relevant_equality` (default `True`), `Encoder.__init__` runs in two passes: pass 1 encodes every fluent (needed to run `_compute_relevant_actions`'s `HMax` reachability and then `_compute_relevant_fluents` itself over *some* numbering), and if the fixpoint found anything to drop, pass 2 (`Encoder._encode`) rebuilds `_fluents`/`_fluent_ids`/`_fluent_types`/`_converter`/`_actions_duration`/`_events` restricted to the keep-set — a fresh `Converter` each time, since `Converter` is a `DagWalker` that memoizes conversions per `FNode` and can't be reused across a renumbering. `_convert_effects` silently drops an effect whose *target* fluent wasn't kept (nothing reads it, so nothing observes its absence); every fluent a *read* site resolves is guaranteed present by construction of the fixpoint, so a missing id there is a bug, not an expected case — `_convert_effects`/`_convert_fluent` raise (`KeyError`/`self.fluent_ids[...]`) rather than silently falling back to a full encoding. `relevant_equality=False` skips pass 2 entirely and behaves exactly like `main`: every fluent gets a slot.

This means `state.assignments` — for dedup, for heuristic evaluation, for effect application, for goal checking — is *already* restricted to relevant fluents whenever `relevant_equality` is on. There is no separate filter left to apply anywhere:

- **Dedup** just operates on the (possibly compacted) state directly. `WeakEqState` (`core.search` in both cores) no longer carries a `fluents` field — it exists solely to add the `todo` (durative actions in progress) comparison the temporal `weak_equality` dedup path needs on top of the full `assignments` compare/hash; that comparison is a no-op on the classical (`not is_temporal`) path, since `todo` is provably always empty there. The Rust memory-bounded bloom-filter path hashes `state.assignments` directly (no more `DedupKey` wrapper). One behavior change from the old two-consumer design: compaction is no longer gated on `is_temporal`/`weak_equality` the way the old `dedup_relevant_fluents` view was, so a temporal problem solved without `weak_equality` (which gets no dedup at all, before or after this change) now still gets a smaller `state.assignments` and a smaller heuristic cache key — compaction was never actually dedup-specific, that gate was an artifact of dedup being its only consumer at the time.
- **Heuristics** (`HFF`/`HAdd`/`HMax`/`HMaxExplicit`, both cores) no longer take a `relevant_fluents` constructor argument at all. `TamerLite._get_heuristic` (`engine.py`) still builds them over `encoder.considered_actions` rather than the full `applicable_actions`, independent of fluent compaction and unaffected by this section. The per-evaluation restriction the heuristics used to apply on top of an uncompacted state (pruned effects, a projected `internal_caching` key, a restricted fixpoint seed) is gone because there's nothing left to restrict: `fluent_types`/`objects`/`goal`/`events` already describe the compacted encoding, so every heuristic construction path is exactly what it was before relevance analysis existed (`internal_caching` keys on the whole `state.assignments`; `HMaxExplicit`'s fixpoint seed covers every fluent, including the extra bookkeeping ones — this also fixes a pre-existing Python/Rust divergence, where only the Rust core restricted that seed).
- **Custom heuristics** (`StateWrapper.get_value`, `engine.py`) read fluents by name via `Encoder.fluent_ids[str(fluent)]`, which raises `KeyError` for a fluent the problem never defined *or* one compaction dropped as irrelevant — both cases are caught and re-raised as `UPStateMissingFluentError`. A custom heuristic that needs to read a fluent nothing else in the problem reads must be run with `relevant_equality=False`.

Because `tests/test_engine.py::check_metrics_equality` asserts identical `expanded_states`/`goal_depth` between the Python and Rust backends, any divergence between the two cores' compaction — or their `considered_actions` narrowing — fails that check; keep both cores in sync when touching either. One visible consequence of compacting the encoding itself (rather than filtering after the fact, as before): a durative action's events lose the effects that used to write a since-dropped fluent, which can loosen `MutexChecker`/`PrecedenceChecker` ordering on temporal problems and change `expanded_states`/the exact plan produced relative to `main` — still correct (an unread fluent can't gate anything a validator checks), just no longer purely a search-side optimization invisible to metrics.

### Engine ([src/tamerlite/engine.py](src/tamerlite/engine.py))

`TamerLite` implements both `OneshotPlannerMixin` and `AnytimePlannerMixin`.

**Solve pipeline** (`_solve` / `_solve_ground_problem`):
1. Compile the UP problem: remove undefined numeric initials → ground.
2. If all actions are *compression-safe*, further compile temporal → sequential via UP's `TimedToSequential`.
3. Build an `Encoder` from the grounded problem.
4. Run the selected search with the selected heuristic.
5. Reconstruct and map back the plan.

**Anytime** (`_get_solutions_with_params`): iteratively tightens the quality constraint and re-solves until UNSAT or timeout.

**`internal_heuristic_cache` is an upper bound, not a switch.** The heuristic's internal cache key (`assignments` plus the per-action todo index, both cores) is exactly the equivalence relation the search dedups successors on before ever calling `eval_gen` (`State.__eq__`), and that dedup is unconditionally active on a non-temporal problem (`not ss.is_temporal or weak_equality` is true whenever `not ss.is_temporal` is) — so on a classical problem every cache lookup is a guaranteed miss, pure cost with no effect on `expanded_states`/the plan. Both branches of `_solve_ground_problem` (single-queue and multiqueue) therefore gate the flag with `self._params.internal_heuristic_cache and encoder.search_space.is_temporal` before constructing the heuristic(s): `SearchParams.internal_heuristic_cache=True` only actually enables the cache on a temporal problem, regardless of `weak_equality` or which search algorithm is selected. No warning is emitted — the default is `True`, so most classical solves would gate silently and a warning there would be noise.

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
