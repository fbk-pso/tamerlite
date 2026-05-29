# TamerLite

**TamerLite** is a heuristic search-based temporal planner designed to address planning problems with rich temporal dynamics. It is capable of handling *Intermediate Conditions and Effects* (ICE), allowing it to model actions that have requirements and consequences not only at their start or end, but also at intermediate points during their execution. This makes TamerLite suitable for complex real-world scenarios where temporal expressiveness is crucial.


## Installation

TamerLite is not currently available on PyPI and must be installed from source.
It includes a core module written in Rust (under the `rustamer` and `rustamer-base` directories), which is compiled using [Maturin](https://github.com/PyO3/maturin).

### Prerequisites

- Python 3.10+
- Rust toolchain (only needed if building the Rust extension from source)
- [uv](https://github.com/astral-sh/uv) — Python project & environment manager (recommended)
- [just](https://github.com/casey/just) — command runner (optional, for the shortcuts below)

### Quick start (recommended, with `uv`)

```bash
git clone https://github.com/fbk-pso/tamerlite.git
cd tamerlite
uv sync --all-extras    # creates .venv, builds rustamer workspace member, installs dev deps
uv run pre-commit install
```

Common tasks via [`just`](justfile):

```bash
just              # list recipes
just install      # uv sync --all-extras
just build-rust   # rebuild the rustamer Rust extension
just test         # pytest
just lint         # ruff check + ruff format --check
just format       # ruff format + ruff check --fix
just typecheck    # mypy
just precommit    # run all pre-commit hooks
just build        # build sdist + wheel for tamerlite
```

### Manual installation (pip)

If you prefer pip:

```bash
pip install crates/rustamer   # build & install the Rust extension
pip install .                 # install tamerlite (uses src/ layout)
```

The Rust sources live under [`crates/`](crates/) as a Cargo workspace (root `Cargo.toml`).
Run `cargo build --workspace` to build both crates with a single `target/`.

### Development tooling

This repository is configured with:

- **uv** workspace (root + `rustamer/`) — single `uv.lock` reproduces both Python and Rust dependencies.
- **ruff** for linting and formatting (`pyproject.toml` → `[tool.ruff]`).
- **mypy** for static type checking (`pyproject.toml` → `[tool.mypy]`).
- **pre-commit** hooks (`.pre-commit-config.yaml`) — enforced in CI.
- **just** task runner (`justfile`).

## Usage

TamerLite is fully integrated with the [Unified Planning](https://github.com/aiplan4eu/unified-planning) framework. You must register the planner engines with the Unified Planning environment:

```python
from unified_planning.shortcuts import *

# Register TamerLite engine
env = get_environment()
env.factory.add_engine("tamerlite", "tamerlite.engine", "TamerLite")

# Define your temporal planning problem
problem = ...

# Solve with TamerLite
with OneshotPlanner(name="tamerlite") as planner:
   result = planner.solve(problem)
   print(result.plan)
```

## Parameters

TamerLite supports configurable search strategies via structured parameter classes.
The main options are `SearchParams` for defining a single search configuration, and `MultiqueueParams` for combining multiple strategies.

### `SearchParams`

Defines parameters for a single search strategy.

| Field                      | Type              | Description                                                                 |
|----------------------------|-------------------|-----------------------------------------------------------------------------|

| Field                      | Type              | Description                                                                 |
|----------------------------|-------------------|-----------------------------------------------------------------------------|
| `search`                   | `Optional[str]`   | Search algorithm to use. Supported values: `"astar"`, `"wastar"`, `"gbfs"`, `"bfs"`, `"dfs"`, `"ehs"`. Default: `"wastar"`. |
| `heuristic`                | `Optional[str]`   | Heuristic used by heuristic search algorithms. Supported values: `"hff"`, `"hadd"`, `"hmax"`, `"hmax_explicit"`, `"blind"`, `"custom"`. Default: `"hff"`. |
| `weight`                   | `Optional[float]` | Heuristic weight used by weighted search variants like `wastar`. Must be between 0 and 1. Default: `0.8`. |
| `internal_heuristic_cache` | `bool`            | Enable internal caching within the heuristic. Default: `True`. |
| `inadmissible_numeric_heuristic_variant` | `bool` | Enable the inadmissible numeric variant for `hff`, `hadd`, `hmax` heuristics. Default: `False`. |
| `early_termination`        | `bool`            | Stop as soon as a generated successor state satisfies the goal, instead of waiting until the state is selected for expansion. Default: `False`. |
| `weak_equality`            | `bool`            | Use weaker state equality on temporal problems. If no plan is found, retry with weak equality disabled. Default: `False`. |
| `symmetry_breaking`        | `bool`            | Prune equivalent symmetric states during search. Default: `True`. |
| `compression_safe_actions` | `bool`            | Enable contiguous expansion of compression-safe temporal actions. Default: `True`. |
| `relevance_analysis`       | `bool`            | Filter out actions that cannot contribute to the goal. Default: `True`. |
| `incomplete_memory_bounded_search` | `bool`    | Use incomplete memory-bounded variants of `"wastar"`, `"astar"`, and `"gbfs"`. Default: `False`. |

---

### `MultiqueueParams`

Defines a multi-queue search strategy composed of multiple `SearchParams`.

| Field                      | Type                    | Description                                           |
|----------------------------|-------------------------|-------------------------------------------------------|
| `queues`                   | `List[HeuristicParams]` | A list of independent heuristic configurations. |
| `internal_heuristic_cache` | `bool`            | Enable internal caching within the heuristic. Default: `True`. |
| `inadmissible_numeric_heuristic_variant` | `bool` | Enable the inadmissible numeric variant for `hff`, `hadd`, `hmax` heuristics. Default: `False`. |
| `early_termination`        | `bool`            | Stop as soon as a generated successor state satisfies the goal, instead of waiting until the state is selected for expansion. Default: `False`. |
| `weak_equality`            | `bool`            | Use weaker state equality on temporal problems. If no plan is found, retry with weak equality disabled. Default: `False`. |
| `symmetry_breaking`        | `bool`            | Prune equivalent symmetric states during search. Default: `True`. |
| `compression_safe_actions` | `bool`            | Enable contiguous expansion of compression-safe temporal actions. Default: `True`. |
| `relevance_analysis`       | `bool`            | Filter out actions that cannot contribute to the goal. Default: `True`. |

---

### Example

```python
params = SearchParams(
   search="wastar",
   heuristic="hadd",
   weight=0.8
)

with OneshotPlanner(name="tamerlite", params={"search": params}) as planner:
   result = planner.solve(problem)
   print(result.plan)
```

Or using multiple queues:

```python
multi_params = MultiqueueParams(queues=[
   SearchParams(heuristic="hadd", weight=0.8),
   SearchParams(heuristic="hmax", weight=0.5)
])

with OneshotPlanner(name="tamerlite", params={"search": multi_params}) as planner:
   result = planner.solve(problem)
   print(result.plan)
```

## References

TamerLite is based on the following research paper:

- Valentini, A., Micheli, A., & Cimatti, A. (2020). *Temporal planning with intermediate conditions and effects.* **AAAI 2020**

## License

TamerLite is released under the GNU General Public License v3.0 (GPL-3.0).
See the `LICENSE` file for full details.

## Contact

For questions, bug reports, or contributions, please open an issue on GitHub or contact the authors.
