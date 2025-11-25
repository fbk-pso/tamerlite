# TamerLite

**TamerLite** is a heuristic search-based temporal planner designed to address planning problems with rich temporal dynamics. It is capable of handling *Intermediate Conditions and Effects* (ICE), allowing it to model actions that have requirements and consequences not only at their start or end, but also at intermediate points during their execution. This makes TamerLite suitable for complex real-world scenarios where temporal expressiveness is crucial.


## Installation

TamerLite is not currently available on PyPI and must be installed from source.
It includes a core module written in Rust (under the `rustamer` and `rustamer-base` directories), which must be compiled using [Maturin](https://github.com/PyO3/maturin) before installing the package.

### Prerequisites

Make sure the following tools are installed:

- Python 3.10+
- Rust toolchain (only needed if building the Rust extension from source)
- Maturin (`pip install maturin`, only needed if building from source)

### Installation steps

1. Clone the repository:
   ```bash
   git clone https://github.com/fbk-pso/tamerlite.git
   cd tamerlite
   ```

2. Build and install the Rust extension:
   ```bash
   pip install rustamer/
   ```

3. Install the remaining Python code:
   ```bash
   pip install .
   ```

> **Note:** Precompiled wheels for `rustamer` are available as artifacts from the GitHub Actions CI on the `main` branch. If you download and install the precompiled rustamer wheel manually, you can skip step 2 and proceed directly to step 4. You can find the artifacts in the [Actions tab](https://github.com/fbk-pso/tamerlite/actions) of the GitHub repository.

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
| `search`                   | `Optional[str]`   | The search algorithm to use. Supported values: `"astar"`, `"wastar"`, `"gbfs"`, `"bfs"`, `"dfs"`, `"ehs"`. Default: `"wastar"`. |
| `heuristic`                | `Optional[str]`   | The heuristic function to use. Supported values: `"hff"`, `"hadd"`, `"hmax"`, `"hmax_numeric"`, `"blind"`, `"custom"`. Default: `"hff"`. |
| `weight`                   | `Optional[float]` | A numeric value between 0 and 1 used by weighted search variants like `wastar`. Default: `0.8`. |
| `internal_heuristic_cache` | `Optional[bool]`  | Enables internal caching within the heuristic if set to `True`. Default: `True`. |

---

### `MultiqueueParams`

Defines a multi-queue search strategy composed of multiple `SearchParams`.

| Field     | Type                    | Description                                           |
|-----------|-------------------------|-------------------------------------------------------|
| `queues`  | `List[SearchParams]`    | A list of independent search configurations. The `search` field is ignored in this case. |

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
   SearchParams(heuristic="hmax_numeric", weight=0.5)
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