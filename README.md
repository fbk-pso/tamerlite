# TamerLite

**TamerLite** is a heuristic search-based temporal planner designed to address planning problems with rich temporal dynamics. It is capable of handling *Intermediate Conditions and Effects* (ICE), allowing it to model actions that have requirements and consequences not only at their start or end, but also at intermediate points during their execution. This makes TamerLite suitable for complex real-world scenarios where temporal expressiveness is crucial.


## Installation

```bash
pip install "tamerlite[rust]"   # pure-Python wrapper + Rust acceleration wheel
# or
pip install tamerlite           # pure-Python only (slower; no Rust toolchain needed)
```

To try the latest unreleased build, install a wheel directly from the rolling
[`dev` pre-release](https://github.com/fbk-pso/tamerlite/releases/tag/dev):

```bash
pip install --pre <url-of-wheel-on-dev-release>
```

### Install from source

```bash
git clone https://github.com/fbk-pso/tamerlite.git
cd tamerlite
pip install ./crates/rustamer   # builds the Rust acceleration wheel (requires a Rust toolchain + maturin)
pip install .                   # installs the tamerlite Python wheel
```

For a contributor setup (uv + just + pre-commit + dev tools), see
[CONTRIBUTING.md](CONTRIBUTING.md) instead.

> **Note:** the Rust acceleration is optional. If `rustamer` is not installed,
> or if you set the environment variable `DISABLE_RUSTAMER=1`, TamerLite
> transparently falls back to a slower pure-Python implementation.

## Usage

TamerLite is fully integrated with the [Unified Planning](https://github.com/aiplan4eu/unified-planning) framework. Register the planner engine with the Unified Planning environment:

```python
from unified_planning.shortcuts import *

env = get_environment()
env.factory.add_engine("tamerlite", "tamerlite.engine", "TamerLite")

problem = ...  # your temporal planning problem

with OneshotPlanner(name="tamerlite") as planner:
    result = planner.solve(problem)
    print(result.plan)
```

## Parameters

TamerLite supports configurable search strategies via structured parameter classes. The main options are `SearchParams` for a single search configuration, and `MultiqueueParams` for combining multiple strategies.

### `SearchParams`

Defines parameters for a single search strategy.

| Field                                    | Type              | Description                                                                                                                                                |
|------------------------------------------|-------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `search`                                 | `Optional[str]`   | Search algorithm to use. Supported values: `"astar"`, `"wastar"`, `"gbfs"`, `"bfs"`, `"dfs"`, `"ehs"`. Default: `"wastar"`.                                |
| `heuristic`                              | `Optional[str]`   | Heuristic used by heuristic search algorithms. Supported values: `"hff"`, `"hadd"`, `"hmax"`, `"hmax_explicit"`, `"blind"`, `"custom"`. Default: `"hff"`.  |
| `weight`                                 | `Optional[float]` | Heuristic weight used by weighted search variants like `wastar`. Must be between 0 and 1. Default: `0.8`.                                                  |
| `internal_heuristic_cache`               | `bool`            | Enable internal caching within the heuristic. Default: `True`.                                                                                             |
| `inadmissible_numeric_heuristic_variant` | `bool`            | Enable the inadmissible numeric variant for `hff`, `hadd`, `hmax` heuristics. Default: `False`.                                                            |
| `early_termination`                      | `bool`            | Stop as soon as a generated successor state satisfies the goal, instead of waiting until the state is selected for expansion. Default: `False`.            |
| `weak_equality`                          | `bool`            | Use weaker state equality on temporal problems. If no plan is found, retry with weak equality disabled. Default: `False`.                                  |
| `symmetry_breaking`                      | `bool`            | Prune equivalent symmetric states during search. Default: `True`.                                                                                          |
| `compression_safe_actions`               | `bool`            | Enable contiguous expansion of compression-safe temporal actions. Default: `True`.                                                                         |
| `relevance_analysis`                     | `bool`            | Filter out actions that cannot contribute to the goal. Default: `True`.                                                                                    |
| `incomplete_memory_bounded_search`       | `bool`            | Use incomplete memory-bounded variants of `"wastar"`, `"astar"`, and `"gbfs"`. Default: `False`.                                                           |

### `MultiqueueParams`

Defines a multi-queue search strategy composed of multiple `SearchParams`.

| Field                                    | Type                    | Description                                                                                                                                                |
|------------------------------------------|-------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `queues`                                 | `List[HeuristicParams]` | A list of independent heuristic configurations.                                                                                                            |
| `internal_heuristic_cache`               | `bool`                  | Enable internal caching within the heuristic. Default: `True`.                                                                                             |
| `inadmissible_numeric_heuristic_variant` | `bool`                  | Enable the inadmissible numeric variant for `hff`, `hadd`, `hmax` heuristics. Default: `False`.                                                            |
| `early_termination`                      | `bool`                  | Stop as soon as a generated successor state satisfies the goal, instead of waiting until the state is selected for expansion. Default: `False`.            |
| `weak_equality`                          | `bool`                  | Use weaker state equality on temporal problems. If no plan is found, retry with weak equality disabled. Default: `False`.                                  |
| `symmetry_breaking`                      | `bool`                  | Prune equivalent symmetric states during search. Default: `True`.                                                                                          |
| `compression_safe_actions`               | `bool`                  | Enable contiguous expansion of compression-safe temporal actions. Default: `True`.                                                                         |
| `relevance_analysis`                     | `bool`                  | Filter out actions that cannot contribute to the goal. Default: `True`.                                                                                    |

### Example

```python
params = SearchParams(
    search="wastar",
    heuristic="hadd",
    weight=0.8,
)

with OneshotPlanner(name="tamerlite", params={"search": params}) as planner:
    result = planner.solve(problem)
    print(result.plan)
```

Or using multiple queues:

```python
multi_params = MultiqueueParams(queues=[
    SearchParams(heuristic="hadd", weight=0.8),
    SearchParams(heuristic="hmax", weight=0.5),
])

with OneshotPlanner(name="tamerlite", params={"search": multi_params}) as planner:
    result = planner.solve(problem)
    print(result.plan)
```

## Tutorial

An end-to-end runnable tutorial covering classical, numeric, temporal and
temporal-numeric planning lives at
[`tamerlite_tutorial.ipynb`](tamerlite_tutorial.ipynb).

## References

TamerLite is based on the following research paper:

- Valentini, A., Micheli, A., & Cimatti, A. (2020). *Temporal planning with intermediate conditions and effects.* **AAAI 2020**

## License

TamerLite is released under the GNU General Public License v3.0 (GPL-3.0).
See the [LICENSE](LICENSE) file for full details.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the contribution workflow, dev
setup, the `just` task runner, and the release flow. All contributors must
sign our
[Contributor License Agreement](https://gist.github.com/alvalentini/a8c5e371be4e7e43b79035c67dc2a1ac)
on their first pull request and agree to follow our
[Code of Conduct](CODE_OF_CONDUCT.md).

## Contact

For questions, bug reports, or contributions, please open an issue on GitHub or
contact the authors at <pso-tools@fbk.eu>.
