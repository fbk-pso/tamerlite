# TamerLite

**TamerLite** is a heuristic search-based temporal planner designed to address planning problems with rich temporal dynamics. It is capable of handling *Intermediate Conditions and Effects* (ICE), allowing it to model actions that have requirements and consequences not only at their start or end, but also at intermediate points during their execution. This makes TamerLite suitable for complex real-world scenarios where temporal expressiveness is crucial.


## Installation

TamerLite is not currently available on PyPI and must be installed from source.
It includes a core module written in Rust (under the `rustamer/` directory), which must be compiled using [Maturin](https://github.com/PyO3/maturin) before installing the package.

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

2. Build the Rust extension (wheel format):
   ```bash
   cd rustamer
   maturin build --release
   ```

3. Install the generated wheel (replace `*.whl` with the actual filename):
   ```bash
   pip install target/wheels/*.whl
   cd ..
   ```

4. Install the remaining Python code:
   ```bash
   pip install .
   ```

> **Note:** Precompiled wheels for `rustamer` are available as artifacts from the GitHub Actions CI on the `main` branch. If you download and install the precompiled rustamer wheel manually, you can skip steps 2 and 3 above and proceed directly to step 4. You can find the artifacts in the [Actions tab](https://github.com/fbk-pso/tamerlite/actions) of the GitHub repository.

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

TODO

## References

TamerLite is based on the following research paper:

- Valentini, A., Micheli, A., & Cimatti, A. (2020). *Temporal planning with intermediate conditions and effects.* **AAAI 2020**

## License

TamerLite is released under the GNU Lesser General Public License v3.0 (LGPL-3.0).
See the `LICENSE` file for full details.

## Contact

For questions, bug reports, or contributions, please open an issue on GitHub or contact the authors.