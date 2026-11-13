#!/bin/bash
# Build the anytime-on-main port in its own venv (keeps .pruning_venv untouched).
set -euo pipefail
cd /storage/PSO/alafarciola/pruning_rules/AAAI27/tamerlite-anytime-main
export PATH="$HOME/.cargo/bin:$PATH"
[ -d .venv ] || /usr/bin/python3.13 -m venv .venv
source .venv/bin/activate
pip install -q --upgrade pip maturin
pip install -q "unified-planning @ git+https://github.com/aiplan4eu/unified-planning.git@0be391f84420098249c4cec8fb891f288dc99a7f"
pip install -q -e .
maturin develop --release --manifest-path crates/rustamer/Cargo.toml
python -c "import rustamer, tamerlite.core as c; print(\"rust core:\", c.use_rustamer, hasattr(rustamer, \"wastar_search_enumerate\"))"
