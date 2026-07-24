# Type stub for `tamerlite.core`'s dual-backend dispatch (PEP 561).
#
# `tamerlite/core/__init__.py` binds each public name at runtime to whichever
# backend is active -- the pure-Python implementation below, or the
# dynamically-`importlib`-loaded `rustamer` extension (see `rustamer.pyi`).
# The checker can't see through either the runtime dispatch or the dynamic
# import, so without this stub every name ends up Unknown for anyone
# importing from `tamerlite.core` and using it in annotation position.
#
# Both backends expose an identical interface (see CLAUDE.md), so this stub
# re-exports the canonical pure-Python definitions as the single type each
# name presents to the checker, regardless of which backend actually runs.
# Keep in sync with `__all__` in `__init__.py` and with `rustamer.pyi`.
#
# `__all__` below is what marks these as re-exports rather than private
# stub-local imports (PEP 484); it must list every name importable from
# `tamerlite.core`.

from fractions import Fraction

from tamerlite.core.heuristics import HFF, CustomHeuristic, HAdd, HMax, HMaxExplicit
from tamerlite.core.multiqueue import multiqueue_search
from tamerlite.core.search import (
    astar_search,
    astar_search_memory_bounded,
    bfs_search,
    dfs_search,
    ehc_search,
    gbfs_search,
    gbfs_search_memory_bounded,
    wastar_search,
    wastar_search_memory_bounded,
)
from tamerlite.core.search_space import (
    Action,
    Effect,
    Event,
    Expression,
    SearchSpace,
    State,
    Timing,
    get_fluents,
    make_bool_constant_node,
    make_fluent_node,
    make_int_constant_node,
    make_object_node,
    make_operator_node,
    make_rational_constant_node,
    shift_expression,
    simplify,
)

# `evaluate`/`get_fluent_value` are NOT literal re-exports at runtime:
# `core/__init__.py` wraps each backend's raw evaluator to unwrap an
# object-valued result down to its plain `int` id at this boundary (both
# backends keep the tagged/`ObjectNode` form internally). Give them their own
# signature here instead of importing the `search_space` originals, whose
# return type still includes `ObjectNode`.
def evaluate(exp: Expression, state: State) -> bool | int | Fraction: ...
def get_fluent_value(fluent: int, state: State) -> bool | int | Fraction: ...

__all__ = [
    "HFF",
    "Action",
    "CustomHeuristic",
    "Effect",
    "Event",
    "Expression",
    "HAdd",
    "HMax",
    "HMaxExplicit",
    "SearchSpace",
    "State",
    "Timing",
    "astar_search",
    "astar_search_memory_bounded",
    "bfs_search",
    "dfs_search",
    "ehc_search",
    "evaluate",
    "gbfs_search",
    "gbfs_search_memory_bounded",
    "get_fluent_value",
    "get_fluents",
    "make_bool_constant_node",
    "make_fluent_node",
    "make_int_constant_node",
    "make_object_node",
    "make_operator_node",
    "make_rational_constant_node",
    "multiqueue_search",
    "shift_expression",
    "simplify",
    "wastar_search",
    "wastar_search_memory_bounded",
]
