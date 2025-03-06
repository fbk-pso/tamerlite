import os
import sys

use_rustamer = True
if 'DISABLE_RUSTAMER' in os.environ:
    if os.environ['DISABLE_RUSTAMER'].lower() in ("1", "true", "yes"):
        use_rustamer = False
    elif os.environ['DISABLE_RUSTAMER'].lower() in ("0", "false", "no"):
        use_rustamer = True
    else:
        sys.exit("The DISABLE_RUSTAMER environment variable has an invalid value.")

if use_rustamer:
    try:
        import rustamer
    except ImportError:
        use_rustamer = False

if not use_rustamer:
    print("use_rustamer", use_rustamer)

    from tamerlite.core.search import wastar_search, astar_search, gbfs_search
    from tamerlite.core.search import bfs_search, dfs_search, ehc_search
    from tamerlite.core.multiqueue import multiqueue_search
    from tamerlite.core.search_space import SearchSpace
    from tamerlite.core.heuristics import HFF, HAdd, HMax, HMaxNumeric, CustomHeuristic, RLRank, RLHeuristic
    from tamerlite.core.search_space import Timing, Effect, Event
    from tamerlite.core.search_space import Expression, evaluate, get_fluents, simplify
    from tamerlite.core.search_space import (
        make_bool_constant_node,
        make_fluent_node,
        make_int_constant_node,
        make_object_node,
        make_operator_node,
        make_rational_constant_node,
        shift_expression,
    )
    from tamerlite.core.state_encoder import CoreStateEncoder
else:
    print("use_rustamer", use_rustamer)
    
    from tamerlite.rustamer import wastar_search, astar_search, gbfs_search
    from tamerlite.rustamer import bfs_search, dfs_search, ehc_search
    from tamerlite.rustamer import multiqueue_search
    from tamerlite.rustamer import SearchSpace
    from tamerlite.rustamer import HFF, HAdd, HMax, HMaxNumeric, CustomHeuristic, RLRank, RLHeuristic
    from tamerlite.rustamer import Timing, Effect, Event
    from tamerlite.rustamer import Expression, evaluate, get_fluents, simplify
    from tamerlite.rustamer import (
        make_bool_constant_node,
        make_fluent_node,
        make_int_constant_node,
        make_object_node,
        make_operator_node,
        make_rational_constant_node,
        shift_expression,
    )
    from tamerlite.rustamer import CoreStateEncoder
