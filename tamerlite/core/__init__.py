if False:
    from tamerlite.core.search import wastar_search, astar_search, gbfs_search
    from tamerlite.core.search import bfs_search, dfs_search, ehc_search
    from tamerlite.core.multiqueue import multiqueue_search
    from tamerlite.core.search_space import SearchSpace
    from tamerlite.core.heuristics import HFF, HAdd, CustomHeuristic
    from tamerlite.core.search_space import Timing, Effect, Event
    from tamerlite.core.search_space import Expression, evaluate, get_fluents
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
    from tamerlite.rustamer import wastar_search, astar_search, gbfs_search
    from tamerlite.rustamer import bfs_search, dfs_search, ehc_search
    from tamerlite.rustamer import SearchSpace
    from tamerlite.rustamer import HFF, HAdd, CustomHeuristic
    from tamerlite.rustamer import Timing, Effect, Event
    from tamerlite.rustamer import Expression, evaluate, get_fluents
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

    def multiqueue_search(*args, **kwargs):
        raise NotImplementedError