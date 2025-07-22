# Copyright (C) 2025 PSO Unit, Fondazione Bruno Kessler
# This file is part of TamerLite.
#
# TamerLite is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# TamerLite is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#

import os
import sys
from warnings import warn

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
    warn("Using the pure Python implementation of TamerLite.")
    from tamerlite.core.search import wastar_search, astar_search, gbfs_search
    from tamerlite.core.search import bfs_search, dfs_search, ehc_search
    from tamerlite.core.multiqueue import multiqueue_search, entropy_dual_queue_search
    from tamerlite.core.search_space import SearchSpace, get_fluent_value
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
    from tamerlite.rustamer import wastar_search, astar_search, gbfs_search
    from tamerlite.rustamer import bfs_search, dfs_search, ehc_search
    from tamerlite.rustamer import multiqueue_search, entropy_dual_queue_search
    from tamerlite.rustamer import SearchSpace, get_fluent_value
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
