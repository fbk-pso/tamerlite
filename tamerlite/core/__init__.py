# Copyright (C) 2025 PSO Unit, Fondazione Bruno Kessler
# This file is part of TamerLite.
#
# TamerLite is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# TamerLite is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#

import os
import sys
import warnings
import importlib

use_rustamer = True
if "DISABLE_RUSTAMER" in os.environ:
    if os.environ["DISABLE_RUSTAMER"].lower() in ("1", "true", "yes"):
        use_rustamer = False
    elif os.environ["DISABLE_RUSTAMER"].lower() in ("0", "false", "no"):
        use_rustamer = True
    else:
        sys.exit("The DISABLE_RUSTAMER environment variable has an invalid value.")

if use_rustamer:
    try:
        libname = os.environ.get("RUSTAMER_LIB", "rustamer")
        rustamer_lib = importlib.import_module(libname)
    except ImportError:
        use_rustamer = False

if not use_rustamer:
    warnings.warn(
        "Tamerlite is using the Python core implementation instead of the Rust one. "
        "For better performance, ensure rustamer is installed and not disabled."
    )
    from tamerlite.core.search import (
        wastar_search,
        wastar_search_memory_bounded,
        astar_search,
        astar_search_memory_bounded,
        gbfs_search,
        gbfs_search_memory_bounded,
    )
    from tamerlite.core.search import bfs_search, dfs_search, ehc_search
    from tamerlite.core.multiqueue import multiqueue_search
    from tamerlite.core.search_space import SearchSpace, get_fluent_value
    from tamerlite.core.heuristics import (
        HFF,
        HAdd,
        HMax,
        HMaxExplicit,
        CustomHeuristic,
    )
    from tamerlite.core.search_space import Timing, Effect, Event, Action
    from tamerlite.core.search_space import Expression, evaluate, simplify
    from tamerlite.core.search_space import (
        make_bool_constant_node,
        make_fluent_node,
        make_int_constant_node,
        make_object_node,
        make_operator_node,
        make_rational_constant_node,
        shift_expression,
        get_fluents,
    )
else:
    from fractions import Fraction
    from typing import List, Union, Iterator

    (
        wastar_search,
        wastar_search_memory_bounded,
        astar_search,
        astar_search_memory_bounded,
        gbfs_search,
        gbfs_search_memory_bounded,
    ) = (
        rustamer_lib.wastar_search,
        rustamer_lib.wastar_search_memory_bounded,
        rustamer_lib.astar_search,
        rustamer_lib.astar_search_memory_bounded,
        rustamer_lib.gbfs_search,
        rustamer_lib.gbfs_search_memory_bounded,
    )
    ehc_search, bfs_search, dfs_search = (
        rustamer_lib.ehc_search,
        rustamer_lib.bfs_search,
        rustamer_lib.dfs_search,
    )
    multiqueue_search = rustamer_lib.multiqueue_search
    SearchSpace, Timing, Effect, Event, Action = (
        rustamer_lib.SearchSpace,
        rustamer_lib.Timing,
        rustamer_lib.Effect,
        rustamer_lib.Event,
        rustamer_lib.Action,
    )
    Expression = List[rustamer_lib.ExpressionNode]
    State = rustamer_lib.State

    (
        make_bool_constant_node,
        make_fluent_node,
        make_int_constant_node,
        make_object_node,
        make_operator_node,
        make_rational_constant_node,
        shift_expression,
        simplify,
        get_fluents,
    ) = (
        rustamer_lib.make_bool_constant_node,
        rustamer_lib.make_fluent_node,
        rustamer_lib.make_int_constant_node,
        rustamer_lib.make_object_node,
        rustamer_lib.make_operator_node,
        rustamer_lib.make_rational_constant_node,
        rustamer_lib.shift_expression,
        rustamer_lib.simplify,
        rustamer_lib.get_fluents,
    )

    (
        HFF,
        HAdd,
        HMax,
        HMaxExplicit,
        CustomHeuristic,
    ) = (
        rustamer_lib.Heuristic.hff,
        rustamer_lib.Heuristic.hadd,
        rustamer_lib.Heuristic.hmax,
        rustamer_lib.Heuristic.hmax_explicit,
        rustamer_lib.Heuristic.custom,
    )

    def get_fluent_value(fluent: int, state: State) -> Union[bool, int, Fraction, str]:
        exp = state.get_py_value(fluent)
        if exp.bool_constant is not None:
            return exp.bool_constant
        elif exp.object is not None:
            return exp.object
        elif exp.int_constant is not None:
            return exp.int_constant
        elif exp.real_constant is not None:
            n, d = exp.real_constant
            return Fraction(n, d)
        else:
            raise NotImplementedError("Unreachable code")

    def evaluate(exp: Expression, state: State) -> Union[bool, int, Fraction, str]:
        r = rustamer_lib.evaluate(exp, state)
        if r.bool_constant is not None:
            return r.bool_constant
        elif r.object is not None:
            return r.object
        elif r.int_constant is not None:
            return r.int_constant
        elif r.real_constant is not None:
            n, d = r.real_constant
            return Fraction(n, d)
        else:
            raise NotImplementedError("Unreachable code")
