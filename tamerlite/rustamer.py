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

from fractions import Fraction
from typing import List, Union
from rustamer import wastar_search, astar_search, gbfs_search
from rustamer import ehc_search, bfs_search, dfs_search
from rustamer import multiqueue_search
from rustamer import SearchSpace, evaluate as rustevaluate
from rustamer import Timing, Effect, Event, ExpressionNode
from rustamer import (
    make_bool_constant_node,
    make_fluent_node,
    make_int_constant_node,
    make_object_node,
    make_operator_node,
    make_rational_constant_node,
    shift_expression,
    simplify,
)
from rustamer import CoreStateEncoder, Heuristic

def HFF(fluents, objects, events, goals, internal_caching, cache_value_in_state):
    return Heuristic.hff(fluents, objects, events, goals, internal_caching, cache_value_in_state)

def HAdd(fluents, objects, events, goals, internal_caching, cache_value_in_state):
    return Heuristic.hadd(fluents, objects, events, goals, internal_caching, cache_value_in_state)

def HMaxNumeric(fluents, objects, events, goals, internal_caching, cache_value_in_state):
    return Heuristic.hmax_numeric(fluents, objects, events, goals, internal_caching, cache_value_in_state)

def HMax(fluents, objects, events, goals, internal_caching, cache_value_in_state):
    return Heuristic.hmax(fluents, objects, events, goals, internal_caching, cache_value_in_state)

def RLRank(state_encoder, model, ModelClass, other_params, sym_h, cache_value_in_state):
    from tamerlite.rl_heuristics import RLRank
    h = RLRank(state_encoder, model, ModelClass, other_params, sym_h, cache_value_in_state)
    return Heuristic.hrl(h.name, state_encoder._general_state_encoder._cse, state_encoder._goals_vec, state_encoder._constants_vec, h.eval_state_vecs, sym_h, cache_value_in_state)

def RLHeuristic(state_encoder, model, ModelClass, other_params, sym_h, cache_value_in_state):
    from tamerlite.rl_heuristics import RLHeuristic
    h = RLHeuristic(state_encoder, model, ModelClass, other_params, sym_h, cache_value_in_state)
    return Heuristic.hrl(h.name, state_encoder._general_state_encoder._cse, state_encoder._goals_vec, state_encoder._constants_vec, h.eval_state_vecs, sym_h, cache_value_in_state)

def CustomHeuristic(callable, cache_value_in_state):
    return Heuristic.custom(callable, cache_value_in_state)

def get_fluents(exp):
    for e in exp:
        f = e.fluent
        if f is not None:
            yield f

def get_fluent_value(fluent: str, state) -> Union[bool, int, Fraction, str]:
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

def evaluate(exp, state):
    r = rustevaluate(exp, state)
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

Expression = List[ExpressionNode]
