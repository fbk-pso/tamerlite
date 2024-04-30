from fractions import Fraction
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

def HFF(fluents, objects, events, goals):
    return Heuristic.hff(fluents, objects, events, goals)

def HAdd(fluents, objects, events, goals):
    return Heuristic.hadd(fluents, objects, events, goals)

def RLRank(state_encoder, model, ModelClass, config, sym_h):
    from tamerlite.rl_heuristics import RLRank
    h = RLRank(state_encoder, model, ModelClass, config, sym_h)
    return Heuristic.hrl(state_encoder._general_state_encoder._cse, state_encoder._goals_vec, state_encoder._constants_vec, h.eval_state_vec)

def RLHeuristic(state_encoder, model, ModelClass, config, sym_h):
    from tamerlite.rl_heuristics import RLHeuristic
    h = RLHeuristic(state_encoder, model, ModelClass, config, sym_h)
    return Heuristic.hrl(state_encoder._general_state_encoder._cse, state_encoder._goals_vec, state_encoder._constants_vec, h.eval_state_vec)

def CustomHeuristic(callable):
    return Heuristic.custom(callable)

def get_fluents(exp):
    for e in exp:
        f = e.fluent
        if f is not None:
            yield f

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

from typing import List
Expression = List[ExpressionNode]
