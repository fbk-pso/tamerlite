from dataclasses import dataclass
from typing import Callable, List, Dict, Tuple, Union, Optional, Set

from tamerlite.core.search_space import Event, SearchSpace, Expression, State, Timing, evaluate
from tamerlite.core.search_space import OperatorNode as Op, split_expression


@dataclass(eq=True, frozen=True)
class Operator:
    action: str
    conditions: Tuple[Expression]
    effects: Tuple[Tuple[str, Union[bool, str]]]
    cost: float


class Heuristic:
    def eval(self, state: State, ss: SearchSpace) -> Optional[float]:
        raise NotImplementedError


class CustomHeuristic(Heuristic):
    def __init__(self, callable: Callable[[State], Optional[float]]):
        self.callable = callable

    def eval(self, state: State, ss: SearchSpace) -> Optional[float]:
        return self.callable(state)

def RLRank(encoder, state_encoder, model, ModelClass, config, sym_h):
    from tamerlite.rl_heuristics import RLRank
    return RLRank(encoder, state_encoder, model, ModelClass, config, sym_h)

def RLHeuristic(encoder, state_encoder, model, ModelClass, config, sym_h):
    from tamerlite.rl_heuristics import RLHeuristic
    return RLHeuristic(encoder, state_encoder, model, ModelClass, config, sym_h)

def HAdd(fluents: Dict[str, str], objects: Dict[str, List[str]],
         events: Dict[str, List[Tuple[Timing, Event]]], goals: Expression):
    return HFF(fluents, objects, events, goals, True)


class HFF(Heuristic):
    def __init__(self, fluents: Dict[str, str], objects: Dict[str, List[str]],
                 events: Dict[str, List[Tuple[Timing, Event]]], goals: Expression, return_hadd=False):
        self._fluents = fluents
        self._objects = objects
        self._events = events
        self._return_hadd = return_hadd
        self._operators = []
        self._extra_fluents = {}
        for a, le in events.items():
            self._extra_fluents[a] = []
            cond = (f"__f_{a}_{len(le)-1}", )
            for i, (_, e) in enumerate(le):
                effects = []
                f = f"__f_{a}_{i}"
                self._extra_fluents[a].append((f, ))
                effects.append((f, True))
                for eff in e.effects:
                    t = fluents[eff.fluent]
                    if t == "bool":
                        if len(eff.value) == 1 and isinstance(eff.value[0], bool):
                            effects.append((eff.fluent, eff.value[0]))
                        else:
                            effects.append((eff.fluent, True))
                            effects.append((eff.fluent, False))
                    elif t != "real" and t != "int":
                        if len(eff.value) == 1 and isinstance(eff.value[0], str) and eff.value[0] not in fluents:
                            effects.append((eff.fluent, eff.value[0]))
                        else:
                            for obj in objects[fluents[eff.fluent]]:
                                effects.append((eff.fluent, obj))
                if len(e.conditions) == 0 or e.conditions == (True,):
                    conditions = (cond, )
                else:
                    conditions = split_expression(e.conditions) + (cond, )
                cond = (f, )
                if (False, ) not in conditions:
                    self._operators.append(Operator(a, conditions, tuple(effects), 1))
        self._extra_goals = tuple([fe[-1] for fe in self._extra_fluents.values()])
        self._goals = split_expression(goals)
        self._precondition_of = {}
        self._numeric_conds = set()
        self._empty_pre_operators = []
        for o in self._operators:
            if len(o.conditions) == 0 or o.conditions == ((True,),):
                self._empty_pre_operators.append(o)
            for c in o.conditions:
                if self._is_numeric_condition(c):
                    self._numeric_conds.add(c)
                else:
                    if c not in self._precondition_of:
                        self._precondition_of[c] = []
                    self._precondition_of[c].append(o)
        for c in self._goals:
            if self._is_numeric_condition(c):
                self._numeric_conds.add(c)

    def _is_numeric_condition(self, exp: Expression) -> bool:
        if isinstance(exp[-1], bool): # boolean constant
            return False
        if isinstance(exp[-1], str): # boolean fluent expression
            return False
        if isinstance(exp[-1], Op) and exp[-1].kind == "not": # not of a boolean fluent expression
            i = exp[-1].operands[0]
            if isinstance(exp[i], str):
                return False
        if isinstance(exp[-1], Op) and exp[-1].kind == "==": # equals between a fluent and an object
            i1 = exp[-1].operands[0]
            i2 = exp[-1].operands[1]
            if isinstance(exp[i1], str) and isinstance(exp[i2], str):
                if exp[i1] in self._fluents and self._fluents[exp[i1]] in self._objects and exp[i2] in self._objects[self._fluents[exp[i1]]]:
                    return False
                if exp[i2] in self._fluents and self._fluents[exp[i2]] in self._objects and exp[i1] in self._objects[self._fluents[exp[i2]]]:
                    assert False, "An expression of this form should not be present"
        return True

    def eval(self, state: State, ss: SearchSpace) -> Optional[float]:
        costs = {}
        lp = []

        assignments = state.assignments
        for f, v in assignments.items():
            if v == True:
                k = (f, )
            elif v == False:
                k = (f, Op("not", (0, )))
            else:
                k = (f, v, Op("==", (0, 1)))
            costs[k] = 0
            lp.append(k)

        for x in self._numeric_conds:
            if evaluate(x, state):
                costs[x] = 0
            else:
                costs[x] = 1
            lp.append(x)

        for a in self._events.keys():
            j, _ = state.todo.get(a, (None, None))
            if j is None:
                x = self._extra_fluents[a][-1]
            else:
                x = self._extra_fluents[a][j-1]
            costs[x] = 0
            lp.append(x)

        reached_by = {}
        while len(lp) > 0:
            lo = list(self._empty_pre_operators)
            for p in lp:
                if p in self._precondition_of:
                    lo.extend(self._precondition_of[p])
            lp = []
            new_costs = {}
            for o in set(lo):
                c = self._cost(o.conditions, costs)
                if c is not None:
                    for f, v in o.effects:
                        if v == True:
                            k = (f, )
                        elif v == False:
                            k = (f, Op("not", (0, )))
                        else:
                            k = (f, v, Op("==", (0, 1)))
                        new_cost_k = new_costs.get(k, None)
                        cost_k = costs.get(k, None)
                        if ((new_cost_k is not None and new_cost_k > c + o.cost) or
                            (new_cost_k is None and cost_k is None) or
                            (new_cost_k is None and cost_k > c + o.cost)):
                            reached_by[k] = o
                            new_costs[k] = c + o.cost
                            lp.append(k)
                        elif ((new_cost_k is not None and new_cost_k == c + o.cost) or
                            (new_cost_k is None and cost_k == c + o.cost)) and o.action > reached_by[k].action:
                            reached_by[k] = o

            for k, v in new_costs.items():
                costs[k] = v

        h = self._cost(self._goals, costs)

        if h is None:
            return None

        if self._return_hadd:
            eh = self._cost(self._extra_goals, costs)
            return h + eh

        res = 0
        for a, (j, _) in state.todo.items():
            res += len(self._events[a]) - j

        if h == 0:
            return res

        relaxed_plan = set()
        stack = [g for g in self._goals]
        while len(stack) > 0:
            g = stack.pop()
            o = reached_by.get(g, None)
            if o is None:
                continue
            relaxed_plan.add(o.action)
            if len(o.conditions) > 0:
                for g in o.conditions:
                    stack.append(g)

        for a in relaxed_plan:
            if a not in state.todo:
                res += len(self._events[a])

        return res

    def _cost(self, exp: Tuple[Expression], costs: Dict[Expression, float]) -> Optional[float]:
        if len(exp) == 0:
            return 0
        res = 0
        for g in exp:
            c = costs.get(g, None)
            if c is None:
                return None
            res += c
        return res
