from dataclasses import dataclass
from enum import Enum
from typing import Callable, List, Dict, Tuple, Union, Optional, Set, Iterator
from fractions import Fraction
from collections import defaultdict
import itertools

from tamerlite.core.search_space import Event, SearchSpace, Expression, State, Timing, evaluate
from tamerlite.core.search_space import OperatorNode as Op, split_expression


@dataclass(eq=True, frozen=True)
class Operator:
    action: str
    conditions: Tuple[Expression, ...]
    effects: Tuple[Tuple[str, Union[Expression, bool, int, str]], ...]
    cost: float

class HeuristicKind(Enum):
    HFF = 1
    HADD = 2
    HMAX = 3

class Heuristic:
    def eval(self, state: State, ss: SearchSpace) -> Optional[float]:
        raise NotImplementedError

class CustomHeuristic(Heuristic):
    def __init__(self, callable: Callable[[State], Optional[float]]):
        self.callable = callable

    def eval(self, state: State, ss: SearchSpace) -> Optional[float]:
        return self.callable(state)


def RLRank(state_encoder, model, ModelClass, other_params):
    from tamerlite.rl_heuristics import RLRank
    return RLRank(state_encoder, model, ModelClass, other_params)

def RLHeuristic(state_encoder, model, ModelClass, other_params):
    from tamerlite.rl_heuristics import RLHeuristic
    return RLHeuristic(state_encoder, model, ModelClass, other_params)

def HFF(fluents: Dict[str, str], objects: Dict[str, List[str]],
         events: Dict[str, List[Tuple[Timing, Event]]], goals: Expression, cache_states: bool):
    return DeleteRelaxationHeuristic(fluents, objects, events, goals, HeuristicKind.HFF, cache_states)

def HAdd(fluents: Dict[str, str], objects: Dict[str, List[str]],
         events: Dict[str, List[Tuple[Timing, Event]]], goals: Expression, cache_states: bool):
    return DeleteRelaxationHeuristic(fluents, objects, events, goals, HeuristicKind.HADD, cache_states)

def HMax(fluents: Dict[str, str], objects: Dict[str, List[str]],
         events: Dict[str, List[Tuple[Timing, Event]]], goals: Expression, cache_states: bool):
    return DeleteRelaxationHeuristic(fluents, objects, events, goals, HeuristicKind.HMAX, cache_states)


class _DeleteRelaxationHeuristicBase(Heuristic):
    def __init__(
        self,
        fluents: Dict[str, str],
        objects: Dict[str, List[str]],
        events: Dict[str, List[Tuple[Timing, Event]]],
        goals: Expression,
        cache_states: bool,
        ignore_real_int: bool = False,
    ):
        self._fluents = fluents
        self._objects = objects
        self._events = events
        self._operators: List[Operator] = []
        self._extra_fluents: Dict[str, List[Tuple[str]]] = {}
        self._all_fluents: Set[str] = set(fluents.keys())
        for a, le in events.items():
            self._extra_fluents[a] = []
            cond = (f"__f_{a}_{len(le)-1}", )
            for i, (_, e) in enumerate(le):
                effects = []
                f = f"__f_{a}_{i}"
                self._extra_fluents[a].append((f, ))
                self._all_fluents.add(f)
                effects.append((f, True))
                for eff in e.effects:
                    t = fluents[eff.fluent]
                    if t == "bool":
                        if len(eff.value) == 1 and isinstance(eff.value[0], bool):
                            effects.append((eff.fluent, eff.value[0]))
                        else:
                            effects.append((eff.fluent, True))
                            effects.append((eff.fluent, False))
                    elif not ignore_real_int and (t == "real" or t == "int"):
                        if len(eff.value) == 1:
                            effects.append((eff.fluent, eff.value[0]))
                        else:
                            effects.append((eff.fluent, eff.value))
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
        self._extra_goals: Tuple[Expression, ...] = tuple([fe[-1] for fe in self._extra_fluents.values()])
        self._goals = split_expression(goals)

        self._ordered_fluents = list(self._all_fluents)
        self._cache_states = {} if cache_states else None
        # TODO: remove
        self._fn_eval_counter = 0
        self._cache_state_hits = 0

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


class DeleteRelaxationHeuristic(_DeleteRelaxationHeuristicBase):
    def __init__(
        self,
        fluents: Dict[str, str],
        objects: Dict[str, List[str]],
        events: Dict[str, List[Tuple[Timing, Event]]],
        goals: Expression,
        heuristic_kind: HeuristicKind,
        cache_states: bool,
    ):
        super().__init__(fluents, objects, events, goals, cache_states, ignore_real_int=True)
        self._heuristic_kind = heuristic_kind

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

    def eval(self, state: State, ss: SearchSpace) -> Optional[float]:
        self._fn_eval_counter += 1
        # print(self._fn_eval_counter, self._cache_state_hits)

        if self._cache_states is not None:
            # TODO: the eval initial state is different?
            # TODO: remove this code duplicated
            assignments: Dict[str, Set[Union[bool, int, Fraction, str]]] = {}
            # add state assignments to assignments
            for f, v in state.assignments.items():
                assignments[f] = {v}

            # add extra fluents to assignments
            for action in self._events.keys():
                j, _ = state.todo.get(action, (None, None))
                if j is None:
                    idx = len(self._extra_fluents[action]) - 1
                else:
                    idx = j - 1

                for i, f in enumerate(self._extra_fluents[action]):
                    assignments[f[0]] = {i == idx}

            assignments_values = tuple(list(assignments[f])[0] for f in self._ordered_fluents)
            if assignments_values in self._cache_states:
                self._cache_state_hits += 1
                return self._cache_states[assignments_values]
        
        costs = {}
        lp = []

        for f, v in state.assignments.items():
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
            if self._cache_states is not None:
                self._cache_states[assignments_values] = None
            return None

        if self._heuristic_kind != HeuristicKind.HFF:
            eh = self._cost(self._extra_goals, costs)
            
            if self._heuristic_kind == HeuristicKind.HMAX:
                res = max(h, eh)
            else:
                res = h + eh

            if self._cache_states is not None:
                self._cache_states[assignments_values] = res
            return res

        res = 0
        for a, (j, _) in state.todo.items():
            res += len(self._events[a]) - j

        if h == 0:
            if self._cache_states is not None:
                self._cache_states[assignments_values] = res
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

        if self._cache_states is not None:
            self._cache_states[assignments_values] = res
        return res

    def _cost(self, exp: Tuple[Expression], costs: Dict[Expression, float]) -> Optional[float]:
        if len(exp) == 0:
            return 0
        res = 0
        for g in exp:
            c = costs.get(g, None)
            if c is None:
                return None
            
            if self._heuristic_kind == HeuristicKind.HMAX:
                res = max(res, c)
            else:
                res += c

        return res


class HMaxNumeric(_DeleteRelaxationHeuristicBase):
    def __init__(
        self,
        fluents: Dict[str, str],
        objects: Dict[str, List[str]],
        events: Dict[str, List[Tuple[Timing, Event]]],
        goals: Expression,
        cache_states: bool
    ):
        super().__init__(fluents, objects, events, goals, cache_states, ignore_real_int=False)

        self._operator_conditions_fluents: List[Set[str]] = []
        for operator in self._operators:
            self._operator_conditions_fluents.append(set())
            for cond in operator.conditions:
                for expr_node in cond:
                    if expr_node in self._all_fluents:
                        self._operator_conditions_fluents[-1].add(expr_node)

        self._operator_effects_fluents: List[Set[str]] = []
        for operator in self._operators:
            self._operator_effects_fluents.append(set())
            for fluent, eff in operator.effects:
                if isinstance(eff, str):
                    self._operator_effects_fluents[-1].add(eff)
                elif isinstance(eff, tuple):
                    self._operator_effects_fluents[-1].update(
                        filter(
                            lambda expression_node: isinstance(expression_node, str),
                            eff,
                        )
                    )        

    def _extract_fluents(
        self,
        exp: Expression,
        assignments: Dict[str, Set[Union[bool, int, Fraction, str]]],
        cache_extract_fluents: Dict[int, Set[str]],
    ) -> Set[str]:
        if id(exp) not in cache_extract_fluents:
            cache_extract_fluents[id(exp)] = set(
                filter(lambda expression_node: expression_node in assignments, exp)
            )
        return cache_extract_fluents[id(exp)]

    def _possible_values(
        self,
        exp: Union[Expression, bool, int, Fraction, str],
        assignments: Dict[str, Set[Union[bool, int, Fraction, str]]],
        cache_extract_fluents: Dict[int, Set[str]],
        exp_fluents: Set[str] = None,
    ) -> Iterator[Union[bool, int, Fraction, str]]:
        if isinstance(exp, tuple):
            if exp_fluents is None:
                exp_fluents = self._extract_fluents(
                    exp, assignments, cache_extract_fluents
                )
            values = map(lambda f: assignments[f], exp_fluents)
            for assignments_values in itertools.product(*values):
                assignments_values = dict(zip(exp_fluents, assignments_values))
                state = State(assignments_values, None, None, None, None, None)
                yield evaluate(exp, state)
        else:
            yield exp

    def _exp_can_be_true(
        self,
        exp: Expression,
        assignments: Dict[str, Set[Union[bool, int, Fraction, str]]],
        assignments_changes: Set[str],
        cache_can_be_true: Dict[int, bool],
        cache_extract_fluents: Dict[int, Set[str]],
    ) -> bool:
        exp_fluents = None
        id_exp = id(exp)
        if id_exp in cache_can_be_true:
            if cache_can_be_true[id_exp]:
                return True

            exp_fluents = self._extract_fluents(exp, assignments, cache_extract_fluents)
            if exp_fluents.isdisjoint(assignments_changes):
                return False

        possible_values = self._possible_values(
            exp, assignments, cache_extract_fluents, exp_fluents
        )
        for value in possible_values:
            if value == True:
                cache_can_be_true[id_exp] = True
                return True

        cache_can_be_true[id_exp] = False
        return False

    def _can_be_true(
        self,
        expressions: Tuple[Expression, ...],
        assignments: Dict[str, Set[Union[bool, int, Fraction, str]]],
        assignments_changes: Set[str],
        cache_can_be_true: Dict[int, bool],
        cache_extract_fluents: Dict[int, Set[str]],
    ) -> bool:
        for exp in expressions:
            if not self._exp_can_be_true(
                exp,
                assignments,
                assignments_changes,
                cache_can_be_true,
                cache_extract_fluents,
            ):
                return False
        return True

    def eval(self, state: State, ss: SearchSpace) -> Optional[float]:
        self._fn_eval_counter += 1
        if self._fn_eval_counter % 100 == 0:
            print(self._fn_eval_counter, self._cache_state_hits)

        assignments: Dict[str, Set[Union[bool, int, Fraction, str]]] = {}

        # add state assignments to assignments
        for f, v in state.assignments.items():
            assignments[f] = {v}

        # add extra fluents to assignments
        for action in self._events.keys():
            j, _ = state.todo.get(action, (None, None))
            if j is None:
                idx = len(self._extra_fluents[action]) - 1
            else:
                idx = j - 1

            for i, f in enumerate(self._extra_fluents[action]):
                assignments[f[0]] = {i == idx}

        if self._cache_states is not None:
            assignments_values = tuple(next(iter(assignments[f])) for f in self._ordered_fluents)
            if assignments_values in self._cache_states:
                self._cache_state_hits += 1
                return self._cache_states[assignments_values]
        
        cache_can_be_true: Dict[int, bool] = {}
        cache_extract_fluents: Dict[int, Set[str]] = {}
        applied_operators = [False] * len(self._operators)

        assignments_changes = set(assignments.keys())
        depth = 0
        while len(assignments_changes) > 0:
            if self._can_be_true(
                self._goals + self._extra_goals,
                assignments,
                assignments_changes,
                cache_can_be_true,
                cache_extract_fluents,
            ):
                # goal satisfied
                if self._cache_states is not None:
                    self._cache_states[assignments_values] = depth
                return depth

            new_assignments: Dict[str, Set[Union[bool, int, Fraction, str]]] = (
                defaultdict(set)
            )
            for i, operator in enumerate(self._operators):
                if applied_operators[i]:
                    # operator already applied
                    if assignments_changes.isdisjoint(
                        self._operator_effects_fluents[i]
                    ):
                        # no changes in the effect fluents
                        continue

                elif assignments_changes.isdisjoint(
                    self._operator_conditions_fluents[i]
                ):
                    # operator never applied, but no changes in the condition fluents
                    continue

                elif not self._can_be_true(
                    operator.conditions,
                    assignments,
                    assignments_changes,
                    cache_can_be_true,
                    cache_extract_fluents,
                ):
                    # operator cannot be applied
                    continue

                else:
                    # first time applied
                    applied_operators[i] = True

                for effect in operator.effects:
                    fluent, value = effect
                    possible_values = self._possible_values(
                        value, assignments, cache_extract_fluents
                    )
                    new_assignments[fluent].update(possible_values)

            # update assignments
            assignments_changes = set()
            for fluent, vv in new_assignments.items():
                prev_len = len(assignments[fluent])
                assignments[fluent].update(vv)
                if len(assignments[fluent]) > prev_len:
                    assignments_changes.add(fluent)

            depth += 1

        if self._cache_states is not None:
            self._cache_states[assignments_values] = None
        return None
