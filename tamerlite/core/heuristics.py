from dataclasses import dataclass
from typing import Callable, List, Dict, Tuple, Union, Optional, Set, Iterable
from fractions import Fraction
from collections import defaultdict
import itertools

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

def RLRank(state_encoder, model, ModelClass, other_params):
    from tamerlite.rl_heuristics import RLRank
    return RLRank(state_encoder, model, ModelClass, other_params)

def RLHeuristic(state_encoder, model, ModelClass, other_params):
    from tamerlite.rl_heuristics import RLHeuristic
    return RLHeuristic(state_encoder, model, ModelClass, other_params)

def HAdd(fluents: Dict[str, str], objects: Dict[str, List[str]],
         events: Dict[str, List[Tuple[Timing, Event]]], goals: Expression):
    return HFF(fluents, objects, events, goals, True)


class DeleteRelaxationHeuristic(Heuristic):
    def __init__(
        self,
        fluents: Dict[str, str],
        objects: Dict[str, List[str]],
        events: Dict[str, List[Tuple[Timing, Event]]],
        goals: Expression,
        ignore_real_int=False,
    ):
        self._fluents = fluents
        self._objects = objects
        self._events = events
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


class HFF(DeleteRelaxationHeuristic):
    def __init__(
        self,
        fluents: Dict[str, str],
        objects: Dict[str, List[str]],
        events: Dict[str, List[Tuple[Timing, Event]]],
        goals: Expression,
        return_hadd=False,
    ):
        super().__init__(fluents, objects, events, goals, ignore_real_int=True)
        self._return_hadd = return_hadd

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
            # TODO: remove
            # return max(h, eh)

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

            # TODO: remove
            # h_max heuristic
            # res = max(res, c)
        return res


class HMax(DeleteRelaxationHeuristic):
    def __init__(
        self,
        fluents: Dict[str, str],
        objects: Dict[str, List[str]],
        events: Dict[str, List[Tuple[Timing, Event]]],
        goals: Expression,
    ):
        super().__init__(fluents, objects, events, goals, ignore_real_int=False)

        self.assignments: Dict[str, Set[Union[bool, int, Fraction, str]]] = {}
        self.assignments_changes: Dict[str, Set[Union[bool, int, Fraction, str]]] = {}
        self.cache_can_be_true: Dict[int, bool] = {}

    # @cache
    def _extract_fluents(
        self,
        exp: Expression,
    ) -> Iterable:
        # return list(filter(lambda expression_node: expression_node in self.assignments, exp))
        return filter(lambda expression_node: expression_node in self.assignments, exp)

    def _possible_values(
        self,
        exp,
        assignments_changes: Set = None,
        exp_fluents=None,
    ):
        if isinstance(exp, tuple):
            if exp_fluents is None:
                exp_fluents = self._extract_fluents(exp)
            values = map(lambda f: self.assignments[f], exp_fluents)
            for state_assignments in itertools.product(*values):
                state_assignments = dict(zip(exp_fluents, state_assignments))
                state = State(state_assignments, None, None, None, None, None)
                yield evaluate(exp, state)
        else:
            yield exp

    def _exp_can_be_true(
        self,
        exp: Expression,
        assignments_changes: Set,
    ) -> bool:

        exp_fluents = None
        id_exp = id(exp)
        if id_exp in self.cache_can_be_true:
            if self.cache_can_be_true[id_exp]:
                return True

            exp_fluents = list(self._extract_fluents(exp))
            if not any(map(lambda f: f in assignments_changes, exp_fluents)):
                return False

            possible_values = self._possible_values(
                exp, assignments_changes, exp_fluents=exp_fluents
            )
        else:
            exp_fluents = list(self._extract_fluents(exp))
            possible_values = self._possible_values(
                exp, assignments_changes=None, exp_fluents=exp_fluents
            )

        for value in possible_values:
            assert isinstance(value, bool)
            if value == True:
                self.cache_can_be_true[id_exp] = True
                return True

        self.cache_can_be_true[id_exp] = False
        return False

    def _can_be_true(
        self,
        expressions: Tuple[Expression, ...],
        assignments_changes: Set,
    ) -> bool:

        for exp in expressions:
            if not self._exp_can_be_true(exp, assignments_changes):
                return False
        return True

    def eval(self, state: State, ss: SearchSpace) -> Optional[float]:
        self.assignments = {}
        self.cache_can_be_true = {}

        for f, v in state.assignments.items():
            assert f not in self.assignments
            self.assignments[f] = {v}

        # add extra fluents to assignments
        for action in self._events.keys():
            j, _ = state.todo.get(action, (None, None))
            if j is None:
                idx = len(self._extra_fluents[action]) - 1
            else:
                idx = j - 1

            for i, f in enumerate(self._extra_fluents[action]):
                assert f[0] not in self.assignments
                self.assignments[f[0]] = {i == idx}

        assignments_changes = set(self.assignments.keys())
        assignments_changed = True
        depth = 0

        while assignments_changed:
            if self._can_be_true(self._goals + self._extra_goals, assignments_changes):
                # goal satisfied
                # print("return", depth)
                return depth

            new_assignments: Dict[str, Set[Union[bool, int, Fraction, str]]] = (
                defaultdict(set)
            )
            for i, operator in enumerate(self._operators):
                if not self._can_be_true(
                    operator.conditions, assignments_changes
                ):
                    # operator cannot be applied
                    continue

                for effect in operator.effects:
                    fluent, value = effect
                    possible_values = self._possible_values(value)
                    new_assignments[fluent].update(possible_values)

            # update assignments
            assignments_changes = set()
            assignments_changed = False
            for fluent, vv in new_assignments.items():
                prev_len = len(self.assignments[fluent])
                self.assignments[fluent].update(vv)
                if len(self.assignments[fluent]) > prev_len:
                    assignments_changes.add(fluent)
                    assignments_changed = True

            depth += 1

        # print("return", None)
        return None
