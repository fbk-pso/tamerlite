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

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Iterable, List, Dict, Tuple, Union, Optional, Set, Iterator
from fractions import Fraction
from collections import defaultdict
import itertools
from abc import ABC, abstractmethod

from tamerlite.core.search_space import (
    Event,
    SearchSpaceABC,
    Expression,
    FluentNode,
    State,
    Timing,
    evaluate,
)
from tamerlite.core.search_space import OperatorNode as Op


@dataclass(eq=True, frozen=True)
class AndNode:
    num_operands: int

@dataclass(eq=True, frozen=True)
class OrNode:
    num_operands: int

@dataclass(eq=True, frozen=True)
class LeafNode:
    expression: Expression

HeuristicExpressionNode = Union[AndNode, OrNode, LeafNode]

@dataclass(eq=True, frozen=True)
class Operator:
    action: str
    conditions: Tuple[HeuristicExpressionNode]
    effects: Tuple[Tuple[int, Union[Expression, bool, int, str]], ...]
    cost: float

class HeuristicKind(Enum):
    HFF = 1
    HADD = 2
    HMAX = 3

class Heuristic(ABC):
    def __init__(self, cache_value_in_state: bool = False):
        self.cache_value_in_state = cache_value_in_state

    def eval(self, state: State, ss: SearchSpaceABC) -> Optional[float]:
        if self.cache_value_in_state and self.name in state.heuristic_cache:
            return state.heuristic_cache[self.name]

        h = self._eval(state, ss)
        if self.cache_value_in_state:
            state.heuristic_cache[self.name] = h
        return h

    def eval_gen(self, states: Iterable[State], ss: SearchSpaceABC) -> Iterable[Tuple[State, Optional[float]]]:
        '''
        This function is used to evaluate multiple states at once.
        '''
        for state in states:
            yield state, self.eval(state, ss)

    @abstractmethod
    def _eval(self, state: State, ss: SearchSpaceABC) -> Optional[float]:
        pass

    @property
    @abstractmethod
    def name(self):
        pass


class CustomHeuristic(Heuristic):
    def __init__(self, callable: Callable[[State], Optional[float]], cache_value_in_state: bool = False):
        super().__init__(cache_value_in_state)
        self.callable = callable

    def _eval(self, state: State, ss: SearchSpaceABC) -> Optional[float]:
        return self.callable(state)

    @property
    def name(self):
        return "custom"

def HFF(fluent_types: List[str], objects: Dict[str, List[str]],
         events: Dict[str, List[Tuple[Timing, Event]]], goals: Expression, internal_caching: bool, cache_value_in_state: bool):
    return DeleteRelaxationHeuristic(fluent_types, objects, events, goals, HeuristicKind.HFF, internal_caching, cache_value_in_state)

def HAdd(fluent_types: List[str], objects: Dict[str, List[str]],
         events: Dict[str, List[Tuple[Timing, Event]]], goals: Expression, internal_caching: bool, cache_value_in_state: bool):
    return DeleteRelaxationHeuristic(fluent_types, objects, events, goals, HeuristicKind.HADD, internal_caching, cache_value_in_state)

def HMax(fluent_types: List[str], objects: Dict[str, List[str]],
         events: Dict[str, List[Tuple[Timing, Event]]], goals: Expression, internal_caching: bool, cache_value_in_state: bool):
    return DeleteRelaxationHeuristic(fluent_types, objects, events, goals, HeuristicKind.HMAX, internal_caching, cache_value_in_state)


class _DeleteRelaxationHeuristicBase(Heuristic):
    def __init__(
        self,
        fluent_types: List[str],
        objects: Dict[str, List[str]],
        events: Dict[str, List[Tuple[Timing, Event]]],
        goals: Expression,
        internal_caching: bool,
        cache_value_in_state: bool,
        ignore_real_int: bool = False,
    ):
        super().__init__(cache_value_in_state)
        self._fluent_types = fluent_types
        self._objects = objects
        self._events = events
        self._operators: List[Operator] = []
        self._extra_fluents: Dict[str, List[int]] = {}
        self._num_fluents = len(self._fluent_types)

        for a, le in events.items():
            self._extra_fluents[a] = []
            f_cond = self._num_fluents + len(le) - 1
            cond = FluentNode(f_cond)
            for _, e in le:
                effects = []
                f = self._num_fluents
                self._num_fluents += 1
                self._extra_fluents[a].append(f)
                effects.append((f, True))
                for eff in e.effects:
                    t = self._fluent_types[eff.fluent]
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
                        if len(eff.value) == 1 and isinstance(eff.value[0], str):
                            # eff.value[0] is an object
                            effects.append((eff.fluent, eff.value[0]))
                        else:
                            for obj in objects[self._fluent_types[eff.fluent]]:
                                effects.append((eff.fluent, obj))
                is_applicable, conditions = self.build_operator_conditions(e.conditions, cond)
                if is_applicable:
                    self._operators.append(Operator(a, conditions, tuple(effects), 1))
                cond = FluentNode(f)
        self._goals = self.convert_to_heuristic_expression(goals)
        extra_goals = tuple(FluentNode(fe[-1]) for fe in self._extra_fluents.values())
        extra_goals += (Op("and", tuple(range(len(extra_goals)))), )
        self._extra_goals = self.convert_to_heuristic_expression(extra_goals)

        self._ordered_actions = list(self._events.keys())
        self._internal_caching = {} if internal_caching else None

    def build_operator_conditions(
        self, conditions: Expression, extra_fluent: FluentNode
    ) -> Tuple[bool, Tuple[HeuristicExpressionNode]]:
        conditions = tuple(conditions[::])
        if conditions == (False,):
            return False, tuple()
        if len(conditions) == 0 or conditions == (True,):
            conditions = (extra_fluent,)
        elif isinstance(conditions[-1], Op) and conditions[-1].kind == "and":
            and_op = Op("and", conditions[-1].operands + (len(conditions) - 1,))
            conditions = conditions[:-1] + (extra_fluent, and_op)
        else:
            conditions = conditions + (
                extra_fluent,
                Op("and", (len(conditions) - 1, len(conditions))),
            )

        return True, self.convert_to_heuristic_expression(conditions)

    def convert_to_heuristic_expression(
        self, exp: Expression
    ) -> Tuple[HeuristicExpressionNode]:
        result = []
        stack = [(len(exp) - 1, False)]
        while len(stack) > 0:
            idx, processed = stack.pop()
            e = exp[idx]

            if (
                isinstance(e, bool)
                or isinstance(e, int)
                or isinstance(e, Fraction)
                or isinstance(e, str)
                or isinstance(e, FluentNode)
            ):
                result.append(LeafNode((e,)))
            elif isinstance(e, Op) and e.kind == "and":
                if not processed:
                    stack.append((idx, True))
                    for i in e.operands:
                        stack.append((i, False))
                else:
                    result.append(AndNode(len(e.operands)))
            elif isinstance(e, Op) and e.kind == "or":
                if not processed:
                    stack.append((idx, True))
                    for i in e.operands:
                        stack.append((i, False))
                else:
                    result.append(OrNode(len(e.operands)))
            else:
                result.append(LeafNode(self._extract_sub_expression(exp, idx)))

        return tuple(result)

    def _is_numeric_leaf_expression(self, exp: Expression) -> bool:
        is_numeric = True
        if isinstance(exp[-1], bool):  # boolean constant
            is_numeric = False
        elif isinstance(exp[-1], FluentNode):  # boolean fluent expression
            is_numeric = False
        elif (
            isinstance(exp[-1], Op) and exp[-1].kind == "not"
        ):  # not of a boolean fluent expression
            i = exp[-1].operands[0]
            if isinstance(exp[i], FluentNode):
                is_numeric = False
        elif (
            isinstance(exp[-1], Op) and exp[-1].kind == "=="
        ):  # equals between a fluent and an object
            i1 = exp[-1].operands[0]
            i2 = exp[-1].operands[1]
            if isinstance(exp[i1], FluentNode) and isinstance(exp[i2], str):
                is_numeric = False

        return is_numeric

    def _extract_sub_expression(self, exp: Expression, idx: int) -> Expression:
        # find the start index of the sub-expression
        i = idx
        while isinstance(exp[i], Op):
            i = min(exp[i].operands)

        offset = i
        res = []
        for j in range(i, idx + 1):
            e = exp[j]
            if isinstance(e, Op):
                res.append(Op(e.kind, tuple([o - offset for o in e.operands])))
            else:
                res.append(e)

        return tuple(res)


class DeleteRelaxationHeuristic(_DeleteRelaxationHeuristicBase):
    def __init__(
        self,
        fluent_types: List[str],
        objects: Dict[str, List[str]],
        events: Dict[str, List[Tuple[Timing, Event]]],
        goals: Expression,
        heuristic_kind: HeuristicKind,
        internal_caching: bool,
        cache_value_in_state: bool,
    ):
        super().__init__(
            fluent_types,
            objects,
            events,
            goals,
            internal_caching,
            cache_value_in_state,
            ignore_real_int=True,
        )
        self._heuristic_kind = heuristic_kind

        self._precondition_of: Dict[Expression, List[Operator]] = {}
        self._numeric_conds: Set[Expression] = set()
        self._empty_pre_operators: List[Operator] = []
        for o in self._operators:
            if len(o.conditions) == 0:
                self._empty_pre_operators.append(o)
            else:
                for node in o.conditions:
                    if isinstance(node, LeafNode):
                        if self._is_numeric_leaf_expression(node.expression):
                            self._numeric_conds.add(node.expression)
                        else:
                            if node.expression not in self._precondition_of:
                                self._precondition_of[node.expression] = []
                            self._precondition_of[node.expression].append(o)
        for node in self._goals:
            if isinstance(node, LeafNode):
                if self._is_numeric_leaf_expression(node.expression):
                    self._numeric_conds.add(node.expression)

    @property
    def name(self):
        if self._heuristic_kind == HeuristicKind.HFF:
            return "hff"
        if self._heuristic_kind == HeuristicKind.HADD:
            return "hadd"
        if self._heuristic_kind == HeuristicKind.HMAX:
            return "hmax"

    def _eval(self, state: State, ss: SearchSpaceABC) -> Optional[float]:
        if self._internal_caching is not None:
            assignments_values = tuple(state.assignments) + tuple(
                state.todo.get(action, (None, None))[0]
                for action in self._ordered_actions
            )
            if assignments_values in self._internal_caching:
                return self._internal_caching[assignments_values]

            res = self._eval_core(state)
            self._internal_caching[assignments_values] = res
        else:
            res = self._eval_core(state)

        return res

    def _eval_core(self, state: State) -> Optional[float]:
        costs = {}
        for f, v in enumerate(state.assignments):
            if v == True:
                k = (FluentNode(f), )
            elif v == False:
                k = (FluentNode(f), Op("not", (0, )))
            else:
                k = (FluentNode(f), v, Op("==", (0, 1)))
            costs[k] = 0

        # TODO: lazy eval?
        for x in self._numeric_conds:
            if evaluate(x, state):
                costs[x] = 0
            else:
                costs[x] = 1

        for a in self._events.keys():
            j, _ = state.todo.get(a, (None, None))
            if j is None:
                f = self._extra_fluents[a][-1]
            else:
                f = self._extra_fluents[a][j-1]
            x = (FluentNode(f), )
            costs[x] = 0

        lp = list(costs.keys())
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
                            k = (FluentNode(f), )
                        elif v == False:
                            k = (FluentNode(f), Op("not", (0, )))
                        else:
                            k = (FluentNode(f), v, Op("==", (0, 1)))
                        new_cost_k = new_costs.get(k, None)
                        cost_k = costs.get(k, None)
                        if ((new_cost_k is not None and new_cost_k > c + o.cost) or
                            (new_cost_k is None and cost_k is None) or
                            (new_cost_k is None and cost_k > c + o.cost)):
                            if self._heuristic_kind == HeuristicKind.HFF:
                                reached_by[k] = o
                            new_costs[k] = c + o.cost
                            lp.append(k)
                        elif (
                            self._heuristic_kind == HeuristicKind.HFF
                            and (
                                (new_cost_k is not None and new_cost_k == c + o.cost)
                                or (new_cost_k is None and cost_k == c + o.cost)
                            )
                            and o.action > reached_by[k].action
                        ):
                            reached_by[k] = o

            costs.update(new_costs)

        h = self._cost(self._goals, costs)
        if h is None:
            return None

        if self._heuristic_kind != HeuristicKind.HFF:
            eh = self._cost(self._extra_goals, costs)

            if self._heuristic_kind == HeuristicKind.HMAX:
                res = max(h, eh)
            else:
                res = h + eh

            return res

        res = 0
        for a, (j, _) in state.todo.items():
            res += len(self._events[a]) - j

        if h == 0:
            return res

        # FIXME
        relaxed_plan = set()
        stack = [node.expression for node in self._goals if isinstance(node, LeafNode)]
        while len(stack) > 0:
            g = stack.pop()
            o = reached_by.get(g, None)
            if o is None:
                continue
            relaxed_plan.add(o.action)
            stack += [
                node.expression for node in o.conditions if isinstance(node, LeafNode)
            ]

        for a in relaxed_plan:
            if a not in state.todo:
                res += len(self._events[a])

        return res

    def _cost(
        self, exp: Tuple[HeuristicExpressionNode], costs: Dict[Expression, float]
    ) -> Optional[float]:
        if isinstance(exp[-1], LeafNode):
            return costs.get(exp[-1].expression, None)

        res = []
        for node in exp:
            if isinstance(node, LeafNode):
                res.append(costs.get(node.expression, None))
            elif isinstance(node, AndNode):
                v = 0
                operands_values = [res.pop() for i in range(node.num_operands)]
                for ov in operands_values:
                    if isinstance(ov, int):
                        if self._heuristic_kind == HeuristicKind.HMAX:
                            v = max(v, ov)
                        else:
                            v += ov
                    else:
                        v = None
                        break
                res.append(v)
            elif isinstance(node, OrNode):
                operands_values = [res.pop() for i in range(node.num_operands)]
                operands_values = [ov for ov in operands_values if isinstance(ov, int)]
                if len(operands_values) > 0:
                    res.append(min(operands_values))
                else:
                    res.append(None)

        assert len(res) == 1
        return res[-1]


class HMaxNumeric(_DeleteRelaxationHeuristicBase):
    def __init__(
        self,
        fluent_types: List[str],
        objects: Dict[str, List[str]],
        events: Dict[str, List[Tuple[Timing, Event]]],
        goals: Expression,
        internal_caching: bool,
        cache_value_in_state: bool
    ):
        super().__init__(
            fluent_types,
            objects,
            events,
            goals,
            internal_caching,
            cache_value_in_state,
            ignore_real_int=False,
        )

        self._operator_conditions_fluents: List[Set[int]] = []
        for operator in self._operators:
            self._operator_conditions_fluents.append(set())
            for cond in operator.conditions:
                for expr_node in cond:
                    if isinstance(expr_node, FluentNode):
                        self._operator_conditions_fluents[-1].add(expr_node.fluent)

        self._operator_effects_fluents: List[Set[int]] = []
        for operator in self._operators:
            self._operator_effects_fluents.append(set())
            for fluent, eff in operator.effects:
                if isinstance(eff, FluentNode):
                    self._operator_effects_fluents[-1].add(eff.fluent)
                elif isinstance(eff, tuple):
                    self._operator_effects_fluents[-1].update(
                        expression_node.fluent
                        for expression_node in eff
                        if isinstance(expression_node, FluentNode)
                    )

    @property
    def name(self):
        return "hmax_numeric"

    def _extract_fluents(
        self,
        exp: Expression,
        cache_extract_fluents: Dict[int, Set[str]],
    ) -> Set[int]:
        if id(exp) not in cache_extract_fluents:
            cache_extract_fluents[id(exp)] = set(
                expression_node.fluent
                for expression_node in exp
                if isinstance(expression_node, FluentNode)
            )
        return cache_extract_fluents[id(exp)]

    def _possible_values(
        self,
        exp: Union[Expression, bool, int, Fraction, str],
        assignments: List[Set[Union[bool, int, Fraction, str]]],
        cache_extract_fluents: Dict[int, Set[str]],
        exp_fluents: Set[int] = None,
    ) -> Iterator[Union[bool, int, Fraction, str]]:
        if isinstance(exp, tuple):
            if exp_fluents is None:
                exp_fluents = self._extract_fluents(
                    exp, cache_extract_fluents
                )
            values = map(lambda f: assignments[f], exp_fluents)
            state_assignments = [None] * len(assignments)
            for assignments_values in itertools.product(*values):
                for f, v in zip(exp_fluents, assignments_values):
                    state_assignments[f] = v
                state = State(state_assignments, None, None, None, None, None)
                yield evaluate(exp, state)
        else:
            yield exp

    def _exp_can_be_true(
        self,
        exp: Expression,
        assignments: List[Set[Union[bool, int, Fraction, str]]],
        assignments_changes: Set[int],
        cache_can_be_true: Dict[int, bool],
        cache_extract_fluents: Dict[int, Set[str]],
    ) -> bool:
        exp_fluents = None
        id_exp = id(exp)
        if id_exp in cache_can_be_true:
            if cache_can_be_true[id_exp]:
                return True

            exp_fluents = self._extract_fluents(exp, cache_extract_fluents)
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
        assignments: List[Set[Union[bool, int, Fraction, str]]],
        assignments_changes: Set[int],
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

    def _eval(self, state: State, ss: SearchSpaceABC) -> Optional[float]:
        if self._internal_caching is not None:
            assignments_values = tuple(state.assignments) + tuple(
                state.todo.get(action, (None, None))[0]
                for action in self._ordered_actions
            )
            if assignments_values in self._internal_caching:
                return self._internal_caching[assignments_values]

            res = self._eval_core(state)
            self._internal_caching[assignments_values] = res
        else:
            res = self._eval_core(state)

        return res

    def _eval_core(self, state: State) -> Optional[float]:
        assignments: List[Set[Union[bool, int, Fraction, str]]] = [
            {v} for v in state.assignments
        ] + [{} for _ in range(self._num_fluents - len(state.assignments))]

        # add extra fluents to assignments
        for action in self._ordered_actions:
            j, _ = state.todo.get(action, (None, None))
            if j is None:
                idx = len(self._extra_fluents[action]) - 1
            else:
                idx = j - 1

            for i, f in enumerate(self._extra_fluents[action]):
                assignments[f] = {i == idx}

        cache_can_be_true: Dict[int, bool] = {}
        cache_extract_fluents: Dict[int, Set[str]] = {}
        applied_operators = [False] * len(self._operators)

        assignments_changes = set(range(self._num_fluents))
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
                return depth

            new_assignments: Dict[int, Set[Union[bool, int, Fraction, str]]] = (
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

        return None
