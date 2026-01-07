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
    ExpressionNode,
    Expression,
    FluentNode,
    State,
    Timing,
    evaluate,
)
from tamerlite.core.search_space import OperatorNode as Op, split_expression


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
HeuristicExpression = Tuple[HeuristicExpressionNode, ...]


@dataclass(eq=True, frozen=True)
class Operator:
    action: str
    conditions: HeuristicExpression
    effects: Tuple[Tuple[int, Union[bool, str]], ...]
    cost: float


@dataclass(eq=True, frozen=True)
class OperatorHmax:
    action: str
    conditions: Tuple[Expression, ...]
    effects: Tuple[Tuple[int, Union[Expression, bool, int, Fraction, str]], ...]
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

    def eval_gen(
        self, states: Iterable[State], ss: SearchSpaceABC
    ) -> Iterable[Tuple[State, Optional[float]]]:
        """
        This function is used to evaluate multiple states at once.
        """
        for state in states:
            yield state, self.eval(state, ss)

    @abstractmethod
    def _eval(self, state: State, ss: SearchSpaceABC) -> Optional[float]:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass


class CustomHeuristic(Heuristic):
    def __init__(
        self,
        callable: Callable[[State], Optional[float]],
        cache_value_in_state: bool = False,
    ):
        super().__init__(cache_value_in_state)
        self.callable = callable

    def _eval(self, state: State, ss: SearchSpaceABC) -> Optional[float]:
        return self.callable(state)

    @property
    def name(self) -> str:
        return "custom"


class DeleteRelaxationHeuristic(Heuristic):
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
        super().__init__(cache_value_in_state)
        self._heuristic_kind = heuristic_kind
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
                effects: List[Tuple[int, Union[bool, str]]] = []
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
                    elif t != "real" and t != "int":
                        if len(eff.value) == 1 and isinstance(eff.value[0], str):
                            # eff.value[0] is an object
                            effects.append((eff.fluent, eff.value[0]))
                        else:
                            for obj in objects[self._fluent_types[eff.fluent]]:
                                effects.append((eff.fluent, obj))
                is_applicable, conditions = self._build_operator_condition(
                    e.conditions, cond
                )
                if is_applicable:
                    self._operators.append(Operator(a, conditions, tuple(effects), 1.0))
                cond = FluentNode(f)
        self._goals = self._convert_to_heuristic_expression(goals)
        extra_goals: Expression = tuple(
            FluentNode(fe[-1]) for fe in self._extra_fluents.values()
        )
        extra_goals += (Op("and", tuple(range(len(extra_goals)))),)
        self._extra_goals = self._convert_to_heuristic_expression(extra_goals)

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

        self._ordered_actions = list(self._events.keys())
        self._internal_caching: Optional[
            Dict[Tuple[Union[bool, int, Fraction, str, None], ...], Optional[float]]
        ] = ({} if internal_caching else None)

    @property
    def name(self) -> str:
        if self._heuristic_kind == HeuristicKind.HFF:
            return "hff"
        if self._heuristic_kind == HeuristicKind.HADD:
            return "hadd"
        if self._heuristic_kind == HeuristicKind.HMAX:
            return "hmax"

    def _build_operator_condition(
        self, condition: Expression, extra_fluent: FluentNode
    ) -> Tuple[bool, HeuristicExpression]:
        """
        Build the operator condition as a `HeuristicExpression`.

        This method takes an existing condition (represented as an `Expression`)
        and add an additional fluent (`extra_fluent`). The final result is converted
        into a `HeuristicExpression`.

        Args:
            condition (Expression): The condition of the operator.
            extra_fluent (FluentNode): The additional fluent to include in the condition.

        Returns:
            Tuple[bool, HeuristicExpression]: A tuple where:
                - The first element is a boolean indicating whether the operator is applicable
                (i.e., the condition is not explicitly False)
                - The second element is the resulting `HeuristicExpression`.
        """

        # If the condition is explicitly False, the operator is not applicable
        if condition == (False,):
            return False, tuple()

        # If the condition is empty or trivially True, the condition become the extra_fluent
        if len(condition) == 0 or condition == (True,):
            condition = (extra_fluent,)

        # If the last node is an AND operation, add the new fluent as operand
        elif isinstance(condition[-1], Op) and condition[-1].kind == "and":
            and_op = Op("and", condition[-1].operands + (len(condition) - 1,))
            condition = condition[:-1] + (extra_fluent, and_op)

        # Otherwise, combine the condition and extra_fluent using a new AND operation
        else:
            condition = condition + (
                extra_fluent,
                Op("and", (len(condition) - 1, len(condition))),
            )

        return True, self._convert_to_heuristic_expression(condition)

    def _convert_to_heuristic_expression(self, exp: Expression) -> HeuristicExpression:
        """
        Convert an expression into a `HeuristicExpression`.

        A `HeuristicExpression` represents the input expression where:
        - Only `AND` and `OR` operations are internal nodes.
        - All other elements are represented as `LeafNode`s.

        Args:
            exp (Expression): The input expression to convert.

        Returns:
            HeuristicExpression: A tuple representing the converted expression
                with `AndNode`, `OrNode`, and `LeafNode` elements.
        """

        result: List[HeuristicExpressionNode] = []
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
        """
        Determine if a leaf expression represents a numeric expression.
        A leaf expression is assumed to contain no AND or OR nodes.

        Args:
            exp (Expression): The leaf expression to check.

        Returns:
            bool: True if the expression is numeric, False otherwise.
        """

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
        """
        Extract the sub-expression from a given expression rooted at a specified index.
        All operands in the extracted sub-expression are re-indexed relative to the
        start of the sub-expression.

        Args:
            exp (Expression): The full expression from which to extract the sub-expression.
            idx (int): The index of the root node of the sub-expression.

        Returns:
            Expression: A tuple representing the extracted sub-expression with operands
                re-indexed relative to the sub-expression start.
        """

        # find the start index of the sub-expression
        i = idx
        while isinstance(exp[i], Op):
            i = min(exp[i].operands)  # type: ignore[union-attr]

        offset = i
        res: List[ExpressionNode] = []
        for j in range(i, idx + 1):
            e = exp[j]
            if isinstance(e, Op):
                res.append(Op(e.kind, tuple([o - offset for o in e.operands])))
            else:
                res.append(e)

        return tuple(res)

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
        costs: Dict[Expression, float] = {}
        for f, v in enumerate(state.assignments):
            if v == True:
                k: Expression = (FluentNode(f),)
            elif v == False:
                k = (FluentNode(f), Op("not", (0,)))
            else:
                k = (FluentNode(f), v, Op("==", (0, 1)))
            costs[k] = 0.0

        for x in self._numeric_conds:
            if evaluate(x, state):
                costs[x] = 0.0
            else:
                costs[x] = 1.0

        for a in self._events.keys():
            j, _ = state.todo.get(a, (None, None))
            if j is None:
                f = self._extra_fluents[a][-1]
            else:
                f = self._extra_fluents[a][j - 1]
            x = (FluentNode(f),)
            costs[x] = 0.0

        lp = list(costs.keys())
        reached_by: Dict[Expression, Tuple[Operator, List[Expression]]] = {}
        while len(lp) > 0:
            lo = list(self._empty_pre_operators)
            for p in lp:
                if p in self._precondition_of:
                    lo.extend(self._precondition_of[p])
            lp = []
            new_costs: Dict[Expression, float] = {}
            for o in set(lo):
                c, l = self._cost(o.conditions, costs)
                if c is not None:
                    for f, e in o.effects:
                        if e == True:
                            k: Expression = (FluentNode(f),)
                        elif e == False:
                            k = (FluentNode(f), Op("not", (0,)))
                        else:
                            k = (FluentNode(f), e, Op("==", (0, 1)))
                        new_cost_k = new_costs.get(k, None)
                        cost_k = costs.get(k, None)
                        if (new_cost_k is not None and new_cost_k > c + o.cost) or (
                            new_cost_k is None
                            and (cost_k is None or cost_k > c + o.cost)
                        ):
                            if self._heuristic_kind == HeuristicKind.HFF:
                                reached_by[k] = (o, l)
                            new_costs[k] = c + o.cost
                            lp.append(k)
                        elif (
                            self._heuristic_kind == HeuristicKind.HFF
                            and (
                                (new_cost_k is not None and new_cost_k == c + o.cost)
                                or (new_cost_k is None and cost_k == c + o.cost)
                            )
                            and o.action > reached_by[k][0].action
                        ):
                            reached_by[k] = (o, l)

            costs.update(new_costs)

        h, _ = self._cost(self._goals, costs)
        if h is None:
            return None

        if self._heuristic_kind != HeuristicKind.HFF:
            eh, _ = self._cost(self._extra_goals, costs)
            assert eh is not None

            if self._heuristic_kind == HeuristicKind.HMAX:
                res = max(h, eh)
            else:
                res = h + eh

            return res

        res = 0
        for a, (j, _) in state.todo.items():
            res += len(self._events[a]) - j

        if h == 0.0:
            return float(res)

        relaxed_plan = set()
        stack = list(set(self._cost(self._goals, costs)[1]))
        visited_expressions = set()
        while len(stack) > 0:
            g = stack.pop()
            if g not in reached_by:
                continue
            o, l = reached_by[g]
            relaxed_plan.add(o.action)
            for exp in l:
                if exp not in visited_expressions:
                    visited_expressions.add(exp)
                    stack.append(exp)

        for a in relaxed_plan:
            if a not in state.todo:
                res += len(self._events[a])

        return float(res)

    def _cost(
        self, exp: HeuristicExpression, costs: Dict[Expression, float]
    ) -> Tuple[Optional[float], List[Expression]]:
        """
        Calculate the cost of an expression along with the leaf expressions that
        contributed to the computed cost.

        Leaf expressions are collected according to the type of node:
        - AND nodes: all leaf expressions from the operands are included
        - OR nodes: only the leaf expressions from the operand with the minimum cost are included

        Args:
            exp (HeuristicExpression): The expression to evaluate.
            costs (Dict[Expression, float]): A mapping from leaf expressions to their costs.

        Returns:
            Tuple[Optional[float], List[Expression]]:
                - The total cost of the expression
                - A list of leaf expressions that were considered in computing the cost
        """

        if isinstance(exp[-1], LeafNode):
            return costs.get(exp[-1].expression, None), [exp[-1].expression]

        res: List[Tuple[Optional[float], List[Expression]]] = []
        for node in exp:
            if isinstance(node, LeafNode):
                res.append((costs.get(node.expression, None), [node.expression]))
            elif isinstance(node, AndNode):
                v = 0.0
                l = []
                operands_values = [res.pop() for i in range(node.num_operands)]
                for ov, ol in operands_values:
                    if ov is not None:
                        if self._heuristic_kind == HeuristicKind.HMAX:
                            v = max(v, ov)  # type: ignore[type-var]
                        else:
                            v += ov
                            l.extend(ol)
                    else:
                        v = None  # type: ignore[assignment]
                        l = []
                        break
                res.append((v, l))
            elif isinstance(node, OrNode):
                operands_values = [res.pop() for _ in range(node.num_operands)]
                operands_values = [
                    (ov, ol) for ov, ol in operands_values if ov is not None
                ]
                if len(operands_values) > 0:
                    mv, ml = operands_values[0]
                    for ov, ol in operands_values:
                        if ov < mv:  # type: ignore[operator]
                            mv = ov
                            ml = ol
                    res.append((mv, ml))
                else:
                    res.append((None, []))

        assert len(res) == 1
        return res[-1]


def HFF(
    fluent_types: List[str],
    objects: Dict[str, List[str]],
    events: Dict[str, List[Tuple[Timing, Event]]],
    goals: Expression,
    internal_caching: bool,
    cache_value_in_state: bool,
) -> DeleteRelaxationHeuristic:
    return DeleteRelaxationHeuristic(
        fluent_types,
        objects,
        events,
        goals,
        HeuristicKind.HFF,
        internal_caching,
        cache_value_in_state,
    )


def HAdd(
    fluent_types: List[str],
    objects: Dict[str, List[str]],
    events: Dict[str, List[Tuple[Timing, Event]]],
    goals: Expression,
    internal_caching: bool,
    cache_value_in_state: bool,
) -> DeleteRelaxationHeuristic:
    return DeleteRelaxationHeuristic(
        fluent_types,
        objects,
        events,
        goals,
        HeuristicKind.HADD,
        internal_caching,
        cache_value_in_state,
    )


def HMax(
    fluent_types: List[str],
    objects: Dict[str, List[str]],
    events: Dict[str, List[Tuple[Timing, Event]]],
    goals: Expression,
    internal_caching: bool,
    cache_value_in_state: bool,
) -> DeleteRelaxationHeuristic:
    return DeleteRelaxationHeuristic(
        fluent_types,
        objects,
        events,
        goals,
        HeuristicKind.HMAX,
        internal_caching,
        cache_value_in_state,
    )


class HMaxNumeric(Heuristic):
    def __init__(
        self,
        fluent_types: List[str],
        objects: Dict[str, List[str]],
        events: Dict[str, List[Tuple[Timing, Event]]],
        goals: Expression,
        internal_caching: bool,
        cache_value_in_state: bool,
    ):
        super().__init__(cache_value_in_state)
        self._fluent_types = fluent_types
        self._objects = objects
        self._events = events
        self._operators: List[OperatorHmax] = []
        self._extra_fluents: Dict[str, List[int]] = {}
        self._num_fluents = len(self._fluent_types)

        for a, le in events.items():
            self._extra_fluents[a] = []
            f_cond = self._num_fluents + len(le) - 1
            cond = (FluentNode(f_cond),)
            for _, e in le:
                effects: List[
                    Tuple[int, Union[Expression, bool, int, Fraction, str]]
                ] = []
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
                    elif t == "real" or t == "int":
                        if len(eff.value) == 1:
                            assert not isinstance(eff.value[0], Op)
                            effects.append((eff.fluent, eff.value[0]))  # type: ignore[arg-type]
                        else:
                            effects.append((eff.fluent, eff.value))
                    else:
                        if len(eff.value) == 1 and isinstance(eff.value[0], str):
                            # eff.value[0] is an object
                            effects.append((eff.fluent, eff.value[0]))
                        else:
                            for obj in objects[self._fluent_types[eff.fluent]]:
                                effects.append((eff.fluent, obj))
                if len(e.conditions) == 0 or e.conditions == (True,):
                    conditions: Tuple[Expression, ...] = (cond,)
                else:
                    conditions = split_expression(e.conditions) + (cond,)
                cond = (FluentNode(f),)
                if (False,) not in conditions:
                    self._operators.append(
                        OperatorHmax(a, conditions, tuple(effects), 1.0)
                    )
        self._extra_goals: Tuple[Expression, ...] = tuple(
            [(FluentNode(fe[-1]),) for fe in self._extra_fluents.values()]
        )
        self._goals = split_expression(goals)

        self._operator_conditions_fluents: List[Set[int]] = []
        for operator in self._operators:
            self._operator_conditions_fluents.append(set())
            for c in operator.conditions:
                for expr_node in c:
                    if isinstance(expr_node, FluentNode):
                        self._operator_conditions_fluents[-1].add(expr_node.fluent)

        self._operator_effects_fluents: List[Set[int]] = []
        for operator in self._operators:
            self._operator_effects_fluents.append(set())
            for fluent, effect in operator.effects:
                if isinstance(effect, FluentNode):
                    self._operator_effects_fluents[-1].add(effect.fluent)
                elif isinstance(effect, tuple):
                    self._operator_effects_fluents[-1].update(
                        expression_node.fluent
                        for expression_node in effect
                        if isinstance(expression_node, FluentNode)
                    )

        self._ordered_actions = list(self._events.keys())
        self._internal_caching: Optional[
            Dict[Tuple[Union[bool, int, Fraction, str, None], ...], Optional[float]]
        ] = ({} if internal_caching else None)

    @property
    def name(self) -> str:
        return "hmax_numeric"

    def _extract_fluents(
        self,
        exp: Expression,
        cache_extract_fluents: Dict[int, Set[int]],
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
        cache_extract_fluents: Dict[int, Set[int]],
        exp_fluents: Optional[Set[int]] = None,
    ) -> Iterator[Union[bool, int, Fraction, str]]:
        if isinstance(exp, tuple):
            if exp_fluents is None:
                exp_fluents = self._extract_fluents(exp, cache_extract_fluents)
            values = map(lambda f: assignments[f], exp_fluents)
            state_assignments: List[Union[bool, int, Fraction, str, None]] = [
                None
            ] * len(assignments)
            for assignments_values in itertools.product(*values):
                for f, v in zip(exp_fluents, assignments_values):
                    state_assignments[f] = v
                state = State(state_assignments, None, None, None, None, None)  # type: ignore
                yield evaluate(exp, state)
        else:
            yield exp

    def _exp_can_be_true(
        self,
        exp: Expression,
        assignments: List[Set[Union[bool, int, Fraction, str]]],
        assignments_changes: Set[int],
        cache_can_be_true: Dict[int, bool],
        cache_extract_fluents: Dict[int, Set[int]],
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
        cache_extract_fluents: Dict[int, Set[int]],
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
        ] + [set() for _ in range(self._num_fluents - len(state.assignments))]

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
        cache_extract_fluents: Dict[int, Set[int]] = {}
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
                return float(depth)

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
