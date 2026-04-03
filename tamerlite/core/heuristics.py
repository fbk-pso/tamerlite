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

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Iterable, List, Dict, Tuple, Union, Optional, Set, Iterator
import math
from fractions import Fraction
from collections import defaultdict
import itertools
from abc import ABC, abstractmethod

from tamerlite.core.search_space import (
    Action,
    Event,
    Effect,
    SearchSpaceABC,
    Expression,
    ExpressionNode,
    FluentNode,
    State,
    Timing,
    evaluate,
    shift_expression,
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
    id: int
    action: Action = field(compare=False)
    conditions: HeuristicExpression = field(compare=False)
    effects: Tuple[Tuple[int, Union[bool, str]], ...] = field(compare=False)
    constant_increase_effects: Dict[int, Union[int, Fraction]] = field(compare=False)
    constant_assign_effects: Dict[int, Union[int, Fraction]] = field(compare=False)
    complex_numeric_effects: Dict[int, Expression] = field(compare=False)
    cost: float = field(compare=False)


@dataclass(eq=True, frozen=True)
class OperatorHmax:
    action: Action
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


def get_event_conditions(event: Event) -> List[Expression]:
    conditions_set = set()
    conditions = []
    for c in split_expression(event.conditions) + event.end_conditions:
        if c not in conditions_set:
            conditions.append(c)
            conditions_set.add(c)
    return conditions


class DeleteRelaxationHeuristic(Heuristic):
    def __init__(
        self,
        actions: List[Action],
        fluent_types: List[str],
        objects: Dict[str, List[str]],
        events: Dict[Action, List[Tuple[Timing, Event]]],
        goals: Expression,
        heuristic_kind: HeuristicKind,
        internal_caching: bool,
        cache_value_in_state: bool,
        disable_numeric_reasoning: bool,
    ):
        super().__init__(cache_value_in_state)
        self._heuristic_kind = heuristic_kind
        self._actions = actions
        self._fluent_types = fluent_types
        self._objects = objects
        self._events = events
        self._operators: List[Operator] = []
        self._extra_fluents: Dict[Action, List[int]] = {}
        self._num_fluents = len(self._fluent_types)
        self._disable_numeric_reasoning = disable_numeric_reasoning

        for a in actions:
            if a not in events:
                continue
            le = events[a]
            self._extra_fluents[a] = []
            f_cond = self._num_fluents + len(le) - 1
            cond = FluentNode(f_cond)
            for _, e in le:
                effects: List[Tuple[int, Union[bool, str]]] = []
                constant_increase_effects: Dict[int, Union[int, Fraction]] = {}
                constant_assign_effects: Dict[int, Union[int, Fraction]] = {}
                complex_numeric_effects: Dict[int, Expression] = {}
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
                        assert (
                            eff.fluent not in constant_increase_effects
                            and eff.fluent not in constant_assign_effects
                            and eff.fluent not in complex_numeric_effects
                        )
                        self._update_numeric_effects(
                            eff,
                            constant_increase_effects,
                            constant_assign_effects,
                            complex_numeric_effects,
                        )
                    else:
                        if len(eff.value) == 1 and isinstance(eff.value[0], str):
                            # eff.value[0] is an object
                            effects.append((eff.fluent, eff.value[0]))
                        else:
                            for obj in objects[self._fluent_types[eff.fluent]]:
                                effects.append((eff.fluent, obj))
                is_applicable, conditions = self._build_operator_condition(
                    get_event_conditions(e), cond
                )
                if is_applicable:
                    self._operators.append(
                        Operator(
                            len(self._operators),
                            a,
                            conditions,
                            tuple(effects),
                            constant_increase_effects,
                            constant_assign_effects,
                            complex_numeric_effects,
                            1.0,
                        )
                    )
                cond = FluentNode(f)
        self._goals = self._simplify_condition(
            self._convert_to_heuristic_expression(goals)
        )
        extra_goals: Expression = tuple(
            FluentNode(fe[-1]) for fe in self._extra_fluents.values()
        )
        extra_goals += (Op("and", tuple(range(len(extra_goals)))),)
        self._extra_goals = self._convert_to_heuristic_expression(extra_goals)

        self._precondition_of: Dict[Expression, List[Operator]] = {}
        self._simple_numeric_conds: Dict[Expression, Tuple[List[int], List[float]]] = {}
        self._lt_simple_numeric_conds: Set[Expression] = set()
        self._complex_numeric_conds: Set[Expression] = set()
        self._empty_pre_operators: List[Operator] = []
        for o in self._operators:
            if len(o.conditions) == 0:
                self._empty_pre_operators.append(o)
            else:
                for node in o.conditions:
                    if isinstance(node, LeafNode):
                        if self._is_numeric_leaf_expression(node):
                            self._update_numeric_conditions(node)

                        if node.expression not in self._precondition_of:
                            self._precondition_of[node.expression] = []
                        self._precondition_of[node.expression].append(o)

        for node in self._goals:
            if isinstance(node, LeafNode):
                if self._is_numeric_leaf_expression(node):
                    self._update_numeric_conditions(node)

        self._max_net_effect = float("-inf")
        self._achieved_simple_numeric_conds: List[List[Expression]] = [
            [] for _ in self._operators
        ]
        for o in self._operators:
            for c in self._simple_numeric_conds:
                if self._achieves(o, c):
                    self._achieved_simple_numeric_conds[o.id].append(c)

        epsilon = -self._max_net_effect / 2
        for simple_cond in self._lt_simple_numeric_conds:
            _, weights = self._simple_numeric_conds[simple_cond]
            weights[-1] += epsilon

        self._internal_caching: Optional[
            Dict[Tuple[Union[bool, int, Fraction, str, None], ...], Optional[float]]
        ] = ({} if internal_caching else None)

    @property
    def name(self) -> str:
        if self._heuristic_kind == HeuristicKind.HFF:
            name = "hff"
        if self._heuristic_kind == HeuristicKind.HADD:
            name = "hadd"
        if self._heuristic_kind == HeuristicKind.HMAX:
            name = "hmax"

        if self._disable_numeric_reasoning:
            name += "_no_numbers"
        return name

    def _simplify_condition(
        self, condition: HeuristicExpression
    ) -> HeuristicExpression:
        """Simplify leaf expressions in a condition.

        Each `LeafNode` in the condition is rewritten when possible.
        The following simplifications are applied:

        - Simple numeric leaf expressions containing logical negation (`not`) or equality
        (`==`) are simplified, unless numeric reasoning is disabled.
        - Fluent-object inequality expressions (`fluent != object`) are
        rewritten into an equivalent form.

        Non-leaf nodes or leaf nodes that do not match any simplification rule
        are left unchanged.

        Args:
            condition: A heuristic expression.

        Returns:
            A new heuristic expression with simplified leaf nodes.
        """

        new_condition: List[HeuristicExpressionNode] = []
        for node in condition:
            new_nodes = None
            if isinstance(node, LeafNode):
                if (
                    not self._disable_numeric_reasoning
                    and self._is_numeric_leaf_expression(node)
                ):
                    new_nodes = self._simplify_numeric_leaf_node(node)
                else:
                    new_nodes = self._simplify_fluent_not_equals_object_expression(node)

            if new_nodes is None:
                new_condition.append(node)
            else:
                new_condition.extend(new_nodes)

        return tuple(new_condition)

    def _simplify_numeric_leaf_node(
        self, node: LeafNode
    ) -> Optional[HeuristicExpression]:
        """Simplify a simple numeric leaf node expression.

        This method rewrites numeric leaf expressions containing logical negation (`not`)
        or equality (`==`) into simpler equivalent expressions suitable for heuristic
        evaluation. Specifically, it transforms:

        - `a == b` into `a <= b and b <= a`.
        - `not(a == b)` into `a < b or b < a`
        - `not(a < b)` into `b <= a`
        - `not(a <= b)` into `b < a`

        Args:
            node: A `LeafNode` containing a numeric expression.

        Returns:
            A new `HeuristicExpression` if simplification is possible; otherwise, `None`.
        """

        def inverted_operands(exp: Expression, op: Op):
            op1, op2 = op.operands
            op1_exp = exp[: op1 + 1]
            op2_exp = exp[op1 + 1 : op2 + 1]
            return shift_expression(op2_exp, -len(op1_exp)), shift_expression(
                op1_exp, len(op2_exp)
            )

        nodes: Optional[HeuristicExpression] = None
        exp = node.expression
        if isinstance(exp[-1], Op):
            if exp[-1].kind == "==":
                exp1 = exp[:-1] + (Op("<=", exp[-1].operands),)
                op1, op2 = inverted_operands(exp, exp[-1])
                exp2 = op1 + op2 + (Op("<=", (len(op1) - 1, len(op1) + len(op2) - 1)),)
                nodes = (LeafNode(exp1), LeafNode(exp2), AndNode(2))
            elif exp[-1].kind == "not":
                negated = exp[exp[-1].operands[0]]
                if isinstance(negated, Op):
                    if negated.kind == "==":
                        exp1 = exp[:-2] + (Op("<", negated.operands),)
                        op1, op2 = inverted_operands(exp, negated)
                        exp2 = (
                            op1
                            + op2
                            + (Op("<", (len(op1) - 1, len(op1) + len(op2) - 1)),)
                        )
                        nodes = (LeafNode(exp1), LeafNode(exp2), OrNode(2))
                    elif negated.kind == "<":
                        op1, op2 = inverted_operands(exp, negated)
                        nodes = (
                            LeafNode(
                                op1
                                + op2
                                + (Op("<=", (len(op1) - 1, len(op1) + len(op2) - 1)),)
                            ),
                        )
                    elif negated.kind == "<=":
                        op1, op2 = inverted_operands(exp, negated)
                        nodes = (
                            LeafNode(
                                op1
                                + op2
                                + (Op("<", (len(op1) - 1, len(op1) + len(op2) - 1)),)
                            ),
                        )

        if nodes is not None:
            # Check whether the leaf node represents a simple numeric expression
            exp = nodes[0].expression  # type: ignore[union-attr]
            polynomial_exp = exp[:-1] + (Op("-", exp[-1].operands),)  # type: ignore
            try:
                self._to_linear_polynomial(polynomial_exp)
            except ValueError:
                return None

        return nodes

    def _simplify_fluent_not_equals_object_expression(
        self, node: LeafNode
    ) -> Optional[HeuristicExpression]:
        """Simplify a leaf expression of the form `fluent != object`.

        This method rewrites inequality expressions between a fluent and a specific
        object into an equivalent disjunction of equalities:

            `fluent != objX` into `fluent == obj1 or fluent == obj2 or ...`

        where `obj1, obj2, ...` are all objects of the fluent's type except `objX`.

        Args:
            node: A `LeafNode` potentially representing a `fluent != object` expression.

        Returns:
            A new `HeuristicExpression` representing the disjunction of equality
            expressions if simplification is possible; otherwise, `None`.
        """

        exp = node.expression
        if (
            len(exp) == 4
            and isinstance(exp[0], FluentNode)
            and isinstance(exp[1], str)
            and isinstance(exp[2], Op)
            and exp[2].kind == "=="
            and isinstance(exp[3], Op)
            and exp[3].kind == "not"
        ):
            nodes: List[HeuristicExpressionNode] = []
            for obj in self._objects[self._fluent_types[exp[0].fluent]]:
                if obj != exp[1]:
                    nodes.append(LeafNode((exp[0], obj, Op("==", (0, 1)))))
            if len(nodes) == 0:
                return (LeafNode((False,)),)
            if len(nodes) > 1:
                nodes.append(OrNode(len(nodes)))
            return tuple(nodes)
        return None

    def _build_operator_condition(
        self, conditions: List[Expression], extra_fluent: FluentNode
    ) -> Tuple[bool, HeuristicExpression]:
        """
        Build the operator condition as a `HeuristicExpression`.

        This method takes the operator conditions and add the `extra_fluent`.
        The final result is converted into a `HeuristicExpression`.

        Args:
            conditions (Expression): The conditions of the operator.
            extra_fluent (FluentNode): The additional fluent to include in the condition.

        Returns:
            Tuple[bool, HeuristicExpression]: A tuple where:
                - The first element is a boolean indicating whether the operator is applicable
                (i.e., the condition is not explicitly False)
                - The second element is the resulting `HeuristicExpression`.
        """

        condition: List[ExpressionNode] = []
        operands = []
        for c in conditions:
            if c == (False,):
                # If the condition is explicitly False, the operator is not applicable
                return False, tuple()
            elif len(c) > 0 and c != (True,):
                condition.extend(shift_expression(c, len(condition)))
                operands.append(len(condition) - 1)
        condition.append(extra_fluent)
        operands.append(len(condition) - 1)
        if len(operands) > 1:
            condition.append(Op("and", tuple(operands)))

        return True, self._simplify_condition(
            self._convert_to_heuristic_expression(tuple(condition))
        )

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
            i = exp[i].operands[0]  # type: ignore[union-attr]

        return shift_expression(exp[i : idx + 1], -i)

    def _update_numeric_effects(
        self,
        effect: Effect,
        constant_increase_effects: Dict[int, Union[int, Fraction]],
        constant_assign_effects: Dict[int, Union[int, Fraction]],
        complex_numeric_effects: Dict[int, Expression],
    ):
        """Processes a numeric effect and categorizes it into one of three
        types:

        1. **Constant assignment:** If the effect is a single numeric value, it is
        stored in `constant_assign_effects`.
        2. **Constant increase:** If the effect represents a linear increase of a
        fluent by a constant amount, it is stored in `constant_increase_effects`.
        3. **Complex numeric effect:** If the effect is non-linear or cannot be
        simplified to a constant increase, it is stored in `complex_numeric_effects`.

        Args:
            effect: The numeric effect to process.
            constant_increase_effects: Mapping from fluent to constant increase
                values, updated if the effect is a simple increase.
            constant_assign_effects: Mapping from fluent to constant assignment
                values, updated if the effect is a constant numeric assignment.
            complex_numeric_effects: Mapping from fluent to expressions for
                effects that are complex.
        """

        if len(effect.value) == 1 and isinstance(effect.value[0], (int, Fraction)):
            constant_assign_effects[effect.fluent] = effect.value[0]
            return

        try:
            polynomial = self._to_linear_polynomial(effect.value)
        except ValueError:
            complex_numeric_effects[effect.fluent] = effect.value
            return

        assert effect.fluent in polynomial
        k = polynomial.pop(None, 0)
        if len(polynomial) == 1 and polynomial[effect.fluent] == 1:
            constant_increase_effects[effect.fluent] = k
        else:
            complex_numeric_effects[effect.fluent] = effect.value

    def _is_numeric_leaf_expression(self, node: LeafNode) -> bool:
        """
        Determine if a leaf expression represents a numeric expression.
        A leaf expression is assumed to contain no AND or OR nodes.

        Args:
            node (LeafNode): The leaf node to check.

        Returns:
            bool: True if the expression is numeric, False otherwise.
        """

        exp = node.expression
        if isinstance(exp[-1], Op):
            i = -1
            if exp[-1].kind == "not":
                i = exp[-1].operands[0]

            if isinstance(exp[i], Op):
                exp_node: Op = exp[i]  # type: ignore[assignment]
                if exp_node.kind != "==":
                    return True

                op1, op2 = exp_node.operands
                if not isinstance(exp[op1], str) and not isinstance(exp[op2], str):
                    return True

        return False

    def _update_numeric_conditions(self, numeric_condition: LeafNode):
        """Processes a numeric condition represented as a `LeafNode` and
        classifies it as either **simple** or **complex**:

        - If numeric reasoning is disabled, the condition is always treated as complex.
        - If the condition can be represented as a simple linear numeric expression,
        it is stored in `_simple_numeric_conds` along with its fluents and weights.
        - Conditions that cannot be simplified are stored in `_complex_numeric_conds`.

        Args:
            numeric_condition: A `LeafNode` representing a numeric condition.
        """

        if self._disable_numeric_reasoning:
            self._complex_numeric_conds.add(numeric_condition.expression)
            return

        fluents_weights = self._extract_fluents_weights_simple_numeric_condition(
            numeric_condition
        )
        if fluents_weights is None:
            self._complex_numeric_conds.add(numeric_condition.expression)
        else:
            fluents, weights, is_lt = fluents_weights
            self._simple_numeric_conds[numeric_condition.expression] = (
                fluents,
                weights,
            )
            if is_lt:
                self._lt_simple_numeric_conds.add(numeric_condition.expression)

    def _extract_fluents_weights_simple_numeric_condition(
        self, node: LeafNode
    ) -> Optional[Tuple[List[int], List[float], bool]]:
        """Extracts fluents and weights from a simple numeric condition.

        This method attempts to interpret a numeric condition of the form
        `linear-expression < constant` or `linear-expression <= constant` as a
        linear polynomial and extract its components:

        - `fluents`: List of fluents appearing in the expression.
        - `weights`: Corresponding coefficients of the fluents, with the constant
        term appended as the last element.
        - `is_lt`: True if the original operator was `<`, False if `<=`.

        Args:
            node: A `LeafNode` representing a condition.

        Returns:
            A tuple `(fluents, weights, is_lt)` if the condition is a simple linear
            numeric condition; otherwise, `None`.
        """

        exp = node.expression
        if not (isinstance(exp[-1], Op) and exp[-1].kind in ("<", "<=")):
            return None

        polynomial_exp = exp[:-1] + (Op("-", exp[-1].operands),)
        try:
            polynomial = self._to_linear_polynomial(polynomial_exp)
        except ValueError:
            return None

        k = float(polynomial.pop(None, 0))
        fluents: List[int] = list(polynomial.keys())  # type: ignore[arg-type]
        weights: List[float] = [float(polynomial[f]) for f in fluents] + [k]
        return fluents, weights, exp[-1].kind == "<"

    def _to_linear_polynomial(
        self, exp: Expression
    ) -> Dict[Optional[int], Union[int, Fraction]]:
        """Converts an expression into a linear polynomial representation.

        This method attempts to represent a numeric expression as a linear
        polynomial of the form:

            w1 * f1 + w2 * f2 + ... + k

        where `fi` are fluents, `wi` are their coefficients, and `k` is a constant
        term.

        Supported operations in the expression are `+`, `-`, `*`, and `/`, provided
        they maintain linearity. If the expression is non-linear (e.g., product of
        two fluents, division by a fluent), a `ValueError` is raised.

        Args:
            exp: The numeric expression.

        Returns:
            A dictionary mapping fluents to coefficients and `None` to the constant term.

        Raises:
            ValueError: If the expression is non-linear or contains unsupported operations.
        """

        res: List[Dict[Optional[int], Union[int, Fraction]]] = []
        for node in exp:
            if isinstance(node, (int, Fraction)):
                res.append({None: node})

            elif isinstance(node, FluentNode):
                res.append({node.fluent: 1})

            elif isinstance(node, Op) and node.kind in ("-", "+", "/", "*"):
                operands = [res.pop() for _ in node.operands]

                def is_constant(polynomial: Dict[Optional[int], Union[int, Fraction]]):
                    return len(polynomial) == 1 and None in polynomial

                if node.kind == "-":
                    result = operands[1]
                    for f, w in operands[0].items():
                        result[f] = result.get(f, 0) - w

                elif node.kind == "+":
                    result = {}
                    for operand in operands:
                        for f, w in operand.items():
                            result[f] = result.get(f, 0) + w

                elif node.kind == "/":
                    dividend = operands[1]
                    divisor = operands[0]
                    if not is_constant(divisor):
                        raise ValueError("non-linear polynomial")

                    result = {
                        f: Fraction(w) / divisor[None] for f, w in dividend.items()
                    }

                elif node.kind == "*":
                    const_multiplier: Fraction = Fraction(1)
                    polynomial = None
                    for operand in operands:
                        if is_constant(operand):
                            const_multiplier *= operand[None]
                        elif polynomial is not None:
                            raise ValueError("non-linear polynomial")
                        else:
                            polynomial = operand

                    if polynomial is None:
                        result = {None: const_multiplier}
                    else:
                        for f in polynomial:
                            polynomial[f] *= const_multiplier
                        result = polynomial

                res.append(result)

            else:
                raise ValueError("non-linear polynomial")

        assert len(res) == 1
        return res[-1]

    def reachable_actions(self, state: State) -> Set[Action]:
        _, reachable_operators = self._eval_core(state, reachability_analysis=True)
        assert reachable_operators is not None

        action_operators = {}
        for o in self._operators:
            if o.action not in action_operators:
                action_operators[o.action] = 1
            else:
                action_operators[o.action] += 1

        action_reachable_operators = {}
        for o in reachable_operators:
            if o.action not in action_reachable_operators:
                action_reachable_operators[o.action] = 1
            else:
                action_reachable_operators[o.action] += 1

        reachable_actions = set(
            a
            for a in action_reachable_operators
            if action_reachable_operators[a] == action_operators[a]
        )
        return reachable_actions

    def _eval(self, state: State, ss: SearchSpaceABC) -> Optional[float]:
        if self._internal_caching is not None:
            assignments_values = tuple(state.assignments) + tuple(
                state.todo.get(action, (None, None))[0] for action in self._actions
            )
            if assignments_values in self._internal_caching:
                return self._internal_caching[assignments_values]

            res, _ = self._eval_core(state)
            self._internal_caching[assignments_values] = res
        else:
            res, _ = self._eval_core(state)

        return res

    def _eval_core(
        self, state: State, reachability_analysis: bool = False
    ) -> Tuple[Optional[float], Optional[List[Operator]]]:
        """Compute the heuristic value for a given state.

        This method evaluates the state using the selected delete-relaxation heuristic,
        which can be one of `hmax`, `hadd`, or `hff`. The returned value estimates
        the cost to reach the goal from the given state.

        If `reachability_analysis` is enabled, the method performs reachability
        analysis instead of computing the heuristic value and returns the set of
        reachable operators.

        Args:
            state: The state to evaluate.
            reachability_analysis: If True, perform reachability analysis and return
                the reachable operators instead of the heuristic value.

        Returns:
            A tuple containing:
                - heuristic: The heuristic value as a float, or `None` if not computed.
                - reachable_operators: A list of reachable `Operator` instances if
                    `reachability_analysis` is True; otherwise `None`.
        """

        costs: Dict[Expression, float] = {}
        for f, v in enumerate(state.assignments):
            if v == True:
                k: Expression = (FluentNode(f),)
            elif v == False:
                k = (FluentNode(f), Op("not", (0,)))
            else:
                k = (FluentNode(f), v, Op("==", (0, 1)))
            costs[k] = 0.0

        for cond in self._simple_numeric_conds:
            if evaluate(cond, state):
                costs[cond] = 0.0
        for cond in self._complex_numeric_conds:
            if evaluate(cond, state):
                costs[cond] = 0.0
            else:
                costs[cond] = 1.0

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
        operator_cost: Dict[Operator, float] = {}
        poss: Dict[Expression, Set[Operator]] = {}
        while len(lp) > 0:
            lo = list(self._empty_pre_operators)
            for p in lp:
                if p in self._precondition_of:
                    lo.extend(self._precondition_of[p])
            lp = []
            new_costs: Dict[Expression, float] = {}
            for o in set(lo):
                c, l = self._cost(o.conditions, costs)
                if c is not None and (o not in operator_cost or operator_cost[o] > c):
                    operator_cost[o] = c

                    achieved_expressions = []
                    for f, e in o.effects:
                        if e == True:
                            k: Expression = (FluentNode(f),)
                        elif e == False:
                            k = (FluentNode(f), Op("not", (0,)))
                        else:
                            k = (FluentNode(f), e, Op("==", (0, 1)))
                        achieved_expressions.append((k, o.cost + c))

                    for simple_cond in self._achieved_simple_numeric_conds[o.id]:
                        if costs.get(simple_cond, None) == 0.0:
                            # condition satisfied in state
                            continue

                        rep = self._repetitions(o, simple_cond, state)
                        assert rep is not None

                        if self._heuristic_kind == HeuristicKind.HMAX:
                            if simple_cond not in poss:
                                poss[simple_cond] = set()
                            poss[simple_cond].add(o)

                            exp_cost = float(rep) * o.cost + min(  # type: ignore[operator,type-var]
                                self._cost(o.conditions, costs)[0]
                                for o in poss[simple_cond]
                            )
                        else:
                            exp_cost = float(rep) * o.cost + c
                        achieved_expressions.append((simple_cond, exp_cost))

                    for exp, exp_cost in achieved_expressions:
                        if exp in new_costs:
                            prev_exp_cost = new_costs[exp]
                        elif exp in costs:
                            prev_exp_cost = costs[exp]
                        else:
                            prev_exp_cost = None

                        if prev_exp_cost is None or exp_cost < prev_exp_cost:
                            if self._heuristic_kind == HeuristicKind.HFF:
                                reached_by[exp] = (o, l)
                            new_costs[exp] = exp_cost
                        elif (
                            prev_exp_cost == exp_cost
                            and self._heuristic_kind == HeuristicKind.HFF
                            and o.id > reached_by[exp][0].id
                        ):
                            reached_by[exp] = (o, l)

            lp = list(new_costs.keys())
            costs.update(new_costs)

        if reachability_analysis:
            return None, list(operator_cost.keys())

        h, _ = self._cost(self._goals, costs)
        if h is None:
            return None, None

        if self._heuristic_kind != HeuristicKind.HFF:
            eh, _ = self._cost(self._extra_goals, costs)
            assert eh is not None

            if self._heuristic_kind == HeuristicKind.HMAX:
                res = max(h, eh)
            else:
                res = h + eh

            return res, None

        res = 0
        for a, (j, _) in state.todo.items():
            res += len(self._events[a]) - j

        if h == 0.0:
            return float(res), None

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

        return float(res), None

    def _achieves(self, operator: Operator, simple_condition: Expression) -> bool:
        """Check whether an operator achieves a given simple numeric condition.

        The check considers:
        - If the operator has a constant assignment or complex effect on any of the
        condition's fluents, the condition is considered achieved.
        - Otherwise, it computes the net effect of the operator on the
        condition. If the net effect is negative, the condition is potentially
        achieved.

        Args:
            operator: The operator whose effects are being evaluated.
            simple_condition: A simple numeric condition expression.

        Returns:
            True if the operator achieves the condition; otherwise, False.
        """

        fluents, weights = self._simple_numeric_conds[simple_condition]
        net_effect = 0.0
        for f, w in zip(fluents, weights):
            if (
                f in operator.constant_assign_effects
                or f in operator.complex_numeric_effects
            ):
                return True
            if f in operator.constant_increase_effects:
                k = operator.constant_increase_effects[f]
                net_effect += w * k

        if net_effect < 0.0 and net_effect > self._max_net_effect:
            self._max_net_effect = net_effect

        return net_effect < 0.0

    def _repetitions(
        self, operator: Operator, simple_condition: Expression, state: State
    ) -> Optional[int]:
        """Estimate how many applications of an operator are needed to satisfy a simple numeric condition.

        This method computes the minimum number of times `operator` must be applied
        to a given `state` for the `simple_condition` to become satisfied.

        The computation follows these rules:
        - If the operator has a constant assignment or complex effect on any fluent
        in the condition, return 1, assuming one application is sufficient.
        - Otherwise, compute the net effect of the operator on the condition and
        return the minimum number of repetitions needed.

        Args:
            operator: The operator whose effects are being evaluated.
            simple_condition: A simple numeric condition expression.
            state: The state on which the condition is evaluated.

        Returns:
            The minimum number of operator applications required to satisfy the
            condition, 0 if already satisfied, 1 if a constant/complex effect applies,
            or `None` if the condition cannot be satisfied.
        """

        fluents, weights = self._simple_numeric_conds[simple_condition]
        v = weights[-1]
        for f, w in zip(fluents, weights):
            v += w * state.get_value(f)  # type: ignore[operator]

        if v <= 0.0:
            # condition satisfied in state
            return 0

        for f in fluents:
            if (
                f in operator.constant_assign_effects
                or f in operator.complex_numeric_effects
            ):
                return 1

        net_effect = 0.0
        for f, w in zip(fluents, weights):
            if f in operator.constant_increase_effects:
                k = operator.constant_increase_effects[f]
                net_effect += w * k

        if net_effect >= 0.0:
            return None

        return math.ceil(-v / net_effect)

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
    actions: List[Action],
    fluent_types: List[str],
    objects: Dict[str, List[str]],
    events: Dict[Action, List[Tuple[Timing, Event]]],
    goals: Expression,
    internal_caching: bool,
    cache_value_in_state: bool,
    disable_numeric_reasoning: bool = False,
) -> DeleteRelaxationHeuristic:
    return DeleteRelaxationHeuristic(
        actions,
        fluent_types,
        objects,
        events,
        goals,
        HeuristicKind.HFF,
        internal_caching,
        cache_value_in_state,
        disable_numeric_reasoning,
    )


def HAdd(
    actions: List[Action],
    fluent_types: List[str],
    objects: Dict[str, List[str]],
    events: Dict[Action, List[Tuple[Timing, Event]]],
    goals: Expression,
    internal_caching: bool,
    cache_value_in_state: bool,
    disable_numeric_reasoning: bool = False,
) -> DeleteRelaxationHeuristic:
    return DeleteRelaxationHeuristic(
        actions,
        fluent_types,
        objects,
        events,
        goals,
        HeuristicKind.HADD,
        internal_caching,
        cache_value_in_state,
        disable_numeric_reasoning,
    )


def HMax(
    actions: List[Action],
    fluent_types: List[str],
    objects: Dict[str, List[str]],
    events: Dict[Action, List[Tuple[Timing, Event]]],
    goals: Expression,
    internal_caching: bool,
    cache_value_in_state: bool,
    disable_numeric_reasoning: bool = False,
) -> DeleteRelaxationHeuristic:
    return DeleteRelaxationHeuristic(
        actions,
        fluent_types,
        objects,
        events,
        goals,
        HeuristicKind.HMAX,
        internal_caching,
        cache_value_in_state,
        disable_numeric_reasoning,
    )


class HMaxExplicit(Heuristic):
    def __init__(
        self,
        actions: List[Action],
        fluent_types: List[str],
        objects: Dict[str, List[str]],
        events: Dict[Action, List[Tuple[Timing, Event]]],
        goals: Expression,
        internal_caching: bool,
        cache_value_in_state: bool,
    ):
        super().__init__(cache_value_in_state)
        self._actions = actions
        self._fluent_types = fluent_types
        self._objects = objects
        self._events = events
        self._operators: List[OperatorHmax] = []
        self._extra_fluents: Dict[Action, List[int]] = {}
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
                conditions: List[Tuple[ExpressionNode, ...]] = [cond]
                for c in get_event_conditions(e):
                    if len(c) > 0 and c != (True,):
                        conditions.extend(split_expression(c))
                cond = (FluentNode(f),)
                if (False,) not in conditions:
                    self._operators.append(
                        OperatorHmax(a, tuple(conditions), tuple(effects), 1.0)
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

        self._internal_caching: Optional[
            Dict[Tuple[Union[bool, int, Fraction, str, None], ...], Optional[float]]
        ] = ({} if internal_caching else None)

    @property
    def name(self) -> str:
        return "hmax_explicit"

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
                state.todo.get(action, (None, None))[0] for action in self._actions
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
        for action in self._events:
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
