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


from collections.abc import Callable
from typing import Any

from unified_planning.model import FNode, InterpretedFunction, Object, Problem
from unified_planning.model.walkers import DagWalker

from tamerlite.core import (
    Expression,
    make_bool_constant_node,
    make_fluent_node,
    make_int_constant_node,
    make_interpreted_function_node,
    make_object_node,
    make_operator_node,
    make_rational_constant_node,
    shift_expression,
)


def _unresolvable_fluent_message(expression: FNode) -> str:
    """Explain why `expression` names no grounded fluent of the problem."""
    env = expression.environment
    ifuns = set(env.interpreted_functions_extractor.get(expression))
    nested_fluents = set(env.free_vars_extractor.get(expression))
    nested_fluents.discard(expression)
    if ifuns:
        cause = f"contains interpreted functions in its arguments: {ifuns}"
    elif nested_fluents:
        cause = f"contains other fluents in its arguments: {nested_fluents}"
    else:
        cause = "does not name a grounded fluent of the problem"
    return (
        f"TamerLite does not support the fluent expression `{expression}`: it "
        f"{cause}, so it cannot be resolved to a single grounded fluent at "
        "encoding time."
    )


class Converter(DagWalker):
    def __init__(
        self,
        problem: Problem,
        fluent_ids: dict[str, int],
        object_ids: dict[str, int],
        objects_by_id: list[Object],
        if_cache: dict[InterpretedFunction, dict[tuple, Any]] | None = None,
    ):
        DagWalker.__init__(self)
        self._fluent_ids = fluent_ids
        self._object_ids = object_ids
        self._objects_by_id = objects_by_id
        self.static_fluents = problem.get_static_fluents()
        self._if_wrappers: dict[InterpretedFunction, Callable] = {}
        self._if_cache: dict[InterpretedFunction, dict[tuple, Any]] = (
            if_cache if if_cache is not None else {}
        )

    def convert(self, expression: FNode) -> Expression:
        """Converts the given expression."""
        result: Expression = self.walk(expression)
        return result

    def walk_and(self, expression: FNode, args: list[Expression]) -> Expression:
        if len(args) == 0:
            return (True,)
        elif len(args) == 1:
            return args[0]
        else:
            res = args[0]
            offset = len(res) - 1
            operands = [offset]
            for i in range(1, len(args)):
                res += tuple(shift_expression(args[i], offset + 1))
                offset += len(args[i])
                operands.append(offset)
            res += (make_operator_node("and", tuple(operands)),)
            return res

    def walk_or(self, expression: FNode, args: list[Expression]) -> Expression:
        if len(args) == 0:
            return (False,)
        elif len(args) == 1:
            return args[0]
        else:
            res = args[0]
            offset = len(res) - 1
            operands = [offset]
            for i in range(1, len(args)):
                res += tuple(shift_expression(args[i], offset + 1))
                offset += len(args[i])
                operands.append(offset)
            res += (make_operator_node("or", tuple(operands)),)
            return res

    def walk_not(self, expression: FNode, args: list[Expression]) -> Expression:
        assert len(args) == 1
        return args[0] + (make_operator_node("not", (len(args[0]) - 1,)),)

    def walk_plus(self, expression: FNode, args: list[Expression]) -> Expression:
        assert len(args) >= 2
        res = args[0]
        offset = len(res) - 1
        operands = [offset]
        for i in range(1, len(args)):
            res += tuple(shift_expression(args[i], offset + 1))
            offset += len(args[i])
            operands.append(offset)
        res += (make_operator_node("+", tuple(operands)),)
        return res

    def walk_minus(self, expression: FNode, args: list[Expression]) -> Expression:
        assert len(args) == 2
        return (
            args[0]
            + tuple(shift_expression(args[1], len(args[0])))
            + (
                make_operator_node(
                    "-", (len(args[0]) - 1, len(args[0]) + len(args[1]) - 1)
                ),
            )
        )

    def walk_times(self, expression: FNode, args: list[Expression]) -> Expression:
        assert len(args) >= 2
        res = args[0]
        offset = len(res) - 1
        operands = [offset]
        for i in range(1, len(args)):
            res += tuple(shift_expression(args[i], offset + 1))
            offset += len(args[i])
            operands.append(offset)
        res += (make_operator_node("*", tuple(operands)),)
        return res

    def walk_div(self, expression: FNode, args: list[Expression]) -> Expression:
        assert len(args) == 2
        return (
            args[0]
            + tuple(shift_expression(args[1], len(args[0])))
            + (
                make_operator_node(
                    "/", (len(args[0]) - 1, len(args[0]) + len(args[1]) - 1)
                ),
            )
        )

    def walk_le(self, expression: FNode, args: list[Expression]) -> Expression:
        assert len(args) == 2
        return (
            args[0]
            + tuple(shift_expression(args[1], len(args[0])))
            + (
                make_operator_node(
                    "<=", (len(args[0]) - 1, len(args[0]) + len(args[1]) - 1)
                ),
            )
        )

    def walk_lt(self, expression: FNode, args: list[Expression]) -> Expression:
        assert len(args) == 2
        return (
            args[0]
            + tuple(shift_expression(args[1], len(args[0])))
            + (
                make_operator_node(
                    "<", (len(args[0]) - 1, len(args[0]) + len(args[1]) - 1)
                ),
            )
        )

    def walk_equals(self, expression: FNode, args: list[Expression]) -> Expression:
        assert len(args) == 2
        if (
            not expression.arg(0).is_fluent_exp()
            or expression.arg(0).fluent() in self.static_fluents
        ) and expression.arg(1).is_fluent_exp():
            a0 = args[1]
            a1 = args[0]
        else:
            a0 = args[0]
            a1 = args[1]
        return (
            a0
            + tuple(shift_expression(a1, len(a0)))
            + (make_operator_node("==", (len(a0) - 1, len(a0) + len(a1) - 1)),)
        )

    def walk_fluent_exp(self, expression: FNode, args: list[Expression]) -> Expression:
        fluent = str(expression)
        try:
            return (make_fluent_node(self._fluent_ids[fluent]),)
        except KeyError:
            raise NotImplementedError(
                _unresolvable_fluent_message(expression)
            ) from None

    def walk_object_exp(self, expression: FNode, args: list[Expression]) -> Expression:
        assert len(args) == 0
        return (make_object_node(self._object_ids[expression.object().name]),)

    def walk_bool_constant(
        self, expression: FNode, args: list[Expression]
    ) -> Expression:
        assert len(args) == 0
        return (make_bool_constant_node(expression.is_true()),)

    def walk_real_constant(
        self, expression: FNode, args: list[Expression]
    ) -> Expression:
        assert len(args) == 0
        v = expression.constant_value()
        return (make_rational_constant_node(v.numerator, v.denominator),)

    def walk_int_constant(
        self, expression: FNode, args: list[Expression]
    ) -> Expression:
        assert len(args) == 0
        return (make_int_constant_node(expression.int_constant_value()),)

    def _get_interpreted_function_wrapper(
        self, interpreted_function: InterpretedFunction
    ) -> Callable:
        """Returns the single, memoizing wrapper callable for
        `interpreted_function`, shared by every occurrence of the same
        interpreted function across the whole problem.

        Sharing the wrapper keeps `InterpretedFunctionNode` equality/hashing
        meaningful within one encoding. The wrapper's *result* cache
        (`self._if_cache`) is a separate concern and may be shared across
        several Converters re-encoding the same problem (see
        `TamerLite._get_solutions_with_params`, which re-encodes on every
        anytime iteration): it's keyed by the already-unwrapped,
        table-agnostic argument values (real `Object`s, not this Converter's
        internal ids), so a shared cache stays correct even if two Converters
        happen to number their objects differently. Only the id<->`Object`
        translation actually depends on that numbering, so it always runs
        fresh against *this* Converter's own `_objects_by_id`/`_object_ids`,
        on every call, cache hit or not -- never against whichever Converter
        first populated a shared cache. This assumes interpreted functions
        are deterministic and side-effect-free.
        """
        cached = self._if_wrappers.get(interpreted_function)
        if cached is not None:
            return cached

        return_type = interpreted_function.return_type
        # Object-typed parameters/return values are exposed to `evaluate` as
        # internal `ObjectNode`s, but the real callable expects/returns actual
        # UP `Object`s -- translate both directions.
        object_params = tuple(
            p.type.is_user_type() for p in interpreted_function.signature
        )
        wraps_result = return_type.is_user_type()
        result_cache = self._if_cache.setdefault(interpreted_function, {})

        def wrapper(*call_args):
            if any(object_params):
                real_args = tuple(
                    self._objects_by_id[a.object] if is_obj else a
                    for a, is_obj in zip(call_args, object_params, strict=True)
                )
            else:
                real_args = call_args
            if real_args in result_cache:
                raw_result = result_cache[real_args]
            else:
                raw_result = interpreted_function.function(*real_args)
                result_cache[real_args] = raw_result
            if wraps_result:
                return make_object_node(self._object_ids[raw_result.name])
            return raw_result

        self._if_wrappers[interpreted_function] = wrapper
        return wrapper

    def walk_interpreted_function_exp(
        self, expression: FNode, args: list[Expression]
    ) -> Expression:
        # `Converter` only sees interpreted-function calls with at least one
        # non-constant argument: UP's `Grounder` already
        # folds any call whose arguments are all constant -- including
        # static fluents, resolved to their initial value -- by calling the
        # real Python function during grounding. What remains here
        # must be re-evaluated at search time.
        interpreted_function = expression.interpreted_function()
        return_type = interpreted_function.return_type
        if return_type.is_bool_type():
            return_type_str = "bool"
        elif return_type.is_int_type():
            return_type_str = "int"
        elif return_type.is_real_type():
            return_type_str = "real"
        elif return_type.is_user_type():
            return_type_str = "object"
        else:
            raise NotImplementedError(
                f"Unsupported interpreted function return type: {return_type}"
            )

        function = self._get_interpreted_function_wrapper(interpreted_function)

        if len(args) == 0:
            return (make_interpreted_function_node(function, return_type_str, ()),)
        res = args[0]
        offset = len(res) - 1
        operands = [offset]
        for i in range(1, len(args)):
            res += tuple(shift_expression(args[i], offset + 1))
            offset += len(args[i])
            operands.append(offset)
        res += (
            make_interpreted_function_node(function, return_type_str, tuple(operands)),
        )
        return res

    def walk_implies(self, expression: FNode, args: list[Expression]) -> Expression:
        raise NotImplementedError

    def walk_iff(self, expression: FNode, args: list[Expression]) -> Expression:
        raise NotImplementedError

    def walk_param_exp(self, expression: FNode, args: list[Expression]) -> Expression:
        raise NotImplementedError
