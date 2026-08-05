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


from unified_planning.model import FNode, Object, Problem
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
    ):
        DagWalker.__init__(self)
        self._fluent_ids = fluent_ids
        self._object_ids = object_ids
        self._objects_by_id = objects_by_id
        self.static_fluents = problem.get_static_fluents()

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
        function = interpreted_function.function
        if return_type.is_bool_type():
            return_type_str = "bool"
        elif return_type.is_int_type():
            return_type_str = "int"
        elif return_type.is_real_type():
            return_type_str = "real"
        elif return_type.is_user_type():
            return_type_str = "object"

        # Object-typed parameters/return values are exposed to `evaluate` as
        # internal `ObjectNode`s (see `search_space.evaluate`), but the real
        # callable expects/returns actual UP `Object`s -- wrap it to translate
        # both directions, mirroring `walk_object_exp`'s name <-> id lookup.
        # Only installed when actually needed, so plain bool/int/real IFs
        # keep storing `interpreted_function.function` verbatim.
        object_params = tuple(
            p.type.is_user_type() for p in interpreted_function.signature
        )
        wraps_result = return_type.is_user_type()
        if any(object_params) or wraps_result:

            def function(*call_args):
                if any(object_params):
                    call_args = tuple(
                        self._objects_by_id[a.object] if is_obj else a
                        for a, is_obj in zip(call_args, object_params, strict=True)
                    )
                r = interpreted_function.function(*call_args)
                if wraps_result:
                    return make_object_node(self._object_ids[r.name])
                return r

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
