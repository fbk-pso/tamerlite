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

from typing import List

from unified_planning.model import FNode
from unified_planning.model.walkers import DagWalker, Dnf

from tamerlite.core import Expression
from tamerlite.core import (
    make_bool_constant_node,
    make_fluent_node,
    make_int_constant_node,
    make_object_node,
    make_operator_node,
    make_rational_constant_node,
    shift_expression,
)


class Converter(DagWalker):
    def __init__(self, problem):
        DagWalker.__init__(self)
        self.static_fluents = problem.get_static_fluents()

    def convert(self, expression: 'FNode') -> Expression:
        """Converts the given expression."""
        w = Dnf(expression.environment)
        return self.walk(w.get_dnf_expression(expression))

    def walk_and(self, expression: 'FNode',
                 args: List[Expression]) -> Expression:
        if len(args) == 0:
            return (True, )
        elif len(args) == 1:
            return args[0]
        else:
            res = args[0]
            l = len(res)-1
            operands = [l]
            for i in range(1, len(args)):
                res += tuple(shift_expression(args[i], l+1))
                l += len(args[i])
                operands.append(l)
            res += (make_operator_node("and", tuple(operands)), )
            return res

    def walk_or(self, expression: 'FNode',
                args: List[Expression]) -> Expression:
        if len(args) == 0:
            return (False, )
        elif len(args) == 1:
            return args[0]
        else:
            raise NotImplementedError

    def walk_not(self, expression: 'FNode',
                 args: List[Expression]) -> Expression:
        assert len(args) == 1
        return args[0] + (make_operator_node("not", (len(args[0])-1, )), )

    def walk_plus(self, expression: 'FNode',
                  args: List[Expression]) -> Expression:
        assert len(args) >= 2
        res = args[0]
        l = len(res)-1
        operands = [l]
        for i in range(1, len(args)):
            res += tuple(shift_expression(args[i], l+1))
            l += len(args[i])
            operands.append(l)
        res += (make_operator_node("+", tuple(operands)), )
        return res

    def walk_minus(self, expression: 'FNode',
                   args: List[Expression]) -> Expression:
        assert len(args) == 2
        return args[0] + tuple(shift_expression(args[1], len(args[0]))) + (make_operator_node("-", (len(args[0])-1, len(args[0])+len(args[1])-1)), )

    def walk_times(self, expression: 'FNode',
                   args: List[Expression]) -> Expression:
        assert len(args) >= 2
        res = args[0]
        l = len(res)-1
        operands = [l]
        for i in range(1, len(args)):
            res += tuple(shift_expression(args[i], l+1))
            l += len(args[i])
            operands.append(l)
        res += (make_operator_node("*", tuple(operands)), )
        return res

    def walk_div(self, expression: 'FNode',
                 args: List[Expression]) -> Expression:
        assert len(args) == 2
        return args[0] + tuple(shift_expression(args[1], len(args[0]))) + (make_operator_node("/", (len(args[0])-1, len(args[0])+len(args[1])-1)), )

    def walk_le(self, expression: 'FNode',
                args: List[Expression]) -> Expression:
        assert len(args) == 2
        return args[0] + tuple(shift_expression(args[1], len(args[0]))) + (make_operator_node("<=", (len(args[0])-1, len(args[0])+len(args[1])-1)), )

    def walk_lt(self, expression: 'FNode',
                args: List[Expression]) -> Expression:
        assert len(args) == 2
        return args[0] + tuple(shift_expression(args[1], len(args[0]))) + (make_operator_node("<", (len(args[0])-1, len(args[0])+len(args[1])-1)), )

    def walk_equals(self, expression: 'FNode',
                    args: List[Expression]) -> Expression:
        assert len(args) == 2
        if (not expression.arg(0).is_fluent_exp() or expression.arg(0).fluent() in self.static_fluents) and expression.arg(1).is_fluent_exp():
            a0 = args[1]
            a1 = args[0]
        else:
            a0 = args[0]
            a1 = args[1]
        return a0 + tuple(shift_expression(a1, len(a0))) + (make_operator_node("==", (len(a0)-1, len(a0)+len(a1)-1)), )

    def walk_fluent_exp(self, expression: 'FNode',
                        args: List[Expression]) -> Expression:
        return (make_fluent_node(str(expression)), )

    def walk_object_exp(self, expression: 'FNode',
                        args: List[Expression]) -> Expression:
        assert len(args) == 0
        return (make_object_node(str(expression)), )

    def walk_bool_constant(self, expression: 'FNode',
                           args: List[Expression]) -> Expression:
        assert len(args) == 0
        return (make_bool_constant_node(expression.is_true()), )

    def walk_real_constant(self, expression: 'FNode',
                           args: List[Expression]) -> Expression:
        assert len(args) == 0
        v = expression.constant_value()
        return (make_rational_constant_node(v.numerator, v.denominator), )

    def walk_int_constant(self, expression: 'FNode',
                          args: List[Expression]) -> Expression:
        assert len(args) == 0
        return (make_int_constant_node(expression.constant_value()), )

    def walk_implies(self, expression: 'FNode',
                     args: List[Expression]) -> Expression:
        raise NotImplementedError

    def walk_iff(self, expression: 'FNode',
                 args: List[Expression]) -> Expression:
        raise NotImplementedError

    def walk_param_exp(self, expression: 'FNode',
                       args: List[Expression]) -> Expression:
        raise NotImplementedError
