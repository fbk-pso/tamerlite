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

import ast
import random
from typing import List

from unified_planning.model import Problem

from tamerlite.core import (
    Expression,
    make_bool_constant_node,
    make_fluent_node,
    make_int_constant_node,
    make_object_node,
    make_operator_node,
    make_rational_constant_node,
)


def compile_problem(problem: Problem):
    with problem.environment.factory.Compiler(
        compilation_kind="UNDEFINED_INITIAL_NUMERIC_REMOVING",
        problem_kind=problem.kind,
    ) as compiler:
        compilation_res = compiler.compile(problem)
        undefined_map_back_action_instance = compilation_res.map_back_action_instance
        problem = compilation_res.problem

    with problem.environment.factory.Compiler(
        compilation_kind="GROUNDING", problem_kind=problem.kind
    ) as compiler:
        compilation_res = compiler.compile(problem)
        ground_map_back_action_instance = compilation_res.map_back_action_instance
        ground_problem = compilation_res.problem
        lifted_problem = problem

    map_back_action_instance = lambda ai: undefined_map_back_action_instance(
        ground_map_back_action_instance(ai)
    )

    return lifted_problem, ground_problem, map_back_action_instance


def is_strictly_increasing(l: List):
    for i in range(len(l) - 1):
        if l[i] >= l[i + 1]:
            return False
    return True


def is_temporal_problem(problem: Problem):
    return problem.kind.has_continuous_time() or problem.kind.has_discrete_time()


def is_numeric_problem(problem: Problem):
    return problem.kind.has_int_fluents() or problem.kind.has_real_fluents()


def construct_numeric_exp_rec(offset=0, depth=0) -> tuple:
    kinds = ["int", "rational", "fluent", "+", "-", "*", "/"]

    if depth == 0:
        r = random.randint(0, 1)
    else:
        r = random.randint(0, len(kinds) - 1)

    kind = kinds[r]
    if kind == "int":
        return (make_int_constant_node(random.randint(-10, 10)),)
    elif kind == "rational":
        return (
            make_rational_constant_node(
                random.choice([1, -1]) * random.randint(1, 10), random.randint(1, 10)
            ),
        )
    elif kind == "fluent":
        return (make_fluent_node(1),)

    num_operands = 2
    if kind in ["+", "*"]:
        num_operands = random.randint(2, 4)

    res = ()
    operands = []
    for i in range(num_operands):
        sub_exp = construct_numeric_exp_rec(offset + len(res), depth - 1)
        res += sub_exp
        operands.append(offset + len(res) - 1)
    res += (make_operator_node(kind, tuple(operands)),)
    return res


def construct_exp_rec(offset=0, depth=0) -> tuple:
    kinds = ["bool", "fluent", "and", "or", "not", "==", "<=", "<"]

    if depth == 0:
        r = random.randint(0, 1)
    else:
        r = random.randint(0, len(kinds) - 1)

    kind = kinds[r]
    if kind == "bool":
        return (make_bool_constant_node(bool(random.randint(0, 1))),)
    elif kind == "fluent":
        return (make_fluent_node(0),)

    num_operands = 2
    if kind == "not":
        num_operands = 1
    elif kind in ["and", "or"]:
        num_operands = random.randint(2, 5)

    res = ()
    operands = []
    for i in range(num_operands):
        if kind in ["<=", "<", "=="]:
            sub_exp = construct_numeric_exp_rec(offset + len(res), depth - 1)
        else:
            sub_exp = construct_exp_rec(offset + len(res), depth - 1)
        res += sub_exp
        operands.append(offset + len(res) - 1)
    res += (make_operator_node(kind, tuple(operands)),)
    return res


def construct_expressions(
    num_expressions: int, max_depth: int, random_seed: int = 0
) -> List[Expression]:
    random.seed(random_seed)
    return [
        construct_exp_rec(offset=0, depth=max_depth) for _ in range(num_expressions)
    ]


def parse_expression_rec(node):
    if isinstance(node, ast.Tuple):
        return tuple(parse_expression_rec(e) for e in node.elts)

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return make_int_constant_node(-node.operand.value)

    if isinstance(node, ast.Constant):
        value = node.value
        if isinstance(value, bool):
            return make_bool_constant_node(value)
        if isinstance(value, int):
            return make_int_constant_node(value)
        if isinstance(value, str):
            return make_object_node(value)
        raise TypeError(f"Unsupported literal: {value}")

    if isinstance(node, ast.Name):
        if node.id in ("True", "False"):
            return make_bool_constant_node(node.id == "True")
        return make_object_node(node.id)

    if isinstance(node, ast.Call):
        fname = node.func.id

        if fname == "Fraction":
            a = node.args[0]
            if isinstance(a, ast.UnaryOp) and isinstance(a.op, ast.USub):
                a = -a.operand.value
            else:
                a = a.value
            b = node.args[1].value
            return make_rational_constant_node(a, b)

        if fname == "FluentNode":
            kwargs = {kw.arg: kw.value.value for kw in node.keywords}
            return make_fluent_node(kwargs["fluent"])

        if fname == "OperatorNode":
            kwargs = {kw.arg: kw.value for kw in node.keywords}
            kind = kwargs["kind"].value
            operands = tuple(o.value for o in kwargs["operands"].elts)
            return make_operator_node(kind, operands)

    raise ValueError(f"Unknown node: {fname}")


def parse_expression(exp: str):
    tree = ast.parse(exp, mode="eval")
    return parse_expression_rec(tree.body)
