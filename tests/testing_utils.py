from tamerlite.core import *
import random
from tamerlite.core import (
    make_bool_constant_node,
    make_fluent_node,
    make_int_constant_node,
    make_operator_node,
    make_rational_constant_node,
)


def construct_numeric_exp_rec(offset=0, depth=0) -> tuple:
    kinds = ["int", "rational", "fluent", "+", "-", "*", "/"]

    if depth == 0:
        r = random.randint(0, 1)
    else:
        r = random.randint(0, len(kinds) - 1)

    kind = kinds[r]
    if kind == "int":
        return (make_int_constant_node(random.randint(1, 100)),)
    elif kind == "rational":
        return (
            make_rational_constant_node(random.randint(1, 100), random.randint(1, 100)),
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
