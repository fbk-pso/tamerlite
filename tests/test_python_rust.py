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

import unified_planning
from unified_planning.shortcuts import *
from unified_planning.engines import PlanGenerationResult, PlanGenerationResultStatus
import unified_planning.test
import unified_planning.test.examples
import up_test_cases.builtin

import tamerlite
from tamerlite.core.heuristics import Heuristic
from tamerlite.core import HFF, HAdd, HMax, HMaxExplicit, CustomHeuristic
from tamerlite.core import simplify
from tamerlite.core.search_space import SearchSpaceABC
from tamerlite.encoder import Encoder
import tamerlite.encoder
import tamerlite.engine

import problems_generator
import testing_utils
import pytest
import importlib
import os
import types
from functools import partial

env = get_environment()
env.factory.add_engine("tamerlite", "tamerlite.engine", "TamerLite")


@pytest.fixture
def problems():
    test_problems = [
        problems_generator.get_problem_logistics(1, 1, 4, 2),
        problems_generator.get_problem_numeric(),
    ]

    up_example_problems = list(
        unified_planning.test.examples.get_example_problems().values()
    )
    up_test_problems = list(up_test_cases.builtin.get_test_cases().values())
    for test_case in up_example_problems + up_test_problems:
        if test_case.solvable and tamerlite.engine.TamerLite.supports(
            test_case.problem.kind
        ):
            test_problems.append(test_case.problem)

    names = set()
    for problem in test_problems:
        if problem.name in names:
            # name duplicated
            i = 0
            new_name = problem.name
            while new_name in names:
                new_name = problem.name + str(i)
                i += 1
            problem.name = new_name
        names.add(problem.name)

    return test_problems


@pytest.fixture
def expressions():
    import pathlib
    import json

    data_path = os.path.join(
        pathlib.Path(__file__).parent.resolve(),
        "test_python_rust",
        "test_simplify_fixed_expressions.json",
    )
    with open(data_path) as f:
        data = json.load(f)

    expressions = []
    for e in data["expressions"]:
        expressions.append((e["exp"], e["simplified_exp"]))
    return expressions


def reload_package(package):
    assert hasattr(package, "__package__")
    fn = package.__file__
    fn_dir = os.path.dirname(fn) + os.sep
    module_visit = {fn}
    del fn

    def reload_recursive_ex(module):
        importlib.reload(module)

        for module_child in vars(module).values():
            if isinstance(module_child, types.ModuleType):
                fn_child = getattr(module_child, "__file__", None)
                if (fn_child is not None) and fn_child.startswith(fn_dir):
                    if fn_child not in module_visit:
                        module_visit.add(fn_child)
                        reload_recursive_ex(module_child)

    return reload_recursive_ex(package)


def reload_tamerlite(disable_rustamer: bool):
    os.environ["DISABLE_RUSTAMER"] = str(disable_rustamer)
    reload_package(tamerlite)


def skip(problem, search, heuristic, disable_rustamer, internal_heuristic_cache):
    return (
        (problem.name == "robot_fluent_of_user_type" and search == "dfs")
        or (problem.name == "robot_loader" and search == "dfs")
        or (problem.name == "robot_loader_mod" and search == "dfs")
        or (problem.name == "robot_loader_adv" and search == "dfs")
        or (problem.name == "robot_fluent_of_user_type_with_int_id" and search == "dfs")
        or (problem.name == "depots_p01" and search in ["dfs", "bfs"])
        or (problem.name == "RoboLogistics" and search == "dfs")
        or (problem.name == "NumericProblem" and search == "dfs")
    )


def max_generated_states(problem):
    if problem.name in [
        "nonlinear_increase_effects",
        "constant_increase_effect",
        "constant_decrease_effect",
        "disjunctive_linear_conditions",
    ]:
        return 2
    if problem.name in ["constant_increase_effect_2", "constant_decrease_effect_2"]:
        return 4
    return 1000


def generate_states(ss: SearchSpaceABC, state, num_states: int):
    states = [state]
    i = 0
    while i < len(states) and len(states) < num_states:
        state = states[i]
        states += list(ss.get_successor_states(state))
        i += 1
    return states


def check_metrics_equality(results: List[PlanGenerationResult]):
    for i in range(len(results) - 1):
        res1: PlanGenerationResult = results[i]
        res2: PlanGenerationResult = results[i + 1]
        assert len(res1.metrics) == len(res2.metrics)
        assert int(res1.metrics["expanded_states"]) == int(
            res2.metrics["expanded_states"]
        )
        assert int(res1.metrics["goal_depth"]) == int(res2.metrics["goal_depth"])


def test_heuristics(problems):
    for problem in problems:
        if testing_utils.is_temporal_problem(problem):
            weak_equality_flags = [True, False]
        else:
            weak_equality_flags = [False]

        search_kind = "wastar"
        for heuristic in [
            "hff",
            "hadd",
            "hmax",
            "hff_no_numbers",
            "hadd_no_numbers",
            "hmax_no_numbers",
            "hmax_explicit",
        ]:
            for weak_equality in weak_equality_flags:
                results = []
                for disable_rustamer in [True, False]:
                    reload_tamerlite(disable_rustamer)
                    for internal_heuristic_cache in [True, False]:
                        if skip(
                            problem,
                            search_kind,
                            heuristic,
                            disable_rustamer,
                            internal_heuristic_cache,
                        ):
                            continue

                        search = tamerlite.SearchParams(
                            search=search_kind,
                            heuristic=heuristic,
                            weight=0.8,
                            internal_heuristic_cache=internal_heuristic_cache,
                            weak_equality=weak_equality,
                        )

                        with OneshotPlanner(
                            name="tamerlite", params={"search": search}
                        ) as planner:
                            planner: tamerlite.engine.TamerLite
                            res: PlanGenerationResult = planner.solve(
                                problem, heuristic=heuristic, timeout=None
                            )
                            assert (
                                res.status
                                == PlanGenerationResultStatus.SOLVED_SATISFICING
                            )
                            results.append(res)
                            with PlanValidator(problem_kind=problem.kind) as v:
                                assert v.validate(problem, res.plan)

                check_metrics_equality(results)


def test_heuristic_fixed_values():
    problems = [
        (
            problems_generator.get_problem_logistics(1, 1, 2, 1),
            {
                "hmax_explicit": [4, 4, 3, 3, 2, 1, 0],
                "hmax": [4, 4, 3, 3, 2, 1, 0],
                "hadd": [9, 8, 5, 8, 4, 1, 0],
                "hff": [6, 6, 4, 3, 2, 1, 0],
            },
            [
                "load_at_depot_r0_plt0_p1",
                "move_r0_p1_p0",
                "make_treatment_r0_plt0_p0_t0",
                "make_treatment_r0_plt0_p0_t0",
                "load_r0_plt0_p0_t0",
                "make_treatment_r0_plt0_p0_t0",
            ],
        ),
        (
            problems_generator.get_problem_numeric(),
            {
                "hmax_explicit": [5, 4, 3, 3, 3, 2],
                "hmax": [5, 4, 3, 3, 3, 2],
                "hadd": [12, 9, 6, 5, 4, 4],
                "hff": [3, 2, 3, 3, 3, 2],
            },
            ["action1", "action2", "action3", "action3", "action4"],
        ),
    ]
    for problem, values, path in problems:
        for disable_rustamer in [False]:
            reload_tamerlite(disable_rustamer)
            from tamerlite.core import HFF, HAdd, HMax, HMaxExplicit

            with problem.environment.factory.Compiler(
                compilation_kind="GROUNDING", problem_kind=problem.kind
            ) as compiler:
                compilation_res = compiler.compile(problem)
            new_problem = compilation_res.problem
            map_back_action_instance = compilation_res.map_back_action_instance
            encoder = Encoder(new_problem, problem, map_back_action_instance)
            ss: SearchSpaceABC = encoder.search_space
            init_state = ss.initial_state()

            states = [init_state]
            for action_name in path:
                state = ss.get_successor_state(
                    states[-1], encoder.get_action(action_name)
                )
                states.append(state)

            for heuristic_class, heuristic_name in [
                (HFF, "hff"),
                (HAdd, "hadd"),
                (HMax, "hmax"),
                (HMaxExplicit, "hmax_explicit"),
            ]:
                for internal_caching in [True, False]:
                    heuristic: Heuristic = heuristic_class(
                        encoder.actions,
                        encoder.fluent_types,
                        encoder.objects,
                        encoder.events,
                        encoder.goal,
                        internal_caching=internal_caching,
                        cache_value_in_state=False,
                    )

                    for i, state in enumerate(states):
                        h_val = heuristic.eval(state, ss)
                        if h_val is not None:
                            h_val = int(h_val)
                        assert values[heuristic_name][i] == h_val


def test_heuristic_values(problems, data_regression):
    heuristic_values = {}
    for problem in problems:
        values = {}
        for disable_rustamer in [True, False]:
            reload_tamerlite(disable_rustamer)
            from tamerlite.core import HFF, HAdd, HMax, HMaxExplicit

            with problem.environment.factory.Compiler(
                compilation_kind="GROUNDING", problem_kind=problem.kind
            ) as compiler:
                compilation_res = compiler.compile(problem)
            new_problem = compilation_res.problem
            map_back_action_instance = compilation_res.map_back_action_instance
            encoder = Encoder(new_problem, problem, map_back_action_instance)
            ss: SearchSpaceABC = encoder.search_space
            init_state = ss.initial_state()

            states = generate_states(
                ss, init_state, num_states=max_generated_states(problem)
            )
            for heuristic_class, heuristic_name in [
                (HFF, "hff"),
                (HAdd, "hadd"),
                (HMax, "hmax"),
                (partial(HFF, disable_numeric_reasoning=True), "hff_no_numbers"),
                (partial(HAdd, disable_numeric_reasoning=True), "hadd_no_numbers"),
                (partial(HMax, disable_numeric_reasoning=True), "hmax_no_numbers"),
                (HMaxExplicit, "hmax_explicit"),
            ]:
                for internal_caching in [True, False]:
                    if skip(
                        problem,
                        "wastar",
                        heuristic_name,
                        disable_rustamer,
                        internal_caching,
                    ):
                        continue

                    heuristic: Heuristic = heuristic_class(
                        encoder.actions,
                        encoder.fluent_types,
                        encoder.objects,
                        encoder.events,
                        encoder.goal,
                        internal_caching=internal_caching,
                        cache_value_in_state=False,
                    )

                    if heuristic_name not in values:
                        values[heuristic_name] = []
                        for state in states:
                            h_val = heuristic.eval(state, ss)
                            if h_val is not None:
                                h_val = int(h_val)
                            values[heuristic_name].append(h_val)

                    else:
                        assert len(states) == len(values[heuristic_name])
                        for i, state in enumerate(states):
                            h_val = heuristic.eval(state, ss)
                            if h_val is not None:
                                h_val = int(h_val)
                            assert h_val == values[heuristic_name][i]

        heuristic_values[problem.name] = values

    data_regression.check(heuristic_values)


def test_custom_heuristic(problems):
    search_kind = "wastar"
    heuristic = "custom"

    def custom_heuristic(state: State):
        return 1

    for problem in problems:
        if testing_utils.is_temporal_problem(problem):
            weak_equality_flags = [True, False]
        else:
            weak_equality_flags = [False]

        for weak_equality in weak_equality_flags:
            results = []
            for disable_rustamer in [True, False]:
                reload_tamerlite(disable_rustamer)

                for internal_heuristic_cache in [True, False]:
                    if skip(
                        problem,
                        search_kind,
                        heuristic,
                        disable_rustamer,
                        internal_heuristic_cache,
                    ):
                        continue

                    search = tamerlite.SearchParams(
                        search=search_kind,
                        heuristic=heuristic,
                        weight=0.1,
                        internal_heuristic_cache=internal_heuristic_cache,
                        weak_equality=weak_equality,
                    )

                    with OneshotPlanner(
                        name="tamerlite", params={"search": search}
                    ) as planner:
                        planner: tamerlite.engine.TamerLite
                        res: PlanGenerationResult = planner.solve(
                            problem, heuristic=custom_heuristic, timeout=None
                        )
                        assert (
                            res.status == PlanGenerationResultStatus.SOLVED_SATISFICING
                        )
                        results.append(res)
                        with PlanValidator(problem_kind=problem.kind) as v:
                            assert v.validate(problem, res.plan)

            check_metrics_equality(results)


def test_search_algorithms(problems):
    for problem in problems:
        heuristic = "hff"
        for search_kind in ["wastar", "astar", "gbfs", "dfs", "bfs", "ehs"]:
            if testing_utils.is_temporal_problem(problem) and search_kind not in (
                "dfs",
                "bfs",
            ):
                weak_equality_flags = [True, False]
            else:
                weak_equality_flags = [False]

            for weak_equality in weak_equality_flags:
                results = []
                for disable_rustamer in [True, False]:
                    if skip(problem, search_kind, heuristic, disable_rustamer, True):
                        continue

                    reload_tamerlite(disable_rustamer)
                    search = tamerlite.SearchParams(
                        search=search_kind,
                        heuristic=heuristic,
                        weak_equality=weak_equality,
                    )

                    with OneshotPlanner(
                        name="tamerlite", params={"search": search}
                    ) as planner:
                        planner: tamerlite.engine.TamerLite
                        res: PlanGenerationResult = planner.solve(
                            problem, heuristic=heuristic, timeout=None
                        )
                        assert (
                            res.status == PlanGenerationResultStatus.SOLVED_SATISFICING
                            or (
                                search_kind == "ehs"
                                and res.status
                                == PlanGenerationResultStatus.UNSOLVABLE_INCOMPLETELY
                            )
                        )
                        if res.status == PlanGenerationResultStatus.SOLVED_SATISFICING:
                            results.append(res)
                            with PlanValidator(problem_kind=problem.kind) as v:
                                assert v.validate(problem, res.plan)

                check_metrics_equality(results)


def test_multiqueue_search(problems):
    for problem in problems:
        if testing_utils.is_temporal_problem(problem):
            weak_equality_flags = [True, False]
        else:
            weak_equality_flags = [False]

        for weak_equality in weak_equality_flags:
            results = []
            for disable_rustamer in [True, False]:
                if skip(
                    problem,
                    "multiqueue",
                    heuristic=None,
                    disable_rustamer=disable_rustamer,
                    internal_heuristic_cache=True,
                ):
                    continue

                reload_tamerlite(disable_rustamer)

                search = tamerlite.engine.MultiqueueParams(
                    queues=[
                        tamerlite.HeuristicParams(heuristic="hff", weight=0.8),
                        tamerlite.HeuristicParams(heuristic="hadd", weight=0.8),
                        tamerlite.HeuristicParams(heuristic="hmax", weight=0.8),
                    ],
                    weak_equality=weak_equality,
                )
                with OneshotPlanner(
                    name="tamerlite", params={"search": search}
                ) as planner:
                    res: PlanGenerationResult = planner.solve(problem, timeout=None)
                    assert res.status == PlanGenerationResultStatus.SOLVED_SATISFICING
                    results.append(res)
                    with PlanValidator(problem_kind=problem.kind) as v:
                        assert v.validate(problem, res.plan)

            check_metrics_equality(results)


def test_search_space(problems):
    for problem in problems:
        states = {}
        for disable_rustamer in [True, False]:
            reload_tamerlite(disable_rustamer)
            reload_package(tamerlite.encoder)
            from tamerlite.encoder import Encoder

            with problem.environment.factory.Compiler(
                compilation_kind="GROUNDING", problem_kind=problem.kind
            ) as compiler:
                compilation_res = compiler.compile(problem)
            new_problem = compilation_res.problem
            map_back_action_instance = compilation_res.map_back_action_instance
            encoder = Encoder(new_problem, problem, map_back_action_instance)
            ss: SearchSpaceABC = encoder.search_space

            init_state = ss.initial_state()
            l = "python" if disable_rustamer else "rust"
            states[l] = generate_states(
                ss, init_state, num_states=max_generated_states(problem)
            )

        assert len(states["python"]) == len(states["rust"])
        for i in range(len(states["python"])):
            state1 = states["python"][i]
            state2 = states["rust"][i]

            assert len(state1.path) == len(state2.path)
            actions1 = list(map(lambda e: encoder.get_action_name(e[0]), state1.path))
            actions2 = list(map(lambda e: encoder.get_action_name(e[0]), state2.path))
            assert actions1 == actions2

            assert len(state1.todo) == len(state2.todo)
            todo2 = {k.idx: v for k, v in state2.todo.items()}
            for k in state1.todo:
                assert k.idx in todo2
                assert state1.todo[k][0] == todo2[k.idx][0]

            assert state1.g == state2.g


def test_simplify():
    num_expressions = 100
    results = {True: [None] * num_expressions, False: [None] * num_expressions}
    for disable_rustamer in [True, False]:
        reload_tamerlite(disable_rustamer)
        reload_package(testing_utils)
        from tamerlite.core import simplify
        from testing_utils import (
            construct_expressions,
            parse_expression,
            is_strictly_increasing,
        )

        expressions = construct_expressions(num_expressions, max_depth=20)
        for i, exp in enumerate(expressions):
            try:
                results[disable_rustamer][i] = simplify(exp, {})
            except ZeroDivisionError:
                results[disable_rustamer][i] = "ZeroDivisionError"

            if not disable_rustamer:
                if (
                    results[True][i] == "ZeroDivisionError"
                    or results[False][i] == "ZeroDivisionError"
                ):
                    assert results[True][i] == results[False][i]

                else:
                    py_exp = results[True][i]
                    rs_exp = parse_expression(str(py_exp))
                    assert str(list(rs_exp)) == str(results[False][i])

    # verify that operands are in ascending order
    for i in range(num_expressions):
        for node in results[True][i]:
            try:
                assert is_strictly_increasing(node.operands)
            except AttributeError:
                pass


def test_simplify_fixed_expressions(expressions):
    for disable_rustamer in [True, False]:
        reload_tamerlite(disable_rustamer)
        reload_package(testing_utils)
        from tamerlite.core import simplify
        from testing_utils import parse_expression

        for exp, simplified_exp in expressions:
            exp = parse_expression(exp)
            if not disable_rustamer:
                simplified_exp = str(list(parse_expression(simplified_exp)))
            assert str(simplify(exp, {})) == simplified_exp
