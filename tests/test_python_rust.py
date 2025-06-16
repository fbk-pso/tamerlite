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
import tamerlite.core
from tamerlite.core.heuristics import Heuristic
from tamerlite.core import HFF, HAdd, HMax, HMaxNumeric
from tamerlite.core.search_space import SearchSpace
from tamerlite.encoder import Encoder
import tamerlite.encoder
import tamerlite.engine

import problems_generator
import pytest
import importlib
import os
import types


@pytest.fixture
def problems():
    test_problems = [problems_generator.get_problem_logistics(1, 1, 4, 2)]

    up_example_problems = list(
        unified_planning.test.examples.get_example_problems().values()
    )
    up_test_problems = list(up_test_cases.builtin.get_test_cases().values())
    for test_case in up_example_problems + up_test_problems:
        if test_case.solvable and tamerlite.engine.TamerLite.supports(
            test_case.problem.kind
        ):
            test_problems.append(test_case.problem)

    return test_problems


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
    )


def max_generated_states(problem):
    if problem.name in [
        "nonlinear_increase_effects",
        "constant_increase_effect",
        "constant_decrease_effect",
    ]:
        return 2
    if problem.name in ["constant_increase_effect_2", "constant_decrease_effect_2"]:
        return 4
    return 1000


def generate_states(ss: SearchSpace, state, num_states: int):
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
        search_kind = "wastar"
        for heuristic in ["hff", "hadd", "hmax", "hmax_numeric"]:
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
                        )
                        results.append(res)
                        with PlanValidator(problem_kind=problem.kind) as v:
                            assert v.validate(problem, res.plan)

            check_metrics_equality(results)


def test_heuristic_values(problems):
    for problem in problems:
        values = {}
        for disable_rustamer in [True, False]:
            reload_tamerlite(disable_rustamer)
            from tamerlite.core import HFF, HAdd, HMax, HMaxNumeric

            with problem.environment.factory.Compiler(
                compilation_kind="GROUNDING", problem_kind=problem.kind
            ) as compiler:
                compilation_res = compiler.compile(problem)
            new_problem = compilation_res.problem
            encoder = Encoder(new_problem)
            ss: SearchSpace = encoder.search_space
            init_state = ss.initial_state()

            states = generate_states(
                ss, init_state, num_states=max_generated_states(problem)
            )
            for heuristic_class, heuristic_name in [
                (HFF, "HFF"),
                (HAdd, "HAdd"),
                (HMax, "HMax"),
                (HMaxNumeric, "HMaxNumeric"),
            ]:
                for internal_caching in [True, False]:
                    heuristic: Heuristic = heuristic_class(
                        encoder.fluents,
                        encoder.objects,
                        encoder.events,
                        encoder.goal,
                        internal_caching=internal_caching,
                        cache_value_in_state=False
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


def test_search_algorithms(problems):
    for problem in problems:
        heuristic = "hff"
        for search_kind in ["wastar", "astar", "gbfs", "dfs", "bfs", "ehs"]:
            results = []
            for disable_rustamer in [True, False]:
                if skip(problem, search_kind, heuristic, disable_rustamer, True):
                    continue

                reload_tamerlite(disable_rustamer)
                search = tamerlite.SearchParams(search=search_kind, heuristic=heuristic)

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
                [
                    tamerlite.SearchParams(search="wastar", heuristic="hff"),
                    tamerlite.SearchParams(search="astar", heuristic="hadd"),
                    tamerlite.SearchParams(search="bfs", heuristic="hmax"),
                ]
            )
            with OneshotPlanner(name="tamerlite", params={"search": search}) as planner:
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
            encoder = Encoder(new_problem)
            ss: tamerlite.core.SearchSpace = encoder.search_space

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
            actions1 = list(map(lambda e: e[0], state1.path))
            actions2 = list(map(lambda e: e[0], state2.path))
            assert actions1 == actions2

            assert len(state1.todo) == len(state2.todo)
            for k in state1.todo:
                assert k in state2.todo
                assert state1.todo[k][0] == state2.todo[k][0]

            assert state1.g == state2.g
