from unified_planning.shortcuts import *
from unified_planning.engines import PlanGenerationResult, PlanGenerationResultStatus

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
    return [
        problems_generator.get_problem_logistics(1, 1, 4, 2),
        problems_generator.get_problem_matchcellar(3),
    ]


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
        for heuristic in ["hff", "hadd", "hmax", "hmax_numeric"]:
            results = []
            for disable_rustamer in [True, False]:
                reload_tamerlite(disable_rustamer)
                for enable_heuristic_cache in [True, False]:
                    search = tamerlite.SearchParams(
                        search="wastar",
                        heuristic=heuristic,
                        weight=0.8,
                        enable_heuristic_cache=enable_heuristic_cache,
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

            states = generate_states(ss, init_state, num_states=2000)
            for heuristic_class, heuristic_name in [
                (HFF, "HFF"),
                (HAdd, "HAdd"),
                (HMax, "HMax"),
                (HMaxNumeric, "HMaxNumeric"),
            ]:
                for cache_states in [True, False]:
                    heuristic: Heuristic = heuristic_class(
                        encoder.fluents,
                        encoder.objects,
                        encoder.events,
                        encoder.goal,
                        cache_states=cache_states,
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
                reload_tamerlite(disable_rustamer)
                search = tamerlite.SearchParams(search=search_kind, heuristic=heuristic)

                with OneshotPlanner(
                    name="tamerlite", params={"search": search}
                ) as planner:
                    planner: tamerlite.engine.TamerLite
                    res: PlanGenerationResult = planner.solve(
                        problem, heuristic=heuristic, timeout=30
                    )
                    assert res.status in (
                        PlanGenerationResultStatus.SOLVED_SATISFICING,
                        PlanGenerationResultStatus.TIMEOUT,
                    ) or (
                        search_kind == "ehs"
                        and res.status
                        == PlanGenerationResultStatus.UNSOLVABLE_INCOMPLETELY
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
            states[l] = generate_states(ss, init_state, num_states=2000)

        assert len(states["python"]) == len(states["rust"])
        for i in range(len(states["python"])):
            state1 = states["python"][i]
            state2 = states["rust"][i]

            assert len(state1.path) == len(state2.path)
            actions1 = list(map(lambda e: e[0].action, state1.path))
            actions2 = list(map(lambda e: e[0], state2.path))
            assert actions1 == actions2

            assert len(state1.todo) == len(state2.todo)
            for k in state1.todo:
                assert k in state2.todo
                assert state1.todo[k][0] == state2.todo[k][0]

            assert state1.g == state2.g
