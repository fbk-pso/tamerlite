import importlib
from unified_planning.shortcuts import *
from unified_planning.engines import PlanGenerationResult, PlanGenerationResultStatus
import os
import types
import problems_generator
import tamerlite
import pytest

import tamerlite.engine


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

            for i in range(len(results) - 1):
                res1: PlanGenerationResult = results[i]
                res2: PlanGenerationResult = results[i + 1]
                assert len(res1.metrics) == len(res2.metrics)
                assert (
                    res1.metrics["expanded_states"] == res2.metrics["expanded_states"]
                )


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

            for i in range(len(results) - 1):
                res1: PlanGenerationResult = results[i]
                res2: PlanGenerationResult = results[i + 1]
                assert len(res1.metrics) == len(res2.metrics)
                assert (
                    res1.metrics["expanded_states"] == res2.metrics["expanded_states"]
                )
