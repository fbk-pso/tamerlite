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

import contextlib
import importlib
import os
import types
from collections.abc import Callable
from functools import partial

import pytest
import unified_planning
import unified_planning.test
import unified_planning.test.examples
import up_test_cases.builtin
from unified_planning.engines import PlanGenerationResult, ValidationResult
from unified_planning.engines import PlanGenerationResultStatus as ResultStatus
from unified_planning.exceptions import UPUsageError
from unified_planning.plans import TimeTriggeredPlan
from unified_planning.shortcuts import *

import problems_generator
import tamerlite
import tamerlite.encoder
import tamerlite.engine
import testing_utils
from tamerlite.core.heuristics import Heuristic
from tamerlite.core.search_space import SearchSpaceABC
from tamerlite.encoder import Encoder

env = get_environment()
env.factory.add_engine("tamerlite", "tamerlite.engine", "TamerLite")


HEURISTICS = [
    "hff",
    "hadd",
    "hmax",
    "hff_no_numbers",
    "hadd_no_numbers",
    "hmax_no_numbers",
    "hmax_explicit",
]


def _build_problems():
    test_problems = [
        problems_generator.get_problem_logistics(1, 1, 4, 2),
        problems_generator.get_problem_numeric(),
        problems_generator.get_problem_satellite(),
        problems_generator.get_problem_hierarchical_types(),
        problems_generator.get_problem_temporal_flight(),
        problems_generator.get_problem_flight(),
    ]

    up_example_problems = list(
        unified_planning.test.examples.get_example_problems().values()
    )
    up_test_problems = list(up_test_cases.builtin.get_test_cases().values())
    test_problems.extend(
        test_case.problem
        for test_case in up_example_problems + up_test_problems
        if test_case.solvable
        and tamerlite.engine.TamerLite.supports(test_case.problem.kind)
    )

    names: set[str | None] = {None}
    for problem in test_problems:
        if problem.name in names:
            # name duplicated
            i = 0
            base_name = problem.name if problem.name is not None else ""
            new_name = base_name + str(i)
            while new_name in names:
                i += 1
                new_name = base_name + str(i)
            problem.name = new_name
        names.add(problem.name)

    return test_problems


def _build_anytime_problems(problems):
    test_problems = [
        problems_generator.get_problem_temporal_flight_minimize_makespan(),
        problems_generator.get_problem_temporal_flight_minimize_fuel(),
        problems_generator.get_problem_temporal_flight_maximize_fuel(),
        problems_generator.get_problem_flight_minimize_plan_length(),
        problems_generator.get_problem_flight_minimize_fuel(),
        problems_generator.get_problem_flight_maximize_fuel(),
    ]
    test_problems.extend(filter(lambda p: len(p.quality_metrics) == 1, problems))
    return test_problems


# Built once at import time (once per xdist worker) so tests can be parametrized
# per-problem instead of looping internally. Collection order is deterministic
# (insertion-ordered UP dicts + in-order de-dup), which xdist requires.
PROBLEMS = _build_problems()
ANYTIME_PROBLEMS = _build_anytime_problems(PROBLEMS)

# depots_pfile1 is by far the largest instance in the set; its unbounded
# (timeout=None) solves run for minutes each and used to dominate the suite's
# wall-clock. We exclude it from the full-solve tests but deliberately KEEP it
# in the cheap large-instance equivalence checks -- test_search_space (Rust vs
# Python state expansion / encoding) and test_heuristic_values (heuristic values
# on generated states) -- so backend agreement on a hard instance is still
# exercised, just without the multi-minute searches.
EXPENSIVE_SOLVE_EXCLUDE = {"depots_pfile1"}


def _solve_problems():
    return [p for p in PROBLEMS if p.name not in EXPENSIVE_SOLVE_EXCLUDE]


@pytest.fixture
def expressions():
    import json
    import pathlib

    data_path = os.path.join(
        pathlib.Path(__file__).parent.resolve(),
        "test_engine",
        "test_simplify_fixed_expressions.json",
    )
    with open(data_path) as f:
        data = json.load(f)

    expressions = [(e["exp"], e["simplified_exp"]) for e in data["expressions"]]
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
                if (
                    (fn_child is not None)
                    and fn_child.startswith(fn_dir)
                    and fn_child not in module_visit
                ):
                    module_visit.add(fn_child)
                    reload_recursive_ex(module_child)

    return reload_recursive_ex(package)


def reload_tamerlite(disable_rustamer: bool):
    os.environ["DISABLE_RUSTAMER"] = str(disable_rustamer)
    reload_package(tamerlite)


def skip(
    problem,
    search,
    heuristic,
    weak_equality,
    symmetry_breaking,
    disable_rustamer,
    internal_heuristic_cache,
):
    """Whether this parametrization should be skipped.

    Every rule below is a performance prune, not a correctness exclusion: on the
    heaviest instances these combinations run for minutes under an unbounded
    (timeout=None) search and/or exhaust available memory, dominating the suite's
    wall-clock. Each search/heuristic involved is still exercised on many other
    problems, so dropping these specific combinations costs little coverage.
    """
    return (
        (problem.name == "robot_fluent_of_user_type" and search == "dfs")
        or (problem.name == "robot_loader" and search == "dfs")
        or (problem.name == "robot_loader_mod" and search == "dfs")
        or (problem.name == "robot_loader_adv" and search == "dfs")
        or (problem.name == "robot_fluent_of_user_type_with_int_id" and search == "dfs")
        or (problem.name == "depots_p01" and search in ["dfs", "bfs"])
        or (problem.name == "RoboLogistics" and search == "dfs")
        or (problem.name == "NumericProblem" and search == "dfs")
        or (problem.name == "hierarchical-types" and search in ["dfs", "bfs"])
        or (problem.name == "hierarchical_blocks_world" and search == "dfs")
        or (
            problem.name == "hierarchical_blocks_world_object_as_root"
            and search == "dfs"
        )
        or (problem.name == "hierarchical_blocks_world_with_object" and search == "dfs")
        or (problem.name == "tpp_p01" and search == "dfs")
        or (
            problem.name == "satellite"
            and (
                search in ["dfs", "bfs"]
                or heuristic == "custom"
                or (
                    heuristic in ["hmax", "hmax_no_numbers", "hmax_explicit"]
                    and not weak_equality
                )
            )
        )
        or (
            problem.name == "robot_holding"
            and (
                search in ["dfs", "bfs"]
                or (not weak_equality and (heuristic == "custom" or search == "gbfs"))
            )
        )
        or (problem.name == "timed_connected_locations" and search == "dfs")
        or (problem.name == "hierarchical_blocks_world_exists" and search == "dfs")
        or (problem.name == "existential_linear_conditions" and search == "dfs")
        or (
            problem.name == "plant_watering_4_1"
            and (search == "multiqueue" or heuristic != "hadd")
        )
        or (problem.name == "rovers_pfile2" and search in ["dfs", "bfs"])
        or (
            problem.name == "block_grouping_5_5_1_1"
            and (
                heuristic
                in ["custom", "hmax_no_numbers", "hff_no_numbers", "hmax_explicit"]
                or search in ["bfs", "ehc"]
            )
        )
        or (
            problem.name == "farmland_2_100_1229"
            and (search in ["dfs", "bfs", "ehc"] or heuristic == "hmax_explicit")
        )
        or (
            problem.name in ["depots_pfile1", "depots_pfile10"]
            and search in ["dfs", "bfs"]
        )
        or (
            problem.name == "depots_pfile1"
            and search in ["wastar", "astar", "gbfs", "multiqueue"]
            and not weak_equality
        )
        or (
            problem.name == "universal_existential_linear_conditions"
            and search == "dfs"
        )
        or (
            problem.name == "RoboLogistics"
            and (search == "bfs" or heuristic == "custom")
        )
        or (
            problem.name == "block_grouping_5_5_1_1"
            and (search == "astar" or (heuristic == "hmax" and not weak_equality))
        )
        or (problem.name == "rovers_pfile2" and heuristic == "custom")
        or (
            problem.name == "logistic"
            and not weak_equality
            and (
                search in ["gbfs", "dfs", "bfs"]
                or (
                    search == "wastar"
                    and heuristic in ["hff", "hff_no_numbers", "custom"]
                )
            )
        )
    )


def max_generated_states(problem):
    if problem.name in {
        "nonlinear_increase_effects",
        "constant_increase_effect",
        "constant_decrease_effect",
        "disjunctive_linear_conditions",
        "basic_undef_numeric",
        "equality_linear_conditions",
    }:
        return 2
    if problem.name in {"constant_increase_effect_2", "constant_decrease_effect_2"}:
        return 4
    if problem.name in {"existential_linear_conditions", "nonlinear_assign_effects"}:
        return 3
    if problem.name == "farmland_2_100_1229":
        return 100
    return 1000


def max_generated_plans(problem):
    if problem.name in {"satellite"}:
        return 1

    return 4


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
        assert res1.metrics is not None and res2.metrics is not None
        assert len(res1.metrics) == len(res2.metrics)
        assert int(res1.metrics["expanded_states"]) == int(
            res2.metrics["expanded_states"]
        )
        assert int(res1.metrics["goal_depth"]) == int(res2.metrics["goal_depth"])


def _inadmissible_flags(problem, heuristic):
    if testing_utils.is_numeric_problem(problem) and heuristic in {
        "hff",
        "hadd",
        "hmax",
    }:
        return [True, False]
    return [False]


def _weak_flags(problem):
    return [True, False] if testing_utils.is_temporal_problem(problem) else [False]


def _compression_flags(problem):
    return [True, False] if testing_utils.is_temporal_problem(problem) else [False]


def _heuristics_cases():
    return [
        pytest.param(
            problem,
            heuristic,
            inadmissible_numeric_heuristic,
            weak_equality,
            symmetry_breaking,
            id=f"{problem.name}-{heuristic}"
            f"-inadm{int(inadmissible_numeric_heuristic)}"
            f"-weak{int(weak_equality)}-sym{int(symmetry_breaking)}",
        )
        for problem in _solve_problems()
        for heuristic in HEURISTICS
        for inadmissible_numeric_heuristic in _inadmissible_flags(problem, heuristic)
        for weak_equality in _weak_flags(problem)
        for symmetry_breaking in [True, False]
    ]


@pytest.mark.parametrize(
    "problem,heuristic,inadmissible_numeric_heuristic,weak_equality,symmetry_breaking",
    _heuristics_cases(),
)
def test_heuristics(
    problem,
    heuristic,
    inadmissible_numeric_heuristic,
    weak_equality,
    symmetry_breaking,
):
    search_kind = "wastar"
    results = []
    for disable_rustamer in [True, False]:
        reload_tamerlite(disable_rustamer)
        for internal_heuristic_cache in [True, False]:
            if skip(
                problem,
                search_kind,
                heuristic,
                weak_equality,
                symmetry_breaking,
                disable_rustamer,
                internal_heuristic_cache,
            ):
                continue

            search = tamerlite.SearchParams(
                search=search_kind,
                heuristic=heuristic,
                weight=0.8,
                internal_heuristic_cache=internal_heuristic_cache,
                inadmissible_numeric_heuristic_variant=inadmissible_numeric_heuristic,
                weak_equality=weak_equality,
                symmetry_breaking=symmetry_breaking,
                compression_safe_actions=False,
            )

            with OneshotPlanner(name="tamerlite", params={"search": search}) as planner:
                planner: tamerlite.engine.TamerLite
                res: PlanGenerationResult = planner.solve(problem, timeout=None)
                assert res.status == ResultStatus.SOLVED_SATISFICING
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

            lifted_problem, ground_problem, map_back_action_instance = (
                testing_utils.compile_problem(problem)
            )
            encoder = Encoder(
                ground_problem,
                lifted_problem,
                map_back_action_instance,
                symmetry_breaking=False,
                compression_safe_actions=False,
                relevance_analysis=False,
            )
            ss: SearchSpaceABC = encoder.search_space
            init_state = ss.initial_state()

            states = [init_state]
            for action_name in path:
                state = ss.get_successor_state(
                    states[-1], encoder.get_action(action_name)
                )
                assert state is not None
                states.append(state)

            heuristic_classes: list[tuple[Callable[..., Heuristic], str]] = [
                (HFF, "hff"),
                (HAdd, "hadd"),
                (HMax, "hmax"),
                (HMaxExplicit, "hmax_explicit"),
            ]
            for heuristic_class, heuristic_name in heuristic_classes:
                for internal_caching in [True, False]:
                    heuristic: Heuristic = heuristic_class(
                        encoder.actions,
                        encoder.fluent_types,
                        encoder.objects,
                        encoder.events,
                        encoder.goal,
                        internal_caching=internal_caching,
                        cache_value_in_state=False,
                        inadmissible_numeric_heuristic_variant=False,
                    )

                    for i, state in enumerate(states):
                        h_val = heuristic.eval(state, ss)
                        if h_val is not None:
                            h_val = int(h_val)
                        assert values[heuristic_name][i] == h_val


@pytest.mark.parametrize("problem", PROBLEMS, ids=[p.name for p in PROBLEMS])
def test_heuristic_values(problem, data_regression):
    values: dict[str, list[int | None]] = {}
    for disable_rustamer in [True, False]:
        reload_tamerlite(disable_rustamer)
        from tamerlite.core import HFF, HAdd, HMax, HMaxExplicit

        lifted_problem, ground_problem, map_back_action_instance = (
            testing_utils.compile_problem(problem)
        )
        encoder = Encoder(
            ground_problem,
            lifted_problem,
            map_back_action_instance,
            symmetry_breaking=False,
            compression_safe_actions=False,
            relevance_analysis=False,
        )
        ss: SearchSpaceABC = encoder.search_space
        init_state = ss.initial_state()

        states = generate_states(
            ss, init_state, num_states=max_generated_states(problem)
        )
        heuristic_classes: list[tuple[Callable[..., Heuristic], str]] = [
            (HFF, "hff"),
            (HAdd, "hadd"),
            (HMax, "hmax"),
            (partial(HFF, disable_numeric_reasoning=True), "hff_no_numbers"),
            (partial(HAdd, disable_numeric_reasoning=True), "hadd_no_numbers"),
            (partial(HMax, disable_numeric_reasoning=True), "hmax_no_numbers"),
            (HMaxExplicit, "hmax_explicit"),
        ]
        for heuristic_class, heuristic_name in heuristic_classes:
            inadmissible_numeric_heuristic_flags = [False]
            if testing_utils.is_numeric_problem(problem) and heuristic_name in {
                "hff",
                "hadd",
                "hmax",
            }:
                inadmissible_numeric_heuristic_flags = [True, False]
            for inadmissible_numeric_heuristic in inadmissible_numeric_heuristic_flags:
                for internal_caching in [True, False]:
                    if skip(
                        problem,
                        "wastar",
                        heuristic_name,
                        True,
                        True,
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
                        inadmissible_numeric_heuristic_variant=inadmissible_numeric_heuristic,
                    )

                    values_key = heuristic_name + (
                        "_inadmissible" if inadmissible_numeric_heuristic else ""
                    )
                    if values_key not in values:
                        values[values_key] = []
                        for state in states:
                            h_val = heuristic.eval(state, ss)
                            if h_val is not None:
                                h_val = int(h_val)
                            values[values_key].append(h_val)

                    else:
                        assert len(states) == len(values[values_key])
                        for i, state in enumerate(states):
                            h_val = heuristic.eval(state, ss)
                            if h_val is not None:
                                h_val = int(h_val)
                            assert h_val == values[values_key][i]

    data_regression.check(values)


def _weak_sym_compression_cases():
    """(problem, weak_equality, symmetry_breaking, compression_safe_actions) tuples;
    temporal problems additionally vary weak_equality and compression_safe_actions."""
    return [
        pytest.param(
            problem,
            weak_equality,
            symmetry_breaking,
            compression_safe_actions,
            id=f"{problem.name}-weak{int(weak_equality)}"
            f"-sym{int(symmetry_breaking)}"
            f"-csa{int(compression_safe_actions)}",
        )
        for problem in _solve_problems()
        for weak_equality in _weak_flags(problem)
        for symmetry_breaking in [True, False]
        for compression_safe_actions in _compression_flags(problem)
    ]


@pytest.mark.parametrize(
    "problem,weak_equality,symmetry_breaking,compression_safe_actions",
    _weak_sym_compression_cases(),
)
def test_custom_heuristic(
    problem, weak_equality, symmetry_breaking, compression_safe_actions
):
    search_kind = "wastar"
    heuristic = "custom"

    def custom_heuristic(state: State):
        return 1

    results = []
    for disable_rustamer in [True, False]:
        reload_tamerlite(disable_rustamer)

        for internal_heuristic_cache in [True, False]:
            if skip(
                problem,
                search_kind,
                heuristic,
                weak_equality,
                symmetry_breaking,
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
                symmetry_breaking=symmetry_breaking,
                compression_safe_actions=compression_safe_actions,
            )

            with OneshotPlanner(name="tamerlite", params={"search": search}) as planner:
                planner: tamerlite.engine.TamerLite
                res: PlanGenerationResult = planner.solve(
                    problem, heuristic=custom_heuristic, timeout=None
                )
                assert res.status == ResultStatus.SOLVED_SATISFICING
                results.append(res)
                with PlanValidator(problem_kind=problem.kind) as v:
                    assert v.validate(problem, res.plan)

    check_metrics_equality(results)


def _search_algo_weak_flags(problem, search_kind):
    if testing_utils.is_temporal_problem(problem) and search_kind not in ("dfs", "bfs"):
        return [True, False]
    return [False]


def _search_algo_memory_bounded_flags(problem, search_kind):
    if not testing_utils.is_temporal_problem(problem) and search_kind in {
        "wastar",
        "astar",
        "gbfs",
    }:
        return [True, False]
    return [False]


def _search_algorithms_cases():
    return [
        pytest.param(
            problem,
            search_kind,
            memory_bounded,
            weak_equality,
            symmetry_breaking,
            compression_safe_actions,
            id=f"{problem.name}-{search_kind}"
            f"-mb{int(memory_bounded)}"
            f"-weak{int(weak_equality)}"
            f"-sym{int(symmetry_breaking)}"
            f"-csa{int(compression_safe_actions)}",
        )
        for problem in _solve_problems()
        for search_kind in ["wastar", "astar", "gbfs", "dfs", "bfs", "ehc"]
        for memory_bounded in _search_algo_memory_bounded_flags(problem, search_kind)
        for weak_equality in _search_algo_weak_flags(problem, search_kind)
        if not (memory_bounded and weak_equality)
        for symmetry_breaking in [True, False]
        for compression_safe_actions in _compression_flags(problem)
    ]


@pytest.mark.parametrize(
    "problem,search_kind,memory_bounded,weak_equality,"
    "symmetry_breaking,compression_safe_actions",
    _search_algorithms_cases(),
)
def test_search_algorithms(
    problem,
    search_kind,
    memory_bounded,
    weak_equality,
    symmetry_breaking,
    compression_safe_actions,
):
    heuristic = "hff"
    results = []
    for disable_rustamer in [True, False]:
        if skip(
            problem,
            search_kind,
            heuristic,
            weak_equality,
            symmetry_breaking,
            disable_rustamer,
            True,
        ):
            continue

        reload_tamerlite(disable_rustamer)
        search = tamerlite.SearchParams(
            search=search_kind,
            heuristic=heuristic,
            weak_equality=weak_equality,
            symmetry_breaking=symmetry_breaking,
            compression_safe_actions=compression_safe_actions,
            incomplete_memory_bounded_search=memory_bounded,
        )

        with OneshotPlanner(name="tamerlite", params={"search": search}) as planner:
            planner: tamerlite.engine.TamerLite
            res: PlanGenerationResult = planner.solve(problem, timeout=None)
            assert res.status == ResultStatus.SOLVED_SATISFICING or (
                search_kind == "ehc"
                and res.status == ResultStatus.UNSOLVABLE_INCOMPLETELY
            )
            if res.status == ResultStatus.SOLVED_SATISFICING:
                results.append(res)
                with PlanValidator(problem_kind=problem.kind) as v:
                    assert v.validate(problem, res.plan)

    check_metrics_equality(results)


@pytest.mark.parametrize(
    "problem,weak_equality,symmetry_breaking,compression_safe_actions",
    _weak_sym_compression_cases(),
)
def test_multiqueue_search(
    problem, weak_equality, symmetry_breaking, compression_safe_actions
):
    results = []
    for disable_rustamer in [True, False]:
        if skip(
            problem,
            "multiqueue",
            heuristic=None,
            weak_equality=weak_equality,
            symmetry_breaking=symmetry_breaking,
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
            symmetry_breaking=symmetry_breaking,
            compression_safe_actions=compression_safe_actions,
        )
        with OneshotPlanner(name="tamerlite", params={"search": search}) as planner:
            res: PlanGenerationResult = planner.solve(problem, timeout=None)
            assert res.status == ResultStatus.SOLVED_SATISFICING
            results.append(res)
            with PlanValidator(problem_kind=problem.kind) as v:
                assert v.validate(problem, res.plan)

    check_metrics_equality(results)


@pytest.mark.parametrize("problem", PROBLEMS, ids=[p.name for p in PROBLEMS])
def test_search_space(problem):
    states = {}
    for disable_rustamer in [True, False]:
        reload_tamerlite(disable_rustamer)
        reload_package(tamerlite.encoder)
        from tamerlite.encoder import Encoder

        lifted_problem, ground_problem, map_back_action_instance = (
            testing_utils.compile_problem(problem)
        )
        encoder = Encoder(
            ground_problem,
            lifted_problem,
            map_back_action_instance,
            symmetry_breaking=False,
            compression_safe_actions=False,
            relevance_analysis=False,
        )
        ss: SearchSpaceABC = encoder.search_space

        init_state = ss.initial_state()
        backend = "python" if disable_rustamer else "rust"
        states[backend] = generate_states(
            ss, init_state, num_states=max_generated_states(problem)
        )

    assert len(states["python"]) == len(states["rust"])
    for i in range(len(states["python"])):
        state1 = states["python"][i]
        state2 = states["rust"][i]

        assert len(state1.path) == len(state2.path)
        actions1 = [encoder.get_action_name(e[0]) for e in state1.path]
        actions2 = [encoder.get_action_name(e[0]) for e in state2.path]
        assert actions1 == actions2

        assert len(state1.todo) == len(state2.todo)
        todo2 = {k.idx: v for k, v in state2.todo.items()}
        for k in state1.todo:
            assert k.idx in todo2
            assert state1.todo[k][0] == todo2[k.idx][0]

        assert state1.g == state2.g


def _anytime_cases():
    return [
        pytest.param(
            problem,
            weak_equality,
            symmetry_breaking,
            disable_rustamer,
            id=f"{problem.name}-weak{int(weak_equality)}"
            f"-sym{int(symmetry_breaking)}"
            f"-{'py' if disable_rustamer else 'rs'}",
        )
        for problem in ANYTIME_PROBLEMS
        if problem.name not in EXPENSIVE_SOLVE_EXCLUDE
        for weak_equality in _weak_flags(problem)
        for symmetry_breaking in [True, False]
        for disable_rustamer in [True, False]
    ]


@pytest.mark.parametrize(
    "problem,weak_equality,symmetry_breaking,disable_rustamer",
    _anytime_cases(),
)
def test_anytime_planner(problem, weak_equality, symmetry_breaking, disable_rustamer):
    quality_metric = problem.quality_metrics[0]
    is_minimization_metric = (
        quality_metric.is_minimize_action_costs()
        or quality_metric.is_minimize_sequential_plan_length()
        or quality_metric.is_minimize_makespan()
        or quality_metric.is_minimize_expression_on_final_state()
    )

    search_kind = "wastar"
    heuristic = "hff"
    internal_heuristic_cache = True
    max_plans = max_generated_plans(problem)

    reload_tamerlite(disable_rustamer)
    search = tamerlite.SearchParams(
        search=search_kind,
        heuristic=heuristic,
        weight=0.8,
        internal_heuristic_cache=internal_heuristic_cache,
        weak_equality=weak_equality,
        symmetry_breaking=symmetry_breaking,
        compression_safe_actions=False,
    )

    if skip(
        problem,
        search_kind,
        heuristic,
        weak_equality,
        symmetry_breaking,
        disable_rustamer,
        internal_heuristic_cache,
    ):
        return

    with AnytimePlanner(name="tamerlite", params={"search": search}) as planner:
        for counter, res in enumerate(planner.get_solutions(problem, timeout=None)):
            assert res.status in {
                ResultStatus.INTERMEDIATE,
                ResultStatus.SOLVED_SATISFICING,
                ResultStatus.SOLVED_OPTIMALLY,
            }
            prev_metric_value = None
            with PlanValidator(problem_kind=problem.kind) as v:
                val_res: ValidationResult = v.validate(problem, res.plan)
                assert val_res
                assert (
                    val_res.metric_evaluations is not None
                    and len(val_res.metric_evaluations) == 1
                )
                metric_value = next(iter(val_res.metric_evaluations.values()))
                if prev_metric_value is not None:
                    if res.status == ResultStatus.INTERMEDIATE:
                        if is_minimization_metric:
                            assert metric_value < prev_metric_value
                        else:
                            assert metric_value > prev_metric_value
                    else:
                        if is_minimization_metric:
                            assert metric_value <= prev_metric_value
                        else:
                            assert metric_value >= prev_metric_value
                else:
                    prev_metric_value = metric_value

            if counter + 1 == max_plans:
                break


def test_simplify():
    num_expressions = 100
    results: dict[bool, list] = {
        True: [None] * num_expressions,
        False: [None] * num_expressions,
    }
    for disable_rustamer in [True, False]:
        reload_tamerlite(disable_rustamer)
        reload_package(testing_utils)
        from tamerlite.core import simplify
        from testing_utils import (
            construct_expressions,
            is_strictly_increasing,
            parse_expression,
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
            with contextlib.suppress(AttributeError):
                assert is_strictly_increasing(node.operands)


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


def test_temporal_fluent_duration():
    problem = problems_generator.get_problem_temporal_fluent_duration()

    for disable_rustamer in [True, False]:
        reload_tamerlite(disable_rustamer)

        search = tamerlite.SearchParams(
            search="wastar",
            heuristic="hff",
            weight=0.8,
            compression_safe_actions=False,
        )

        with OneshotPlanner(name="tamerlite", params={"search": search}) as planner:
            planner: tamerlite.engine.TamerLite
            res: PlanGenerationResult = planner.solve(problem, timeout=None)
            assert res.status == ResultStatus.SOLVED_SATISFICING

            with PlanValidator(problem_kind=problem.kind) as v:
                assert v.validate(problem, res.plan)

            assert isinstance(res.plan, TimeTriggeredPlan)
            timed_actions = res.plan.timed_actions
            assert len(timed_actions) == 1
            _, _, duration = timed_actions[0]
            assert duration == Fraction(5)


def test_symmetry_breaking_object_valued_fluents():
    """
    Symmetry breaking must not conflate two objects that are only distinguished by
    appearing as the *value* of an object-valued fluent (as opposed to a fluent
    argument), whether in the initial state (explicit or default) or in the goal.
    """

    problems = [
        problems_generator.get_problem_object_value_symmetry_initial(),
        problems_generator.get_problem_object_value_symmetry_goal(),
    ]
    for problem in problems:
        for disable_rustamer in [True, False]:
            reload_tamerlite(disable_rustamer)

            for symmetry_breaking in [True, False]:
                search = tamerlite.SearchParams(
                    search="wastar",
                    heuristic="hadd",
                    symmetry_breaking=symmetry_breaking,
                    compression_safe_actions=False,
                )
                with OneshotPlanner(
                    name="tamerlite", params={"search": search}
                ) as planner:
                    planner: tamerlite.engine.TamerLite
                    res: PlanGenerationResult = planner.solve(problem, timeout=None)
                    assert res.status == ResultStatus.SOLVED_SATISFICING
                    with PlanValidator(problem_kind=problem.kind) as v:
                        assert v.validate(problem, res.plan)


def test_symmetry_breaking_retains_legitimate_symmetry():
    """
    Complement to `test_symmetry_breaking_object_valued_fluents`: objects that
    are genuinely symmetric via an object-valued fluent (swapping them
    preserves the initial state) must still be grouped together.
    """

    problem = problems_generator.get_problem_object_value_symmetry_retained()
    lifted_problem, ground_problem, map_back_action_instance = (
        testing_utils.compile_problem(problem)
    )
    encoder = Encoder(
        ground_problem,
        lifted_problem,
        map_back_action_instance,
        symmetry_breaking=False,
        compression_safe_actions=False,
        relevance_analysis=False,
    )
    groups = encoder._compute_equivalent_objects()
    assert {"a", "b"} in [{obj.name for obj in group} for group in groups]


def test_symmetry_breaking_default_value_objects():
    """
    An object used as a fluent's *default* value is not automatically
    non-equivalent to everything else: if every other grounding of that
    fluent is consistent with the swap, the objects are still symmetric
    (e.g. with only two objects of that type, `f` defaulting to `o1` and
    `f(o2)` explicitly set to `o2` is a genuine symmetry). But a default that
    is also shared by some *other*, unrelated object's grounding does break
    the symmetry (a third object silently falling through to the same
    default is enough to distinguish the pair).
    """

    problem = problems_generator.get_problem_default_value_object_symmetry()
    lifted_problem, ground_problem, map_back_action_instance = (
        testing_utils.compile_problem(problem)
    )
    encoder = Encoder(
        ground_problem,
        lifted_problem,
        map_back_action_instance,
        symmetry_breaking=False,
        compression_safe_actions=False,
        relevance_analysis=False,
    )
    groups = encoder._compute_equivalent_objects()
    assert {"o1", "o2"} in [{obj.name for obj in group} for group in groups]

    asymmetric_problem = problems_generator.get_problem_default_value_object_asymmetry()
    lifted_problem, ground_problem, map_back_action_instance = (
        testing_utils.compile_problem(asymmetric_problem)
    )
    encoder = Encoder(
        ground_problem,
        lifted_problem,
        map_back_action_instance,
        symmetry_breaking=False,
        compression_safe_actions=False,
        relevance_analysis=False,
    )
    groups = encoder._compute_equivalent_objects()
    assert [{obj.name for obj in group} for group in groups] == [{"x"}, {"y"}, {"z"}]


def test_symmetry_breaking_goal_taint_is_per_object():
    """
    An unrecognized goal conjunct (one _extract_goal_obj_to_fluent_map doesn't
    know how to precisely reason about) must only exclude the objects it
    actually references from equivalence, not every object in the problem.
    This is what makes symmetry breaking still useful when the goal contains
    e.g. an injected quality-bound literal that doesn't mention the objects
    being compared, as happens on every re-solve of TamerLite's anytime loop
    (e.g. LT(plan_length, v) for the auto-added
    MinimizeSequentialPlanLength metric).
    """

    def groups_for(problem):
        lifted_problem, ground_problem, map_back_action_instance = (
            testing_utils.compile_problem(problem)
        )
        encoder = Encoder(
            ground_problem,
            lifted_problem,
            map_back_action_instance,
            symmetry_breaking=False,
            compression_safe_actions=False,
            relevance_analysis=False,
        )
        return [
            {obj.name for obj in group}
            for group in encoder._compute_equivalent_objects()
        ]

    # Partial taint: an Or(...) goal conjunct is unrecognized and taints the
    # objects it references (p1, p4), but must not affect p2/p3, which only
    # appear in recognized (plain fluent) goal conjuncts and remain
    # genuinely symmetric.
    groups = groups_for(problems_generator.get_problem_goal_taint_partial())
    assert {"p2", "p3"} in groups
    assert {"p1"} in groups
    assert {"p4"} in groups

    # Equals(fluent, fluent) (neither side a constant) is also an
    # unrecognized shape and must taint its objects as one opaque unit, not
    # be decomposed into independent per-fluent literals -- doing so would
    # assign the wrong meaning (Equals(fl(a), fr(b)) only requires the two
    # values to match, e.g. both zero; it does not require either to hold
    # any specific value on its own).
    groups = groups_for(
        problems_generator.get_problem_goal_taint_equals_fluent_fluent()
    )
    assert groups == [{"a"}, {"b"}]

    # End-to-end regression: TamerLite's anytime loop injects a quality-bound
    # literal into the goal on every re-solve after the first. Before the
    # per-object taint fix, that single unrecognized literal flipped a global
    # "is the goal a conjunction of recognized literals" flag, which made
    # every object in the problem -- even ones nowhere near the injected
    # literal -- ineligible for equivalence. Verify the equivalence classes
    # for genuinely symmetric objects survive across a real anytime re-solve.
    orig_compute_equivalent_objects = (
        tamerlite.encoder.Encoder._compute_equivalent_objects
    )
    observed_groups: list[list[set[str]]] = []

    def instrumented(self):
        result = orig_compute_equivalent_objects(self)
        observed_groups.append([{obj.name for obj in group} for group in result])
        return result

    tamerlite.encoder.Encoder._compute_equivalent_objects = (  # type: ignore[method-assign]
        instrumented
    )
    try:
        anytime_problem = problems_generator.get_problem_anytime_symmetric_delivery()
        search = tamerlite.SearchParams(
            search="wastar",
            heuristic="hadd",
            symmetry_breaking=True,
            compression_safe_actions=False,
        )
        planner = tamerlite.engine.TamerLite(search)
        for i, _ in enumerate(planner.get_solutions(anytime_problem, timeout=None)):
            if i >= 1:
                break
    finally:
        tamerlite.encoder.Encoder._compute_equivalent_objects = (  # type: ignore[method-assign]
            orig_compute_equivalent_objects
        )

    assert len(observed_groups) >= 2
    for pkg_groups in observed_groups[:2]:
        assert {"p1", "p2", "p3"} in pkg_groups


# --- Interpreted-function tests --------------------------------------------
#
# Interpreted functions are only evaluated by the pure-Python backend (see
# `InterpretedFunctionNode` / `evaluate` in `tamerlite.core.search_space`), so
# every test below forces the Python backend via `reload_tamerlite(True)`.
# They are never routed to the Rust backend: `TamerLite.supported_kind()`
# only advertises `INTERPRETED_FUNCTIONS_IN_*` when it is active.

IF_PROBLEMS = [
    problems_generator.get_problem_if_bool_condition(),
    problems_generator.get_problem_if_numeric_effect(),
    problems_generator.get_problem_if_minimal_chain(),
]

# Delete-relaxation heuristics reason about conditions/effects structurally
# and don't know how to handle an interpreted-function node; only "blind"
# and "custom" evaluate a plain search `State` and are IF-safe.
IF_DISALLOWED_HEURISTICS = [
    "hff",
    "hadd",
    "hmax",
    "hmax_explicit",
    "hff_no_numbers",
    "hadd_no_numbers",
    "hmax_no_numbers",
]


@pytest.mark.parametrize("problem", IF_PROBLEMS, ids=[p.name for p in IF_PROBLEMS])
def test_interpreted_functions_supports(problem):
    for disable_rustamer in [True, False]:
        reload_tamerlite(disable_rustamer)
        assert tamerlite.engine.TamerLite.supports(problem.kind) == disable_rustamer


@pytest.mark.parametrize("problem", IF_PROBLEMS, ids=[p.name for p in IF_PROBLEMS])
@pytest.mark.parametrize("search_kind", ["gbfs", "wastar", "bfs"])
def test_interpreted_functions_solve_with_blind_heuristic(problem, search_kind):
    results = []
    for disable_rustamer in [True]:
        reload_tamerlite(disable_rustamer)
        search = tamerlite.SearchParams(search=search_kind, heuristic="blind")
        with OneshotPlanner(name="tamerlite", params={"search": search}) as planner:
            planner: tamerlite.engine.TamerLite
            res: PlanGenerationResult = planner.solve(problem, timeout=None)
            assert res.status == ResultStatus.SOLVED_SATISFICING
            results.append(res)
            with PlanValidator(problem_kind=problem.kind) as v:
                assert v.validate(problem, res.plan)

    check_metrics_equality(results)


@pytest.mark.parametrize("problem", IF_PROBLEMS, ids=[p.name for p in IF_PROBLEMS])
def test_interpreted_functions_solve_with_custom_heuristic(problem):
    def custom_heuristic(state: State):
        return 0.0

    results = []
    for disable_rustamer in [True]:
        reload_tamerlite(disable_rustamer)
        search = tamerlite.SearchParams(search="gbfs", heuristic="custom")
        with OneshotPlanner(name="tamerlite", params={"search": search}) as planner:
            planner: tamerlite.engine.TamerLite
            res: PlanGenerationResult = planner.solve(
                problem, heuristic=custom_heuristic, timeout=None
            )
            assert res.status == ResultStatus.SOLVED_SATISFICING
            results.append(res)
            with PlanValidator(problem_kind=problem.kind) as v:
                assert v.validate(problem, res.plan)

    check_metrics_equality(results)


@pytest.mark.parametrize("problem", IF_PROBLEMS, ids=[p.name for p in IF_PROBLEMS])
@pytest.mark.parametrize("heuristic", IF_DISALLOWED_HEURISTICS)
def test_interpreted_functions_delete_relaxation_heuristics_raise(problem, heuristic):
    for disable_rustamer in [True]:
        reload_tamerlite(disable_rustamer)
        search = tamerlite.SearchParams(search="gbfs", heuristic=heuristic)
        with (
            OneshotPlanner(name="tamerlite", params={"search": search}) as planner,
            pytest.raises(UPUsageError),
        ):
            planner.solve(problem, timeout=None)


def test_interpreted_functions_object_typed_argument_not_supported():
    problem = problems_generator.get_problem_if_object_argument()
    for disable_rustamer in [True]:
        reload_tamerlite(disable_rustamer)
        search = tamerlite.SearchParams(search="gbfs", heuristic="blind")
        with (
            OneshotPlanner(name="tamerlite", params={"search": search}) as planner,
            pytest.raises(NotImplementedError),
        ):
            planner.solve(problem, timeout=None)
