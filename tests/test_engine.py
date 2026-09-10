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
import gc
import importlib
import os
import types
import warnings
import weakref
from collections import OrderedDict
from collections.abc import Callable
from functools import partial
from typing import Any, NamedTuple, cast

import pytest
import unified_planning
import unified_planning.test
import unified_planning.test.examples
import up_test_cases.builtin
from unified_planning.engines import PlanGenerationResult, ValidationResult
from unified_planning.engines import PlanGenerationResultStatus as ResultStatus
from unified_planning.plans import TimeTriggeredPlan
from unified_planning.shortcuts import *

import problems_generator
import tamerlite
import tamerlite.encoder
import tamerlite.engine
import testing_utils
from tamerlite.core.heuristics import Heuristic
from tamerlite.core.search_space import ConstantNode, SearchSpaceABC
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
    "blind",
]


def _build_problems():
    test_problems = [
        problems_generator.get_problem_logistics(1, 1, 4, 2),
        problems_generator.get_problem_numeric(),
        problems_generator.get_problem_satellite(),
        problems_generator.get_problem_hierarchical_types(),
        problems_generator.get_problem_temporal_flight(),
        problems_generator.get_problem_flight(),
        problems_generator.get_problem_if_bool_condition(),
        problems_generator.get_problem_if_numeric_effect(),
        problems_generator.get_problem_if_object_effect(),
        problems_generator.get_problem_if_object_argument_and_return(),
        problems_generator.get_problem_if_undefined_initial_numeric(),
        problems_generator.get_problem_if_temporal_compression_safe(),
        problems_generator.get_problem_if_duration(),
        problems_generator.get_problem_if_conditions_and_effects(),
        problems_generator.get_problem_if_signature_shapes(),
        problems_generator.get_problem_if_numeric_symmetry_retained(),
        problems_generator.get_problem_if_object_argument_symmetry_unsound(),
        problems_generator.get_problem_if_hierarchical_type_argument(),
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
        # Children must be reloaded *before* `module` itself: `module`'s own
        # top-level code re-runs `from child import name` statements, and
        # those need `child` already holding its freshest definitions --
        # otherwise `name` rebinds to whatever `child` had before this reload
        # pass touched it, leaving two live, non-identical objects (e.g. two
        # `Enum` classes) that compare unequal despite being "the same" type
        # conceptually. For this same reason, callers must always reload the
        # whole `tamerlite` package (via `reload_tamerlite`) and never a lone
        # submodule on its own: e.g. calling `reload_package(tamerlite.encoder)`
        # again *after* `reload_tamerlite` rebuilds `tamerlite.encoder.Encoder`
        # a second time without anything re-running `tamerlite.engine`'s
        # `from tamerlite.encoder import Encoder`, leaving
        # `tamerlite.engine.Encoder is not tamerlite.encoder.Encoder` for the
        # rest of the process -- silently, since both classes share
        # `__globals__` and behave identically except for object identity.
        children = []
        for module_child in vars(module).values():
            if isinstance(module_child, types.ModuleType):
                fn_child = getattr(module_child, "__file__", None)
                if (
                    (fn_child is not None)
                    and fn_child.startswith(fn_dir)
                    and fn_child not in module_visit
                ):
                    module_visit.add(fn_child)
                    children.append(module_child)

        for module_child in children:
            reload_recursive_ex(module_child)

        importlib.reload(module)

    return reload_recursive_ex(package)


def reload_tamerlite(disable_rustamer: bool):
    os.environ["DISABLE_RUSTAMER"] = str(disable_rustamer)
    reload_package(tamerlite)


class PruneCase(NamedTuple):
    """The dimensions `PERFORMANCE_PRUNES` predicates can look at -- the subset
    of solve-matrix parameters that actually affect runtime."""

    search: str
    heuristic: str | None
    weak_equality: bool
    symmetry_breaking: bool


# Problem name -> which uninformed-style traversals are known to blow up on
# it: "dfs" (can wander arbitrarily deep down a bad branch) and/or "blind".
UNINFORMED_SEARCH_RISK: dict[str, set[str]] = {
    "robot_fluent_of_user_type": {"dfs"},
    "robot_loader": {"dfs"},
    "robot_loader_mod": {"dfs"},
    "robot_loader_adv": {"dfs"},
    "robot_fluent_of_user_type_with_int_id": {"dfs"},
    "depots_p01": {"dfs", "blind"},
    "RoboLogistics": {"dfs", "blind"},
    "NumericProblem": {"dfs"},
    "hierarchical-types": {"dfs", "blind"},
    "hierarchical_blocks_world": {"dfs"},
    "hierarchical_blocks_world_object_as_root": {"dfs"},
    "hierarchical_blocks_world_with_object": {"dfs"},
    "tpp_p01": {"dfs"},
    "satellite": {"dfs", "blind"},
    "robot_holding": {"dfs", "blind"},
    "timed_connected_locations": {"dfs"},
    "hierarchical_blocks_world_exists": {"dfs"},
    "existential_linear_conditions": {"dfs"},
    "rovers_pfile2": {"dfs", "blind"},
    "depots_pfile1": {"dfs"},
    "depots_pfile10": {"dfs", "blind"},
    "universal_existential_linear_conditions": {"dfs"},
    "interpreted_functions_minimal_chain_of_assignments": {"dfs"},
    "treasure_hunting_robot_simple": {"dfs"},
    "if_bool_condition": {"dfs"},
    "if_signature_shapes": {"dfs"},
    "block_grouping_5_5_1_1": {"blind"},
    "farmland_2_100_1229": {"blind"},
    "if_reals_condition_effect_pizza": {"blind"},
}

# Problems where `hmax_explicit` legitimately diverges between backends -- see
# the `PERFORMANCE_PRUNES` entry below and `test_heuristic_values`.
HMAX_EXPLICIT_BACKEND_DIVERGENT = ("if_object_effect", "if_object_argument_and_return")

# (problem_name(s), predicate, reason). Every entry is a performance prune, not
# a correctness exclusion: on the heaviest instances these combinations run for
# minutes under an unbounded (timeout=None) search and/or exhaust available
# memory, dominating the suite's wall-clock. Each search/heuristic involved is
# still exercised on many other problems, so pruning a specific combination
# here costs little coverage.
PERFORMANCE_PRUNES: list[
    tuple[str | tuple[str, ...], Callable[[PruneCase], bool], str]
] = [
    *(
        (name, lambda c: c.search == "dfs", "dfs can wander arbitrarily deep")
        for name, risk in UNINFORMED_SEARCH_RISK.items()
        if "dfs" in risk
    ),
    *(
        (
            name,
            lambda c: c.heuristic == "blind",
            "blind is as uninformed as bfs here",
        )
        for name, risk in UNINFORMED_SEARCH_RISK.items()
        if "blind" in risk
    ),
    (
        (
            "depots_p01",
            "hierarchical-types",
            "satellite",
            "robot_holding",
            "rovers_pfile2",
        ),
        lambda c: c.search == "bfs",
        "bfs explores the whole state space",
    ),
    (
        ("depots_pfile1", "depots_pfile10"),
        lambda c: c.search == "bfs",
        "bfs explores the whole state space",
    ),
    (
        "satellite",
        lambda c: c.heuristic == "custom",
        "custom heuristic times out",
    ),
    (
        "satellite",
        lambda c: (
            c.heuristic in {"hmax", "hmax_no_numbers", "hmax_explicit"}
            and not c.weak_equality
        ),
        "non-weak hmax family times out",
    ),
    (
        "robot_holding",
        lambda c: (
            not c.weak_equality and (c.heuristic == "custom" or c.search == "gbfs")
        ),
        "non-weak custom heuristic / gbfs times out",
    ),
    (
        "plant_watering_4_1",
        lambda c: c.search == "multiqueue" or c.heuristic != "hadd",
        "only hadd (non-multiqueue) solves in reasonable time",
    ),
    (
        "block_grouping_5_5_1_1",
        lambda c: (
            c.heuristic
            in {"custom", "hmax_no_numbers", "hff_no_numbers", "hmax_explicit"}
            or c.search in {"bfs", "ehc"}
        ),
        "weak heuristics / uninformed-ish search time out",
    ),
    (
        "farmland_2_100_1229",
        lambda c: c.search in {"dfs", "bfs", "ehc"} or c.heuristic == "hmax_explicit",
        "uninformed search / hmax_explicit time out on this instance",
    ),
    (
        "depots_pfile1",
        lambda c: (
            c.search in {"wastar", "astar", "gbfs", "multiqueue"}
            and not c.weak_equality
        ),
        "non-weak search exhausts memory on this instance",
    ),
    (
        "RoboLogistics",
        lambda c: c.search == "bfs" or c.heuristic == "custom",
        "bfs / custom heuristic time out",
    ),
    (
        "block_grouping_5_5_1_1",
        lambda c: (
            c.search == "astar" or (c.heuristic == "hmax" and not c.weak_equality)
        ),
        "astar / non-weak hmax time out",
    ),
    (
        "rovers_pfile2",
        lambda c: c.heuristic == "custom",
        "custom heuristic times out",
    ),
    (
        "logistic",
        lambda c: (
            not c.weak_equality
            and (
                c.search in {"gbfs", "dfs", "bfs"}
                or (
                    c.search == "wastar"
                    and c.heuristic in {"hff", "hff_no_numbers", "custom", "blind"}
                )
            )
        ),
        "non-weak uninformed/gbfs search or wastar+hff-family/blind times out",
    ),
    (
        "if_reals_condition_effect_pizza",
        lambda c: c.search in {"bfs", "ehc"},
        "uninformed search explores the whole state space",
    ),
    (
        "if_numeric_symmetry_retained",
        lambda c: c.heuristic == "hmax_explicit",
        "hmax_explicit never terminates on this problem's unbounded "
        "monotonic interpreted-function effect",
    ),
    # Not a performance prune like everything above -- a known, documented
    # heuristic-precision divergence. `hmax_explicit` cross-products an
    # effect's argument fluents' already-reachable values on the Rust side,
    # while Python's classifier over-approximates any non-constant object
    # effect to "every object of the type"; the two backends can legitimately
    # reach different (both admissible) values on these two problems' non-
    # constant object effects. `test_heuristic_values` pins both values under
    # backend-specific keys instead of asserting them equal; the generic
    # `test_heuristics`/`test_search_algorithms`, which only compare recorded
    # metrics across backends, have no such escape hatch, so the combination
    # is pruned here instead of forced to (dis)agree.
    (
        HMAX_EXPLICIT_BACKEND_DIVERGENT,
        lambda c: c.heuristic == "hmax_explicit",
        "hmax_explicit legitimately diverges between backends on this "
        "problem's non-constant object effect",
    ),
]


def prune_reason(
    problem, search, heuristic, weak_equality, symmetry_breaking
) -> str | None:
    """The reason this parametrization should be pruned, or `None` if it
    shouldn't be. See `PERFORMANCE_PRUNES`."""
    case = PruneCase(search, heuristic, weak_equality, symmetry_breaking)
    for names, predicate, reason in PERFORMANCE_PRUNES:
        matches_name = (
            problem.name == names if isinstance(names, str) else problem.name in names
        )
        if matches_name and predicate(case):
            return reason
    return None


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
    reason = prune_reason(
        problem, search_kind, heuristic, weak_equality, symmetry_breaking
    )
    if reason is not None:
        pytest.skip(reason)

    results = []
    for disable_rustamer in [True, False]:
        reload_tamerlite(disable_rustamer)
        for internal_heuristic_cache in [True, False]:
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
    """Regression-pins the heuristic values every heuristic (and its
    `_no_numbers`/`_inadmissible` variants where applicable) computes on
    generated states of every problem in `PROBLEMS`, asserting the two
    backends agree exactly -- except `hmax_explicit` on
    `HMAX_EXPLICIT_BACKEND_DIVERGENT`, which is deliberately excluded from
    that cross-check (see the `PERFORMANCE_PRUNES` entry for why) and instead
    recorded under backend-specific keys so the divergence stays pinned
    rather than silently accepted or forced to agree by weakening either
    implementation."""

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
            # `hmax_explicit` legitimately diverges between backends on these
            # problems (see `HMAX_EXPLICIT_BACKEND_DIVERGENT`); pin both
            # values under backend-specific keys instead of skipping.
            backend_divergent = (
                heuristic_name == "hmax_explicit"
                and problem.name in HMAX_EXPLICIT_BACKEND_DIVERGENT
            )
            reason = prune_reason(problem, "wastar", heuristic_name, True, True)
            if reason is not None and not backend_divergent:
                continue

            inadmissible_numeric_heuristic_flags = [False]
            if testing_utils.is_numeric_problem(problem) and heuristic_name in {
                "hff",
                "hadd",
                "hmax",
            }:
                inadmissible_numeric_heuristic_flags = [True, False]
            for inadmissible_numeric_heuristic in inadmissible_numeric_heuristic_flags:
                for internal_caching in [True, False]:
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

                    values_key = heuristic_name
                    if backend_divergent:
                        values_key += "_python" if disable_rustamer else "_rust"
                    elif inadmissible_numeric_heuristic:
                        values_key += "_inadmissible"
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

    reason = prune_reason(
        problem, search_kind, heuristic, weak_equality, symmetry_breaking
    )
    if reason is not None:
        pytest.skip(reason)

    def custom_heuristic(state: State):
        return 1

    results = []
    for disable_rustamer in [True, False]:
        reload_tamerlite(disable_rustamer)

        for internal_heuristic_cache in [True, False]:
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
    reason = prune_reason(
        problem, search_kind, heuristic, weak_equality, symmetry_breaking
    )
    if reason is not None:
        pytest.skip(reason)

    results = []
    for disable_rustamer in [True, False]:
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
    reason = prune_reason(
        problem,
        "multiqueue",
        heuristic=None,
        weak_equality=weak_equality,
        symmetry_breaking=symmetry_breaking,
    )
    if reason is not None:
        pytest.skip(reason)

    results = []
    for disable_rustamer in [True, False]:
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
@pytest.mark.parametrize("relevance_analysis", [True, False])
def test_search_space(problem, relevance_analysis):
    states = {}
    for disable_rustamer in [True, False]:
        reload_tamerlite(disable_rustamer)
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
            relevance_analysis=relevance_analysis,
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

    reason = prune_reason(
        problem, search_kind, heuristic, weak_equality, symmetry_breaking
    )
    if reason is not None:
        pytest.skip(reason)

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


# --- Cross-backend differential tests for `evaluate()` -----------------------
#
# Unlike `test_simplify` above, there was no test asserting the two
# `evaluate()` implementations agree. These build a `SearchSpace` directly
# (no actions), skipping `Encoder`/`Converter`: both backends' `initial_state()`
# accept a plain list of already-built constant nodes, and each backend's own
# `make_*_node` constructors already produce exactly what it expects.


def test_evaluate_fixed_cases():
    """Cross-backend differential test for `evaluate()`. Pins expected
    values for every shape that used to diverge between
    `src/tamerlite/core/search_space.py` and
    `crates/rustamer-base/src/expressions_utils.rs`:

    - `+ - * /` collapsing an integral result down to `int`/`Int`.
    - `Equals` between an integer and an integral-valued real agreeing on
      both backends even though `make_rational_constant_node` does *not*
      normalize a denominator-1 value at construction (a real-typed fluent
      holding an integral value stays a `Fraction`/`Rational`, see
      `fluent_real_3` below): Rust's `internal_evaluate` gives `Equals` a
      numeric fallback for exactly this case (mirroring `simplify`'s
      identical fallback), rather than relying on construction-time
      normalization to make the structural comparison always agree.
    - Division, including by zero.

    Plus one fluent read of every constant type (bool, int, real, object) as
    a baseline."""

    (
        F_BOOL_T,
        F_BOOL_F,
        F_INT_3,
        F_INT_6,
        F_INT_NEG2,
        F_INT_0,
        F_REAL_3,
        F_REAL_7_2,
        F_OBJ_L1,
        F_OBJ_L2,
    ) = range(10)

    for disable_rustamer in [True, False]:
        reload_tamerlite(disable_rustamer)
        from tamerlite.core import (
            SearchSpace,
            evaluate,
            make_bool_constant_node,
            make_fluent_node,
            make_int_constant_node,
            make_object_node,
            make_operator_node,
            make_rational_constant_node,
        )

        search_space = SearchSpace([], {}, [], None, None, None)
        # `make_*_node` is typed to return the general `ExpressionNode`
        # union (per `tamerlite/core/__init__.pyi`'s stub comment, the
        # pure-Python signatures are what the checker sees for both
        # backends); each call here always produces a leaf constant, which
        # is what `initial_state` actually wants (`list[ConstantNode]`).
        state = search_space.initial_state(
            cast(
                "list[ConstantNode]",
                [
                    make_bool_constant_node(True),
                    make_bool_constant_node(False),
                    make_int_constant_node(3),
                    make_int_constant_node(6),
                    make_int_constant_node(-2),
                    make_int_constant_node(0),
                    make_rational_constant_node(3, 1),
                    make_rational_constant_node(7, 2),
                    make_object_node(0),
                    make_object_node(1),
                ],
            )
        )

        bool_t = make_bool_constant_node(True)
        i1 = make_int_constant_node(1)
        i2 = make_int_constant_node(2)
        i3 = make_int_constant_node(3)
        i4 = make_int_constant_node(4)
        i7 = make_int_constant_node(7)
        r7_2 = make_rational_constant_node(7, 2)
        r1_2 = make_rational_constant_node(1, 2)
        r0_1 = make_rational_constant_node(0, 1)
        obj_l1 = make_object_node(0)
        f_bool_t = make_fluent_node(F_BOOL_T)
        f_bool_f = make_fluent_node(F_BOOL_F)
        f_int_3 = make_fluent_node(F_INT_3)
        f_int_6 = make_fluent_node(F_INT_6)
        f_int_neg2 = make_fluent_node(F_INT_NEG2)
        f_int_0 = make_fluent_node(F_INT_0)
        f_real_3 = make_fluent_node(F_REAL_3)
        f_real_7_2 = make_fluent_node(F_REAL_7_2)
        f_obj_l1 = make_fluent_node(F_OBJ_L1)
        f_obj_l2 = make_fluent_node(F_OBJ_L2)

        op = testing_utils._op_tree
        cases: list[tuple[str, object, tuple]] = [
            ("fluent_bool_true", f_bool_t, ("ok", "bool", True)),
            ("fluent_bool_false", f_bool_f, ("ok", "bool", False)),
            ("fluent_int_3", f_int_3, ("ok", "int", 3)),
            ("fluent_real_3", f_real_3, ("ok", "real", 3, 1)),
            ("fluent_real_7_2", f_real_7_2, ("ok", "real", 7, 2)),
            ("fluent_object_l1", f_obj_l1, ("ok", "object", 0)),
            ("real_minus_real_to_int", op("-", r7_2, r1_2), ("ok", "int", 3)),
            ("real_times_int_to_int", op("*", r1_2, i4), ("ok", "int", 2)),
            ("real_div_real_to_int", op("/", r7_2, r7_2), ("ok", "int", 1)),
            ("real_plus_int_stays_real", op("+", r7_2, i1), ("ok", "real", 9, 2)),
            ("int_minus_real_stays_real", op("-", i3, r1_2), ("ok", "real", 5, 2)),
            ("int_div_int_exact", op("/", f_int_6, i3), ("ok", "int", 2)),
            ("int_div_int_inexact", op("/", i7, i2), ("ok", "real", 7, 2)),
            ("negative_div", op("/", f_int_neg2, i4), ("ok", "real", -1, 2)),
            (
                "div_by_zero_fluent",
                op("/", i3, f_int_0),
                ("raise", "ZeroDivisionError"),
            ),
            (
                "div_by_zero_literal",
                op("/", i3, r0_1),
                ("raise", "ZeroDivisionError"),
            ),
            (
                "int_fluent_eq_nonintegral_real",
                op("==", f_int_3, r7_2),
                ("ok", "bool", False),
            ),
            (
                "real_fluent_eq_int_literal_numeric_fallback",
                op("==", f_real_3, i3),
                ("ok", "bool", True),
            ),
            (
                "real_fluent_eq_real_literal_exact",
                op("==", f_real_7_2, r7_2),
                ("ok", "bool", True),
            ),
            (
                "le_mixed_representations",
                op("<=", f_int_3, f_real_7_2),
                ("ok", "bool", True),
            ),
            (
                "lt_mixed_representations_false",
                op("<", f_real_3, f_int_3),
                ("ok", "bool", False),
            ),
            ("object_eq_same", op("==", f_obj_l1, obj_l1), ("ok", "bool", True)),
            (
                "object_eq_different",
                op("==", f_obj_l1, f_obj_l2),
                ("ok", "bool", False),
            ),
            (
                "and_over_fluent_and_literal",
                op("and", f_bool_t, bool_t),
                ("ok", "bool", True),
            ),
            ("not_fluent", op("not", f_bool_t), ("ok", "bool", False)),
        ]

        for name, tree, expected in cases:
            exp = testing_utils._flatten_expression_tree(tree, make_operator_node)
            got = testing_utils._evaluate_outcome(evaluate, exp, state)
            assert got == expected, (
                f"case {name!r} on {'python' if disable_rustamer else 'rust'} "
                f"backend: expected {expected}, got {got}"
            )


def test_evaluate_interpreted_function_normalization():
    """Cross-backend differential test for interpreted-function calls
    through `evaluate()`, covering every `IfReturnType`: a `REAL` return
    always stays a `Fraction`/`Rational` on both backends
    (`InterpretedFunctionNode.call`'s `REAL` branch,
    `interpreted_function_result`'s `Real` arm), even when the value is
    integral -- mirroring UP's own `Simplifier.walk_interpreted_function_exp`.
    Also covers `BOOL`/`OBJECT` returns and an object-typed argument, and a
    nested interpreted-function call (the outer call's argument is itself
    the inner call's `Fraction` result)."""

    for disable_rustamer in [True, False]:
        reload_tamerlite(disable_rustamer)
        from tamerlite.core import (
            IfReturnType,
            SearchSpace,
            evaluate,
            make_int_constant_node,
            make_interpreted_function_node,
            make_object_node,
        )

        search_space = SearchSpace([], {}, [], None, None, None)
        state = search_space.initial_state([])

        def to_real(x):
            return Fraction(x)

        def half(x):
            return Fraction(x, 2)

        def obj_identity(o):
            return make_object_node(o.object)

        def is_first_object(o):
            return o.object == 0

        three = make_int_constant_node(3)

        to_real_node = make_interpreted_function_node(to_real, IfReturnType.REAL, (0,))
        assert testing_utils._evaluate_outcome(
            evaluate, (three, to_real_node), state
        ) == (
            "ok",
            "real",
            3,
            1,
        )

        half_node = make_interpreted_function_node(half, IfReturnType.REAL, (0,))
        assert testing_utils._evaluate_outcome(evaluate, (three, half_node), state) == (
            "ok",
            "real",
            3,
            2,
        )

        obj0 = make_object_node(0)
        obj_identity_node = make_interpreted_function_node(
            obj_identity, IfReturnType.OBJECT, (0,)
        )
        assert testing_utils._evaluate_outcome(
            evaluate, (obj0, obj_identity_node), state
        ) == (
            "ok",
            "object",
            0,
        )

        is_first_object_node = make_interpreted_function_node(
            is_first_object, IfReturnType.BOOL, (0,)
        )
        assert testing_utils._evaluate_outcome(
            evaluate, (obj0, is_first_object_node), state
        ) == (
            "ok",
            "bool",
            True,
        )

        # Nested call: the outer `to_real` receives the inner one's `Fraction`
        # result directly (`Fraction(Fraction(3, 1)) == Fraction(3, 1)`).
        inner = make_interpreted_function_node(to_real, IfReturnType.REAL, (0,))
        outer = make_interpreted_function_node(to_real, IfReturnType.REAL, (1,))
        assert testing_utils._evaluate_outcome(
            evaluate, (three, inner, outer), state
        ) == (
            "ok",
            "real",
            3,
            1,
        )


def test_evaluate_random_differential():
    """Cross-backend differential test mirroring `test_simplify` above:
    generates the same 100 random expressions on both backends via
    `testing_utils.construct_expressions` and asserts `evaluate()` agrees on
    every one, including which ones raise `ZeroDivisionError`. Unlike the
    fixed matrix, this is bool-rooted (`construct_exp_rec`'s only root
    shape) and exercises whatever nesting of `and`/`or`/`==`/`<=`/`</+`/`-`/
    `*`/`/` the generator happens to produce, catching combinations the
    fixed matrix didn't think to write down -- in particular, `==` over a
    numeric subexpression is exactly the shape that used to surface the
    integral-rational structural-equality divergence."""

    num_expressions = 100
    results: dict[bool, list[tuple]] = {True: [], False: []}

    for disable_rustamer in [True, False]:
        reload_tamerlite(disable_rustamer)
        reload_package(testing_utils)
        from tamerlite.core import (
            SearchSpace,
            evaluate,
            make_bool_constant_node,
            make_int_constant_node,
        )
        from testing_utils import construct_expressions

        search_space = SearchSpace([], {}, [], None, None, None)
        # Matches `construct_exp_rec`/`construct_numeric_exp_rec`'s hardcoded
        # `make_fluent_node(0)` (bool) / `make_fluent_node(1)` (numeric).
        state = search_space.initial_state(
            cast(
                "list[ConstantNode]",
                [make_bool_constant_node(True), make_int_constant_node(5)],
            )
        )

        expressions = construct_expressions(num_expressions, max_depth=20)
        for exp in expressions:
            results[disable_rustamer].append(
                testing_utils._evaluate_outcome(evaluate, exp, state)
            )

    for i in range(num_expressions):
        assert results[True][i] == results[False][i], (
            f"expression #{i} disagrees between backends: "
            f"python={results[True][i]!r} rust={results[False][i]!r}"
        )


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


def test_relevance_analysis_keeps_duration_only_writer():
    """
    `_compute_relevant_actions`'s backward goal-dependency walk must count
    fluents read by an action's duration bounds as dependencies too, not just
    fluents read by its conditions/effect values -- otherwise an action whose
    sole role is to write a fluent read only by another action's duration
    (`tune` writing `charge`, which only `run`'s duration reads) is wrongly
    pruned as irrelevant. `get_problem_duration_fluent_relevance` makes this
    an actual solvability difference (not just a suboptimal plan): without
    `tune`, `run`'s duration bound is never satisfiable, so pruning `tune`
    makes the problem UNSOLVABLE.

    The `PlanValidator` round-trip additionally covers the ordering side of
    the same relationship: `tune` and `run` share no condition, only a
    duration-read against an effect-write on `charge`. `run`'s start event
    carries its duration bounds' fluents in its read set, which makes it
    mutex with `tune`'s write; without that the reconstructed plan places
    them at the same timestamp and an external validator sees `run` as
    inapplicable even though the search's own state sequence runs `tune`
    first.
    """
    problem = problems_generator.get_problem_duration_fluent_relevance()
    lifted_problem, ground_problem, map_back_action_instance = (
        testing_utils.compile_problem(problem)
    )

    for disable_rustamer in [True, False]:
        reload_tamerlite(disable_rustamer)

        encoder = Encoder(
            ground_problem,
            lifted_problem,
            map_back_action_instance,
            symmetry_breaking=False,
            compression_safe_actions=False,
            relevance_analysis=True,
        )
        assert encoder.relevant_actions is not None
        relevant_actions = {
            name
            for name, action in encoder.action_by_name.items()
            if action in encoder.relevant_actions
        }
        assert "tune" in relevant_actions
        assert "run" in relevant_actions

        search = tamerlite.SearchParams(
            search="wastar",
            heuristic="hff",
            weight=0.8,
            compression_safe_actions=False,
            relevance_analysis=True,
        )
        with OneshotPlanner(name="tamerlite", params={"search": search}) as planner:
            planner: tamerlite.engine.TamerLite
            res: PlanGenerationResult = planner.solve(problem, timeout=None)
            assert res.status == ResultStatus.SOLVED_SATISFICING

            with PlanValidator(problem_kind=problem.kind) as v:
                assert v.validate(problem, res.plan)


def test_dedup_relevant_fluents_excludes_bookkeeping_fluents():
    """
    `Encoder._compute_dedup_relevant_fluents` restricts the search's
    duplicate-state detection key to fluents that matter for state identity:
    the least fixpoint of a backward slice seeded from the fluents read by a
    precondition/effect-condition/goal/duration-bound, closed under "an
    effect's RHS matters only if the fluent it writes matters". That closure
    is what makes exclusion of an effect's own target fluent from its own
    right-hand side fall out for free: `_convert_effects` desugars
    `increase`/`decrease` into a self-referencing assignment (`cost := cost +
    1`), and without the closure a pure bookkeeping fluent that only bumps
    itself would trivially mark itself relevant and the reduction would
    collapse to `None` -- exactly the bug this analysis exists to avoid. The
    closure also correctly drops fluents that are only *transitively*
    bookkeeping (fluent `A` feeds only fluent `B`, and `B` is read nowhere) --
    see `get_problem_dedup_relevant_transitive`, which a one-step "everything
    read by an effect's RHS is relevant" rule would get wrong even with the
    same-fluent exclusion, since the two fluents involved are never equal.

    Covers all three dedup regimes that consume the reduced set
    (`core.search.state_representation`) via
    `get_problem_dedup_relevant_classical` (plain `not is_temporal` dedup
    path, direct self-reference), `get_problem_dedup_relevant_temporal`
    (temporal `weak_equality` dedup path), and
    `get_problem_dedup_relevant_transitive` (plain `not is_temporal` dedup
    path, transitive chain) -- see their docstrings for why each is shaped
    the way it is. Also covers `Encoder`'s `relevant_equality` flag: `False`
    must leave the reduction at `None` even on a fixture that would
    otherwise qualify. And covers the `is_temporal and not weak_equality`
    skip: that regime has no dedup at all, so `Encoder` must leave the
    reduction at `None` there too, by default, without needing
    `relevant_equality=False`.
    """
    classical = problems_generator.get_problem_dedup_relevant_classical()
    temporal = problems_generator.get_problem_dedup_relevant_temporal()
    transitive = problems_generator.get_problem_dedup_relevant_transitive()

    for disable_rustamer in [True, False]:
        reload_tamerlite(disable_rustamer)

        for problem, excluded_names, kept_names in [
            (classical, ["cost"], ["ready"]),
            (temporal, ["tcost"], ["charge", "done"]),
            (transitive, ["counter", "log"], ["ready"]),
        ]:
            lifted_problem, ground_problem, map_back_action_instance = (
                testing_utils.compile_problem(problem)
            )
            encoder = Encoder(
                ground_problem,
                lifted_problem,
                map_back_action_instance,
                symmetry_breaking=False,
                compression_safe_actions=False,
                relevance_analysis=True,
                # Required for the temporal fixture: `is_temporal and not
                # weak_equality` skips computing the reduction entirely (see
                # the dedicated check below). No-op for the classical fixture,
                # which ignores this flag.
                weak_equality=True,
            )
            assert encoder.dedup_relevant_fluents is not None
            assert len(encoder.dedup_relevant_fluents) < len(encoder.fluents)
            dedup_names = {encoder.fluents[i] for i in encoder.dedup_relevant_fluents}
            for excluded_name in excluded_names:
                assert excluded_name not in dedup_names
            for name in kept_names:
                assert name in dedup_names

            if problem is transitive:
                # Sharp regression guard for the transitive-chain case: the
                # pre-fixpoint one-step rule kept `counter` (read by `log`'s
                # RHS) even though `log` itself is never seeded as relevant.
                # Assert the exact set, not just the two checks above, so a
                # regression to the one-step rule is caught even if it
                # happens to keep `ready` and drop `log`.
                assert dedup_names == set(kept_names)

            # Round-trips through the PyO3 getter/setter on the Rust backend too.
            assert (
                encoder.search_space.dedup_relevant_fluents
                == encoder.dedup_relevant_fluents
            )

            # relevant_equality=False must skip the reduction entirely,
            # even though the fixture's own fluents would otherwise qualify --
            # this is the flag's whole purpose, distinct from a problem simply
            # having nothing to exclude (covered by the numeric_problem case
            # below).
            disabled_encoder = Encoder(
                ground_problem,
                lifted_problem,
                map_back_action_instance,
                symmetry_breaking=False,
                compression_safe_actions=False,
                relevance_analysis=True,
                relevant_equality=False,
            )
            assert disabled_encoder.dedup_relevant_fluents is None
            assert disabled_encoder.search_space.dedup_relevant_fluents is None

            if problem is temporal:
                # `is_temporal and not weak_equality` has no dedup at all, so
                # the reduction is never consulted there -- Encoder must skip
                # computing it, matching the `weak_equality=False` default
                # most temporal solves use. Contrast with the `weak_equality=True`
                # encoder above, built from the same fixture, whose reduction
                # is non-trivial: this isn't a coincidental None.
                weak_equality_default_encoder = Encoder(
                    ground_problem,
                    lifted_problem,
                    map_back_action_instance,
                    symmetry_breaking=False,
                    compression_safe_actions=False,
                    relevance_analysis=True,
                )
                assert weak_equality_default_encoder.dedup_relevant_fluents is None
                assert (
                    weak_equality_default_encoder.search_space.dedup_relevant_fluents
                    is None
                )

        # A problem where every fluent is read somewhere leaves the reduction
        # as None -- the pre-existing full-assignments dedup path stays
        # reachable unchanged.
        numeric_problem = problems_generator.get_problem_numeric()
        lifted_problem, ground_problem, map_back_action_instance = (
            testing_utils.compile_problem(numeric_problem)
        )
        encoder = Encoder(
            ground_problem,
            lifted_problem,
            map_back_action_instance,
            symmetry_breaking=False,
            compression_safe_actions=False,
            relevance_analysis=True,
        )
        assert encoder.dedup_relevant_fluents is None
        assert encoder.search_space.dedup_relevant_fluents is None

        # Solve the temporal problem under weak_equality=True, exercising the
        # new WeakEqState.fluents subset comparison end to end.
        search = tamerlite.SearchParams(
            search="wastar",
            heuristic="hff",
            weight=0.8,
            compression_safe_actions=False,
            relevance_analysis=True,
            weak_equality=True,
        )
        with OneshotPlanner(name="tamerlite", params={"search": search}) as planner:
            planner: tamerlite.engine.TamerLite
            res: PlanGenerationResult = planner.solve(temporal, timeout=None)
            assert res.status == ResultStatus.SOLVED_SATISFICING
            with PlanValidator(problem_kind=temporal.kind) as v:
                assert v.validate(temporal, res.plan)


def test_weak_equality_warns_on_non_temporal_problem():
    """`weak_equality` only changes dedup behavior on temporal problems --
    `WeakEqState`'s extra `todo` comparison (`core.search.WeakEqState.__eq__`)
    is a guaranteed no-op on a classical problem, since `SearchSpace._open_action`
    only ever populates `todo` inside its `is_temporal` branch. Setting
    `weak_equality=True` on a non-temporal problem is therefore always a no-op,
    most plausibly a leftover from reusing params built for a temporal problem
    -- worth a warning, unlike the `is_temporal and not weak_equality` case
    (`test_dedup_relevant_fluents_excludes_bookkeeping_fluents`), which is the
    default for every temporal solve and would make the warning pure noise.

    Covers both `SearchParams` and `MultiqueueParams`, since the check lives once
    in `_solve_ground_problem`, common to both branches, rather than duplicated
    inside `_get_search` (which `MultiqueueParams` never calls).
    """
    classical = problems_generator.get_problem_dedup_relevant_classical()
    temporal = problems_generator.get_problem_dedup_relevant_temporal()

    for disable_rustamer in [True, False]:
        reload_tamerlite(disable_rustamer)

        param_cases: list[tuple[type, dict[str, Any]]] = [
            (tamerlite.SearchParams, {"search": "wastar", "heuristic": "hff"}),
            (
                tamerlite.engine.MultiqueueParams,
                {"queues": [tamerlite.HeuristicParams(heuristic="hff")]},
            ),
        ]
        for params_cls, extra_kwargs in param_cases:
            # weak_equality=True on a non-temporal problem: warns.
            search = params_cls(
                weak_equality=True, compression_safe_actions=False, **extra_kwargs
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                with OneshotPlanner(name="tamerlite", params={"search": search}) as p:
                    res: PlanGenerationResult = p.solve(classical, timeout=None)
                    assert res.status == ResultStatus.SOLVED_SATISFICING
            messages = [str(w.message) for w in caught]
            assert any("weak_equality" in m for m in messages)

            # weak_equality=False (default) on the same non-temporal problem: silent.
            search = params_cls(compression_safe_actions=False, **extra_kwargs)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                with OneshotPlanner(name="tamerlite", params={"search": search}) as p:
                    res = p.solve(classical, timeout=None)
                    assert res.status == ResultStatus.SOLVED_SATISFICING
            messages = [str(w.message) for w in caught]
            assert not any("weak_equality" in m for m in messages)

            # weak_equality=True on a temporal problem: silent -- this is where
            # the flag actually does something.
            search = params_cls(
                weak_equality=True, compression_safe_actions=False, **extra_kwargs
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                with OneshotPlanner(name="tamerlite", params={"search": search}) as p:
                    res = p.solve(temporal, timeout=None)
                    assert res.status == ResultStatus.SOLVED_SATISFICING
            messages = [str(w.message) for w in caught]
            assert not any("weak_equality" in m for m in messages)


def test_temporal_no_start_event():
    """Every durative action must own an event at its start timepoint.

    The search opens an action when its first event fires and reads the
    duration bounds from that state, so an action whose first event sits
    elsewhere -- `finish`, which only has an `at end` condition/effect, or
    the degenerate `noop`, which has no conditions and no effects at all --
    would be sized from the wrong state. `finish`'s duration reads a fluent
    that `setup` overwrites at its end, so an encoding that loses the start
    event emits a 10-long `finish` that the validator rejects.
    """
    problem = problems_generator.get_problem_temporal_no_start_event()

    for disable_rustamer in [True, False]:
        reload_tamerlite(disable_rustamer)
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
        for name in ["noop", "setup", "finish"]:
            events = encoder.events[encoder.action_by_name[name]]
            assert events, f"{name} has no events"
            timing, _ = events[0]
            assert timing.is_from_start() and timing.delay == 0, (
                f"{name}'s first event is not at the start timepoint"
            )

        noop = encoder.action_by_name["noop"]
        assert noop in encoder.applicable_actions

        ss: SearchSpaceABC = encoder.search_space
        succ_state = ss.get_successor_state(ss.initial_state(), noop)
        assert succ_state is not None

        scheduled = ss.build_plan([noop])
        assert len(scheduled) == 1
        start, scheduled_action, duration = scheduled[0]
        # `scheduled_action` is reconstructed by `build_plan`, so it may not
        # be `noop` itself (the Rust backend's pyo3 `Action` doesn't wire up
        # value equality); compare by `idx` instead.
        assert scheduled_action.idx == noop.idx
        assert start == 0
        assert duration == Fraction(2)

        # Opening `finish` after `setup`'s end event reads `d == 10`, so the
        # scheduler must keep it there: the duration fluents are read at the
        # start event, which makes it mutex with the event writing them.
        setup = encoder.action_by_name["setup"]
        finish = encoder.action_by_name["finish"]
        scheduled = ss.build_plan([setup, setup, finish, finish])
        by_idx = {a.idx: (start, duration) for start, a, duration in scheduled}
        assert by_idx[setup.idx] == (Fraction(0), Fraction(20))
        finish_start, finish_duration = by_idx[finish.idx]
        assert finish_duration == Fraction(10)
        assert finish_start is not None and finish_start > Fraction(20)

        with OneshotPlanner(name="tamerlite") as planner:
            planner: tamerlite.engine.TamerLite
            res: PlanGenerationResult = planner.solve(problem, timeout=None)
            assert res.status == ResultStatus.SOLVED_SATISFICING

            with PlanValidator(problem_kind=problem.kind) as v:
                assert v.validate(problem, res.plan)

            assert isinstance(res.plan, TimeTriggeredPlan)


def test_temporal_condition_before_start_is_rejected():
    """An event the ICE fold maps before the action's own start would sort
    ahead of the start event and defeat the invariant that event 0 is the
    start, so the encoder must reject the action instead of encoding it.
    """
    problem = problems_generator.get_problem_temporal_condition_before_start()
    lifted_problem, ground_problem, map_back_action_instance = (
        testing_utils.compile_problem(problem)
    )

    for disable_rustamer in [True, False]:
        reload_tamerlite(disable_rustamer)
        from tamerlite.encoder import Encoder

        with pytest.raises(Exception, match="before the start of a durative action"):
            Encoder(
                ground_problem,
                lifted_problem,
                map_back_action_instance,
                symmetry_breaking=False,
                compression_safe_actions=False,
                relevance_analysis=False,
            )


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


def test_interpreted_functions_supported_kind():
    """`_build_problems()` only picks up UP's IF example problems because
    `TamerLite.supported_kind()` declares all five IF feature flags -- pin
    that declaration directly instead of re-deriving it indirectly through
    which problems happen to get filtered in."""
    kind = tamerlite.engine.TamerLite.supported_kind()
    assert kind.has_interpreted_functions_in_conditions()
    assert kind.has_interpreted_functions_in_boolean_assignments()
    assert kind.has_interpreted_functions_in_numeric_assignments()
    assert kind.has_interpreted_functions_in_object_assignments()
    assert kind.has_interpreted_functions_in_durations()


def test_converter_shares_interpreted_function_wrapper():
    """Two calls to the *same* interpreted function -- even with different
    argument expressions -- must share one wrapper callable, built once by
    `Converter._get_interpreted_function_wrapper` and reused for every
    occurrence."""

    reload_tamerlite(True)
    from tamerlite.converter import Converter
    from tamerlite.core.search_space import (
        InterpretedFunctionNode,
        MultiSet,
        ObjectNode,
        State,
        evaluate,
    )

    Loc = UserType("Loc")
    l1 = Object("l1", Loc)
    l2 = Object("l2", Loc)

    def is_l1(loc):
        return loc == l1

    IF_is_l1 = InterpretedFunction("is_l1", BoolType(), OrderedDict(loc=Loc), is_l1)

    at1 = Fluent("at1", Loc)
    at2 = Fluent("at2", Loc)
    problem = Problem("shared_if")
    problem.add_fluent(at1)
    problem.add_fluent(at2)
    problem.add_object(l1)
    problem.add_object(l2)

    exp1 = IF_is_l1(at1())
    exp2 = IF_is_l1(at2())
    assert exp1 is not exp2

    converter = Converter(
        problem,
        fluent_ids={"at1": 0, "at2": 1},
        object_ids={"l1": 0, "l2": 1},
        objects_by_id=[l1, l2],
    )
    converted1 = converter.convert(exp1)
    converted2 = converter.convert(exp2)

    node1 = converted1[-1]
    node2 = converted2[-1]
    assert isinstance(node1, InterpretedFunctionNode)
    assert isinstance(node2, InterpretedFunctionNode)
    assert node1.function is node2.function

    state = State([ObjectNode(0), ObjectNode(1)], None, {}, MultiSet(), 0, [])
    assert evaluate(converted1, state) is True
    assert evaluate(converted2, state) is False


def test_converter_shares_if_cache_across_converters_with_different_object_tables():
    """`Converter._if_cache` -- unlike `_if_wrappers`, see
    `Converter._get_interpreted_function_wrapper`'s docstring -- is safe to
    share across Converters even when their object numbering differs: it's
    keyed by the already-unwrapped, table-agnostic real argument (an
    `Object`, never an internal id), and the id<->`Object` translation always
    runs against the *calling* Converter's own tables, never a stale one
    baked into a shared closure. The two Converters below deliberately
    number `l1`/`l2` oppositely to exercise exactly that."""

    reload_tamerlite(True)
    from tamerlite.converter import Converter
    from tamerlite.core.search_space import MultiSet, ObjectNode, State, evaluate

    Loc = UserType("Loc")
    l1 = Object("l1", Loc)
    l2 = Object("l2", Loc)

    calls = []

    def is_l1(loc):
        calls.append(loc)
        return loc == l1

    IF_is_l1 = InterpretedFunction("is_l1", BoolType(), OrderedDict(loc=Loc), is_l1)

    at = Fluent("at", Loc)
    problem = Problem("shared_if_cache")
    problem.add_fluent(at)
    problem.add_object(l1)
    problem.add_object(l2)

    exp = IF_is_l1(at())

    if_cache: dict = {}
    converter_a = Converter(
        problem,
        fluent_ids={"at": 0},
        object_ids={"l1": 0, "l2": 1},
        objects_by_id=[l1, l2],
        if_cache=if_cache,
    )
    converter_b = Converter(
        problem,
        fluent_ids={"at": 0},
        object_ids={"l1": 1, "l2": 0},
        objects_by_id=[l2, l1],
        if_cache=if_cache,
    )

    converted_a = converter_a.convert(exp)
    converted_b = converter_b.convert(exp)

    # Same real object (l1), opposite internal ids under each converter.
    state_a_l1 = State([ObjectNode(0)], None, {}, MultiSet(), 0, [])  # a: 0 -> l1
    state_b_l1 = State([ObjectNode(1)], None, {}, MultiSet(), 0, [])  # b: 1 -> l1

    assert evaluate(converted_a, state_a_l1) is True
    assert evaluate(converted_b, state_b_l1) is True
    # The second call hit the cache populated by the first -- the real
    # callable ran exactly once, despite the two converters' opposite
    # numbering.
    assert calls == [l1]

    # A different real object (l2) must not collide with l1's cache entry.
    state_a_l2 = State([ObjectNode(1)], None, {}, MultiSet(), 0, [])  # a: 1 -> l2
    assert evaluate(converted_a, state_a_l2) is False
    assert calls == [l1, l2]


def test_interpreted_function_receives_real_argument_types():
    """The wrapped callable always receives real Python/`up.model` values --
    an `int`, a `Fraction`, a `bool`, and a real `up.model.Object` -- never an
    internal node (`FluentNode`, `ObjectNode`, ...), regardless of the
    argument's declared type. `test_converter_shares_interpreted_function_
    wrapper`/`..._if_cache_across_converters...` already exercise the object
    case implicitly (via `loc == l1`); this pins all four argument shapes at
    once and asserts their exact Python types, not just their values."""

    reload_tamerlite(True)
    from tamerlite.converter import Converter
    from tamerlite.core.search_space import MultiSet, ObjectNode, State, evaluate

    Loc = UserType("Loc")
    l1 = Object("l1", Loc)

    received = {}

    def record(i, r, b, o):
        received["i"] = i
        received["r"] = r
        received["b"] = b
        received["o"] = o
        return True

    IF_record = InterpretedFunction(
        "record",
        BoolType(),
        OrderedDict(i=IntType(), r=RealType(), b=BoolType(), o=Loc),
        record,
    )

    fi = Fluent("fi", IntType())
    fr = Fluent("fr", RealType())
    fb = Fluent("fb", BoolType())
    fo = Fluent("fo", Loc)
    problem = Problem("if_arg_types")
    problem.add_fluent(fi)
    problem.add_fluent(fr)
    problem.add_fluent(fb)
    problem.add_fluent(fo)
    problem.add_object(l1)

    exp = IF_record(fi(), fr(), fb(), fo())
    converter = Converter(
        problem,
        fluent_ids={"fi": 0, "fr": 1, "fb": 2, "fo": 3},
        object_ids={"l1": 0},
        objects_by_id=[l1],
    )
    converted = converter.convert(exp)

    state = State([3, Fraction(1, 2), True, ObjectNode(0)], None, {}, MultiSet(), 0, [])
    assert evaluate(converted, state) is True
    assert received["i"] == 3 and type(received["i"]) is int
    assert received["r"] == Fraction(1, 2) and isinstance(received["r"], Fraction)
    assert received["b"] is True
    assert received["o"] is l1


def test_interpreted_function_registry_reuses_func_id_for_shared_wrapper():
    """Rust registers each distinct wrapper callable once, deduped by
    `function.as_ptr()` (`crates/rustamer-base/src/interpreted_functions.rs::
    register_interpreted_function`), and only ever frees an entry once
    `tamerlite.converter.interpreted_function_scope`'s live-scope count
    returns to zero (none of this test's three `Encoder(...)` calls opens
    one, so nothing gets freed here) -- the registry's strong `Py<PyAny>`
    reference is exactly what keeps a still-registered pointer from being
    recycled and silently aliasing a different callable's `func_id`.
    Encoding the same problem twice (as TamerLite's anytime loop does on
    every re-solve) must therefore still evaluate correctly: the second
    `Encoder`'s IF nodes have to resolve through whichever `func_id` the
    registry assigns *this* time, not silently reuse a stale id from the
    first encoding.

    None of the three `Encoder(...)` calls below pass `if_wrappers`, so each
    gets a fresh, unshared dict and therefore a fresh `func_id` every time --
    the same behavior `TamerLite.engine.py` relies on across *unrelated*
    solves, as opposed to the shared-`if_wrappers` path it uses within one
    top-level solve."""

    reload_tamerlite(False)
    problem = problems_generator.get_problem_if_bool_condition()
    for _ in range(3):
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
        state = ss.initial_state()
        # `f` reads `IF_int_to_int(ithree)`, evaluated once per encoding;
        # if a stale/aliased `func_id` were ever resolved this would raise
        # or evaluate against the wrong callable instead of quietly working.
        states = generate_states(ss, state, num_states=10)
        assert len(states) >= 1


def test_interpreted_function_unsupported_return_type_raises():
    """`Converter.walk_interpreted_function_exp`'s return-type dispatch
    (bool/int/real/user-type) has no reachable fifth case through the public
    `InterpretedFunction` API -- `InterpretedFunction.__init__` only asserts
    the type is registered with the environment, not that it's one of those
    four. Simulate an unsupported return type the same way: build a normal,
    validly-constructed `InterpretedFunction` and then swap its declared
    return type for UP's own `TIME` type, which none of the four
    `is_*_type()` checks recognize (a hand-rolled fake type won't survive
    UP's own type-checker, which needs the *full* `Type` interface -- `TIME`
    is a real, fully-formed one that just isn't bool/int/real/user).
    `counter` is written by `bump`, so it isn't static and UP's Grounder
    can't constant-fold `IF_broken(counter)` away before it ever reaches
    TamerLite (same reasoning as `test_interpreted_function_in_fluent_
    argument_raises`)."""
    from unified_planning.model.types import TIME

    counter = Fluent("counter", IntType())
    IF_broken = InterpretedFunction(
        "broken", IntType(), OrderedDict(x=IntType()), lambda x: x
    )
    IF_broken._return_type = TIME

    bump = InstantaneousAction("bump")
    bump.add_effect(counter, Plus(counter, 1))
    act = InstantaneousAction("act")
    act.add_precondition(GE(IF_broken(counter), 1))

    problem = Problem("if_unsupported_return_type")
    problem.add_fluent(counter)
    problem.add_action(bump)
    problem.add_action(act)
    problem.set_initial_value(counter, 1)
    problem.add_goal(GE(counter, 1))

    for disable_rustamer in [True, False]:
        reload_tamerlite(disable_rustamer)
        search = tamerlite.SearchParams(search="gbfs", heuristic="blind")
        with (
            OneshotPlanner(name="tamerlite", params={"search": search}) as planner,
            pytest.raises(
                NotImplementedError,
                match="Unsupported interpreted function return type",
            ),
        ):
            planner.solve(problem, timeout=None)


def test_interpreted_function_bad_return_value_raises():
    """A callable declared to return an object but that actually returns
    something else (a user bug, not a modeling error) must fail loudly on
    both backends. Rust's `interpreted_function_result` raises a
    `PyValueError` when it can't extract an `ObjectNode`
    (`crates/rustamer-base/src/interpreted_functions.rs`). Python's own wrapper
    (`Converter._get_interpreted_function_wrapper`) doesn't validate the raw
    result before wrapping it -- `make_object_node(self._object_ids[raw_
    result.name])` -- so a non-`Object` return surfaces as a bare
    `AttributeError` right there (`src/tamerlite/converter.py`), not as the
    cleaner `assert isinstance(r, ObjectNode)` in `InterpretedFunctionNode.
    call`, which only runs for calls the wrapper doesn't already resolve to
    an object. `counter` is written by `bump`, so it isn't static and UP's
    own Grounder/Simplifier can't constant-fold `IF_broken(counter)` away
    first -- without that, the bad return would instead surface as an
    `AttributeError` from UP's own `Simplifier.walk_interpreted_function_exp`
    during grounding, one layer before TamerLite ever sees it."""
    Loc = UserType("Loc")
    l1 = Object("l1", Loc)
    l2 = Object("l2", Loc)

    def broken(x):
        return "not an object"

    IF_broken = InterpretedFunction(
        "broken_object_return", Loc, OrderedDict(x=IntType()), broken
    )

    counter = Fluent("counter", IntType())
    at = Fluent("at", Loc)
    bump = InstantaneousAction("bump")
    bump.add_effect(counter, Plus(counter, 1))
    act = InstantaneousAction("act")
    act.add_precondition(GE(counter, 0))
    act.add_effect(at, IF_broken(counter))

    problem = Problem("if_bad_return_value")
    problem.add_fluent(counter)
    problem.add_fluent(at)
    problem.add_object(l1)
    problem.add_object(l2)
    problem.add_action(bump)
    problem.add_action(act)
    problem.set_initial_value(counter, 0)
    # `at` starts at `l2`, so reaching the goal requires actually applying
    # `act` -- if it started at `l1` already, the goal would be trivially
    # true and `act`'s broken effect would never get evaluated at all.
    problem.set_initial_value(at, l2)
    problem.add_goal(Equals(at, l1))

    for disable_rustamer in [True, False]:
        reload_tamerlite(disable_rustamer)
        search = tamerlite.SearchParams(search="gbfs", heuristic="blind")
        with (
            OneshotPlanner(name="tamerlite", params={"search": search}) as planner,
            pytest.raises((AssertionError, ValueError, TypeError, AttributeError)),
        ):
            planner.solve(problem, timeout=None)


def test_hmax_explicit_partial_callable_can_raise():
    """`hmax_explicit` cross-products an effect's argument fluents'
    already-reachable values and evaluates the interpreted function on every
    combination (`heuristics.py`'s `HMaxExplicit` docstring;
    `crates/rustamer-base/src/heuristics.rs::possible_values`) -- including
    combinations that never jointly occur in any real reachable state. A
    *partial* callable (one that's only defined for some inputs) can
    therefore be called out-of-domain and raise, purely as an artifact of the
    over-approximation -- documented in `TODO.txt` as an open question, not a
    bug to silently swallow. hff/hadd/hmax, by contrast, evaluate an IF
    effect as an opaque `complex_numeric_effect` and never call the callable
    during heuristic computation at all, so they must NOT raise here."""

    Loc = UserType("Loc")
    l1 = Object("l1", Loc)
    l2 = Object("l2", Loc)

    def partial_next(loc, allowed):
        # Only defined when `allowed` is `True` -- `hmax_explicit`'s
        # reachable-value cross-product doesn't know that constraint and may
        # call this with `allowed=False` regardless.
        if not allowed:
            raise ValueError("next_loc is undefined when not allowed")
        return l2 if loc == l1 else l1

    IF_partial_next = InterpretedFunction(
        "partial_next", Loc, OrderedDict(loc=Loc, allowed=BoolType()), partial_next
    )

    at = Fluent("at", Loc)
    allowed = Fluent("allowed", BoolType())
    move = InstantaneousAction("move")
    move.add_precondition(allowed)
    move.add_effect(at, IF_partial_next(at, allowed))

    problem = Problem("if_hmax_explicit_partial_callable")
    problem.add_fluent(at)
    problem.add_fluent(allowed)
    problem.add_object(l1)
    problem.add_object(l2)
    problem.add_action(move)
    problem.set_initial_value(at, l1)
    problem.set_initial_value(allowed, True)
    problem.add_goal(Equals(at, l2))

    for disable_rustamer in [True, False]:
        reload_tamerlite(disable_rustamer)
        from tamerlite.core import HFF, HAdd, HMax

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

        heuristic_classes: list[Callable[..., Heuristic]] = [HFF, HAdd, HMax]
        for heuristic_class in heuristic_classes:
            heuristic: Heuristic = heuristic_class(
                encoder.actions,
                encoder.fluent_types,
                encoder.objects,
                encoder.events,
                encoder.goal,
                internal_caching=True,
                cache_value_in_state=False,
                inadmissible_numeric_heuristic_variant=False,
            )
            # Must not raise: the IF effect is opaque to these heuristics.
            heuristic.eval(init_state, ss)


def test_interpreted_functions_compression_safe_actions_reached():
    """`get_problem_if_temporal_compression_safe` exists specifically to be
    solved with `compression_safe_actions=True` -- exercising `Encoder.
    _compute_compression_safe_actions` and the `TimedToSequential` recompile
    on an IF problem, per its own docstring -- but nothing calls it with that
    flag directly; `test_search_algorithms`/`test_heuristics` only reach it
    via the generic `_compression_flags` matrix, indistinguishable from any
    other flag combination. Pin the specific case this problem was built
    for."""
    problem = problems_generator.get_problem_if_temporal_compression_safe()
    assert testing_utils.is_temporal_problem(problem)

    for disable_rustamer in [True, False]:
        reload_tamerlite(disable_rustamer)
        search = tamerlite.SearchParams(
            search="wastar", heuristic="hff", compression_safe_actions=True
        )
        with OneshotPlanner(name="tamerlite", params={"search": search}) as planner:
            planner: tamerlite.engine.TamerLite
            res: PlanGenerationResult = planner.solve(problem, timeout=None)
            assert res.status == ResultStatus.SOLVED_SATISFICING
            with PlanValidator(problem_kind=problem.kind) as v:
                assert v.validate(problem, res.plan)


def test_interpreted_functions_problem_not_picklable():
    """`InterpretedFunction.__getstate__` deliberately nulls out `_function`
    before pickling instead of letting the pickler raise on it (its own
    comment: "removing the function here so that pickler does not get mad at
    us... interpreted functions in parallel problems won't work") -- so
    `pickle.dumps` itself succeeds, but a round-tripped `InterpretedFunction`
    comes back with no callable at all. That's the real shape of "`Parallel`
    is off the table for any IF problem" (see `interpreted-functions-
    report.md`): the break surfaces the first time something tries to call
    the function, not at pickle time."""
    import pickle

    def double_it(x):
        return x * 2

    IF_double = InterpretedFunction(
        "double_it", IntType(), OrderedDict(x=IntType()), double_it
    )
    assert IF_double.function is double_it

    restored: InterpretedFunction = pickle.loads(pickle.dumps(IF_double))
    assert restored.function is None


def test_interpreted_functions_bounded_types_examples_excluded():
    """8 of UP's 13 bundled interpreted-function examples require
    `BOUNDED_TYPES`, which `TamerLite.supported_kind` doesn't declare at all
    -- a pre-existing gap unrelated to interpreted-function support (see
    `problems_generator.py`'s comment above the local `get_problem_if_*`
    builders). Pin the exact excluded set so this list visibly shrinks --
    rather than silently changing size -- if `BOUNDED_TYPES` support ever
    lands."""
    examples_module = unified_planning.test.examples.interpreted_functions_examples
    examples = examples_module.get_example_problems()
    excluded = {
        name
        for name, test_case in examples.items()
        if not tamerlite.engine.TamerLite.supports(test_case.problem.kind)
    }
    assert excluded == {
        "IF_in_conditions_complex_1",
        "go_home_with_rain_and_interpreted_functions",
        "interpreted_functions_in_conditions",
        "interpreted_functions_in_conditions_always_impossible",
        "interpreted_functions_in_durative_conditions",
        "interpreted_functions_in_boolean_assignment",
        "interpreted_functions_in_numeric_assignment",
        "interpreted_functions_in_durative_start_effects",
    }


def test_interpreted_functions_real_return_backend_normalization():
    """Both backends always keep a real-typed IF return as a `Fraction`/
    `Rational`, even when the value is integral: Rust's
    `interpreted_function_result`
    (`crates/rustamer-base/src/interpreted_functions.rs`) and Python's
    `InterpretedFunctionNode.call` (`src/tamerlite/core/search_space.py`)
    both mirror UP's own `Simplifier.walk_interpreted_function_exp`, which
    does the same unconditionally. This test's own `to_int` deliberately
    returns an integral value (`4`) so that's actually exercised --
    `get_problem_if_signature_shapes`'s real-return case just needs to prove
    the solve path works at all, not pin this specific rule.

    Observed through `evaluate()`, not `State.get_value()`: the latter
    returns the Rust backend's raw internal `ExpressionNode` wrapper, which
    doesn't compare equal to a plain Python `int`/`Fraction` at all --
    `evaluate()` is what actually normalizes a fluent read down to a native
    Python value on both backends."""

    def to_int(x):
        return Fraction(x)

    IF_to_int = InterpretedFunction(
        "to_int", RealType(), OrderedDict(x=IntType()), to_int
    )

    n = Fluent("n", IntType())
    problem = Problem("if_real_return_integral_probe")
    problem.add_fluent(n)
    problem.set_initial_value(n, 4)
    exp = IF_to_int(n())

    for disable_rustamer in [True, False]:
        reload_tamerlite(disable_rustamer)
        from tamerlite.converter import Converter
        from tamerlite.core import evaluate

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
        init_state = encoder.search_space.initial_state()
        converter = Converter(
            ground_problem,
            fluent_ids=encoder.fluent_ids,
            object_ids={},
            objects_by_id=[],
        )
        converted = converter.convert(exp)
        value = evaluate(converted, init_state)
        assert value == 4 and type(value) is Fraction


def test_simplify_with_interpreted_functions():
    """`simplify(..., evaluate_interpreted_functions=True)` is implemented in
    both backends (`src/tamerlite/core/search_space.py`,
    `crates/rustamer-base/src/expressions_utils.rs`) but called from nowhere
    in `src/tamerlite/` and, before this test, from nowhere in the test suite
    either -- `TODO.txt` records this gap explicitly ("test simplify with
    interpreted functions"). `testing_utils.parse_expression` has no syntax
    for an IF node, so this builds the postfix expression tuple directly with
    the real internal constructors, the same ones `Converter` itself uses.
    With the flag off (the default), a fully-constant IF call is still opaque
    data and must be left alone; with it on, it must fold away into its
    actual return value."""

    def double_it(x):
        return x * 2

    for disable_rustamer in [True, False]:
        reload_tamerlite(disable_rustamer)
        from tamerlite.core import (
            IfReturnType,
            make_int_constant_node,
            make_interpreted_function_node,
            simplify,
        )

        five = make_int_constant_node(5)
        if_node = make_interpreted_function_node(double_it, IfReturnType.INT, (0,))
        exp = (five, if_node)

        # Without the flag, the IF node is untouched: still length 2, its
        # operand (`5`) still live. (Comparing the raw element to a plain
        # `5` isn't portable -- unlike `evaluate()`, `simplify()`'s raw
        # output elements are the Rust backend's internal `ExpressionNode`
        # wrapper on that side, which doesn't support `==` against a plain
        # Python `int`; the shape check is the meaningful assertion here.)
        not_evaluated = tuple(simplify(exp, {}))
        assert len(not_evaluated) == 2

        # With it, the call folds away into its result and the now-dead
        # operand is dropped by simplify's reachable-node rebuild -- the
        # whole expression collapses to a single node holding `10`. (Same
        # portability note as above: compare the repr, not the raw value.)
        evaluated = tuple(simplify(exp, {}, evaluate_interpreted_functions=True))
        assert len(evaluated) == 1
        assert "10" in str(evaluated[0])


def test_interpreted_function_cache_avoids_python_call():
    """The Rust backend now memoizes interpreted-function results in
    `IF_RESULTS` (`crates/rustamer-base/src/interpreted_functions.rs`), keyed on
    `(func_id, return_type, args)`, in front of `call_interpreted_function` --
    so a second, identical call never touches Python at all. This registers
    the raw counting callable directly (no `Converter`, hence no
    `Converter._if_cache`), so any dedup observed here provably comes from
    the new Rust-side cache and not the pre-existing Python-side one.

    This is a deliberate, call-count-only divergence between backends: the
    pure-Python `evaluate`/`simplify` has no memo of its own
    (`InterpretedFunctionNode.call`, `src/tamerlite/core/search_space.py`),
    so it calls the raw callable every time. In production, both backends
    memoize -- Python via `Converter._if_cache`, Rust via `IF_RESULTS` -- this
    difference is only visible when a raw callable bypasses the wrapper, as
    in this test."""

    calls = []

    def double_it(x):
        calls.append(x)
        return x * 2

    for disable_rustamer in [True, False]:
        reload_tamerlite(disable_rustamer)
        from tamerlite.core import (
            IfReturnType,
            make_int_constant_node,
            make_interpreted_function_node,
            simplify,
        )

        calls.clear()
        five = make_int_constant_node(5)
        if_node = make_interpreted_function_node(double_it, IfReturnType.INT, (0,))
        exp = (five, if_node)

        simplify(exp, {}, evaluate_interpreted_functions=True)
        simplify(exp, {}, evaluate_interpreted_functions=True)

        if disable_rustamer:
            assert calls == [5, 5]
        else:
            assert calls == [5]


def test_interpreted_function_cache_distinguishes_return_type():
    """`make_interpreted_function_node` is a public constructor that can
    register the same raw callable under two different `IfReturnType`s (as
    this test does) -- something `Converter._get_interpreted_function_wrapper`
    never does in production, since one `InterpretedFunction` always has one
    declared `return_type`. `register_interpreted_function` dedups by pointer
    identity alone, so both registrations resolve to the same `func_id`; the
    Rust cache's key must still include `return_type`, or the two calls below
    would collide and the second would wrongly return the first's cached,
    differently-coerced result. `half_up` returns a non-integral value so the
    two coercions (`int()` truncates, `Fraction()` doesn't) actually produce
    different results -- if the cache ever collided on `func_id` alone, the
    second (real) call would wrongly come back truncated like the first."""

    def half_up(x):
        return x + Fraction(1, 2)

    for disable_rustamer in [True, False]:
        reload_tamerlite(disable_rustamer)
        from tamerlite.core import (
            IfReturnType,
            make_int_constant_node,
            make_interpreted_function_node,
            simplify,
        )

        three = make_int_constant_node(3)
        int_node = make_interpreted_function_node(half_up, IfReturnType.INT, (0,))
        real_node = make_interpreted_function_node(half_up, IfReturnType.REAL, (0,))

        int_result = tuple(
            simplify((three, int_node), {}, evaluate_interpreted_functions=True)
        )
        real_result = tuple(
            simplify((three, real_node), {}, evaluate_interpreted_functions=True)
        )
        assert len(int_result) == 1
        assert len(real_result) == 1
        assert str(int_result[0]) != str(real_result[0])


def test_interpreted_function_cache_is_scoped_to_object_table():
    """Rust counterpart of `test_converter_shares_if_cache_across_converters_
    with_different_object_tables`, which pins the same invariant for the
    pre-existing Python-side `_if_cache` (Python-only, via `reload_
    tamerlite(True)`). Here two `Converter`s number `l1`/`l2` oppositely and
    do NOT share an `if_cache`, so a correct result can only come from each
    Converter's own wrapper closure -- and therefore its own `func_id` in the
    Rust-side `IF_RESULTS` cache underneath it. `is_l1` discriminates on the
    real object's identity, not just round-trips it, so a leak between the
    two converters' cache entries would flip one result's boolean rather than
    silently agreeing. This is the regression test for the *default*
    (unshared) case: `_if_wrappers` may now be explicitly shared across
    Converters when the caller guarantees identical object numbering (see
    `TamerLite.engine.py`, and `_get_interpreted_function_wrapper`'s
    docstring for when that's sound) -- but absent that opt-in, each
    Converter still gets its own, independently-scoped wrapper/`func_id`,
    which is exactly what these two (neither passes `if_wrappers`) pin.

    Built directly from the wrapper + the real node constructors (as
    `test_simplify_with_interpreted_functions` does) and evaluated via
    `simplify`, not `evaluate`/`State`: Rust's `State` has no Python
    constructor, so tests can't build one directly for the Rust backend."""

    reload_tamerlite(False)
    from tamerlite.converter import Converter
    from tamerlite.core import (
        IfReturnType,
        make_interpreted_function_node,
        make_object_node,
        simplify,
    )

    Loc = UserType("Loc")
    l1 = Object("l1", Loc)
    l2 = Object("l2", Loc)

    def is_l1(loc):
        return loc == l1

    IF_is_l1 = InterpretedFunction("is_l1", BoolType(), OrderedDict(loc=Loc), is_l1)

    problem = Problem("if_cache_object_table_scoping")
    problem.add_object(l1)
    problem.add_object(l2)

    converter_a = Converter(
        problem,
        fluent_ids={},
        object_ids={"l1": 0, "l2": 1},
        objects_by_id=[l1, l2],
    )
    converter_b = Converter(
        problem,
        fluent_ids={},
        object_ids={"l1": 1, "l2": 0},
        objects_by_id=[l2, l1],
    )

    wrapper_a = converter_a._get_interpreted_function_wrapper(IF_is_l1)
    wrapper_b = converter_b._get_interpreted_function_wrapper(IF_is_l1)
    node_a = make_interpreted_function_node(wrapper_a, IfReturnType.BOOL, (0,))
    node_b = make_interpreted_function_node(wrapper_b, IfReturnType.BOOL, (0,))

    # oid 0 means l1 under converter_a and l2 under converter_b -- correct
    # results must diverge accordingly.
    obj0 = make_object_node(0)

    result_a = tuple(simplify((obj0, node_a), {}, evaluate_interpreted_functions=True))
    result_b = tuple(simplify((obj0, node_b), {}, evaluate_interpreted_functions=True))
    assert len(result_a) == 1 and len(result_b) == 1
    # Compare the repr, not the raw value -- same portability note as
    # `test_simplify_with_interpreted_functions`: `simplify()`'s raw output
    # elements are the Rust backend's internal `ExpressionNode` wrapper here,
    # which doesn't support `==` against a plain Python `bool`.
    assert "true" in str(result_a[0])
    assert "false" in str(result_b[0])


def test_clear_interpreted_function_cache_drops_memoized_results():
    """`clear_interpreted_function_cache` (`crates/rustamer-base/src/
    interpreted_functions.rs` on Rust, a no-op stub in
    `src/tamerlite/core/search_space.py` on Python)
    drops every entry of `IF_RESULTS` *and* resets the callable registry
    (`INTERPRETED_FUNCTIONS`/`IF_IDS_BY_PTR`) -- so a node built before the
    clear must never be reused afterward; only a freshly-built node (fresh
    registration) is safe post-clear. Registers the raw counting callable
    directly (no `Converter`), so there is no `Converter._if_cache` to
    confound the count: on Rust, the second `simplify` call (pre-clear) is a
    hit (no append); the fresh post-clear node is always a miss (an append)
    on both backends -- on Python, which has no memo of its own, every call
    is a miss regardless, proving the stub is importable, callable, and
    inert."""

    calls = []

    def double_it(x):
        calls.append(x)
        return x * 2

    for disable_rustamer in [True, False]:
        reload_tamerlite(disable_rustamer)
        from tamerlite.core import (
            IfReturnType,
            clear_interpreted_function_cache,
            make_int_constant_node,
            make_interpreted_function_node,
            simplify,
        )

        calls.clear()
        five = make_int_constant_node(5)
        if_node = make_interpreted_function_node(double_it, IfReturnType.INT, (0,))
        exp = (five, if_node)

        simplify(exp, {}, evaluate_interpreted_functions=True)
        simplify(exp, {}, evaluate_interpreted_functions=True)
        # Python has no memo: every call is a miss. Rust: the second call is
        # a cache hit.
        before_clear = 2 if disable_rustamer else 1
        assert len(calls) == before_clear

        clear_interpreted_function_cache()

        # A *fresh* node (fresh registration) for the post-clear call --
        # `if_node` above must never be reused now that the registry itself
        # gets reset too.
        fresh_if_node = make_interpreted_function_node(
            double_it, IfReturnType.INT, (0,)
        )
        simplify((five, fresh_if_node), {}, evaluate_interpreted_functions=True)
        # Clearing forces this call back to a miss on both backends (a
        # genuine miss on Rust, an unconditional miss on Python).
        assert len(calls) == before_clear + 1


def test_clear_interpreted_function_cache_resets_for_fresh_use():
    """`clear_interpreted_function_cache` resets the callable registry
    (`INTERPRETED_FUNCTIONS`/`IF_IDS_BY_PTR`), not just `IF_RESULTS` -- so
    anything built *before* a clear (a wrapper, a node) must never be
    evaluated again afterward; only freshly-built wrappers/nodes are safe.
    Clears first, then builds two Converters with opposite object numbering
    entirely *after* the reset (mirroring
    `test_interpreted_function_cache_is_scoped_to_object_table`), and
    confirms both still evaluate correctly and independently -- proving the
    reset leaves the registry in a clean, working state for subsequent,
    correctly-scoped use, not that pre-reset state survives (which it no
    longer does)."""

    reload_tamerlite(False)
    from tamerlite.converter import Converter
    from tamerlite.core import (
        IfReturnType,
        clear_interpreted_function_cache,
        make_interpreted_function_node,
        make_object_node,
        simplify,
    )

    clear_interpreted_function_cache()

    Loc = UserType("Loc")
    l1 = Object("l1", Loc)
    l2 = Object("l2", Loc)

    def is_l1(loc):
        return loc == l1

    IF_is_l1 = InterpretedFunction("is_l1", BoolType(), OrderedDict(loc=Loc), is_l1)

    problem = Problem("if_clear_resets_for_fresh_use")
    problem.add_object(l1)
    problem.add_object(l2)

    converter_a = Converter(
        problem,
        fluent_ids={},
        object_ids={"l1": 0, "l2": 1},
        objects_by_id=[l1, l2],
    )
    converter_b = Converter(
        problem,
        fluent_ids={},
        object_ids={"l1": 1, "l2": 0},
        objects_by_id=[l2, l1],
    )

    wrapper_a = converter_a._get_interpreted_function_wrapper(IF_is_l1)
    wrapper_b = converter_b._get_interpreted_function_wrapper(IF_is_l1)
    node_a = make_interpreted_function_node(wrapper_a, IfReturnType.BOOL, (0,))
    node_b = make_interpreted_function_node(wrapper_b, IfReturnType.BOOL, (0,))
    obj0 = make_object_node(0)

    result_a = tuple(simplify((obj0, node_a), {}, evaluate_interpreted_functions=True))
    assert "true" in str(result_a[0])

    result_b = tuple(simplify((obj0, node_b), {}, evaluate_interpreted_functions=True))
    assert "false" in str(result_b[0])


def test_shared_if_wrappers_reuse_rust_cache_across_converters():
    """The core new behavior: sharing one `if_wrappers` dict between two
    Converters (exactly what `TamerLite.engine.py` now does across every
    re-encode of one top-level solve) makes them resolve to the *identical*
    wrapper closure for the same `InterpretedFunction` -- and therefore the
    *same* Rust `func_id` -- so a result cached via `converter_a` is served
    to `converter_b` without a second Python-callable invocation. Contrast
    with `test_interpreted_function_cache_is_scoped_to_object_table`, which
    pins the opposite (default, unshared) behavior."""

    reload_tamerlite(False)
    from tamerlite.converter import Converter
    from tamerlite.core import (
        IfReturnType,
        make_int_constant_node,
        make_interpreted_function_node,
        simplify,
    )

    calls = []

    def double_it(x):
        calls.append(x)
        return x * 2

    IF_double = InterpretedFunction(
        "double_it", IntType(), OrderedDict(x=IntType()), double_it
    )

    problem = Problem("if_shared_wrappers")
    if_wrappers: dict = {}

    converter_a = Converter(
        problem, fluent_ids={}, object_ids={}, objects_by_id=[], if_wrappers=if_wrappers
    )
    converter_b = Converter(
        problem, fluent_ids={}, object_ids={}, objects_by_id=[], if_wrappers=if_wrappers
    )

    wrapper_a = converter_a._get_interpreted_function_wrapper(IF_double)
    wrapper_b = converter_b._get_interpreted_function_wrapper(IF_double)
    assert wrapper_a is wrapper_b

    five = make_int_constant_node(5)
    node_a = make_interpreted_function_node(wrapper_a, IfReturnType.INT, (0,))
    node_b = make_interpreted_function_node(wrapper_b, IfReturnType.INT, (0,))

    result_a = tuple(simplify((five, node_a), {}, evaluate_interpreted_functions=True))
    assert len(calls) == 1
    assert "10" in str(result_a[0])

    # Same logical call through converter_b's node: must be a Rust cache hit
    # (the wrapper is the same object, hence the same func_id), not a second
    # invocation of the real callable.
    result_b = tuple(simplify((five, node_b), {}, evaluate_interpreted_functions=True))
    assert len(calls) == 1
    assert "10" in str(result_b[0])


def test_encoder_threads_if_wrappers_to_converter():
    """`Encoder.__init__` must thread a caller-supplied `if_wrappers` dict
    through to `Converter` unchanged, by identity, not just by value -- this
    is the wiring that lets `TamerLite.engine.py` share wrapper closures
    (hence `func_id`s) across every re-encode of one solve. Builds two
    Encoders for the same problem sharing one `if_wrappers` dict and
    confirms both Converters hold that exact object, and that it actually
    got populated."""

    reload_tamerlite(False)
    problem = problems_generator.get_problem_if_bool_condition()
    if_cache: dict = {}
    if_wrappers: dict = {}
    for _ in range(2):
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
            if_cache=if_cache,
            if_wrappers=if_wrappers,
        )
        assert encoder._converter._if_wrappers is if_wrappers
        assert encoder._converter._if_cache is if_cache
    assert len(if_wrappers) > 0


def test_interpreted_function_cache_does_not_cache_errors():
    """A `PyErr` result must never be memoized: a replayed error would carry a
    stale traceback and could permanently poison a callable that raises once
    and later succeeds. Calling the same failing IF twice must raise both
    times, and the callable must actually run both times (`raises == 2`), not
    just once with the second call replaying a cached exception."""

    raises = []

    def always_fails(x):
        raises.append(x)
        raise ValueError("nope")

    for disable_rustamer in [True, False]:
        reload_tamerlite(disable_rustamer)
        from tamerlite.core import (
            IfReturnType,
            make_int_constant_node,
            make_interpreted_function_node,
            simplify,
        )

        raises.clear()
        five = make_int_constant_node(5)
        if_node = make_interpreted_function_node(always_fails, IfReturnType.INT, (0,))
        exp = (five, if_node)

        for _ in range(2):
            with pytest.raises(ValueError):
                simplify(exp, {}, evaluate_interpreted_functions=True)
        assert len(raises) == 2


def test_interpreted_function_reentrant_callable():
    """The Rust cache probes/inserts into a `thread_local! RefCell` around
    the actual Python call (`IF_RESULTS` in `interpreted_functions.rs`) -- the borrow
    must never be held while that call runs, since the callable is arbitrary
    Python and can re-enter `evaluate`/`simplify`, which can call back into
    `call_interpreted_function` for a *different* IF node. If the two borrow
    scopes (probe, insert) were ever merged into one, this would panic with
    "already borrowed" instead of returning a value."""

    def outer(x):
        from tamerlite.core import (
            IfReturnType,
            make_int_constant_node,
            make_interpreted_function_node,
            simplify,
        )

        def inner(y):
            return y + 1

        one = make_int_constant_node(1)
        inner_node = make_interpreted_function_node(inner, IfReturnType.INT, (0,))
        result = tuple(
            simplify((one, inner_node), {}, evaluate_interpreted_functions=True)
        )
        assert "2" in str(result[0])
        return x * 10

    for disable_rustamer in [True, False]:
        reload_tamerlite(disable_rustamer)
        from tamerlite.core import (
            IfReturnType,
            make_int_constant_node,
            make_interpreted_function_node,
            simplify,
        )

        five = make_int_constant_node(5)
        outer_node = make_interpreted_function_node(outer, IfReturnType.INT, (0,))
        result = tuple(
            simplify((five, outer_node), {}, evaluate_interpreted_functions=True)
        )
        assert len(result) == 1
        assert "50" in str(result[0])


def test_interpreted_function_real_arg_int_normalization():
    """A real-typed argument position can arrive as either `Int` or
    `Rational`, depending on whether a prior computation normalized it down
    (Rust's `interpreted_function_result` collapses an integral `Real` result
    to `Int`, `crates/rustamer-base/src/interpreted_functions.rs`). The Rust memo keys
    on the raw `ExpressionNode`, so `Int(3)` and `Rational(3, 1)` are two
    distinct cache entries -- this callable is invoked twice, once receiving
    a Python `int` and once a `Fraction`, never sharing a cached answer. This
    is finer-grained than the pre-existing Python-side `_if_cache`
    (`Converter._get_interpreted_function_wrapper`), whose dict key silently
    merges the two (`Fraction(3) == 3`, equal hashes) -- a pre-existing
    wrinkle this change neither fixes nor worsens.

    Uses `evaluate()`, not `simplify()`: `simplify`'s own node-rebuild
    normalizes any bare, integral `Rational` leaf down to `Int` as a general
    simplification rule, independent of interpreted functions
    (`expressions_utils.rs`), which would silently erase the very
    distinction this test needs. `internal_evaluate`'s leaf case does not
    (`other => (*other).clone()`), so a raw, un-normalized `Rational(3, 1)`
    node survives through to the interpreted-function call. `evaluate()`
    needs a real `State`, which the two expressions below never read (they
    contain no `Fluent` node) -- it exists only because Rust's `State` has
    no Python constructor, so one has to come from a real (otherwise
    irrelevant) `Encoder`."""

    received_types = []

    def record_type(x):
        received_types.append(type(x))
        return True

    reload_tamerlite(False)
    from tamerlite.core import (
        IfReturnType,
        evaluate,
        make_int_constant_node,
        make_interpreted_function_node,
        make_rational_constant_node,
    )

    problem = Problem("if_real_arg_int_normalization_dummy")
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
    state = encoder.search_space.initial_state()

    three_int = make_int_constant_node(3)
    three_real = make_rational_constant_node(3, 1)
    if_node_for_int = make_interpreted_function_node(
        record_type, IfReturnType.BOOL, (0,)
    )
    if_node_for_real = make_interpreted_function_node(
        record_type, IfReturnType.BOOL, (0,)
    )

    evaluate((three_int, if_node_for_int), state)
    evaluate((three_real, if_node_for_real), state)

    assert received_types == [int, Fraction]


def test_interpreted_functions_duration():
    problem = problems_generator.get_problem_if_duration()
    assert problem.kind.has_interpreted_functions_in_durations()

    for disable_rustamer in [True, False]:
        reload_tamerlite(disable_rustamer)

        search = tamerlite.SearchParams(
            search="wastar",
            heuristic="blind",
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
            # charge_time(battery=4) == 10 - 4 == 6
            assert duration == Fraction(6)


def test_interpreted_function_in_fluent_argument_raises():
    """`value(choose(sel))` addresses `value` through an interpreted-function
    call rather than a plain object -- no such grounded fluent exists in
    `problem.initial_values` (only `value(l1)`/`value(l2)` do), so encoding
    must fail loudly instead of crashing with a bare `KeyError`. `sel` is
    written by `pick`'s effect, so it stays non-static and `choose(sel)`
    survives grounding instead of being constant-folded away."""
    Loc = UserType("Loc")
    l1 = Object("l1", Loc)
    l2 = Object("l2", Loc)

    def choose(loc):
        return l2 if loc == l1 else l1

    IF_choose = InterpretedFunction("choose", Loc, OrderedDict(loc=Loc), choose)

    value = Fluent("value", IntType(), loc=Loc)
    sel = Fluent("sel", Loc)

    check = InstantaneousAction("check")
    check.add_precondition(GE(value(IF_choose(sel)), 2))
    pick = InstantaneousAction("pick")
    pick.add_effect(sel, l2)

    problem = Problem("if_in_fluent_argument")
    problem.add_fluent(value)
    problem.add_fluent(sel)
    problem.add_object(l1)
    problem.add_object(l2)
    problem.add_action(check)
    problem.add_action(pick)
    problem.set_initial_value(value(l1), 1)
    problem.set_initial_value(value(l2), 5)
    problem.set_initial_value(sel, l1)
    problem.add_goal(GE(value(l2), 2))

    for disable_rustamer in [True, False]:
        reload_tamerlite(disable_rustamer)
        search = tamerlite.SearchParams(search="gbfs", heuristic="blind")
        with (
            OneshotPlanner(name="tamerlite", params={"search": search}) as planner,
            pytest.raises(
                NotImplementedError, match="interpreted functions in its arguments"
            ),
        ):
            planner.solve(problem, timeout=None)


def test_nested_fluent_in_fluent_argument_raises():
    """`value(sel)` addresses `value` through another fluent rather than a
    plain object -- nothing in UP forbids this on the read side (only
    `Effect.__init__` rejects it, and only for effect *targets*), so it
    reaches TamerLite's encoder unchecked. No grounded `value(sel)` instance
    exists in `problem.initial_values` (only `value(l1)`/`value(l2)` do), so
    this must fail loudly rather than crash with a bare `KeyError`. `sel` is
    written by `pick`'s effect, so it stays non-static and survives
    grounding instead of being constant-folded away."""
    Loc = UserType("Loc")
    l1 = Object("l1", Loc)
    l2 = Object("l2", Loc)

    value = Fluent("value", IntType(), loc=Loc)
    sel = Fluent("sel", Loc)

    check = InstantaneousAction("check")
    check.add_precondition(GE(value(sel), 2))
    pick = InstantaneousAction("pick")
    pick.add_effect(sel, l2)

    problem = Problem("nested_fluent_in_fluent_argument")
    problem.add_fluent(value)
    problem.add_fluent(sel)
    problem.add_object(l1)
    problem.add_object(l2)
    problem.add_action(check)
    problem.add_action(pick)
    problem.set_initial_value(value(l1), 1)
    problem.set_initial_value(value(l2), 5)
    problem.set_initial_value(sel, l1)
    problem.add_goal(GE(value(l2), 2))

    for disable_rustamer in [True, False]:
        reload_tamerlite(disable_rustamer)
        search = tamerlite.SearchParams(search="gbfs", heuristic="blind")
        with (
            OneshotPlanner(name="tamerlite", params={"search": search}) as planner,
            pytest.raises(NotImplementedError, match="other fluents in its arguments"),
        ):
            planner.solve(problem, timeout=None)


def test_symmetry_breaking_interpreted_function_numeric_is_retained():
    """A numeric-only interpreted function (signature and return type both
    plain numbers) must not defeat symmetry breaking: `{s1, s2}` stays a
    legitimate equivalence class."""

    problem = problems_generator.get_problem_if_numeric_symmetry_retained()
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
    assert {"s1", "s2"} in [{obj.name for obj in group} for group in groups]


def test_symmetry_breaking_interpreted_function_object_argument_taint():
    """An interpreted function taking an object-typed argument must taint
    every object of that type: `l1`/`l2` must come back as singletons, not
    grouped together.

    This is also the soundness regression: before
    `Encoder._extract_interpreted_function_tainted_objects`, `{l1, l2}` were
    wrongly grouped (their initial state and goal are symmetric; the IF's
    distinguishing logic lives entirely in a Python closure invisible to
    `_extract_domain_objects`), which made `enter(l2)`/`finish(l2)` require
    an `l1`-using action already on the plan prefix -- but `enter(l1)` can
    never fire, so the goal became unreachable even though
    `enter(l2); finish(l2)` is a trivial valid plan. Solving with
    `symmetry_breaking=True` must still find and validate that plan.
    """

    problem = problems_generator.get_problem_if_object_argument_symmetry_unsound()
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
    group_sets = [{obj.name for obj in group} for group in groups]
    assert {"l1"} in group_sets
    assert {"l2"} in group_sets
    assert {"l1", "l2"} not in group_sets

    for disable_rustamer in [True, False]:
        reload_tamerlite(disable_rustamer)
        search = tamerlite.SearchParams(
            search="wastar", heuristic="blind", symmetry_breaking=True
        )
        with OneshotPlanner(name="tamerlite", params={"search": search}) as planner:
            planner: tamerlite.engine.TamerLite
            res: PlanGenerationResult = planner.solve(problem, timeout=None)
            assert res.status == ResultStatus.SOLVED_SATISFICING
            with PlanValidator(problem_kind=problem.kind) as v:
                assert v.validate(problem, res.plan)


def test_symmetry_breaking_interpreted_function_metric_action_cost_taint():
    """Same soundness bug as
    `test_symmetry_breaking_interpreted_function_object_argument_taint`, but
    the IF call lives only inside a `MinimizeActionCosts` quality metric
    (`Encoder._iter_metric_expressions`) rather than an action precondition:
    `l1`/`l2` are otherwise fully symmetric (same default-initial `visited`,
    a goal that doesn't mention either), so
    `_extract_interpreted_function_tainted_objects` must also scan
    quality-metric expressions for IF calls, or it wrongly groups
    `{l1, l2}` as equivalent."""

    problem = (
        problems_generator.get_problem_if_metric_action_cost_object_argument_taint()
    )
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
    group_sets = [{obj.name for obj in group} for group in groups]
    assert {"l1"} in group_sets
    assert {"l2"} in group_sets
    assert {"l1", "l2"} not in group_sets


def test_symmetry_breaking_interpreted_function_object_return_taint():
    """An object-returning interpreted function must taint every object of
    its return type (`{l1, l2}` become singletons), while leaving a second,
    unrelated type's genuine symmetry untouched (`{i1, i2}` stays grouped) --
    the taint must be scoped to the types an IF can actually observe or
    produce, not a blanket effect on the whole problem."""

    problem = problems_generator.get_problem_if_object_return_symmetry()
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
    group_sets = [{obj.name for obj in group} for group in groups]
    assert {"l1"} in group_sets
    assert {"l2"} in group_sets
    assert {"i1", "i2"} in group_sets


def test_symmetry_breaking_interpreted_function_hierarchical_type_argument_taint():
    """An interpreted function's tainted-type scan taints subtype instances
    too, not just objects declared exactly as the argument's declared
    (super)type: `is_heavy(vehicle: Vehicle)` must taint `v1` (a `Van`,
    a subtype of `Vehicle`) the same way it taints `c1` (a plain `Vehicle`)."""

    problem = problems_generator.get_problem_if_hierarchical_type_argument()
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
    group_sets = [{obj.name for obj in group} for group in groups]
    assert {"c1"} in group_sets
    assert {"v1"} in group_sets
    assert {"c1", "v1"} not in group_sets


def test_interpreted_functions_symmetry_breaking_and_relevance_analysis_not_disabled():
    problem = problems_generator.get_problem_if_bool_condition()
    for disable_rustamer in [True, False]:
        reload_tamerlite(disable_rustamer)
        search = tamerlite.SearchParams(
            search="gbfs",
            heuristic="blind",
            symmetry_breaking=True,
            relevance_analysis=True,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with OneshotPlanner(name="tamerlite", params={"search": search}) as planner:
                planner: tamerlite.engine.TamerLite
                res: PlanGenerationResult = planner.solve(problem, timeout=None)
                assert res.status == ResultStatus.SOLVED_SATISFICING

        messages = [str(w.message) for w in caught]
        assert not any("relevance_analysis" in m for m in messages)
        assert not any("symmetry_breaking" in m for m in messages)


def test_out_of_range_func_id_raises_instead_of_panicking():
    """`get_interpreted_function` (`interpreted_functions.rs`) used to index the
    registry unchecked (`registry[func_id]`), so a `func_id` past the current
    registry length aborted the whole process with a Rust panic surfaced as
    `pyo3_runtime.PanicException`. Registers two callables (ids 0 and 1),
    clears (which also empties the registry), then registers only *one*
    callable post-clear -- so the retained id-1 node's `func_id` is now out
    of range -- and asserts evaluating it raises an ordinary `RuntimeError`
    instead of panicking.

    Note this only exercises the *out-of-range* half of a stale `func_id`. An
    *in-range* stale id (recycled by the next epoch to a genuinely different
    callable) resolves silently to that callable by design -- pointer dedup,
    not id-range checking, is what keeps `func_id`s honest within one epoch,
    and nothing on the Rust side can tell "recycled" apart from "still
    mine". That's why `tamerlite.converter.interpreted_function_scope` (the
    live-solve counter gating when `clear_interpreted_function_cache`
    actually runs) is the real guard against stale-id resolution, not this
    error path."""

    reload_tamerlite(False)
    from tamerlite.core import (
        IfReturnType,
        clear_interpreted_function_cache,
        make_int_constant_node,
        make_interpreted_function_node,
        simplify,
    )

    def fn_x(v):
        return v

    def fn_y(v):
        return v

    five = make_int_constant_node(5)
    make_interpreted_function_node(fn_x, IfReturnType.INT, (0,))  # func_id 0
    node_y = make_interpreted_function_node(fn_y, IfReturnType.INT, (0,))  # func_id 1

    clear_interpreted_function_cache()

    def fn_z(v):
        return v

    make_interpreted_function_node(fn_z, IfReturnType.INT, (0,))  # fresh func_id 0

    with pytest.raises(RuntimeError, match="no longer registered"):
        simplify((five, node_y), {}, evaluate_interpreted_functions=True)


def _drive_if_counting_problem(name: str) -> list:
    """Drives `problems_generator.get_problem_if_counting_chain(name, ...)`
    through two anytime `next()` calls in isolation (no other solve ever in
    flight) and returns the resulting call log -- the baseline
    `test_interleaved_anytime_runs_keep_if_cache_warm` compares an
    interleaved run against. Needed because this problem's own
    heuristic/relevance-analysis machinery re-probes some interpreted-function
    values on every re-encode regardless of any caching bug (each solve calls
    the IF once per value the search actually needs to check reachability
    for), so a plain "must be exactly N calls" assertion can't distinguish
    that inherent, pre-existing cost from extra cost caused by interleaving
    -- only a same-problem, solo-vs-interleaved comparison can."""

    calls: list = []
    problem = problems_generator.get_problem_if_counting_chain(name, calls)
    search = tamerlite.SearchParams(search="gbfs", heuristic="hff")
    with AnytimePlanner(name="tamerlite", params={"search": search}) as planner:
        gen = planner.get_solutions(problem, timeout=None)
        next(gen)
        next(gen, None)
        gen.close()
    return calls


def test_interleaved_anytime_runs_keep_if_cache_warm():
    """Interleaving two anytime `get_solutions()` runs must not cost either
    one its own warm `IF_RESULTS` (Rust side): before this change,
    `clear_interpreted_function_cache()` ran unconditionally at the start of
    *every* top-level solve (`engine.py`, both the anytime and the oneshot
    path), so starting B's run while A's generator merely sat suspended
    between `yield`s wiped A's callable registry and result cache out from
    under it -- forcing A's next re-encode to recompute values that were
    already cached from its *own* first solve (confirmed empirically:
    pre-fix, an uninterrupted solo run of `get_problem_if_counting_chain`
    logs `[0, 1, 2, 3, 0, 1, 2]` -- the trailing `2` is as far as the second
    encode's own relevance probing needs to go, `3` stays a hit from the
    first solve -- while the same run interleaved with an unrelated anytime
    solve over a second such problem logs `[0, 1, 2, 3, 0, 1, 2, 3]`: the
    unrelated solve's start wiped the cache entry for `3`, forcing an extra,
    otherwise-unnecessary recompute).

    Builds two `get_problem_if_counting_chain`s, drives A interleaved with an
    entire unrelated anytime run over B, and asserts A's call log is
    *identical* to the solo baseline `_drive_if_counting_problem` produces
    for the same problem run alone -- i.e. interleaving added zero extra
    interpreted-function calls. Rust only -- the pure-Python backend has no
    registry for interleaving to disturb."""

    reload_tamerlite(False)

    baseline_calls = _drive_if_counting_problem("if_warm_baseline")

    calls_a: list = []
    calls_b: list = []
    problem_a = problems_generator.get_problem_if_counting_chain("if_warm_a", calls_a)
    problem_b = problems_generator.get_problem_if_counting_chain("if_warm_b", calls_b)
    search = tamerlite.SearchParams(search="gbfs", heuristic="hff")

    with (
        AnytimePlanner(name="tamerlite", params={"search": search}) as planner_a,
        AnytimePlanner(name="tamerlite", params={"search": search}) as planner_b,
    ):
        gen_a = planner_a.get_solutions(problem_a, timeout=None)
        res_a1 = next(gen_a)
        assert res_a1.status == ResultStatus.INTERMEDIATE
        assert calls_a == baseline_calls[: len(calls_a)]

        gen_b = planner_b.get_solutions(problem_b, timeout=None)
        res_b1 = next(gen_b)
        assert res_b1.status == ResultStatus.INTERMEDIATE

        res_a2 = next(gen_a, None)
        assert res_a2 is not None

        # An entire unrelated anytime solve ran to its first yield while A's
        # generator was merely suspended (not executing) in between -- A's
        # call log must come out identical to the uninterrupted baseline.
        assert calls_a == baseline_calls

        gen_a.close()
        gen_b.close()


def test_if_registrations_released_when_last_scope_exits():
    """`interpreted_function_scope` (`tamerlite.converter`) reclaims a run's
    IF registrations as soon as the live-scope count returns to zero --
    sooner than the old unconditional-clear-at-start design, which pinned a
    finished run's registrations until some later, unrelated solve happened
    to start. Registers a raw callable that captures a sentinel object via a
    default argument, exits the scope that registered it, drops every other
    reference, and asserts the sentinel is collected -- proving the Rust
    registry actually released its `Py<PyAny>` strong ref on the callable,
    not just that the Python side dropped its own.

    Deliberately bypasses the full engine/UP pipeline and calls
    `make_interpreted_function_node` directly: going through a real
    `InterpretedFunction`/`Problem` would confound the measurement, since
    UP's global expression manager permanently memoizes every `FNode` it
    ever builds (including one wrapping this callable) for the life of the
    process -- keeping the callable, and hence the sentinel, alive
    regardless of anything this change does."""

    reload_tamerlite(False)
    from tamerlite.converter import interpreted_function_scope
    from tamerlite.core import IfReturnType, make_interpreted_function_node

    class Sentinel:
        pass

    def register_and_discard() -> weakref.ReferenceType:
        # `sentinel`'s only binding lives in this frame -- once it returns,
        # nothing but the Rust registry (if it still holds `fn`) keeps the
        # sentinel alive.
        sentinel = Sentinel()
        sentinel_ref = weakref.ref(sentinel)

        def make_fn():
            def fn(x, _keep=sentinel):
                return x

            return fn

        with interpreted_function_scope():
            make_interpreted_function_node(make_fn(), IfReturnType.INT, (0,))

        return sentinel_ref

    sentinel_ref = register_and_discard()
    gc.collect()

    assert sentinel_ref() is None


def test_abandoned_anytime_generator_exits_its_scope():
    """`interpreted_function_scope`'s decrement runs in a `finally`, so it
    fires on `GeneratorExit` too -- not just on normal exhaustion -- which is
    what makes abandoning a suspended anytime generator (rather than
    explicitly `close()`-ing or draining it, the common real-world pattern)
    still release its live-scope count. Suspends a generator mid-run,
    abandons it without `close()`, forces GC, then runs a second, unrelated
    solve and confirms it isn't blocked or corrupted by anything the first
    left behind -- which would only be possible if the first's scope had
    actually exited."""

    reload_tamerlite(False)

    calls_a: list = []
    calls_b: list = []
    problem_a = problems_generator.get_problem_if_counting_chain(
        "if_abandoned_a", calls_a
    )
    problem_b = problems_generator.get_problem_if_counting_chain(
        "if_abandoned_b", calls_b
    )
    search = tamerlite.SearchParams(search="gbfs", heuristic="hff")

    with AnytimePlanner(name="tamerlite", params={"search": search}) as planner_a:
        gen_a = planner_a.get_solutions(problem_a, timeout=None)
        res_a1 = next(gen_a)
        assert res_a1.status == ResultStatus.INTERMEDIATE
        del gen_a  # abandoned mid-run, not closed

    gc.collect()

    with AnytimePlanner(name="tamerlite", params={"search": search}) as planner_b:
        gen_b = planner_b.get_solutions(problem_b, timeout=None)
        res_b1 = next(gen_b)
        assert res_b1.status == ResultStatus.INTERMEDIATE
        gen_b.close()


def test_interleaved_anytime_if_runs_produce_valid_plans():
    """End-to-end guard on the new plumbing (both backends): two anytime
    planners over the two IF fixtures, interleaved `next()` calls, every
    non-`None` plan validated against the original problem. This does *not*
    catch the original bug on its own (interleaving was already safe at the
    engine level, see `interpreted_function_scope`'s docstring) -- its job is
    to catch the opposite mistake this change could introduce: a scope that
    releases *too early*. If that happened, A's resumed `next()` would raise
    ("no longer registered") instead of returning a plan."""

    for disable_rustamer in [True, False]:
        reload_tamerlite(disable_rustamer)
        problem_a = problems_generator.get_problem_if_bool_condition()
        problem_b = problems_generator.get_problem_if_numeric_effect()
        search = tamerlite.SearchParams(
            search="gbfs", heuristic="hff", compression_safe_actions=False
        )

        with (
            AnytimePlanner(name="tamerlite", params={"search": search}) as planner_a,
            AnytimePlanner(name="tamerlite", params={"search": search}) as planner_b,
        ):
            gen_a = planner_a.get_solutions(problem_a, timeout=None)
            gen_b = planner_b.get_solutions(problem_b, timeout=None)

            results = [
                next(gen_a),
                next(gen_b),
                next(gen_a, None),
                next(gen_b, None),
            ]
            gen_a.close()
            gen_b.close()

        for problem, res in [
            (problem_a, results[0]),
            (problem_b, results[1]),
            (problem_a, results[2]),
            (problem_b, results[3]),
        ]:
            if res is None:
                continue
            assert res.status in {
                ResultStatus.INTERMEDIATE,
                ResultStatus.SOLVED_SATISFICING,
                ResultStatus.SOLVED_OPTIMALLY,
            }
            if res.plan is not None:
                with PlanValidator(problem_kind=problem.kind) as v:
                    val_res: ValidationResult = v.validate(problem, res.plan)
                    assert val_res


def test_oneshot_solve_during_suspended_anytime_run():
    """The scope must cover oneshot `solve()` too, not just anytime
    generators: a `solve()` call while an anytime generator sits suspended is
    exactly the second case the old unconditional-clear-at-start design
    could corrupt (`_solve_ground_problem` cleared unconditionally whenever
    it wasn't handed an existing `if_wrappers`, i.e. on every standalone
    oneshot call, regardless of what else was in flight). Suspends an
    anytime generator over one IF problem, runs a full, independent oneshot
    `solve()` over another, then resumes the suspended generator and asserts
    it still produces a valid plan instead of raising."""

    for disable_rustamer in [True, False]:
        reload_tamerlite(disable_rustamer)
        problem_a = problems_generator.get_problem_if_bool_condition()
        problem_b = problems_generator.get_problem_if_numeric_effect()
        search = tamerlite.SearchParams(
            search="gbfs", heuristic="hff", compression_safe_actions=False
        )

        with AnytimePlanner(name="tamerlite", params={"search": search}) as planner_a:
            gen_a = planner_a.get_solutions(problem_a, timeout=None)
            res_a1 = next(gen_a)
            assert res_a1.status == ResultStatus.INTERMEDIATE

            with OneshotPlanner(
                name="tamerlite", params={"search": search}
            ) as planner_b:
                res_b = planner_b.solve(problem_b, timeout=None)
                assert res_b.status == ResultStatus.SOLVED_SATISFICING

            res_a2 = next(gen_a, None)
            if res_a2 is not None and res_a2.plan is not None:
                with PlanValidator(problem_kind=problem_a.kind) as v:
                    val_res: ValidationResult = v.validate(problem_a, res_a2.plan)
                    assert val_res

            gen_a.close()
