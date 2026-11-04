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
"""Tests for strict separation: without a `Problem.epsilon`, the search keeps
mutex events apart by any positive amount `δ` instead of a fixed epsilon, and a
plan's times are reported at a concrete `δ` chosen once it is found. The
fixed-epsilon comparisons set `Problem.epsilon` to 1/100. Every test runs on
both cores, and `test_cross_backend_parity` checks that they solve
identically.
"""

import warnings
from fractions import Fraction

import pytest
from unified_planning.engines import PlanGenerationResult
from unified_planning.engines import PlanGenerationResultStatus as ResultStatus
from unified_planning.model import Problem
from unified_planning.plans import Plan, TimeTriggeredPlan
from unified_planning.shortcuts import AnytimePlanner, OneshotPlanner, PlanValidator

import problems_generator
import tamerlite
import testing_utils

reload_tamerlite = testing_utils.reload_tamerlite


@pytest.fixture(params=[True, False], ids=["python", "rust"])
def backend(request):
    reload_tamerlite(request.param)
    if not request.param and not tamerlite.core.use_rustamer:
        pytest.skip("rustamer is not installed")


_EPS = Fraction(1, 100)


def _with_epsilon(problem: Problem, epsilon: Fraction) -> Problem:
    problem = problem.clone()
    problem.epsilon = epsilon
    return problem


def _solve(problem: Problem, **params) -> PlanGenerationResult:
    search = tamerlite.SearchParams(**params)
    with OneshotPlanner(name="tamerlite", params={"search": search}) as planner:
        res: PlanGenerationResult = planner.solve(problem, timeout=60)
    return res


def _assert_valid(problem: Problem, plan) -> None:
    with PlanValidator(problem_kind=problem.kind) as validator:
        assert validator.validate(problem, plan)


def _timed_actions(plan: Plan | None) -> list[tuple[Fraction, str, Fraction | None]]:
    assert isinstance(plan, TimeTriggeredPlan)
    return [(start, ai.action.name, d) for start, ai, d in plan.timed_actions]


def _starts(plan: Plan | None) -> list[Fraction]:
    return sorted(start for start, _, _ in _timed_actions(plan))


def _encode(problem: Problem, **kwargs):
    lifted_problem, ground_problem, map_back_action_instance = (
        testing_utils.compile_problem(problem)
    )
    from tamerlite.encoder import Encoder

    return Encoder(
        ground_problem,
        lifted_problem,
        map_back_action_instance,
        symmetry_breaking=False,
        compression_safe_actions=False,
        relevance_analysis=False,
        **kwargs,
    )


def test_timed_instants_closer_than_epsilon(backend):
    """Timed instants closer than the problem's epsilon are reported as
    `UNSUPPORTED_PROBLEM`, and solved without one: the three mutex steps fit
    strictly inside the window, reported at `δ = 1/1000`."""
    problem = problems_generator.get_problem_tight_timed_window(3, Fraction(1, 200))
    res = _solve(_with_epsilon(problem, _EPS))
    assert res.status == ResultStatus.UNSUPPORTED_PROBLEM
    assert any("closer than epsilon" in m.message for m in res.log_messages or [])
    res = _solve(problem)
    assert res.status == ResultStatus.SOLVED_SATISFICING
    _assert_valid(problem, res.plan)
    assert _starts(res.plan) == [1 + Fraction(k, 1000) for k in range(1, 4)]


def test_delta_is_the_largest_power_of_ten_that_fits(backend):
    """Ten steps and two window ends need `11 δ <= 1/200`, so the plan is
    reported at `δ = 1/10000`, the largest power of ten that fits."""
    problem = problems_generator.get_problem_tight_timed_window(10, Fraction(1, 200))
    res = _solve(problem)
    assert res.status == ResultStatus.SOLVED_SATISFICING
    _assert_valid(problem, res.plan)
    assert _starts(res.plan) == [1 + Fraction(k, 10000) for k in range(1, 11)]


def test_plans_are_unchanged_when_an_epsilon_of_1_100_fits(backend):
    """When 1/100 fits, `δ` is 1/100 and the search makes the same choices, so
    the plan is the one found with `Problem.epsilon = 1/100`."""
    problem = problems_generator.get_problem_timed_effects_window()
    strict, fixed = _solve(problem), _solve(_with_epsilon(problem, _EPS))
    assert strict.status == fixed.status == ResultStatus.SOLVED_SATISFICING
    assert strict.plan == fixed.plan
    testing_utils.check_metrics_equality([strict, fixed])


_HALF = Fraction(1, 2)


@pytest.mark.parametrize(
    "start_delays, end_delays, lower, upper, right_open, dropped",
    [
        pytest.param(
            [0, Fraction(1, 200)], [], 1, 1, False, False, id="close_from_start"
        ),
        pytest.param([0], [0], 0, Fraction(1, 200), False, False, id="short_interval"),
        pytest.param([0], [0], 0, Fraction(1, 200), True, False, id="short_right_open"),
        pytest.param([0], [0], 0, 0, False, True, id="zero_duration"),
        pytest.param([_HALF], [0], 0, _HALF, False, True, id="closed_at_gap"),
        pytest.param([_HALF], [0], 0, _HALF, True, True, id="right_open_at_gap"),
        pytest.param(
            [_HALF],
            [0],
            0,
            _HALF + Fraction(1, 200),
            True,
            False,
            id="right_open_past_gap",
        ),
    ],
)
def test_durative_events_closer_than_epsilon(
    backend, start_delays, end_delays, lower, upper, right_open, dropped
):
    """Without a problem epsilon, a durative action whose events are closer
    than 1/100 is kept; only one whose events can't be apart at all, for any
    duration it allows, is dropped. An open upper bound costs nothing: a
    duration in `(gap, upper)` exists exactly when `upper > gap`, however
    little above it."""
    problem = problems_generator.get_problem_durative_event_gaps(
        start_delays, end_delays, lower, upper, right_open
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        encoder = _encode(problem)
    messages = [str(w.message) for w in caught if "never be applied" in str(w.message)]
    assert len(messages) == (1 if dropped else 0)
    names = {
        encoder.get_action_name(a)
        for a in encoder.applicable_actions
        if a != encoder.timed_action
    }
    assert ("act" not in names) == dropped


def test_open_duration_bounds_are_strict(backend):
    """A duration in `(1, 1 + 1/200)` has no room for an epsilon of 1/100 on
    either side, but does for `δ`: the plan reports it at `1 + δ`."""
    problem = problems_generator.get_problem_narrow_duration(1, 1 + Fraction(1, 200))
    # Compression would solve the sequential compilation, which drops the
    # duration, and then fail to schedule its plan.
    fixed = _solve(_with_epsilon(problem, _EPS), compression_safe_actions=False)
    assert fixed.plan is None
    res = _solve(problem)
    assert res.status == ResultStatus.SOLVED_SATISFICING
    _assert_valid(problem, res.plan)
    [(_, _, duration)] = _timed_actions(res.plan)
    assert duration == 1 + Fraction(1, 1000)


def test_symbolic_deadline(backend):
    """A symbolic deadline `(c, k)` asks for a plan shorter for every small
    enough `δ`: `(2, 1)` excludes `load` then `go` (makespan `2 + δ`) but not
    `load_and_go` (makespan 2), and `(2, 0)` excludes both."""
    problem = problems_generator.get_problem_strict_separation_chain()
    problem.clear_quality_metrics()
    from tamerlite.core import bfs_search

    encoder = _encode(problem)
    path, _ = bfs_search(encoder.search_space)
    assert path is not None
    assert encoder.build_plan(path)[1] in ((Fraction(2), 0), (Fraction(2), 1))

    encoder = _encode(problem, deadline=(Fraction(2), 1))
    path, _ = bfs_search(encoder.search_space)
    assert path is not None
    assert [encoder.get_action_name(a) for a in path] == ["load_and_go"] * 2
    assert encoder.build_plan(path)[1] == (Fraction(2), 0)

    encoder = _encode(problem, deadline=(Fraction(2), 0))
    path, _ = bfs_search(encoder.search_space)
    assert path is None


@pytest.mark.parametrize("c", [3, Fraction(7, 2)], ids=["int", "fraction"])
def test_delta_time_hashes_like_the_number_it_equals(c):
    """`DeltaTime(c, 0)` equals the plain number `c`, so it must hash like it,
    or a dict or set mixing the two (dominance's makespans, STN distances)
    would hold both as distinct keys. `_time` never builds one, but the
    constructor allows it. With `k != 0` it equals no plain number."""
    from tamerlite.core.search_space import DeltaTime

    assert DeltaTime(c, 0) == c
    assert hash(DeltaTime(c, 0)) == hash(c)
    assert len({DeltaTime(c, 0), c}) == 1
    assert DeltaTime(c, 1) != c
    assert len({DeltaTime(c, 1), c}) == 2


def test_feasible_delta_rejects_an_inconsistent_network():
    """The Python core's `_RecordingSTN.feasible_delta` reports an
    inconsistent network instead of failing on a timepoint the network
    stopped adding once inconsistent (2 here). The Rust core's
    `RecordingSTN` has the same check, tested in `search_space.rs`."""
    from tamerlite.core.search_space import DeltaTime, _RecordingSTN

    tn = _RecordingSTN()
    tn.add(0, 1, DeltaTime(0, -1))
    tn.add(1, 0, 0)
    tn.add(1, 2, 0)
    with pytest.raises(AssertionError, match="inconsistent"):
        tn.feasible_delta()


@pytest.mark.parametrize(
    "bound, go_start",
    [
        pytest.param(None, 1 + Fraction(1, 100), id="no_bound"),
        pytest.param(Fraction(3), 1 + Fraction(1, 100), id="loose"),
        pytest.param(Fraction(201, 100), 1 + Fraction(1, 1000), id="tight"),
        pytest.param(Fraction(2), 1 + Fraction(1, 100), id="unreachable"),
    ],
)
def test_makespan_bound_lowers_delta(backend, bound, go_start):
    """`build_plan`'s `makespan_bound`, the previous anytime plan's reported
    makespan, lowers `δ` until the plan's reported makespan is below it:
    `load` then `go` has makespan `2 + δ`, 2.01 at the default `δ`, so a bound
    of 2.01 reports it at `δ = 1/1000`. A bound no `δ` gets below leaves `δ`
    alone, and the symbolic makespan never changes."""
    problem = problems_generator.get_problem_strict_separation_chain(shortcut=False)
    problem.clear_quality_metrics()
    from tamerlite.core import bfs_search

    encoder = _encode(problem)
    path, _ = bfs_search(encoder.search_space)
    assert path is not None
    plan, makespan = encoder.build_plan(path, bound)
    assert makespan == (Fraction(2), 1)
    assert _timed_actions(plan) == [
        (Fraction(0), "load", Fraction(1)),
        (go_start, "go", Fraction(1)),
    ]


@pytest.mark.parametrize(
    "shortcut, expected",
    [
        pytest.param(True, [(0, "load_and_go", 2)], id="shortcut"),
        pytest.param(
            False,
            [(0, "load", 1), (1 + Fraction(1, 100), "go", 1)],
            id="strict_gap_only",
        ),
    ],
)
def test_anytime_makespan_terminates_optimally(backend, shortcut, expected):
    """The anytime makespan improvement bounds each search by the previous
    plan's symbolic makespan, so it proves the best plan optimal. Without the
    shortcut, every plan has makespan `2 + δ`: bounding the next search by
    the makespan at the reported `δ` instead would find it again and again.
    Compression-safe actions are off because they keep the engine from
    claiming makespan optimality."""
    problem = problems_generator.get_problem_strict_separation_chain(shortcut)
    search = tamerlite.SearchParams(compression_safe_actions=False)
    with AnytimePlanner(name="tamerlite", params={"search": search}) as planner:
        results = list(planner.get_solutions(problem, timeout=60))
    assert results[-1].status == ResultStatus.SOLVED_OPTIMALLY
    plan = results[-1].plan
    _assert_valid(problem, plan)
    assert _timed_actions(plan) == expected


@pytest.mark.parametrize(
    "make_problem",
    [
        pytest.param(
            lambda: problems_generator.get_problem_tight_timed_window(
                10, Fraction(1, 200)
            ),
            id="tight_timed_window",
        ),
        pytest.param(
            lambda: problems_generator.get_problem_narrow_duration(
                1, 1 + Fraction(1, 200)
            ),
            id="narrow_duration",
        ),
        pytest.param(
            problems_generator.get_problem_timed_effects_window, id="timed_window"
        ),
    ],
)
def test_cross_backend_parity(make_problem):
    """Both cores expand the same states and report the same plan without a
    problem epsilon."""
    problem = make_problem()
    results = []
    for disable_rustamer in [True, False]:
        reload_tamerlite(disable_rustamer)
        results.append(_solve(problem))
    testing_utils.check_metrics_equality(results)
    assert results[0].plan == results[1].plan
