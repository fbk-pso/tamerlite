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

"""Tests for `TamerLite`'s Heuristic operation mode (`Heuristic(problem,
name="tamerlite", params=...)`)."""

from collections.abc import Callable

import pytest
from unified_planning.engines import HeuristicGuarantee
from unified_planning.engines import PlanGenerationResultStatus as ResultStatus
from unified_planning.engines.compilers.grounder import Grounder
from unified_planning.exceptions import UPUsageError
from unified_planning.model import UPState
from unified_planning.model.multi_agent import MultiAgentProblem
from unified_planning.shortcuts import (
    Heuristic,
    OneshotPlanner,
    PlanValidator,
    Problem,
    SequentialSimulator,
    get_environment,
)

import problems_generator
import tamerlite
import tamerlite.converter
import tamerlite.encoder
from testing_utils import reload_tamerlite

get_environment().credits_stream = None

HEURISTICS = ["hff", "hadd", "hmax"]


def _params(heuristic: str, **kwargs) -> dict[str, tamerlite.HeuristicModeParams]:
    """`Heuristic`'s `params` selecting `heuristic`; read through `tamerlite`
    at call time, so it is the class of the currently loaded backend."""
    return {"heuristic": tamerlite.HeuristicModeParams(heuristic=heuristic, **kwargs)}


CLASSICAL_PROBLEMS: dict[str, Callable[[], Problem]] = {
    "visit_all": problems_generator.get_problem_visit_all,
    "counter": problems_generator.get_problem_counter,
    "numeric": problems_generator.get_problem_numeric,
    "flight": problems_generator.get_problem_flight,
    "object_equality": problems_generator.get_problem_object_equality_fluents,
}

PROBLEMS: dict[str, Callable[[], Problem]] = {
    **CLASSICAL_PROBLEMS,
    "temporal_flight": problems_generator.get_problem_temporal_flight,
    "if_bool_condition": problems_generator.get_problem_if_bool_condition,
}


def _core_value(problem: Problem, heuristic: str) -> float | None:
    """The core heuristic's value on the initial state of an encoding built
    with the heuristic mode's own options -- the oracle for `value`."""
    from tamerlite.core import HFF, HAdd, HMax

    ground = Grounder().compile(problem)
    assert isinstance(ground.problem, Problem)
    assert ground.map_back_action_instance is not None
    encoder = tamerlite.encoder.Encoder(
        ground.problem,
        problem,
        ground.map_back_action_instance,
        symmetry_breaking=False,
        compression_safe_actions=False,
        relevance_analysis=False,
        relevant_equality=False,
    )
    assert encoder.goal is not None
    h = {"hff": HFF, "hadd": HAdd, "hmax": HMax}[heuristic](
        encoder.considered_actions,
        encoder.fluent_domains,
        encoder.events,
        encoder.goal,
        internal_caching=False,
        cache_value_in_state=False,
        inadmissible_numeric_heuristic_variant=False,
    )
    ss = encoder.search_space
    return h.eval(ss.initial_state(), ss)


def _plan_states(problem: Problem) -> list[UPState]:
    """The states a TamerLite plan for classical `problem` goes through."""
    with OneshotPlanner(name="tamerlite") as planner:
        res = planner.solve(problem)
    assert res.status == ResultStatus.SOLVED_SATISFICING
    assert res.plan is not None
    with SequentialSimulator(problem) as sim:
        states = [sim.get_initial_state()]
        for ai in res.plan.actions:
            states.append(sim.apply(states[-1], ai))
        assert sim.is_goal(states[-1])
    return states


@pytest.mark.parametrize("disable_rustamer", [True, False])
@pytest.mark.parametrize("heuristic", HEURISTICS)
@pytest.mark.parametrize("problem_name", PROBLEMS)
def test_initial_value_matches_core(disable_rustamer, heuristic, problem_name):
    """`value` of the problem's initial state, given as a UP state, is the
    core heuristic's value on the encoder's own initial state."""
    reload_tamerlite(disable_rustamer)
    problem = PROBLEMS[problem_name]()
    with Heuristic(problem, name="tamerlite", params=_params(heuristic)) as h:
        value = h.value(UPState(problem.initial_values, problem))
    assert value == _core_value(problem, heuristic)


@pytest.mark.parametrize("heuristic", HEURISTICS)
@pytest.mark.parametrize("problem_name", CLASSICAL_PROBLEMS)
def test_values_along_plan(heuristic, problem_name):
    """Along a plan, both backends agree, every value is finite, the goal
    state is 0 (GOAL_AWARE), and hmax never exceeds the remaining number of
    actions."""
    problem = CLASSICAL_PROBLEMS[problem_name]()
    values = []
    for disable_rustamer in [True, False]:
        reload_tamerlite(disable_rustamer)
        states = _plan_states(problem)
        with Heuristic(problem, name="tamerlite", params=_params(heuristic)) as h:
            values.append([h.value(s) for s in states])
    assert values[0] == values[1]
    backend_values = values[0]
    assert all(v is not None for v in backend_values)
    assert backend_values[-1] == 0
    if heuristic == "hmax":
        remaining = len(backend_values) - 1
        for i, v in enumerate(backend_values):
            assert v is not None and v <= remaining - i


@pytest.mark.parametrize("disable_rustamer", [True, False])
@pytest.mark.parametrize("heuristic", HEURISTICS)
def test_dead_end(disable_rustamer, heuristic):
    reload_tamerlite(disable_rustamer)
    problem = problems_generator.get_problem_unreachable_goal()
    with Heuristic(problem, name="tamerlite", params=_params(heuristic)) as h:
        assert h.value(UPState(problem.initial_values, problem)) is None


@pytest.mark.parametrize("disable_rustamer", [True, False])
def test_temporal_goal_state(disable_rustamer):
    """A temporal problem's state is evaluated with nothing in progress: one
    that already satisfies the goal is 0."""
    reload_tamerlite(disable_rustamer)
    problem = problems_generator.get_problem_temporal_flight()
    at = problem.fluent("at")
    values = dict(problem.initial_values)
    values[at(problem.object("A"))] = problem.environment.expression_manager.FALSE()
    values[at(problem.object("D"))] = problem.environment.expression_manager.TRUE()
    for heuristic in HEURISTICS:
        with Heuristic(problem, name="tamerlite", params=_params(heuristic)) as h:
            assert h.value(UPState(values, problem)) == 0


@pytest.mark.parametrize("disable_rustamer", [True, False])
def test_params(disable_rustamer):
    """Every keyword reaches the engine through `params`."""
    reload_tamerlite(disable_rustamer)
    problem = problems_generator.get_problem_numeric()
    state = UPState(problem.initial_values, problem)
    for heuristic in HEURISTICS:
        for params in (
            {"disable_numeric_reasoning": True},
            {"inadmissible_numeric_heuristic_variant": True},
            {"internal_heuristic_cache": True},
        ):
            with Heuristic(
                problem,
                name="tamerlite",
                params=_params(heuristic, **params),
            ) as h:
                assert h.value(state) is not None
                # A second evaluation of the same state (a cache hit with
                # `internal_heuristic_cache`) returns the same value.
                assert h.value(state) == h.value(state)


def test_guarantees():
    TamerLite = tamerlite.engine.TamerLite
    assert TamerLite.is_heuristic()
    assert TamerLite.guarantees(HeuristicGuarantee.GOAL_AWARE)
    assert TamerLite.guarantees(HeuristicGuarantee.SAFE)
    assert TamerLite.guarantees(HeuristicGuarantee.GOAL_AWARE | HeuristicGuarantee.SAFE)
    assert not TamerLite.guarantees(HeuristicGuarantee.ADMISSIBLE)
    assert not TamerLite.guarantees(HeuristicGuarantee.CONSISTENT)
    assert not TamerLite.guarantees(
        HeuristicGuarantee.GOAL_AWARE | HeuristicGuarantee.ADMISSIBLE
    )


def test_rejections():
    TamerLite = tamerlite.engine.TamerLite
    problem = problems_generator.get_problem_visit_all()
    HeuristicModeParams = tamerlite.HeuristicModeParams
    with pytest.raises(UPUsageError, match="Unknown heuristic"):
        TamerLite(
            heuristic=HeuristicModeParams(heuristic="hmax_explicit"), problem=problem
        )
    with pytest.raises(UPUsageError, match="needs a problem"):
        TamerLite(heuristic=HeuristicModeParams())
    with pytest.raises(UPUsageError, match="must be a HeuristicModeParams"):
        TamerLite(heuristic="hadd", problem=problem)  # type: ignore[arg-type]
    with pytest.raises(UPUsageError, match="must be a SearchParams"):
        TamerLite(search=HeuristicModeParams(), problem=problem)  # type: ignore[arg-type]
    with pytest.raises(UPUsageError, match="only supports a Problem"):
        TamerLite(problem=MultiAgentProblem("agents"))
    with pytest.raises(TypeError, match="weight"):
        TamerLite(problem=problem, weight=0.8)  # type: ignore[call-arg]
    with pytest.raises(UPUsageError, match="not built as a heuristic"):
        TamerLite().value(UPState(problem.initial_values, problem))
    h = TamerLite(problem=problem)
    h.destroy()
    with pytest.raises(UPUsageError, match="destroyed"):
        h.value(UPState(problem.initial_values, problem))
    with pytest.raises(UPUsageError, match="undefined initial values"):
        TamerLite(problem=problems_generator.get_problem_if_undefined_initial_numeric())


def test_interpreted_function_scope():
    """An engine over a problem with interpreted functions keeps the
    interpreted-function scope open until it is destroyed; one without them,
    or a planner, never opens it."""
    problem = problems_generator.get_problem_if_bool_condition()
    before = tamerlite.converter._live_if_scopes
    with Heuristic(problem, name="tamerlite") as h:
        assert tamerlite.converter._live_if_scopes == before + 1
        assert h.value(UPState(problem.initial_values, problem)) is not None
    assert tamerlite.converter._live_if_scopes == before
    with Heuristic(problems_generator.get_problem_visit_all(), name="tamerlite"):
        assert tamerlite.converter._live_if_scopes == before
    with OneshotPlanner(name="tamerlite"):
        assert tamerlite.converter._live_if_scopes == before
    # A solve on an instance that is also a heuristic leaves the heuristic's
    # scope open, so its `func_id`s stay valid after the solve.
    engine = tamerlite.engine.TamerLite(problem=problem)
    state = UPState(problem.initial_values, problem)
    value = engine.value(state)
    assert engine.solve(problem).status == ResultStatus.SOLVED_SATISFICING
    assert tamerlite.converter._live_if_scopes == before + 1
    assert engine.value(state) == value
    engine.destroy()
    assert tamerlite.converter._live_if_scopes == before


def test_heuristic_drives_planner():
    """The engine's `value` is a valid custom heuristic for the planner.

    The planner's states must keep every fluent the engine reads, i.e. every
    non-static one: `relevant_equality=False`, as for any custom heuristic
    reading a fluent the search itself never reads (`fuel_used` here)."""
    problem = problems_generator.get_problem_flight()
    search = tamerlite.SearchParams(relevant_equality=False)
    with (
        Heuristic(problem, name="tamerlite") as h,
        OneshotPlanner(name="tamerlite", params={"search": search}) as planner,
    ):
        res = planner.solve(problem, heuristic=h.value)
    assert res.status == ResultStatus.SOLVED_SATISFICING
    with PlanValidator(problem_kind=problem.kind) as v:
        assert v.validate(problem, res.plan)


@pytest.mark.parametrize("disable_rustamer", [True, False])
def test_planner_and_heuristic_interleaved(disable_rustamer):
    """One instance is both a planner (`search`) and a heuristic (`heuristic`),
    and its `value` can drive its own `solve`."""
    reload_tamerlite(disable_rustamer)
    problem = problems_generator.get_problem_flight()
    state = UPState(problem.initial_values, problem)
    with tamerlite.engine.TamerLite(
        search=tamerlite.SearchParams(relevant_equality=False),
        heuristic=tamerlite.HeuristicModeParams(heuristic="hadd"),
        problem=problem,
        # As `Heuristic(problem, name="tamerlite")`: the flight problem's kind
        # is not covered by `supported_kind`.
        error_on_failed_checks=False,
    ) as engine:
        value = engine.value(state)
        plans = [engine.solve(problem), engine.solve(problem, heuristic=engine.value)]
        assert engine.value(state) == value
    with PlanValidator(problem_kind=problem.kind) as v:
        for res in plans:
            assert res.status == ResultStatus.SOLVED_SATISFICING
            assert v.validate(problem, res.plan)
