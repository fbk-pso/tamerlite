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

"""`max_expanded_states` (`TamerLite._solve_with_params`): a oneshot solve
expands at most that many states per search run, and reports
`UNSOLVABLE_INCOMPLETELY` with the limit as its `expanded_states` metric when
it runs out."""

import pytest
from unified_planning.engines import PlanGenerationResultStatus as ResultStatus
from unified_planning.shortcuts import OneshotPlanner, get_environment

import problems_generator
import tamerlite
import tamerlite.engine
import testing_utils

env = get_environment()
env.factory.add_engine("tamerlite", "tamerlite.engine", "TamerLite")

BACKENDS = [pytest.param(True, id="python"), pytest.param(False, id="rust")]


def _search_params(kind: str):
    if kind == "multiqueue":
        return tamerlite.engine.MultiqueueParams(
            queues=[
                tamerlite.HeuristicParams(heuristic="hff", weight=0.8),
                tamerlite.HeuristicParams(heuristic="hadd", weight=0.8),
            ]
        )
    if kind == "wastar_memory_bounded":
        return tamerlite.SearchParams(
            search="wastar", incomplete_memory_bounded_search=True
        )
    return tamerlite.SearchParams(search=kind)


SEARCHES = [
    "bfs",
    "ehc",
    "wastar",
    "astar",
    "gbfs",
    "wastar_memory_bounded",
    "novbfs_hg",
    "novbfs_lg",
    "multiqueue",
]


def _solve(search_kind: str, problem, **kwargs):
    with OneshotPlanner(
        name="tamerlite", params={"search": _search_params(search_kind)}
    ) as planner:
        return planner.solve(problem, **kwargs)


@pytest.mark.parametrize("disable_rustamer", BACKENDS)
@pytest.mark.parametrize("search_kind", ["dfs", *SEARCHES])
def test_limit_reached_returns_unsolvable_incompletely(search_kind, disable_rustamer):
    testing_utils.reload_tamerlite(disable_rustamer)
    problem = problems_generator.get_problem_logistics(1, 1, 4, 2)
    res = _solve(search_kind, problem, max_expanded_states=1)
    assert res.status == ResultStatus.UNSOLVABLE_INCOMPLETELY
    assert res.plan is None
    assert res.metrics == {"expanded_states": "1"}


@pytest.mark.parametrize("disable_rustamer", BACKENDS)
@pytest.mark.parametrize("search_kind", SEARCHES)
def test_limit_is_inclusive(search_kind, disable_rustamer):
    """A limit equal to the unlimited solve's expansion count still solves,
    identically; one less does not."""
    testing_utils.reload_tamerlite(disable_rustamer)
    problem = problems_generator.get_problem_logistics(1, 1, 4, 2)
    unlimited = _solve(search_kind, problem)
    assert unlimited.status == ResultStatus.SOLVED_SATISFICING
    expanded = int(unlimited.metrics["expanded_states"])
    assert expanded > 1

    exact = _solve(search_kind, problem, max_expanded_states=expanded)
    assert exact.status == ResultStatus.SOLVED_SATISFICING
    assert exact.metrics == unlimited.metrics
    assert exact.plan == unlimited.plan

    short = _solve(search_kind, problem, max_expanded_states=expanded - 1)
    assert short.status == ResultStatus.UNSOLVABLE_INCOMPLETELY
    assert short.metrics == {"expanded_states": str(expanded - 1)}


@pytest.mark.parametrize("disable_rustamer", BACKENDS)
def test_limit_zero_allows_initial_goal_only(disable_rustamer):
    testing_utils.reload_tamerlite(disable_rustamer)
    problem = problems_generator.get_problem_goal_satisfied_initially()
    params = tamerlite.SearchParams(early_termination=True)
    with OneshotPlanner(name="tamerlite", params={"search": params}) as planner:
        res = planner.solve(problem, max_expanded_states=0)
    assert res.status == ResultStatus.SOLVED_SATISFICING
    assert res.metrics["expanded_states"] == "0"

    res = _solve(
        "wastar",
        problems_generator.get_problem_logistics(1, 1, 4, 2),
        max_expanded_states=0,
    )
    assert res.status == ResultStatus.UNSOLVABLE_INCOMPLETELY
    assert res.metrics == {"expanded_states": "0"}


@pytest.mark.parametrize("disable_rustamer", BACKENDS)
def test_limit_skips_weak_equality_retry(disable_rustamer):
    testing_utils.reload_tamerlite(disable_rustamer)
    problem = problems_generator.get_problem_temporal_flight()
    params = tamerlite.SearchParams(weak_equality=True, compression_safe_actions=False)
    with OneshotPlanner(name="tamerlite", params={"search": params}) as planner:
        res = planner.solve(problem, max_expanded_states=1)
    assert res.status == ResultStatus.UNSOLVABLE_INCOMPLETELY
    assert res.metrics == {"expanded_states": "1"}


@pytest.mark.parametrize("disable_rustamer", BACKENDS)
def test_core_raises_expansion_limit_reached(disable_rustamer):
    testing_utils.reload_tamerlite(disable_rustamer)
    from tamerlite.core import ExpansionLimitReached, bfs_search
    from tamerlite.encoder import Encoder

    assert not issubclass(ExpansionLimitReached, TimeoutError)
    lifted_problem, ground_problem, map_back_action_instance = (
        testing_utils.compile_problem(
            problems_generator.get_problem_logistics(1, 1, 4, 2)
        )
    )
    encoder = Encoder(
        ground_problem,
        lifted_problem,
        map_back_action_instance,
        symmetry_breaking=False,
        compression_safe_actions=False,
        relevance_analysis=False,
    )
    with pytest.raises(ExpansionLimitReached):
        bfs_search(encoder.search_space, max_expanded_states=1)


@pytest.mark.parametrize(
    "value,error",
    [(-1, ValueError), (True, TypeError), (1.5, TypeError), ("3", TypeError)],
)
def test_invalid_limit_rejected(value, error):
    problem = problems_generator.get_problem_logistics(1, 1, 4, 2)
    with pytest.raises(error, match="max_expanded_states"):
        _solve("wastar", problem, max_expanded_states=value)


def test_unknown_kwarg_rejected():
    problem = problems_generator.get_problem_logistics(1, 1, 4, 2)
    with pytest.raises(TypeError, match="max_expanded"):
        _solve("wastar", problem, max_expanded=10)


def test_warm_start_plan_warns():
    problem = problems_generator.get_problem_logistics(1, 1, 4, 2)
    plan = _solve("wastar", problem).plan
    assert plan is not None
    with pytest.warns(UserWarning, match="warm_start_plan"):
        res = _solve("wastar", problem, warm_start_plan=plan)
    assert res.status == ResultStatus.SOLVED_SATISFICING
