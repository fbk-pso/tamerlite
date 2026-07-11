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

from collections import deque
from types import SimpleNamespace

import pytest
from unified_planning.engines import PlanGenerationResultStatus
from unified_planning.plans import SequentialPlan
from unified_planning.shortcuts import Problem

import tamerlite.engine as engine_module


class _FakeEncoder:
    def __init__(self, *args, **kwargs):
        self.search_space = SimpleNamespace(is_temporal=True)

    def are_all_actions_compression_safe(self):
        return False

    def build_plan(self, path):
        return SequentialPlan([])


def _run_planner(
    monkeypatch,
    *,
    use_multiqueue,
    outcomes,
    timeout=10.0,
    weak_equality_fallback=True,
):
    params: engine_module.SearchParams | engine_module.MultiqueueParams
    if use_multiqueue:
        params = engine_module.MultiqueueParams(
            queues=[engine_module.HeuristicParams(heuristic="hff")],
            max_expanded_states=100,
            weak_equality=True,
            weak_equality_fallback=weak_equality_fallback,
            symmetry_breaking=False,
            compression_safe_actions=False,
        )
    else:
        params = engine_module.SearchParams(
            search="wastar",
            heuristic="hff",
            max_expanded_states=100,
            weak_equality=True,
            weak_equality_fallback=weak_equality_fallback,
            symmetry_breaking=False,
            compression_safe_actions=False,
        )

    planner = engine_module.TamerLite(params)
    calls = []
    pending = deque(outcomes)

    def record_search(call_timeout, max_expanded_states, weak_equality):
        calls.append((weak_equality, call_timeout, max_expanded_states))
        outcome = pending.popleft()
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    def fake_single_search(
        search_space,
        timeout=None,
        max_expanded_states=None,
        early_termination=False,
        weak_equality=False,
    ):
        return record_search(timeout, max_expanded_states, weak_equality)

    def fake_multiqueue_search(
        search_space,
        heuristics,
        timeout=None,
        max_expanded_states=None,
        early_termination=False,
        weak_equality=False,
    ):
        return record_search(timeout, max_expanded_states, weak_equality)

    monkeypatch.setattr(engine_module, "Encoder", _FakeEncoder)
    monkeypatch.setattr(planner, "_get_heuristic", lambda *args: (object(), 0.8))
    if use_multiqueue:
        monkeypatch.setattr(engine_module, "multiqueue_search", fake_multiqueue_search)
    else:
        monkeypatch.setattr(
            planner,
            "_get_search",
            lambda *args: ("wastar", fake_single_search),
        )

    return planner._solve(Problem("fallback_test"), timeout=timeout), calls


@pytest.mark.parametrize("use_multiqueue", [False, True])
def test_fallback_shares_timeout_expansion_budget_and_metrics(
    monkeypatch, use_multiqueue
):
    clock = iter((100.0, 102.0))
    monkeypatch.setattr(engine_module.time, "monotonic", lambda: next(clock))

    result, calls = _run_planner(
        monkeypatch,
        use_multiqueue=use_multiqueue,
        outcomes=(
            (None, {"expanded_states": "30"}),
            (None, {"expanded_states": "20"}),
        ),
    )

    assert result.status == PlanGenerationResultStatus.UNSOLVABLE_INCOMPLETELY
    assert calls == [(True, 10.0, 100), (False, 8.0, 70)]
    assert result.metrics == {
        "expanded_states": "50",
        "weak_equality_fallback_attempted": "1",
        "weak_equality_fallback_solved": "0",
        "weak_expanded_states": "30",
        "strong_expanded_states": "20",
    }


@pytest.mark.parametrize("use_multiqueue", [False, True])
def test_fallback_success_keeps_strong_goal_depth(monkeypatch, use_multiqueue):
    clock = iter((100.0, 101.0))
    monkeypatch.setattr(engine_module.time, "monotonic", lambda: next(clock))

    result, calls = _run_planner(
        monkeypatch,
        use_multiqueue=use_multiqueue,
        outcomes=(
            (None, {"expanded_states": "30"}),
            ([], {"expanded_states": "15", "goal_depth": "4"}),
        ),
    )

    assert result.status == PlanGenerationResultStatus.SOLVED_SATISFICING
    assert calls == [(True, 10.0, 100), (False, 9.0, 70)]
    assert result.metrics["expanded_states"] == "45"
    assert result.metrics["goal_depth"] == "4"
    assert result.metrics["weak_equality_fallback_solved"] == "1"


@pytest.mark.parametrize("use_multiqueue", [False, True])
def test_weak_success_does_not_enter_fallback(monkeypatch, use_multiqueue):
    monkeypatch.setattr(engine_module.time, "monotonic", lambda: 100.0)
    weak_metrics = {"expanded_states": "12", "goal_depth": "3"}

    result, calls = _run_planner(
        monkeypatch,
        use_multiqueue=use_multiqueue,
        outcomes=(([], weak_metrics),),
    )

    assert result.status == PlanGenerationResultStatus.SOLVED_SATISFICING
    assert calls == [(True, 10.0, 100)]
    assert result.metrics == weak_metrics


@pytest.mark.parametrize("use_multiqueue", [False, True])
def test_fallback_can_be_disabled_for_explicit_weak_only(monkeypatch, use_multiqueue):
    monkeypatch.setattr(engine_module.time, "monotonic", lambda: 100.0)
    weak_metrics = {"expanded_states": "30"}

    result, calls = _run_planner(
        monkeypatch,
        use_multiqueue=use_multiqueue,
        outcomes=((None, weak_metrics),),
        weak_equality_fallback=False,
    )

    assert result.status == PlanGenerationResultStatus.UNSOLVABLE_INCOMPLETELY
    assert calls == [(True, 10.0, 100)]
    assert result.metrics == weak_metrics


@pytest.mark.parametrize("use_multiqueue", [False, True])
def test_no_timeout_is_preserved_for_fallback(monkeypatch, use_multiqueue):
    monkeypatch.setattr(engine_module.time, "monotonic", lambda: 100.0)

    result, calls = _run_planner(
        monkeypatch,
        use_multiqueue=use_multiqueue,
        outcomes=(
            (None, {"expanded_states": "30"}),
            (None, {"expanded_states": "20"}),
        ),
        timeout=None,
    )

    assert result.status == PlanGenerationResultStatus.UNSOLVABLE_INCOMPLETELY
    assert calls == [(True, None, 100), (False, None, 70)]
    assert result.metrics["expanded_states"] == "50"


@pytest.mark.parametrize("use_multiqueue", [False, True])
def test_timeout_in_either_phase_returns_timeout(monkeypatch, use_multiqueue):
    monkeypatch.setattr(engine_module.time, "monotonic", lambda: 100.0)

    weak_timeout, weak_calls = _run_planner(
        monkeypatch,
        use_multiqueue=use_multiqueue,
        outcomes=(TimeoutError(),),
    )
    assert weak_timeout.status == PlanGenerationResultStatus.TIMEOUT
    assert weak_calls == [(True, 10.0, 100)]

    strong_timeout, strong_calls = _run_planner(
        monkeypatch,
        use_multiqueue=use_multiqueue,
        outcomes=((None, {"expanded_states": "30"}), TimeoutError()),
    )
    assert strong_timeout.status == PlanGenerationResultStatus.TIMEOUT
    assert strong_calls == [(True, 10.0, 100), (False, 10.0, 70)]


def test_remaining_timeout_is_never_negative(monkeypatch):
    monkeypatch.setattr(engine_module.time, "monotonic", lambda: 112.0)

    assert engine_module._remaining_timeout(10.0, 100.0) == 0.0
    assert engine_module._remaining_timeout(None, 100.0) is None


def test_remaining_expanded_budget_requires_a_valid_count_when_capped():
    remaining = engine_module.TamerLite._remaining_expanded_budget

    assert remaining(None, {}) is None
    assert remaining(100, {"expanded_states": "30"}) == 70
    assert remaining(100, {"expanded_states": "130"}) == 0
    with pytest.raises(KeyError):
        remaining(100, {})
    with pytest.raises(ValueError):
        remaining(100, {"expanded_states": "invalid"})


def test_fallback_metrics_require_expansion_counts():
    with pytest.raises(KeyError):
        engine_module.TamerLite._fallback_metrics({}, {"expanded_states": "1"}, False)
    with pytest.raises(ValueError):
        engine_module.TamerLite._fallback_metrics(
            {"expanded_states": "invalid"}, {"expanded_states": "1"}, False
        )


def test_new_fallback_flag_preserves_historical_positional_parameter_order():
    search = engine_module.SearchParams(
        "hff", 0.8, "wastar", 123, False, True, True, False, False
    )
    multiqueue = engine_module.MultiqueueParams(
        [engine_module.HeuristicParams("hadd", 0.7)],
        456,
        False,
        True,
        True,
        False,
        False,
    )

    assert search.symmetry_breaking is False
    assert search.compression_safe_actions is False
    assert search.weak_equality_fallback is True
    assert multiqueue.symmetry_breaking is False
    assert multiqueue.compression_safe_actions is False
    assert multiqueue.weak_equality_fallback is True
