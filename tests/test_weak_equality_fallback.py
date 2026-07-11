# Copyright (C) 2025 PSO Unit, Fondazione Bruno Kessler
# This file is part of TamerLite.
#
# TamerLite is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.
#
# TamerLite is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

from types import SimpleNamespace

import pytest
from unified_planning.engines import PlanGenerationResultStatus

import tamerlite.engine as engine_module


class _FakeEncoder:
    def __init__(self, *args, **kwargs):
        self.search_space = SimpleNamespace(is_temporal=True)

    def are_all_actions_compression_safe(self):
        return False

    def is_any_action_compression_safe(self):
        return False


@pytest.mark.parametrize("use_multiqueue", [False, True])
def test_weak_equality_fallback_uses_remaining_timeout(monkeypatch, use_multiqueue):
    if use_multiqueue:
        params = engine_module.MultiqueueParams(
            queues=[engine_module.HeuristicParams(heuristic="hff")],
            weak_equality=True,
            symmetry_breaking=False,
            compression_safe_actions=False,
            relevance_analysis=False,
        )
    else:
        params = engine_module.SearchParams(
            search="wastar",
            heuristic="hff",
            weak_equality=True,
            symmetry_breaking=False,
            compression_safe_actions=False,
            relevance_analysis=False,
        )

    planner = engine_module.TamerLite(params)
    calls = []

    def record_search(timeout, weak_equality):
        calls.append((timeout, weak_equality))
        return None, {"expanded_states": "1"}

    def fake_single_search(
        search_space,
        timeout=None,
        early_termination=False,
        weak_equality=False,
    ):
        return record_search(timeout, weak_equality)

    def fake_multiqueue_search(
        search_space,
        heuristics,
        timeout=None,
        early_termination=False,
        weak_equality=False,
    ):
        return record_search(timeout, weak_equality)

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

    clock = iter((100.0, 102.0))
    monkeypatch.setattr(engine_module.time, "monotonic", lambda: next(clock))

    result, _, _ = planner._solve_ground_problem(
        object(),
        object(),
        lambda action: action,
        timeout=10.0,
    )

    assert result.status == PlanGenerationResultStatus.UNSOLVABLE_INCOMPLETELY
    assert calls == [(10.0, True), (8.0, False)]


def test_remaining_timeout_is_never_negative(monkeypatch):
    monkeypatch.setattr(engine_module.time, "monotonic", lambda: 112.0)

    assert engine_module._remaining_timeout(10.0, 100.0) == 0.0
    assert engine_module._remaining_timeout(None, 100.0) is None
