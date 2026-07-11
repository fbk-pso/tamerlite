# Copyright (C) 2025 PSO Unit, Fondazione Bruno Kessler
# This file is part of TamerLite.
#
# TamerLite is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import pytest

from tamerlite.core import multiqueue
from tamerlite.core import search


class _SearchSpace:
    is_temporal = True

    @staticmethod
    def initial_state():
        return object()


@pytest.mark.parametrize(
    ("timeout", "max_expanded_states", "reason"),
    (
        (-1.0, None, "timeout"),
        (None, 5, "max_expanded_states"),
    ),
)
def test_python_search_limits_preserve_reason_and_progress(
    timeout, max_expanded_states, reason
):
    with pytest.raises(TimeoutError) as raised:
        search._check_search_limits(
            st=0.0,
            timeout=timeout,
            expanded_states=5,
            max_expanded_states=max_expanded_states,
        )

    assert raised.value.reason == reason
    assert raised.value.expanded_states == 5
    assert str(raised.value) == ""


def test_python_multiqueue_limit_preserves_reason_and_progress():
    with pytest.raises(TimeoutError) as raised:
        multiqueue.multiqueue_search(
            _SearchSpace(),
            [(object(), 1.0)],
            max_expanded_states=0,
        )

    assert raised.value.reason == "max_expanded_states"
    assert raised.value.expanded_states == 0
