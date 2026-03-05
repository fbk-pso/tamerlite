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

import heapq
from dataclasses import dataclass
import time
from typing import List, Tuple, Dict, Optional
from tamerlite.core.search_space import SearchSpaceABC, State, Action
from tamerlite.core.heuristics import Heuristic
from tamerlite.core.search import state_representation, extract_path
from abc import ABC, abstractmethod


@dataclass
class StateContainer:
    state: State
    expanded: bool


@dataclass
class PrioritizedItem:
    heuristic: float
    state_container: StateContainer

    def __lt__(self, other):
        if self.heuristic < other.heuristic:
            return True
        if self.heuristic > other.heuristic:
            return False
        return len(self.state_container.state.todo) < len(
            other.state_container.state.todo
        )


class MQSwitchPolicy(ABC):
    """Abstract class for multi-queue switching policies."""

    @abstractmethod
    def switching_policy(self, i: int) -> int:
        """Given the number of expansions done so far, return the index of the
        next queue to use."""
        pass

    def notify_push(self, i: int, item: PrioritizedItem) -> None:
        """Called by algorithm to notify the policy that an item has been pushed
        to queue i."""
        pass

    def notify_pop(self, i: int, item: PrioritizedItem) -> None:
        """Called by algorithm to notify the policy that an item has been
        removed from queue i (and marked as expanded in all other queues)."""
        pass


class RoundRobinSwitchPolicy(MQSwitchPolicy):
    """The simple round-robin switching policy."""

    def __init__(self, num_queues: int):
        self.num_queues = num_queues

    def switching_policy(self, i: int) -> int:
        return i % self.num_queues


def multiqueue_search(
    ss: SearchSpaceABC,
    heuristics: List[Tuple[Heuristic, float]],
    timeout: Optional[float] = None,
    early_termination: bool = False,
    weak_equality: bool = False,
) -> Tuple[Optional[List[Action]], Dict[str, str]]:
    return _multiqueue_search(
        ss=ss,
        heuristics=heuristics,
        switch_policy=RoundRobinSwitchPolicy(len(heuristics)),
        timeout=timeout,
        early_termination=early_termination,
        weak_equality=weak_equality,
    )


def _multiqueue_search(
    ss: SearchSpaceABC,
    heuristics: List[Tuple[Heuristic, float]],
    switch_policy: MQSwitchPolicy,
    timeout: Optional[float] = None,
    early_termination: bool = False,
    weak_equality: bool = False,
) -> Tuple[Optional[List[Action]], Dict[str, str]]:
    st = time.time()
    opens = []
    init = ss.initial_state()
    if not ss.is_temporal or weak_equality:
        visited_states = {state_representation(init, weak_equality)}
    states_expanded = 0
    if early_termination and ss.goal_reached(init):
        return extract_path(init), {
            "expanded_states": str(states_expanded),
            "goal_depth": str(init.g),
        }

    item = PrioritizedItem(0.0, StateContainer(init, False))
    for i, _ in enumerate(heuristics):
        open: List[PrioritizedItem] = []
        heapq.heappush(open, item)
        opens.append(open)
        switch_policy.notify_push(i, item)

    while True:
        if timeout is not None and time.time() - st > timeout:
            raise TimeoutError
        if any(len(o) == 0 for o in opens):
            break

        i = switch_policy.switching_policy(states_expanded)
        open = opens[i]
        item = heapq.heappop(open)
        switch_policy.notify_pop(i, item)
        sc = item.state_container
        if sc.expanded:
            continue
        sc.expanded = True
        state = sc.state
        states_expanded += 1
        if not early_termination and ss.goal_reached(state):
            return extract_path(state), {
                "expanded_states": str(states_expanded),
                "goal_depth": str(state.g),
            }

        # Here, we create a temporary list of the successor states to reuse it among multiple heuristics
        candidate_states = []
        for s in ss.get_successor_states(state):
            if early_termination and ss.goal_reached(s):
                return extract_path(s), {
                    "expanded_states": str(states_expanded),
                    "goal_depth": str(s.g),
                }
            if not ss.is_temporal or weak_equality:
                state_repr = state_representation(s, weak_equality)
                if state_repr not in visited_states:
                    visited_states.add(state_repr)
                    candidate_states.append(s)
            else:
                candidate_states.append(s)

        candidate_containers = [StateContainer(s, False) for s in candidate_states]
        for i, (heuristic, weight) in enumerate(heuristics):
            for j, (succ_state, h) in enumerate(
                heuristic.eval_gen(candidate_states, ss)
            ):
                if h is not None:
                    f = (1.0 - weight) * succ_state.g + weight * h
                    item = PrioritizedItem(f, candidate_containers[j])
                    heapq.heappush(opens[i], item)
                    switch_policy.notify_push(i, item)
    return None, {"expanded_states": str(states_expanded)}
