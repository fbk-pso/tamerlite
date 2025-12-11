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
#

from collections import deque
import heapq
import time
from dataclasses import dataclass
from tamerlite.core.search_space import SearchSpaceABC, State
from tamerlite.core.heuristics import Heuristic


@dataclass
class PrioritizedItem:
    heuristic: int
    state: State

    def __lt__(self, other):
        if self.heuristic < other.heuristic:
            return True
        if self.heuristic > other.heuristic:
            return False
        return len(self.state.todo) < len(other.state.todo)


@dataclass
class WeakEqState:
    state: State

    def __hash__(self) -> int:
        return hash(tuple(self.state.assignments))

    def __eq__(self, oth) -> bool:
        if (
            len(self.state.todo) != len(oth.state.todo)
            or self.state.assignments != oth.state.assignments
        ):
            return False

        for a in self.state.todo:
            idx = self.state.todo[a][0]
            idx_id = oth.state.todo.get(a, None)
            if idx_id is None or idx_id[0] != idx:
                return False

        return True


def state_representation(state: State, weak_equality: bool):
    if weak_equality:
        return WeakEqState(state)
    return state


def bfs_search(ss: SearchSpaceABC, timeout=None, early_termination: bool = False):
    return _basic_search(ss, True, timeout, early_termination)


def dfs_search(ss: SearchSpaceABC, timeout=None, early_termination: bool = False):
    return _basic_search(ss, False, timeout, early_termination)


def _basic_search(
    ss: SearchSpaceABC, bfs: bool, timeout, early_termination: bool = False
):
    st = time.time()
    init = ss.initial_state()
    open = deque()
    counter = 0

    if early_termination and ss.goal_reached(init):
        return ss.build_plan(init), {"expanded_states": counter, "goal_depth": init.g}
    open.append(init)

    while len(open) > 0:
        if timeout is not None and time.time() - st > timeout:
            raise TimeoutError
        if bfs:
            state = open.popleft()
        else:
            state = open.pop()
        counter += 1
        if not early_termination and ss.goal_reached(state):
            return ss.build_plan(state), {
                "expanded_states": counter,
                "goal_depth": state.g,
            }
        for succ_state in ss.get_successor_states(state):
            if early_termination and ss.goal_reached(succ_state):
                return ss.build_plan(succ_state), {
                    "expanded_states": counter,
                    "goal_depth": succ_state.g,
                }
            open.append(succ_state)
    return None, {"expanded_states": str(counter)}


def astar_search(
    ss: SearchSpaceABC,
    heuristic: Heuristic,
    timeout=None,
    early_termination: bool = False,
    weak_equality: bool = False,
):
    return wastar_search(ss, heuristic, 0.5, timeout, early_termination, weak_equality)


def gbfs_search(
    ss: SearchSpaceABC,
    heuristic: Heuristic,
    timeout=None,
    early_termination: bool = False,
    weak_equality: bool = False,
):
    return wastar_search(ss, heuristic, 1, timeout, early_termination, weak_equality)


def wastar_search(
    ss: SearchSpaceABC,
    heuristic: Heuristic,
    weight: float = 0.5,
    timeout=None,
    early_termination: bool = False,
    weak_equality: bool = False,
):
    st = time.time()
    open = []
    init = ss.initial_state()
    if not ss.is_temporal or weak_equality:
        visited_states = {state_representation(init, weak_equality)}
    counter = 0
    if early_termination and ss.goal_reached(init):
        return ss.build_plan(init), {
            "expanded_states": str(counter),
            "goal_depth": str(init.g),
        }

    init_h = heuristic.eval(init, ss)
    if init_h is None:
        return None, {"expanded_states": str(0)}
    heapq.heappush(open, PrioritizedItem(init_h, init))
    while open:
        if timeout is not None and time.time() - st > timeout:
            raise TimeoutError
        item = heapq.heappop(open)
        state = item.state
        counter += 1
        if not early_termination and ss.goal_reached(state):
            return ss.build_plan(state), {
                "expanded_states": str(counter),
                "goal_depth": str(state.g),
            }

        candidate_states = []
        for state in ss.get_successor_states(state):
            if not ss.is_temporal or weak_equality:
                s_repr = state_representation(state, weak_equality)
                if s_repr not in visited_states:
                    visited_states.add(s_repr)
                    candidate_states.append(state)
            else:
                candidate_states.append(state)

        for succ_state, h in heuristic.eval_gen(candidate_states, ss):
            if early_termination and ss.goal_reached(succ_state):
                return ss.build_plan(succ_state), {
                    "expanded_states": str(counter),
                    "goal_depth": str(succ_state.g),
                }
            if h is not None:
                f = (1 - weight) * succ_state.g + weight * h
                heapq.heappush(open, PrioritizedItem(f, succ_state))

    return None, {"expanded_states": str(counter)}


def ehc_search(
    ss: SearchSpaceABC,
    heuristic: Heuristic,
    timeout=None,
    early_termination: bool = False,
):
    st = time.time()
    init = ss.initial_state()
    counter = 0
    if early_termination and ss.goal_reached(init):
        return ss.build_plan(init), {
            "expanded_states": str(counter),
            "goal_depth": str(init.g),
        }

    open = deque()
    open.append(init)
    best_h = heuristic.eval(init, ss)
    if best_h is None:
        return None, {"expanded_states": str(0)}
    while len(open) > 0:
        if timeout is not None and time.time() - st > timeout:
            raise TimeoutError
        state = open.popleft()
        counter += 1
        if not early_termination and ss.goal_reached(state):
            return ss.build_plan(state), {
                "expanded_states": str(counter),
                "goal_depth": str(state.g),
            }
        for succ_state, h in heuristic.eval_gen(ss.get_successor_states(state), ss):
            if early_termination and ss.goal_reached(succ_state):
                return ss.build_plan(succ_state), {
                    "expanded_states": str(counter),
                    "goal_depth": str(succ_state.g),
                }
            if h is not None:
                if h < best_h:
                    best_h = h
                    open.clear()
                open.append(succ_state)
    return None, {"expanded_states": str(counter)}
