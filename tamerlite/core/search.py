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
from tamerlite.core.search_space import SearchSpace, State
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

def bfs_search(ss: SearchSpace, timeout=None):
    return _basic_search(ss, True, timeout)

def dfs_search(ss: SearchSpace, timeout=None):
    return _basic_search(ss, False, timeout)

def _basic_search(ss: SearchSpace, bfs: bool, timeout):
    st = time.time()
    init = ss.initial_state()
    open = deque()
    open.append(init)
    counter = 0
    while len(open) > 0:
        if timeout is not None and time.time() - st > timeout:
            raise TimeoutError
        if bfs:
            state = open.popleft()
        else:
            state = open.pop()
        counter += 1
        if ss.goal_reached(state):
            return state.extract_solution(), {"expanded_states": counter, "goal_depth": state.g}
        for succ_state in ss.get_successor_states(state):
            open.append(succ_state)
    return None, {"expanded_states": str(counter)}

def astar_search(ss: SearchSpace, heuristic: Heuristic, timeout=None):
    return wastar_search(ss, heuristic, 0.5, timeout)

def gbfs_search(ss: SearchSpace, heuristic: Heuristic, timeout=None):
    return wastar_search(ss, heuristic, 1, timeout)

def wastar_search(ss: SearchSpace, heuristic: Heuristic, weight: float = 0.5, timeout=None):
    st = time.time()
    open = []
    closed_set = set()
    open_set = set()
    init = ss.initial_state()
    init_h = heuristic.eval(init, ss)
    if init_h is None:
        return None, {"expanded_states": str(0)}
    heapq.heappush(open, PrioritizedItem(init_h, init))
    counter = 0
    while open:
        if timeout is not None and time.time() - st > timeout:
            raise TimeoutError
        item = heapq.heappop(open)
        state = item.state
        if not ss.is_temporal:
            closed_set.add(state)
            open_set.discard(state)
        counter += 1
        if ss.goal_reached(state):
            return state.extract_solution(), {"expanded_states": str(counter), "goal_depth": str(state.g)}

        candidate_states = (s for s in ss.get_successor_states(state) if s not in closed_set and s not in open_set)
        for succ_state, h in heuristic.eval_gen(candidate_states, ss):
            if h is not None:
                f = (1-weight)*succ_state.g + weight*h
                heapq.heappush(open, PrioritizedItem(f, succ_state))
                if not ss.is_temporal:
                    open_set.add(succ_state)
    return None, {"expanded_states": str(counter)}

def ehc_search(ss: SearchSpace, heuristic: Heuristic, timeout=None):
    st = time.time()
    init = ss.initial_state()
    open = deque()
    open.append(init)
    best_h = heuristic.eval(init, ss)
    if best_h is None:
        return None, {"expanded_states": str(0)}
    counter = 0
    while len(open) > 0:
        if timeout is not None and time.time() - st > timeout:
            raise TimeoutError
        state = open.popleft()
        counter += 1
        if ss.goal_reached(state):
            return state.extract_solution(), {"expanded_states": str(counter), "goal_depth": str(state.g)}
        for succ_state, h in heuristic.eval_gen(ss.get_successor_states(state), ss):
            if h is not None:
                if h < best_h:
                    best_h = h
                    open.clear()
                open.append(succ_state)
    return None, {"expanded_states": str(counter)}
