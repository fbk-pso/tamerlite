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

import heapq
from dataclasses import dataclass
import time
from typing import List, Tuple
from tamerlite.core.search_space import SearchSpace, State
from tamerlite.core.heuristics import Heuristic


@dataclass
class StateContainer:
    state: State
    expanded: bool


@dataclass
class PrioritizedItem:
    heuristic: int
    state_container: StateContainer

    def __lt__(self, other):
        if self.heuristic < other.heuristic:
            return True
        if self.heuristic > other.heuristic:
            return False
        return len(self.state_container.state.todo) < len(other.state_container.state.todo)


def multiqueue_search(ss: SearchSpace, heuristics: List[Tuple[Heuristic, float]], timeout: float = None):
    st = time.time()
    opens = []
    closed_set = set()
    open_set = set()
    init = ss.initial_state()
    item = PrioritizedItem(0, StateContainer(init, False))
    for _ in heuristics:
        open = []
        heapq.heappush(open, item)
        opens.append(open)
    counter = 0
    states_expanded = 0
    while True:
        if timeout is not None and time.time() - st > timeout:
            raise TimeoutError
        if any(len(o) == 0 for o in opens):
            break
        i = counter % len(opens)
        open = opens[i]
        item = heapq.heappop(open)
        sc = item.state_container
        if sc.expanded:
            continue
        sc.expanded = True
        state = sc.state
        if not ss.is_temporal:
            closed_set.add(state)
            open_set.discard(state)
        counter += 1
        states_expanded += 1
        if ss.goal_reached(state):
            return state.extract_solution(), {"expanded_states": str(states_expanded), "goal_depth": str(state.g)}

        # Here, we create a temporary list of the successor states to reuse it among multiple heuristics
        candidate_states = []
        for s in ss.get_successor_states(state):
            if not ss.is_temporal:
                if s in closed_set or s in open_set:
                    continue
                open_set.add(s)
            candidate_states.append(s)
        candidate_containers = [StateContainer(s, False) for s in candidate_states]
        for i, (heuristic, weight) in enumerate(heuristics):
            for j, (succ_state, h) in enumerate(heuristic.eval_gen(candidate_states, ss)):
                if h is not None:
                    f = (1-weight)*succ_state.g + weight*h
                    item = PrioritizedItem(f, candidate_containers[j])
                    heapq.heappush(opens[i], item)
    return None, {"expanded_states": str(states_expanded)}
