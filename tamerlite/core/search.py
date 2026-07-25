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

from collections import deque
import heapq
import time
from dataclasses import dataclass
from tamerlite.core.search_space import SearchSpaceABC, State, Action
from tamerlite.core.heuristics import Heuristic
from typing import Tuple, List, Dict, Deque, Optional, Union
from fractions import Fraction


@dataclass
class PrioritizedItem:
    heuristic: float
    state: State
    idx: int

    def __lt__(self, other):
        return (self.heuristic, len(self.state.todo), self.idx) < (
            other.heuristic,
            len(other.state.todo),
            other.idx,
        )

@dataclass
class WeakEqState:
    state: State

    def __hash__(self) -> int:
        return hash(tuple(self.state.assignments))

    def __eq__(self, oth) -> bool:
        if (len(self.state.todo) != len(oth.state.todo)
                or self.state.assignments != oth.state.assignments):
            return False

        for a in self.state.todo:
            idx = self.state.todo[a][0]
            idx_id = oth.state.todo.get(a, None)
            if idx_id is None or idx_id[0] != idx:
                return False

        return True


def state_representation(state: State, weak_equality: bool) -> Union[State, WeakEqState]:
    if weak_equality:
        return WeakEqState(state)
    return state


def bfs_search(
    ss: SearchSpaceABC,
    timeout: Optional[float] = None,
    early_termination: bool = False
) -> Tuple[
        Optional[List[Tuple[Optional[Fraction], Action, Optional[Fraction]]]],
        Dict[str, str],
]:
    return _basic_search(ss, True, timeout, early_termination)


def dfs_search(
    ss: SearchSpaceABC,
    timeout: Optional[float] = None,
    early_termination: bool = False
) -> Tuple[
        Optional[List[Tuple[Optional[Fraction], Action, Optional[Fraction]]]],
        Dict[str, str],
]:
    return _basic_search(ss, False, timeout, early_termination)


def _basic_search(
    ss: SearchSpaceABC,
    bfs: bool,
    timeout: Optional[float] = None,
    early_termination: bool = False,
) -> Tuple[
        Optional[List[Tuple[Optional[Fraction], Action, Optional[Fraction]]]],
        Dict[str, str],
]:
    st = time.time()
    init = ss.initial_state()
    open: Deque[State] = deque()
    expanded_states = 0

    if early_termination and ss.goal_reached(init):
        return ss.build_plan(init), {
            "expanded_states": str(expanded_states),
            "goal_depth": str(init.g),
        }
    open.append(init)

    while len(open) > 0:
        if timeout is not None and time.time() - st > timeout:
            raise TimeoutError({"expanded_states": str(expanded_states)})
        if bfs:
            state = open.popleft()
        else:
            state = open.pop()
        expanded_states += 1
        if not early_termination and ss.goal_reached(state):
            return ss.build_plan(state), {
                "expanded_states": str(expanded_states),
                "goal_depth": str(state.g),
            }
        for succ_state in ss.get_successor_states(state):
            if early_termination and ss.goal_reached(succ_state):
                return ss.build_plan(succ_state), {
                    "expanded_states": str(expanded_states),
                    "goal_depth": str(succ_state.g),
                }
            open.append(succ_state)
    return None, {"expanded_states": str(expanded_states)}


def astar_search(
    ss: SearchSpaceABC,
    heuristic: Heuristic,
    timeout: Optional[float] = None,
    early_termination: bool = False,
    weak_equality: bool = False,
) -> Tuple[
        Optional[List[Tuple[Optional[Fraction], Action, Optional[Fraction]]]],
        Dict[str, str],
]:
    return wastar_search(ss, heuristic, 0.5, timeout, early_termination, weak_equality)


def gbfs_search(
    ss: SearchSpaceABC,
    heuristic: Heuristic,
    timeout: Optional[float] = None,
    early_termination: bool = False,
    weak_equality: bool = False,
) -> Tuple[
        Optional[List[Tuple[Optional[Fraction], Action, Optional[Fraction]]]],
        Dict[str, str],
]:
    return wastar_search(ss, heuristic, 1, timeout, early_termination, weak_equality)


def wastar_search(
    ss: SearchSpaceABC,
    heuristic: Heuristic,
    weight: float = 0.5,
    timeout: Optional[float] = None,
    early_termination: bool = False,
    weak_equality: bool = False,
    max_len: Optional[float] = None,
):
    st = time.time()
    open: List[PrioritizedItem] = []
    init = ss.initial_state()
    if not ss.is_temporal or weak_equality:
        visited_states = {state_representation(init, weak_equality)}
    expanded_states = 0
    generated_states = 1
    if early_termination and ss.goal_reached(init):
        return ss.build_plan(init), {
            "expanded_states": str(expanded_states),
            "goal_depth": str(init.g),
            "len_plan": len([a for a, _, _ in init.path])
        }

    init_h = heuristic.eval(init, ss)
    if init_h is None:
        return None, {"expanded_states": str(0)}
    heapq.heappush(open, PrioritizedItem(init_h, init, 0))
    plans = []
    if max_len is not None:            
        assert timeout is not None
    while open:
        if max_len is not None and time.time() - st > timeout:
            raise TimeoutError({"expanded_states": str(expanded_states), "plans": plans})
        if timeout is not None and time.time() - st > timeout:
            raise TimeoutError({"expanded_states": str(expanded_states)})
        item = heapq.heappop(open)
        state = item.state
        expanded_states += 1
        if not early_termination and ss.goal_reached(state):
            path = [a for a, _, _ in state.path]
            if max_len is not None: 
                if len(path) <= max_len:
                    print(f"{len(plans)}) Found plan of length {len(path)}")
                    plans.append(path)
                    if max_len == float("inf"):
                        max_len = len(path)
            else:
                return ss.build_plan(state), {
                    "expanded_states": str(expanded_states),
                    "goal_depth": str(state.g),
                    "len_plan": len([a for a, _, _ in state.path])
                }

        candidate_states = []
        candidate_indices: Dict[int, int] = {}
        for succ_state in ss.get_successor_states(state):
            if early_termination and ss.goal_reached(succ_state):
                return ss.build_plan(succ_state), {
                    "expanded_states": str(expanded_states),
                    "goal_depth": str(succ_state.g),
                    "len_plan": len([a for a, _, _ in state.path])
                }

            if not ss.is_temporal or weak_equality:
                state_repr = state_representation(succ_state, weak_equality)
                if state_repr not in visited_states:
                    visited_states.add(state_repr)
                    if max_len is None or succ_state.g <= max_len:
                        candidate_indices[id(succ_state)] = generated_states
                        generated_states += 1
                        candidate_states.append(succ_state)
            else:
                if max_len is None or succ_state.g <= max_len:
                    candidate_indices[id(succ_state)] = generated_states
                    generated_states += 1
                    candidate_states.append(succ_state)

        for succ_state, h in heuristic.eval_gen(candidate_states, ss):
            if h is not None:
                f = (1 - weight) * succ_state.g + weight * h
                heapq.heappush(
                    open,
                    PrioritizedItem(f, succ_state, candidate_indices[id(succ_state)]),
                )
    res = {"expanded_states": str(expanded_states)} if max_len is None else {"expanded_states": str(expanded_states), "plans": plans}
    return None , res


def ehc_search(
    ss: SearchSpaceABC,
    heuristic: Heuristic,
    timeout: Optional[float] = None,
    early_termination: bool = False,
    weak_equality: bool = False,
    max_len: Optional[float] = None,
) -> Tuple[
        Optional[List[Tuple[Optional[Fraction], Action, Optional[Fraction]]]],
        Dict[str, str],
]:
    st = time.time()
    init = ss.initial_state()
    expanded_states = 0
    if early_termination and ss.goal_reached(init):
        return ss.build_plan(init), {
            "expanded_states": str(expanded_states),
            "goal_depth": str(init.g),
        }

    open: Deque[State] = deque()
    open.append(init)
    best_h = heuristic.eval(init, ss)
    if best_h is None:
        return None, {"expanded_states": str(0)}

    closed = set()
    plans: List[List[Action]] = []
    if max_len is not None:
        assert timeout is not None

        # EHC restarts by wiping `open`/`closed` whenever a strictly better
        # successor is found, so a goal state sitting in `open` can be
        # discarded before it is ever popped. To not lose it, every state is
        # checked for goal_reached exactly once, right when it's generated
        # (here, and for each successor below), instead of only at pop time.
        def log_if_goal(candidate: State) -> None:
            nonlocal max_len
            if ss.goal_reached(candidate):
                path = [a for a, _, _ in candidate.path]
                if len(path) <= max_len:
                    print(f"{len(plans)}) Found plan of length {len(path)}")
                    plans.append(path)
                    if max_len == float("inf"):
                        max_len = len(path)

        if time.time() - st > timeout:
            raise TimeoutError({"expanded_states": str(expanded_states), "plans": plans})
        log_if_goal(init)

    while len(open) > 0:
        if max_len is not None and time.time() - st > timeout:
            raise TimeoutError({"expanded_states": str(expanded_states), "plans": plans})
        if timeout is not None and time.time() - st > timeout:
            raise TimeoutError({"expanded_states": str(expanded_states)})
        state = open.popleft()
        expanded_states += 1
        if not ss.is_temporal or weak_equality:
            closed.add(state_representation(state, weak_equality))

        if max_len is None and not early_termination and ss.goal_reached(state):
            return ss.build_plan(state), {
                "expanded_states": str(expanded_states),
                "goal_depth": str(state.g),
            }

        candidate_states = []
        for succ_state in ss.get_successor_states(state):
            if early_termination and ss.goal_reached(succ_state):
                return ss.build_plan(succ_state), {
                    "expanded_states": str(expanded_states),
                    "goal_depth": str(succ_state.g),
                }

            if max_len is not None:
                log_if_goal(succ_state)

            if not ss.is_temporal or weak_equality:
                state_repr = state_representation(succ_state, weak_equality)
                if state_repr not in closed:
                    if max_len is None or succ_state.g <= max_len:
                        candidate_states.append(succ_state)
            else:
                if max_len is None or succ_state.g <= max_len:
                    candidate_states.append(succ_state)

        for succ_state, h in heuristic.eval_gen(candidate_states, ss):
            if h is not None:
                if h < best_h:
                    best_h = h
                    closed.clear()
                    open.clear()
                    open.append(succ_state)
                    break
                else:
                    open.append(succ_state)
    res = (
        {"expanded_states": str(expanded_states)}
        if max_len is None
        else {"expanded_states": str(expanded_states), "plans": plans}
    )
    return None, res
