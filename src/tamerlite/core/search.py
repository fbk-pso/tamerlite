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
import time
from collections import deque
from dataclasses import dataclass
from fractions import Fraction
from typing import Deque, Dict, List, Optional, Tuple, Union

from bloom_filter2 import BloomFilter
from min_max_heap import MinMaxHeap

from tamerlite.core.heuristics import Heuristic
from tamerlite.core.search_space import Action, SearchSpaceABC, State


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

    def __le__(self, other):
        return self < other


class BoundedPriorityQueue:
    """A bounded priority queue that keeps only the top-N smallest items.

    Args:
        bound: Maximum number of elements to keep.
    """

    def __init__(self, bound: int):
        assert bound > 0, "bound must be positive"
        self._bound = bound
        self._heap = MinMaxHeap()

    def push(self, item: PrioritizedItem) -> bool:
        """Push item if it belongs in the top-N smallest elements."""

        if self._heap.size() < self._bound:
            self._heap.push(item)
            return True

        # Heap is at capacity: only insert if item improves the collection.
        current_max = self._heap.max()
        if item < current_max:
            self._heap.pop_max()  # evict the largest element
            self._heap.push(item)
            return True

        return False  # item rejected

    def pop(self) -> PrioritizedItem:
        return self._heap.pop_min()

    def __len__(self) -> int:
        return self._heap.size()


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


def state_representation(
    state: State, weak_equality: bool
) -> Union[State, WeakEqState]:
    if weak_equality:
        return WeakEqState(state)
    return state


def extract_path(state: State) -> List[Action]:
    return [a for a, _, _ in state.path]


def bfs_search(
    ss: SearchSpaceABC, timeout: Optional[float] = None, early_termination: bool = False
) -> Tuple[Optional[List[Action]], Dict[str, str]]:
    return _basic_search(ss, True, timeout, early_termination)


def dfs_search(
    ss: SearchSpaceABC, timeout: Optional[float] = None, early_termination: bool = False
) -> Tuple[Optional[List[Action]], Dict[str, str]]:
    return _basic_search(ss, False, timeout, early_termination)


def _basic_search(
    ss: SearchSpaceABC,
    bfs: bool,
    timeout: Optional[float] = None,
    early_termination: bool = False,
) -> Tuple[Optional[List[Action]], Dict[str, str]]:
    st = time.time()
    init = ss.initial_state()
    open: Deque[State] = deque()
    expanded_states = 0

    if early_termination and ss.goal_reached(init):
        return extract_path(init), {
            "expanded_states": str(expanded_states),
            "goal_depth": str(init.g),
        }
    open.append(init)

    while len(open) > 0:
        if timeout is not None and time.time() - st > timeout:
            raise TimeoutError
        if bfs:
            state = open.popleft()
        else:
            state = open.pop()
        expanded_states += 1
        if not early_termination and ss.goal_reached(state):
            return extract_path(state), {
                "expanded_states": str(expanded_states),
                "goal_depth": str(state.g),
            }
        for succ_state in ss.get_successor_states(state):
            if early_termination and ss.goal_reached(succ_state):
                return extract_path(succ_state), {
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
) -> Tuple[Optional[List[Action]], Dict[str, str]]:
    return wastar_search(ss, heuristic, 0.5, timeout, early_termination, weak_equality)


def gbfs_search(
    ss: SearchSpaceABC,
    heuristic: Heuristic,
    timeout: Optional[float] = None,
    early_termination: bool = False,
    weak_equality: bool = False,
) -> Tuple[Optional[List[Action]], Dict[str, str]]:
    return wastar_search(ss, heuristic, 1, timeout, early_termination, weak_equality)


def wastar_search(
    ss: SearchSpaceABC,
    heuristic: Heuristic,
    weight: float = 0.5,
    timeout: Optional[float] = None,
    early_termination: bool = False,
    weak_equality: bool = False,
) -> Tuple[Optional[List[Action]], Dict[str, str]]:
    st = time.time()
    open: List[PrioritizedItem] = []
    init = ss.initial_state()
    if not ss.is_temporal or weak_equality:
        visited_states = {state_representation(init, weak_equality)}
    expanded_states = 0
    generated_states = 1
    if early_termination and ss.goal_reached(init):
        return extract_path(init), {
            "expanded_states": str(expanded_states),
            "goal_depth": str(init.g),
        }

    init_h = heuristic.eval(init, ss)
    if init_h is None:
        return None, {"expanded_states": str(0)}
    heapq.heappush(open, PrioritizedItem(init_h, init, 0))
    while open:
        if timeout is not None and time.time() - st > timeout:
            raise TimeoutError
        item = heapq.heappop(open)
        state = item.state
        expanded_states += 1
        if not early_termination and ss.goal_reached(state):
            return extract_path(state), {
                "expanded_states": str(expanded_states),
                "goal_depth": str(state.g),
            }

        candidate_states = []
        for succ_state in ss.get_successor_states(state):
            if early_termination and ss.goal_reached(succ_state):
                return extract_path(succ_state), {
                    "expanded_states": str(expanded_states),
                    "goal_depth": str(succ_state.g),
                }

            if not ss.is_temporal or weak_equality:
                state_repr = state_representation(succ_state, weak_equality)
                if state_repr not in visited_states:
                    visited_states.add(state_repr)
                    candidate_states.append(succ_state)
            else:
                candidate_states.append(succ_state)

        for succ_state, h in heuristic.eval_gen(candidate_states, ss):
            if h is not None:
                f = (1 - weight) * succ_state.g + weight * h
                heapq.heappush(open, PrioritizedItem(f, succ_state, generated_states))
            generated_states += 1

    return None, {"expanded_states": str(expanded_states)}


def astar_search_memory_bounded(
    ss: SearchSpaceABC,
    heuristic: Heuristic,
    timeout: Optional[float] = None,
    early_termination: bool = False,
    weak_equality: bool = False,
) -> Tuple[Optional[List[Action]], Dict[str, str]]:
    return wastar_search_memory_bounded(
        ss, heuristic, 0.5, timeout, early_termination, weak_equality
    )


def gbfs_search_memory_bounded(
    ss: SearchSpaceABC,
    heuristic: Heuristic,
    timeout: Optional[float] = None,
    early_termination: bool = False,
    weak_equality: bool = False,
) -> Tuple[Optional[List[Action]], Dict[str, str]]:
    return wastar_search_memory_bounded(
        ss, heuristic, 1, timeout, early_termination, weak_equality
    )


def wastar_search_memory_bounded(
    ss: SearchSpaceABC,
    heuristic: Heuristic,
    weight: float = 0.5,
    timeout: Optional[float] = None,
    early_termination: bool = False,
    weak_equality: bool = False,
) -> Tuple[Optional[List[Action]], Dict[str, str]]:
    st = time.time()
    init = ss.initial_state()
    expanded_states = 0
    generated_states = 1
    if early_termination and ss.goal_reached(init):
        return extract_path(init), {
            "expanded_states": str(expanded_states),
            "goal_depth": str(init.g),
        }

    def bloom_key(state: State) -> bytes:
        key = []
        for v in state.assignments:
            if isinstance(v, bool):
                key.append(f"{int(v)}")
            elif isinstance(v, int):
                key.append(f"{v}")
            elif isinstance(v, Fraction):
                key.append(f"{v.numerator}/{v.denominator}")
            elif isinstance(v, str):
                key.append(v)
        return "|".join(key).encode("utf-8")

    if not ss.is_temporal or weak_equality:
        BLOOM_ITEMS = 20_000_000
        BLOOM_FP_RATE = 1e-4
        visited_states = BloomFilter(max_elements=BLOOM_ITEMS, error_rate=BLOOM_FP_RATE)
        visited_states.add(bloom_key(init))

    init_h = heuristic.eval(init, ss)
    if init_h is None:
        return None, {"expanded_states": str(0)}

    QUEUE_BOUND = 400_000
    open = BoundedPriorityQueue(QUEUE_BOUND)
    open.push(PrioritizedItem(init_h, init, generated_states))
    while len(open) > 0:
        if timeout is not None and time.time() - st > timeout:
            raise TimeoutError
        item = open.pop()
        state = item.state
        expanded_states += 1
        if not early_termination and ss.goal_reached(state):
            return extract_path(state), {
                "expanded_states": str(expanded_states),
                "goal_depth": str(state.g),
            }

        candidate_states = []
        for succ_state in ss.get_successor_states(state):
            if early_termination and ss.goal_reached(succ_state):
                return extract_path(succ_state), {
                    "expanded_states": str(expanded_states),
                    "goal_depth": str(succ_state.g),
                }

            if not ss.is_temporal or weak_equality:
                succ_state_key = bloom_key(succ_state)
                if succ_state_key not in visited_states:
                    visited_states.add(succ_state_key)
                    candidate_states.append(succ_state)
            else:
                candidate_states.append(succ_state)

        for succ_state, h in heuristic.eval_gen(candidate_states, ss):
            if h is not None:
                f = (1 - weight) * succ_state.g + weight * h
                open.push(PrioritizedItem(f, succ_state, generated_states))
            generated_states += 1

    return None, {"expanded_states": str(expanded_states)}


def ehc_search(
    ss: SearchSpaceABC,
    heuristic: Heuristic,
    timeout: Optional[float] = None,
    early_termination: bool = False,
    weak_equality: bool = False,
) -> Tuple[Optional[List[Action]], Dict[str, str]]:
    st = time.time()
    init = ss.initial_state()
    expanded_states = 0
    if early_termination and ss.goal_reached(init):
        return extract_path(init), {
            "expanded_states": str(expanded_states),
            "goal_depth": str(init.g),
        }

    open: Deque[State] = deque()
    open.append(init)
    best_h = heuristic.eval(init, ss)
    if best_h is None:
        return None, {"expanded_states": str(0)}

    closed = set()
    while len(open) > 0:
        if timeout is not None and time.time() - st > timeout:
            raise TimeoutError
        state = open.popleft()
        expanded_states += 1
        if not ss.is_temporal or weak_equality:
            closed.add(state_representation(state, weak_equality))

        if not early_termination and ss.goal_reached(state):
            return extract_path(state), {
                "expanded_states": str(expanded_states),
                "goal_depth": str(state.g),
            }

        candidate_states = []
        for succ_state in ss.get_successor_states(state):
            if early_termination and ss.goal_reached(succ_state):
                return extract_path(succ_state), {
                    "expanded_states": str(expanded_states),
                    "goal_depth": str(succ_state.g),
                }

            if not ss.is_temporal or weak_equality:
                state_repr = state_representation(succ_state, weak_equality)
                if state_repr not in closed:
                    candidate_states.append(succ_state)
            else:
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
    return None, {"expanded_states": str(expanded_states)}
