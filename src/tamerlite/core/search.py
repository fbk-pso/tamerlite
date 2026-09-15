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
import logging
import time
from collections import deque
from dataclasses import dataclass
from fractions import Fraction

from bloom_filter2 import BloomFilter
from min_max_heap import MinMaxHeap

from tamerlite.core.heuristics import Heuristic
from tamerlite.core.novelty import NumericNovelty
from tamerlite.core.search_space import Action, ObjectNode, SearchSpaceABC, State

logger = logging.getLogger(__name__)


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
        item: PrioritizedItem = self._heap.pop_min()
        return item

    def __len__(self) -> int:
        return int(self._heap.size())


@dataclass
class WeakEqState:
    """Wraps a State so the visited-state dedup set also compares `todo` (the
    durative actions currently in progress), which is what the temporal
    `weak_equality` dedup path needs on top of the full `assignments` compare/hash.
    """

    state: State

    def __hash__(self) -> int:
        return hash(tuple(self.state.assignments))

    def __eq__(self, oth) -> bool:
        if len(self.state.todo) != len(oth.state.todo):
            return False
        if self.state.assignments != oth.state.assignments:
            return False

        for a in self.state.todo:
            idx = self.state.todo[a][0]
            idx_id = oth.state.todo.get(a, None)
            if idx_id is None or idx_id[0] != idx:
                return False

        return True


def state_representation(state: State, weak_equality: bool) -> State | WeakEqState:
    if weak_equality:
        return WeakEqState(state)
    return state


def extract_path(state: State) -> list[Action]:
    return [a for a, _, _ in state.path]


def bfs_search(
    ss: SearchSpaceABC, timeout: float | None = None, early_termination: bool = False
) -> tuple[list[Action] | None, dict[str, str]]:
    return _basic_search(ss, True, timeout, early_termination)


def dfs_search(
    ss: SearchSpaceABC, timeout: float | None = None, early_termination: bool = False
) -> tuple[list[Action] | None, dict[str, str]]:
    return _basic_search(ss, False, timeout, early_termination)


def _basic_search(
    ss: SearchSpaceABC,
    bfs: bool,
    timeout: float | None = None,
    early_termination: bool = False,
) -> tuple[list[Action] | None, dict[str, str]]:
    name = "bfs" if bfs else "dfs"
    logger.info("%s: timeout=%s early_termination=%s", name, timeout, early_termination)
    st = time.monotonic()
    init = ss.initial_state()
    open: deque[State] = deque()
    expanded_states = 0
    generated_states = 1

    if early_termination and ss.goal_reached(init):
        return extract_path(init), {
            "expanded_states": str(expanded_states),
            "goal_depth": str(init.g),
        }
    open.append(init)

    while len(open) > 0:
        if timeout is not None and time.monotonic() - st > timeout:
            raise TimeoutError
        state = open.popleft() if bfs else open.pop()
        expanded_states += 1
        if expanded_states % 10_000 == 0:
            logger.debug(
                "%s: expanded=%d generated=%d open=%d",
                name,
                expanded_states,
                generated_states,
                len(open),
            )
        if not early_termination and ss.goal_reached(state):
            logger.info(
                "%s: goal found — expanded=%d depth=%s", name, expanded_states, state.g
            )
            return extract_path(state), {
                "expanded_states": str(expanded_states),
                "goal_depth": str(state.g),
            }
        for succ_state in ss.get_successor_states(state):
            if early_termination and ss.goal_reached(succ_state):
                logger.info(
                    "%s: goal found — expanded=%d depth=%s",
                    name,
                    expanded_states,
                    succ_state.g,
                )
                return extract_path(succ_state), {
                    "expanded_states": str(expanded_states),
                    "goal_depth": str(succ_state.g),
                }
            open.append(succ_state)
            generated_states += 1
    logger.info("%s: no solution found — expanded=%d", name, expanded_states)
    return None, {"expanded_states": str(expanded_states)}


def astar_search(
    ss: SearchSpaceABC,
    heuristic: Heuristic,
    timeout: float | None = None,
    early_termination: bool = False,
    weak_equality: bool = False,
) -> tuple[list[Action] | None, dict[str, str]]:
    return wastar_search(ss, heuristic, 0.5, timeout, early_termination, weak_equality)


def gbfs_search(
    ss: SearchSpaceABC,
    heuristic: Heuristic,
    timeout: float | None = None,
    early_termination: bool = False,
    weak_equality: bool = False,
) -> tuple[list[Action] | None, dict[str, str]]:
    return wastar_search(ss, heuristic, 1, timeout, early_termination, weak_equality)


def wastar_search(
    ss: SearchSpaceABC,
    heuristic: Heuristic,
    weight: float = 0.5,
    timeout: float | None = None,
    early_termination: bool = False,
    weak_equality: bool = False,
) -> tuple[list[Action] | None, dict[str, str]]:
    logger.info(
        "wastar_search: weight=%s timeout=%s early_termination=%s weak_equality=%s",
        weight,
        timeout,
        early_termination,
        weak_equality,
    )
    st = time.monotonic()
    open: list[PrioritizedItem] = []
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
        if timeout is not None and time.monotonic() - st > timeout:
            raise TimeoutError
        item = heapq.heappop(open)
        state = item.state
        expanded_states += 1
        if expanded_states % 10_000 == 0:
            logger.debug(
                "wastar_search: expanded=%d generated=%d open=%d",
                expanded_states,
                generated_states,
                len(open),
            )
        if not early_termination and ss.goal_reached(state):
            logger.info(
                "wastar_search: goal found — expanded=%d depth=%s",
                expanded_states,
                state.g,
            )
            return extract_path(state), {
                "expanded_states": str(expanded_states),
                "goal_depth": str(state.g),
            }

        candidate_states = []
        for succ_state in ss.get_successor_states(state):
            if early_termination and ss.goal_reached(succ_state):
                logger.info(
                    "wastar_search: goal found — expanded=%d depth=%s",
                    expanded_states,
                    succ_state.g,
                )
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

    logger.info("wastar_search: no solution found — expanded=%d", expanded_states)
    return None, {"expanded_states": str(expanded_states)}


def astar_search_memory_bounded(
    ss: SearchSpaceABC,
    heuristic: Heuristic,
    timeout: float | None = None,
    early_termination: bool = False,
    weak_equality: bool = False,
) -> tuple[list[Action] | None, dict[str, str]]:
    return wastar_search_memory_bounded(
        ss, heuristic, 0.5, timeout, early_termination, weak_equality
    )


def gbfs_search_memory_bounded(
    ss: SearchSpaceABC,
    heuristic: Heuristic,
    timeout: float | None = None,
    early_termination: bool = False,
    weak_equality: bool = False,
) -> tuple[list[Action] | None, dict[str, str]]:
    return wastar_search_memory_bounded(
        ss, heuristic, 1, timeout, early_termination, weak_equality
    )


def wastar_search_memory_bounded(
    ss: SearchSpaceABC,
    heuristic: Heuristic,
    weight: float = 0.5,
    timeout: float | None = None,
    early_termination: bool = False,
    weak_equality: bool = False,
) -> tuple[list[Action] | None, dict[str, str]]:
    logger.info(
        "wastar_search_memory_bounded: weight=%s timeout=%s "
        "early_termination=%s weak_equality=%s",
        weight,
        timeout,
        early_termination,
        weak_equality,
    )
    st = time.monotonic()
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
            elif isinstance(v, ObjectNode):
                key.append(f"{v.object}")
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
        if timeout is not None and time.monotonic() - st > timeout:
            raise TimeoutError
        item = open.pop()
        state = item.state
        expanded_states += 1
        if expanded_states % 10_000 == 0:
            logger.debug(
                "wastar_search_memory_bounded: expanded=%d generated=%d open=%d",
                expanded_states,
                generated_states,
                len(open),
            )
        if not early_termination and ss.goal_reached(state):
            logger.info(
                "wastar_search_memory_bounded: goal found — expanded=%d depth=%s",
                expanded_states,
                state.g,
            )
            return extract_path(state), {
                "expanded_states": str(expanded_states),
                "goal_depth": str(state.g),
            }

        candidate_states = []
        for succ_state in ss.get_successor_states(state):
            if early_termination and ss.goal_reached(succ_state):
                logger.info(
                    "wastar_search_memory_bounded: goal found — expanded=%d depth=%s",
                    expanded_states,
                    succ_state.g,
                )
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

    logger.info(
        "wastar_search_memory_bounded: no solution found — expanded=%d", expanded_states
    )
    return None, {"expanded_states": str(expanded_states)}


def ehc_search(
    ss: SearchSpaceABC,
    heuristic: Heuristic,
    timeout: float | None = None,
    early_termination: bool = False,
    weak_equality: bool = False,
) -> tuple[list[Action] | None, dict[str, str]]:
    logger.info(
        "ehc_search: timeout=%s early_termination=%s weak_equality=%s",
        timeout,
        early_termination,
        weak_equality,
    )
    st = time.monotonic()
    init = ss.initial_state()
    expanded_states = 0
    generated_states = 1
    if early_termination and ss.goal_reached(init):
        return extract_path(init), {
            "expanded_states": str(expanded_states),
            "goal_depth": str(init.g),
        }

    open: deque[State] = deque()
    open.append(init)
    best_h = heuristic.eval(init, ss)
    if best_h is None:
        return None, {"expanded_states": str(0)}
    logger.debug("ehc_search: initial h=%.4g", best_h)

    closed = set()
    while len(open) > 0:
        if timeout is not None and time.monotonic() - st > timeout:
            raise TimeoutError
        state = open.popleft()
        expanded_states += 1
        if not ss.is_temporal or weak_equality:
            closed.add(state_representation(state, weak_equality))

        if not early_termination and ss.goal_reached(state):
            logger.info(
                "ehc_search: goal found — expanded=%d depth=%s",
                expanded_states,
                state.g,
            )
            return extract_path(state), {
                "expanded_states": str(expanded_states),
                "goal_depth": str(state.g),
            }

        candidate_states = []
        for succ_state in ss.get_successor_states(state):
            if early_termination and ss.goal_reached(succ_state):
                logger.info(
                    "ehc_search: goal found — expanded=%d depth=%s",
                    expanded_states,
                    succ_state.g,
                )
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
            generated_states += 1
            if h is not None:
                if h < best_h:
                    best_h = h
                    logger.debug(
                        "ehc_search: improved h=%.4g expanded=%d generated=%d",
                        best_h,
                        expanded_states,
                        generated_states,
                    )
                    closed.clear()
                    open.clear()
                    open.append(succ_state)
                    break
                else:
                    open.append(succ_state)
    logger.info("ehc_search: no solution found — expanded=%d", expanded_states)
    return None, {"expanded_states": str(expanded_states)}


@dataclass
class NovBFSItem:
    """Open-list entry for `novbfs_search`: the numeric-novelty tie-break
    chain `(novelty, h^add, +-g)` plus an `idx` insertion-order tie-break for
    determinism, matching every other search's `PrioritizedItem`."""

    novelty: int  # 1 (most novel) .. 3 (not novel)
    h: float  # h^add
    g_key: float  # state.g, sign-flipped by `prefer_higher_g` at push time
    idx: int
    state: State
    partition: int  # this state's novelty partition, needed by its children

    def __lt__(self, other):
        return (self.novelty, self.h, self.g_key, self.idx) < (
            other.novelty,
            other.h,
            other.g_key,
            other.idx,
        )

    def __le__(self, other):
        return self < other


def novbfs_search(
    ss: SearchSpaceABC,
    heuristic: Heuristic,
    novelty: NumericNovelty,
    prefer_higher_g: bool,
    timeout: float | None = None,
    early_termination: bool = False,
    weak_equality: bool = False,
) -> tuple[list[Action] | None, dict[str, str]]:
    """A single open list ordered lexicographically on
    `(novelty, h^add, +-g)` -- numeric novelty first, `h^add` only as a
    tie-breaker, plan cost `g` last. `prefer_higher_g=True` is `novbfs_hg`
    (cost-*maximizing* final tie-break, to dive into longer committed plans
    and find *a* solution fast); `prefer_higher_g=False` is `novbfs_lg`
    (cost-*minimizing*). `heuristic` must be an h^add instance -- see
    `TamerLite._solve_ground_problem`'s novbfs dispatch branch, which always
    builds one internally regardless of the configured heuristic.

    `novelty` must not have had `start()` called yet -- this function calls
    it once, on the initial state, and constructs a fresh instance per
    search call (including per anytime cold-restart iteration, which
    tamerlite already implements generically -- see
    `TamerLite._anytime_solutions` -- so no restart logic needs to live
    here).

    Dedup is the standard tamerlite "generate-once" strategy shared with
    every other search here (`state_representation`/`visited_states`,
    closed at generation time), not g-based reopening: a state re-reached
    later via a strictly cheaper path is simply dropped rather than
    re-queued and re-scored against the novelty tables.

    Calls `novelty.begin_expansion()` once per popped state, before scoring
    its surviving successors -- this is what lets `NumericNovelty.eval`
    cache the parent's own features across all of one expansion's children
    instead of recomputing them per child; see its class docstring."""

    logger.info(
        "novbfs_search: prefer_higher_g=%s timeout=%s early_termination=%s "
        "weak_equality=%s",
        prefer_higher_g,
        timeout,
        early_termination,
        weak_equality,
    )
    st = time.monotonic()
    open: list[NovBFSItem] = []
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
    init_partition = novelty.start(init, init_h)
    # Seed the tables (return discarded); the root's *stored* novelty is
    # hard-coded to 1 below regardless.
    novelty.eval(init, init_partition, None, None)

    def g_key(g: int) -> float:
        return -g if prefer_higher_g else g

    heapq.heappush(open, NovBFSItem(1, init_h, g_key(init.g), 0, init, init_partition))
    while open:
        if timeout is not None and time.monotonic() - st > timeout:
            raise TimeoutError
        item = heapq.heappop(open)
        state = item.state
        expanded_states += 1
        if expanded_states % 10_000 == 0:
            logger.debug(
                "novbfs_search: expanded=%d generated=%d open=%d",
                expanded_states,
                generated_states,
                len(open),
            )
        if not early_termination and ss.goal_reached(state):
            logger.info(
                "novbfs_search: goal found — expanded=%d depth=%s",
                expanded_states,
                state.g,
            )
            return extract_path(state), {
                "expanded_states": str(expanded_states),
                "goal_depth": str(state.g),
            }

        candidate_states = []
        for succ_state in ss.get_successor_states(state):
            if early_termination and ss.goal_reached(succ_state):
                logger.info(
                    "novbfs_search: goal found — expanded=%d depth=%s",
                    expanded_states,
                    succ_state.g,
                )
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

        novelty.begin_expansion()
        for succ_state, h in heuristic.eval_gen(candidate_states, ss):
            if h is not None:
                succ_partition = novelty.partition_of(h)
                nov = novelty.eval(succ_state, succ_partition, state, item.partition)
                heapq.heappush(
                    open,
                    NovBFSItem(
                        nov,
                        h,
                        g_key(succ_state.g),
                        generated_states,
                        succ_state,
                        succ_partition,
                    ),
                )
            generated_states += 1

    logger.info("novbfs_search: no solution found — expanded=%d", expanded_states)
    return None, {"expanded_states": str(expanded_states)}
