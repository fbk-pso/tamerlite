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
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Generic, Protocol, TypeVar

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


class _SearchItem(Protocol):
    """Structural bound satisfied by `PrioritizedItem`/`NovBFSItem`: every
    open-list item `_priority_search` handles carries the `State` it was
    generated for and is totally ordered (required by `heapq`)."""

    state: State

    def __lt__(self, other: Any) -> bool: ...


ItemT = TypeVar("ItemT", bound=_SearchItem)


class _OpenList(Protocol[ItemT]):
    def push(self, item: ItemT) -> None: ...
    def pop(self) -> ItemT: ...
    def __len__(self) -> int: ...


class _PriorityQueue(Generic[ItemT]):
    """A `heapq`-backed, unbounded open list."""

    def __init__(self) -> None:
        self._heap: list[ItemT] = []

    def push(self, item: ItemT) -> None:
        heapq.heappush(self._heap, item)

    def pop(self) -> ItemT:
        return heapq.heappop(self._heap)

    def __len__(self) -> int:
        return len(self._heap)


class _BoundedPriorityQueue(Generic[ItemT]):
    """Adapts `BoundedPriorityQueue` (the module-level, `MinMaxHeap`-backed
    class this wraps -- not to be confused with this class itself) to
    `_OpenList`. `push`'s accepted/rejected return value is intentionally
    discarded. Holds `PrioritizedItem`s (`wastar_search_memory_bounded`) or
    `NovBFSItem`s (`novbfs_search_memory_bounded`); `BoundedPriorityQueue`
    only ever compares items with `<`, which both define."""

    def __init__(self, bound: int) -> None:
        self._queue: BoundedPriorityQueue = BoundedPriorityQueue(bound)

    def push(self, item: ItemT) -> None:
        self._queue.push(item)  # type: ignore[arg-type]

    def pop(self) -> ItemT:
        return self._queue.pop()  # type: ignore[return-value]

    def __len__(self) -> int:
        return len(self._queue)


# Open-list capacity shared by every `*_memory_bounded` search.
_QUEUE_BOUND = 400_000


class _Dedup(Protocol):
    def is_new(self, state: State) -> bool: ...


class _SetDedup:
    """Set-backed dedup keyed by `state_representation`, gated on
    `not ss.is_temporal or weak_equality` -- a disabled instance treats
    every state as new."""

    def __init__(self, ss: SearchSpaceABC, weak_equality: bool) -> None:
        self._weak_equality = weak_equality
        self._enabled = not ss.is_temporal or weak_equality
        self._seen: set[State | WeakEqState] = set()

    def is_new(self, state: State) -> bool:
        if not self._enabled:
            return True
        repr_ = state_representation(state, self._weak_equality)
        if repr_ in self._seen:
            return False
        self._seen.add(repr_)
        return True


def _bloom_key(state: State) -> bytes:
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
    todo = ",".join(f"{a.idx}:{i}" for a, (i, _) in sorted(state.todo.items()))
    return f"{'|'.join(key)}#{todo}".encode()


class _BloomDedup:
    """Bloom-filter-backed dedup keyed on the same equivalence as
    `WeakEqState` (`state.assignments` plus each in-progress durative action's
    `todo` event index) -- used only by `wastar_search_memory_bounded` and
    `novbfs_search_memory_bounded`.
    False positives make it strictly more aggressive (and incomplete) than
    `_SetDedup`; the two are not interchangeable."""

    BLOOM_ITEMS = 20_000_000
    BLOOM_FP_RATE = 1e-4

    def __init__(self, ss: SearchSpaceABC, weak_equality: bool) -> None:
        self._filter: BloomFilter | None = (
            BloomFilter(max_elements=self.BLOOM_ITEMS, error_rate=self.BLOOM_FP_RATE)
            if (not ss.is_temporal or weak_equality)
            else None
        )

    def is_new(self, state: State) -> bool:
        if self._filter is None:
            return True
        key = _bloom_key(state)
        if key in self._filter:
            return False
        self._filter.add(key)
        return True


def _priority_search(
    ss: SearchSpaceABC,
    heuristic: Heuristic,
    timeout: float | None,
    early_termination: bool,
    *,
    name: str,
    open: _OpenList[ItemT],
    dedup: _Dedup,
    make_root: Callable[[State, float], ItemT],
    make_child: Callable[[ItemT, State, float, int], ItemT],
    begin_expansion: Callable[[], None] | None = None,
) -> tuple[list[Action] | None, dict[str, str]]:
    """Shared skeleton for every priority-queue-based search in this module
    (`wastar_search`, `wastar_search_memory_bounded`, `novbfs_search`,
    `novbfs_search_memory_bounded`)."""

    st = time.monotonic()
    init = ss.initial_state()
    dedup.is_new(init)  # seed; return discarded, the root is never re-checked
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
    open.push(make_root(init, init_h))

    while len(open) > 0:
        if timeout is not None and time.monotonic() - st > timeout:
            raise TimeoutError
        item = open.pop()
        state = item.state
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

        candidate_states = []
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

            if dedup.is_new(succ_state):
                candidate_states.append(succ_state)

        if begin_expansion is not None:
            begin_expansion()
        for succ_state, h in heuristic.eval_gen(candidate_states, ss):
            if h is not None:
                open.push(make_child(item, succ_state, h, generated_states))
            generated_states += 1

    logger.info("%s: no solution found — expanded=%d", name, expanded_states)
    return None, {"expanded_states": str(expanded_states)}


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
    return _priority_search(
        ss,
        heuristic,
        timeout,
        early_termination,
        name="wastar_search",
        open=_PriorityQueue(),
        dedup=_SetDedup(ss, weak_equality),
        make_root=lambda init, h: PrioritizedItem(h, init, 0),
        make_child=lambda _item, s, h, idx: PrioritizedItem(
            (1 - weight) * s.g + weight * h, s, idx
        ),
    )


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
    return _priority_search(
        ss,
        heuristic,
        timeout,
        early_termination,
        name="wastar_search_memory_bounded",
        open=_BoundedPriorityQueue(_QUEUE_BOUND),
        dedup=_BloomDedup(ss, weak_equality),
        make_root=lambda init, h: PrioritizedItem(h, init, 0),
        make_child=lambda _item, s, h, idx: PrioritizedItem(
            (1 - weight) * s.g + weight * h, s, idx
        ),
    )


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
    chain `(novelty, h, +-g)`, then `len(state.todo)` (fewer durative
    actions in flight first, matching every other search's
    `PrioritizedItem`), then an `idx` insertion-order tie-break for
    determinism. `todo_len` is inert on classical problems -- `state.todo` is
    only ever populated on the temporal path -- so it only breaks otherwise-
    real ties among temporal states."""

    novelty: int  # 1 (most novel) .. 3 (not novel)
    h: float  # the search heuristic's value
    g_key: float  # state.g, sign-flipped by `prefer_higher_g` at push time
    idx: int
    state: State
    partition: int  # this state's novelty partition, needed by its children

    def __lt__(self, other):
        return (
            self.novelty,
            self.h,
            self.g_key,
            len(self.state.todo),
            self.idx,
        ) < (
            other.novelty,
            other.h,
            other.g_key,
            len(other.state.todo),
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
    `(novelty, h, +-g, len(state.todo))` -- numeric novelty first,
    the heuristic value `h` only as a tie-breaker, plan cost `g` next, and the count of
    durative actions in flight (fewer first) as a final tie-break before
    insertion order -- see `NovBFSItem`. That last key is a no-op on
    classical problems, where `state.todo` is always empty.
    `prefer_higher_g=True` is `novbfs_hg`
    (cost-*maximizing* final tie-break, to dive into longer committed plans
    and find *a* solution fast); `prefer_higher_g=False` is `novbfs_lg`
    (cost-*minimizing*). `heuristic` is any `Heuristic`: its raw value both
    picks the novelty partition (`floor(h)`) and breaks ties after novelty.
    It must never return a negative value (see `NumericNovelty.partition_of`).

    This function calls `novelty.start()` once, on the initial state's heuristic
    value, before doing anything else -- safe to call even if `novelty` was
    already used by a previous `novbfs_search` call on the same instance
    (e.g. `TamerLite._solve_ground_problem`'s `weak_equality` retry, which
    invokes the same bound `partial` twice), since `start()` fully resets
    its partition tables and parent-feature caches; see `NumericNovelty`'s
    class docstring. A fresh instance is still constructed per anytime
    cold-restart iteration, which tamerlite already implements generically
    -- see `TamerLite._anytime_solutions` -- but nothing here requires that
    beyond `start()` being called again.

    Dedup is the standard tamerlite "generate-once" strategy shared with
    every other search here (`state_representation`/`visited_states`,
    closed at generation time), not g-based reopening: a state re-reached
    later via a strictly cheaper path is simply dropped rather than
    re-queued and re-scored against the novelty tables.

    Calls `novelty.begin_expansion()` once per popped state, before scoring
    its surviving successors -- this is what lets `NumericNovelty.eval`
    cache the parent's own features across all of one expansion's children
    instead of recomputing them per child; see its class docstring."""

    return _novbfs(
        ss,
        heuristic,
        novelty,
        prefer_higher_g,
        timeout,
        early_termination,
        name="novbfs_search",
        open=_PriorityQueue(),
        dedup=_SetDedup(ss, weak_equality),
        weak_equality=weak_equality,
    )


def novbfs_search_memory_bounded(
    ss: SearchSpaceABC,
    heuristic: Heuristic,
    novelty: NumericNovelty,
    prefer_higher_g: bool,
    timeout: float | None = None,
    early_termination: bool = False,
    weak_equality: bool = False,
) -> tuple[list[Action] | None, dict[str, str]]:
    """Memory-bounded variant of `novbfs_search`: same open-list order and
    `novelty` contract, but the open list is a `BoundedPriorityQueue`
    (capacity `_QUEUE_BOUND`, evicting the worst item once full) and dedup is
    a `_BloomDedup`. Both make the search incomplete -- an evicted state is
    never regenerated, and a Bloom false positive drops a genuinely new one.
    While the bound is never hit and no false positive occurs, it expands
    exactly the same states as `novbfs_search`."""

    return _novbfs(
        ss,
        heuristic,
        novelty,
        prefer_higher_g,
        timeout,
        early_termination,
        name="novbfs_search_memory_bounded",
        open=_BoundedPriorityQueue(_QUEUE_BOUND),
        dedup=_BloomDedup(ss, weak_equality),
        weak_equality=weak_equality,
    )


def _novbfs(
    ss: SearchSpaceABC,
    heuristic: Heuristic,
    novelty: NumericNovelty,
    prefer_higher_g: bool,
    timeout: float | None,
    early_termination: bool,
    *,
    name: str,
    open: _OpenList[NovBFSItem],
    dedup: _Dedup,
    weak_equality: bool,
) -> tuple[list[Action] | None, dict[str, str]]:
    """Shared body of `novbfs_search`/`novbfs_search_memory_bounded`, which
    differ only in `open`/`dedup` (`weak_equality` is only logged here; it's
    already baked into `dedup`)."""

    logger.info(
        "%s: prefer_higher_g=%s timeout=%s early_termination=%s weak_equality=%s",
        name,
        prefer_higher_g,
        timeout,
        early_termination,
        weak_equality,
    )

    def g_key(g: int) -> float:
        return -g if prefer_higher_g else g

    def make_root(init: State, init_h: float) -> NovBFSItem:
        partition = novelty.start(init_h)
        # Seed the tables (return discarded); the root's *stored* novelty is
        # hard-coded to 1 below regardless.
        novelty.eval(init, partition, None, None)
        return NovBFSItem(1, init_h, g_key(init.g), 0, init, partition)

    def make_child(item: NovBFSItem, s: State, h: float, idx: int) -> NovBFSItem:
        partition = novelty.partition_of(h)
        nov = novelty.eval(s, partition, item.state, item.partition)
        return NovBFSItem(nov, h, g_key(s.g), idx, s, partition)

    return _priority_search(
        ss,
        heuristic,
        timeout,
        early_termination,
        name=name,
        open=open,
        dedup=dedup,
        make_root=make_root,
        make_child=make_child,
        begin_expansion=novelty.begin_expansion,
    )
