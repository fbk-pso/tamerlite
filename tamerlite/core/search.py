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
            print("expanded states:", counter)
            return state.extract_solution()
        for succ_state in ss.get_successor_states(state):
            open.append(succ_state)
    return None

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
        return None
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
        # print([ev.action for (ev, _) in state.path], item.heuristic)
        counter += 1
        if ss.goal_reached(state):
            print("expanded states:", counter)
            return state.extract_solution()
        for succ_state in ss.get_successor_states(state):
            if succ_state in closed_set or succ_state in open_set:
                continue
            h = heuristic.eval(succ_state, ss) if weight > 0 else 0
            if h is not None:
                f = (1-weight)*succ_state.g + weight*h
                heapq.heappush(open, PrioritizedItem(f, succ_state))
                if not ss.is_temporal:
                    open_set.add(succ_state)
    return None

def ehc_search(ss: SearchSpace, heuristic: Heuristic, timeout=None):
    st = time.time()
    init = ss.initial_state()
    open = deque()
    open.append(init)
    best_h = heuristic.eval(init, ss)
    if best_h is None:
        return None
    counter = 0
    while len(open) > 0:
        if timeout is not None and time.time() - st > timeout:
            raise TimeoutError
        state = open.popleft()
        counter += 1
        if ss.goal_reached(state):
            print("expanded states:", counter)
            return state.extract_solution()
        for succ_state in ss.get_successor_states(state):
            h = heuristic.eval(succ_state, ss)
            if h is not None:
                if h < best_h:
                    best_h = h
                    open.clear()
                open.append(succ_state)
    return None
