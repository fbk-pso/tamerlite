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
    state_expanded = 0
    while True:
        if timeout is not None and time.time() - st > timeout:
            raise TimeoutError
        if sum([len(o) for o in opens]) == 0:
            break
        i = counter % len(opens)
        open = opens[i]
        if len(open) == 0:
            counter += 1
            continue
        item = heapq.heappop(open)
        sc = item.state_container
        if sc.expanded:
            continue
        sc.expanded = True
        state = sc.state
        if not ss.is_temporal:
            closed_set.add(state)
            open_set.discard(state)
        # print(state.path, item.heuristic)
        counter += 1
        state_expanded += 1
        if ss.goal_reached(state):
            print("Expanded states:", state_expanded)
            return state.extract_solution()
        for succ_state in ss.get_successor_states(state):
            if succ_state in closed_set or succ_state in open_set:
                continue
            if not ss.is_temporal:
                open_set.add(succ_state)
            sc = StateContainer(succ_state, False)
            for i, (heuristic, weight) in enumerate(heuristics):
                h = heuristic.eval(succ_state, ss) if weight > 0 else 0
                if h is not None:
                    f = (1-weight)*succ_state.g + weight*h
                    item = PrioritizedItem(f, sc)
                    heapq.heappush(opens[i], item)
    return None
