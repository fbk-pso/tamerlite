from itertools import combinations
from typing import List, Set, Tuple, Any, Iterator

Action = Any

def get_all_simultaneity_actions_groups(actions: List[Action], mutex: Set[Tuple[Tuple[Action, int], Tuple[Action, int]]]) -> List[Set[Tuple[Action, int]]]:
    def valid_subset(subset):
        for a, b in combinations(subset, 2):
            if (a, b) in mutex:
                return False
        return True

    results = []
    snap_actions = [(a, 0) for a in actions] + [(a, 1) for a in actions]
    for k in range(2, len(snap_actions) + 1):
        for subset in combinations(snap_actions, k):
            if valid_subset(subset):
                results.append(set(subset))

    return results

def get_simultaneity_actions_groups(actions: List[Action], mutex: Set[Tuple[Tuple[Action, int], Tuple[Action, int]]]) -> List[Set[Tuple[Action, int]]]:
    raise NotImplementedError