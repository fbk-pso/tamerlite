from itertools import combinations
from typing import Dict, List, Set, Tuple, Any
from tamerlite.core import Action


def build_compatibility_graph(
    actions: List[Tuple[Action, int]],
    mutex: Set[Tuple[Tuple[Action, int], Tuple[Action, int]]],
) -> Dict[Tuple[Action, int], Set[Tuple[Action, int]]]:
    mutex_set = set(mutex) | {(b, a) for (a, b) in mutex}
    graph: Dict[Tuple[Action, int], Set[Tuple[Action, int]]] = {
        a: set() for a in actions
    }

    for a, b in combinations(actions, 2):
        if (a, b) not in mutex_set:
            graph[a].add(b)
            graph[b].add(a)
    return graph


def find_valid_subsets(
    actions: List[Tuple[Action, int]],
    mutex: Set[Tuple[Tuple[Action, int], Tuple[Action, int]]],
) -> List[Set[Tuple[Action, int]]]:

    graph = build_compatibility_graph(actions, mutex)
    results = []

    def backtrack(current_set, candidates):
        if len(current_set) >= 2:
            results.append(current_set.copy())

        while candidates:
            a = candidates.pop()
            new_set = current_set | {a}
            new_candidates = candidates & graph[a]
            backtrack(new_set, new_candidates)

    backtrack(set(), set(actions))
    return results


def get_all_simultaneity_actions_groups(
    actions: List[Tuple[Action, int]],
    mutex: Set[Tuple[Tuple[Action, int], Tuple[Action, int]]],
) -> List[Set[Tuple[Action, int]]]:
    return find_valid_subsets(actions, mutex)


def build_graph(
    actions: List[Tuple[Action, int]],
    arcs: Set[Tuple[Tuple[Action, int], Tuple[Action, int]]],
) -> Dict[Tuple[Action, int], Set[Tuple[Action, int]]]:
    graph: Dict[Tuple[Action, int], Set[Tuple[Action, int]]] = {
        a: set() for a in actions
    }
    for u, v in arcs:
        graph[u].add(v)
    return graph


def build_mutex_map(
    actions: List[Tuple[Action, int]],
    mutex: Set[Tuple[Tuple[Action, int], Tuple[Action, int]]],
) -> Dict[Tuple[Action, int], Set[Tuple[Action, int]]]:
    m: Dict[Tuple[Action, int], Set[Tuple[Action, int]]] = {a: set() for a in actions}
    for a, b in mutex:
        m[a].add(b)
        m[b].add(a)
    return m


def find_mutex_free_cycles(
    actions: List[Tuple[Action, int]],
    mutex: Set[Tuple[Tuple[Action, int], Tuple[Action, int]]],
    arcs: Set[Tuple[Tuple[Action, int], Tuple[Action, int]]],
) -> List[Set[Tuple[Action, int]]]:

    graph = build_graph(actions, arcs)
    mutex_map = build_mutex_map(actions, mutex)

    cycles = set()

    def is_mutex_free(node, path_set):
        return all(node not in mutex_map[p] for p in path_set)

    def canonicalize_cycle(path):
        n = len(path)
        rotations = (tuple(path[i:] + path[:i]) for i in range(n))
        return min(rotations)

    def dfs(start, current, path, path_set):
        for nxt in graph[current]:
            if nxt == start and len(path) >= 2:
                cyc = canonicalize_cycle(path.copy())
                cycles.add(cyc)
                continue

            if nxt in path_set:
                continue

            if not is_mutex_free(nxt, path_set):
                continue

            path.append(nxt)
            path_set.add(nxt)
            dfs(start, nxt, path, path_set)
            path.pop()
            path_set.remove(nxt)

    for a in actions:
        dfs(a, a, [a], {a})

    return [set(c) for c in cycles]


def get_simultaneity_actions_groups(
    actions: List[Tuple[Action, int]],
    mutex: Set[Tuple[Tuple[Action, int], Tuple[Action, int]]],
    precedence: Set[Tuple[Tuple[Action, int], Tuple[Action, int]]],
    sim_set: Set[frozenset[Tuple[Action, int]]],
) -> List[Set[Tuple[Action, int]]]:
    unique_subsets = set()

    for s in sim_set:
        subsets = find_valid_subsets(list(s), mutex)
        for subset in subsets:
            unique_subsets.add(frozenset(subset))

    for c in find_mutex_free_cycles(actions, mutex, precedence):
        unique_subsets.add(frozenset(c))

    return [set(s) for s in unique_subsets]
