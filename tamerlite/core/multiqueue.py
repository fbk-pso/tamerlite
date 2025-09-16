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

import functools
import heapq
from dataclasses import dataclass
#import math
from decimal import Decimal, getcontext
import time
from typing import Callable, List, Optional, Tuple, Type
from tamerlite.core.search_space import SearchSpace, State
from tamerlite.core.heuristics import Heuristic


@dataclass
class StateContainer:
    state: State
    expanded: bool


@dataclass
class PrioritizedItem:
    heuristic: float
    state_container: StateContainer

    def __lt__(self, other):
        if self.heuristic < other.heuristic:
            return True
        if self.heuristic > other.heuristic:
            return False
        return len(self.state_container.state.todo) < len(
            other.state_container.state.todo
        )

# def compute_entropy(queue: List[PrioritizedItem]) -> float:
#     if len(queue) == 0:
#         return 0.0
#     logits = [-item.heuristic for item in queue if not item.state_container.expanded]
#     logits.sort()
#     #print("Logits for entropy computation:", logits)
#     exps = [math.exp(logit) for logit in logits]
#     sum_exps = sum(exps)
#     #print("Sum of exp(logit):", sum_exps)
#     softmax = [exp / sum_exps for exp in exps]
#     entropy = -sum(p * math.log(p) for p in softmax if p > 0)
#     print("Entropy computed:", entropy)
#     normalized_entropy = entropy / math.log(len(logits)) if len(logits) > 1 else 0.0
#     print("Denominator for normalization:", len(logits))
#     return normalized_entropy


class MQSwitchPolicy():
    def switching_policy(self, i: int):
        raise NotImplementedError()

    def notify_push(self, i: int, item: PrioritizedItem):
        pass

    def notify_pop(self, i: int, item: PrioritizedItem):
        pass


class RoundRobinSwitchPolicy(MQSwitchPolicy):
    def __init__(self, num_queues: int):
        self.num_queues = num_queues

    def switching_policy(self, i: int):
        return i % self.num_queues


class EntropySwitchPolicy(MQSwitchPolicy):
    def __init__(self, threshold: float, max_successive_steps: Optional[int]):
        self.threshold = threshold
        self.max_successive_steps = max_successive_steps
        self.exp_sum = Decimal(0.0) # sum of exp(logit)
        self.exp_logit_sum = Decimal(0.0) # sum of exp(logit) * logit
        self.n = 0
        self._successive_steps = 0

    def switching_policy(self, i: int):
        nentropy = self._compute_normalized_entropy()
        # print(f"Normalized Entropy: {nentropy:.4f}")
        if (self.max_successive_steps is not None and self._successive_steps >= self.max_successive_steps) or nentropy > self.threshold:
            self._successive_steps = 0
            return 0 # Use the astar queue
        else:
            self._successive_steps += 1
            return 1 # Use the rank queue

    def _compute_normalized_entropy(self) -> Decimal:
        if self.n <= 1:
            return Decimal(0.0)
        entropy = self.exp_sum.ln() - self.exp_logit_sum / self.exp_sum
        return entropy / Decimal(self.n).ln()

    def notify_push(self, i: int, item: PrioritizedItem):
        if i == 1: # Only consider push on rank queue
            self.n += 1
            logit = Decimal(-item.heuristic)
            self.exp_sum += logit.exp()
            self.exp_logit_sum += logit.exp() * logit

    def notify_pop(self, i: int, item: PrioritizedItem):
        if not item.state_container.expanded: # Only consider the first time I pop this state
            self.n -= 1
            if i == 1 or self.n == 0:
                logit = Decimal(-item.heuristic)
            else:
                logit = Decimal(-item.state_container.state.heuristic_cache["rlrank"])
            self.exp_sum -= logit.exp()
            self.exp_logit_sum -= logit.exp() * logit

def entropy_dual_queue_search(
    ss: SearchSpace,
    astar_h: Tuple[Heuristic, float],
    rank_policy: Heuristic,
    threshold: float,
    max_successive_steps: Optional[int],
    timeout: float = None,
    early_termination: bool = False,
):
    getcontext().prec = 77
    return _multiqueue_search(
        ss=ss,
        heuristics=[astar_h, (rank_policy, 1)],
        switch_policy=EntropySwitchPolicy(threshold=threshold, max_successive_steps=max_successive_steps),
        timeout=timeout,
        early_termination=early_termination,
    )

def multiqueue_search(
    ss: SearchSpace, heuristics: List[Tuple[Heuristic, float]], timeout: float = None, early_termination: bool = False
):
    return _multiqueue_search(
        ss=ss,
        heuristics=heuristics,
        switch_policy=RoundRobinSwitchPolicy(len(heuristics)),
        timeout=timeout,
        early_termination=early_termination,
    )

def _multiqueue_search(
    ss: SearchSpace,
    heuristics: List[Tuple[Heuristic, float]],
    switch_policy: MQSwitchPolicy,
    timeout: float = None,
    early_termination: bool = False,
):
    st = time.time()
    opens = []
    closed_set = set()
    open_set = set()
    init = ss.initial_state()
    item = PrioritizedItem(0, StateContainer(init, False))
    for i, _ in enumerate(heuristics):
        open = []
        heapq.heappush(open, item)
        switch_policy.notify_push(i, item)
        opens.append(open)
    counter = 0
    states_expanded = 0
    while True:
        if timeout is not None and time.time() - st > timeout:
            raise TimeoutError
        if any(len(o) == 0 for o in opens):
            break
        i = switch_policy.switching_policy(counter)
        open = opens[i]
        item = heapq.heappop(open)
        switch_policy.notify_pop(i, item)
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
            return state.extract_solution(), {
                "expanded_states": str(states_expanded),
                "goal_depth": str(state.g),
            }

        # Here, we create a temporary list of the successor states to reuse it among multiple heuristics
        candidate_states = []
        for s in ss.get_successor_states(state):
            if early_termination and ss.goal_reached(s):
                return s.extract_solution(), {
                    "expanded_states": str(states_expanded),
                    "goal_depth": str(s.g),
                }
            if not ss.is_temporal:
                if s in closed_set or s in open_set:
                    continue
                open_set.add(s)
            candidate_states.append(s)
        candidate_containers = [StateContainer(s, False) for s in candidate_states]
        for i, (heuristic, weight) in enumerate(heuristics):
            for j, (succ_state, h) in enumerate(
                heuristic.eval_gen(candidate_states, ss)
            ):
                if h is not None:
                    f = (1 - weight) * succ_state.g + weight * h
                    item = PrioritizedItem(f, candidate_containers[j])
                    heapq.heappush(opens[i], item)
                    switch_policy.notify_push(i, item)
    return None, {"expanded_states": str(states_expanded)}
