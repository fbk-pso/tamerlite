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

from unified_planning.model import DeltaSimpleTemporalNetwork
from dataclasses import dataclass, field
from fractions import Fraction
from typing import List, Tuple, Dict, Iterator, Optional, Union, Set
from abc import ABC, abstractmethod


@dataclass(eq=True, frozen=True)
class OperatorNode:
    kind: str
    operands: Tuple[int, ...]


@dataclass(eq=True, frozen=True)
class FluentNode:
    fluent: int


ExpressionNode = Union[OperatorNode, FluentNode, bool, int, Fraction, str]
Expression = Tuple[ExpressionNode, ...]


def make_operator_node(kind: str, operands: Tuple[int, ...]) -> ExpressionNode:
    return OperatorNode(kind, operands)


def make_bool_constant_node(v: bool) -> ExpressionNode:
    return v


def make_int_constant_node(v: int) -> ExpressionNode:
    return v


def make_rational_constant_node(numerator: int, denominator: int) -> ExpressionNode:
    return Fraction(numerator=numerator, denominator=denominator)


def make_object_node(name: str) -> ExpressionNode:
    return name


def make_fluent_node(fluent: int) -> ExpressionNode:
    return FluentNode(fluent)


def contains_operator(kind: str, exp: Expression) -> bool:
    for e in exp:
        if isinstance(e, OperatorNode) and e.kind == kind:
            return True
    return False


def shift_expression(exp: Expression, offset: int) -> Expression:
    res: List[ExpressionNode] = []
    for e in exp:
        if isinstance(e, OperatorNode):
            res.append(OperatorNode(e.kind, tuple([o + offset for o in e.operands])))
        else:
            res.append(e)
    return tuple(res)


def split_expression(exp: Expression) -> Tuple[Expression, ...]:
    if not isinstance(exp[-1], OperatorNode) or not exp[-1].kind == "and":
        return (exp,)
    res = []
    last = 0
    for i in exp[-1].operands:
        new_exp: List[ExpressionNode] = []
        for e in exp[last : i + 1]:
            if isinstance(e, OperatorNode):
                new_operands = tuple([j - last for j in e.operands])
                new_exp.append(OperatorNode(e.kind, new_operands))
            else:
                new_exp.append(e)
        res.append(tuple(new_exp))
        last = i + 1
    return tuple(res)


def get_fluents(exp: Expression) -> Iterator[int]:
    for e in exp:
        if isinstance(e, FluentNode):
            yield e.fluent


@dataclass(eq=True, frozen=True)
class Effect:
    fluent: int
    value: Expression


@dataclass(eq=True, frozen=True)
class Timing:
    start: bool
    delay: Fraction

    def is_from_start(self) -> bool:
        return self.start

    def is_from_end(self) -> bool:
        return not self.start


@dataclass(order=True, frozen=True)
class Action:
    idx: int


@dataclass(eq=True, frozen=True)
class Event:
    action: Action
    pos: int
    conditions: Expression
    start_conditions: Tuple[Expression, ...]
    end_conditions: Tuple[Expression, ...]
    effects: Tuple[Effect, ...]

    def __repr__(self):
        return f"Event(action={self.action}, pos={self.pos}, conditions={self.conditions}, start_conditions={self.start_conditions}, end_conditions={self.end_conditions}, effects={self.effects})"


class MultiSet:
    def __init__(self):
        self._elements = {}

    def __repr__(self):
        return str(self._elements)

    def __contains__(self, e):
        return e in self._elements

    def __iter__(self):
        return iter(self._elements.keys())

    def clone(self):
        n = MultiSet()
        n._elements = {k: v for k, v in self._elements.items()}
        return n

    def add(self, e):
        self._elements.setdefault(e, 0)
        self._elements[e] += 1

    def remove(self, e):
        self._elements[e] -= 1
        if self._elements[e] == 0:
            del self._elements[e]


@dataclass
class State:
    assignments: List[Union[bool, int, Fraction, str]]
    temporal_network: Optional[DeltaSimpleTemporalNetwork]
    todo: Dict[Action, Tuple[int, int]]
    active_conditions: MultiSet
    g: int
    path: List[Tuple[Action, int, int]]
    heuristic_cache: Dict[str, Optional[float]] = field(default_factory=dict)

    def __hash__(self) -> int:
        return hash(tuple(self.assignments))

    def __eq__(self, oth) -> bool:
        if self.temporal_network is None:
            return self.assignments == oth.assignments
        else:
            return False

    def get_value(self, fluent: int) -> Union[bool, int, Fraction, str]:
        return self.assignments[fluent]

    def clone(self):
        assignments = list(self.assignments)
        todo = self.todo.copy()
        tn = self.temporal_network.copy_stn() if self.temporal_network else None
        return State(
            assignments, tn, todo, self.active_conditions.clone(), self.g, self.path[:]
        )


def get_fluent_value(fluent: int, state: State) -> Union[bool, int, Fraction, str]:
    return state.assignments[fluent]


def evaluate(exp: Expression, state: State) -> Union[bool, int, Fraction, str]:
    res: List[ExpressionNode] = []
    for e in exp:
        if isinstance(e, bool) or isinstance(e, int) or isinstance(e, Fraction):
            res.append(e)
        elif isinstance(e, FluentNode):
            res.append(state.assignments[e.fluent])
        elif isinstance(e, str):
            res.append(e)
        else:
            assert isinstance(e, OperatorNode)
            if e.kind == "and":
                bv = True
                for i in e.operands:
                    if not res[i]:
                        bv = False
                        break
                res.append(bv)
            elif e.kind == "or":
                bv = False
                for i in e.operands:
                    if res[i]:
                        bv = True
                        break
                res.append(bv)
            elif e.kind == "not":
                res.append(not res[e.operands[0]])
            elif e.kind == "==":
                res.append(res[e.operands[0]] == res[e.operands[1]])
            elif e.kind == "<=":
                res.append(res[e.operands[0]] <= res[e.operands[1]])  # type: ignore[operator]
            elif e.kind == "<":
                res.append(res[e.operands[0]] < res[e.operands[1]])  # type: ignore[operator]
            elif e.kind == "+":
                v: Union[int, Fraction] = 0
                for i in e.operands:
                    v += res[i]  # type: ignore[operator]
                res.append(v)
            elif e.kind == "-":
                res.append(res[e.operands[0]] - res[e.operands[1]])  # type: ignore[operator]
            elif e.kind == "*":
                v = 1
                for i in e.operands:
                    v *= res[i]  # type: ignore[operator,assignment]
                res.append(v)
            elif e.kind == "/":
                res.append(Fraction(res[e.operands[0]], res[e.operands[1]]))  # type: ignore[arg-type]
    assert (
        isinstance(res[-1], bool)
        or isinstance(res[-1], int)
        or isinstance(res[-1], Fraction)
        or isinstance(res[-1], str)
    )
    return res[-1]


def simplify(
    exp: Expression, assignments: Dict[int, Union[bool, int, Fraction, str]]
) -> Expression:
    """This function simplifies the given expression using the given assignments"""

    # We iterate over the expression elements and we store the simplified value in the res vector
    res: List[ExpressionNode] = []
    for e in exp:
        if isinstance(e, bool) or isinstance(e, int):
            res.append(e)
        elif isinstance(e, Fraction):
            if e.denominator == 1:
                res.append(int(e))
            else:
                res.append(e)
        elif isinstance(e, FluentNode):
            v = assignments.get(e.fluent, None)
            if v is None:
                res.append(e)
            else:
                res.append(v)
        elif isinstance(e, str):
            res.append(e)
        else:
            assert isinstance(e, OperatorNode)
            if e.kind == "and":
                is_false = False
                operands = []
                for i in e.operands:
                    if isinstance(res[i], bool):
                        if not res[i]:
                            is_false = True
                            break
                    else:
                        operands.append(i)
                if is_false:
                    res.append(False)
                else:
                    if len(operands) == 0:
                        res.append(True)
                    elif len(operands) == 1:
                        res.append(res[operands[0]])
                    else:
                        res.append(OperatorNode("and", tuple(operands)))
            elif e.kind == "or":
                is_true = False
                operands = []
                for i in e.operands:
                    if isinstance(res[i], bool):
                        if res[i]:
                            is_true = True
                            break
                    else:
                        operands.append(i)
                if is_true:
                    res.append(True)
                else:
                    if len(operands) == 0:
                        res.append(False)
                    elif len(operands) == 1:
                        res.append(res[operands[0]])
                    else:
                        res.append(OperatorNode("or", tuple(operands)))
            elif e.kind == "not":
                v: Union[bool, OperatorNode, FluentNode] = res[e.operands[0]]
                if isinstance(v, bool):
                    res.append(not v)
                else:
                    res.append(e)
            elif e.kind == "==":
                v1 = res[e.operands[0]]
                v2 = res[e.operands[1]]
                if v1 == v2 or (
                    (isinstance(v1, int) or isinstance(v1, Fraction))
                    and (isinstance(v2, int) or isinstance(v2, Fraction))
                ):
                    res.append(v1 == v2)
                else:
                    res.append(e)
            elif e.kind in ["<=", "<", "-", "/"]:
                v1 = res[e.operands[0]]
                v2 = res[e.operands[1]]
                if (isinstance(v1, int) or isinstance(v1, Fraction)) and (
                    isinstance(v2, int) or isinstance(v2, Fraction)
                ):
                    r: Union[bool, int, Fraction]
                    if e.kind == "<=":
                        r = v1 <= v2
                    elif e.kind == "<":
                        r = v1 < v2
                    elif e.kind == "-":
                        r = v1 - v2
                    elif e.kind == "/":
                        r = Fraction(v1, v2)

                    if isinstance(r, Fraction) and r.denominator == 1:
                        r = int(r)
                    res.append(r)
                else:
                    res.append(e)
            elif e.kind in ["+", "*"]:
                v = 0 if e.kind == "+" else 1
                first_constant_operand = None
                operands = []
                for i in e.operands:
                    v1 = res[i]
                    if isinstance(v1, int) or isinstance(v1, Fraction):
                        if e.kind == "+":
                            v += v1
                        else:
                            v *= v1

                        if first_constant_operand is None:
                            first_constant_operand = i
                            operands.append(i)
                    else:
                        operands.append(i)

                if first_constant_operand is None:
                    res.append(e)
                else:
                    if isinstance(v, Fraction) and v.denominator == 1:
                        v = int(v)

                    if len(operands) == 1:
                        res.append(v)
                    else:
                        res[first_constant_operand] = v
                        res.append(OperatorNode(e.kind, tuple(operands)))

    # Keep only the nodes reachable from the root using a depth-first search
    final_res: List[ExpressionNode] = []
    stack = [(len(res) - 1, False)]
    operands_stack = []
    while len(stack) > 0:
        idx, processed = stack.pop()
        e = res[idx]
        if (
            isinstance(e, bool)
            or isinstance(e, int)
            or isinstance(e, Fraction)
            or isinstance(e, FluentNode)
            or isinstance(e, str)
        ):
            operands_stack.append(len(final_res))
            final_res.append(e)
        else:
            if processed:
                operands = [operands_stack.pop() for _ in e.operands]
                operands.reverse()
                operands_stack.append(len(final_res))
                final_res.append(OperatorNode(e.kind, tuple(operands)))
            else:
                stack.append((idx, True))
                for i in e.operands[::-1]:
                    stack.append((i, False))

    return tuple(final_res)


class SearchSpaceABC(ABC):
    @property
    @abstractmethod
    def is_temporal(self) -> bool:
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def initial_state(
        self,
        initial_state: Optional[List[Union[bool, int, Fraction, str]]] = None,
    ) -> State:
        pass

    @abstractmethod
    def get_successor_state(self, state: State, action: Action) -> Optional[State]:
        pass

    @abstractmethod
    def get_successor_states(self, state: State) -> Iterator[State]:
        pass

    @abstractmethod
    def goal_reached(self, state: State, goal: Optional[Expression] = None) -> bool:
        pass

    @abstractmethod
    def subgoals_sat(
        self, state: State, goal: Optional[Expression] = None
    ) -> Set[Expression]:
        pass

    @abstractmethod
    def build_plan(
        self, state: State
    ) -> List[Tuple[Optional[Fraction], Action, Optional[Fraction]]]:
        pass


class SearchSpace(SearchSpaceABC):
    def __init__(
        self,
        actions_duration: List[Optional[Tuple[Expression, Expression, bool, bool]]],
        events: Dict[Action, List[Tuple[Timing, Event]]],
        actions: List[Action],
        mutex: Set[Tuple[Tuple[Action, int], Tuple[Action, int]]],
        precedence: Set[Tuple[Tuple[Action, int], Tuple[Action, int]]],
        simultaneity_groups: List[Set[Tuple[Action, int]]],
        initial_state: Optional[List[Union[bool, int, Fraction, str]]] = None,
        goal: Optional[Expression] = None,
        epsilon: Optional[Fraction] = None,
    ):
        self._actions_duration = actions_duration
        self._events = events
        self._actions = actions
        self._mutex = mutex
        self._precedence = precedence
        self._simultaneity_groups = simultaneity_groups
        self._initial_state = initial_state
        self._goal = goal
        self._epsilon = Fraction(1, 100) if epsilon is None else epsilon
        self._is_temporal = any(v is not None for v in actions_duration)
        self._counter = 0

    @property
    def is_temporal(self) -> bool:
        return self._is_temporal

    def reset(self):
        pass

    def initial_state(
        self,
        initial_state: Optional[List[Union[bool, int, Fraction, str]]] = None,
    ) -> State:
        tn = DeltaSimpleTemporalNetwork() if self._is_temporal else None
        if initial_state is not None:
            return State(initial_state, tn, {}, MultiSet(), 0, [])
        else:
            # `initial_state` can be None if the initial state was already provided when instantiating the class
            assert self._initial_state is not None
            return State(self._initial_state, tn, {}, MultiSet(), 0, [])

    def get_successor_state(self, state: State, action: Action) -> Optional[State]:
        events = self._events[action]
        new_state = state.clone()
        new_state.g = state.g + 1
        if action in state.todo:
            index, id = state.todo[action]
            _, e = events[index]
            if index + 1 >= len(events):
                new_state.todo.pop(action)
            else:
                new_state.todo[action] = index + 1, id + 1
            new_state = self._expand_event(state, new_state, e, index, id)
        else:
            new_state = self._open_action(state, new_state, action, events)
        return new_state

    def get_successor_state_from_simultaneity_group(
        self, state: State, actions: Set[Tuple[Action, int]]
    ) -> Optional[State]:
        for c in state.active_conditions:
            if not evaluate(c, state):
                return None

        previous_state = state.clone()
        for action, i in actions:
            index, _ = state.todo.get(action, (0, 0))
            if i != index:
                return None
            _, e = self._events[action][index]
            for c in e.end_conditions:
                previous_state.active_conditions.remove(c)

        overall_conditions = []
        last_ev = None
        for action, i in actions:
            events = self._events[action]
            new_state = previous_state.clone()
            new_state.g = new_state.g + 1
            if i == 0:
                new_state = self._open_action(
                    previous_state, new_state, action, events, False
                )
            else:
                index, id = state.todo[action]
                assert index == i
                _, e = events[index]
                if index + 1 >= len(events):
                    new_state.todo.pop(action)
                else:
                    new_state.todo[action] = index + 1, id + 1
                new_state = self._expand_event(
                    previous_state, new_state, e, index, id, False
                )
            if new_state is None:
                return None
            if last_ev is not None:
                new_state.temporal_network.insert_interval(
                    last_ev, new_state.path[-1], left_bound=0, right_bound=0
                )
                if not new_state.temporal_network.check_stn():
                    return None
            overall_conditions.extend(list(events[i][1].start_conditions))
            last_ev = new_state.path[-1]
            previous_state = new_state

        for c in overall_conditions:
            if not evaluate(c, new_state):
                return None
            new_state.active_conditions.add(c)

        return new_state

    def get_successor_states(self, state: State) -> Iterator[State]:
        for action in self._actions:
            new_state = self.get_successor_state(state, action)
            if new_state:
                yield new_state
        for action_group in self._simultaneity_groups:
            new_state = self.get_successor_state_from_simultaneity_group(
                state, action_group
            )
            if new_state:
                yield new_state

    def goal_reached(self, state: State, goal: Optional[Expression] = None) -> bool:
        if len(state.todo) > 0:
            return False
        if goal is not None:
            res = evaluate(goal, state)
        else:
            # `goal` can be None if the goal was already provided when instantiating the class
            assert self._goal is not None
            res = evaluate(self._goal, state)
        assert isinstance(res, bool)
        return res

    def subgoals_sat(
        self, state: State, goal: Optional[Expression] = None
    ) -> Set[Expression]:
        if goal is not None:
            goals = split_expression(goal)
        else:
            # `goal` can be None if the goal was already provided when instantiating the class
            assert self._goal is not None
            goals = split_expression(self._goal)
        res = set()
        for g in goals:
            if evaluate(g, state):
                res.add(g)
        return res

    def _expand_event(
        self,
        state: State,
        new_state: State,
        e: Event,
        index: int,
        id: int,
        use_durative_conditions: bool = True,
    ) -> Optional[State]:
        new_state.path.append((e.action, e.pos, id))
        # check conditions
        if not evaluate(e.conditions, state):
            return None
        # check active conditions
        if use_durative_conditions:
            for c in new_state.active_conditions:
                if not evaluate(c, state):
                    return None
        # remove end conditions
        if use_durative_conditions:
            for c in e.end_conditions:
                new_state.active_conditions.remove(c)
        # insert start conditions
        if use_durative_conditions:
            for c in e.start_conditions:
                new_state.active_conditions.add(c)
        # apply effects
        for eff in e.effects:
            f = eff.fluent
            v = evaluate(eff.value, state)
            new_state.assignments[f] = v
        # check active conditions
        if use_durative_conditions:
            for c in new_state.active_conditions:
                if not evaluate(c, new_state):
                    return None
        if self._is_temporal:
            # update TN
            assert new_state.temporal_network is not None
            e_id = (e.action, index)
            if len(state.path) > 0:
                for e2_action, e2_pos, id2 in state.path:
                    e2_id = (e2_action, e2_pos)
                    if (e_id, e2_id) in self._mutex:
                        new_state.temporal_network.add(
                            (e2_action, e2_pos, id2),
                            (e.action, e.pos, id),
                            -self._epsilon,
                        )
                    else:
                        new_state.temporal_network.add(
                            (e2_action, e2_pos, id2), (e.action, e.pos, id), 0
                        )
            for a, i in new_state.todo.items():
                id2 = i[1]
                for j in range(len(self._events[a][i[0] :])):
                    e2_id = (a, i[0] + j)
                    e2 = (a, i[0] + j, id2)
                    if (e_id, e2_id) in self._mutex:
                        new_state.temporal_network.add(
                            (e.action, e.pos, id), e2, -self._epsilon
                        )
                    else:
                        new_state.temporal_network.add((e.action, e.pos, id), e2, 0)
                    id2 += 1
            # check TN
            if not new_state.temporal_network.check_stn():
                return None
        return new_state

    def _open_action(
        self,
        state: State,
        new_state: State,
        action: Action,
        events: List[Tuple[Timing, Event]],
        use_durative_conditions: bool = True,
    ) -> Optional[State]:
        if self._is_temporal:
            assert new_state.temporal_network is not None
            start = (action, True, self._counter)
            end = (action, False, self._counter)
            self._counter += 1
            duration = self._actions_duration[action.idx]
            l: Union[int, Fraction]
            u: Union[int, Fraction]
            if duration is None:
                l, u = 0, 0
            else:
                l = evaluate(duration[0], state)  # type: ignore[assignment]
                assert isinstance(l, int) or isinstance(l, Fraction)
                if duration[2]:
                    l += self._epsilon
                u = evaluate(duration[1], state)  # type: ignore[assignment]
                assert isinstance(u, int) or isinstance(u, Fraction)
                if duration[3]:
                    u -= self._epsilon
            new_state.temporal_network.insert_interval(
                start, end, left_bound=l, right_bound=u
            )
            id = self._counter
            for t, e in events:
                ev = (e.action, e.pos, self._counter)
                if t.is_from_start():
                    new_state.temporal_network.insert_interval(
                        start, ev, left_bound=t.delay, right_bound=t.delay
                    )
                else:
                    new_state.temporal_network.insert_interval(
                        end, ev, left_bound=t.delay, right_bound=t.delay
                    )
                self._counter += 1
            if len(events) > 1:
                new_state.todo[action] = 1, id + 1
        else:
            id = self._counter
        return self._expand_event(
            state, new_state, events[0][1], 0, id, use_durative_conditions
        )

    def build_plan(
        self, state: State
    ) -> List[Tuple[Optional[Fraction], Action, Optional[Fraction]]]:
        if not self.is_temporal:
            return [(None, e[0], None) for e in state.path]

        if len(self._simultaneity_groups) > 0:
            return self._build_plan_from_stn(state.temporal_network)

        all_path = state.path
        tn = DeltaSimpleTemporalNetwork()
        todo: Dict[Action, Tuple[int, int]] = {}
        path: List[Tuple[Event, int]] = []
        counter = 0
        state = self.initial_state()
        for action, _, _ in all_path:
            succ_state = self.get_successor_state(state, action)
            assert succ_state is not None
            state = succ_state

            action_events = self._events[action]
            if action in todo:
                index, id = todo[action]
                if index + 1 >= len(action_events):
                    todo.pop(action)
                else:
                    todo[action] = (index + 1, id + 1)

                _, e = action_events[index]
                for e2, id2 in path:
                    if ((e.action, e.pos), (e2.action, e2.pos)) in self._mutex:
                        b = -self._epsilon
                        tn.add((e2.action, e2.pos, id2), (e.action, e.pos, id), b)
                    elif ((e2.action, e2.pos), (e.action, e.pos)) in self._precedence:
                        tn.add((e2.action, e2.pos, id2), (e.action, e.pos, id), 0)

                for a, i in todo.items():
                    id2 = i[1]
                    for j in range(i[0], len(self._events[a])):
                        _, e2 = self._events[a][j]
                        if ((e.action, e.pos), (e2.action, e2.pos)) in self._mutex:
                            b = -self._epsilon
                            tn.add((e.action, e.pos, id), (e2.action, e2.pos, id2), b)
                        id2 += 1

                path.append((e, id))

            else:
                start = (action, True, counter)
                end = (action, False, counter)
                counter += 1
                duration = self._actions_duration[action.idx]
                lb: Union[int, Fraction]
                ub: Union[int, Fraction]
                if duration is None:
                    lb = 0
                    ub = 0
                else:
                    l = evaluate(duration[0], state)
                    u = evaluate(duration[1], state)
                    assert isinstance(l, int) or isinstance(l, Fraction)
                    assert isinstance(u, int) or isinstance(u, Fraction)
                    lb = -l
                    ub = u
                    if duration[2]:
                        lb -= self._epsilon
                    if duration[3]:
                        ub -= self._epsilon

                tn.add(start, end, lb)
                tn.add(end, start, ub)
                id = counter
                for t, e in action_events:
                    ev = (e.action, e.pos, counter)
                    b1 = -t.delay
                    b2 = t.delay
                    if t.is_from_start():
                        tn.add(start, ev, b1)
                        tn.add(ev, start, b2)
                    else:
                        tn.add(end, ev, b1)
                        tn.add(ev, end, b2)
                    counter += 1

                e = action_events[0][1]
                ev = (e.action, e.pos, id)
                for e2, id2 in path:
                    ev2 = (e2.action, e2.pos, id2)
                    if ((e.action, e.pos), (e2.action, e2.pos)) in self._mutex:
                        b = -self._epsilon
                        tn.add(ev2, ev, b)
                    elif ((e2.action, e2.pos), (e.action, e.pos)) in self._precedence:
                        tn.add(ev2, ev, 0)

                for a, i in todo.items():
                    id2 = i[1]
                    for j in range(i[0], len(self._events[a])):
                        _, e2 = self._events[a][j]
                        ev2 = (e2.action, e2.pos, id2)
                        if ((e.action, e.pos), (e2.action, e2.pos)) in self._mutex:
                            b = -self._epsilon
                            tn.add(ev, ev2, b)
                        id2 += 1

                path.append((e, id))
                if len(action_events) > 1:
                    todo[action] = (1, id + 1)

        return self._build_plan_from_stn(tn)

    def _build_plan_from_stn(
        self, tn: DeltaSimpleTemporalNetwork
    ) -> List[Tuple[Optional[Fraction], Action, Optional[Fraction]]]:
        res: List[Tuple[Optional[Fraction], Action, Optional[Fraction]]] = []
        start_time: Dict[Tuple[Action, int], Fraction] = {}
        end_time: Dict[Tuple[Action, int], Fraction] = {}
        for ev, t in tn.distances.items():
            if not isinstance(ev[1], bool):
                continue

            if ev[1]:
                start_time[(ev[0], ev[2])] = -t
            else:
                end_time[(ev[0], ev[2])] = -t

        for a_id, st in start_time.items():
            et = end_time[a_id]
            if (et - st) == 0:
                d = None
            else:
                d = et - st
            res.append((st, a_id[0], d))

        res.sort()
        return res
