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

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from fractions import Fraction

from unified_planning.model import DeltaSimpleTemporalNetwork


@dataclass(eq=True, frozen=True)
class OperatorNode:
    kind: str
    operands: tuple[int, ...]


@dataclass(eq=True, frozen=True)
class FluentNode:
    fluent: int


@dataclass(eq=True, frozen=True)
class ObjectNode:
    object: int


@dataclass(eq=True, frozen=True)
class InterpretedFunctionNode:
    """An interpreted function call, evaluated by invoking a real Python
    callable at search time (see `evaluate`). `operands` are indices of the
    (inlined) argument sub-expression roots, exactly like `OperatorNode`."""

    function: Callable
    return_type: str  # "bool" | "int" | "real"
    operands: tuple[int, ...]

    def call(self, *arg_values: "ConstantNode") -> "ConstantNode":
        """Calls the underlying Python callable and coerces its result to
        the declared `return_type` -- the raw callable is free to return
        any Python-native type (e.g. a plain `float` for a "real" function),
        so this normalizes it to the exact type the rest of the search space
        expects (mirroring `Simplifier.walk_interpreted_function_exp`, which
        does the analogous normalization on UP's side)."""
        r = self.function(*arg_values)
        if self.return_type == "bool":
            return bool(r)
        elif self.return_type == "int":
            return int(r)
        else:
            return Fraction(r)


ConstantNode = bool | int | Fraction | ObjectNode
ExpressionNode = OperatorNode | FluentNode | InterpretedFunctionNode | ConstantNode
Expression = tuple[ExpressionNode, ...]


def make_operator_node(kind: str, operands: tuple[int, ...]) -> ExpressionNode:
    return OperatorNode(kind, operands)


def make_interpreted_function_node(
    function: Callable, return_type: str, operands: tuple[int, ...]
) -> ExpressionNode:
    return InterpretedFunctionNode(function, return_type, operands)


def make_bool_constant_node(v: bool) -> ExpressionNode:
    return v


def make_int_constant_node(v: int) -> ExpressionNode:
    return v


def make_rational_constant_node(numerator: int, denominator: int) -> ExpressionNode:
    return Fraction(numerator=numerator, denominator=denominator)


def make_object_node(oid: int) -> ExpressionNode:
    return ObjectNode(oid)


def make_fluent_node(fluent: int) -> ExpressionNode:
    return FluentNode(fluent)


def shift_expression(exp: Expression, offset: int) -> Expression:
    res: list[ExpressionNode] = []
    for e in exp:
        if isinstance(e, OperatorNode):
            res.append(OperatorNode(e.kind, tuple([o + offset for o in e.operands])))
        elif isinstance(e, InterpretedFunctionNode):
            res.append(
                InterpretedFunctionNode(
                    e.function,
                    e.return_type,
                    tuple([o + offset for o in e.operands]),
                )
            )
        else:
            res.append(e)
    return tuple(res)


def split_expression(exp: Expression) -> tuple[Expression, ...]:
    if not isinstance(exp[-1], OperatorNode) or exp[-1].kind != "and":
        return (exp,)
    res = []
    last = 0
    for i in exp[-1].operands:
        new_exp: list[ExpressionNode] = []
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
    start_conditions: tuple[Expression, ...]
    end_conditions: tuple[Expression, ...]
    effects: tuple[Effect, ...]

    def __repr__(self):
        return (
            f"Event(action={self.action}, pos={self.pos}, "
            f"conditions={self.conditions}, "
            f"start_conditions={self.start_conditions}, "
            f"end_conditions={self.end_conditions}, effects={self.effects})"
        )


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
        n._elements = dict(self._elements.items())
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
    assignments: list[ConstantNode]
    temporal_network: DeltaSimpleTemporalNetwork | None
    todo: dict[Action, tuple[int, int]]
    active_conditions: MultiSet
    g: int
    path: list[tuple[Action, int, int]]
    heuristic_cache: dict[str, float | None] = field(default_factory=dict)

    def __hash__(self) -> int:
        return hash(tuple(self.assignments))

    def __eq__(self, oth) -> bool:
        if self.temporal_network is None:
            return bool(self.assignments == oth.assignments)
        else:
            return False

    def get_value(self, fluent: int) -> ConstantNode:
        return self.assignments[fluent]

    def clone(self):
        assignments = list(self.assignments)
        todo = self.todo.copy()
        tn = self.temporal_network.copy_stn() if self.temporal_network else None
        return State(
            assignments, tn, todo, self.active_conditions.clone(), self.g, self.path[:]
        )


class MutexChecker:
    def __init__(
        self,
        event_fluents: list[
            list[tuple[set[int], set[int], set[int], set[int], set[int]]]
        ],
    ):
        self._event_fluents = event_fluents
        self._cache: dict[tuple[tuple[Action, int], tuple[Action, int]], bool] = {}

    def __contains__(
        self, events_pair: tuple[tuple[Action, int], tuple[Action, int]]
    ) -> bool:
        (a1, i1), (a2, i2) = events_pair
        if a1 == a2:
            return True

        are_mutex = self._cache.get(events_pair, None)
        if are_mutex is None:
            (_, a_writes, a_read_writes, _, _) = self._event_fluents[a1.idx][i1]
            (b_reads, b_writes, _, _, _) = self._event_fluents[a2.idx][i2]
            are_mutex = not (
                b_reads.isdisjoint(a_writes) and a_read_writes.isdisjoint(b_writes)
            )
            self._cache[events_pair] = are_mutex
        return are_mutex


class PrecedenceChecker:
    def __init__(
        self,
        event_fluents: list[
            list[tuple[set[int], set[int], set[int], set[int], set[int]]]
        ],
    ):
        self._event_fluents = event_fluents
        self._cache: dict[tuple[tuple[Action, int], tuple[Action, int]], bool] = {}

    def __contains__(
        self, events_pair: tuple[tuple[Action, int], tuple[Action, int]]
    ) -> bool:
        (a1, i1), (a2, i2) = events_pair
        if a1 == a2:
            return False

        res = self._cache.get(events_pair, None)
        if res is None:
            (_, a_writes, _, _, a_end_cond_reads) = self._event_fluents[a1.idx][i1]
            (_, b_writes, _, b_start_cond_reads, _) = self._event_fluents[a2.idx][i2]
            res = not (
                a_writes.isdisjoint(b_start_cond_reads)
                and b_writes.isdisjoint(a_end_cond_reads)
            )
            self._cache[events_pair] = res
        return res


def get_fluent_value(fluent: int, state: State) -> ConstantNode:
    return state.get_value(fluent)


def evaluate(exp: Expression, state: State) -> ConstantNode:
    res: list[ExpressionNode] = []
    for e in exp:
        if isinstance(e, (bool, int, Fraction)):
            res.append(e)
        elif isinstance(e, FluentNode):
            res.append(get_fluent_value(e.fluent, state))
        elif isinstance(e, ObjectNode):
            res.append(e)
        elif isinstance(e, InterpretedFunctionNode):
            arg_values = [res[i] for i in e.operands]
            res.append(e.call(*arg_values))  # type: ignore[arg-type]
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
                v: int | Fraction = 0
                for i in e.operands:
                    v += res[i]  # type: ignore[operator]
                res.append(v)
            elif e.kind == "-":
                res.append(res[e.operands[0]] - res[e.operands[1]])  # type: ignore[operator]
            elif e.kind == "*":
                v = 1
                for i in e.operands:
                    v *= res[i]  # type: ignore[operator]
                res.append(v)
            elif e.kind == "/":
                res.append(Fraction(res[e.operands[0]], res[e.operands[1]]))  # type: ignore[arg-type]
    assert isinstance(res[-1], (bool, int, Fraction, ObjectNode))
    return res[-1]


def simplify(
    exp: Expression,
    assignments: dict[int, ConstantNode],
    evaluate_interpreted_functions: bool = False,
) -> Expression:
    """This function simplifies the given expression using the given assignments.

    If `evaluate_interpreted_functions` is True, an interpreted function whose
    operands have all been folded to constants is actually called and
    replaced by its result; otherwise (the default) it is always re-emitted
    unchanged."""

    # We iterate over the expression elements and we store the simplified value
    # in the res vector
    res: list[ExpressionNode] = []
    for e in exp:
        if isinstance(e, (bool, int)):
            res.append(e)
        elif isinstance(e, Fraction):
            if e.denominator == 1:
                res.append(int(e))
            else:
                res.append(e)
        elif isinstance(e, FluentNode):
            v = assignments.get(e.fluent)
            if v is None:
                res.append(e)
            else:
                res.append(v)
        elif isinstance(e, ObjectNode):
            res.append(e)
        elif isinstance(e, InterpretedFunctionNode):
            operand_values = [res[i] for i in e.operands]
            if evaluate_interpreted_functions and all(
                isinstance(v, (bool, int, Fraction, ObjectNode)) for v in operand_values
            ):
                res.append(e.call(*operand_values))  # type: ignore[arg-type]
            else:
                # Either not all operands have been folded to constants yet,
                # or the caller opted out (the default): re-emit the node unchanged.
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
                v: bool | OperatorNode | FluentNode = res[e.operands[0]]
                if isinstance(v, bool):
                    res.append(not v)
                else:
                    res.append(e)
            elif e.kind == "==":
                v1 = res[e.operands[0]]
                v2 = res[e.operands[1]]
                if v1 == v2 or (
                    (isinstance(v1, (int, Fraction)))
                    and (isinstance(v2, (int, Fraction)))
                ):
                    res.append(v1 == v2)
                else:
                    res.append(e)
            elif e.kind in ["<=", "<", "-", "/"]:
                v1 = res[e.operands[0]]
                v2 = res[e.operands[1]]
                if (isinstance(v1, (int, Fraction))) and (
                    isinstance(v2, (int, Fraction))
                ):
                    r: bool | int | Fraction
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
                    if isinstance(v1, (int, Fraction)):
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
    final_res: list[ExpressionNode] = []
    stack = [(len(res) - 1, False)]
    operands_stack = []
    while len(stack) > 0:
        idx, processed = stack.pop()
        e = res[idx]
        if isinstance(e, (bool, int, Fraction, FluentNode, ObjectNode)):
            operands_stack.append(len(final_res))
            final_res.append(e)
        else:
            assert isinstance(e, (OperatorNode, InterpretedFunctionNode))
            if processed:
                operands = [operands_stack.pop() for _ in e.operands]
                operands.reverse()
                operands_stack.append(len(final_res))
                if isinstance(e, InterpretedFunctionNode):
                    final_res.append(
                        InterpretedFunctionNode(
                            e.function, e.return_type, tuple(operands)
                        )
                    )
                else:
                    final_res.append(OperatorNode(e.kind, tuple(operands)))
            else:
                stack.append((idx, True))
                stack.extend((i, False) for i in e.operands[::-1])

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
        initial_state: list[ConstantNode] | None = None,
    ) -> State:
        pass

    @abstractmethod
    def get_successor_state(self, state: State, action: Action) -> State | None:
        pass

    @abstractmethod
    def get_successor_states(self, state: State) -> Iterator[State]:
        pass

    @abstractmethod
    def goal_reached(self, state: State, goal: Expression | None = None) -> bool:
        pass

    @abstractmethod
    def subgoals_sat(
        self, state: State, goal: Expression | None = None
    ) -> set[Expression]:
        pass

    @abstractmethod
    def build_plan(
        self, path: list[Action]
    ) -> list[tuple[Fraction | None, Action, Fraction | None]]:
        pass


class SearchSpace(SearchSpaceABC):
    def __init__(
        self,
        actions_duration: list[tuple[Expression, Expression, bool, bool] | None],
        events: dict[Action, list[tuple[Timing, Event]]],
        actions: list[Action],
        compression_safe_actions: list[bool] | None,
        action_objects: list[list[int]] | None,
        obj_to_prev_actions_map: list[set[Action]] | None,
        initial_state: list[ConstantNode] | None = None,
        goal: Expression | None = None,
        relevant_actions: list[Action] | None = None,
        deadline: Fraction | None = None,
        epsilon: Fraction | None = None,
    ):
        self._actions_duration = actions_duration
        self._events = events
        self._relevant_actions = (
            relevant_actions if relevant_actions is not None else list(actions)
        )
        self._compression_safe_actions = compression_safe_actions
        self._action_objects = action_objects
        self._obj_to_prev_actions_map = obj_to_prev_actions_map
        self._initial_state = initial_state
        self._goal = goal
        self._deadline = deadline
        self._start_plan = "start_plan"
        self._end_plan = "end_plan"
        self._epsilon = Fraction(1, 100) if epsilon is None else epsilon
        self._is_temporal = any(v is not None for v in actions_duration)
        self._counter = 0

        event_fluents: list[
            list[tuple[set[int], set[int], set[int], set[int], set[int]]]
        ] = [[] for _ in actions]
        for a, le in self._events.items():
            duration = self._actions_duration[a.idx]
            for i, (_, e) in enumerate(le):
                reads = set(get_fluents(e.conditions))
                reads.update(x for eff in e.effects for x in get_fluents(eff.value))
                if i == 0 and duration is not None:
                    # The duration bounds are read when the action is opened,
                    # i.e. at its first (start) event, so that event reads the
                    # fluents they mention just like a condition would. Without
                    # this, nothing orders the start against the events writing
                    # those fluents and `build_plan` is free to schedule the
                    # action where its duration does not hold.
                    reads.update(get_fluents(duration[0]))
                    reads.update(get_fluents(duration[1]))
                writes = {eff.fluent for eff in e.effects}
                read_writes = reads.union(writes)
                start_cond_reads = {
                    f for c in e.start_conditions for f in get_fluents(c)
                }
                end_cond_reads = {f for c in e.end_conditions for f in get_fluents(c)}
                event_fluents[a.idx].append(
                    (reads, writes, read_writes, start_cond_reads, end_cond_reads)
                )
        self._mutex = MutexChecker(event_fluents)
        self._precedence = PrecedenceChecker(event_fluents)

    @property
    def is_temporal(self) -> bool:
        return self._is_temporal

    @property
    def relevant_actions(self) -> list[Action]:
        return self._relevant_actions

    @relevant_actions.setter
    def relevant_actions(self, relevant_actions: list[Action]):
        self._relevant_actions = relevant_actions

    def reset(self):
        pass

    def initial_state(
        self,
        initial_state: list[ConstantNode] | None = None,
    ) -> State:
        if self._is_temporal:
            tn = DeltaSimpleTemporalNetwork()
            if self._deadline is not None:
                tn.insert_interval(
                    self._start_plan,
                    self._end_plan,
                    left_bound=self._deadline,
                    right_bound=self._deadline,
                )
        else:
            tn = None
        if initial_state is not None:
            return State(initial_state, tn, {}, MultiSet(), 0, [])
        else:
            # `initial_state` can be None if the initial state was already
            # provided when instantiating the class
            assert self._initial_state is not None
            return State(self._initial_state, tn, {}, MultiSet(), 0, [])

    def get_successor_state(self, state: State, action: Action) -> State | None:
        return self.get_successor_state_with_compression(state, action, True)

    def get_successor_state_with_compression(
        self, state: State, action: Action, enable_compression_safe_actions: bool
    ) -> State | None:
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
            if (
                enable_compression_safe_actions
                and self._compression_safe_actions is not None
                and self._compression_safe_actions[action.idx]
                and new_state is not None
                and len(events) > 1
            ):
                _, id = new_state.todo.pop(action)
                for index in range(1, len(events)):
                    state = new_state.clone()
                    new_state.g += 1
                    _, e = events[index]
                    new_state = self._expand_event(state, new_state, e, index, id)
                    id += 1

        result: State | None = new_state
        return result

    def get_successor_states(self, state: State) -> Iterator[State]:
        for action in self._relevant_actions:
            new_state = self.get_successor_state(state, action)
            if new_state:
                yield new_state

    def goal_reached(self, state: State, goal: Expression | None = None) -> bool:
        if len(state.todo) > 0:
            return False
        if goal is not None:
            res = evaluate(goal, state)
        else:
            # `goal` can be None if the goal was already provided when
            # instantiating the class
            assert self._goal is not None
            res = evaluate(self._goal, state)
        assert isinstance(res, bool)
        return res

    def subgoals_sat(
        self, state: State, goal: Expression | None = None
    ) -> set[Expression]:
        if goal is not None:
            goals = split_expression(goal)
        else:
            # `goal` can be None if the goal was already provided when
            # instantiating the class
            assert self._goal is not None
            goals = split_expression(self._goal)
        res = set()
        for g in goals:
            if evaluate(g, state):
                res.add(g)
        return res

    def _expand_event(
        self, state: State, new_state: State, e: Event, index: int, id: int
    ) -> State | None:
        new_state.path.append((e.action, e.pos, id))
        # check conditions
        if not evaluate(e.conditions, state):
            return None
        # check active conditions
        for c in new_state.active_conditions:
            if not evaluate(c, state):
                return None
        # remove end conditions
        for c in e.end_conditions:
            new_state.active_conditions.remove(c)
        # insert start conditions
        for c in e.start_conditions:
            new_state.active_conditions.add(c)
        # apply effects
        for eff in e.effects:
            f = eff.fluent
            v = evaluate(eff.value, state)
            new_state.assignments[f] = v
        # check active conditions
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
        events: list[tuple[Timing, Event]],
    ) -> State | None:
        if (
            self._action_objects is not None
            and self._obj_to_prev_actions_map is not None
        ):
            for obj in self._action_objects[action.idx]:
                prev_actions = self._obj_to_prev_actions_map[obj]
                if not prev_actions or action in prev_actions:
                    continue

                if not any(a in prev_actions for a, _, _ in state.path):
                    return None

        if self._is_temporal:
            assert new_state.temporal_network is not None
            start = (action, True, self._counter)
            end = (action, False, self._counter)
            self._counter += 1
            duration = self._actions_duration[action.idx]
            lower: int | Fraction
            upper: int | Fraction
            if duration is None:
                lower, upper = 0, 0
            else:
                evaluated_lower = evaluate(duration[0], state)
                assert isinstance(evaluated_lower, (int, Fraction))
                lower = evaluated_lower
                if duration[2]:
                    lower += self._epsilon
                evaluated_upper = evaluate(duration[1], state)
                assert isinstance(evaluated_upper, (int, Fraction))
                upper = evaluated_upper
                if duration[3]:
                    upper -= self._epsilon
            new_state.temporal_network.insert_interval(
                start, end, left_bound=lower, right_bound=upper
            )
            new_state.temporal_network.add(self._start_plan, start, 0)
            new_state.temporal_network.add(end, self._end_plan, -self._epsilon)
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
        return self._expand_event(state, new_state, events[0][1], 0, id)

    def build_plan(
        self, path: list[Action]
    ) -> list[tuple[Fraction | None, Action, Fraction | None]]:
        if not self.is_temporal:
            return [(None, a, None) for a in path]

        tn = DeltaSimpleTemporalNetwork()
        todo: dict[Action, tuple[int, int]] = {}
        event_path: list[tuple[Event, int]] = []
        counter = 0
        state = self.initial_state()
        for action in path:
            action_events = self._events[action]
            if action in todo:
                index, id = todo[action]
                if index + 1 >= len(action_events):
                    todo.pop(action)
                else:
                    todo[action] = (index + 1, id + 1)

                _, e = action_events[index]
                for e2, id2 in event_path:
                    if ((e.action, e.pos), (e2.action, e2.pos)) in self._mutex:
                        b = -self._epsilon
                        tn.add((e2.action, e2.pos, id2), (e.action, e.pos, id), b)
                    elif ((e2.action, e2.pos), (e.action, e.pos)) in self._precedence:
                        tn.add(
                            (e2.action, e2.pos, id2), (e.action, e.pos, id), Fraction(0)
                        )

                for a, i in todo.items():
                    id2 = i[1]
                    for j in range(i[0], len(self._events[a])):
                        _, e2 = self._events[a][j]
                        if ((e.action, e.pos), (e2.action, e2.pos)) in self._mutex:
                            b = -self._epsilon
                            tn.add((e.action, e.pos, id), (e2.action, e2.pos, id2), b)
                        id2 += 1

                event_path.append((e, id))

            else:
                start = (action, True, counter)
                end = (action, False, counter)
                counter += 1
                duration = self._actions_duration[action.idx]
                lb: Fraction
                ub: Fraction
                if duration is None:
                    lb = Fraction(0)
                    ub = Fraction(0)
                else:
                    lower = evaluate(duration[0], state)
                    upper = evaluate(duration[1], state)
                    assert isinstance(lower, (int, Fraction))
                    assert isinstance(upper, (int, Fraction))
                    lb = Fraction(-lower)
                    ub = Fraction(upper)
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
                for e2, id2 in event_path:
                    ev2 = (e2.action, e2.pos, id2)
                    if ((e.action, e.pos), (e2.action, e2.pos)) in self._mutex:
                        b = -self._epsilon
                        tn.add(ev2, ev, b)
                    elif ((e2.action, e2.pos), (e.action, e.pos)) in self._precedence:
                        tn.add(ev2, ev, Fraction(0))

                for a, i in todo.items():
                    id2 = i[1]
                    for j in range(i[0], len(self._events[a])):
                        _, e2 = self._events[a][j]
                        ev2 = (e2.action, e2.pos, id2)
                        if ((e.action, e.pos), (e2.action, e2.pos)) in self._mutex:
                            b = -self._epsilon
                            tn.add(ev, ev2, b)
                        id2 += 1

                event_path.append((e, id))
                if len(action_events) > 1:
                    todo[action] = (1, id + 1)

            # Advance to the successor state only after evaluating the action's
            # duration bounds above, so they are evaluated against the pre-action state
            succ_state = self.get_successor_state_with_compression(state, action, False)
            assert succ_state is not None
            state = succ_state

        res: list[tuple[Fraction | None, Action, Fraction | None]] = []
        start_time: dict[tuple[Action, int], Fraction] = {}
        end_time: dict[tuple[Action, int], Fraction] = {}
        for ev, dist in tn.distances.items():
            if not isinstance(ev[1], bool):
                continue

            if ev[1]:
                start_time[(ev[0], ev[2])] = -dist
            else:
                end_time[(ev[0], ev[2])] = -dist

        for a_id, st in start_time.items():
            et = end_time[a_id]
            d = None if et - st == 0 else et - st
            res.append((st, a_id[0], d))

        res.sort()
        return res
