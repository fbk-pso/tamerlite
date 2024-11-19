from unified_planning.model import DeltaSimpleTemporalNetwork
from dataclasses import dataclass, field
from fractions import Fraction
from typing import List, Tuple, Dict, Iterator, Optional, Union, Set

from tamerlite.core.utils import PrefixTree


@dataclass(eq=True, frozen=True)
class OperatorNode:
    kind: str
    operands: Tuple[int, ...]


ExpressionNode = Union[OperatorNode, bool, int, Fraction, str]
Expression = Tuple[ExpressionNode, ...]


def make_operator_node(kind: str,  operands: Tuple[int, ...]) -> ExpressionNode:
    return OperatorNode(kind, operands)

def make_bool_constant_node(v: bool) -> ExpressionNode:
    return v

def make_int_constant_node(v: int) -> ExpressionNode:
    return v

def make_rational_constant_node(numerator: int, denominator:int) -> ExpressionNode:
    return Fraction(numerator=numerator, denominator=denominator)

def make_object_node(name: str) -> ExpressionNode:
    return name

def make_fluent_node(name: str) -> ExpressionNode:
    return name

def shift_expression(exp: Expression, offset: int) -> Expression:
    res = []
    for e in exp:
        if isinstance(e, OperatorNode):
            res.append(OperatorNode(e.kind, tuple([o+offset for o in e.operands])))
        else:
            res.append(e)
    return tuple(res)

def split_expression(exp: Expression) -> Tuple[Expression]:
    if not isinstance(exp[-1], OperatorNode) or not exp[-1].kind == "and":
        return (exp, )
    res = []
    last = 0
    for i in exp[-1].operands:
        new_exp = []
        for e in exp[last:i+1]:
            if isinstance(e, OperatorNode):
                new_operands = tuple([j-last for j in e.operands])
                new_exp.append(OperatorNode(e.kind, new_operands))
            else:
                new_exp.append(e)
        res.append(tuple(new_exp))
        last = i+1
    return tuple(res)

def get_fluents(exp: Expression) -> Iterator[str]:
    for e in exp:
        if isinstance(e, str):
            yield e


@dataclass(eq=True, frozen=True)
class Effect:
    fluent: str
    value: Expression


@dataclass(eq=True, frozen=True)
class Timing:
    start: bool
    delay: Fraction

    def is_from_start(self):
        return self.start

    def is_from_end(self):
        return not self.start


@dataclass(eq=True, frozen=True)
class Event:
    action: str
    pos: int
    conditions: Expression
    start_conditions: Tuple[Expression, ...]
    end_conditions: Tuple[Expression, ...]
    effects: Tuple[Effect, ...]

    def __repr__(self):
        return f"Event(action={self.action}, conditions={self.conditions}, start_conditions={self.start_conditions}, end_conditions={self.end_conditions}, effects={self.effects})"


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
    assignments: Dict[str, Union[bool, int, Fraction, str]]
    temporal_network: Optional[DeltaSimpleTemporalNetwork]
    todo: Dict[str, Tuple[int, int]]
    active_conditions: MultiSet
    g: int
    path: List[Tuple[Event, int]]
    selection: Optional[List[str]] = None
    father: Optional['State'] = None
    is_skipped: Optional[bool] = False

    def __hash__(self) -> int:
        res = 0
        for k, v in self.assignments.items():
            res += hash(k) + hash(v)
        return res

    def __eq__(self, oth) -> bool:
        if self.temporal_network is None:
            return self.assignments == oth.assignments
        else:
            return False

    def get_value(self, fluent: str) -> Union[bool, int, Fraction, str]:
        return self.assignments[fluent]

    def clone(self):
        assignments = {k : v for k, v in self.assignments.items()}
        todo = {k : v for k, v in self.todo.items()}
        tn = self.temporal_network.copy_stn() if self.temporal_network else None
        return State(assignments, tn, todo, self.active_conditions.clone(), self.g, self.path[:])

    def extract_solution(self) -> List[Tuple[Optional[Fraction], str, Optional[Fraction]]]:
        if self.temporal_network:
            start_time = {}
            end_time = {}
            for e, t in self.temporal_network.distances.items():
                if len(e) == 3:
                    if e[1]:
                        start_time[(e[0], e[2])] = -t
                    else:
                        end_time[(e[0], e[2])] = -t
            l = []
            for a, st in start_time.items():
                et = end_time[a]
                d = et-st
                l.append((st, a[0], None if d==0 else d))
            l = sorted(l, key=lambda x: x[0])
            return l
        else:
            l = []
            for e in self.path:
                l.append((None, e[0].action, None))
            return l

    def extract_used_macro(self) -> List[str]:
        #count = 0

        macro_selected = []
        fath=self.father
        sel=self.selection
        while fath is not None:
            if sel is not None:
                macro_selected.append(sel)
            fath = fath.father
            if fath is not None:
                sel = fath.selection

        return macro_selected

def evaluate(exp: Expression, state: State) -> Union[bool, int, Fraction, str]:
    res = []
    for e in exp:
        if isinstance(e, bool) or isinstance(e, int) or isinstance(e, Fraction):
            res.append(e)
        elif isinstance(e, str):
            v = state.assignments.get(e, None)
            if v is None:
                res.append(e)
            else:
                res.append(v)
        else:
            assert isinstance(e, OperatorNode)
            if e.kind == "and":
                v = True
                for i in e.operands:
                    if not res[i]:
                        v = False
                        break
                res.append(v)
            if e.kind == "not":
                res.append(not res[e.operands[0]])
            elif e.kind == "==":
                res.append(res[e.operands[0]] == res[e.operands[1]])
            elif e.kind == "<=":
                res.append(res[e.operands[0]] <= res[e.operands[1]])
            elif e.kind == "<":
                res.append(res[e.operands[0]] < res[e.operands[1]])
            elif e.kind == "+":
                v = 0
                for i in e.operands:
                    v += res[i]
                res.append(v)
            elif e.kind == "-":
                res.append(res[e.operands[0]] - res[e.operands[1]])
            elif e.kind == "*":
                v = 1
                for i in e.operands:
                    v *= res[i]
                res.append(v)
            elif e.kind == "/":
                res.append(res[e.operands[0]] / res[e.operands[1]])
    return res[-1]


def simplify(exp: Expression, assignments: Dict[str, Union[bool, int, Fraction, str]]) -> Expression:
    """This function simplify the given expression using the given assignments"""

    # We iterate over the expression elements and we store the simplified value in the res vector
    # In the to_remove vector we store the index of the elements that can be removed
    res = []
    to_remove = []
    for e in exp:
        if isinstance(e, bool) or isinstance(e, int) or isinstance(e, Fraction):
            res.append(e)
        elif isinstance(e, str):
            v = assignments.get(e, None)
            if v is None:
                res.append(e)
            else:
                res.append(v)
        else:
            assert isinstance(e, OperatorNode)
            if e.kind == "and":
                v = True
                unresolved = False
                true_to_remove = []
                for i in e.operands:
                    if isinstance(res[i], bool) and not res[i]:
                        v = False
                        break
                    elif isinstance(res[i], bool):
                        true_to_remove.append(i)
                    else:
                        unresolved = True
                if not unresolved:
                    to_remove.extend(e.operands)
                    res.append(v)
                else:
                    to_remove.extend(true_to_remove)
                    res.append(e)
            if e.kind == "not":
                v = res[e.operands[0]]
                if isinstance(v, bool):
                    to_remove.extend(e.operands)
                    res.append(not v)
                else:
                    res.append(e)
            elif e.kind == "==":
                v1 = res[e.operands[0]]
                v2 = res[e.operands[1]]
                if v1 == v2 or ((isinstance(v1, int) or isinstance(v1, Fraction)) and (isinstance(v2, int) or isinstance(v2, Fraction))):
                    to_remove.extend(e.operands)
                    res.append(v1 == v2)
                else:
                    res.append(e)
            elif e.kind in ["<=", "<", "-", "/"]:
                v1 = res[e.operands[0]]
                v2 = res[e.operands[1]]
                if (isinstance(v1, int) or isinstance(v1, Fraction)) and (isinstance(v2, int) or isinstance(v2, Fraction)):
                    to_remove.extend(e.operands)
                    if e.kind == "<=":
                        res.append(v1 <= v2)
                    elif e.kind == "<":
                        res.append(v1 < v2)
                    elif e.kind == "-":
                        res.append(v1 - v2)
                    elif e.kind == "/":
                        res.append(v1 / v2)
                else:
                    res.append(e)
            elif e.kind in ["+", "*"]:
                v = 0 if e.kind == "+" else 1
                to_simplified = True
                for i in e.operands:
                    v1 = res[i]
                    if (isinstance(v1, int) or isinstance(v1, Fraction)):
                        if e.kind == "+":
                            v += v1
                        else:
                            v *= v1
                    else:
                        to_simplified = False
                        break
                if to_simplified:
                    to_remove.extend(e.operands)
                    res.append(v)
                else:
                    res.append(e)

    # We build the simplified expression iterating over the res elements, removing
    # the ones that are not needed and updating the operands indexes
    final_res = []
    to_remove = set(to_remove)
    for i, e in enumerate(res):
        if i not in to_remove:
            if isinstance(e, OperatorNode):
                operands = []
                for o in e.operands:
                    if o not in to_remove:
                        operands.append(o - len(to_remove.intersection(range(o))))
                final_res.append(OperatorNode(e.kind, tuple(operands)))
            else:
                final_res.append(e)

    return tuple(final_res)


class SearchSpace:

    def __init__(self,
                 actions_duration: Dict[str, Optional[Tuple[Expression, Expression, bool, bool]]],
                 events: Dict[str, List[Tuple[Timing, Event]]],
                 mutex: Set[Tuple[Event, Event]],
                 initial_state: Optional[Dict[str, Union[bool, int, Fraction, str]]] = None,
                 goal: Optional[Expression] = None,
                 epsilon: Optional[Fraction] = None):
        self._actions_duration = actions_duration
        self._events = events
        self._actions = sorted(events.keys())
        self._mutex = mutex
        self._initial_state = initial_state
        self._goal = goal
        self._epsilon = Fraction(1, 100) if epsilon is None else epsilon
        self._is_temporal = False if all([v is None for v in actions_duration.values()]) else True
        self._counter = 0

    @property
    def is_temporal(self) -> bool:
        return self._is_temporal

    def reset(self):
        pass

    def initial_state(self,
                      initial_state: Optional[Dict[str, Union[bool, int, Fraction, str]]] = None) -> State:
        # initial_state parameter can be None if it was passed to the class constructor
        assert initial_state is not None or self._initial_state is not None
        tn = DeltaSimpleTemporalNetwork() if self._is_temporal else None
        if initial_state is not None:
            return State(initial_state, tn, {}, MultiSet(), 0, [])
        else:
            return State(self._initial_state, tn, {}, MultiSet(), 0, [])

    def get_successor_state(self, state: State, action: str) -> Optional[State]:
        events = self._events[action]
        new_state = state.clone()
        new_state.g = state.g + 1
        if action in state.todo:
            index, id = state.todo[action]
            _, e = events[index]
            if index+1 >= len(events):
                new_state.todo.pop(action)
            else:
                new_state.todo[action] = index+1, id+1
            new_state = self._expand_event(state, new_state, e, index, id)
        else:
            new_state = self._open_action(state, new_state, action, events)
        return new_state

    def get_successor_states(self, state: State, search_trie : Optional[PrefixTree]) -> Iterator[State]:
        for action in self._actions:
            new_state = self.get_successor_state(state, action)
            if new_state:
                yield new_state

    def goal_reached(self, state: State, goal: Optional[Fraction] = None) -> bool:
        # goal parameter can be None if it was passed to the class constructor
        assert goal is not None or self._goal is not None
        if len(state.todo) > 0:
            return False
        if goal is not None:
            return evaluate(goal, state)
        else:
            return evaluate(self._goal, state)

    def subgoals_sat(self, state: State, goal: Optional[Fraction] = None) -> Set[Expression]:
        # goal parameter can be None if it was passed to the class constructor
        assert goal is not None or self._goal is not None
        if goal is not None:
            goals = split_expression(goal)
        else:
            goals = split_expression(self._goal)
        res = set()
        for g in goals:
            if evaluate(g, state):
                res.add(g)
        return res

    def _expand_event(self, state, new_state, e, index, id):
        new_state.path.append((e, id))
        # check conditions
        if not evaluate(e.conditions, state):
            return None
        # remove end conditions
        for c in e.end_conditions:
            new_state.active_conditions.remove(c)
        # check active conditions
        for c in new_state.active_conditions:
            if not evaluate(c, state):
                return None
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
            e_id = (e.action, index)
            if len(state.path) > 0:
                for e2, id2 in state.path:
                    e2_id = (e2.action, e2.pos)
                    if (e_id, e2_id) in self._mutex:
                        new_state.temporal_network.add((e2, id2), (e, id), -self._epsilon)
                    else:
                        new_state.temporal_network.add((e2, id2), (e, id), 0)
            for a, i in new_state.todo.items():
                id2 = i[1]
                for j, (_, e2) in enumerate(self._events[a][i[0]:]):
                    e2_id = (a, i[0]+j)
                    if (e_id, e2_id) in self._mutex:
                        new_state.temporal_network.add((e, id), (e2, id2), -self._epsilon)
                    else:
                        new_state.temporal_network.add((e, id), (e2, id2), 0)
                    id2 += 1
            # check TN
            if not new_state.temporal_network.check_stn():
                return None
        return new_state

    def _open_action(self, state, new_state, action, events):
        if self._is_temporal:
            start = (action, True, self._counter)
            end = (action, False, self._counter)
            self._counter += 1
            duration = self._actions_duration[action]
            if duration is None:
                l, u = 0, 0
            else:
                l = evaluate(duration[0], state)
                if duration[2]:
                    l += self._epsilon
                u = evaluate(duration[1], state)
                if duration[3]:
                    u -= self._epsilon
            new_state.temporal_network.insert_interval(start, end, left_bound=l, right_bound=u)
            id = self._counter
            for t, e in events:
                ev = (e, self._counter)
                if t.is_from_start():
                    new_state.temporal_network.insert_interval(start, ev, left_bound=t.delay, right_bound=t.delay)
                else:
                    new_state.temporal_network.insert_interval(end, ev, left_bound=t.delay, right_bound=t.delay)
                self._counter += 1
            if len(events) > 1:
                new_state.todo[action] = 1, id+1
        else:
            id = self._counter
        return self._expand_event(state, new_state, events[0][1], 0, id)


class SearchSpaceMacroAction:

    CACHE_MISS = -1

    def __init__(self, ss : SearchSpace, macros: Optional[List[str]], macros_usage: Optional[str]):
        self._ss = ss
        self._macros = macros
        self._macros_usage = macros_usage
        assert self._macros_usage is not None
        self._fa = "FA" in self._macros_usage
        self._minus = "-" in self._macros_usage
        self._cache_equal_path = True

    @property
    def is_temporal(self) -> bool:
        return self._ss.is_temporal

    def reset(self):
        pass

    def initial_state(self,
                      initial_state: Optional[Dict[str, Union[bool, int, Fraction, str]]] = None) -> State:
        return self._ss.initial_state(initial_state)

    def get_successor_state(self, state: State, action: str) -> Optional[State]:
        if action in list(self._ss._events.keys()):
            return self._ss.get_successor_state(state, action)
        else:
            return None


    def get_successor_states(self, state: State, search_trie : Optional[PrefixTree] = None) -> Iterator[State]:
        macro_cache = {}
        if search_trie is not None:
            path = tuple(ev.action for (ev, _) in state.path)
        for action in self._ss._actions:
            if self._cache_equal_path and not search_trie.insert(path + (action,)):
                search_trie.counter_skip += 1
                continue
            new_state = self.get_successor_state(state, action)
            macro_cache[(action,)] = new_state
            if new_state:
                new_state.father = state
                yield new_state
        if self._macros:
            for ma in self._macros:
                new_states = []
                s = state
                macro_so_far = []
                for i, a in enumerate(ma):
                    macro_so_far.append(a)
                    key = tuple(macro_so_far)
                    new_state = macro_cache.get(key, self.CACHE_MISS)
                    if new_state == self.CACHE_MISS:
                        new_state = self.get_successor_state(s, a)
                        macro_cache[key] = new_state
                    s = new_state
                    new_states.append(new_state)
                    if not new_state: # applicablity
                        break
                if self._fa: #fully applicable
                    if self._minus: #without
                        if new_state:
                            new_state.selection = ma
                            new_state.father = state
                            if not self._cache_equal_path or search_trie.insert(path + tuple(ma)):
                                yield new_state
                            else:
                                search_trie.counter_skip += 1
                                # new_state.is_skipped = True
                    else: #with
                        if new_state:
                            for i, ns in enumerate(new_states[1:]):
                                assert ns is not None
                                if not self._cache_equal_path or search_trie.insert(path + tuple(a for a in ma[:i+2])):
                                    ns.selection = ma
                                    ns.father = state
                                    yield ns
                                else:
                                    search_trie.counter_skip += 1
                                    # ns.is_skipped = True
                else: #partial applicable
                    if self._minus: #without
                        if len(new_states) > 1:
                            len_ma = len(ma)
                            if not new_state:
                                assert new_states[-2] is not None
                                new_state = new_states[-2]
                                len_ma = len(new_states[:-1])
                            if not self._cache_equal_path or search_trie.insert(path + tuple(a for a in ma[:len_ma])):
                                new_state.selection = ma
                                new_state.father = state
                                yield new_state
                            else:
                                search_trie.counter_skip += 1
                                # new_state.is_skipped = True
                    else:
                        if len(new_states) > 1: #with
                            if not new_state: 
                                new_states = new_states[:-1]
                            for i, ns in enumerate(new_states[1:]):
                                assert ns is not None
                                if not self._cache_equal_path or search_trie.insert(path + tuple(a for a in ma[:i+2])):
                                    ns.selection = ma
                                    ns.father = state
                                    yield ns
                                else:
                                    search_trie.counter_skip += 1
                                    # ns.is_skipped = True

    def goal_reached(self, state: State, goal: Optional[Fraction] = None) -> bool:
        return self._ss.goal_reached(state, goal)


    def subgoals_sat(self, state: State, goal: Optional[Fraction] = None) -> Set[Expression]:
        return self._ss.subgoals_sat(state, goal)