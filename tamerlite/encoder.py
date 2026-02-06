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

import unified_planning as up
from unified_planning.plans import TimeTriggeredPlan, SequentialPlan, Plan
from unified_planning.model import Problem, FNode
from fractions import Fraction
from typing import List, Tuple, Dict, Optional, Union, Any

from tamerlite.core import (
    Expression,
    Effect,
    Timing,
    Event,
    Action,
    SearchSpace,
    get_fluents,
    contains_operator,
    make_bool_constant_node,
)
from tamerlite.core.search_space import SearchSpaceABC
from tamerlite.converter import Converter
from tamerlite.simultaneity_utils import (
    get_all_simultaneity_actions_groups,
    get_simultaneity_actions_groups,
)


PlanType = List[
    Tuple[Optional[Union[Fraction, str]], Action, Optional[Union[Fraction, str]]]
]


class Encoder:
    """
    This class takes in input a Problem and builds its search space.
    If full is True, the initial and goal states are already initialized
    in the search space.
    """

    def __init__(self, problem: Problem, full: bool = True, simultaneity: str = "NO"):
        self._problem = problem
        self._simultaneity = simultaneity
        if full:
            self._simplifier = up.model.walkers.Simplifier(problem.environment, problem)
        else:
            self._simplifier = problem.environment.simplifier

        fluent_types = {}
        for f in problem.initial_values.keys():
            if f.type.is_bool_type():
                t = "bool"
            elif f.type.is_int_type():
                t = "int"
            elif f.type.is_real_type():
                t = "real"
            elif f.type.is_user_type():
                t = f.type.name
            else:
                raise NotImplementedError
            fluent_types[self._convert_fluent(f)] = t
        self._fluents: List[str] = sorted(fluent_types.keys())
        self._fluent_ids = dict((f, i) for i, f in enumerate(self._fluents))
        self._fluent_types = [fluent_types[f] for f in self._fluents]

        self._converter = Converter(problem, self._fluent_ids)
        self._action_names: List[str] = sorted(
            action.name for action in problem.actions
        )
        self._action_by_name: Dict[str, Action] = {
            name: Action(index) for index, name in enumerate(self._action_names)
        }
        self._actions: List[Action] = [
            self._action_by_name[name] for name in self._action_names
        ]
        actions_duration_map: Dict[
            str, Optional[Tuple[Expression, Expression, bool, bool]]
        ] = {}
        self._is_temporal = False
        for a in problem.actions:
            if isinstance(a, up.model.DurativeAction):
                self._is_temporal = True
                lb = self._convert_expression(a.duration.lower)
                ub = self._convert_expression(a.duration.upper)
                actions_duration_map[a.name] = (
                    lb,
                    ub,
                    a.duration.is_left_open(),
                    a.duration.is_right_open(),
                )
            else:
                actions_duration_map[a.name] = None
        actions_duration = [actions_duration_map[a] for a in self._action_names]
        self._build_events()
        self._build_mutex()

        initial_state = None
        self._goal = None
        if full:
            initial_state = self.initial_state(problem.initial_values)
            self._goal = self._convert_expression(
                problem.environment.expression_manager.And(problem.goals)
            )
        self._search_space = SearchSpace(
            actions_duration,
            self._events,
            self._actions,
            self._mutex,
            self._precedence,
            self._simultaneity_groups,
            initial_state,  # type: ignore[arg-type]
            self._goal,
            problem.epsilon,
        )
        self._objects = {}
        for ut in problem.user_types:
            self._objects[ut.name] = [o.name for o in problem.objects(ut)]

    @property
    def problem(self) -> Problem:
        return self._problem

    def initial_state(self, initial_values: Dict[FNode, FNode]) -> Expression:
        initial_state_values = {}
        for f, v in initial_values.items():
            initial_state_values[self._convert_fluent(f)] = self._convert_expression(v)[
                0
            ]

        initial_state = []
        for f in self._fluents:
            initial_state.append(initial_state_values[f])
        return initial_state  # type: ignore[return-value]

    def goals(self, goals: List[FNode]) -> Expression:
        return self._convert_expression(
            self._problem.environment.expression_manager.And(goals)
        )

    @property
    def search_space(self) -> SearchSpaceABC:
        return self._search_space

    @property
    def fluents(self) -> List[str]:
        return self._fluents

    @property
    def fluent_ids(self) -> Dict[str, int]:
        return self._fluent_ids

    @property
    def fluent_types(self) -> List[str]:
        return self._fluent_types

    @property
    def objects(self) -> Dict[str, List[str]]:
        return self._objects

    @property
    def events(self) -> Dict[Action, List[Tuple[Timing, Event]]]:
        return self._events

    @property
    def actions(self) -> List[Action]:
        return self._actions

    @property
    def action_names(self) -> List[str]:
        return self._action_names

    @property
    def action_by_name(self) -> Dict[str, Action]:
        return self._action_by_name

    @property
    def applicable_actions(self) -> List[Action]:
        return self._applicable_actions

    @property
    def goal(self) -> Optional[Expression]:
        return self._goal

    def get_action(self, name: str) -> Action:
        return self.action_by_name[name]

    def get_action_name(self, action: Action) -> str:
        return self.action_names[action.idx]

    def build_plan(self, plan: PlanType) -> Plan:
        if self._is_temporal:
            assert all(map(lambda e: e[0] is not None, plan))
            return TimeTriggeredPlan(
                [
                    (
                        Fraction(s),  # type: ignore[arg-type]
                        self._problem.action(self.get_action_name(a))(),
                        Fraction(d) if d is not None else None,
                    )
                    for s, a, d in plan
                ]
            )
        else:
            return SequentialPlan(
                [self._problem.action(self.get_action_name(a))() for _, a, _ in plan]
            )

    def _convert_fluent(self, fluent_exp: FNode) -> str:
        return str(fluent_exp)

    def _convert_expression(self, expression: FNode) -> Expression:
        expression = self._simplifier.simplify(expression)
        return self._converter.convert(expression)

    def _convert_timing(self, timing: "up.model.Timing") -> Timing:
        return Timing(timing.is_from_start(), timing.delay)

    def _convert_effect(self, effect: "up.model.Effect") -> Effect:
        env = self._problem.environment
        em = env.expression_manager
        if effect.is_increase():
            value = em.Plus(effect.fluent, effect.value)
        elif effect.is_decrease():
            value = em.Minus(effect.fluent, effect.value)
        else:
            value = effect.value

        f = self.fluent_ids[self._convert_fluent(effect.fluent)]
        v = self._convert_expression(value)
        return Effect(f, v)

    def _build_events(self):
        env = self._problem.environment
        em = env.expression_manager
        self._events: Dict[Action, List[Tuple[Timing, Event]]] = {}
        self._applicable_actions = []
        for a in self._problem.actions:
            if isinstance(a, up.model.DurativeAction):
                from_start: Dict[Any, Any] = {}
                from_end: Dict[Any, Any] = {}
                action_events = []
                is_applicable = True
                for i, lc in a.conditions.items():
                    l = i.lower
                    u = i.upper
                    if l == u:  # conditions
                        action_events.append((l.delay, l, 1, lc))
                    else:
                        # lower: start conditions
                        if not i.is_left_open():
                            action_events.append((l.delay, l, 1, lc))
                        action_events.append((l.delay, l, 2, [em.And(lc)]))
                        # upper: end conditions
                        if not i.is_right_open():
                            action_events.append((u.delay, u, 1, lc))
                        action_events.append((u.delay, u, 3, [em.And(lc)]))
                    is_applicable = (
                        is_applicable
                        and not self._simplifier.simplify(em.And(lc)).is_false()
                    )
                if is_applicable:
                    self._applicable_actions.append(self.get_action(a.name))

                for t, le in a.effects.items():
                    action_events.append((t.delay, t, 4, le))

                has_ice_from_start = False
                has_ice_from_end = False
                for d, t, p, e in action_events:
                    if t.is_from_start():
                        from_start.setdefault(d, (t, [], [], [], []))
                        from_start[d][p].extend(e)
                        if d > 0:
                            has_ice_from_start = True
                    else:
                        from_end.setdefault(d, (t, [], [], [], []))
                        from_end[d][p].extend(e)
                        if d < 0:
                            has_ice_from_end = True

                if has_ice_from_start and has_ice_from_end:
                    lower, upper = a.duration.lower, a.duration.upper
                    if (
                        lower.is_constant()
                        and upper.is_constant()
                        and lower.constant_value() == upper.constant_value()
                    ):
                        duration = lower.constant_value()
                        for d in from_end:
                            t, lc, lsc, lec, le = from_end[d]
                            d_from_start = duration + d
                            from_start.setdefault(d_from_start, (t, [], [], [], []))
                            from_start[d_from_start][1].extend(lc)
                            from_start[d_from_start][2].extend(lsc)
                            from_start[d_from_start][3].extend(lec)
                            from_start[d_from_start][4].extend(le)
                        from_end.clear()
                    else:
                        raise Exception(
                            "TamerLite does not support ICE from start and from end inside the same action!"
                        )

                self._events[self.get_action(a.name)] = []
                pos = 0
                for d in sorted(from_start):
                    t, lc, lsc, lec, le = from_start[d]
                    t = self._convert_timing(t)
                    c = self._convert_expression(em.And(lc))
                    tsc = tuple([self._convert_expression(sc) for sc in lsc])
                    tec = tuple([self._convert_expression(ec) for ec in lec])
                    te = tuple([self._convert_effect(e) for e in le])
                    self._events[self.get_action(a.name)].append(
                        (t, Event(self.get_action(a.name), pos, c, tsc, tec, te))
                    )
                    pos += 1
                for d in sorted(from_end):
                    t, lc, lsc, lec, le = from_end[d]
                    t = self._convert_timing(t)
                    c = self._convert_expression(em.And(lc))
                    tsc = tuple([self._convert_expression(sc) for sc in lsc])
                    tec = tuple([self._convert_expression(ec) for ec in lec])
                    te = tuple([self._convert_effect(e) for e in le])
                    self._events[self.get_action(a.name)].append(
                        (t, Event(self.get_action(a.name), pos, c, tsc, tec, te))
                    )
                    pos += 1
            else:
                t = Timing(True, Fraction(0))
                te = tuple([self._convert_effect(e) for e in a.effects])
                self._events[self.get_action(a.name)] = [
                    (
                        t,
                        Event(
                            self.get_action(a.name),
                            0,
                            self._convert_expression(em.And(a.preconditions)),
                            tuple(),
                            tuple(),
                            te,
                        ),
                    )
                ]
                if not self._simplifier.simplify(em.And(a.preconditions)).is_false():
                    self._applicable_actions.append(self.get_action(a.name))

    def _build_mutex(self):
        self._mutex = set()
        self._precedence = set()
        ev = {}
        ev_list = []
        durative_conds = []
        f_to_actions = {}
        for a, le in self._events.items():
            for i, (_, e1) in enumerate(le):
                a_p = set(get_fluents(e1.conditions))
                a_p.update(x for e in e1.effects for x in get_fluents(e.value))
                a_e = set(e.fluent for e in e1.effects)
                a_e_pos = set(
                    e.fluent
                    for e in e1.effects
                    if e.value == (make_bool_constant_node(True),)
                )
                a_e_neg = set(
                    e.fluent
                    for e in e1.effects
                    if e.value == (make_bool_constant_node(False),)
                )
                for f in a_e:
                    f_to_actions.setdefault(f, []).append((a, i))
                a_sc = {f for c in e1.start_conditions for f in get_fluents(c)}
                a_ec = {f for c in e1.end_conditions for f in get_fluents(c)}
                if len(a_sc) > 0 and any(
                    contains_operator("or", c) for c in e1.start_conditions
                ):
                    durative_conds.append((a, a_sc))
                sdc = not any(
                    contains_operator("or", c) or contains_operator("not", c)
                    for c in e1.start_conditions + e1.end_conditions
                )
                ev[(a, i)] = (a_p, a_e, a_sc, a_ec, a_e_pos, a_e_neg, sdc)
                ev_list.append((a, i))
        sim_set = set()
        for a, fs in durative_conds:
            s = {(b, i) for f in fs for (b, i) in f_to_actions.get(f, []) if a != b}
            if len(s) >= 2:
                sim_set.add(frozenset(s))
        sim_arcs = set()
        for a1, i1 in ev_list:
            for a2, i2 in ev_list:
                if a1 == a2:
                    # Since we do not allow self-overlapping, events of the same action are always mutex
                    self._mutex.add(((a1, i1), (a2, i2)))
                else:
                    (a_p, a_e, _, a_ec, a_e_pos, _, a_sdc) = ev[(a1, i1)]
                    (b_p, b_e, b_sc, _, _, b_e_neg, b_sdc) = ev[(a2, i2)]
                    if (
                        not a_p.isdisjoint(b_e)
                        or not b_p.isdisjoint(a_e)
                        or not a_e.isdisjoint(b_e)
                    ):
                        self._mutex.add(((a1, i1), (a2, i2)))
                    if not a_e.isdisjoint(b_sc):
                        self._precedence.add(((a1, i1), (a2, i2)))
                        if not b_sdc or not a_e_pos.isdisjoint(b_sc):
                            sim_arcs.add(((a1, i1), (a2, i2)))
                    if not b_e.isdisjoint(a_ec):
                        self._precedence.add(((a1, i1), (a2, i2)))
                        if not a_sdc or not b_e_neg.isdisjoint(a_ec):
                            sim_arcs.add(((a1, i1), (a2, i2)))
        if self._simultaneity == "NO":
            self._simultaneity_groups = []
        elif self._simultaneity == "ALL":
            self._simultaneity_groups = get_all_simultaneity_actions_groups(
                ev_list, self._mutex
            )
        else:
            self._simultaneity_groups = get_simultaneity_actions_groups(
                ev_list, self._mutex, sim_arcs, sim_set
            )
