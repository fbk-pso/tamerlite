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

import unified_planning as up
import unified_planning.model
from unified_planning.model.fluent import get_all_fluent_exp
from unified_planning.plans import TimeTriggeredPlan, SequentialPlan, Plan
from unified_planning.engines.compilers import Grounder
from fractions import Fraction
from typing import List, Tuple, Dict, Union, Optional

from tamerlite.core import Expression, Effect, Timing, Event, SearchSpace, get_fluents
from tamerlite.state_encoder import StateEncoder, GeneralStateEncoder
from tamerlite.converter import Converter


def get_encoders(domain, problem=None):
    """
    Returns the problem encoder, the state encoder and the callable to map back
    the ground actions to their lifted version.
    If both the domain and the problem are specified, the problem is updated
    adding all the domain's objects "disabling" the ones not already present in
    the problem, using an extra fluent.
    If only the domain is specified, returns the general state encoder instead of
    the state encoder.
    """
    full = problem is not None

    if problem is None:
        problem = domain.clone()
    else:
        problem = problem.clone()

    is_active_fluents = {}
    for ut in problem.user_types:
        f = up.model.Fluent(f"_is_active_{ut.name}", p=ut)
        is_active_fluents[ut] = f
        problem.add_fluent(f, default_initial_value=False)
        if full:
            for obj in problem.objects(ut):
                problem.set_initial_value(f(obj), True)
    if full:
        for obj in domain.all_objects:
            if obj not in problem.all_objects:
                problem.add_object(obj)
        domain_initial_values = domain.initial_values
        initial_values = problem.initial_values
        for f in problem.fluents:
            for f_exp in get_all_fluent_exp(problem, f):
                if f_exp not in initial_values and f_exp in domain_initial_values:
                    problem.set_initial_value(f_exp, domain_initial_values[f_exp])

    em = problem.environment.expression_manager
    for f in problem.fluents:
        if f in problem.fluents_defaults:
            pass
        elif f.type.is_bool_type():
            problem.fluents_defaults[f] = em.FALSE()
        elif f.type.is_int_type():
            problem.fluents_defaults[f] = em.Int(0)
        elif f.type.is_real_type():
            problem.fluents_defaults[f] = em.Real(Fraction(0))
        elif f.type.is_user_type():
            problem.fluents_defaults[f] = em.ObjectExp(next(problem.objects(f.type)))
        else:
            raise NotImplementedError()

    for a in problem.actions:
        for p in a.parameters:
            if p.type.is_user_type():
                f = is_active_fluents[p.type]
                if isinstance(a, up.model.InstantaneousAction):
                    a.add_precondition(f(p))
                else:
                    a.add_condition(up.model.StartTiming(), f(p))

    grounder = Grounder(prune_actions=False)
    grounding_result = grounder.compile(problem)

    ground_problem = grounding_result.problem

    encoder = Encoder(ground_problem, full)
    gen_state_encoder = GeneralStateEncoder(problem, grounding_result, encoder.events, encoder.search_space)
    if full:
        state_encoder = StateEncoder(problem.environment, gen_state_encoder, problem.initial_values, problem.goals)
    else:
        state_encoder = gen_state_encoder

    return encoder, state_encoder, grounding_result.map_back_action_instance


class Encoder:
    """
    This class takes in input a Problem and builds its search space.
    If full is True, the initial and goal states are already initialized
    in the search space.
    """

    def __init__(self, problem: "up.model.Problem", full: bool = True):
        self._problem = problem
        if full:
            self._simplifier = up.model.walkers.Simplifier(problem.environment, problem)
        else:
            self._simplifier = problem.environment.simplifier
        self._converter = Converter(problem)
        actions_duration = {}
        self._is_temporal = False
        for a in problem.actions:
            if isinstance(a, up.model.DurativeAction):
                self._is_temporal = True
                lb = self._convert_expression(a.duration.lower)
                ub = self._convert_expression(a.duration.upper)
                actions_duration[a.name] = (lb, ub, a.duration.is_left_open(), a.duration.is_right_open())
            else:
                actions_duration[a.name] = None
        self._build_events()
        self._build_mutex()
        if full:
            initial_state = {}
            for f, v in problem.initial_values.items():
                initial_state[self._convert_fluent(f)] = self._convert_expression(v)[0]
            self._goal = self._convert_expression(problem.environment.expression_manager.And(problem.goals))
        else:
            initial_state = None
            self._goal = None
        self._search_space = SearchSpace(actions_duration, self._events, self._mutex,
                                         initial_state, self._goal, problem.epsilon)
        self._objects = {}
        for ut in problem.user_types:
            self._objects[ut.name] = [o.name for o in problem.objects(ut)]
        self._fluents = {}
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
            self._fluents[self._convert_fluent(f)] = t

    @property
    def problem(self):
        return self._problem

    def initial_state(self, initial_values: Dict["up.model.FNode", "up.model.FNode"]):
        initial_state = {}
        for f, v in initial_values.items():
            initial_state[self._convert_fluent(f)] = self._convert_expression(v)[0]
        return initial_state

    def goals(self, goals: List["up.model.FNode"]):
        return self._convert_expression(self._problem.environment.expression_manager.And(goals))

    @property
    def search_space(self) -> SearchSpace:
        return self._search_space

    @property
    def fluents(self) -> Dict[str, str]:
        return self._fluents

    @property
    def objects(self) -> Dict[str, List[str]]:
        return self._objects

    @property
    def events(self) -> Dict[str, List[Tuple[Timing, Event]]]:
        return self._events

    @property
    def applicable_actions(self) -> List[str]:
        return self._applicable_actions

    @property
    def goal(self) -> Expression:
        return self._goal

    def build_plan(self, plan: List[Tuple[Optional[Fraction], str, Optional[Fraction]]]) -> Plan:
        if self._is_temporal:
            return TimeTriggeredPlan([(Fraction(s), self._problem.action(a)(), Fraction(d) if d else None) for s, a, d in plan])
        else:
            return SequentialPlan([self._problem.action(a)() for _, a, _ in plan])

    def _convert_fluent(self, fluent_exp: "up.model.FNode") -> str:
        return str(fluent_exp)

    def _convert_expression(self, expression: "up.model.FNode") -> Expression:
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
        f = self._convert_fluent(effect.fluent)
        v = self._convert_expression(value)
        return Effect(f, v)

    def _build_events(self):
        env = self._problem.environment
        em = env.expression_manager
        self._events = {}
        self._applicable_actions = []
        for a in self._problem.actions:
            if isinstance(a, up.model.DurativeAction):
                from_start = {}
                from_end = {}
                action_events = []
                is_applicable = True
                for i, lc in a.conditions.items():
                    l = i.lower
                    u = i.upper
                    if l == u: # conditions
                        action_events.append((l.delay, l, 1, lc))
                    else:
                        # lower: start conditions
                        if not i.is_left_open():
                            action_events.append((l.delay, l, 1, lc))
                        action_events.append((l.delay, l, 2, [em.And(lc)]))
                        # upper: end conditions
                        action_events.append((u.delay, u, 1, lc))
                        action_events.append((u.delay, u, 3, [em.And(lc)]))
                    is_applicable = is_applicable and not self._simplifier.simplify(em.And(lc)).is_false()
                if is_applicable:
                    self._applicable_actions.append(a.name)

                for t, le in a.effects.items():
                    action_events.append((t.delay, t, 4, le))

                for d, t, p, e in action_events:
                    if t.is_from_start():
                        from_start.setdefault(d, (t, [], [], [], []))
                        from_start[d][p].extend(e)
                    else:
                        from_end.setdefault(d, (t, [], [], [], []))
                        from_end[d][p].extend(e)

                self._events[a.name] = []
                has_ice = False
                pos = 0
                for d in sorted(from_start):
                    if d > 0:
                        has_ice = True
                    t, lc, lsc, lec, le = from_start[d]
                    t = self._convert_timing(t)
                    c = self._convert_expression(em.And(lc))
                    tsc = tuple([self._convert_expression(sc) for sc in lsc])
                    tec = tuple([self._convert_expression(ec) for ec in lec])
                    te = tuple([self._convert_effect(e) for e in le])
                    self._events[a.name].append((t, Event(a.name, pos, c, tsc, tec, te)))
                    pos += 1
                for d in sorted(from_end):
                    if d < 0 and has_ice:
                        raise Exception("TamerLite does not support ICE from start and from end inside the same action!")
                    t, lc, lsc, lec, le = from_end[d]
                    t = self._convert_timing(t)
                    c = self._convert_expression(em.And(lc))
                    tsc = tuple([self._convert_expression(sc) for sc in lsc])
                    tec = tuple([self._convert_expression(ec) for ec in lec])
                    te = tuple([self._convert_effect(e) for e in le])
                    self._events[a.name].append((t, Event(a.name, pos, c, tsc, tec, te)))
                    pos += 1
            else:
                t = Timing(True, Fraction(0))
                te = tuple([self._convert_effect(e) for e in a.effects])
                self._events[a.name] = [(t, Event(a.name, 0, self._convert_expression(em.And(a.preconditions)), tuple(), tuple(), te))]
                if not self._simplifier.simplify(em.And(a.preconditions)).is_false():
                    self._applicable_actions.append(a.name)

    def _build_mutex(self):
        self._mutex = set()
        ev = {}
        ev_list = []
        for (a, le) in self._events.items():
            for i, (_, e1) in enumerate(le):
                a_p = set(get_fluents(e1.conditions))
                a_p.update(x for e in e1.effects for x in get_fluents(e.value))
                a_e = set(e.fluent for e in e1.effects)
                ev[(a, i)] = (a_p, a_e)
                ev_list.append((a, i))
        for a1, i1 in ev_list:
            for a2, i2 in ev_list:
                if a1 == a2:
                    # Since we do not allow self-overlapping, events of the same action are always mutex
                    self._mutex.add(((a1, i1), (a2, i2)))
                else:
                    (a_p, a_e) = ev[(a1, i1)]
                    (b_p, b_e) = ev[(a2, i2)]
                    if not a_p.isdisjoint(b_e) or not b_p.isdisjoint(a_e) or not a_e.isdisjoint(b_e):
                        self._mutex.add(((a1, i1), (a2, i2)))
