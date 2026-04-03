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
from unified_planning.plans import (
    TimeTriggeredPlan,
    SequentialPlan,
    Plan,
    ActionInstance,
)
from unified_planning.model import Problem, FNode, Object, Type, Fluent, TimepointKind
from fractions import Fraction
from typing import List, Tuple, Dict, Optional, Union, Any, Set, Callable, Iterable

from tamerlite.core import Expression, Effect, Timing, Event, Action, SearchSpace
from tamerlite.core.search_space import SearchSpaceABC
from tamerlite.core import HMax, get_fluents
from tamerlite.converter import Converter


def extract_objects(exp: FNode) -> Iterable[Object]:
    stack: List[FNode] = [exp]
    while len(stack) > 0:
        exp = stack.pop()
        if exp.is_object_exp():
            yield exp.object()
        else:
            stack.extend(exp.args)


def extract_fluents(exp: FNode) -> Iterable[Fluent]:
    stack: List[FNode] = [exp]
    while len(stack) > 0:
        exp = stack.pop()
        if exp.is_fluent_exp():
            yield exp.fluent()
        else:
            stack.extend(exp.args)


def extract_and_arguments(expressions: List[FNode]) -> Iterable[FNode]:
    stack: List[FNode] = list(expressions)
    while len(stack) > 0:
        exp = stack.pop()
        if exp.is_and():
            stack.extend(exp.args)
        else:
            yield exp


PlanType = List[
    Tuple[Optional[Union[Fraction, str]], Action, Optional[Union[Fraction, str]]]
]


class Encoder:
    """
    This class takes in input a Problem and builds its search space.
    If full is True, the initial and goal states are already initialized
    in the search space.
    """

    def __init__(
        self,
        problem: Problem,
        lifted_problem: Problem,
        map_back_action_instance: Callable[[ActionInstance], Optional[ActionInstance]],
        symmetry_breaking: bool,
        compression_safe_actions: bool,
        relevance_analysis: bool,
        full: bool = True,
    ):
        self._problem = problem
        self._lifted_problem = lifted_problem
        self._map_back_action_instance = map_back_action_instance
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

        initial_state = None
        self._goal = None
        action_objects = None
        obj_to_prev_actions_map = None
        self._compression_safe_actions = None
        if full:
            initial_state = self.initial_state(problem.initial_values)
            self._goal = self.goals(problem.goals)

            if symmetry_breaking:
                action_objects, obj_to_prev_actions_map = (
                    self._compute_obj_to_prev_actions_map()
                )
                if len(obj_to_prev_actions_map) == 0:
                    # Symmetry breaking is not beneficial because there are no equivalent objects
                    action_objects = None
                    obj_to_prev_actions_map = None

            if compression_safe_actions:
                self._compression_safe_actions = (
                    self._compute_compression_safe_actions()
                )
                if not any(self._compression_safe_actions):
                    # No actions are safe for compression
                    self._compression_safe_actions = None

        self._search_space = SearchSpace(
            actions_duration,
            self._events,
            self._actions,
            self._compression_safe_actions,
            action_objects,
            obj_to_prev_actions_map,
            initial_state,  # type: ignore[arg-type]
            self._goal,
            self._applicable_actions,
            problem.epsilon,
        )
        self._objects = {}
        for ut in problem.user_types:
            self._objects[ut.name] = [o.name for o in problem.objects(ut)]

        self._relevant_actions = None
        if full and relevance_analysis:
            self._relevant_actions = self._compute_relevant_actions()
            if len(self._relevant_actions) < len(self.applicable_actions):
                self._search_space.relevant_actions = self._relevant_actions

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

    def _compute_relevant_actions(self) -> List[Action]:
        events = {a: e for a, e in self.events.items() if a in self.applicable_actions}
        heuristic = HMax(
            self.actions,
            self.fluent_types,
            self.objects,
            events,
            self.goal,  # type: ignore[arg-type]
            internal_caching=False,
            cache_value_in_state=False,
        )
        reachable_actions = {
            a.idx
            for a in heuristic.reachable_actions(self._search_space.initial_state())
        }

        actions_affecting_fluent: Dict[int, Set[int]] = {}
        action_to_condition_fluents: Dict[int, Set[int]] = {}
        for a, le in events.items():
            if a.idx not in reachable_actions:
                continue

            action_to_condition_fluents[a.idx] = set()
            for _, e in le:
                for eff in e.effects:
                    if eff.fluent not in actions_affecting_fluent:
                        actions_affecting_fluent[eff.fluent] = {a.idx}
                    else:
                        actions_affecting_fluent[eff.fluent].add(a.idx)

                for cond in list(e.end_conditions) + [e.conditions]:
                    action_to_condition_fluents[a.idx].update(get_fluents(cond))

        checked_fluents = [False] * len(self._fluents)
        stack = list(get_fluents(self.goal))  # type: ignore[arg-type]
        for f in stack:
            checked_fluents[f] = True

        relevant_actions: Set[int] = set()
        while len(stack) > 0 and len(relevant_actions) < len(
            action_to_condition_fluents
        ):
            f = stack.pop()
            relevant_actions.update(actions_affecting_fluent.get(f, set()))
            for action_idx in actions_affecting_fluent.get(f, set()):
                for f in action_to_condition_fluents[action_idx]:
                    if not checked_fluents[f]:
                        checked_fluents[f] = True
                        stack.append(f)

        return [a for a in self._actions if a.idx in relevant_actions]

    def _compute_obj_to_prev_actions_map(
        self,
    ) -> Tuple[List[List[str]], Dict[str, Set[Action]]]:
        """
        This method produces two outputs:
            1. A list of lists of object names, where each inner list corresponds
                to the objects used as parameters for the action.
            2. A dictionary mapping each object name to the set of actions that
                include the previous equivalent object as a parameter.

        Returns:
            Tuple[List[List[str]], Dict[str, Set[Action]]]:
                - List of object lists for each action.
                - Mapping from object names to the set of actions.
        """

        equivalent_objects = self._compute_equivalent_objects()
        prev_equivalent_object = {}
        for group in equivalent_objects:
            for i, obj in enumerate(group):
                prev_equivalent_object[obj] = None if i == 0 else group[i - 1]

        obj_to_actions_map: Dict[Object, Set[Action]] = {}
        action_objects: List[List[str]] = [[]] * len(self.actions)
        for action in self._problem.actions:
            ai = self._map_back_action_instance(action())
            assert ai is not None
            objects = [p.object() for p in ai.actual_parameters if p.is_object_exp()]
            action_objects[self.action_by_name[action.name].idx] = [
                obj.name for obj in objects
            ]
            for obj in objects:
                if obj not in obj_to_actions_map:
                    obj_to_actions_map[obj] = set()
                obj_to_actions_map[obj].add(self._action_by_name[action.name])

        obj_to_prev_actions_map = {}
        for obj, prev_obj in prev_equivalent_object.items():
            if prev_obj is not None and prev_obj in obj_to_actions_map:
                obj_to_prev_actions_map[obj.name] = obj_to_actions_map[prev_obj]

        return action_objects, obj_to_prev_actions_map

    def _compute_equivalent_objects(self) -> List[List[Object]]:
        """
        Compute groups of equivalent objects in the problem.

        Returns:
            List[List[Object]]: A list of equivalence classes, where each inner
            list contains objects that are equivalent to each other.
        """

        domain_objects = self._extract_domain_objects()
        goal_obj_to_fluent_map, goal_exp_is_conjunction = (
            self._extract_goal_obj_to_fluent_map()
        )

        objects: Dict[Type, List[Object]] = {}
        for obj in self._problem.all_objects:
            if obj.type not in objects:
                objects[obj.type] = []
            objects[obj.type].append(obj)

        groups = []
        for type, objs in objects.items():
            grouped = [False] * len(objs)
            for i, obj1 in enumerate(objs):
                if grouped[i]:
                    continue

                grouped[i] = True
                groups.append([obj1])

                if obj1 in domain_objects:
                    # treat all domain objects as non-equivalent objects
                    continue

                for j in range(i + 1, len(objs)):
                    obj2 = objs[j]
                    if grouped[j]:
                        continue

                    if self._are_equivalent_objects(
                        obj1, obj2, goal_obj_to_fluent_map, goal_exp_is_conjunction
                    ):
                        grouped[j] = True
                        groups[-1].append(obj2)

                groups[-1].sort(key=lambda obj: obj.name)

        return groups

    def _extract_domain_objects(self) -> Set[Object]:
        """
        Extract all objects that appear in the problem's domain.

        Returns:
            Set[Object]: A set of all objects that appear in the domain.
        """

        domain_objects: Set[Object] = set()
        for a in self._lifted_problem.actions:
            if isinstance(a, up.model.InstantaneousAction):
                for p in a.preconditions:
                    domain_objects.update(extract_objects(p))
                for e in a.effects:
                    if e.is_conditional():
                        domain_objects.update(extract_objects(e.condition))
                    domain_objects.update(extract_objects(e.fluent))
                    domain_objects.update(extract_objects(e.value))
            elif isinstance(a, up.model.DurativeAction):
                domain_objects.update(extract_objects(a.duration.lower))
                domain_objects.update(extract_objects(a.duration.upper))
                for interval, cl in a.conditions.items():
                    for c in cl:
                        domain_objects.update(extract_objects(c))
                for t, el in a.effects.items():
                    for e in el:
                        if e.is_conditional():
                            domain_objects.update(extract_objects(e.condition))
                        domain_objects.update(extract_objects(e.fluent))
                        domain_objects.update(extract_objects(e.value))
        return domain_objects

    def _extract_goal_obj_to_fluent_map(
        self,
    ) -> Tuple[Dict[Object, Set[Tuple[Fluent, Tuple[Object], Any]]], bool]:
        """
        Build a mapping from objects to goal fluents they appear in.

        Returns:
            Tuple[Dict[Object, Set[Tuple[Fluent, Tuple[Object], Any]]], bool]:
                - A dictionary mapping each object to the set of associated fluents.
                - A boolean indicating whether the goal expression is a conjunction.
        """

        obj_to_fluent_map: Dict[Object, Set[Tuple[Fluent, Tuple[Object], Any]]] = {
            obj: set() for obj in self._problem.all_objects
        }

        def extract_fluent_equals_constant_exp(
            arg1: FNode, arg2: FNode, is_negated: bool
        ) -> bool:
            fluent_exp = None
            if arg1.is_fluent_exp() and arg2.is_constant():
                fluent_exp = arg1
                v = arg2.constant_value()
            elif arg2.is_fluent_exp() and arg1.is_constant():
                fluent_exp = arg2
                v = arg1.constant_value()

            if fluent_exp is None:
                return False
            else:
                if is_negated:
                    v = (v, False)
                fluent = fluent_exp.fluent()
                objs = tuple(
                    arg.object() for arg in fluent_exp.args if arg.is_object_exp()
                )
                for obj in objs:
                    obj_to_fluent_map[obj].add((fluent, objs, v))

                return True

        is_conjunction = True
        stack: List[FNode] = list(self._problem.goals)
        while len(stack) > 0:
            exp = stack.pop()
            if exp.is_fluent_exp():
                fluent = exp.fluent()
                objs = tuple(arg.object() for arg in exp.args if arg.is_object_exp())
                for obj in objs:
                    obj_to_fluent_map[obj].add((fluent, objs, True))

            elif exp.is_not() and exp.args[0].is_fluent_exp():
                exp = exp.args[0]
                fluent = exp.fluent()
                objs = tuple(arg.object() for arg in exp.args if arg.is_object_exp())
                for obj in objs:
                    obj_to_fluent_map[obj].add((fluent, objs, False))

            elif exp.is_equals():
                arg1, arg2 = exp.args
                if not extract_fluent_equals_constant_exp(arg1, arg2, False):
                    is_conjunction = False
                    stack.extend(exp.args)

            elif exp.is_not() and exp.args[0].is_equals():
                arg1, arg2 = exp.args[0].args
                if not extract_fluent_equals_constant_exp(arg1, arg2, True):
                    is_conjunction = False
                    stack.extend(exp.args)

            elif exp.is_and():
                stack.extend(exp.args)

            else:
                is_conjunction = False
                stack.extend(exp.args)

        return obj_to_fluent_map, is_conjunction

    def _are_equivalent_objects(
        self,
        obj1: Object,
        obj2: Object,
        goal_obj_to_fluent_map: Dict[Object, Set[Tuple[Fluent, Tuple[Object], Any]]],
        goal_exp_is_conjunction: bool,
    ) -> bool:
        """
        Determine whether two objects are equivalent in the problem.

        Args:
            obj1 (Object): The first object to compare.
            obj2 (Object): The second object to compare.
            goal_obj_to_fluent_map (Dict[Object, Set[Tuple[Fluent, Tuple[Object], Any]]]):
                Mapping from objects to the goal fluents they appear in.
            goal_exp_is_conjunction (bool):
                Flag indicating whether the goal expression is a conjunction.

        Returns:
            bool: True if the objects are equivalent; False otherwise.
        """

        if goal_exp_is_conjunction:
            if len(goal_obj_to_fluent_map[obj1]) != len(goal_obj_to_fluent_map[obj2]):
                # the two objects appear in a different number of goal fluents
                return False

            # for each goal fluent involving obj1, ensure the corresponding fluent exists for obj2
            for fluent, objs1, v in goal_obj_to_fluent_map[obj1]:
                objs2 = list(objs1)
                for i, obj in enumerate(objs1):
                    if obj == obj1:
                        objs2[i] = obj2
                if (fluent, tuple(objs2), v) not in goal_obj_to_fluent_map[obj2]:
                    return False

        elif not (
            len(goal_obj_to_fluent_map[obj1]) == 0
            and len(goal_obj_to_fluent_map[obj2]) == 0
        ):
            # the goal is not a conjunction and at least one of the objects appears in it
            return False

        # For each fluent with an explicit initial value, swap obj1 and obj2 in its
        # arguments and verify that the resulting fluent has the same initial value
        obj1_exp = self._problem.environment.expression_manager.ObjectExp(obj1)
        obj2_exp = self._problem.environment.expression_manager.ObjectExp(obj2)
        for fluent_exp, value_exp in self._problem.explicit_initial_values.items():
            fluent = fluent_exp.fluent()
            if fluent.arity == 0:
                continue

            new_args = list(fluent_exp.args)
            args_changed = False
            for i, arg in enumerate(new_args):
                if arg.is_object_exp():
                    if arg == obj1_exp:
                        new_args[i] = obj2_exp
                        args_changed = True
                    elif arg == obj2_exp:
                        new_args[i] = obj1_exp
                        args_changed = True

            if args_changed:
                new_fluent_exp = self._problem.environment.expression_manager.FluentExp(
                    fluent, new_args
                )
                if self._problem.initial_value(new_fluent_exp) != value_exp:
                    return False

        return True

    def _compute_compression_safe_actions(self) -> List[bool]:
        actions = [False] * len(self.action_names)
        fluent_to_conditions, complex_condition_fluents = self._extract_conditions()
        for action_name in self.action_names:
            action = self._problem.action(action_name)
            if (
                isinstance(action, up.model.DurativeAction)
                and not self._has_intermediate_conditions(action)
                and self._end_conditions_contained_in_overall_conditions(action)
                and not self._effects_interfere_with_conditions(
                    action, fluent_to_conditions, complex_condition_fluents
                )
            ):
                actions[self.action_by_name[action_name].idx] = True

        return actions

    def _extract_conditions(self) -> Tuple[Dict[Fluent, Set[bool]], Set[Fluent]]:
        fluent_to_conditions: Dict[Fluent, Set[bool]] = {}
        complex_condition_fluents: Set[Fluent] = set()
        for action in self._problem.actions:
            action_conditions = (
                list(action.conditions.values())
                if isinstance(action, up.model.DurativeAction)
                else [action.preconditions]
            )
            for conds in action_conditions:
                for c in extract_and_arguments(conds):
                    f = None
                    if c.is_fluent_exp():
                        f = c.fluent()
                        v = True
                    elif c.is_not() and c.arg(0).is_fluent_exp():
                        f = c.arg(0).fluent()
                        v = False
                    else:
                        complex_condition_fluents.update(extract_fluents(c))

                    if f is not None:
                        if f not in fluent_to_conditions:
                            fluent_to_conditions[f] = set()
                        fluent_to_conditions[f].add(v)

        return fluent_to_conditions, complex_condition_fluents

    def _has_intermediate_conditions(self, action: "up.model.Action") -> bool:
        return any(
            interval.lower.delay != 0 or interval.upper.delay != 0
            for interval in action.conditions
        )

    def _end_conditions_contained_in_overall_conditions(
        self, action: "up.model.Action"
    ) -> bool:
        end_conditions: Set[FNode] = set()
        overall_conditions: Set[FNode] = set()
        for interval, conditions in action.conditions.items():
            if (
                interval.lower == interval.upper
                and interval.lower.timepoint.kind == TimepointKind.END
                and interval.lower.delay == 0
            ):
                end_conditions.update(extract_and_arguments(conditions))

            elif (
                interval.lower.timepoint.kind == TimepointKind.START
                and interval.upper.timepoint.kind == TimepointKind.END
                and interval.lower.delay == 0
                and interval.upper.delay == 0
            ):
                overall_conditions.update(extract_and_arguments(conditions))

        return all(condition in overall_conditions for condition in end_conditions)

    def _effects_interfere_with_conditions(
        self,
        action: "up.model.Action",
        fluent_to_conditions: Dict[Fluent, Set[bool]],
        complex_condition_fluents: Set[Fluent],
    ) -> bool:
        for timing, effects in action.effects.items():
            if timing.timepoint.kind == TimepointKind.START and timing.delay == 0:
                continue

            for eff in effects:
                f = eff.fluent.fluent()
                if not eff.value.is_bool_constant():
                    return True

                negated_value = not eff.value.bool_constant_value()
                if (
                    f in complex_condition_fluents
                    or negated_value in fluent_to_conditions.get(f, set())
                ):
                    return True

        return False

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
    def relevant_actions(self) -> Optional[List[Action]]:
        return self._relevant_actions

    @property
    def compression_safe_actions(self) -> List[Action]:
        if self._compression_safe_actions is None:
            return []
        return [a for a in self._actions if self._compression_safe_actions[a.idx]]

    @property
    def goal(self) -> Optional[Expression]:
        return self._goal

    def get_action(self, name: str) -> Action:
        return self.action_by_name[name]

    def get_action_name(self, action: Action) -> str:
        return self.action_names[action.idx]

    def are_all_actions_compression_safe(self) -> bool:
        return self._compression_safe_actions is not None and all(
            self._compression_safe_actions
        )

    def build_plan(self, path: List[Action]) -> Plan:
        plan = self.search_space.build_plan(path)
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
        applicable_actions = set()
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
                    applicable_actions.add(self.get_action(a.name))

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
                    applicable_actions.add(self.get_action(a.name))

        self._applicable_actions = [a for a in self._actions if a in applicable_actions]
