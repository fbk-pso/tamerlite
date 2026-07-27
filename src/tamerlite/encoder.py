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

from collections.abc import Callable, Iterable
from fractions import Fraction
from typing import Any, cast

import unified_planning as up
from unified_planning.model import Fluent, FNode, Object, Problem, TimepointKind, Type
from unified_planning.model.types import _UserType
from unified_planning.model.walkers import ExpressionQuantifiersRemover, Nnf
from unified_planning.plans import (
    ActionInstance,
    Plan,
    SequentialPlan,
    TimeTriggeredPlan,
)

from tamerlite.converter import Converter
from tamerlite.core import (
    Action,
    Effect,
    Event,
    Expression,
    HMax,
    SearchSpace,
    Timing,
    get_fluents,
)
from tamerlite.core.search_space import ConstantNode, SearchSpaceABC


def extract_objects(exp: FNode) -> Iterable[Object]:
    stack: list[FNode] = [exp]
    while len(stack) > 0:
        exp = stack.pop()
        if exp.is_object_exp():
            yield exp.object()
        else:
            stack.extend(exp.args)


def extract_fluents(exp: FNode) -> Iterable[Fluent]:
    stack: list[FNode] = [exp]
    while len(stack) > 0:
        exp = stack.pop()
        if exp.is_fluent_exp():
            yield exp.fluent()
        else:
            stack.extend(exp.args)


def extract_and_arguments(expressions: list[FNode]) -> Iterable[FNode]:
    stack: list[FNode] = list(expressions)
    while len(stack) > 0:
        exp = stack.pop()
        if exp.is_and():
            stack.extend(exp.args)
        else:
            yield exp


# Value recorded for a fluent appearing in a goal conjunct: the raw constant
# (bool/int/Fraction/Object), or a `(value, False)` pair marking a negated
# fluent-equals-constant comparison.
ConstantValue = bool | int | Fraction | Object
GoalFluentValue = ConstantValue | tuple[ConstantValue, bool]


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
        map_back_action_instance: Callable[[ActionInstance], ActionInstance | None],
        symmetry_breaking: bool,
        compression_safe_actions: bool,
        relevance_analysis: bool,
        full: bool = True,
        deadline: Fraction | None = None,
    ):
        self._problem = problem
        self._lifted_problem = lifted_problem
        self._map_back_action_instance = map_back_action_instance
        if full:
            self._simplifier = up.model.walkers.Simplifier(problem.environment, problem)
        else:
            self._simplifier = problem.environment.simplifier
        self._qrm = ExpressionQuantifiersRemover(problem.environment)
        self._nnf = Nnf(problem.environment)

        self._problem_initial_values = problem.initial_values
        fluent_types = {}
        for f in self._problem_initial_values:
            if f.type.is_bool_type():
                t = "bool"
            elif f.type.is_int_type():
                t = "int"
            elif f.type.is_real_type():
                t = "real"
            elif f.type.is_user_type():
                t = cast(_UserType, f.type).name
            else:
                raise NotImplementedError
            fluent_types[self._convert_fluent(f)] = t
        self._fluents: list[str] = sorted(fluent_types.keys())
        self._fluent_ids = {f: i for i, f in enumerate(self._fluents)}
        self._fluent_types = [fluent_types[f] for f in self._fluents]

        self._object_names: list[str] = sorted(o.name for o in problem.all_objects)
        self._object_ids = {name: i for i, name in enumerate(self._object_names)}

        self._converter = Converter(problem, self._fluent_ids, self._object_ids)
        self._action_names: list[str] = sorted(
            action.name for action in problem.actions
        )
        self._action_by_name: dict[str, Action] = {
            name: Action(index) for index, name in enumerate(self._action_names)
        }
        self._actions: list[Action] = [
            self._action_by_name[name] for name in self._action_names
        ]
        actions_duration_map: dict[
            str, tuple[Expression, Expression, bool, bool] | None
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
            initial_state = self.initial_state(self._problem_initial_values)
            self._goal = self.goals(problem.goals)

            if symmetry_breaking:
                action_objects, obj_to_prev_actions_map = (
                    self._compute_obj_to_prev_actions_map()
                )
                if not any(obj_to_prev_actions_map):
                    # Symmetry breaking is not beneficial because there are no
                    # equivalent objects
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
            initial_state,
            self._goal,
            self._applicable_actions,
            deadline,
            problem.epsilon,
        )
        self._objects = {}
        for ut in problem.user_types:
            self._objects[cast(_UserType, ut).name] = [
                self._object_ids[o.name] for o in problem.objects(ut)
            ]

        self._relevant_actions = None
        if full and relevance_analysis:
            self._relevant_actions = self._compute_relevant_actions()
            if len(self._relevant_actions) < len(self.applicable_actions):
                self._search_space.relevant_actions = self._relevant_actions

    @property
    def problem(self) -> Problem:
        return self._problem

    def initial_state(self, initial_values: dict[FNode, FNode]) -> list[ConstantNode]:
        initial_state_values = {}
        for f, v in initial_values.items():
            initial_state_values[self._convert_fluent(f)] = self._convert_expression(v)[
                0
            ]

        initial_state = [initial_state_values[f] for f in self._fluents]
        # Initial values are always constants (no OperatorNode/FluentNode), so
        # narrowing the wider ExpressionNode element type to ConstantNode is safe.
        return cast(list[ConstantNode], initial_state)

    def _compute_relevant_actions(self) -> list[Action]:
        assert self.goal is not None
        events = {a: e for a, e in self.events.items() if a in self.applicable_actions}
        heuristic = HMax(
            self.actions,
            self.fluent_types,
            self.objects,
            events,
            self.goal,
            internal_caching=False,
            cache_value_in_state=False,
            inadmissible_numeric_heuristic_variant=False,
        )
        reachable_actions = {
            a.idx
            for a in heuristic.reachable_actions(self._search_space.initial_state())
        }

        actions_affecting_fluent: dict[int, set[int]] = {}
        action_to_condition_fluents: dict[int, set[int]] = {}
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

                for cond in [*list(e.end_conditions), e.conditions]:
                    action_to_condition_fluents[a.idx].update(get_fluents(cond))

        checked_fluents = [False] * len(self._fluents)
        stack = list(get_fluents(self.goal))
        for f in stack:
            checked_fluents[f] = True

        relevant_actions: set[int] = set()
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
    ) -> tuple[list[list[int]], list[set[Action]]]:
        """
        This method produces two outputs:
            1. A list of lists of object ids, where each inner list corresponds
                to the objects used as parameters for the action.
            2. A list, indexed by object id, of the set of actions that include
                the previous equivalent object as a parameter (empty set if the
                object has no such constraint).

        Returns:
            Tuple[List[List[int]], List[Set[Action]]]:
                - List of object id lists for each action.
                - List, indexed by object id, of the set of actions.
        """

        equivalent_objects = self._compute_equivalent_objects()
        prev_equivalent_object = {}
        for group in equivalent_objects:
            for i, obj in enumerate(group):
                prev_equivalent_object[obj] = None if i == 0 else group[i - 1]

        obj_to_actions_map: dict[Object, set[Action]] = {}
        action_objects: list[list[int]] = [[] for _ in range(len(self.actions))]
        for action in self._problem.actions:
            ai = self._map_back_action_instance(action())
            assert ai is not None
            objects = [p.object() for p in ai.actual_parameters if p.is_object_exp()]
            action_objects[self.action_by_name[action.name].idx] = [
                self._object_ids[obj.name] for obj in objects
            ]
            for obj in objects:
                if obj not in obj_to_actions_map:
                    obj_to_actions_map[obj] = set()
                obj_to_actions_map[obj].add(self._action_by_name[action.name])

        obj_to_prev_actions_map: list[set[Action]] = [
            set() for _ in range(len(self._object_names))
        ]
        for obj, prev_obj in prev_equivalent_object.items():
            if prev_obj is not None and prev_obj in obj_to_actions_map:
                obj_to_prev_actions_map[self._object_ids[obj.name]] = (
                    obj_to_actions_map[prev_obj]
                )

        return action_objects, obj_to_prev_actions_map

    def _compute_equivalent_objects(self) -> list[list[Object]]:
        """
        Compute groups of equivalent objects in the problem.

        Returns:
            List[List[Object]]: A list of equivalence classes, where each inner
            list contains objects that are equivalent to each other.
        """

        goal_obj_to_fluent_map, goal_tainted_objects = (
            self._extract_goal_obj_to_fluent_map()
        )
        non_equivalent_objects = self._extract_domain_objects() | goal_tainted_objects
        obj_to_init_assignments = self._compute_obj_to_init_assignments_map()

        objects: dict[Type, list[Object]] = {}
        for obj in self._problem.all_objects:
            if obj.type not in objects:
                objects[obj.type] = []
            objects[obj.type].append(obj)

        groups = []
        for objs in objects.values():
            grouped = [False] * len(objs)
            for i, obj1 in enumerate(objs):
                if grouped[i]:
                    continue

                grouped[i] = True
                groups.append([obj1])

                if obj1 in non_equivalent_objects:
                    # treat all domain objects as non-equivalent objects
                    continue

                for j in range(i + 1, len(objs)):
                    if grouped[j]:
                        continue

                    obj2 = objs[j]
                    if obj2 in non_equivalent_objects:
                        continue

                    if self._are_equivalent_objects(
                        obj1,
                        obj2,
                        goal_obj_to_fluent_map,
                        obj_to_init_assignments,
                    ):
                        grouped[j] = True
                        groups[-1].append(obj2)

                groups[-1].sort(key=lambda obj: obj.name)

        return groups

    def _extract_domain_objects(self) -> set[Object]:
        """
        Extract all objects that appear in the problem's domain.

        Returns:
            Set[Object]: A set of all objects that appear in the domain.
        """

        domain_objects: set[Object] = set()
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
                for cl in a.conditions.values():
                    for c in cl:
                        domain_objects.update(extract_objects(c))
                for el in a.effects.values():
                    for e in el:
                        if e.is_conditional():
                            domain_objects.update(extract_objects(e.condition))
                        domain_objects.update(extract_objects(e.fluent))
                        domain_objects.update(extract_objects(e.value))
        return domain_objects

    def _compute_obj_to_init_assignments_map(
        self,
    ) -> dict[Object, list[tuple[FNode, FNode]]]:
        """
        Build a mapping from each object to the initial-value assignments it
        participates in, either as a fluent argument or as the assigned value.

        Uses `initial_values` (the complete grounded initial state, defaults
        included) rather than `explicit_initial_values`, so that an object
        used only as a fluent's default value is checked precisely by
        transposition in `_are_equivalent_objects` instead of needing to be
        conservatively excluded from equivalence altogether.

        Returns:
            Dict[Object, List[Tuple[FNode, FNode]]]: Mapping from objects to
            the list of (fluent expression, value expression) assignments they
            appear in.
        """

        obj_to_assignments: dict[Object, list[tuple[FNode, FNode]]] = {}
        for fluent_exp, value_exp in self._problem_initial_values.items():
            objs = {arg.object() for arg in fluent_exp.args if arg.is_object_exp()}
            if value_exp.is_object_exp():
                objs.add(value_exp.object())
            for obj in objs:
                obj_to_assignments.setdefault(obj, []).append((fluent_exp, value_exp))
        return obj_to_assignments

    def _extract_goal_obj_to_fluent_map(
        self,
    ) -> tuple[
        dict[Object, set[tuple[Fluent, tuple[Object, ...], GoalFluentValue]]],
        set[Object],
    ]:
        """
        Build a mapping from objects to goal fluents they appear in.

        The goal (`problem.goals`, and recursively any nested conjunction) is
        decomposed into individual conjuncts. A conjunct is precisely
        understood only if it has one of 4 recognized shapes: a fluent, a
        negated fluent, a fluent compared to a constant, or its negation.
        Objects appearing in any OTHER conjunct (of unrecognized shape, e.g. a
        disjunction, an implication, or a comparison between two fluents) are
        collected into a separate "tainted" set instead of being registered in
        the map: we don't know how to verify that swapping them preserves that
        conjunct, so they must be excluded from equivalence altogether -- but
        this must not affect objects that only ever appear in recognized
        conjuncts elsewhere in the goal.

        Returns:
            Tuple[Dict[Object, Set[Tuple[Fluent, Tuple[Object, ...], GoalFluentValue]]],
            Set[Object]]:
                - A dictionary mapping each object to the set of associated
                  recognized-conjunct entries.
                - The set of objects appearing in some unrecognized conjunct,
                  who must be excluded from equivalence.
        """

        obj_to_fluent_map: dict[
            Object, set[tuple[Fluent, tuple[Object, ...], GoalFluentValue]]
        ] = {obj: set() for obj in self._problem.all_objects}

        def extract_fluent_equals_constant_exp(
            arg1: FNode, arg2: FNode, is_negated: bool
        ) -> bool:
            fluent_exp = None
            value_exp = None
            if arg1.is_fluent_exp() and arg2.is_constant():
                fluent_exp = arg1
                value_exp = arg2
                v = arg2.constant_value()
            elif arg2.is_fluent_exp() and arg1.is_constant():
                fluent_exp = arg2
                value_exp = arg1
                v = arg1.constant_value()

            if fluent_exp is None:
                return False
            else:
                value = (v, False) if is_negated else v
                fluent = fluent_exp.fluent()
                objs = tuple(
                    arg.object() for arg in fluent_exp.args if arg.is_object_exp()
                )
                entry_objs = set(objs)
                assert value_exp is not None
                if value_exp.is_object_exp():
                    entry_objs.add(value_exp.object())
                for obj in entry_objs:
                    obj_to_fluent_map[obj].add((fluent, objs, value))

                return True

        tainted_objects: set[Object] = set()
        stack: list[FNode] = list(self._problem.goals)
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
                    tainted_objects.update(extract_objects(exp))

            elif exp.is_not() and exp.args[0].is_equals():
                arg1, arg2 = exp.args[0].args
                if not extract_fluent_equals_constant_exp(arg1, arg2, True):
                    tainted_objects.update(extract_objects(exp))

            elif exp.is_and():
                stack.extend(exp.args)

            else:
                tainted_objects.update(extract_objects(exp))

        return obj_to_fluent_map, tainted_objects

    def _are_equivalent_objects(
        self,
        obj1: Object,
        obj2: Object,
        goal_obj_to_fluent_map: dict[
            Object, set[tuple[Fluent, tuple[Object, ...], GoalFluentValue]]
        ],
        obj_to_init_assignments: dict[Object, list[tuple[FNode, FNode]]],
    ) -> bool:
        """
        Determine whether two objects are equivalent in the problem, i.e.
        whether swapping them everywhere (as fluent arguments and as
        object-valued fluent values) leaves the goal and initial state
        unchanged.

        Args:
            obj1 (Object): The first object to compare.
            obj2 (Object): The second object to compare.
            goal_obj_to_fluent_map
                (Dict[Object, Set[Tuple[Fluent, Tuple[Object, ...], GoalFluentValue]]]):
                Mapping from objects to the recognized goal fluents they
                appear in (as an argument or as the compared value). Objects
                appearing in an unrecognized goal conjunct are excluded from
                equivalence before reaching this method (see
                `_extract_goal_obj_to_fluent_map`), so this map can be trusted
                to precisely and completely describe every goal constraint
                that could possibly distinguish obj1/obj2.
            obj_to_init_assignments (Dict[Object, List[Tuple[FNode, FNode]]]):
                Mapping from objects to the initial-value assignments
                (explicit or default) they appear in (as an argument or as
                the value).

        Returns:
            bool: True if the objects are equivalent; False otherwise.
        """

        def transpose(x: Object) -> Object:
            return obj2 if x == obj1 else obj1 if x == obj2 else x

        def transpose_constant(c: ConstantValue) -> ConstantValue:
            return transpose(c) if isinstance(c, Object) else c

        def transpose_value(v: GoalFluentValue) -> GoalFluentValue:
            if isinstance(v, tuple):
                return (transpose_constant(v[0]), v[1])
            return transpose_constant(v)

        if len(goal_obj_to_fluent_map[obj1]) != len(goal_obj_to_fluent_map[obj2]):
            # the two objects appear in a different number of goal fluents
            return False

        # for each goal fluent involving obj1, ensure the corresponding
        # fluent (with obj1/obj2 swapped in both the arguments and the
        # compared value) exists for obj2
        for fluent, objs1, v in goal_obj_to_fluent_map[obj1]:
            objs2 = tuple(transpose(obj) for obj in objs1)
            v2 = transpose_value(v)
            if (fluent, objs2, v2) not in goal_obj_to_fluent_map[obj2]:
                return False

        # For each initial-value assignment (explicit or default) involving
        # obj1 or obj2 (as an argument or as the value), swap obj1 and obj2
        # throughout and verify that the resulting assignment still holds.
        obj1_exp = self._problem.environment.expression_manager.ObjectExp(obj1)
        obj2_exp = self._problem.environment.expression_manager.ObjectExp(obj2)

        def swap_exp(exp: FNode) -> FNode:
            if exp == obj1_exp:
                return obj2_exp
            if exp == obj2_exp:
                return obj1_exp
            return exp

        seen_fluent_exps: set[FNode] = set()
        assignments = obj_to_init_assignments.get(
            obj1, []
        ) + obj_to_init_assignments.get(obj2, [])
        for fluent_exp, value_exp in assignments:
            if fluent_exp in seen_fluent_exps:
                continue
            seen_fluent_exps.add(fluent_exp)

            new_fluent_exp = self._problem.environment.expression_manager.FluentExp(
                fluent_exp.fluent(), [swap_exp(arg) for arg in fluent_exp.args]
            )
            if self._problem.initial_value(new_fluent_exp) != swap_exp(value_exp):
                return False

        return True

    def _compute_compression_safe_actions(self) -> list[bool]:
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

    def _extract_conditions(self) -> tuple[dict[Fluent, set[bool]], set[Fluent]]:
        fluent_to_conditions: dict[Fluent, set[bool]] = {}
        complex_condition_fluents: set[Fluent] = set()
        for action in self._problem.actions:
            action_conditions: list[list[FNode]]
            if isinstance(action, up.model.DurativeAction):
                action_conditions = list(action.conditions.values())
            else:
                assert isinstance(action, up.model.InstantaneousAction)
                action_conditions = [action.preconditions]
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

    def _has_intermediate_conditions(self, action: "up.model.DurativeAction") -> bool:
        return any(
            interval.lower.delay != 0 or interval.upper.delay != 0
            for interval in action.conditions
        )

    def _end_conditions_contained_in_overall_conditions(
        self, action: "up.model.DurativeAction"
    ) -> bool:
        end_conditions: set[FNode] = set()
        overall_conditions: set[FNode] = set()
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
        action: "up.model.DurativeAction",
        fluent_to_conditions: dict[Fluent, set[bool]],
        complex_condition_fluents: set[Fluent],
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

    def goals(self, goals: list[FNode]) -> Expression:
        return self._convert_expression(
            self._problem.environment.expression_manager.And(goals)
        )

    @property
    def search_space(self) -> SearchSpaceABC:
        return self._search_space

    @property
    def fluents(self) -> list[str]:
        return self._fluents

    @property
    def fluent_ids(self) -> dict[str, int]:
        return self._fluent_ids

    @property
    def fluent_types(self) -> list[str]:
        return self._fluent_types

    @property
    def objects(self) -> dict[str, list[int]]:
        return self._objects

    @property
    def object_ids(self) -> dict[str, int]:
        return self._object_ids

    @property
    def object_names(self) -> list[str]:
        return self._object_names

    @property
    def events(self) -> dict[Action, list[tuple[Timing, Event]]]:
        return self._events

    @property
    def actions(self) -> list[Action]:
        return self._actions

    @property
    def action_names(self) -> list[str]:
        return self._action_names

    @property
    def action_by_name(self) -> dict[str, Action]:
        return self._action_by_name

    @property
    def applicable_actions(self) -> list[Action]:
        return self._applicable_actions

    @property
    def relevant_actions(self) -> list[Action] | None:
        return self._relevant_actions

    @property
    def compression_safe_actions(self) -> list[Action]:
        if self._compression_safe_actions is None:
            return []
        return [a for a in self._actions if self._compression_safe_actions[a.idx]]

    @property
    def goal(self) -> Expression | None:
        return self._goal

    def get_action(self, name: str) -> Action:
        return self.action_by_name[name]

    def get_action_name(self, action: Action) -> str:
        return self.action_names[action.idx]

    def are_all_actions_compression_safe(self) -> bool:
        return self._compression_safe_actions is not None and all(
            self._compression_safe_actions
        )

    def is_any_action_compression_safe(self) -> bool:
        return self._compression_safe_actions is not None and any(
            self._compression_safe_actions
        )

    def build_plan(self, path: list[Action]) -> Plan:
        plan = self.search_space.build_plan(path)
        if self._is_temporal:
            actions = []
            for s, a, d in plan:
                assert s is not None
                actions.append((s, self._problem.action(self.get_action_name(a))(), d))
            return TimeTriggeredPlan(actions)
        else:
            return SequentialPlan(
                [self._problem.action(self.get_action_name(a))() for _, a, _ in plan]
            )

    def _convert_fluent(self, fluent_exp: FNode) -> str:
        return str(fluent_exp)

    def _convert_expression(self, expression: FNode) -> Expression:
        expression = self._qrm.remove_quantifiers(expression, self._problem)
        expression = self._simplifier.simplify(expression)
        expression = self._nnf.get_nnf_expression(expression)
        return self._converter.convert(expression)

    def _convert_timing(self, timing: "up.model.Timing") -> Timing:
        return Timing(timing.is_from_start(), Fraction(timing.delay))

    def _convert_effects(self, effects: list["up.model.Effect"]) -> list[Effect]:
        env = self._problem.environment
        em = env.expression_manager
        fluent_to_effects: dict[FNode, list[list[FNode]]] = {}
        for effect in effects:
            if effect.fluent not in fluent_to_effects:
                fluent_to_effects[effect.fluent] = [[], [], []]

            if effect.is_increase():
                fluent_to_effects[effect.fluent][0].append(effect.value)
            elif effect.is_decrease():
                fluent_to_effects[effect.fluent][1].append(effect.value)
            else:
                fluent_to_effects[effect.fluent][2].append(effect.value)

        converted_effects = []
        for fluent, (
            inc_effects,
            dec_effects,
            assign_effects,
        ) in fluent_to_effects.items():
            some_inc_dec_effects = len(inc_effects) > 0 or len(dec_effects) > 0
            some_assign_effects = len(assign_effects) > 0
            is_bool_type = fluent.fluent().type.is_bool_type()
            assert (some_inc_dec_effects and not some_assign_effects) or (
                not some_inc_dec_effects and some_assign_effects
            )

            if some_assign_effects:
                if len(assign_effects) == 1:
                    value = assign_effects[0]
                elif not is_bool_type:
                    # NOTE: If multiple numeric assignment effects are present,
                    # they are assumed to be identical
                    value = assign_effects[0]
                    for v in assign_effects:
                        assert value == v
                else:
                    value = assign_effects[0]
                    non_constant_assignments = 0
                    for v in assign_effects:
                        if v.is_bool_constant():
                            if v.bool_constant_value():
                                value = v
                                non_constant_assignments = 0
                                break
                        else:
                            value = v
                            non_constant_assignments += 1

                    if non_constant_assignments > 1:
                        raise Exception(
                            "TamerLite does not support multiple non-constant "
                            "boolean assignment effects on the same fluent."
                        )
            else:
                if len(inc_effects) > 0:
                    if len(dec_effects) > 0:
                        value = em.Minus(
                            em.Plus([fluent, *inc_effects]), em.Plus(dec_effects)
                        )
                    else:
                        value = em.Plus([fluent, *inc_effects])
                else:
                    value = em.Minus(fluent, em.Plus(dec_effects))

            f = self.fluent_ids[self._convert_fluent(fluent)]
            converted_value = self._convert_expression(value)
            converted_effects.append(Effect(f, converted_value))

        return converted_effects

    def _build_events(self):
        env = self._problem.environment
        em = env.expression_manager
        self._events: dict[Action, list[tuple[Timing, Event]]] = {}
        applicable_actions = set()
        for a in self._problem.actions:
            if isinstance(a, up.model.DurativeAction):
                from_start: dict[Any, Any] = {}
                from_end: dict[Any, Any] = {}
                action_events: list[
                    tuple[int | Fraction, up.model.Timing, int, list]
                ] = []
                is_applicable = True
                for i, lc in a.conditions.items():
                    lower = i.lower
                    upper = i.upper
                    if lower == upper:  # conditions
                        action_events.append((lower.delay, lower, 1, lc))
                    else:
                        # lower: start conditions
                        if not i.is_left_open():
                            action_events.append((lower.delay, lower, 1, lc))
                        action_events.append((lower.delay, lower, 2, [em.And(lc)]))
                        # upper: end conditions
                        if not i.is_right_open():
                            action_events.append((upper.delay, upper, 1, lc))
                        action_events.append((upper.delay, upper, 3, [em.And(lc)]))
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
                    dur_lower, dur_upper = a.duration.lower, a.duration.upper
                    if (
                        dur_lower.is_constant()
                        and dur_upper.is_constant()
                        and dur_lower.constant_value() == dur_upper.constant_value()
                    ):
                        duration = dur_lower.constant_value()
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
                            "TamerLite does not support ICE from start and from "
                            "end inside the same action!"
                        )

                self._events[self.get_action(a.name)] = []
                pos = 0
                for d in sorted(from_start):
                    t, lc, lsc, lec, le = from_start[d]
                    conv_t = self._convert_timing(t)
                    c = self._convert_expression(em.And(lc))
                    tsc = tuple([self._convert_expression(sc) for sc in lsc])
                    tec = tuple([self._convert_expression(ec) for ec in lec])
                    te = tuple(self._convert_effects(le))
                    self._events[self.get_action(a.name)].append(
                        (conv_t, Event(self.get_action(a.name), pos, c, tsc, tec, te))
                    )
                    pos += 1
                for d in sorted(from_end):
                    t, lc, lsc, lec, le = from_end[d]
                    conv_t = self._convert_timing(t)
                    c = self._convert_expression(em.And(lc))
                    tsc = tuple([self._convert_expression(sc) for sc in lsc])
                    tec = tuple([self._convert_expression(ec) for ec in lec])
                    te = tuple(self._convert_effects(le))
                    self._events[self.get_action(a.name)].append(
                        (conv_t, Event(self.get_action(a.name), pos, c, tsc, tec, te))
                    )
                    pos += 1
            else:
                assert isinstance(a, up.model.InstantaneousAction)
                conv_t = Timing(True, Fraction(0))
                te = tuple(self._convert_effects(a.effects))
                self._events[self.get_action(a.name)] = [
                    (
                        conv_t,
                        Event(
                            self.get_action(a.name),
                            0,
                            self._convert_expression(em.And(a.preconditions)),
                            (),
                            (),
                            te,
                        ),
                    )
                ]
                if not self._simplifier.simplify(em.And(a.preconditions)).is_false():
                    applicable_actions.add(self.get_action(a.name))

        self._applicable_actions = [a for a in self._actions if a in applicable_actions]
