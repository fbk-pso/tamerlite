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

import pytest
from unified_planning.engines import PlanGenerationResultStatus
from unified_planning.model import Problem
from unified_planning.shortcuts import (
    Equals,
    Fluent,
    InstantaneousAction,
    Not,
    Object,
    UserType,
)

from tamerlite.encoder import Encoder
from tamerlite.engine import SearchParams, TamerLite


def _equivalence_classes(problem: Problem) -> set[frozenset[str]]:
    with problem.environment.factory.Compiler(
        compilation_kind="GROUNDING", problem_kind=problem.kind
    ) as compiler:
        compilation_result = compiler.compile(problem)

    encoder = Encoder(
        compilation_result.problem,
        problem,
        compilation_result.map_back_action_instance,
        symmetry_breaking=False,
        compression_safe_actions=False,
    )
    return {
        frozenset(obj.name for obj in group)
        for group in encoder._compute_equivalent_objects()
    }


def _required_object_action_problem(
    use_argument: bool,
) -> tuple[Problem, Object, Object]:
    problem = Problem("object-valued-initial-state")
    value_type = UserType("Value")
    slot_type = UserType("Slot")
    value_a = Object("value-a", value_type)
    value_b = Object("value-b", value_type)
    slot = Object("slot", slot_type)
    problem.add_objects([value_a, value_b, slot])

    if use_argument:
        selected = Fluent("selected", value_type, slot=slot_type)
        problem.add_fluent(selected)
        problem.set_initial_value(selected(slot), value_b)
    else:
        selected = Fluent("selected", value_type)
        problem.add_fluent(selected, default_initial_value=value_b)

    done = Fluent("done")
    problem.add_fluent(done, default_initial_value=False)
    finish = InstantaneousAction("finish", value=value_type)
    value = finish.parameter("value")
    selected_exp = selected(slot) if use_argument else selected()
    finish.add_precondition(Equals(selected_exp, value))
    finish.add_effect(selected_exp, value)
    finish.add_effect(done, True)
    problem.add_action(finish)
    problem.add_goal(done)
    return problem, value_a, value_b


@pytest.mark.parametrize(
    "use_argument",
    [True, False],
    ids=["explicit-parameterized", "default-nullary"],
)
def test_object_valued_initial_state_breaks_equivalence(
    use_argument: bool,
) -> None:
    problem, value_a, value_b = _required_object_action_problem(use_argument)

    classes = _equivalence_classes(problem)

    assert frozenset({value_a.name}) in classes
    assert frozenset({value_b.name}) in classes


def test_symmetry_breaking_keeps_required_object_action() -> None:
    problem, _, _ = _required_object_action_problem(use_argument=True)
    planner = TamerLite(
        search=SearchParams(
            search="bfs",
            heuristic="blind",
            symmetry_breaking=True,
            compression_safe_actions=False,
        )
    )

    result = planner.solve(problem)

    assert result.status == PlanGenerationResultStatus.SOLVED_SATISFICING
    assert result.plan is not None


@pytest.mark.parametrize("negated", [False, True], ids=["equality", "inequality"])
def test_object_valued_goal_breaks_equivalence(negated: bool) -> None:
    problem = Problem("object-valued-goal")
    value_type = UserType("Value")
    value_a = Object("value-a", value_type)
    value_b = Object("value-b", value_type)
    initial_value = Object("initial-value", value_type)
    problem.add_objects([value_a, value_b, initial_value])

    selected = Fluent("selected", value_type)
    problem.add_fluent(selected, default_initial_value=initial_value)
    distinguished = Fluent("distinguished", value=value_type)
    problem.add_fluent(distinguished, default_initial_value=False)
    problem.set_initial_value(distinguished(initial_value), True)
    select = InstantaneousAction("select", value=value_type)
    select.add_effect(selected, select.parameter("value"))
    problem.add_action(select)
    goal = Equals(selected, value_b)
    problem.add_goal(Not(goal) if negated else goal)

    classes = _equivalence_classes(problem)

    assert frozenset({value_a.name}) in classes
    assert frozenset({value_b.name}) in classes


def test_complete_object_transposition_preserves_equivalence() -> None:
    problem = Problem("symmetric-object-values")
    value_type = UserType("Value")
    value_a = Object("value-a", value_type)
    value_b = Object("value-b", value_type)
    problem.add_objects([value_a, value_b])

    partner = Fluent("partner", value_type, value=value_type)
    problem.add_fluent(partner)
    problem.set_initial_value(partner(value_a), value_b)
    problem.set_initial_value(partner(value_b), value_a)
    assign = InstantaneousAction("assign", source=value_type, target=value_type)
    assign.add_effect(partner(assign.parameter("source")), assign.parameter("target"))
    problem.add_action(assign)
    problem.add_goal(Equals(partner(value_a), value_b))
    problem.add_goal(Equals(partner(value_b), value_a))

    classes = _equivalence_classes(problem)

    assert frozenset({value_a.name, value_b.name}) in classes
