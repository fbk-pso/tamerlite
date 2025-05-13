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

from unified_planning.shortcuts import *


def get_problem_matchcellar(n) -> Problem:
    Match, Fuse = UserType("Match"), UserType("Fuse")

    handfree, light = Fluent("handfree"), Fluent("light")
    match_used = Fluent("match_used", BoolType(), match=Match)
    fuse_mended = Fluent("fuse_mended", BoolType(), fuse=Fuse)

    light_match = DurativeAction("light_match", m=Match)
    light_match.set_fixed_duration(6)
    light_match.add_condition(StartTiming(), Not(match_used(light_match.m)))
    light_match.add_effect(StartTiming(), match_used(light_match.m), True)
    light_match.add_effect(StartTiming(), light, True)
    light_match.add_effect(EndTiming(), light, False)

    mend_fuse = DurativeAction("mend_fuse", f=Fuse)
    mend_fuse.set_fixed_duration(5)
    mend_fuse.add_condition(StartTiming(), handfree)
    mend_fuse.add_condition(StartTiming(), Not(fuse_mended(mend_fuse.f)))
    mend_fuse.add_condition(ClosedTimeInterval(StartTiming(), EndTiming()), light)
    mend_fuse.add_effect(StartTiming(), handfree, False)
    mend_fuse.add_effect(EndTiming(), fuse_mended(mend_fuse.f), True)
    mend_fuse.add_effect(EndTiming(), handfree, True)

    problem = Problem("MatchCellar")
    problem.add_fluents([handfree, light])
    problem.add_fluent(match_used, default_initial_value=False)
    problem.add_fluent(fuse_mended, default_initial_value=False)
    problem.add_actions([light_match, mend_fuse])
    problem.set_initial_value(light, False)
    problem.set_initial_value(handfree, True)

    for i in range(1, n + 1):
        f = Object(f"f{i}", Fuse)
        m = Object(f"m{i}", Match)
        problem.add_objects([f, m])
        problem.add_goal(fuse_mended(f))

    return problem


def get_problem_logistics(nRob, nPall, nPos, nTreatment) -> Problem:
    # Setting up Types
    Robot = UserType("Robot")
    Pallet = UserType("Pallet")
    Position = UserType("Position")
    Treatment = UserType("Treatment")

    # Setting up Fluents
    robot_at = Fluent("robot_at", BoolType(), r=Robot, p=Position)
    robot_has = Fluent("robot_has", BoolType(), r=Robot, p=Pallet)
    pallet_at = Fluent("pallet_at", BoolType(), p=Pallet, pos=Position)
    robot_free = Fluent("robot_free", BoolType(), r=Robot)
    position_free = Fluent("position_free", BoolType(), p=Position)
    can_do = Fluent("can_do", BoolType(), p=Position, t=Treatment)
    treated = Fluent("treated", BoolType(), p=Pallet, t=Treatment)
    ready = Fluent("ready", BoolType(), p=Pallet, pos=Position, t=Treatment)
    is_depot = Fluent("is_depot", BoolType(), p=Position)
    battery_level = Fluent("battery_level", IntType(0, 100), r=Robot)
    distance = Fluent("distance", IntType(), pfrom=Position, pto=Position)

    # Setting up Actions:
    move = InstantaneousAction("move", r=Robot, frompos=Position, topos=Position)
    move.add_precondition(Not(Equals(move.frompos, move.topos)))
    move.add_precondition(robot_at(move.r, move.frompos))
    move.add_precondition(GE(battery_level(move.r), distance(move.frompos, move.topos)))
    move.add_effect(robot_at(move.r, move.topos), True)
    move.add_effect(robot_at(move.r, move.frompos), False)
    move.add_decrease_effect(battery_level(move.r), distance(move.frompos, move.topos))

    unload_at_depot = InstantaneousAction(
        "unload_at_depot", r=Robot, pallet=Pallet, pos=Position
    )
    unload_at_depot.add_precondition(is_depot(unload_at_depot.pos))
    unload_at_depot.add_precondition(robot_at(unload_at_depot.r, unload_at_depot.pos))
    unload_at_depot.add_precondition(
        robot_has(unload_at_depot.r, unload_at_depot.pallet)
    )
    unload_at_depot.add_effect(
        pallet_at(unload_at_depot.pallet, unload_at_depot.pos), True
    )
    unload_at_depot.add_effect(robot_free(unload_at_depot.r), True)
    unload_at_depot.add_effect(
        robot_has(unload_at_depot.r, unload_at_depot.pallet), False
    )

    load_at_depot = InstantaneousAction(
        "load_at_depot", r=Robot, pallet=Pallet, pos=Position
    )
    load_at_depot.add_precondition(is_depot(load_at_depot.pos))
    load_at_depot.add_precondition(robot_at(load_at_depot.r, load_at_depot.pos))
    load_at_depot.add_precondition(robot_free(load_at_depot.r))
    load_at_depot.add_precondition(pallet_at(load_at_depot.pallet, load_at_depot.pos))
    load_at_depot.add_effect(robot_free(load_at_depot.r), False)
    load_at_depot.add_effect(robot_has(load_at_depot.r, load_at_depot.pallet), True)
    load_at_depot.add_effect(pallet_at(load_at_depot.pallet, load_at_depot.pos), False)

    make_treat = DurativeAction(
        "make_treatment", r=Robot, pallet=Pallet, pos=Position, t=Treatment
    )
    make_treat.set_fixed_duration(20)
    make_treat.add_condition(StartTiming(), can_do(make_treat.pos, make_treat.t))
    make_treat.add_condition(StartTiming(), position_free(make_treat.pos))
    make_treat.add_condition(StartTiming(), robot_at(make_treat.r, make_treat.pos))
    make_treat.add_condition(StartTiming(), robot_has(make_treat.r, make_treat.pallet))
    make_treat.add_condition(
        StartTiming(), Not(treated(make_treat.pallet, make_treat.t))
    )
    make_treat.add_condition(EndTiming(), treated(make_treat.pallet, make_treat.t))
    make_treat.add_condition(EndTiming(), position_free(make_treat.pos))
    make_treat.add_effect(StartTiming(), position_free(make_treat.pos), False)
    make_treat.add_effect(
        StartTiming(), robot_has(make_treat.r, make_treat.pallet), False
    )
    make_treat.add_effect(
        StartTiming(), pallet_at(make_treat.pallet, make_treat.pos), True
    )
    make_treat.add_effect(StartTiming(), robot_free(make_treat.r), True)
    make_treat.add_effect(
        StartTiming(10), ready(make_treat.pallet, make_treat.pos, make_treat.t), True
    )

    load = InstantaneousAction(
        "load", r=Robot, pallet=Pallet, pos=Position, t=Treatment
    )
    load.add_precondition(ready(load.pallet, load.pos, load.t))
    load.add_precondition(robot_at(load.r, load.pos))
    load.add_precondition(robot_free(load.r))
    load.add_precondition(pallet_at(load.pallet, load.pos))
    load.add_effect(robot_free(load.r), False)
    load.add_effect(ready(load.pallet, load.pos, load.t), False)
    load.add_effect(pallet_at(load.pallet, load.pos), False)
    load.add_effect(robot_has(load.r, load.pallet), True)
    load.add_effect(treated(load.pallet, load.t), True)
    load.add_effect(position_free(load.pos), True)

    problem = Problem("RoboLogistics")

    for f in [
        robot_at,
        robot_free,
        robot_has,
        ready,
        position_free,
        treated,
        pallet_at,
        can_do,
        is_depot,
    ]:
        problem.add_fluent(f, default_initial_value=False)
    problem.add_fluent(battery_level, default_initial_value=0)
    problem.add_fluent(distance, default_initial_value=0)

    problem.add_objects([Object(f"r{i}", Robot) for i in range(nRob)])
    problem.add_objects([Object(f"p{i}", Position) for i in range(nPos)])
    problem.add_objects([Object(f"plt{i}", Pallet) for i in range(nPall)])
    problem.add_objects([Object(f"t{i}", Treatment) for i in range(nTreatment)])

    problem.add_action(move)
    problem.add_action(load)
    problem.add_action(load_at_depot)
    problem.add_action(unload_at_depot)
    problem.add_action(make_treat)

    last_position = problem.object(f"p{nPos-1}")
    # All robots stay at the same position, and so do the pallets
    for i in range(nRob):
        problem.set_initial_value(
            robot_at(problem.object(f"r{i}"), last_position), True
        )
        problem.set_initial_value(robot_free(problem.object(f"r{i}")), True)
    for i in range(nPall):
        problem.set_initial_value(
            pallet_at(problem.object(f"plt{i}"), last_position), True
        )

    for i in range(nRob):
        problem.set_initial_value(
            battery_level(problem.object(f"r{i}")), nPos * nPall * 2
        )

    for i in range(nPos):
        problem.set_initial_value(
            distance(problem.object(f"p{i}"), problem.object(f"p{i}")), 0
        )
        for j in range(i + 1, nPos):
            problem.set_initial_value(
                distance(problem.object(f"p{i}"), problem.object(f"p{j}")), j - i
            )
            problem.set_initial_value(
                distance(problem.object(f"p{j}"), problem.object(f"p{i}")), j - i
            )

    # last position is the depot
    problem.set_initial_value(is_depot(last_position), True)
    for i in range(nPos):
        problem.set_initial_value(position_free(problem.object(f"p{i}")), True)

    # Treatments are done over the various positions
    for i in range(nTreatment):
        treatment_position = i % (nPos - 1)
        problem.set_initial_value(
            can_do(problem.object(f"p{treatment_position}"), problem.object(f"t{i}")),
            True,
        )
        for k in range(nPall):
            problem.add_goal(
                treated(problem.object(f"plt{k}"), problem.object(f"t{i}"))
            )

    return problem
