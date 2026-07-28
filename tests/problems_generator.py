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

import pathlib
from collections import OrderedDict

import unified_planning.test.examples
from unified_planning.io import PDDLReader
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

    last_position = problem.object(f"p{nPos - 1}")
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


def get_problem_numeric() -> Problem:
    problem = Problem("NumericProblem")

    p = Fluent("p", BoolType())
    a = Fluent("a", IntType())
    b = Fluent("b", IntType())
    x = Fluent("x", RealType())
    y = Fluent("y", RealType())

    problem.add_fluent(p, default_initial_value=False)
    problem.add_fluent(a, default_initial_value=0)
    problem.add_fluent(b, default_initial_value=0)
    problem.add_fluent(x, default_initial_value=2.5)
    problem.add_fluent(y, default_initial_value=3.5)

    action1 = InstantaneousAction("action1")
    action1.add_precondition(Not(p))
    action1.add_effect(p, True)

    action2 = InstantaneousAction("action2")
    action2.add_precondition(p)
    action2.add_precondition(Equals(a, 0))
    action2.add_effect(a, 1)
    action2.add_increase_effect(b, 1)
    action2.add_increase_effect(y, Times(x, y))

    action3 = InstantaneousAction("action3")
    action3.add_precondition(Not(Equals(a, 0)))
    action3.add_precondition(GT(x, 1))
    action3.add_precondition(GT(Times(x, y), 10))
    action3.add_precondition(GT(y, 10))
    action3.add_decrease_effect(x, 0.5)

    action4 = InstantaneousAction("action4")
    action4.add_precondition(Equals(a, 1))
    action4.add_effect(a, 0)

    problem.add_actions([action1, action2, action3, action4])

    problem.add_goal(GT(b, 2))
    problem.add_goal(LT(x, 1.5))

    return problem


def get_problem_satellite() -> Problem:
    reader = PDDLReader()
    problem_directory = pathlib.Path(__file__).resolve().parent / "pddl" / "Satellite"
    domain = problem_directory / "domain.pddl"
    instance = problem_directory / "instance.pddl"
    problem = reader.parse_problem(str(domain), str(instance))
    return problem


def get_problem_hierarchical_types() -> Problem:
    problem = Problem("hierarchical-types")

    # Types (hierarchical)
    Vehicle = UserType("Vehicle")
    Truck = UserType("Truck", Vehicle)
    Van = UserType("Van", Vehicle)

    Package = UserType("Package")
    FragilePackage = UserType("FragilePackage", Package)

    Location = UserType("Location")

    # Objects
    truck1 = Object("truck1", Truck)
    van1 = Object("van1", Van)

    pkg1 = Object("pkg1", Package)
    fragile1 = Object("fragile1", FragilePackage)

    loc1 = Object("loc1", Location)
    loc2 = Object("loc2", Location)
    loc3 = Object("loc3", Location)

    problem.add_objects([truck1, van1, pkg1, fragile1, loc1, loc2, loc3])

    # Fluents
    at_vehicle = Fluent("at_vehicle", BoolType(), v=Vehicle, loc=Location)
    at_package = Fluent("at_package", BoolType(), p=Package, loc=Location)

    # Object fluent (function returning a Vehicle)
    carrier = Fluent("carrier", Vehicle, p=Package)

    problem.add_fluent(at_vehicle, default_initial_value=False)
    problem.add_fluent(at_package, default_initial_value=False)
    problem.add_fluent(carrier, default_initial_value=truck1)

    # Initial state
    problem.set_initial_value(at_vehicle(truck1, loc1), True)
    problem.set_initial_value(at_vehicle(van1, loc2), True)

    problem.set_initial_value(at_package(pkg1, loc1), True)
    problem.set_initial_value(at_package(fragile1, loc3), True)

    # Actions
    move = InstantaneousAction("move", v=Vehicle, l_from=Location, l_to=Location)
    v = move.parameter("v")
    l_from = move.parameter("l_from")
    l_to = move.parameter("l_to")

    move.add_precondition(at_vehicle(v, l_from))
    move.add_effect(at_vehicle(v, l_from), False)
    move.add_effect(at_vehicle(v, l_to), True)

    load = InstantaneousAction("load", p=Package, v=Vehicle, loc=Location)
    p = load.parameter("p")
    v = load.parameter("v")
    loc = load.parameter("loc")

    load.add_precondition(at_package(p, loc))
    load.add_precondition(at_vehicle(v, loc))
    load.add_effect(at_package(p, loc), False)
    load.add_effect(carrier(p), v)

    unload = InstantaneousAction("unload", p=Package, v=Vehicle, loc=Location)
    p = unload.parameter("p")
    v = unload.parameter("v")
    loc = unload.parameter("loc")

    unload.add_precondition(Equals(carrier(p), v))
    unload.add_precondition(at_vehicle(v, loc))
    unload.add_effect(at_package(p, loc), True)

    problem.add_actions([move, load, unload])

    # Goals
    problem.add_goal(at_vehicle(truck1, loc2))

    # inequality goal using object fluent
    problem.add_goal(Not(Equals(carrier(pkg1), van1)))

    return problem


def get_problem_flight() -> Problem:
    problem = Problem("flight")
    City = UserType("City")

    # --- Fluents ---
    at = Fluent("at", BoolType(), city=City)
    connected = Fluent("connected", BoolType(), l_from=City, l_to=City)
    fuel_used = Fluent("fuel_used", RealType(0, 1000))

    # --- Actions ---
    fly_fast = InstantaneousAction("fly_fast", l_from=City, l_to=City)
    l_from = fly_fast.parameter("l_from")
    l_to = fly_fast.parameter("l_to")

    fly_fast.add_precondition(at(l_from))
    fly_fast.add_effect(at(l_from), False)
    fly_fast.add_effect(at(l_to), True)
    fly_fast.add_effect(fuel_used, Plus(fuel_used, 100))

    fly_slow = InstantaneousAction("fly_slow", l_from=City, l_to=City)
    l_from2 = fly_slow.parameter("l_from")
    l_to2 = fly_slow.parameter("l_to")

    fly_slow.add_precondition(at(l_from2))
    fly_slow.add_precondition(connected(l_from2, l_to2))
    fly_slow.add_effect(at(l_from2), False)
    fly_slow.add_effect(at(l_to2), True)
    fly_slow.add_effect(fuel_used, Plus(fuel_used, 10))

    # --- Problem ---
    problem.add_fluent(at, default_initial_value=False)
    problem.add_fluent(connected, default_initial_value=False)
    problem.add_fluent(fuel_used, default_initial_value=0)

    problem.add_action(fly_fast)
    problem.add_action(fly_slow)

    # --- Cities ---
    A = Object("A", City)
    B = Object("B", City)
    C = Object("C", City)
    D = Object("D", City)

    cities = [A, B, C, D]
    problem.add_objects(cities)

    # --- Initial state ---
    problem.set_initial_value(at(A), True)

    problem.set_initial_value(connected(A, B), True)
    problem.set_initial_value(connected(B, A), True)
    problem.set_initial_value(connected(B, C), True)
    problem.set_initial_value(connected(C, B), True)
    problem.set_initial_value(connected(C, D), True)
    problem.set_initial_value(connected(D, C), True)

    # --- Goal ---
    problem.add_goal(at(D))

    return problem


def get_problem_flight_minimize_plan_length() -> Problem:
    problem = get_problem_flight()
    problem.name = "flight_minimize_plan_length"
    problem.add_quality_metric(MinimizeSequentialPlanLength())
    return problem


def get_problem_flight_minimize_fuel() -> Problem:
    problem = get_problem_flight()
    problem.name = "flight_minimize_fuel"
    fuel_used = problem.fluent("fuel_used")
    problem.add_quality_metric(MinimizeExpressionOnFinalState(fuel_used))
    return problem


def get_problem_flight_maximize_fuel() -> Problem:
    problem = get_problem_flight()
    problem.name = "flight_maximize_fuel"
    fuel_used = problem.fluent("fuel_used")
    problem.add_quality_metric(MaximizeExpressionOnFinalState(fuel_used))
    return problem


def get_problem_temporal_flight() -> Problem:
    problem = Problem("temporal_flight")
    City = UserType("City")

    # --- Fluents ---
    at = Fluent("at", BoolType(), city=City)
    connected = Fluent("connected", BoolType(), l_from=City, l_to=City)
    fuel_used = Fluent("fuel_used", RealType(0, 1000))

    # --- Actions ---
    fly_fast = DurativeAction("fly_fast", l_from=City, l_to=City)
    l_from = fly_fast.parameter("l_from")
    l_to = fly_fast.parameter("l_to")

    fly_fast.set_fixed_duration(10)
    fly_fast.add_condition(StartTiming(), at(l_from))
    fly_fast.add_effect(StartTiming(), at(l_from), False)
    fly_fast.add_effect(EndTiming(), at(l_to), True)
    fly_fast.add_effect(StartTiming(), fuel_used, Plus(fuel_used, 100))

    fly_slow = DurativeAction("fly_slow", l_from=City, l_to=City)
    l_from2 = fly_slow.parameter("l_from")
    l_to2 = fly_slow.parameter("l_to")

    fly_slow.set_fixed_duration(20)
    fly_slow.add_condition(StartTiming(), at(l_from2))
    fly_slow.add_condition(StartTiming(), connected(l_from2, l_to2))
    fly_slow.add_effect(StartTiming(), at(l_from2), False)
    fly_slow.add_effect(EndTiming(), at(l_to2), True)
    fly_slow.add_effect(StartTiming(), fuel_used, Plus(fuel_used, 10))

    # --- Problem ---
    problem.add_fluent(at, default_initial_value=False)
    problem.add_fluent(connected, default_initial_value=False)
    problem.add_fluent(fuel_used, default_initial_value=0)

    problem.add_action(fly_fast)
    problem.add_action(fly_slow)

    # --- Cities ---
    A = Object("A", City)
    B = Object("B", City)
    C = Object("C", City)
    D = Object("D", City)

    cities = [A, B, C, D]
    problem.add_objects(cities)

    # --- Initial state ---
    problem.set_initial_value(at(A), True)
    problem.set_initial_value(connected(A, B), True)
    problem.set_initial_value(connected(B, A), True)
    problem.set_initial_value(connected(B, C), True)
    problem.set_initial_value(connected(C, B), True)
    problem.set_initial_value(connected(C, D), True)
    problem.set_initial_value(connected(D, C), True)

    # --- Goal ---
    problem.add_goal(at(D))

    return problem


def get_problem_temporal_flight_minimize_makespan() -> Problem:
    problem = get_problem_temporal_flight()
    problem.name = "temporal_flight_minimize_makespan"
    problem.add_quality_metric(MinimizeMakespan())
    return problem


def get_problem_temporal_flight_minimize_fuel() -> Problem:
    problem = get_problem_temporal_flight()
    problem.name = "temporal_flight_minimize_fuel"
    fuel_used = problem.fluent("fuel_used")
    problem.add_quality_metric(MinimizeExpressionOnFinalState(fuel_used))
    return problem


def get_problem_temporal_flight_maximize_fuel() -> Problem:
    problem = get_problem_temporal_flight()
    problem.name = "temporal_flight_maximize_fuel"
    fuel_used = problem.fluent("fuel_used")
    problem.add_quality_metric(MaximizeExpressionOnFinalState(fuel_used))
    return problem


def get_problem_temporal_fluent_duration() -> Problem:
    """A single-action temporal problem whose duration is a fluent that the
    action's own start effect overwrites.
    """
    problem = Problem("temporal_fluent_duration")

    charge = Fluent("charge", RealType(0, 100))
    ready = Fluent("ready")
    done = Fluent("done")

    run = DurativeAction("run")
    run.set_fixed_duration(charge())
    run.add_condition(StartTiming(), ready)
    run.add_effect(StartTiming(), ready, False)
    run.add_effect(StartTiming(), charge, 10)
    run.add_effect(EndTiming(), done, True)

    problem.add_fluent(charge, default_initial_value=0)
    problem.add_fluent(ready, default_initial_value=False)
    problem.add_fluent(done, default_initial_value=False)
    problem.add_action(run)

    problem.set_initial_value(charge, 5)
    problem.set_initial_value(ready, True)

    problem.add_goal(done)

    return problem


def get_problem_object_value_symmetry_initial() -> Problem:
    Token = UserType("Token")
    Slot = UserType("Slot")
    a, b, s = Object("a", Token), Object("b", Token), Object("s", Slot)

    selected = Fluent("selected", Token, slot=Slot)
    done = Fluent("done", BoolType())

    finish = InstantaneousAction("finish", token=Token)
    token = finish.parameter("token")
    finish.add_precondition(Equals(selected(s), token))
    finish.add_effect(selected(s), token)
    finish.add_effect(done, True)

    problem = Problem("object_value_symmetry_initial")
    problem.add_objects([a, b, s])
    problem.add_fluent(selected)
    problem.add_fluent(done, default_initial_value=False)
    problem.set_initial_value(selected(s), b)
    problem.add_action(finish)
    problem.add_goal(done)
    return problem


def get_problem_object_value_symmetry_goal() -> Problem:
    Token = UserType("Token")
    Slot = UserType("Slot")
    a, b, c = Object("a", Token), Object("b", Token), Object("c", Token)
    s = Object("s", Slot)

    selected = Fluent("selected", Token, slot=Slot)

    finish = InstantaneousAction("finish", token=Token)
    token = finish.parameter("token")
    finish.add_precondition(Equals(selected(s), c))
    finish.add_effect(selected(s), token)

    problem = Problem("object_value_symmetry_goal")
    problem.add_objects([a, b, c, s])
    problem.add_fluent(selected)
    problem.set_initial_value(selected(s), c)
    problem.add_action(finish)
    problem.add_goal(Equals(selected(s), b))
    return problem


def get_problem_object_value_symmetry_retained() -> Problem:
    Token = UserType("Token")
    a, b = Object("a", Token), Object("b", Token)
    nxt = Fluent("nxt", Token, x=Token)

    problem = Problem("object_value_symmetry_retained")
    problem.add_objects([a, b])
    problem.add_fluent(nxt)
    problem.set_initial_value(nxt(a), b)
    problem.set_initial_value(nxt(b), a)
    return problem


def get_problem_default_value_object_symmetry() -> Problem:
    Token = UserType("Token")
    o1, o2 = Object("o1", Token), Object("o2", Token)
    f = Fluent("f", Token, x=Token)

    problem = Problem("default_value_object_symmetry")
    problem.add_objects([o1, o2])
    problem.add_fluent(f, default_initial_value=o1)
    problem.set_initial_value(f(o2), o2)
    return problem


def get_problem_default_value_object_asymmetry() -> Problem:
    Token = UserType("Token")
    x, y, z = Object("x", Token), Object("y", Token), Object("z", Token)
    g = Fluent("g", Token, p=Token)

    problem = Problem("default_value_object_asymmetry")
    problem.add_objects([x, y, z])
    problem.add_fluent(g, default_initial_value=x)
    problem.set_initial_value(g(x), y)
    return problem


def get_problem_goal_taint_partial() -> Problem:
    T = UserType("T")
    p1, p2, p3, p4 = (Object(n, T) for n in ["p1", "p2", "p3", "p4"])
    f = Fluent("f", BoolType(), x=T)
    g = Fluent("g", BoolType(), x=T)
    h = Fluent("h", BoolType(), x=T)

    problem = Problem("goal_taint_partial")
    problem.add_objects([p1, p2, p3, p4])
    problem.add_fluent(f, default_initial_value=False)
    problem.add_fluent(g, default_initial_value=False)
    problem.add_fluent(h, default_initial_value=False)
    # Or(...) is an unrecognized goal shape: it taints p1 and p4 (the objects
    # it references), but must not affect p2/p3, which only appear in
    # recognized (plain fluent) goal conjuncts and remain genuinely symmetric.
    problem.add_goal(Or(f(p1), g(p4)))
    problem.add_goal(h(p2))
    problem.add_goal(h(p3))
    return problem


def get_problem_goal_taint_equals_fluent_fluent() -> Problem:
    T = UserType("T")
    a, b = Object("a", T), Object("b", T)
    fl = Fluent("fl", IntType(), x=T)
    fr = Fluent("fr", IntType(), x=T)

    problem = Problem("goal_taint_equals_fluent_fluent")
    problem.add_objects([a, b])
    problem.add_fluent(fl, default_initial_value=0)
    problem.add_fluent(fr, default_initial_value=0)
    # Equals(fluent, fluent) (neither side a constant) is an unrecognized
    # shape: it must taint a and b as a single opaque conjunct, not be
    # decomposed into independent "fl(a) must hold"/"fr(b) must hold"
    # literals (which would be a different, wrong constraint).
    problem.add_goal(Equals(fl(a), fr(b)))
    return problem


def get_problem_anytime_symmetric_delivery() -> Problem:
    Pkg = UserType("Pkg")
    p1, p2, p3 = Object("p1", Pkg), Object("p2", Pkg), Object("p3", Pkg)

    delivered = Fluent("delivered", BoolType(), p=Pkg)
    deliver = InstantaneousAction("deliver", p=Pkg)
    deliver.add_effect(delivered(deliver.parameter("p")), True)

    problem = Problem("anytime_symmetric_delivery")
    problem.add_objects([p1, p2, p3])
    problem.add_fluent(delivered, default_initial_value=False)
    problem.add_action(deliver)
    problem.add_goal(delivered(p1))
    problem.add_goal(delivered(p2))
    problem.add_goal(delivered(p3))
    return problem


# Interpreted-function problems below use plain `IntType()` rather than a
# bounded range: TamerLite's `supported_kind` doesn't declare `BOUNDED_TYPES`
# at all (a pre-existing gap unrelated to interpreted functions), so a
# bounded IntType would make `TamerLite.supports()` reject the problem
# regardless of interpreted-function support.


def get_problem_if_bool_condition() -> Problem:
    """A boolean interpreted function gating a precondition, adapted from
    `unified_planning.test.examples.interpreted_functions_examples
    .IF_in_conditions_complex_1` (dropping the bounded int types)."""

    def integers_to_bool(ina, inb):
        return (ina * inb) == 60

    def int_to_int(inc):
        return inc - 2

    IF_integers_to_bool = InterpretedFunction(
        "integers_to_bool",
        BoolType(),
        OrderedDict(ina=IntType(), inb=IntType()),
        integers_to_bool,
    )
    IF_int_to_int = InterpretedFunction(
        "simple_int_to_int", IntType(), OrderedDict(inc=IntType()), int_to_int
    )

    g = Fluent("g", IntType())
    ione = Fluent("ione", IntType())
    itwo = Fluent("itwo", IntType())
    ithree = Fluent("ithree", IntType())

    a = InstantaneousAction("a")
    a.add_precondition(And(IF_integers_to_bool(ione, itwo), LT(ione, 15)))
    a.add_precondition(LT(g, 10))
    a.add_effect(g, Plus(g, 3))
    c = InstantaneousAction("c")
    c.add_effect(ione, Plus(ione, 1))
    d = InstantaneousAction("d")
    d.add_effect(ione, Minus(ione, 1))
    f = InstantaneousAction("f")
    f.add_precondition(GT(ione, IF_int_to_int(ithree)))
    f.add_effect(itwo, 5)

    problem = Problem("if_bool_condition")
    problem.add_fluent(g)
    problem.add_fluent(ione)
    problem.add_fluent(itwo)
    problem.add_fluent(ithree)
    problem.add_action(a)
    problem.add_action(c)
    problem.add_action(d)
    problem.add_action(f)
    problem.set_initial_value(g, 1)
    problem.set_initial_value(ione, 11)
    problem.set_initial_value(itwo, 1)
    problem.set_initial_value(ithree, 15)
    problem.add_goal(GE(g, 5))
    return problem


def get_problem_if_numeric_effect() -> Problem:
    """A numeric interpreted function used as an effect assignment value."""

    def double_it(x):
        return x * 2

    IF_double = InterpretedFunction(
        "double_it", IntType(), OrderedDict(x=IntType()), double_it
    )

    counter = Fluent("counter", IntType())
    result = Fluent("result", IntType())
    act = InstantaneousAction("act")
    act.add_precondition(LT(counter, 5))
    act.add_effect(counter, Plus(counter, 1))
    act.add_effect(result, IF_double(counter))

    problem = Problem("if_numeric_effect")
    problem.add_fluent(counter)
    problem.add_fluent(result)
    problem.add_action(act)
    problem.set_initial_value(counter, 0)
    problem.set_initial_value(result, 0)
    problem.add_goal(Equals(result, 8))
    return problem


def get_problem_if_minimal_chain() -> Problem:
    """Reuses UP's own bundled example -- the only one of its IF examples
    that doesn't also require `BOUNDED_TYPES`."""
    problem = unified_planning.test.examples.get_example_problems()[
        "interpreted_functions_minimal_chain_of_assignments"
    ].problem
    assert isinstance(problem, Problem)
    return problem


def get_problem_if_object_argument() -> Problem:
    """A boolean interpreted function taking an object-typed argument --
    object-typed interpreted-function parameters aren't supported yet (see
    `Converter.walk_interpreted_function_exp`), so this problem is used to
    check that a clear `NotImplementedError` is raised rather than silently
    doing the wrong thing."""
    Loc = UserType("Loc")
    l1 = Object("l1", Loc)
    l2 = Object("l2", Loc)

    def loc_check(loc):
        return True

    IF_obj = InterpretedFunction(
        "loc_check", BoolType(), OrderedDict(loc=Loc), loc_check
    )
    at = Fluent("at", Loc)
    act = InstantaneousAction("act")
    act.add_precondition(IF_obj(at))
    act.add_effect(at, l2)

    problem = Problem("if_object_argument")
    problem.add_fluent(at)
    problem.add_object(l1)
    problem.add_object(l2)
    problem.add_action(act)
    problem.set_initial_value(at, l1)
    problem.add_goal(Equals(at, l2))
    return problem


def get_problem_if_undefined_initial_numeric() -> Problem:
    """A fluent left undefined for one object, read both inside an
    interpreted-function precondition and an interpreted-function effect
    value -- adapted from UP's `interpreted_functions_undef_numeric` example
    (dropping the object-typed `choose`/`use_chosen` action, since
    object-typed interpreted-function arguments aren't supported yet). The
    `set_value` action (writing `value`, mirroring the UP original) is kept:
    without it `value` would be a static (never-written) fluent and UP's
    Grounder would constant-fold every interpreted-function call away,
    leaving nothing interpreted-function-related in the ground problem this
    combination is meant to exercise."""
    Obj = UserType("Obj")
    o1 = Object("o1", Obj)
    o2 = Object("o2", Obj)

    def is_big(x):
        return x >= 3

    IF_is_big = InterpretedFunction(
        "is_big", BoolType(), OrderedDict(x=IntType()), is_big
    )

    def double_it(x):
        return 2 * x

    IF_double = InterpretedFunction(
        "double", IntType(), OrderedDict(x=IntType()), double_it
    )

    value = Fluent("value", IntType(), o=Obj)
    total = Fluent("total", IntType())

    use_value = InstantaneousAction("use_value", o=Obj)
    use_value.add_precondition(IF_is_big(value(use_value.o)))
    use_value.add_effect(total, IF_double(value(use_value.o)))

    set_value = InstantaneousAction("set_value", o=Obj)
    set_value.add_effect(value(set_value.o), IF_double(value(o1)))

    problem = Problem("if_undefined_initial_numeric")
    problem.add_fluent(value)
    problem.add_fluent(total, default_initial_value=0)
    problem.add_object(o1)
    problem.add_object(o2)
    problem.add_action(use_value)
    problem.add_action(set_value)
    problem.set_initial_value(value(o1), 4)  # value(o2) is left undefined
    problem.add_goal(GE(total, 1))
    return problem


def get_problem_if_temporal_compression_safe() -> Problem:
    """A durative action whose start condition and start effect both use an
    interpreted function. Compression-safe per TamerLite's own
    `Encoder._compute_compression_safe_actions`: its only condition is at
    start, and its only non-start effect assigns a bool constant -- so
    solving it exercises TamerLite's compression-safe-action detection and
    the TimedToSequential recompile, not just the general IF machinery.
    Interpreted functions in a duration remain unsupported (the Grounder
    doesn't declare `INTERPRETED_FUNCTIONS_IN_DURATIONS`), so the duration
    here is a plain constant."""

    def is_low(level):
        return level < 10

    IF_is_low = InterpretedFunction(
        "is_low", BoolType(), OrderedDict(level=IntType()), is_low
    )

    def boost(level):
        return min(level + 3, 10)

    IF_boost = InterpretedFunction(
        "boost", IntType(), OrderedDict(level=IntType()), boost
    )

    battery = Fluent("battery", IntType())
    charged = Fluent("charged")

    charge = DurativeAction("charge")
    charge.set_fixed_duration(1)
    charge.add_condition(StartTiming(), IF_is_low(battery))
    charge.add_effect(StartTiming(), battery, IF_boost(battery))
    charge.add_effect(EndTiming(), charged, True)

    problem = Problem("if_temporal_compression_safe")
    problem.add_fluent(battery)
    problem.add_fluent(charged, default_initial_value=False)
    problem.add_action(charge)
    problem.set_initial_value(battery, 2)
    problem.add_goal(charged)
    return problem
