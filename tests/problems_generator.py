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
from fractions import Fraction

from unified_planning.io import PDDLReader
from unified_planning.shortcuts import *


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


def get_problem_hmax_numeric_reopening() -> Problem:
    """
    Two achievers of the simple numeric condition `x >= 2`: `inc1` needs no
    precondition but two repetitions, `inc2` needs one repetition but sits
    after the chain `a1 -> a2 -> a3`. hmax prices `x >= 2` at
    `rep * cost + min pre` over the achievers expanded so far: 2 after `inc1`
    (closing `r` at 2), then 1 after `inc2`, popped at 3, so `r` must be
    reopened for the goal to drop from 3 to 2.
    """
    problem = Problem("HMaxNumericReopening")

    x = Fluent("x", RealType())
    p1 = Fluent("p1", BoolType())
    p2 = Fluent("p2", BoolType())
    p3 = Fluent("p3", BoolType())
    g = Fluent("g", BoolType())

    problem.add_fluent(x, default_initial_value=0)
    for f in [p1, p2, p3, g]:
        problem.add_fluent(f, default_initial_value=False)

    inc1 = InstantaneousAction("inc1")
    inc1.add_increase_effect(x, 1)

    a1 = InstantaneousAction("a1")
    a1.add_effect(p1, True)

    a2 = InstantaneousAction("a2")
    a2.add_precondition(p1)
    a2.add_effect(p2, True)

    a3 = InstantaneousAction("a3")
    a3.add_precondition(p2)
    a3.add_effect(p3, True)

    inc2 = InstantaneousAction("inc2")
    inc2.add_precondition(p3)
    inc2.add_increase_effect(x, 2)

    r = InstantaneousAction("r")
    r.add_precondition(GE(x, 2))
    r.add_effect(g, True)

    problem.add_actions([inc1, a1, a2, a3, inc2, r])
    problem.add_goal(g)

    return problem


def get_problem_hmax_numeric_single_achiever() -> Problem:
    """
    `get_problem_hmax_numeric_reopening` without `inc2`: `inc1` is the only
    achiever of `x >= 2`, so hmax's cheapest precondition among its achievers
    is always `inc1`'s own, the fix-point is monotone and the heap is not
    drained. hmax prices `x >= 2` at 2, so the goal `g` costs 3.
    """
    problem = Problem("HMaxNumericSingleAchiever")

    x = Fluent("x", RealType())
    p1 = Fluent("p1", BoolType())
    p2 = Fluent("p2", BoolType())
    p3 = Fluent("p3", BoolType())
    g = Fluent("g", BoolType())

    problem.add_fluent(x, default_initial_value=0)
    for f in [p1, p2, p3, g]:
        problem.add_fluent(f, default_initial_value=False)

    inc1 = InstantaneousAction("inc1")
    inc1.add_increase_effect(x, 1)

    a1 = InstantaneousAction("a1")
    a1.add_effect(p1, True)

    a2 = InstantaneousAction("a2")
    a2.add_precondition(p1)
    a2.add_effect(p2, True)

    a3 = InstantaneousAction("a3")
    a3.add_precondition(p2)
    a3.add_effect(p3, True)

    r = InstantaneousAction("r")
    r.add_precondition(GE(x, 2))
    r.add_effect(g, True)

    problem.add_actions([inc1, a1, a2, a3, r])
    problem.add_goal(g)

    return problem


def get_problem_satellite() -> Problem:
    reader = PDDLReader()
    problem_directory = pathlib.Path(__file__).resolve().parent / "pddl" / "Satellite"
    domain = problem_directory / "domain.pddl"
    instance = problem_directory / "instance.pddl"
    problem = reader.parse_problem(str(domain), str(instance))
    return problem


def get_problem_driverlog_pfile1() -> Problem:
    """
    Driverlog IPC pfile1: two drivers, two trucks, two packages, and two
    "connector" locations (p1-0, p1-2) that are each only reachable from a
    different pair of the problem's three hub locations via the static
    `path` fluent. All actions are compression-safe, so a fully-symmetry-
    breaking, fully-compression-safe solve recompiles this problem with
    `TimedToSequential` before building its second `Encoder`. Regression
    fixture for a bug where that compilation pruned `path`/`link` as
    unreferenced (their only reader was a grounding-time-simplified-away
    static precondition), leaving nothing to distinguish p1-0 from p1-2 and
    causing symmetry breaking to wrongly merge them -- see
    `test_symmetry_breaking_compression_safe_prunes_witness_fluent`.
    """
    reader = PDDLReader()
    problem_directory = pathlib.Path(__file__).resolve().parent / "pddl" / "Driverlog"
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


def get_problem_int_vs_rational_equality() -> Problem:
    """Reaches the same real value `x = 3` through two differently encoded
    routes: `set3` assigns the real constant `3.0` (a rational constant
    node), `add3` computes `x + 3` (arithmetic, normalized to an integer).

    Both cores must treat the two resulting states as one: the Python core
    does because `Fraction(3, 1) == 3` with equal hashes; the Rust core only
    does if every integral value is stored as `Int`, never as a
    denominator-1 `Rational`, since its duplicate detection compares and
    hashes values structurally. Without that canonical form the Rust core
    expands one extra state and `check_metrics_equality` fails.
    """
    x = Fluent("x", RealType())
    done = Fluent("done", BoolType())

    # `x < 3` keeps `set3` from being re-applicable forever, which would make
    # dedup-free searches (dfs) never terminate.
    set3 = InstantaneousAction("set3")
    set3.add_precondition(LT(x, 3))
    set3.add_effect(x, Real(Fraction(3)))

    add3 = InstantaneousAction("add3")
    add3.add_effect(x, Plus(x, 3))

    fin = InstantaneousAction("fin")
    fin.add_precondition(GE(x, 12))
    fin.add_effect(done, True)

    problem = Problem("int_vs_rational_equality")
    problem.add_fluent(x, default_initial_value=Real(Fraction(0)))
    problem.add_fluent(done, default_initial_value=False)
    problem.add_actions([set3, add3, fin])
    problem.add_goal(done)
    return problem


def get_problem_object_equality_fluents() -> Problem:
    """Equality (and inequality) between two object-valued *fluents*, not a
    fluent and a literal object -- exercises `DeleteRelaxationHeuristic`'s
    `_simplify_object_equality`, as opposed to
    `_simplify_fluent_not_equals_object_expression` (`fluent != object`),
    which `get_problem_hierarchical_types` above already covers.

    Uses hierarchical types throughout so the two operands of an equality can
    be declared at *different* type names while still sharing objects:

    - `match`'s precondition (`leader == backup`) compares a `Vehicle`
      (superset domain) to a `Truck` (subset domain) -- a non-empty, proper
      intersection.
    - `differ`'s precondition (`not(leader == spare)`) compares a `Vehicle`
      to a `Van` -- also a non-empty intersection, but negated, exercising
      the cross-product expansion instead of the plain intersection one.
    - `impossible`'s precondition (`backup == spare`) compares two *sibling*
      types (`Truck`, `Van`) with disjoint object domains -- always false,
      exercising the empty-intersection ("False" leaf) path. `impossible`
      is never needed to reach the goal (relevance analysis is free to prune
      it), but it's still one of `problem.actions`, so its precondition is
      still converted and classified.
    """
    problem = Problem("object-equality-fluents")

    Vehicle = UserType("Vehicle")
    Truck = UserType("Truck", Vehicle)
    Van = UserType("Van", Vehicle)

    truck1 = Object("truck1", Truck)
    truck2 = Object("truck2", Truck)
    van1 = Object("van1", Van)
    problem.add_objects([truck1, truck2, van1])

    leader = Fluent("leader", Vehicle)
    backup = Fluent("backup", Truck)
    spare = Fluent("spare", Van)
    problem.add_fluent(leader, default_initial_value=truck1)
    problem.add_fluent(backup, default_initial_value=truck2)
    problem.add_fluent(spare, default_initial_value=van1)

    done_match = Fluent("done_match", BoolType())
    done_differ = Fluent("done_differ", BoolType())
    problem.add_fluent(done_match, default_initial_value=False)
    problem.add_fluent(done_differ, default_initial_value=False)

    set_leader = InstantaneousAction("set_leader", v=Vehicle)
    set_leader.add_effect(leader, set_leader.parameter("v"))

    set_backup = InstantaneousAction("set_backup", t=Truck)
    set_backup.add_effect(backup, set_backup.parameter("t"))

    # `spare` has a single possible value (Van has only one object), but it
    # still needs a writer -- an effect-less fluent risks being folded into a
    # static constant at compile time, which would turn every `spare`
    # comparison below into the already-covered `fluent == object` case
    # instead of the fluent-vs-fluent one this problem exists to exercise.
    set_spare = InstantaneousAction("set_spare", vn=Van)
    set_spare.add_effect(spare, set_spare.parameter("vn"))

    match = InstantaneousAction("match")
    match.add_precondition(Equals(leader, backup))
    match.add_effect(done_match, True)

    differ = InstantaneousAction("differ")
    differ.add_precondition(Not(Equals(leader, spare)))
    differ.add_effect(done_differ, True)

    impossible = InstantaneousAction("impossible")
    impossible.add_precondition(Equals(backup, spare))
    impossible.add_effect(done_match, True)

    problem.add_actions([set_leader, set_backup, set_spare, match, differ, impossible])

    problem.add_goal(done_match)
    problem.add_goal(done_differ)

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


def get_problem_duration_fluent_relevance() -> Problem:
    """A two-action temporal problem where `tune`'s sole role is to set
    `charge`, a fluent read *only* by `run`'s duration bound -- no condition,
    effect value, or goal anywhere else mentions `charge`. `tune` is only
    reachable as relevant via the goal (`done`) -> `run` (writes `done`) ->
    `run`'s duration bound -> `charge` -> `tune` (writes `charge`) chain, so
    it isolates whether relevance analysis's backward walk follows duration
    dependencies.

    `run`'s duration is constrained to `[10, charge]`: at the initial
    `charge == 0` this is an empty (lower > upper) interval, so the temporal
    network is inconsistent and `run` can never open -- the goal is only
    reachable by running `tune` first to raise `charge` to `10`. If `tune` is
    wrongly pruned as irrelevant, the problem becomes genuinely unsolvable
    rather than merely producing a worse plan.
    """
    problem = Problem("duration_fluent_relevance")

    charge = Fluent("charge", RealType(0, 100))
    done = Fluent("done")

    tune = InstantaneousAction("tune")
    tune.add_effect(charge, 10)

    run = DurativeAction("run")
    run.set_duration_constraint(DurationInterval(Int(10), charge()))
    run.add_effect(EndTiming(), done, True)

    problem.add_fluent(charge, default_initial_value=0)
    problem.add_fluent(done, default_initial_value=False)
    problem.add_action(tune)
    problem.add_action(run)

    problem.add_goal(done)

    return problem


def get_problem_bookkeeping_fluent_classical() -> Problem:
    """A single-action, non-temporal problem with a pure bookkeeping fluent
    (`cost`, bumped by `turn_on`'s own increase effect and read nowhere else)
    alongside the fluent that actually drives the goal (`ready`). Isolates
    `Encoder._compute_relevant_fluents`'s self-reference exclusion: the
    desugared `cost := cost + 1` assignment reads `cost` on its own
    right-hand side, so a one-step "everything an effect's RHS reads is
    relevant" rule would have this fluent trivially mark itself relevant and
    nothing would ever be compacted away, hiding the very bug this analysis
    exists to prevent. Non-temporal, so it also exercises the plain
    `not is_temporal` dedup path end to end.
    """
    problem = Problem("bookkeeping_fluent_classical")

    ready = Fluent("ready")
    cost = Fluent("cost", IntType())

    turn_on = InstantaneousAction("turn_on")
    turn_on.add_precondition(Not(ready))
    turn_on.add_effect(ready, True)
    turn_on.add_increase_effect(cost, 1)

    problem.add_fluent(ready, default_initial_value=False)
    problem.add_fluent(cost, default_initial_value=0)
    problem.add_action(turn_on)

    problem.add_goal(ready)

    return problem


def get_problem_bookkeeping_fluent_temporal() -> Problem:
    """The temporal counterpart to `get_problem_bookkeeping_fluent_classical`:
    the same self-referencing bookkeeping fluent (`tcost`, bumped by `run`'s
    own start effect), but on a `DurativeAction` whose duration bound reads
    `charge` -- exercising the branch of `_compute_relevant_fluents`
    that pulls duration-bound fluents into the relevant set, mirroring
    `_compute_relevant_actions`'s precedent (see
    `get_problem_duration_fluent_relevance` above).

    `tune` writing `charge` is load-bearing, not incidental: a fluent with
    no writer anywhere is constant-folded by UP's `GROUNDING` compiler
    straight into the duration bound (the ground action ends up with a
    literal `[1, 10]`, no fluent reference left at all), which would make
    `charge` correctly but uninterestingly absent from the ground problem's
    fluents entirely, rather than exercising the duration-bound-read branch.

    Being temporal, it also covers compaction on a problem whose dedup regime
    (`is_temporal and not weak_equality`) dedups only states with no durative
    action in progress (`DedupPolicy.TODO_EMPTY`) -- compaction is
    unconditional on `is_temporal`/`weak_equality`, unlike the dedup-only
    reduction it replaced.
    """
    problem = Problem("bookkeeping_fluent_temporal")

    charge = Fluent("charge", RealType(0, 100))
    done = Fluent("done")
    tcost = Fluent("tcost", IntType())

    tune = InstantaneousAction("tune")
    tune.add_effect(charge, 10)

    run = DurativeAction("run")
    run.set_duration_constraint(DurationInterval(Int(1), charge()))
    run.add_increase_effect(StartTiming(), tcost, 1)
    run.add_effect(EndTiming(), done, True)

    problem.add_fluent(charge, default_initial_value=0)
    problem.add_fluent(done, default_initial_value=False)
    problem.add_fluent(tcost, default_initial_value=0)
    problem.add_action(tune)
    problem.add_action(run)

    problem.add_goal(done)

    return problem


def get_problem_bookkeeping_fluent_transitive() -> Problem:
    """A single-action, non-temporal problem with a *chain* of bookkeeping
    fluents, alongside the fluent that actually drives the goal (`ready`):

    - `counter` is bumped by `turn_on`'s own increase effect (self-referencing
      RHS, like `get_problem_bookkeeping_fluent_classical`'s `cost`).
    - `log` is assigned `counter`'s value, and is read by nothing else.

    Neither fluent is read by any condition, goal, or duration bound, so both
    are irrelevant to search outcome. But a one-step "everything read by an
    effect's RHS is relevant" rule (the reduction's pre-fixpoint shape) would
    incorrectly keep `counter`: `log`'s RHS reads it, and `log != counter` so
    a same-fluent self-reference exclusion alone doesn't filter it out. Only
    the least-fixpoint closure -- an effect's RHS matters only if the fluent
    it writes matters -- correctly drops both, since `log` itself is never
    seeded as relevant. Isolates that transitive case, distinct from
    `get_problem_bookkeeping_fluent_classical`'s direct self-reference.

    Both `counter` and `log` need a writer, like
    `get_problem_bookkeeping_fluent_temporal`'s `charge`: a fluent with no
    writer anywhere is constant-folded away entirely by UP's `GROUNDING`
    compiler / `Simplifier`, rather than exercising the closure.
    """
    problem = Problem("bookkeeping_fluent_transitive")

    ready = Fluent("ready")
    counter = Fluent("counter", IntType())
    log = Fluent("log", IntType())

    turn_on = InstantaneousAction("turn_on")
    turn_on.add_precondition(Not(ready))
    turn_on.add_effect(ready, True)
    turn_on.add_effect(log, counter)
    turn_on.add_increase_effect(counter, 1)

    problem.add_fluent(ready, default_initial_value=False)
    problem.add_fluent(counter, default_initial_value=0)
    problem.add_fluent(log, default_initial_value=0)
    problem.add_action(turn_on)

    problem.add_goal(ready)

    return problem


def get_problem_temporal_no_start_event() -> Problem:
    """A temporal problem with:
    - `noop`, a durative action with no conditions and no effects at all, so
      it has no events of its own (an edge case for the encoder's event list,
      which must never index into it as if it were non-empty).
    - `finish`, a durative action whose only condition/effect is `at end`, so
      it has no event of its own at the start timepoint.

    `finish` reads `d` in its duration, and `setup` -- the only way to make
    `finish`'s end condition `ready` true -- writes `d` at its end. So the
    length of `finish` is decided by whether it starts before or after
    `setup` ends, and any encoding that does not tie `finish`'s start to the
    state its duration was read from produces an invalid plan.
    """
    problem = Problem("temporal_no_start_event")

    d = Fluent("d", IntType())
    ready = Fluent("ready")
    done = Fluent("done")

    noop = DurativeAction("noop")
    noop.set_fixed_duration(2)

    setup = DurativeAction("setup")
    setup.set_fixed_duration(20)
    setup.add_effect(EndTiming(), d, 10)
    setup.add_effect(EndTiming(), ready, True)

    finish = DurativeAction("finish")
    finish.set_fixed_duration(d())
    finish.add_condition(EndTiming(), ready)
    finish.add_effect(EndTiming(), done, True)

    problem.add_fluent(d, default_initial_value=1)
    problem.add_fluent(ready, default_initial_value=False)
    problem.add_fluent(done, default_initial_value=False)
    problem.add_action(noop)
    problem.add_action(setup)
    problem.add_action(finish)

    problem.add_goal(done)

    return problem


def get_problem_temporal_commuting() -> Problem:
    """Three independent durative actions, each runnable once (`not done_x` at
    start, `done_x` at end), goal: all three done. Every ordering of the
    completed actions reaches the same no-action-in-progress state, so this is
    the minimal problem where `DedupPolicy.TODO_EMPTY` (temporal, no
    `weak_equality`, no deadline) prunes something."""
    problem = Problem("temporal_commuting")
    for name, duration in [("a", 2), ("b", 3), ("c", 5)]:
        done = Fluent(f"done_{name}")
        problem.add_fluent(done, default_initial_value=False)
        run = DurativeAction(f"run_{name}")
        run.set_fixed_duration(duration)
        run.add_condition(StartTiming(), Not(done))
        run.add_effect(EndTiming(), done, True)
        problem.add_action(run)
        problem.add_goal(done)
    return problem


def get_problem_temporal_token_chain() -> Problem:
    """`a` (duration 4) holds a token for its whole run and enables `b`
    halfway through; `c` (duration 1) needs the token too. Goal: `b` and `c`
    done. Running `c` before or after `a` reaches the same no-action-in-progress
    states at times at least `epsilon` apart, which is what
    `DedupPolicy.DOMINANCE` (`makespan_dominates`) prunes under a deadline."""
    problem = Problem("temporal_token_chain")
    x, y, z, tok = Fluent("x"), Fluent("y"), Fluent("z"), Fluent("tok")
    for f in (x, y, z):
        problem.add_fluent(f, default_initial_value=False)
    problem.add_fluent(tok, default_initial_value=True)
    a = DurativeAction("a")
    a.set_fixed_duration(4)
    a.add_condition(StartTiming(), tok)
    a.add_effect(StartTiming(), tok, False)
    a.add_effect(StartTiming(2), x, True)
    a.add_effect(EndTiming(), tok, True)
    b = DurativeAction("b")
    b.set_fixed_duration(3)
    b.add_condition(StartTiming(), x)
    b.add_effect(EndTiming(), y, True)
    c = DurativeAction("c")
    c.set_fixed_duration(1)
    c.add_condition(StartTiming(), tok)
    c.add_effect(StartTiming(), tok, False)
    c.add_effect(EndTiming(), tok, True)
    c.add_effect(EndTiming(), z, True)
    for action in (a, b, c):
        problem.add_action(action)
    problem.add_goal(y)
    problem.add_goal(z)
    return problem


def get_problem_quantified_false_precondition() -> Problem:
    """A problem whose single action's precondition is quantified, and whose
    grounder leaves most groundings' `Exists` statically unsatisfiable.

    `move_via(l_from, l_to)` requires a midpoint `mid` connected to both ends
    via the static `connected` fluent -- `Exists(mid, connected(l_from, mid)
    and connected(mid, l_to))`. `connected` only forms the chain `l1 -> l2 ->
    l3 -> l4`, so only the two-hop groundings `(l1, l3)` and `(l2, l4)` have a
    real midpoint; the other 10 of 12 groundings' precondition reduces to the
    constant `false` once quantifiers are removed -- but `Simplifier` alone
    (UP's `walk_exists`/`walk_forall`) can never fold a surviving quantified
    variable to a bool constant, so a caller that simplifies without first
    removing quantifiers reports all 12 groundings applicable instead of 2
    (see `Encoder._build_events`'s applicability check, which must route
    through `Encoder._normalize_expression` -- quantifier removal + simplify
    -- rather than `Encoder._simplifier.simplify` alone).
    """
    Location = UserType("Location")
    connected = Fluent("connected", BoolType(), l1=Location, l2=Location)
    at = Fluent("at", BoolType(), l=Location)

    move_via = InstantaneousAction("move_via", l_from=Location, l_to=Location)
    l_from = move_via.parameter("l_from")
    l_to = move_via.parameter("l_to")
    mid = Variable("mid", Location)
    move_via.add_precondition(at(l_from))
    move_via.add_precondition(Not(Equals(l_from, l_to)))
    move_via.add_precondition(
        Exists(And(connected(l_from, mid), connected(mid, l_to)), mid)
    )
    move_via.add_effect(at(l_from), False)
    move_via.add_effect(at(l_to), True)

    problem = Problem("quantified_false_precondition")
    problem.add_fluent(connected, default_initial_value=False)
    problem.add_fluent(at, default_initial_value=False)
    problem.add_action(move_via)

    l1 = Object("l1", Location)
    l2 = Object("l2", Location)
    l3 = Object("l3", Location)
    l4 = Object("l4", Location)
    problem.add_objects([l1, l2, l3, l4])

    problem.set_initial_value(at(l1), True)
    problem.set_initial_value(connected(l1, l2), True)
    problem.set_initial_value(connected(l2, l3), True)
    problem.set_initial_value(connected(l3, l4), True)

    problem.add_goal(at(l3))

    return problem


def get_problem_temporal_condition_before_start() -> Problem:
    """A malformed temporal problem the encoder must reject.

    `a` mixes an intermediate condition from start with one from end, so the
    encoder folds the end-relative timings onto the start. With a duration of
    2, `end - 5` folds to `start - 3`: a condition required before the action
    begins. Left alone it would sort ahead of `a`'s start event and quietly
    break the invariant that the first event is the action's start.
    """
    problem = Problem("temporal_condition_before_start")

    ok = Fluent("ok")
    done = Fluent("done")

    a = DurativeAction("a")
    a.set_fixed_duration(2)
    a.add_condition(StartTiming() + 1, ok)
    a.add_condition(EndTiming() - 5, ok)
    a.add_effect(EndTiming(), done, True)

    # Unreachable but present so the grounder can't prove `ok` static and
    # simplify away the (always-true) conditions above before the encoder
    # ever sees them.
    unreachable = InstantaneousAction("unreachable")
    unreachable.add_precondition(done)
    unreachable.add_effect(ok, False)

    problem.add_fluent(ok, default_initial_value=True)
    problem.add_fluent(done, default_initial_value=False)
    problem.add_action(a)
    problem.add_action(unreachable)

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
    # Or(f(p1), g(p4)) tells p1 and p4 apart, but must not affect p2/p3,
    # which only appear in h(p2)/h(p3) and remain genuinely symmetric.
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
    # Equals(fl(a), fr(b)) must be compared as a whole, not decomposed into
    # independent "fl(a) must hold"/"fr(b) must hold" literals (which would be
    # a different, wrong constraint); swapped, it is not a goal conjunct.
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


def get_problem_goal_int_argument(symmetric: bool) -> Problem:
    T = UserType("T")
    o1, o2 = Object("o1", T), Object("o2", T)
    f = Fluent("f", BoolType(), o=T, k=IntType(1, 2))
    step = Fluent("step", IntType())

    # Only one `do2` can ever run, so with the asymmetric goal it must be
    # `do2(o2)`, and it must come first.
    do2 = InstantaneousAction("do2", o=T)
    do2.add_precondition(Equals(step, 0))
    do2.add_effect(f(do2.parameter("o"), 2), True)
    do2.add_effect(step, 1)

    do1 = InstantaneousAction("do1", o=T)
    do1.add_precondition(GE(step, 1))
    do1.add_effect(f(do1.parameter("o"), 1), True)

    problem = Problem(f"goal_int_argument_{'' if symmetric else 'a'}symmetric")
    problem.add_objects([o1, o2])
    problem.add_fluent(f, default_initial_value=False)
    problem.add_fluent(step, default_initial_value=0)
    problem.add_actions([do1, do2])
    # The goals of `o1` and `o2` differ only in `f`'s int argument.
    problem.add_goal(f(o1, 1))
    problem.add_goal(f(o2, 1))
    if not symmetric:
        problem.add_goal(f(o2, 2))
    return problem


def _get_problem_refuel_trucks(name: str, n_trucks: int) -> Problem:
    """Trucks `t1`, `t2`, ... with no fuel, and an action adding one unit of
    fuel to a truck. The caller adds the goal."""
    T = UserType("Truck")
    fuel = Fluent("fuel", IntType(), t=T)
    refuel = InstantaneousAction("refuel", t=T)
    refuel.add_increase_effect(fuel(refuel.parameter("t")), 1)

    problem = Problem(name)
    problem.add_objects([Object(f"t{i}", T) for i in range(1, n_trucks + 1)])
    problem.add_fluent(fuel, default_initial_value=0)
    problem.add_action(refuel)
    return problem


def get_problem_goal_numeric_symmetry() -> Problem:
    problem = _get_problem_refuel_trucks("goal_numeric_symmetry", 3)
    fuel = problem.fluent("fuel")
    for t in problem.all_objects:
        problem.add_goal(GE(fuel(t), 2))
    return problem


def get_problem_goal_numeric_asymmetry() -> Problem:
    problem = _get_problem_refuel_trucks("goal_numeric_asymmetry", 2)
    fuel = problem.fluent("fuel")
    problem.add_goal(GE(fuel(problem.object("t1")), 2))
    problem.add_goal(GE(fuel(problem.object("t2")), 3))
    return problem


def get_problem_goal_commutative_sum_symmetry() -> Problem:
    problem = _get_problem_refuel_trucks("goal_commutative_sum_symmetry", 2)
    fuel = problem.fluent("fuel")
    # Swapping `t1` and `t2` turns the sum into `fuel(t2) + fuel(t1)`.
    problem.add_goal(
        GE(Plus(fuel(problem.object("t1")), fuel(problem.object("t2"))), 3)
    )
    return problem


def get_problem_goal_repeated_conjunct_symmetry(reordered: bool) -> Problem:
    """`t1` and `t2` are interchangeable, but the goal states `t1`'s
    condition twice: identically, or with the sum's arguments reordered."""
    suffix = "_reordered" if reordered else ""
    problem = _get_problem_refuel_trucks(f"goal_repeated_conjunct{suffix}", 2)
    fuel = problem.fluent("fuel")
    bonus = Fluent("bonus", IntType())
    problem.add_fluent(bonus, default_initial_value=0)
    t1, t2 = problem.object("t1"), problem.object("t2")
    problem.add_goal(GE(Plus(fuel(t1), bonus), 1))
    if reordered:
        problem.add_goal(GE(Plus(bonus, fuel(t1)), 1))
    else:
        problem.add_goal(GE(Plus(fuel(t1), bonus), 1))
    problem.add_goal(GE(Plus(fuel(t2), bonus), 1))
    return problem


def get_problem_goal_disjunction_symmetry() -> Problem:
    Truck = UserType("Truck")
    Location = UserType("Location")
    t1, t2 = Object("t1", Truck), Object("t2", Truck)
    l0, l1 = Object("l0", Location), Object("l1", Location)
    at = Fluent("at", BoolType(), t=Truck, l=Location)

    move = InstantaneousAction("move", t=Truck, src=Location, dst=Location)
    t, src, dst = move.parameters
    move.add_precondition(at(t, src))
    move.add_precondition(Not(Equals(src, dst)))
    move.add_effect(at(t, src), False)
    move.add_effect(at(t, dst), True)

    problem = Problem("goal_disjunction_symmetry")
    problem.add_objects([t1, t2, l0, l1])
    problem.add_fluent(at, default_initial_value=False)
    problem.add_action(move)
    problem.set_initial_value(at(t1, l0), True)
    problem.set_initial_value(at(t2, l0), True)
    problem.add_goal(Or(at(t1, l1), at(t2, l1)))
    return problem


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


def get_problem_if_object_effect() -> Problem:
    """An interpreted function returning an object, used as an effect
    assignment value."""

    Loc = UserType("Loc")
    l1 = Object("l1", Loc)
    l2 = Object("l2", Loc)

    def choose_location(c):
        return l2 if c >= 1 else l1

    IF_choose = InterpretedFunction(
        "choose_location", Loc, OrderedDict(c=IntType()), choose_location
    )

    counter = Fluent("counter", IntType())
    at = Fluent("at", Loc)

    act = InstantaneousAction("act")
    act.add_precondition(LT(counter, 5))
    act.add_effect(counter, Plus(counter, 1))
    act.add_effect(at, IF_choose(counter))

    problem = Problem("if_object_effect")
    problem.add_fluent(counter)
    problem.add_fluent(at)
    problem.add_object(l1)
    problem.add_object(l2)
    problem.add_action(act)
    problem.set_initial_value(counter, 0)
    problem.set_initial_value(at, l1)
    problem.add_goal(Equals(at, l2))
    return problem


def get_problem_if_object_argument_and_return() -> Problem:
    """An interpreted function returning an object (`next_loc`) and one
    taking an object-typed argument (`is_final`), chained through the same
    object-typed fluent -- exercises both directions of the argument/
    return-unwrapping wrapper in `Converter.walk_interpreted_function_exp`
    together (`is_final` alone already covers plain object-argument
    dispatch: its result depends on the argument's identity, not a
    constant, so it fails loudly if the argument is ever passed through
    unresolved). Three objects so reaching the goal takes two real state
    transitions, not one degenerate step."""
    Loc = UserType("Loc")
    l1 = Object("l1", Loc)
    l2 = Object("l2", Loc)
    l3 = Object("l3", Loc)

    def next_loc(loc):
        return {l1: l2, l2: l3, l3: l3}[loc]

    def is_final(loc):
        return loc == l3

    IF_next = InterpretedFunction("next_loc", Loc, OrderedDict(loc=Loc), next_loc)
    IF_final = InterpretedFunction(
        "is_final", BoolType(), OrderedDict(loc=Loc), is_final
    )

    at = Fluent("at", Loc)
    act = InstantaneousAction("act")
    act.add_precondition(Not(IF_final(at)))
    act.add_effect(at, IF_next(at))

    problem = Problem("if_object_argument_and_return")
    problem.add_fluent(at)
    problem.add_object(l1)
    problem.add_object(l2)
    problem.add_object(l3)
    problem.add_action(act)
    problem.set_initial_value(at, l1)
    problem.add_goal(Equals(at, l3))
    return problem


def get_problem_if_undefined_initial_numeric() -> Problem:
    """A fluent left undefined for one object, read both inside an
    interpreted-function precondition and an interpreted-function effect
    value -- adapted from UP's `interpreted_functions_undef_numeric` example
    (dropping the `choose`/`use_chosen` action: `choose_if()` returns an
    object, and `use_chosen` feeds that object into `undef_value(...)` to
    pick *which fluent instance to read* -- a further capability, dynamic
    fluent-parameter resolution from a runtime-computed object, that's
    separate from and harder than a plain object-valued effect (see
    `get_problem_if_object_effect`) and still out of scope). The
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
    The duration here is a plain constant; see `get_problem_if_duration` for
    interpreted functions used inside a duration itself."""

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


def get_problem_if_duration() -> Problem:
    """A durative action whose *duration* is an interpreted function --
    modelled on UP's `interpreted_functions_in_durative_start_effects`
    example, now that UP's Grounder correctly substitutes into (without
    folding) an interpreted function used in a duration bound.

    `battery` is written by `charge`'s own start effect, so it isn't a
    static fluent -- the encoder's `Simplifier` would otherwise constant-fold
    `charge_time_if(battery)` away during grounding, and the duration would
    reach the search space as a plain literal, silently defeating the point
    of this test.

    The end-only condition (with no matching overall condition) makes
    `charge` fail `Encoder._compute_compression_safe_actions`, so solving
    this exercises the real temporal search path (`_open_action`'s duration
    evaluation), not the `TimedToSequential` recompile, which would discard
    the duration and re-derive it in `build_plan` instead."""

    def charge_time(level):
        return 10 - level

    IF_charge_time = InterpretedFunction(
        "charge_time", IntType(), OrderedDict(level=IntType()), charge_time
    )

    battery = Fluent("battery", IntType())
    charged = Fluent("charged")

    charge = DurativeAction("charge")
    charge.set_fixed_duration(IF_charge_time(battery))
    charge.add_effect(StartTiming(), battery, battery - 2)
    charge.add_condition(EndTiming(), GE(battery, 0))
    charge.add_effect(EndTiming(), charged, True)

    problem = Problem("if_duration")
    problem.add_fluent(battery)
    problem.add_fluent(charged, default_initial_value=False)
    problem.add_action(charge)
    problem.set_initial_value(battery, 4)
    problem.add_goal(charged)
    return problem


def get_problem_if_numeric_symmetry_retained() -> Problem:
    """A numeric-only interpreted function (its signature and return type
    are both `IntType()`) applied to symmetric objects. An IF that only
    ever sees numeric/boolean argument values returns the same result on
    both sides of a genuine object swap -- a real automorphism preserves
    fluent values, and numbers aren't permuted by the swap -- so this
    pattern must not taint the objects it's applied to: `{s1, s2}` stays a
    legitimate equivalence class."""
    Slot = UserType("Slot")
    s1 = Object("s1", Slot)
    s2 = Object("s2", Slot)

    def boost(n):
        return n + 1

    IF_boost = InterpretedFunction("boost", IntType(), OrderedDict(n=IntType()), boost)

    level = Fluent("level", IntType(), s=Slot)
    bump = InstantaneousAction("bump", s=Slot)
    s = bump.parameter("s")
    bump.add_precondition(LT(level(s), 3))
    bump.add_effect(level(s), IF_boost(level(s)))

    problem = Problem("if_numeric_symmetry_retained")
    problem.add_fluent(level)
    problem.add_objects([s1, s2])
    problem.add_action(bump)
    problem.set_initial_value(level(s1), 0)
    problem.set_initial_value(level(s2), 0)
    problem.add_goal(Equals(level(s1), 1))
    problem.add_goal(Equals(level(s2), 1))
    return problem


def get_problem_if_object_argument_symmetry_unsound() -> Problem:
    """Soundness regression: an interpreted function taking an object-typed
    argument distinguishes `l1`/`l2` in a way the initial state and goal
    alone cannot see -- the distinguishing logic lives entirely inside the
    IF's Python closure, so neither `l1` nor `l2` appears as a literal
    `Object` constant anywhere in the lifted action schema, and a symmetry
    breaker that only looks at `_extract_domain_objects`, the goal and the
    initial state wrongly groups `{l1, l2}`.

    That wrong grouping makes any `l2`-using action require some
    `l1`-using action to already be on the plan prefix. But `enter(l1)` can
    never fire (`is_open(l1)` is always `False`), so nothing that uses `l1`
    is ever applicable, and the goal becomes unreachable even though
    `enter(l2); finish(l2)` is a trivial valid plan. A sound symmetry
    breaker must recognise that `is_open`'s object-typed argument means
    `l1`/`l2` cannot be treated as equivalent."""
    Loc = UserType("Loc")
    l1 = Object("l1", Loc)
    l2 = Object("l2", Loc)

    def is_open(loc):
        return loc == l2

    IF_is_open = InterpretedFunction(
        "is_open", BoolType(), OrderedDict(loc=Loc), is_open
    )

    visited = Fluent("visited", BoolType(), x=Loc)
    done = Fluent("done", BoolType())

    enter = InstantaneousAction("enter", x=Loc)
    x = enter.parameter("x")
    enter.add_precondition(And(IF_is_open(x), Not(visited(x))))
    enter.add_effect(visited(x), True)

    finish = InstantaneousAction("finish", x=Loc)
    y = finish.parameter("x")
    finish.add_precondition(visited(y))
    finish.add_effect(done, True)

    problem = Problem("if_object_argument_symmetry_unsound")
    problem.add_fluent(visited, default_initial_value=False)
    problem.add_fluent(done, default_initial_value=False)
    problem.add_objects([l1, l2])
    problem.add_action(enter)
    problem.add_action(finish)
    problem.add_goal(done)
    return problem


def get_problem_if_object_return_symmetry() -> Problem:
    """An object-returning interpreted function must taint every object of
    its return type -- the value it can produce is a dynamically-computed
    domain constant invisible to `_extract_domain_objects` -- while leaving
    a second, unrelated type's genuine symmetry untouched: `{l1, l2}` must
    become singletons, but `{i1, i2}` (never seen by any IF) must stay
    grouped, proving the taint is scoped to the IF's actual types rather
    than a blanket effect on the whole problem."""
    Loc = UserType("Loc")
    l1 = Object("l1", Loc)
    l2 = Object("l2", Loc)
    Item = UserType("Item")
    i1 = Object("i1", Item)
    i2 = Object("i2", Item)

    def pick(n):
        return l2 if n >= 1 else l1

    IF_pick = InterpretedFunction("pick", Loc, OrderedDict(n=IntType()), pick)

    counter = Fluent("counter", IntType())
    at = Fluent("at", Loc)
    held = Fluent("held", BoolType(), item=Item)

    move = InstantaneousAction("move")
    move.add_precondition(LT(counter, 5))
    move.add_effect(counter, Plus(counter, 1))
    move.add_effect(at, IF_pick(counter))

    grab = InstantaneousAction("grab", item=Item)
    grab.add_effect(held(grab.parameter("item")), True)

    problem = Problem("if_object_return_symmetry")
    problem.add_fluent(counter)
    problem.add_fluent(at)
    problem.add_fluent(held, default_initial_value=False)
    problem.add_objects([l1, l2, i1, i2])
    problem.add_action(move)
    problem.add_action(grab)
    problem.set_initial_value(counter, 0)
    problem.set_initial_value(at, l1)
    return problem


def get_problem_if_metric_action_cost_object_argument_taint() -> Problem:
    """Soundness regression for interpreted functions reachable only through
    a quality metric: an IF call inside a `MinimizeActionCosts` cost
    expression can distinguish `l1`/`l2` via the action's own object-typed
    parameter, even though neither the goal, the initial state, nor any
    action precondition/effect ever does. Mirrors
    `get_problem_if_object_argument_symmetry_unsound`, but the discriminating
    IF call sits entirely inside the metric instead of a precondition, so a
    tainted-object scan that skips quality metrics wrongly groups `{l1, l2}`
    as equivalent."""
    Loc = UserType("Loc")
    l1 = Object("l1", Loc)
    l2 = Object("l2", Loc)

    def bonus(loc):
        return 10 if loc == l2 else 0

    IF_bonus = InterpretedFunction("bonus", IntType(), OrderedDict(loc=Loc), bonus)

    visited = Fluent("visited", BoolType(), x=Loc)
    done = Fluent("done", BoolType())

    enter = InstantaneousAction("enter", x=Loc)
    x = enter.parameter("x")
    enter.add_precondition(Not(visited(x)))
    enter.add_effect(visited(x), True)
    enter.add_effect(done, True)

    problem = Problem("if_metric_action_cost_object_argument_taint")
    problem.add_fluent(visited, default_initial_value=False)
    problem.add_fluent(done, default_initial_value=False)
    problem.add_objects([l1, l2])
    problem.add_action(enter)
    problem.add_goal(done)
    problem.add_quality_metric(MinimizeActionCosts({enter: IF_bonus(x)}, default=0))
    return problem


def get_problem_if_conditions_and_effects() -> Problem:
    """Combines five placement gaps that would otherwise each need their own
    minimal problem, all in one durative action fired twice:

    - IF in a durative *overall* condition (`ClosedTimeInterval`) -- every
      other temporal IF problem here uses a start-only or end-only
      condition, checked once rather than held throughout.
    - IF as the value of a *timed* (non-start), boolean-valued effect --
      every other temporal IF problem here only assigns an IF's value at
      `StartTiming()`. `Encoder._extract_interpreted_function_tainted_
      objects` scans timed effects for IF usages separately from start
      effects.
    - IF as the value of an `increase` numeric effect -- every other
      numeric-effect IF problem here uses a plain `assign`; `increase`/
      `decrease` route through a different code path (`Encoder`'s effect
      classification, `_to_linear_polynomial` in the delete-relaxation
      heuristics).
    - IF used directly in the goal -- every other IF problem here only puts
      an IF in a precondition, effect, or duration.
    - A quality metric, so this flows into `ANYTIME_PROBLEMS`
      (`test_engine.py`'s `_build_anytime_problems` filters on
      `len(quality_metrics) == 1`), letting `test_anytime_planner` exercise
      TamerLite's shared `if_cache` across anytime re-solves on a real IF
      problem, not just the synthetic `test_converter_shares_if_cache_
      across_converters_with_different_object_tables`. The IF stays out of
      the metric itself: `TamerLite.supported_kind` doesn't declare any
      interpreted-functions-in-actions-cost feature.

    `flow` needs two non-overlapping occurrences to reach the goal (`total`
    only reaches 4 after two `+2` increases), which is also what exercises
    the overall condition across a real duration rather than at a single
    instant. (Not paired with a timed *goal*: TamerLite's `supported_kind`
    doesn't declare `TIMED_GOALS` at all -- a pre-existing gap unrelated to
    interpreted functions -- so `add_timed_goal` would make `supports()`
    reject the problem outright.)"""

    def is_open(level):
        return level > 0

    IF_is_open = InterpretedFunction(
        "is_open", BoolType(), OrderedDict(level=IntType()), is_open
    )

    def flag_from(level):
        return level >= 2

    IF_flag = InterpretedFunction(
        "flag_from", BoolType(), OrderedDict(level=IntType()), flag_from
    )

    def step(x):
        return x + 2

    IF_step = InterpretedFunction("step", IntType(), OrderedDict(x=IntType()), step)

    def reached(x):
        return x >= 4

    IF_reached = InterpretedFunction(
        "reached", BoolType(), OrderedDict(x=IntType()), reached
    )

    valve = Fluent("valve", IntType())
    level = Fluent("level", IntType())
    ready = Fluent("ready")
    total = Fluent("total", IntType())

    flow = DurativeAction("flow")
    flow.set_fixed_duration(2)
    flow.add_condition(
        ClosedTimeInterval(StartTiming(), EndTiming()), IF_is_open(valve)
    )
    flow.add_effect(StartTiming(), level, 2)
    flow.add_effect(EndTiming(), ready, IF_flag(level))
    flow.add_increase_effect(EndTiming(), total, IF_step(total))

    problem = Problem("if_conditions_and_effects")
    problem.add_fluent(valve)
    problem.add_fluent(level)
    problem.add_fluent(ready, default_initial_value=False)
    problem.add_fluent(total)
    problem.add_action(flow)
    problem.set_initial_value(valve, 1)
    problem.set_initial_value(level, 0)
    problem.set_initial_value(total, 0)
    problem.add_goal(ready)
    problem.add_goal(IF_reached(total))
    # MinimizeMakespan, not MinimizeSequentialPlanLength: this is a
    # continuous-time (durative) problem, and no validator engine supports
    # CONTINUOUS_TIME and PLAN_LENGTH metrics together (only makespan) --
    # matching the existing get_problem_temporal_flight_minimize_makespan
    # pattern rather than get_problem_flight_minimize_plan_length's.
    problem.add_quality_metric(MinimizeMakespan())
    return problem


def get_problem_if_signature_shapes() -> Problem:
    """Combines three argument/return shapes that would otherwise each need
    their own minimal problem:

    - A real-typed return -- proves the solve path works at all (the
      specific integral-vs-fractional backend divergence is pinned
      separately by `test_interpreted_functions_real_return_backend_
      normalization`, which builds its own tiny model and doesn't need this
      problem to be solvable). `half` is deliberately never integral
      (`5/2`): an *integral* real result stored via a plain `assign` effect
      and then compared against a bare int goal literal genuinely doesn't
      solve -- the engine represents the fluent's value as
      `Rational(4, 1)`, which doesn't compare equal to the goal's `Int(4)`
      constant, only to a matching `Fraction` literal. That's exactly the
      trap this problem must not fall into just to look tidy.
    - A bool-typed argument -- the one argument shape not already reachable
      through a *supported* problem elsewhere. A zero-argument IF is already
      covered by UP's own `treasure_hunting_robot_simple` example
      (`check_treasure_map_if()`), and a real-typed argument by
      `if_reals_condition_effect_pizza` (`if_cut`/`if_available`, both over
      a `RealType()` parameter) -- both already arrive in `PROBLEMS` via
      `get_example_problems()`. Bool-typed arguments do appear in UP's
      examples too (`simple_always_false`, `wet_if`), but only inside the
      two that also require `BOUNDED_TYPES`, which TamerLite doesn't
      support.
    - An interpreted function nested directly inside another interpreted
      function's argument (`IF_outer(IF_inner(x))`) -- every other problem
      here only ever nests an IF's argument inside a plain fluent read,
      never inside another IF call.

    `n` and `active` are read once each by a static-looking IF call, but
    `active` is written by `flip` (so `IF_bonus(active)` isn't
    constant-folded away); `n` genuinely is static here since only its
    initial value matters for `to_real`, which is fine -- a real-typed
    return only needs to be *reached*, not re-derived from a written
    fluent."""

    def half(x):
        return Fraction(x, 2)

    IF_half = InterpretedFunction("half", RealType(), OrderedDict(x=IntType()), half)

    def bool_to_bonus(flag):
        return 2 if flag else 0

    IF_bonus = InterpretedFunction(
        "bool_to_bonus", IntType(), OrderedDict(flag=BoolType()), bool_to_bonus
    )

    def inner(x):
        return x + 1

    IF_inner = InterpretedFunction("inner", IntType(), OrderedDict(x=IntType()), inner)

    def outer(y):
        return y >= 5

    IF_outer = InterpretedFunction("outer", BoolType(), OrderedDict(y=IntType()), outer)

    n = Fluent("n", IntType())
    result = Fluent("result", RealType())
    active = Fluent("active", BoolType())
    bonus = Fluent("bonus", IntType())
    counter = Fluent("counter", IntType())

    to_real = InstantaneousAction("to_real")
    to_real.add_precondition(Equals(result, 0))
    to_real.add_effect(result, IF_half(n))

    flip = InstantaneousAction("flip")
    flip.add_effect(active, Not(active))

    apply_bonus = InstantaneousAction("apply_bonus")
    apply_bonus.add_effect(bonus, IF_bonus(active))

    advance = InstantaneousAction("advance")
    advance.add_precondition(Not(IF_outer(IF_inner(counter))))
    advance.add_effect(counter, Plus(counter, 1))

    problem = Problem("if_signature_shapes")
    problem.add_fluent(n)
    problem.add_fluent(result)
    problem.add_fluent(active)
    problem.add_fluent(bonus)
    problem.add_fluent(counter)
    problem.add_action(to_real)
    problem.add_action(flip)
    problem.add_action(apply_bonus)
    problem.add_action(advance)
    problem.set_initial_value(n, 5)
    problem.set_initial_value(result, 0)
    problem.set_initial_value(active, False)
    problem.set_initial_value(bonus, 0)
    problem.set_initial_value(counter, 0)
    problem.add_goal(Equals(result, Fraction(5, 2)))
    problem.add_goal(Equals(bonus, 2))
    problem.add_goal(Equals(counter, 4))
    return problem


def get_problem_if_hierarchical_type_argument() -> Problem:
    """An interpreted function taking an argument typed as a *supertype*, in
    a problem with subtypes of it. `Encoder._extract_interpreted_function_
    tainted_objects` taints "every object of the argument's declared type" --
    this checks that taint also reaches subtype instances (`v1`: `Van`, a
    subtype of `Vehicle`), not just objects declared exactly as the
    supertype."""
    Vehicle = UserType("Vehicle")
    Van = UserType("Van", Vehicle)
    c1 = Object("c1", Vehicle)
    v1 = Object("v1", Van)

    def is_heavy(vehicle):
        return vehicle == v1

    IF_is_heavy = InterpretedFunction(
        "is_heavy", BoolType(), OrderedDict(vehicle=Vehicle), is_heavy
    )

    at_depot = Fluent("at_depot", BoolType(), v=Vehicle)
    dispatched = Fluent("dispatched", BoolType(), v=Vehicle)

    dispatch = InstantaneousAction("dispatch", v=Vehicle)
    v = dispatch.parameter("v")
    dispatch.add_precondition(And(at_depot(v), Not(IF_is_heavy(v))))
    dispatch.add_effect(dispatched(v), True)

    problem = Problem("if_hierarchical_type_argument")
    problem.add_fluent(at_depot, default_initial_value=True)
    problem.add_fluent(dispatched, default_initial_value=False)
    problem.add_objects([c1, v1])
    problem.add_action(dispatch)
    problem.add_goal(dispatched(c1))
    return problem


def get_problem_if_counting_chain(name: str, calls: list) -> Problem:
    """A tiny IF-gated counter problem: the only action, `inc`, is guarded by
    a boolean interpreted function that logs every `n` it's called with (into
    the caller-supplied `calls`) and always returns `n < 3`; the goal needs
    exactly three `inc`s, so this is both the smallest problem with a
    deterministic minimal plan length (for an anytime improvement loop to
    exhaust in one iteration) and a way to observe exactly which fluent
    values the IF was evaluated at. Unlike the other `get_problem_if_*`
    generators here, `name` is caller-supplied rather than fixed, since
    interleaving/warm-cache tests need several distinctly-named instances."""

    def allow(n):
        calls.append(n)
        return n < 3

    IF_allow = InterpretedFunction("allow", BoolType(), OrderedDict(n=IntType()), allow)
    n = Fluent("n", IntType())
    inc = InstantaneousAction("inc")
    inc.add_precondition(IF_allow(n))
    inc.add_effect(n, Plus(n, 1))
    problem = Problem(name)
    problem.add_fluent(n, default_initial_value=0)
    problem.add_action(inc)
    problem.add_goal(GE(n, 3))
    return problem


def get_problem_metric_object_taint() -> Problem:
    T = UserType("T")
    t1, t2 = Object("t1", T), Object("t2", T)
    fuel = Fluent("fuel", IntType(), t=T)
    done = Fluent("done", BoolType())

    refuel = InstantaneousAction("refuel", t=T)
    refuel.add_increase_effect(fuel(refuel.parameter("t")), 1)
    refuel.add_effect(done, True)

    problem = Problem("metric_object_taint")
    problem.add_objects([t1, t2])
    problem.add_fluent(fuel, default_initial_value=0)
    problem.add_fluent(done, default_initial_value=False)
    problem.add_action(refuel)
    problem.add_goal(done)
    # The metric is the only thing that tells `t1` and `t2` apart.
    problem.add_quality_metric(MinimizeExpressionOnFinalState(fuel(t1)))
    return problem


def get_problem_metric_action_cost_object(in_default: bool) -> Problem:
    T = UserType("T")
    t1, t2 = Object("t1", T), Object("t2", T)
    fuel = Fluent("fuel", IntType(), t=T)
    n = Fluent("n", IntType())

    refuel = InstantaneousAction("refuel", t=T)
    refuel.add_precondition(LT(n, 2))
    refuel.add_increase_effect(fuel(refuel.parameter("t")), 1)
    refuel.add_increase_effect(n, 1)

    problem = Problem(
        "metric_default_cost_object" if in_default else "metric_explicit_cost_object"
    )
    problem.add_objects([t1, t2])
    problem.add_fluent(fuel, default_initial_value=0)
    problem.add_fluent(n, default_initial_value=0)
    problem.add_action(refuel)
    problem.add_goal(Equals(n, 2))
    # Every action costs `t1`'s current fuel, so the only zero-cost plans
    # never refuel `t1` before the last step. Only the metric -- its
    # `default`, or `refuel`'s entry in `costs` -- tells `t1` and `t2` apart.
    if in_default:
        problem.add_quality_metric(MinimizeActionCosts({}, default=fuel(t1)))
    else:
        problem.add_quality_metric(MinimizeActionCosts({refuel: fuel(t1)}))
    return problem


# Edge cases for the delete-relaxation heuristics (`hff`/`hadd`/`hmax`),
# regression-pinned by `test_engine.py::test_heuristic_values` over
# `HEURISTIC_EDGE_CASE_PROBLEMS`. Encoder operator ids follow the *sorted*
# action names, so where a tie-break depends on ids the names are chosen to
# fix the order. The initial-state values in the docstrings assume one unit
# of cost per operator.


def _bool_fluents(problem: Problem, *names: str, initial: bool = False) -> list:
    fluents = [Fluent(name, BoolType()) for name in names]
    for f in fluents:
        problem.add_fluent(f, default_initial_value=initial)
    return fluents


def _achiever(name: str, effect, *preconditions) -> InstantaneousAction:
    action = InstantaneousAction(name)
    for pre in preconditions:
        action.add_precondition(pre)
    action.add_effect(effect, True)
    return action


def get_problem_goal_satisfied_initially() -> Problem:
    """The goal `p` holds initially (h = 0 for every heuristic, hff's relaxed
    plan is empty). `clear_p` falsifies it and `set_p` restores it through
    `q`, so successors have a non-zero value."""
    problem = Problem("goal_satisfied_initially")
    (p,) = _bool_fluents(problem, "p", initial=True)
    (q,) = _bool_fluents(problem, "q")

    clear_p = InstantaneousAction("clear_p")
    clear_p.add_precondition(p)
    clear_p.add_effect(p, False)

    problem.add_actions([clear_p, _achiever("get_q", q), _achiever("set_p", p, q)])
    problem.add_goal(p)
    return problem


def get_problem_unreachable_goal() -> Problem:
    """`g` is only achieved by `unlock`, which needs `lock`, and `lock` is
    only achieved by `relock`, which needs `lock` itself: neither is ever
    reachable, so every heuristic reports a dead end (None), even though the
    other goal `s` is reachable."""
    problem = Problem("unreachable_goal")
    g, lock, s = _bool_fluents(problem, "g", "lock", "s")
    problem.add_actions(
        [
            _achiever("unlock", g, lock),
            _achiever("relock", lock, lock),
            _achiever("step", s),
        ]
    )
    problem.add_goal(g)
    problem.add_goal(s)
    return problem


def get_problem_unreachable_or_branch() -> Problem:
    """Goal `Or(g, h)`: `g` is unreachable (as in
    `get_problem_unreachable_goal`), `h` sits after the chain
    `a1 -> a2`. An OR must ignore its unreached operand: hff = hadd = hmax =
    2."""
    problem = Problem("unreachable_or_branch")
    g, h, lock, p1 = _bool_fluents(problem, "g", "h", "lock", "p1")
    problem.add_actions(
        [
            _achiever("unlock", g, lock),
            _achiever("relock", lock, lock),
            _achiever("a1", p1),
            _achiever("a2", h, p1),
        ]
    )
    problem.add_goal(Or(g, h))
    return problem


def get_problem_self_supporting_and_noop() -> Problem:
    """`loop` needs its own effect `p` and is never reachable; `noop` has no
    precondition and no effect; `get_q`/`get_r` have no precondition. Goal
    `Or(p, q)` and `r`: hadd = 2, hmax = 1, hff = 2."""
    problem = Problem("self_supporting_and_noop")
    p, q, r = _bool_fluents(problem, "p", "q", "r")
    noop = InstantaneousAction("noop")
    problem.add_actions(
        [
            _achiever("loop", p, p),
            noop,
            _achiever("get_q", q),
            _achiever("get_r", r),
        ]
    )
    problem.add_goal(Or(p, q))
    problem.add_goal(r)
    return problem


def get_problem_cheaper_path_found_late() -> Problem:
    """`g` is achieved either by `big`, which needs the four facts `f1..f4`
    (each one zero-precondition action away), or by the chain
    `c1 -> c2 -> c3`. A layered sweep first prices `g` through `big`
    (hadd 5, two layers deep) and only later through the chain (hadd 3), so
    the cost of `g` drops after it is first set. hadd = 3, hmax = 2 (through
    `big`), hff = 3 (the chain)."""
    problem = Problem("cheaper_path_found_late")
    fs = _bool_fluents(problem, "f1", "f2", "f3", "f4")
    g, q1, q2 = _bool_fluents(problem, "g", "q1", "q2")
    problem.add_actions([_achiever(f"get_{f.name}", f) for f in fs])
    problem.add_actions(
        [
            _achiever("big", g, *fs),
            _achiever("c1", q1),
            _achiever("c2", q2, q1),
            _achiever("c3", g, q2),
        ]
    )
    problem.add_goal(g)
    return problem


def get_problem_hff_tie_break(swap: bool) -> Problem:
    """Goals `g1` and `g2`. `g2` needs `s`; `g1` has two achievers of equal
    hadd cost 2, one through `s` (shared with `g2`) and one through `t`.
    hff breaks the tie by the larger operator id, i.e. the later action name:
    `g1_x1` reads `t` (hff = 4) unless `swap`, where it reads `s` (hff = 3).
    hadd = 4 and hmax = 2 either way."""
    problem = Problem(f"hff_tie_break_{'swap' if swap else 'noswap'}")
    g1, g2, s, t = _bool_fluents(problem, "g1", "g2", "s", "t")
    first, second = (t, s) if swap else (s, t)
    problem.add_actions(
        [
            _achiever("get_s", s),
            _achiever("get_t", t),
            _achiever("g2_from_s", g2, s),
            _achiever("g1_x0", g1, first),
            _achiever("g1_x1", g1, second),
        ]
    )
    problem.add_goal(g1)
    problem.add_goal(g2)
    return problem


def get_problem_hff_or_operand_tie(swap: bool) -> Problem:
    """Like `get_problem_hff_tie_break`, but the tie is between the operands
    of one OR precondition, `Or(s, t)` (or `Or(t, s)` if `swap`), both of
    cost 1: which operand the relaxed plan follows decides whether `get_t`
    joins it. The first operand wins the tie: hff = 3 for `Or(s, t)`, 4 for
    `Or(t, s)`. hadd = 4, hmax = 2."""
    problem = Problem(f"hff_or_operand_tie_{'swap' if swap else 'noswap'}")
    g1, g2, s, t = _bool_fluents(problem, "g1", "g2", "s", "t")
    problem.add_actions(
        [
            _achiever("get_s", s),
            _achiever("get_t", t),
            _achiever("g2_from_s", g2, s),
            _achiever("fin", g1, Or(t, s) if swap else Or(s, t)),
        ]
    )
    problem.add_goal(g1)
    problem.add_goal(g2)
    return problem


def get_problem_diamond_shared_subgoals() -> Problem:
    """`s` feeds both `left` and `right`, which `top` joins into `g`; the goal
    is `g` and `s`, so `s` is a goal leaf and a precondition at once. hadd
    counts `s` three times (6), hmax = 3, hff counts each action once (4)."""
    problem = Problem("diamond_shared_subgoals")
    g, s, left, right = _bool_fluents(problem, "g", "s", "l", "r")
    problem.add_actions(
        [
            _achiever("get_s", s),
            _achiever("left", left, s),
            _achiever("right", right, s),
            _achiever("top", g, left, right),
        ]
    )
    problem.add_goal(g)
    problem.add_goal(s)
    return problem


def get_problem_duplicate_and_nested_leaves() -> Problem:
    """Preconditions that repeat a leaf (`p` twice in `a1`, `p` also in the
    goal) and nest AND/OR/AND (`a2`), i.e. the shapes where a counter of
    distinct unreached leaves and the OR recomputation path matter. `r`
    needs `p`, so `a2`'s cheaper disjunct changes as costs settle."""
    problem = Problem("duplicate_and_nested_leaves")
    p, q, r, g1, g2 = _bool_fluents(problem, "p", "q", "r", "g1", "g2")
    a1 = InstantaneousAction("a1")
    a1.add_precondition(And(p, q, p))
    a1.add_precondition(p)
    a1.add_effect(g1, True)
    problem.add_actions(
        [
            _achiever("get_p", p),
            _achiever("get_q", q),
            _achiever("get_r", r, p),
            a1,
            _achiever("a2", g2, Or(And(p, r), And(q, Or(p, r)))),
        ]
    )
    problem.add_goal(g1)
    problem.add_goal(g2)
    problem.add_goal(p)
    return problem


def get_problem_negative_and_object_conditions() -> Problem:
    """Negative and object-valued leaves: `pass_closed` needs `not door`
    (initially `door`, so it needs the deleter `close`) and `at != l1`;
    `meet` needs `at == other` and `split` needs `not(at == other)`
    (fluent-vs-fluent, rewritten into ORs over the shared domain). Goal: all
    three done and `at == l3`."""
    problem = Problem("negative_and_object_conditions")
    Location = UserType("Location")
    l1, l2, l3 = (Object(f"l{i}", Location) for i in (1, 2, 3))
    problem.add_objects([l1, l2, l3])
    at = Fluent("at", Location)
    other = Fluent("other", Location)
    problem.add_fluent(at, default_initial_value=l1)
    problem.add_fluent(other, default_initial_value=l2)
    (door,) = _bool_fluents(problem, "door", initial=True)
    g1, g2, g3 = _bool_fluents(problem, "g1", "g2", "g3")

    move = InstantaneousAction("move", src=Location, dst=Location)
    move.add_precondition(Equals(at, move.src))
    move.add_precondition(Not(Equals(move.src, move.dst)))
    move.add_effect(at, move.dst)

    move_other = InstantaneousAction("move_other", dst=Location)
    move_other.add_effect(other, move_other.dst)

    close = InstantaneousAction("close")
    close.add_precondition(door)
    close.add_effect(door, False)

    problem.add_actions(
        [
            move,
            move_other,
            close,
            _achiever("pass_closed", g1, Not(door), Not(Equals(at, l1))),
            _achiever("meet", g2, Equals(at, other)),
            _achiever("split", g3, Not(Equals(at, other))),
        ]
    )
    for g in (g1, g2, g3):
        problem.add_goal(g)
    problem.add_goal(Equals(at, l3))
    return problem


def get_problem_numeric_multi_achievers() -> Problem:
    """Three achievers of `x >= 6`, the cheapest with the largest operator
    id: `inc_a` (+1, no precondition: 6 repetitions), `inc_b` (+3, after
    `get_p1`: 2 repetitions) and `inc_c` (+6, after the chain
    `c1 -> c2 -> c3`: 1 repetition). hadd prices `x >= 6` at
    `min(6 + 0, 2 + 1, 1 + 3) = 3` (g: 4); hmax at `rep` plus the cheapest
    precondition among the achievers, `1 + 0` (g: 2)."""
    problem = Problem("numeric_multi_achievers")
    x = Fluent("x", RealType())
    problem.add_fluent(x, default_initial_value=0)
    g, p1, q1, q2, q3 = _bool_fluents(problem, "g", "p1", "q1", "q2", "q3")

    inc_a = InstantaneousAction("inc_a")
    inc_a.add_increase_effect(x, 1)
    inc_b = InstantaneousAction("inc_b")
    inc_b.add_precondition(p1)
    inc_b.add_increase_effect(x, 3)
    inc_c = InstantaneousAction("inc_c")
    inc_c.add_precondition(q3)
    inc_c.add_increase_effect(x, 6)

    problem.add_actions(
        [
            inc_a,
            inc_b,
            inc_c,
            _achiever("get_p1", p1),
            _achiever("c1", q1),
            _achiever("c2", q2, q1),
            _achiever("c3", q3, q2),
            _achiever("r", g, GE(x, 6)),
        ]
    )
    problem.add_goal(g)
    return problem


def get_problem_numeric_chained() -> Problem:
    """A numeric condition reached through another one: `inc_y` needs
    `x >= 2`, `r` needs `y >= 3`. hadd = hmax = `1 + (3 + 2)` = 6, hff = 3."""
    problem = Problem("numeric_chained")
    x = Fluent("x", RealType())
    y = Fluent("y", RealType())
    problem.add_fluent(x, default_initial_value=0)
    problem.add_fluent(y, default_initial_value=0)
    (g,) = _bool_fluents(problem, "g")

    inc_x = InstantaneousAction("inc_x")
    inc_x.add_increase_effect(x, 1)
    inc_y = InstantaneousAction("inc_y")
    inc_y.add_precondition(GE(x, 2))
    inc_y.add_increase_effect(y, 1)

    problem.add_actions([inc_x, inc_y, _achiever("r", g, GE(y, 3))])
    problem.add_goal(g)
    return problem


def get_problem_numeric_strict_and_decrease() -> Problem:
    """A strict `x < 0` reached by decreasing (`x` starts at 5, `dec_x`
    subtracts 2: the epsilon-shifted weight decides between 3 and 2
    repetitions), and a numeric equality `y == 3` (rewritten into a pair of
    inequalities) with two increase achievers of different step."""
    problem = Problem("numeric_strict_and_decrease")
    x = Fluent("x", RealType())
    y = Fluent("y", IntType())
    problem.add_fluent(x, default_initial_value=5)
    problem.add_fluent(y, default_initial_value=0)
    g1, g2 = _bool_fluents(problem, "g1", "g2")

    dec_x = InstantaneousAction("dec_x")
    dec_x.add_decrease_effect(x, 2)
    inc_y = InstantaneousAction("inc_y")
    inc_y.add_increase_effect(y, 1)
    inc_y2 = InstantaneousAction("inc_y2")
    inc_y2.add_increase_effect(y, 2)

    problem.add_actions(
        [
            dec_x,
            inc_y,
            inc_y2,
            _achiever("r1", g1, LT(x, 0)),
            _achiever("r2", g2, Equals(y, 3)),
        ]
    )
    problem.add_goal(g1)
    problem.add_goal(g2)
    return problem


def get_problem_numeric_assign_and_complex() -> Problem:
    """Achievers that are not constant increases: `set_x` assigns `x := 5`,
    `grow_y` doubles `y` (a complex effect), both counted as one repetition
    (under the inadmissible variant, -1 each, which `inc_x`'s +1 can cancel
    on `x`). `r3` needs the nonlinear `x * y >= 20`, priced 0/1 by
    evaluation."""
    problem = Problem("numeric_assign_and_complex")
    x = Fluent("x", RealType())
    y = Fluent("y", RealType())
    problem.add_fluent(x, default_initial_value=1)
    problem.add_fluent(y, default_initial_value=1)
    g1, g2, g3 = _bool_fluents(problem, "g1", "g2", "g3")

    set_x = InstantaneousAction("set_x")
    set_x.add_effect(x, 5)
    grow_y = InstantaneousAction("grow_y")
    grow_y.add_effect(y, Times(y, 2))
    inc_x = InstantaneousAction("inc_x")
    inc_x.add_increase_effect(x, 1)

    problem.add_actions(
        [
            set_x,
            grow_y,
            inc_x,
            _achiever("r1", g1, GE(x, 4)),
            _achiever("r2", g2, GE(y, 3)),
            _achiever("r3", g3, GE(Times(x, y), 20)),
        ]
    )
    for g in (g1, g2, g3):
        problem.add_goal(g)
    return problem


def get_problem_numeric_unreachable() -> Problem:
    """`x >= 1` only has a wrong-direction writer (`dec_x`), so the goal is
    a dead end (None) though the other goal `s` is one action away."""
    problem = Problem("numeric_unreachable")
    x = Fluent("x", RealType())
    problem.add_fluent(x, default_initial_value=0)
    g, s = _bool_fluents(problem, "g", "s")

    dec_x = InstantaneousAction("dec_x")
    dec_x.add_decrease_effect(x, 1)

    problem.add_actions([dec_x, _achiever("r", g, GE(x, 1)), _achiever("get_s", s)])
    problem.add_goal(g)
    problem.add_goal(s)
    return problem


def get_problem_temporal_in_progress(reachable_end: bool) -> Problem:
    """`work` (start, start + 2, end events) needs `ready` at its end, which
    `prep` provides; `alt` achieves the goal `done` on its own. States reached
    by BFS have `work`/`prep` in progress, exercising the extra goals of
    hadd/hmax and hff's count of remaining events. Unless `reachable_end`,
    `prep` needs `lock`, which nothing reachable achieves: once `work` has
    started, hadd/hmax report a dead end while hff (which ignores the extra
    goals) does not."""
    problem = Problem(
        f"temporal_in_progress_{'reachable' if reachable_end else 'unreachable'}"
    )
    busy, half, ready, done, lock = _bool_fluents(
        problem, "busy", "half", "ready", "done", "lock"
    )

    work = DurativeAction("work")
    work.set_fixed_duration(5)
    work.add_effect(StartTiming(), busy, True)
    work.add_effect(StartTiming(2), half, True)
    work.add_condition(EndTiming(), ready)
    work.add_effect(EndTiming(), done, True)

    prep = DurativeAction("prep")
    prep.set_fixed_duration(2)
    if not reachable_end:
        prep.add_condition(StartTiming(), lock)
    prep.add_effect(EndTiming(), ready, True)

    alt = DurativeAction("alt")
    alt.set_fixed_duration(9)
    alt.add_condition(StartTiming(), Not(busy))
    alt.add_effect(EndTiming(), done, True)

    problem.add_actions([work, prep, alt, _achiever("relock", lock, lock)])
    problem.add_goal(done)
    return problem


def get_problem_wide_equal_costs(n: int = 10) -> Problem:
    """`n` facts `f_i`, each one zero-precondition action away, and `n` goals
    `g_j`, each with two equal-cost achievers (from `f_j` and from
    `f_(3j+1) mod n`), plus `n` goals `h_j` needing two facts. Many operators
    pop at each cost, so this stresses that hff's relaxed plan does not
    depend on the order equal-cost operators are expanded in."""
    problem = Problem(f"wide_equal_costs_{n}")
    fs = _bool_fluents(problem, *(f"f{i}" for i in range(n)))
    gs = _bool_fluents(problem, *(f"g{j}" for j in range(n)))
    hs = _bool_fluents(problem, *(f"h{j}" for j in range(n)))
    problem.add_actions([_achiever(f"get_f{i}", f) for i, f in enumerate(fs)])
    for j in range(n):
        problem.add_actions(
            [
                _achiever(f"ga_{j}", gs[j], fs[j]),
                _achiever(f"gb_{j}", gs[j], fs[(3 * j + 1) % n]),
                _achiever(f"h_{j}", hs[j], fs[(j + 2) % n], fs[(5 * j) % n]),
            ]
        )
    for g in gs + hs:
        problem.add_goal(g)
    return problem
