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

from typing import List
import unified_planning as up
import unified_planning.model

from collections import namedtuple
from unified_planning.model.fluent import get_all_fluent_exp
from unified_planning.model.walkers import Dnf
from unified_planning.model.types import domain_size, domain_item
from tamerlite.core import CoreStateEncoder


StateGeometry = namedtuple('StateGeometry',
                           ['num_fluents', 'num_actions',
                            'num_constants', 'num_goals', 'tn_size'])


class StateEncoder:
    """
    This class provides the functionality to return a state vectorization,
    given a search state. It uses the GeneralStateEncoder to compute the
    vectorization of the state unrelated to the intial state and goal of
    its problem.
    """

    def __init__(self, environment, general_state_encoder, initial_values, goals):
        self._general_state_encoder = general_state_encoder
        objects = self._general_state_encoder._objects
        fluents = self._general_state_encoder._fluents

        # vectorization of the static fluents value
        self._constants_vec = []
        for fe in self._general_state_encoder._constants:
            v = initial_values[fe]
            if v.is_object_exp():
                v = objects[v.object().name]
            elif v.is_bool_constant():
                v = 1.0 if v.constant_value() else 0.0
            else:
                lb = fe.type.lower_bound
                ub = fe.type.upper_bound
                if lb is None or ub is None:
                    v = float(v.constant_value())
                else:
                    v = float((v.constant_value() - lb) / (ub - lb))
            self._constants_vec.append(v)

        # vectorization of the goal
        w = Dnf(environment)
        self._goals = w.walk(environment.expression_manager.And(goals))[0]
        self._goals_vec = [-1.0 for _ in range(self.state_geometry.num_goals)]
        for g in self._goals:
            if g.is_fluent_exp():
                i = fluents.index(g)
                self._goals_vec[i] = 1.0
            elif g.is_not():
                i = fluents.index(g.arg(0))
                self._goals_vec[i] = 0.0
            elif g.is_equals() and g.arg(0).is_fluent_exp():
                i = fluents.index(g.arg(0))
                v = g.arg(1)
                if v.is_object_exp():
                    v = objects[v.object().name]
                else:
                    fe = g.arg(0)
                    lb = fe.type.lower_bound
                    ub = fe.type.upper_bound
                    if lb is None or ub is None:
                        v = float(v.constant_value())
                    else:
                        v = float((v.constant_value() - lb) / (ub - lb))
                self._goals_vec[i] = v
            elif g.is_equals() and g.arg(1).is_fluent_exp():
                i = fluents.index(g.arg(1))
                v = g.arg(0)
                if v.is_object_exp():
                    v = objects[v.object().name]
                else:
                    fe = g.arg(1)
                    lb = fe.type.lower_bound
                    ub = fe.type.upper_bound
                    if lb is None or ub is None:
                        v = float(v.constant_value())
                    else:
                        v = float((v.constant_value() - lb) / (ub - lb))
                self._goals_vec[i] = v
            else:
                raise NotImplementedError

    @property
    def state_geometry(self):
        return self._general_state_encoder.state_geometry

    def get_state_as_vector(self, state):
        return self._general_state_encoder.get_state_as_vector(state, self._constants_vec, self._goals_vec)


class GeneralStateEncoder:
    def __init__(self, problem, grounding_result, events, search_space):
        self._events = events
        self._search_space = search_space
        objects = sorted([o.name for o in problem.all_objects])
        self._objects = {on: float(i / len(objects)) for i, on in enumerate(objects)}
        static_fluents = problem.get_static_fluents()

        # compute the size of the vectorization of the running actions
        num_actions = 0
        for action in problem.actions:
            if len(action.parameters) == 0:
                num_actions += 1
            else:
                type_list = [param.type for param in action.parameters]
                ground_size = 1
                for t in type_list:
                    ds = domain_size(problem, t)
                    ground_size *= ds
                num_actions += ground_size

        # compute the size of the vectorization of the static and not-static fluents
        num_fluents = 0
        num_constants = 0
        fluents = []
        constants = []
        for fluent in problem.fluents:
            type_list = [param.type for param in fluent.signature]
            ground_size = 1
            for t in type_list:
                ds = domain_size(problem, t)
                ground_size *= ds
            f = list(get_all_fluent_exp(problem, fluent))
            if fluent in static_fluents:
                constants.extend(f)
                num_constants += ground_size
            else:
                fluents.extend(f)
                num_fluents += ground_size
        self._fluents = sorted(fluents, key=str)
        self._fluents_str = [(str(fe), fe.type.is_bool_type(), (float(fe.type.lower_bound), float(fe.type.upper_bound)) if fe.type.is_int_type() or fe.type.is_real_type() else (None, None)) for fe in self._fluents]
        self._constants = sorted(constants, key=str)

        # compute the size of the vectorization of the temporal network
        tn_size = 0
        actions = []
        actions_tp = {}
        for action in problem.actions:
            if isinstance(action, up.model.DurativeAction):
                time_points = set()
                for tp in action.effects.keys():
                    time_points.add(tp)
                for i in action.conditions.keys():
                    time_points.add(i.lower)
                    time_points.add(i.upper)
                s = len(time_points)
            else:
                s = 1
            domain_sizes = []
            type_list = [param.type for param in action.parameters]
            ground_size = 1
            for t in type_list:
                ds = domain_size(problem, t)
                domain_sizes.append(ds)
                ground_size *= ds
            for idx in range(ground_size):
                quot = idx
                rem = 0
                actual_parameters = []
                for ds, p in zip(domain_sizes, action.parameters):
                    rem = quot % ds
                    quot //= ds
                    v = domain_item(problem, p.type, rem)
                    actual_parameters.append(v)
                ai = action(*actual_parameters)
                actions.append(str(ai))
                actions_tp[str(ai)] = s
                tn_size += s

        actions = sorted(actions)
        t = 0
        actions_pos = {}
        for a in actions:
            actions_pos[a] = t
            t += actions_tp[a]
        self._actions = {}
        self._actions_pos = {}
        for a in grounding_result.problem.actions:
            oa = grounding_result.map_back_action_instance(a())
            self._actions[a.name] = actions.index(str(oa))
            self._actions_pos[a.name] = actions_pos[str(oa)]
        self._sg = StateGeometry(num_fluents, num_actions, num_constants, num_fluents, tn_size)

        self._cse = CoreStateEncoder(self._sg.num_actions, self._sg.tn_size, self._fluents_str,
                                     self._actions, self._actions_pos, self._objects, self._events)

    @property
    def state_geometry(self):
        return self._sg

    def get_state_as_vector(self, state, constants_vec, goals_vec) -> List[float]:
        # vectorization of the fluents value
        res = self._cse.get_fluents_as_vector(state)

        # vectorization of the running actions
        actions = self._cse.get_running_actions_as_vector(state)
        res.extend(actions)

        res.extend(constants_vec)
        res.extend(goals_vec)

        # vectorization of the temporal network
        tn = self._cse.get_tn_as_vector(state, self._search_space)
        res.extend(tn)

        return res
