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

from fractions import Fraction
from typing import Tuple, Dict, List
from collections import defaultdict

from tamerlite.core.search_space import Event, State, Timing


class CoreStateEncoder:
    def __init__(
            self,
            num_actions: int,
            tn_size: int,
            fluents: List[Tuple[str, bool, Tuple[float, float]]],
            actions_pos: Dict[str, int],
            tn_actions_pos: Dict[str, int],
            objects: Dict[str, float],
            events: Dict[str, List[Tuple[Timing, Event]]],
        ) -> None:
        self._num_actions = num_actions
        self._tn_size = tn_size
        self._fluents = fluents
        self._actions_pos = actions_pos
        self._tn_actions_pos = tn_actions_pos
        self._objects = objects
        self._events = events

    def get_fluents_as_vector(self, state: State) -> List[float]:
        res = []
        for sfe, is_bool, (lb, ub) in self._fluents:
            v = state.get_value(sfe)
            if isinstance(v, str):
                v = self._objects[v]
            elif is_bool:
                v = 1.0 if v else 0.0
            else:
                if lb is None or ub is None:
                    v = float(v)
                else:
                    v = (float(v) - lb) / (ub - lb)
            res.append(v)
        return res

    def get_running_actions_as_vector(self, state: State) -> List[float]:
        actions = [0.0 for _ in range(self._num_actions)]
        for a, i in self._actions_pos.items():
            x, _ = state.todo.get(a, (0, 0))
            if x > 0:
                v = len(self._events[a])-x
            else:
                v = 0
            actions[i] = float(v)
        return actions

    def get_tn_as_vector(self, state: State, search_space) -> List[float]:
        sol = state.temporal_network.distances
        last = -sol[state.path[-1]] if len(state.path) > 0 else -1
        m = {}
        se = defaultdict(int)
        ee = defaultdict(int)
        sa = {}
        for e, t in state.temporal_network.distances.items():
            if -t > last:
                continue
            if not isinstance(e[1], bool):
                ev = (e[0], e[1])
                v = m.get(ev, None)
                if v is None or -t > v:
                    m[ev] = -t
            else:
                oe = (e[0], not e[1], e[2])
                if state.temporal_network.distances[oe] == t:
                    if e[1]:
                        se[-t] += 0
                        ee[-t] += 0
                        sa.setdefault(-t, []).append(e[0])
                elif e[1]:
                    se[-t] += 1
                    sa.setdefault(-t, []).append(e[0])
                else:
                    se[-t] += 0
                    ee[-t] += 1
        t_safe = 0
        c = 0
        actions = []
        for t, nsa in sorted(se.items()):
            if t == last:
                actions.extend(sa.get(t, []))
                break
            nea = ee[t]
            c -= nea
            if c == 0:
                t_safe = t
                actions = []
            c += nsa
            actions.extend(sa.get(t, []))
        tn = [0.0 for _ in range(self._tn_size)]
        for a in actions:
            le = self._events[a]
            p = self._tn_actions_pos[a]
            for i, (_, e) in enumerate(le):
                v = m.get((e.action, e.pos), None)
                if v is None or v-t_safe <= 0:
                    continue
                tn[p+i] = float(v-t_safe+1)
        return tn
