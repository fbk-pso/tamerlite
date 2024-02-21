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

    def get_tn_as_vector(self, state: State) -> List[float]:
        sol = state.temporal_network.distances
        last = -sol[state.path[-1]] if len(state.path) > 0 else -1
        m = {}
        se = defaultdict(int)
        ee = defaultdict(int)
        sa = {}
        for e, t in state.temporal_network.distances.items():
            if -t >= last:
                continue
            if len(e) == 2:
                v = m.get(e[0], None)
                if v is None or -t > v:
                    m[e[0]] = -t
            else:
                if e[1]:
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
                break
            nea = ee[t]
            c -= nea
            if c == 0:
                t_safe = t
                actions = []
            c += nsa
            if nsa > 0:
                actions.extend(sa[t])
        tn = [0.0 for _ in range(self._tn_size)]
        for a in actions:
            le = self._events[a]
            p = self._tn_actions_pos[a]
            for i, (_, e) in enumerate(le):
                v = m.get(e, None)
                if v is None or v-t_safe <= 0:
                    continue
                tn[p+i] = float(v-t_safe+1)
        return tn
