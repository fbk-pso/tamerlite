import math
import torch
import numpy as np
from tamerlite.core.heuristics import Heuristic


class RLRank(Heuristic):
    def __init__(self, state_encoder, model, ModelClass, config, sym_h, cache_value_in_state: bool = False):
        super().__init__(cache_value_in_state)
        self._state_encoder = state_encoder
        self._model = ModelClass(state_encoder.state_geometry, config)
        self._model.load_state_dict(torch.load(model))
        self._model.eval()
        self._deltah_cnt = config.deltah_cnt
        self._residual = config.residual
        self._sym_h = sym_h
        self._reward_signal = config.reward_signal
        self._gamma = config.gamma

    @property
    def name(self):
        return "rlrank"

    def _eval(self, state, ss):
        if ss.goal_reached(state):
            return 0
        state_vec = self._state_encoder.get_state_as_vector(state)
        if self._residual:
            sym_h = self._sym_h.eval(state, ss)
            if sym_h is None:
                return None
        else:
            sym_h = -1
        return self.eval_state_vec(state_vec, sym_h)

    def eval_state_vec(self, state_vec, sym_h):
        s = np.array([state_vec])
        r = self._model(torch.from_numpy(s).float()).detach()[0]
        r = float(r[0])
        if self._residual:
            if self._reward_signal=="cnt":
                r -= sym_h + 3*self._deltah_cnt
            else:
                r += self._gamma**(sym_h-1)
        return -r+3.0


class RLHeuristic(Heuristic):
    def __init__(self, state_encoder, model, ModelClass, config, sym_h, cache_value_in_state: bool = False):
        super().__init__(cache_value_in_state)
        self._state_encoder = state_encoder
        self._model = ModelClass(state_encoder.state_geometry, config)
        self._model.load_state_dict(torch.load(model))
        self._model.eval()
        self._deltah_bin = config.deltah_bin
        self._gamma = config.gamma
        self._reward_signal = config.reward_signal
        self._residual = config.residual
        self._sym_h = sym_h

    @property
    def name(self):
        return "rlh"

    def _eval(self, state, ss):
        if ss.goal_reached(state):
            return 0
        state_vec = self._state_encoder.get_state_as_vector(state)
        if self._residual:
            sym_h = self._sym_h.eval(state, ss)
            if sym_h is None:
                return None
        else:
            sym_h = -1
        return self.eval_state_vec(state_vec, sym_h)

    def eval_state_vec(self, state_vec, sym_h):
        s = np.array([state_vec])
        r = self._model(torch.from_numpy(s).float()).detach()[0]
        r = float(r[0])
        if self._residual:
            if self._reward_signal=="cnt":
                r -= sym_h
            else:
                r += self._gamma**(sym_h-1)
        if self._reward_signal=="bin":
            if r == 0:
                return float(self._deltah_bin)
            elif r < 0:
                return float((2 * self._deltah_bin) - min(self._deltah_bin, (math.log(min(1, -r), self._gamma))))
            else:
                return float(min(self._deltah_bin, (math.log(min(1, r), self._gamma)+1)))
        else:
            return max(0.000001,-r)

    def gen_eval(self, states_generator, ss):
        cached_queue = []
        states_queue = []
        vectors_queue = []
        sym_heuristics_queue = []
        for i, state in enumerate(states_generator):
            h = state.heuristic_cache.get(self.name, -1)
            if h==-1:
                if self._residual:
                    sym_h = self._sym_h.eval(state, ss)
                    if sym_h is None:
                        cached_queue.append((i, state, None))
                        continue
                else:
                    sym_h = -1

                if ss.goal_reached(state):
                    cached_queue.append((i, state, 0))
                else:
                    state_vec = self._state_encoder.get_state_as_vector(state)
                    states_queue.append((i, state))
                    vectors_queue.append(state_vec)
                    sym_heuristics_queue.append(sym_h)
            else:
                cached_queue.append((i, state, h))

        cached_queue_idx = 0
        if len(states_queue) > 0:
            rs = self._model(torch.tensor(vectors_queue, dtype=torch.float32)).detach()
            for (i, s), r, sym_h in zip(states_queue, rs, sym_heuristics_queue, strict=True):
                r = float(r[0])
                if self._residual:
                    if self._reward_signal=="cnt":
                        r -= sym_h
                    else:
                        r += self._gamma**(sym_h-1)
                res = None
                if self._reward_signal=="bin":
                    if r == 0:
                        res = float(self._deltah_bin)
                    elif r < 0:
                        res = float((2 * self._deltah_bin) - min(self._deltah_bin, (math.log(min(1, -r), self._gamma))))
                    else:
                        res = float(min(self._deltah_bin, (math.log(min(1, r), self._gamma)+1)))
                else:
                    res = max(0.000001,-r)

                #assert abs(res - self.eval(s, ss)) < 0.0001, f"{res} != {self.eval(s, ss)}"

                # Yield cached states (if any) with index < i
                while cached_queue_idx < len(cached_queue) and cached_queue[cached_queue_idx][0] < i:
                    yield cached_queue[cached_queue_idx][1], cached_queue[cached_queue_idx][2]
                    cached_queue_idx += 1

                # Yield current state
                yield s, res

        # Yield remaining cached states (if any)
        while cached_queue_idx < len(cached_queue):
            yield cached_queue[cached_queue_idx][1], cached_queue[cached_queue_idx][2]
            cached_queue_idx += 1
