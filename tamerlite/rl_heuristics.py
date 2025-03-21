import math
import torch
import numpy as np
from tamerlite.core.heuristics import Heuristic


class RLHeuristicBase(Heuristic):
    def __init__(self, state_encoder, model, ModelClass, config, sym_h, cache_value_in_state: bool = False):
        super().__init__(cache_value_in_state)
        self._state_encoder = state_encoder
        self._model = ModelClass(state_encoder.state_geometry, config)
        self._model.load_state_dict(torch.load(model))
        self._model.eval()
        self._deltah_cnt = config.deltah_cnt
        self._deltah_bin = config.deltah_bin
        self._residual = config.residual
        self._sym_h = sym_h
        self._reward_signal = config.reward_signal
        self._gamma = config.gamma

    def _compute_return_value(self, model_value, sym_h):
        raise NotImplementedError

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
        return self._compute_return_value(r, sym_h)

    def eval_gen(self, states_generator, ss):
        cached = []
        states_to_eval = []
        vectors_to_eval = []
        sym_heuristics_to_eval = []
        for i, state in enumerate(states_generator):
            h = state.heuristic_cache.get(self.name, -1)
            if h==-1:
                if self._residual:
                    sym_h = self._sym_h.eval(state, ss)
                    if sym_h is None:
                        cached.append((i, state, None))
                        continue
                else:
                    sym_h = -1

                if ss.goal_reached(state):
                    cached.append((i, state, 0))
                else:
                    state_vec = self._state_encoder.get_state_as_vector(state)
                    states_to_eval.append((i, state))
                    vectors_to_eval.append(state_vec)
                    sym_heuristics_to_eval.append(sym_h)
            else:
                cached.append((i, state, h))

        cached_idx = 0
        if len(states_to_eval) > 0:
            rs = self._model(torch.tensor(vectors_to_eval, dtype=torch.float32)).detach()
            for (i, s), r, sym_h in zip(states_to_eval, rs, sym_heuristics_to_eval, strict=True):
                r = float(r[0])
                res = self._compute_return_value(r, sym_h)
                #assert abs(res - self.eval(s, ss)) < 0.0001, f"{res} != {self.eval(s, ss)}"

                # Yield cached states (if any) with index < i
                while cached_idx < len(cached) and cached[cached_idx][0] < i:
                    yield cached[cached_idx][1], cached[cached_idx][2]
                    cached_idx += 1

                # Yield current state
                yield s, res

        # Yield remaining cached states (if any)
        while cached_idx < len(cached):
            yield cached[cached_idx][1], cached[cached_idx][2]
            cached_idx += 1


class RLRank(RLHeuristicBase):

    @property
    def name(self):
        return "rlrank"

    def _compute_return_value(self, model_output, sym_h):
        if self._residual:
            if self._reward_signal=="cnt":
                model_output -= sym_h + 3*self._deltah_cnt
            else:
                model_output += self._gamma**(sym_h-1)
        return -model_output+3.0


class RLHeuristic(RLHeuristicBase):

    @property
    def name(self):
        return "rlh"

    def _compute_return_value(self, model_output, sym_h):
        if self._residual:
            if self._reward_signal=="cnt":
                model_output -= sym_h
            else:
                model_output += self._gamma**(sym_h-1)
        if self._reward_signal=="bin":
            if model_output == 0:
                return float(self._deltah_bin)
            elif model_output < 0:
                return float((2 * self._deltah_bin) - min(self._deltah_bin, (math.log(min(1, -model_output), self._gamma))))
            else:
                return float(min(self._deltah_bin, (math.log(min(1, model_output), self._gamma)+1)))
        else:
            return max(0.000001,-model_output)

