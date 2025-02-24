import math
import torch
import numpy as np
from tamerlite.core.heuristics import Heuristic


class RLRank(Heuristic):
    def __init__(self, state_encoder, model, ModelClass, config, sym_h, cache_enabled: bool = False):
        super().__init__(cache_enabled)
        self._state_encoder = state_encoder
        self._model = ModelClass(state_encoder.state_geometry, config)
        self._model.load_state_dict(torch.load(model))
        self._model.eval()
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
        r = self.eval_state_vec(state_vec)
        if self._residual:
            sym_h = self._sym_h.eval(state, ss)
            if sym_h is None:
                return None
            else:
                if self._reward_signal=="new":
                    r -= sym_h
                else:
                    r += self._gamma**(sym_h-1)
        return -r

    def eval_state_vec(self, state_vec):
        s = np.array([state_vec])
        r = self._model(torch.from_numpy(s).float()).detach()[0]
        return float(r[0])


class RLHeuristic(Heuristic):
    def __init__(self, state_encoder, model, ModelClass, config, sym_h, cache_enabled: bool = False):
        super().__init__(cache_enabled)
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
        r = self.eval_state_vec(state_vec)
        if self._residual:
            sym_h = self._sym_h.eval(state, ss)
            if sym_h is None:
                return None
            if self._reward_signal=="new":
                r -= sym_h
                r = min(0,r)
            else:
                r += self._gamma**(sym_h-1)
        if self._reward_signal=="old":
            if r == 0:
                return float(self._deltah_bin)
            elif r < 0:
                return float((2 * self._deltah_bin) - min(self._deltah_bin, (math.log(min(1, -r), self._gamma))))
            else:
                return float(min(self._deltah_bin, (math.log(min(1, r), self._gamma)+1)))
        else:
            return -r

    def eval_state_vec(self, state_vec):
        s = np.array([state_vec])
        r = self._model(torch.from_numpy(s).float()).detach()[0]
        return float(r[0])   # min(0, -r)
