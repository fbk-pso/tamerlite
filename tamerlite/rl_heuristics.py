import math
import torch
import numpy as np


class RLRank:
    def __init__(self, state_encoder, model, ModelClass, config, sym_h):
        self._state_encoder = state_encoder
        self._model = ModelClass(state_encoder.state_geometry, config)
        self._model.load_state_dict(torch.load(model))
        self._model.eval()
        self._residual = config.residual
        self._sym_h = sym_h
        self._reward_signal = config.reward_signal
        self._gamma = config.gamma

    def eval(self, state, ss, sym_h=None):
        if ss.goal_reached(state):
            return 0
        state_vec = self._state_encoder.get_state_as_vector(state)
        r = self.eval_state_vec(state_vec)
        if self._residual:
            if sym_h is None:
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


class RLHeuristic:
    def __init__(self, state_encoder, model, ModelClass, config, sym_h):
        self._state_encoder = state_encoder
        self._model = ModelClass(state_encoder.state_geometry, config)
        self._model.load_state_dict(torch.load(model))
        self._model.eval()
        self._deltah_bin = config.deltah_bin
        self._gamma = config.gamma
        self._reward_signal = config.reward_signal
        self._residual = config.residual
        self._sym_h = sym_h

    def eval(self, state, ss):
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
