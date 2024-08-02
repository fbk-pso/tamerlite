import math
import torch
import numpy as np


class RLRank:
    def __init__(self, encoder, state_encoder, model, ModelClass, config, sym_h):
        self._encoder = encoder
        self._state_encoder = state_encoder
        self._model = ModelClass(state_encoder.state_geometry, config)
        self._model.load_state_dict(torch.load(model))
        self._model.eval()
        self._residual = config.residual
        self._sym_h = sym_h

    def eval(self, state, ss):
        if ss.goal_reached(state):
            return 0
        state_vec = self._state_encoder.get_state_as_vector(state)
        if self._residual:
            sym_h = self._sym_h.eval(state, self._encoder.search_space)
            if sym_h is None:
                return None
            else:
                return -self.eval_state_vec(state_vec) + sym_h
        else:
            return self.eval_state_vec(state_vec)

    def eval_state_vec(self, state_vec):
        s = np.array([state_vec])
        r = self._model(torch.from_numpy(s).float()).detach()[0]
        return float(-r[0])+2.0


class RLHeuristic:
    def __init__(self, encoder, state_encoder, model, ModelClass, config, sym_h):
        self._encoder = encoder
        self._state_encoder = state_encoder
        self._model = ModelClass(state_encoder.state_geometry, config)
        self._model.load_state_dict(torch.load(model))
        self._model.eval()
        self._delta_h = config.delta_h
        self._gamma = config.gamma
        self._reward_signal = config.reward_signal
        self._residual = config.residual
        self._sym_h = sym_h

    def eval(self, state, ss):
        if ss.goal_reached(state):
            return 0
        state_vec = self._state_encoder.get_state_as_vector(state)
        if self._residual:
            sym_h = self._sym_h.eval(state, self._encoder.search_space)
            if sym_h is None:
                return None
            else:
                return -self.eval_state_vec(state_vec) + sym_h
        else:
            return self.eval_state_vec(state_vec)

    def eval_state_vec(self, state_vec):
        s = np.array([state_vec])
        r = self._model(torch.from_numpy(s).float()).detach()[0]
        r = float(r[0])
        if self._reward_signal=="old":
            if r == 0:
                return float(self._delta_h)
            elif r < 0:
                return float((2 * self._delta_h) - min(self._delta_h, (math.log(min(1, -r), self._gamma))))
            else:
                return float(min(self._delta_h, (math.log(min(1, r), self._gamma)+1)))
        else:
            return -float(r)   # min(0, -r)