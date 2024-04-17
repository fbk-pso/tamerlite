import math
import torch
import numpy as np


class RLRank:
    def __init__(self, state_encoder, model, ModelClass):
        self._state_encoder = state_encoder
        self._model = ModelClass(state_encoder.state_geometry)
        self._model.load_state_dict(torch.load(model))
        self._model.eval()

    def eval(self, state):
        state_vec = self._state_encoder.get_state_as_vector(state)
        return self.eval_state_vec(state_vec)

    def eval_state_vec(self, state_vec):
        s = np.array([state_vec])
        r = self._model(torch.from_numpy(s).float()).detach()[0]
        return float(-r[0])+2.0


class RLHeuristic:
    def __init__(self, state_encoder, model, ModelClass, max_plan_size, gamma):
        self._state_encoder = state_encoder
        self._model = ModelClass(state_encoder.state_geometry)
        self._model.load_state_dict(torch.load(model))
        self._model.eval()
        self._max_plan_size = max_plan_size
        self._gamma = gamma

    def eval(self, state):
        state_vec = self._state_encoder.get_state_as_vector(state)
        return self.eval_state_vec(state_vec)

    def eval_state_vec(self, state_vec):
        s = np.array([state_vec])
        r = self._model(torch.from_numpy(s).float()).detach()[0]
        r = float(r[0])
        if r == 0:
            return float(self._max_plan_size)
        elif r < 0:
            return float((2 * self._max_plan_size) - min(self._max_plan_size, (math.log(min(1, -r), self._gamma))))
        else:
            return float(min(self._max_plan_size, (math.log(min(1, r), self._gamma))))
