import math
import torch
import torch.nn.functional as F
import numpy as np

def apply_outer_activation(x, outer_activation_type, reward_signal, delta_h):
    if outer_activation_type=="none":
        return x
    elif reward_signal == "old":
        if outer_activation_type=="soft":
            return F.softsign(x)
        elif outer_activation_type=="hard":
            return F.hardtanh(x)
    else:
        if outer_activation_type=="soft":
            return -F.softplus(-x)
        elif outer_activation_type=="hard":
            return F.hardtanh(x, min_val=-3*delta_h, max_val=0)


class RLRank:
    def __init__(self, state_encoder, model, ModelClass, config, sym_h):
        self._state_encoder = state_encoder
        self._model = ModelClass(state_encoder.state_geometry, config)
        self._model.load_state_dict(torch.load(model))
        self._model.eval()
        self._residual = config.residual
        self._sym_h = sym_h
        self._delta_h = config.delta_h
        self._reward_signal = config.reward_signal
        self._outer_activation = config.outer_activation
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
                sym_h = torch.tensor(sym_h)
                if self._reward_signal=="new":
                    r -= sym_h + 3*self._delta_h
                else:
                    r += self._gamma**(sym_h-1)
                r = apply_outer_activation(r, self._outer_activation, self._reward_signal, self._delta_h)
        r = float(r)
        if self._reward_signal=="new":
            return -r
        else:
            return -r+2.0

    def eval_state_vec(self, state_vec):
        s = np.array([state_vec])
        r = self._model(torch.from_numpy(s).float()).detach()[0]
        return r[0]


class RLHeuristic:
    def __init__(self, state_encoder, model, ModelClass, config, sym_h):
        self._state_encoder = state_encoder
        self._model = ModelClass(state_encoder.state_geometry, config)
        self._model.load_state_dict(torch.load(model))
        self._model.eval()
        self._delta_h = config.delta_h
        self._gamma = config.gamma
        self._reward_signal = config.reward_signal
        self._residual = config.residual
        self._outer_activation = config.outer_activation
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
            sym_h = torch.tensor(sym_h)
            if self._reward_signal=="new":
                r -= sym_h
            else:
                r += self._gamma**(sym_h-1)
            r = apply_outer_activation(r, self._outer_activation, self._reward_signal, self._delta_h)
        r = float(r)
        if self._reward_signal=="old":
            if r == 0:
                return float(self._delta_h)
            elif r < 0:
                return float((2 * self._delta_h) - min(self._delta_h, (math.log(min(1, -r), self._gamma))))
            else:
                return float(min(self._delta_h, (math.log(min(1, r), self._gamma)+1)))
        else:
            return -min(0,r)

    def eval_state_vec(self, state_vec):
        s = np.array([state_vec])
        r = self._model(torch.from_numpy(s).float()).detach()[0]
        return r[0]   # min(0, -r)
