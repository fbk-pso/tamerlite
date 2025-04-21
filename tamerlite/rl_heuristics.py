# Copyright (C) 2025 PSO Unit, Fondazione Bruno Kessler
# This file is part of TamerLite.
#
# TamerLite is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# TamerLite is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#

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
        r = self._model(torch.from_numpy(s).float(), [sym_h]).detach()[0]
        r = float(r[0])
        if self._residual and self._reward_signal=="cnt":
            r -= 3*self._deltah_cnt
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
        r = self._model(torch.from_numpy(s).float(), [sym_h]).detach()[0]
        r = float(r[0])
        if self._reward_signal=="bin":
            if r == 0:
                return float(self._deltah_bin)
            elif r < 0:
                return float((2 * self._deltah_bin) - min(self._deltah_bin, (math.log(min(1, -r), self._gamma))))
            else:
                return float(min(self._deltah_bin, (math.log(min(1, r), self._gamma)+1)))
        else:
            return max(0.000001,-r)
