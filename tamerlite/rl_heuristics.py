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

import math
import torch
import numpy as np
from tamerlite.core.heuristics import Heuristic

class RLHeuristicBase(Heuristic):
    def __init__(self, state_encoder, model, ModelClass, config, sym_h,
                 cache_value_in_state: bool = False):
        super().__init__(cache_value_in_state)
        self._state_encoder = state_encoder
        self._model = ModelClass(state_encoder.state_geometry, config)
        self._model.load_state_dict(torch.load(model))
        self._model.eval()
        self._residual = config.residual
        self._sym_h = sym_h

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
        raise NotImplementedError("This method should be overridden by subclasses.")


class RLRank(RLHeuristicBase):
    @property
    def name(self):
        return "rlrank"

    def eval_state_vec(self, state_vec, sym_h):
        s = np.array([state_vec])
        r = self._model.get_rank(torch.from_numpy(s).float(), [sym_h])
        return r[0]


class RLHeuristic(RLHeuristicBase):
    @property
    def name(self):
        return "rlh"

    def eval_state_vec(self, state_vec, sym_h):
        s = np.array([state_vec])
        r = self._model.get_heuristic(torch.from_numpy(s).float(), [sym_h])
        return r[0]