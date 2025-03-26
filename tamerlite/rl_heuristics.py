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

import torch
from tamerlite.core.heuristics import Heuristic

class RLHeuristicBase(Heuristic):
    def __init__(self, state_encoder, model, ModelClass, config, sym_h,
                 cache_value_in_state: bool = False):
        super().__init__(cache_value_in_state)
        self._state_encoder = state_encoder
        self._deltah_bin = config.deltah_bin
        self._gamma = config.gamma
        self._reward_signal = config.reward_signal
        self._residual = config.residual
        self._sym_h = sym_h
        self._use_gnn = config.use_gnn

        if self._use_gnn:
            self._inference_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self._model = ModelClass(100).to(self._inference_device)
        else:
            self._inference_device = torch.device("cpu")
            self._model = ModelClass(state_encoder.state_geometry, config).to(self._inference_device)
        self._model.load_state_dict(torch.load(model))
        self._model.eval()

    def _eval(self, state, ss):
        # if ss.goal_reached(state):
        #     return 0

        if self._residual:
            sym_h = self._sym_h.eval(state, ss)
            if sym_h is None:
                return None
        else:
            sym_h = -1

        if self._use_gnn:
            state_vec = self._state_encoder.get_state_as_graph(state)
            return self.eval_state_graph(state_vec, sym_h)
        else:
            state_vec = self._state_encoder.get_state_as_vector(state)
            return self.eval_state_vec(state_vec, sym_h)

    def eval_state_graph(self, G, sym_h):
        r = self._model(torch.tensor(G.nodes, dtype=torch.float, device=self._inference_device),
                        torch.tensor(G.edges, dtype=torch.long, device=self._inference_device),
                        torch.tensor(G.edge_features, dtype=torch.float, device=self._inference_device),
                        torch.zeros(len(G.nodes), dtype=torch.int64, device=self._inference_device)).detach()[0]
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

    def eval_state_vec(self, state_vec, sym_h):
        return self.eval_state_vecs([state_vec], [sym_h])[0]

    def eval_state_vecs(self, states_vectors, sym_hs):
        raise NotImplementedError("This method should be overridden by subclasses.")

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
            rs = self.eval_state_vecs(vectors_to_eval, sym_heuristics_to_eval)
            for (i, s), res in zip(states_to_eval, rs, strict=True):
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

    def eval_state_vecs(self, states_vectors, sym_hs):
        if len(states_vectors) == 0:
            return []
        return self._model.get_rank(torch.tensor(states_vectors, dtype=torch.float32), sym_hs)


class RLHeuristic(RLHeuristicBase):
    @property
    def name(self):
        return "rlh"

    def eval_state_vecs(self, states_vectors, sym_hs):
        if len(states_vectors) == 0:
            return []
        return self._model.get_heuristic(torch.tensor(states_vectors, dtype=torch.float32), sym_hs)
