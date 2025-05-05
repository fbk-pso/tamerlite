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

from dataclasses import dataclass
from functools import partial
import unified_planning as up
import unified_planning.model
import unified_planning.engines
import unified_planning.engines.mixins
from unified_planning.model import ProblemKind, FNode
from unified_planning.model.state import State
from unified_planning.model.walkers import Simplifier
from unified_planning.model.fluent import get_all_fluent_exp
from typing import IO, Any, Callable, List, Optional, Union
from types import SimpleNamespace

from tamerlite.core import wastar_search, astar_search, gbfs_search
from tamerlite.core import bfs_search, dfs_search, ehc_search
from tamerlite.core import multiqueue_search
from tamerlite.core import evaluate, make_fluent_node
from tamerlite.core import HFF, HAdd, HMax, HMaxNumeric, CustomHeuristic, RLRank, RLHeuristic
from tamerlite.core import simplify, Effect, Event
from tamerlite.converter import Converter
from tamerlite.encoder import Encoder, get_encoders


credits = up.engines.Credits('TamerLite',
                  'FBK PSO Unit',
                  'tamer@fbk.eu',
                  'https://tamer.fbk.eu',
                  'Free for Educational Use',
                  '',
                  ''
                )


class StateWrapper(State):
    def __init__(self, problem, search_state):
        self.search_state = search_state
        self.problem = problem
        self.em = problem.environment.expression_manager

    def get_value(self, x: FNode) -> FNode:
        key = (make_fluent_node(str(x)), )
        v = evaluate(key, self.search_state)
        if x.type.is_bool_type():
            return self.em.Bool(v)
        elif x.type.is_int_type():
            return self.em.Int(v)
        elif x.type.is_real_type():
            return self.em.Real(v)
        elif x.type.is_user_type():
            return self.em.ObjectExp(self.problem.object(v))
        else:
            raise NotImplementedError("Unknown value type for expression %s" % x)


@dataclass(frozen=True)
class RLParams:
    domain: up.model.Problem
    model: str
    model_class: Any # Neural Network Class
    other_params: Optional[SimpleNamespace] = None


@dataclass(frozen=True)
class SearchParams:
    search: Optional[str] = None
    heuristic: Optional[str] = None
    internal_heuristic_cache: Optional[bool] = None
    weight: Optional[str] = None
    rl_params: Optional[RLParams] = None
    cache_heuristic_in_state: Optional[bool] = None

    def contains_rl(self) -> bool:
        return self.rl_params is not None

    def domain(self):
        if self.contains_rl():
            return self.rl_params.domain
        return None


@dataclass(frozen=True)
class MultiqueueParams:
    queues: List[SearchParams]

    def contains_rl(self) -> bool:
        return any([q.contains_rl() for q in self.queues])

    def domain(self):
        d = None
        for q in self.queues:
            if q.rl_params and q.rl_params.domain is not None:
                assert d is None or d == q.rl_params.domain
                d = q.rl_params.domain
        return d


class TamerLite(
        unified_planning.engines.Engine,
        unified_planning.engines.mixins.OneshotPlannerMixin,
    ):

    def __init__(self, search: Optional[Union[SearchParams, MultiqueueParams]] = None):
        unified_planning.engines.Engine.__init__(self)
        up.engines.mixins.OneshotPlannerMixin.__init__(self)
        self._params = search

    @property
    def name(self) -> str:
        return "TamerLite"

    @staticmethod
    def get_credits(**kwargs) -> Optional[up.engines.Credits]:
        return credits

    @staticmethod
    def supported_kind() -> ProblemKind:
        supported_kind = ProblemKind()
        supported_kind.set_problem_class('ACTION_BASED')
        supported_kind.set_time('CONTINUOUS_TIME')
        supported_kind.set_time('INTERMEDIATE_CONDITIONS_AND_EFFECTS')
        supported_kind.set_time('DURATION_INEQUALITIES')
        supported_kind.set_expression_duration('STATIC_FLUENTS_IN_DURATIONS')
        supported_kind.set_expression_duration('FLUENTS_IN_DURATIONS')
        supported_kind.set_expression_duration('INT_TYPE_DURATIONS')
        supported_kind.set_numbers('DISCRETE_NUMBERS')
        supported_kind.set_numbers('CONTINUOUS_NUMBERS')
        supported_kind.set_problem_type("SIMPLE_NUMERIC_PLANNING")
        supported_kind.set_problem_type("GENERAL_NUMERIC_PLANNING")
        supported_kind.set_typing('FLAT_TYPING')
        supported_kind.set_parameters("BOOL_FLUENT_PARAMETERS")
        supported_kind.set_parameters("BOUNDED_INT_FLUENT_PARAMETERS")
        supported_kind.set_parameters("BOOL_ACTION_PARAMETERS")
        supported_kind.set_parameters("BOUNDED_INT_ACTION_PARAMETERS")
        supported_kind.set_effects_kind('INCREASE_EFFECTS')
        supported_kind.set_effects_kind('DECREASE_EFFECTS')
        supported_kind.set_effects_kind("STATIC_FLUENTS_IN_BOOLEAN_ASSIGNMENTS")
        supported_kind.set_effects_kind("FLUENTS_IN_BOOLEAN_ASSIGNMENTS")
        supported_kind.set_effects_kind("STATIC_FLUENTS_IN_NUMERIC_ASSIGNMENTS")
        supported_kind.set_effects_kind("FLUENTS_IN_NUMERIC_ASSIGNMENTS")
        supported_kind.set_effects_kind("STATIC_FLUENTS_IN_OBJECT_ASSIGNMENTS")
        supported_kind.set_effects_kind("FLUENTS_IN_OBJECT_ASSIGNMENTS")
        supported_kind.set_conditions_kind('NEGATIVE_CONDITIONS')
        # supported_kind.set_conditions_kind('DISJUNCTIVE_CONDITIONS')
        supported_kind.set_conditions_kind('EQUALITIES')
        supported_kind.set_fluents_type('NUMERIC_FLUENTS')
        supported_kind.set_fluents_type('OBJECT_FLUENTS')
        supported_kind.set_fluents_type('INT_FLUENTS')
        return supported_kind

    @staticmethod
    def supports(problem_kind: 'up.model.ProblemKind') -> bool:
        return problem_kind <= TamerLite.supported_kind()

    @staticmethod
    def satisfies(optimality_guarantee: up.engines.OptimalityGuarantee) -> bool:
        return optimality_guarantee == up.engines.OptimalityGuarantee.SATISFICING

    def _get_heuristic(self, params, heuristic, encoder, state_encoder):
        if params is None:
            h = "custom" if heuristic else "hff"
        else:
            h = "custom" if heuristic and params.heuristic is None else params.heuristic if params.heuristic else "hff"
            rl_params = params.rl_params

        cache_h = False if params is None or params.cache_heuristic_in_state is None else params.cache_heuristic_in_state

        if h == "custom":
            def rewrite_h(search_state):
                return heuristic(StateWrapper(encoder.problem, search_state))
            h = CustomHeuristic(rewrite_h, cache_h)
            w = 1 if params is None or params.weight is None else params.weight
        elif h in ["rl_heuristic", "rl_rank"]:
            assert rl_params is not None and rl_params.other_params is not None
            if rl_params.other_params.residual:
                internal_heuristic_cache = getattr(rl_params.other_params, "internal_heuristic_cache", True)
                cache_heuristic_in_state = getattr(rl_params.other_params, "cache_heuristic_in_state", False)
                hsym_params = SearchParams(heuristic=rl_params.other_params.learning_heuristic, internal_heuristic_cache=internal_heuristic_cache, cache_heuristic_in_state=cache_heuristic_in_state)
                heuristic_for_residual, _ = self._get_heuristic(hsym_params, None, encoder, state_encoder)
            else:
                heuristic_for_residual = None
            if h == "rl_heuristic":
                h = RLHeuristic(state_encoder, rl_params.model, rl_params.model_class, rl_params.other_params, heuristic_for_residual, cache_h)
                w = 0.8 if params is None or params.weight is None else params.weight
            else:
                h = RLRank(state_encoder, rl_params.model, rl_params.model_class, rl_params.other_params, heuristic_for_residual, cache_h)
                w = 1 if params is None or params.weight is None else params.weight
        elif h == "hff":
            internal_heuristic_cache = True if params is None or params.internal_heuristic_cache is None else params.internal_heuristic_cache
            events = {a: e for a, e in encoder.events.items() if a in encoder.applicable_actions}
            h = HFF(encoder.fluents, encoder.objects, events, encoder.goal, internal_caching=internal_heuristic_cache, cache_value_in_state=cache_h)
            w = 0.8 if params is None or params.weight is None else params.weight
        elif h == "hadd":
            internal_heuristic_cache = True if params is None or params.internal_heuristic_cache is None else params.internal_heuristic_cache
            events = {a: e for a, e in encoder.events.items() if a in encoder.applicable_actions}
            h = HAdd(encoder.fluents, encoder.objects, events, encoder.goal, internal_caching=internal_heuristic_cache, cache_value_in_state=cache_h)
            w = 0.8 if params is None or params.weight is None else params.weight
        elif h == "hmax":
            internal_heuristic_cache = True if params is None or params.internal_heuristic_cache is None else params.internal_heuristic_cache
            events = {a: e for a, e in encoder.events.items() if a in encoder.applicable_actions}
            h = HMax(encoder.fluents, encoder.objects, events, encoder.goal, internal_caching=internal_heuristic_cache, cache_value_in_state=cache_h)
            w = 0.8 if params is None or params.weight is None else params.weight
        elif h == "hmax_numeric":
            internal_heuristic_cache = True if params is None or params.internal_heuristic_cache is None else params.internal_heuristic_cache
            events = {a: e for a, e in encoder.events.items() if a in encoder.applicable_actions}
            h = HMaxNumeric(encoder.fluents, encoder.objects, events, encoder.goal, internal_caching=internal_heuristic_cache, cache_value_in_state=cache_h)
            w = 0.8 if params is None or params.weight is None else params.weight
        elif h == "blind":
            h = CustomHeuristic(lambda x: 0.0, cache_h)
            w = 0
        else:
            raise NotImplementedError

        return h, w

    def _get_search(self, params, heuristic, encoder, state_encoder):
        if params is None:
            s = "wastar"
        else:
            s = "wastar" if params.search is None else params.search

        h, w = self._get_heuristic(params, heuristic, encoder, state_encoder)

        if s == "wastar":
            search = partial(wastar_search, heuristic=h, weight=w)
        elif s == "astar":
            search = partial(astar_search, heuristic=h)
        elif s == "gbfs":
            search = partial(gbfs_search, heuristic=h)
        elif s == "dfs":
            search = dfs_search
        elif s == "bfs":
            search = bfs_search
        elif s == "ehs":
            search = partial(ehc_search, heuristic=h)

        return search

    def _solve(self, problem: 'up.model.AbstractProblem',
               heuristic: Optional[Callable[["up.model.state.State"], Optional[float]]] = None,
               timeout: Optional[float] = None,
               output_stream: Optional[IO[str]] = None) -> 'up.engines.results.PlanGenerationResult':
        assert isinstance(problem, up.model.Problem)
        try:
            if self._params is not None and self._params.contains_rl():
                encoder, state_encoder, map_back_action_instance = get_encoders(self._params.domain(), problem)
            else:
                with problem.environment.factory.Compiler(compilation_kind="GROUNDING", problem_kind=problem.kind) as compiler:
                    compilation_res = compiler.compile(problem)
                    map_back_action_instance = compilation_res.map_back_action_instance
                new_problem = compilation_res.problem
                encoder = Encoder(new_problem)
                state_encoder = None

            if isinstance(self._params, MultiqueueParams):
                heuristics = []
                for p in self._params.queues:
                    h, w = self._get_heuristic(p, heuristic, encoder, state_encoder)
                    heuristics.append((h, w))
                plan, metrics = multiqueue_search(encoder.search_space, heuristics, timeout)
            else:
                search = self._get_search(self._params, heuristic, encoder, state_encoder)
                plan, metrics = search(encoder.search_space, timeout=timeout)

            if plan:
                plan = encoder.build_plan(plan)
                plan = plan.replace_action_instances(map_back_action_instance)
                status = up.engines.PlanGenerationResultStatus.SOLVED_SATISFICING
            else:
                status = up.engines.PlanGenerationResultStatus.UNSOLVABLE_INCOMPLETELY
            return up.engines.PlanGenerationResult(status, plan, self.name, metrics)
        except TimeoutError:
            status = up.engines.PlanGenerationResultStatus.TIMEOUT
            return up.engines.PlanGenerationResult(status, None, self.name)
