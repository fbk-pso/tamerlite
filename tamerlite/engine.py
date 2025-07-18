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

from dataclasses import dataclass
from functools import partial
import unified_planning as up
import unified_planning.model
import unified_planning.engines
import unified_planning.engines.mixins
from unified_planning.model import ProblemKind, FNode
from unified_planning.model.state import State
from typing import IO, Any, Callable, List, Optional, Union

from tamerlite.core import use_rustamer
from tamerlite.core import wastar_search, astar_search, gbfs_search
from tamerlite.core import bfs_search, dfs_search, ehc_search
from tamerlite.core import multiqueue_search
from tamerlite.core import evaluate, make_fluent_node
from tamerlite.core import HFF, HAdd, HMax, HMaxNumeric, CustomHeuristic
from tamerlite.encoder import Encoder


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
class SearchParams:
    search: Optional[str] = None
    heuristic: Optional[str] = None
    internal_heuristic_cache: Optional[bool] = None
    weight: Optional[str] = None
    early_termination: Optional[bool] = None


@dataclass(frozen=True)
class MultiqueueParams:
    queues: List[SearchParams]
    early_termination: Optional[bool] = None     # the parameter early_termination inside queues is ignored


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
        supported_kind.set_conditions_kind('DISJUNCTIVE_CONDITIONS')
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

    def _get_heuristic(self, params, heuristic, encoder, problem_has_disjunctive_conditions):
        default_heuristic = "hmax_numeric" if problem_has_disjunctive_conditions else "hff"
        if params is None:
            h = "custom" if heuristic else default_heuristic
        else:
            h = "custom" if heuristic and params.heuristic is None else params.heuristic if params.heuristic else default_heuristic

        cache_h = False # useful only for residual heuristics

        if h == "custom":
            def rewrite_h(search_state):
                return heuristic(StateWrapper(encoder.problem, search_state))
            heuristic = CustomHeuristic(rewrite_h, cache_h)
            w = 1 if params is None or params.weight is None else params.weight
        elif h == "hff":
            internal_heuristic_cache = True if params is None or params.internal_heuristic_cache is None else params.internal_heuristic_cache
            events = {a: e for a, e in encoder.events.items() if a in encoder.applicable_actions}
            heuristic = HFF(encoder.fluents, encoder.objects, events, encoder.goal, internal_caching=internal_heuristic_cache, cache_value_in_state=cache_h)
            w = 0.8 if params is None or params.weight is None else params.weight
        elif h == "hadd":
            internal_heuristic_cache = True if params is None or params.internal_heuristic_cache is None else params.internal_heuristic_cache
            events = {a: e for a, e in encoder.events.items() if a in encoder.applicable_actions}
            heuristic = HAdd(encoder.fluents, encoder.objects, events, encoder.goal, internal_caching=internal_heuristic_cache, cache_value_in_state=cache_h)
            w = 0.8 if params is None or params.weight is None else params.weight
        elif h == "hmax":
            internal_heuristic_cache = True if params is None or params.internal_heuristic_cache is None else params.internal_heuristic_cache
            events = {a: e for a, e in encoder.events.items() if a in encoder.applicable_actions}
            heuristic = HMax(encoder.fluents, encoder.objects, events, encoder.goal, internal_caching=internal_heuristic_cache, cache_value_in_state=cache_h)
            w = 0.8 if params is None or params.weight is None else params.weight
        elif h == "hmax_numeric":
            internal_heuristic_cache = True if params is None or params.internal_heuristic_cache is None else params.internal_heuristic_cache
            events = {a: e for a, e in encoder.events.items() if a in encoder.applicable_actions}
            heuristic = HMaxNumeric(encoder.fluents, encoder.objects, events, encoder.goal, internal_caching=internal_heuristic_cache, cache_value_in_state=cache_h)
            w = 0.8 if params is None or params.weight is None else params.weight
        elif h == "blind":
            heuristic = CustomHeuristic(lambda x: 0.0, cache_h)
            w = 0
        else:
            raise NotImplementedError

        return heuristic, w

    def _get_search(self, params, heuristic, weight):
        if params is None or params.search is None:
            s = "wastar"
        else:
            s = params.search

        if s == "wastar":
            search = partial(wastar_search, heuristic=heuristic, weight=weight)
        elif s == "astar":
            search = partial(astar_search, heuristic=heuristic)
        elif s == "gbfs":
            search = partial(gbfs_search, heuristic=heuristic)
        elif s == "dfs":
            search = dfs_search
        elif s == "bfs":
            search = bfs_search
        elif s == "ehs":
            search = partial(ehc_search, heuristic=heuristic)

        return s, search

    def _solve(self, problem: 'up.model.AbstractProblem',
               heuristic: Optional[Callable[["up.model.state.State"], Optional[float]]] = None,
               timeout: Optional[float] = None,
               output_stream: Optional[IO[str]] = None) -> 'up.engines.results.PlanGenerationResult':
        assert isinstance(problem, up.model.Problem)
        has_disjunctive_conditions = problem.kind.has_disjunctive_conditions()
        try:
            with problem.environment.factory.Compiler(compilation_kind="GROUNDING", problem_kind=problem.kind) as compiler:
                compilation_res = compiler.compile(problem)
                map_back_action_instance = compilation_res.map_back_action_instance
            new_problem = compilation_res.problem
            encoder = Encoder(new_problem, encode_fluents=use_rustamer)

            early_termination = False
            if self._params is not None and self._params.early_termination is not None:
                early_termination = self._params.early_termination

            if isinstance(self._params, MultiqueueParams):
                heuristics = []
                for p in self._params.queues:
                    h, w = self._get_heuristic(p, heuristic, encoder, has_disjunctive_conditions)
                    heuristics.append((h, w))
                    if has_disjunctive_conditions and h.name in ("hff", "hadd", "hmax"):
                        status = up.engines.PlanGenerationResultStatus.UNSUPPORTED_PROBLEM
                        return up.engines.PlanGenerationResult(status, None, self.name)
                plan, metrics = multiqueue_search(encoder.search_space, heuristics, timeout, early_termination=early_termination)
            else:
                h, w = self._get_heuristic(self._params, heuristic, encoder, has_disjunctive_conditions)
                s_name, search = self._get_search(self._params, h, w)
                if (
                    has_disjunctive_conditions
                    and h.name in ("hff", "hadd", "hmax")
                    and s_name in ("wastar", "astar", "gbfs", "ehs")
                ):
                    status = up.engines.PlanGenerationResultStatus.UNSUPPORTED_PROBLEM
                    return up.engines.PlanGenerationResult(status, None, self.name)
                plan, metrics = search(encoder.search_space, timeout=timeout, early_termination=early_termination)

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
