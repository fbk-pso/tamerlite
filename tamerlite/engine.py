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
import time
import unified_planning as up
import unified_planning.engines
import unified_planning.engines.mixins
from unified_planning.model import ProblemKind, FNode
from unified_planning.model.state import State
from unified_planning.engines.compilers.timed_to_sequential import TimedToSequential
from unified_planning.plans import ActionInstance
from typing import IO, Callable, List, Optional, Union, Tuple, Dict

from tamerlite.core import search_space
from tamerlite.core import wastar_search, astar_search, gbfs_search
from tamerlite.core import bfs_search, dfs_search, ehc_search
from tamerlite.core import multiqueue_search
from tamerlite.core import evaluate, make_fluent_node
from tamerlite.core import HFF, HAdd, HMax, HMaxExplicit, CustomHeuristic
from tamerlite.core.heuristics import Heuristic
from tamerlite.encoder import Encoder, PlanType


credits = up.engines.Credits(
    "TamerLite",
    "FBK PSO Unit",
    "tamer@fbk.eu",
    "https://github.com/fbk-pso/tamerlite",
    "LGPLv3",
    "Heuristic search-based temporal planner.",
    "Heuristic search-based temporal planner designed to address planning problems with rich temporal dynamics.",
)


class StateWrapper(State):
    def __init__(self, encoder: Encoder, search_state: search_space.State):
        self.encoder = encoder
        self.search_state = search_state
        self.problem = encoder.problem
        self.em = self.problem.environment.expression_manager

    def get_value(self, x: FNode) -> FNode:
        key = (make_fluent_node(self.encoder.fluent_ids[str(x)]),)
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
class HeuristicParams:
    heuristic: Optional[str] = None
    weight: Optional[float] = None


@dataclass(frozen=True)
class SearchParams(HeuristicParams):
    search: Optional[str] = None
    internal_heuristic_cache: bool = True
    early_termination: bool = False
    weak_equality: bool = False
    symmetry_breaking: bool = True
    compression_safe_actions: bool = True


@dataclass(frozen=True)
class MultiqueueParams:
    queues: List[HeuristicParams]
    internal_heuristic_cache: bool = True
    early_termination: bool = False
    weak_equality: bool = False
    symmetry_breaking: bool = True
    compression_safe_actions: bool = True


class TamerLite(
    unified_planning.engines.Engine,
    unified_planning.engines.mixins.OneshotPlannerMixin,
):

    def __init__(self, search: Union[SearchParams, MultiqueueParams] = SearchParams()):
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
        supported_kind.set_problem_class("ACTION_BASED")
        supported_kind.set_time("CONTINUOUS_TIME")
        supported_kind.set_time("INTERMEDIATE_CONDITIONS_AND_EFFECTS")
        supported_kind.set_time("DURATION_INEQUALITIES")
        supported_kind.set_expression_duration("STATIC_FLUENTS_IN_DURATIONS")
        supported_kind.set_expression_duration("FLUENTS_IN_DURATIONS")
        supported_kind.set_expression_duration("INT_TYPE_DURATIONS")
        supported_kind.set_numbers("DISCRETE_NUMBERS")
        supported_kind.set_numbers("CONTINUOUS_NUMBERS")
        supported_kind.set_problem_type("SIMPLE_NUMERIC_PLANNING")
        supported_kind.set_problem_type("GENERAL_NUMERIC_PLANNING")
        supported_kind.set_typing("FLAT_TYPING")
        supported_kind.set_parameters("BOOL_FLUENT_PARAMETERS")
        supported_kind.set_parameters("BOUNDED_INT_FLUENT_PARAMETERS")
        supported_kind.set_parameters("BOOL_ACTION_PARAMETERS")
        supported_kind.set_parameters("BOUNDED_INT_ACTION_PARAMETERS")
        supported_kind.set_effects_kind("INCREASE_EFFECTS")
        supported_kind.set_effects_kind("DECREASE_EFFECTS")
        supported_kind.set_effects_kind("STATIC_FLUENTS_IN_BOOLEAN_ASSIGNMENTS")
        supported_kind.set_effects_kind("FLUENTS_IN_BOOLEAN_ASSIGNMENTS")
        supported_kind.set_effects_kind("STATIC_FLUENTS_IN_NUMERIC_ASSIGNMENTS")
        supported_kind.set_effects_kind("FLUENTS_IN_NUMERIC_ASSIGNMENTS")
        supported_kind.set_effects_kind("STATIC_FLUENTS_IN_OBJECT_ASSIGNMENTS")
        supported_kind.set_effects_kind("FLUENTS_IN_OBJECT_ASSIGNMENTS")
        supported_kind.set_conditions_kind("NEGATIVE_CONDITIONS")
        supported_kind.set_conditions_kind("DISJUNCTIVE_CONDITIONS")
        supported_kind.set_conditions_kind("EQUALITIES")
        supported_kind.set_fluents_type("NUMERIC_FLUENTS")
        supported_kind.set_fluents_type("OBJECT_FLUENTS")
        supported_kind.set_fluents_type("INT_FLUENTS")
        return supported_kind

    @staticmethod
    def supports(problem_kind: "up.model.ProblemKind") -> bool:
        return problem_kind <= TamerLite.supported_kind()

    @staticmethod
    def satisfies(optimality_guarantee: up.engines.OptimalityGuarantee) -> bool:
        return optimality_guarantee == up.engines.OptimalityGuarantee.SATISFICING

    def _get_heuristic(
        self,
        params: HeuristicParams,
        heuristic: Optional[Callable[[State], Optional[float]]],
        encoder: Encoder,
        internal_heuristic_cache: bool,
        cache_heuristic_in_state: bool = False,
    ) -> Tuple[Heuristic, float]:
        assert encoder.goal is not None
        if params.heuristic is None:
            h_name = "custom" if heuristic is not None else "hff"
        else:
            h_name = params.heuristic

        if h_name == "custom":

            def rewrite_h(search_state: search_space.State):
                return heuristic(StateWrapper(encoder, search_state))  # type: ignore[misc]

            h = CustomHeuristic(rewrite_h, cache_heuristic_in_state)  # type: ignore[assignment]
            w = 1.0 if params.weight is None else params.weight

        elif h_name == "blind":
            h = CustomHeuristic(lambda x: 0.0, cache_heuristic_in_state)  # type: ignore[assignment]
            w = 0.0

        else:
            hh_map = {
                "hff": HFF,
                "hadd": HAdd,
                "hmax": HMax,
                "hmax_explicit": HMaxExplicit,
                "hff_no_numbers": partial(HFF, disable_numeric_reasoning=True),
                "hadd_no_numbers": partial(HAdd, disable_numeric_reasoning=True),
                "hmax_no_numbers": partial(HMax, disable_numeric_reasoning=True),
            }
            if h_name not in hh_map:
                raise NotImplementedError

            events = {
                a: e
                for a, e in encoder.events.items()
                if a in encoder.applicable_actions
            }
            h = hh_map[h_name](  # type: ignore
                encoder.actions,
                encoder.fluent_types,
                encoder.objects,
                events,
                encoder.goal,
                internal_caching=internal_heuristic_cache,
                cache_value_in_state=cache_heuristic_in_state,
            )
            w = 0.8 if params.weight is None else params.weight

        return h, w

    def _get_search(
        self, search_name: Optional[str], heuristic: Heuristic, weight: float
    ) -> Tuple[
        str,
        Callable[
            [search_space.SearchSpaceABC, Optional[float], bool],
            Tuple[Optional[PlanType], Dict[str, str]],
        ],
    ]:
        if search_name is None or search_name == "wastar":
            search = partial(wastar_search, heuristic=heuristic, weight=weight)
        elif search_name == "astar":
            search = partial(astar_search, heuristic=heuristic)
        elif search_name == "gbfs":
            search = partial(gbfs_search, heuristic=heuristic)
        elif search_name == "dfs":
            search = partial(dfs_search)
        elif search_name == "bfs":
            search = partial(bfs_search)
        elif search_name == "ehs":
            search = partial(ehc_search, heuristic=heuristic)

        return search_name, search  # type: ignore[return-value]

    def _solve(
        self,
        problem: "up.model.AbstractProblem",
        heuristic: Optional[Callable[[State], Optional[float]]] = None,
        timeout: Optional[float] = None,
        output_stream: Optional[IO[str]] = None,
    ) -> "up.engines.results.PlanGenerationResult":
        assert isinstance(problem, up.model.Problem)
        try:
            with problem.environment.factory.Compiler(
                compilation_kind="GROUNDING", problem_kind=problem.kind
            ) as compiler:
                compilation_res = compiler.compile(problem)
                map_back_action_instance = compilation_res.map_back_action_instance
            new_problem = compilation_res.problem
            encoder = Encoder(
                new_problem,
                problem,
                map_back_action_instance,
                self._params.symmetry_breaking,
                self._params.compression_safe_actions,
            )

            original_encoder = encoder
            are_all_actions_compression_safe = (
                encoder.are_all_actions_compression_safe()
            )
            if are_all_actions_compression_safe:
                # Compile a temporal planning problem, where all actions are safe to compress,
                # into an equivalent classical planning problem
                t2s_compiler = TimedToSequential()
                t2s_compiler.skip_checks = True
                compilation_res = t2s_compiler.compile(new_problem)
                new_problem_actions = {a.name: a for a in new_problem.actions}

                def new_map_back_action_instance(
                    ai: ActionInstance,
                ) -> Optional[ActionInstance]:
                    action = new_problem_actions.get(ai.action.name, None)
                    if action is not None:
                        return map_back_action_instance(action())
                    return None

                encoder = Encoder(
                    compilation_res.problem,
                    problem,
                    new_map_back_action_instance,
                    self._params.symmetry_breaking,
                    self._params.compression_safe_actions,
                )

            if isinstance(self._params, MultiqueueParams):
                heuristics = []
                for p in self._params.queues:
                    h, w = self._get_heuristic(
                        p, heuristic, encoder, self._params.internal_heuristic_cache
                    )
                    heuristics.append((h, w))

                start = time.time()
                path, metrics = multiqueue_search(
                    encoder.search_space,
                    heuristics,
                    timeout,
                    early_termination=self._params.early_termination,
                    weak_equality=self._params.weak_equality,
                )
                if self._params.weak_equality and path is None:
                    updated_timeout = timeout
                    if updated_timeout is not None:
                        updated_timeout -= start
                    path, metrics = multiqueue_search(
                        encoder.search_space,
                        heuristics,
                        updated_timeout,
                        early_termination=self._params.early_termination,
                        weak_equality=False,
                    )
            else:
                h, w = self._get_heuristic(
                    self._params,
                    heuristic,
                    encoder,
                    self._params.internal_heuristic_cache,
                )
                search_name, search = self._get_search(self._params.search, h, w)

                if self._params.weak_equality and search_name not in ("dfs", "bfs"):
                    start = time.time()
                    path, metrics = search(  # type: ignore
                        encoder.search_space,
                        timeout=timeout,
                        early_termination=self._params.early_termination,
                        weak_equality=True,
                    )
                    if path is None:
                        updated_timeout = timeout
                        if updated_timeout is not None:
                            updated_timeout -= start
                        path, metrics = search(  # type: ignore
                            encoder.search_space,
                            timeout=updated_timeout,
                            early_termination=self._params.early_termination,
                            weak_equality=False,
                        )
                else:
                    path, metrics = search(  # type: ignore
                        encoder.search_space,
                        timeout=timeout,
                        early_termination=self._params.early_termination,
                    )

            if path is not None:
                if are_all_actions_compression_safe:
                    compressed_path = path
                    path = []
                    for action in compressed_path:
                        action = original_encoder.get_action(
                            encoder.action_names[action.idx]
                        )
                        for _ in original_encoder.events[action]:
                            path.append(action)

                plan = original_encoder.build_plan(path)
                plan = plan.replace_action_instances(map_back_action_instance)
                status = up.engines.PlanGenerationResultStatus.SOLVED_SATISFICING
            else:
                plan = None
                status = up.engines.PlanGenerationResultStatus.UNSOLVABLE_INCOMPLETELY
            return up.engines.PlanGenerationResult(status, plan, self.name, metrics)
        except TimeoutError:
            status = up.engines.PlanGenerationResultStatus.TIMEOUT
            return up.engines.PlanGenerationResult(status, None, self.name)
