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
from fractions import Fraction
from functools import partial
import time
import unified_planning as up
import unified_planning.engines
import unified_planning.engines.mixins
from unified_planning.model import ProblemKind, FNode, StartTiming
from unified_planning.model.state import State
from unified_planning.engines.compilers.timed_to_sequential import TimedToSequential
from unified_planning.engines.plan_validator import (
    SequentialPlanValidator,
    TimeTriggeredPlanValidator,
)
from unified_planning.engines.compilers.utils import get_fresh_name
from unified_planning.plans import ActionInstance, PlanKind
from typing import IO, Callable, Iterator, List, Optional, Union, Tuple, Dict
import warnings

from tamerlite.core import search_space
from tamerlite.core import wastar_search, astar_search, gbfs_search
from tamerlite.core import (
    wastar_search_memory_bounded,
    astar_search_memory_bounded,
    gbfs_search_memory_bounded,
)
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
    inadmissible_numeric_heuristic_variant: bool = False
    early_termination: bool = False
    weak_equality: bool = False
    symmetry_breaking: bool = True
    compression_safe_actions: bool = True
    relevance_analysis: bool = True
    incomplete_memory_bounded_search: bool = False


@dataclass(frozen=True)
class MultiqueueParams:
    queues: List[HeuristicParams]
    internal_heuristic_cache: bool = True
    inadmissible_numeric_heuristic_variant: bool = False
    early_termination: bool = False
    weak_equality: bool = False
    symmetry_breaking: bool = True
    compression_safe_actions: bool = True
    relevance_analysis: bool = True


class TamerLite(
    unified_planning.engines.Engine,
    unified_planning.engines.mixins.OneshotPlannerMixin,
    unified_planning.engines.mixins.AnytimePlannerMixin,
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
        supported_kind.set_typing("HIERARCHICAL_TYPING")
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
        supported_kind.set_conditions_kind("EXISTENTIAL_CONDITIONS")
        supported_kind.set_conditions_kind("UNIVERSAL_CONDITIONS")
        supported_kind.set_fluents_type("NUMERIC_FLUENTS")
        supported_kind.set_fluents_type("OBJECT_FLUENTS")
        supported_kind.set_fluents_type("INT_FLUENTS")
        supported_kind.set_fluents_type("REAL_FLUENTS")
        supported_kind.set_quality_metrics("ACTIONS_COST")
        supported_kind.set_quality_metrics("FINAL_VALUE")
        supported_kind.set_quality_metrics("MAKESPAN")
        supported_kind.set_quality_metrics("PLAN_LENGTH")
        supported_kind.set_initial_state("UNDEFINED_INITIAL_NUMERIC")
        return supported_kind

    @staticmethod
    def supports(problem_kind: "up.model.ProblemKind") -> bool:
        return problem_kind <= TamerLite.supported_kind()

    @staticmethod
    def satisfies(optimality_guarantee: up.engines.OptimalityGuarantee) -> bool:
        return optimality_guarantee == up.engines.OptimalityGuarantee.SATISFICING

    @staticmethod
    def ensures(anytime_guarantee: up.engines.AnytimeGuarantee) -> bool:
        return anytime_guarantee == up.engines.AnytimeGuarantee.INCREASING_QUALITY

    def _get_heuristic(
        self,
        params: HeuristicParams,
        heuristic: Optional[Callable[[State], Optional[float]]],
        encoder: Encoder,
        inadmissible_numeric_heuristic_variant: bool,
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
                raise NotImplementedError(
                    f"Unknown heuristic '{h_name}'. "
                    f"Supported values are: custom, blind, {', '.join(sorted(hh_map))}."
                )

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
                inadmissible_numeric_heuristic_variant=inadmissible_numeric_heuristic_variant,
            )
            w = 0.8 if params.weight is None else params.weight

        return h, w

    def _get_search(
        self,
        search_name: Optional[str],
        heuristic: Heuristic,
        weight: float,
        incomplete_memory_bounded_search: bool,
        weak_equality: bool,
        is_temporal: bool,
    ) -> Tuple[
        str,
        Callable[
            [search_space.SearchSpaceABC, Optional[float], bool],
            Tuple[Optional[PlanType], Dict[str, str]],
        ],
    ]:
        if (
            (search_name is None or search_name in {"wastar", "astar", "gbfs"})
            and incomplete_memory_bounded_search
            and is_temporal
            and weak_equality
        ):
            warnings.warn(
                "Memory-bounded search does not support weak equality correctly."
            )

        if search_name is None or search_name == "wastar":
            if incomplete_memory_bounded_search:
                search = partial(
                    wastar_search_memory_bounded, heuristic=heuristic, weight=weight
                )
            else:
                search = partial(wastar_search, heuristic=heuristic, weight=weight)
        elif search_name == "astar":
            if incomplete_memory_bounded_search:
                search = partial(astar_search_memory_bounded, heuristic=heuristic)
            else:
                search = partial(astar_search, heuristic=heuristic)
        elif search_name == "gbfs":
            if incomplete_memory_bounded_search:
                search = partial(gbfs_search_memory_bounded, heuristic=heuristic)
            else:
                search = partial(gbfs_search, heuristic=heuristic)
        elif search_name == "dfs":
            search = partial(dfs_search)
        elif search_name == "bfs":
            search = partial(bfs_search)
        elif search_name == "ehs":
            search = partial(ehc_search, heuristic=heuristic)
        else:
            raise NotImplementedError(
                f"Unknown search '{search_name}'. "
                "Supported values are: wastar, astar, gbfs, dfs, bfs, ehs."
            )

        return search_name, search  # type: ignore[return-value]

    def _compile_problem(self, problem: "up.model.AbstractProblem"):
        with problem.environment.factory.Compiler(
            compilation_kind="UNDEFINED_INITIAL_NUMERIC_REMOVING",
            problem_kind=problem.kind,
        ) as compiler:
            compilation_res = compiler.compile(problem)
            undefined_map_back_action_instance = (
                compilation_res.map_back_action_instance
            )
            problem = compilation_res.problem

        with problem.environment.factory.Compiler(
            compilation_kind="GROUNDING", problem_kind=problem.kind
        ) as compiler:
            compilation_res = compiler.compile(problem)
            ground_map_back_action_instance = compilation_res.map_back_action_instance
            ground_problem = compilation_res.problem
            lifted_problem = problem

        map_back_action_instance = lambda ai: undefined_map_back_action_instance(
            ground_map_back_action_instance(ai)
        )

        return lifted_problem, ground_problem, map_back_action_instance

    def _get_solutions_with_params(
        self,
        problem: "up.model.AbstractProblem",
        timeout: Optional[float] = None,
        output_stream: Optional[IO[str]] = None,
        warm_start_plan: Optional["up.plans.Plan"] = None,
        **kwargs,
    ) -> Iterator["up.engines.results.PlanGenerationResult"]:
        assert isinstance(problem, up.model.Problem)
        if len(problem.quality_metrics) > 1:
            raise NotImplementedError("Multiple quality metrics are not supported")

        start_time = time.time()
        em = problem.environment.expression_manager
        tm = problem.environment.type_manager

        lifted_problem, ground_problem, map_back_action_instance = (
            self._compile_problem(problem)
        )
        original_problem = problem

        elapsed_time = time.time() - start_time
        res, _, _ = self._solve_ground_problem(
            lifted_problem,
            ground_problem,
            map_back_action_instance,
            timeout=timeout - elapsed_time if timeout is not None else None,
            output_stream=output_stream,
            is_intermediate_solution=True,
        )
        yield res
        if res.plan is None:
            return

        if len(ground_problem.quality_metrics) == 0:
            if res.plan.kind == PlanKind.SEQUENTIAL_PLAN:
                quality_metric = up.model.metrics.MinimizeSequentialPlanLength()
            elif res.plan.kind == PlanKind.TIME_TRIGGERED_PLAN:
                quality_metric = up.model.metrics.MinimizeMakespan()
            else:
                raise AssertionError(f"Unknown plan type {res.plan.kind}")

            ground_problem.add_quality_metric(quality_metric)
        else:
            quality_metric = None

        if res.plan.kind == PlanKind.SEQUENTIAL_PLAN:
            validator = SequentialPlanValidator()
        elif res.plan.kind == PlanKind.TIME_TRIGGERED_PLAN:
            validator = TimeTriggeredPlanValidator()
        else:
            raise AssertionError(f"Unknown plan type {res.plan.kind}")

        def validate_plan(
            problem: up.model.AbstractProblem, plan: up.plans.Plan
        ) -> up.engines.results.ValidationResult:
            if quality_metric is not None:
                problem.add_quality_metric(quality_metric)
                res = validator.validate(problem, plan)
                problem.clear_quality_metrics()
            else:
                res = validator.validate(problem, plan)
            return res

        prev_res = up.engines.PlanGenerationResult(
            res.status, res.plan, res.engine_name, res.metrics, res.log_messages
        )
        while res.status == up.engines.PlanGenerationResultStatus.INTERMEDIATE:
            val_res = validate_plan(original_problem, res.plan)
            assert val_res
            assert (
                val_res.metric_evaluations is not None
                and len(val_res.metric_evaluations) == 1
            ), "Expected metric evaluations for plan validation result"

            problem = ground_problem.clone()
            exp = None
            deadline = None
            m, v = list(val_res.metric_evaluations.items())[0]
            if m.is_minimize_expression_on_final_state():
                exp = em.LT(m.expression, v)
            elif m.is_maximize_expression_on_final_state():
                exp = em.GT(m.expression, v)
            elif m.is_minimize_sequential_plan_length():
                plan_length = up.model.Fluent(
                    get_fresh_name(problem, "plan_length"), tm.IntType(0)
                )
                problem.add_fluent(plan_length, default_initial_value=0)
                for a in problem.actions:
                    if isinstance(a, up.model.InstantaneousAction):
                        a.add_increase_effect(plan_length, 1)
                    else:
                        raise NotImplementedError(
                            "Only instantaneous actions supported for plan length metric"
                        )
                exp = em.LT(plan_length, v)
            elif m.is_minimize_action_costs():
                m = list(problem.quality_metrics)[0]
                actions_cost = up.model.Fluent(
                    get_fresh_name(problem, "actions_cost"),
                    tm.RealType(lower_bound=0.0),
                )
                problem.add_fluent(actions_cost, default_initial_value=0)
                for a in problem.actions:
                    cost = m.costs.get(a, m.default)
                    if cost is None:
                        continue
                    if isinstance(a, up.model.InstantaneousAction):
                        a.add_increase_effect(actions_cost, cost)
                    elif isinstance(a, up.model.DurativeAction):
                        a.add_increase_effect(StartTiming(), actions_cost, cost)
                    else:
                        raise AssertionError(f"Unknown action type {type(a)}")
                exp = em.LT(actions_cost, v)
            elif m.is_minimize_makespan():
                deadline = Fraction(v)
            else:
                raise NotImplementedError(f"Unknown metric type for metric {m}")

            if exp is not None:
                for a in problem.actions:
                    if isinstance(a, up.model.InstantaneousAction):
                        a.add_precondition(exp)
                    elif isinstance(a, up.model.DurativeAction):
                        a.add_condition(StartTiming(), exp)
                problem.add_goal(exp)
            else:
                assert deadline is not None

            ground_problem_actions = {a.name: a for a in ground_problem.actions}

            def new_map_back_action_instance(
                ai: ActionInstance,
            ) -> Optional[ActionInstance]:
                action = ground_problem_actions.get(ai.action.name, None)
                if action is not None:
                    return map_back_action_instance(action())
                return None

            elapsed_time = time.time() - start_time
            res, solution_might_exist, is_any_action_compression_safe = (
                self._solve_ground_problem(
                    lifted_problem,
                    problem,
                    new_map_back_action_instance,
                    timeout=timeout - elapsed_time if timeout is not None else None,
                    output_stream=output_stream,
                    deadline=deadline,
                    is_intermediate_solution=True,
                )
            )
            if (
                res.status
                == up.engines.PlanGenerationResultStatus.UNSOLVABLE_INCOMPLETELY
            ):
                if (
                    len(lifted_problem.quality_metrics) == 0
                    or solution_might_exist
                    or (m.is_minimize_makespan() and is_any_action_compression_safe)
                ):
                    prev_res.status = (
                        up.engines.PlanGenerationResultStatus.SOLVED_SATISFICING
                    )
                else:
                    prev_res.status = (
                        up.engines.PlanGenerationResultStatus.SOLVED_OPTIMALLY
                    )
                yield prev_res
            elif res.status == up.engines.PlanGenerationResultStatus.TIMEOUT:
                prev_res.status = up.engines.PlanGenerationResultStatus.TIMEOUT
                yield prev_res
            else:
                assert res.plan is not None
                yield res

            prev_res = up.engines.PlanGenerationResult(
                res.status, res.plan, res.engine_name, res.metrics, res.log_messages
            )

    def _get_solutions(
        self,
        problem: "up.model.AbstractProblem",
        timeout: Optional[float] = None,
        output_stream: Optional[IO[str]] = None,
    ) -> Iterator[up.engines.results.PlanGenerationResult]:
        return self._get_solutions_with_params(problem, timeout, output_stream)

    def _solve(
        self,
        problem: "up.model.AbstractProblem",
        heuristic: Optional[Callable[[State], Optional[float]]] = None,
        timeout: Optional[float] = None,
        output_stream: Optional[IO[str]] = None,
    ) -> "up.engines.results.PlanGenerationResult":
        assert isinstance(problem, up.model.Problem)
        start_time = time.time()
        lifted_problem, ground_problem, map_back_action_instance = (
            self._compile_problem(problem)
        )
        elapsed_time = time.time() - start_time
        res, _, _ = self._solve_ground_problem(
            lifted_problem,
            ground_problem,
            map_back_action_instance,
            heuristic=heuristic,
            timeout=timeout - elapsed_time if timeout is not None else None,
            output_stream=output_stream,
            is_intermediate_solution=False,
        )
        return res

    def _solve_ground_problem(
        self,
        problem: "up.model.Problem",
        ground_problem: "up.model.Problem",
        map_back_action_instance: Callable[[ActionInstance], Optional[ActionInstance]],
        heuristic: Optional[Callable[[State], Optional[float]]] = None,
        timeout: Optional[float] = None,
        output_stream: Optional[IO[str]] = None,
        deadline: Optional[Fraction] = None,
        is_intermediate_solution: bool = False,
    ) -> Tuple["up.engines.results.PlanGenerationResult", bool, bool]:
        try:
            encoder = Encoder(
                ground_problem,
                problem,
                map_back_action_instance,
                self._params.symmetry_breaking,
                self._params.compression_safe_actions,
                self._params.relevance_analysis,
                deadline=deadline,
            )

            original_encoder = encoder
            are_all_actions_compression_safe = (
                not is_intermediate_solution
                and encoder.are_all_actions_compression_safe()
            )
            is_any_action_compression_safe = (
                are_all_actions_compression_safe
                or encoder.is_any_action_compression_safe()
            )
            if are_all_actions_compression_safe:
                # Compile a temporal planning problem, where all actions are safe to compress,
                # into an equivalent classical planning problem
                t2s_compiler = TimedToSequential()
                t2s_compiler.skip_checks = True
                compilation_res = t2s_compiler.compile(ground_problem)
                ground_problem_actions = {a.name: a for a in ground_problem.actions}

                def new_map_back_action_instance(
                    ai: ActionInstance,
                ) -> Optional[ActionInstance]:
                    action = ground_problem_actions.get(ai.action.name, None)
                    if action is not None:
                        return map_back_action_instance(action())
                    return None

                encoder = Encoder(
                    compilation_res.problem,
                    problem,
                    new_map_back_action_instance,
                    self._params.symmetry_breaking,
                    self._params.compression_safe_actions,
                    self._params.relevance_analysis,
                    deadline=deadline,
                )

            if isinstance(self._params, MultiqueueParams):
                search_name = "multiqueue"
                heuristics = []
                for p in self._params.queues:
                    h, w = self._get_heuristic(
                        p,
                        heuristic,
                        encoder,
                        self._params.inadmissible_numeric_heuristic_variant,
                        self._params.internal_heuristic_cache,
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
                    self._params.inadmissible_numeric_heuristic_variant,
                    self._params.internal_heuristic_cache,
                )
                search_name, search = self._get_search(
                    self._params.search,
                    h,
                    w,
                    self._params.incomplete_memory_bounded_search,
                    self._params.weak_equality,
                    encoder.search_space.is_temporal,
                )

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
                status = (
                    up.engines.PlanGenerationResultStatus.INTERMEDIATE
                    if is_intermediate_solution
                    else up.engines.PlanGenerationResultStatus.SOLVED_SATISFICING
                )
                solution_might_exist = True
            else:
                plan = None
                status = up.engines.PlanGenerationResultStatus.UNSOLVABLE_INCOMPLETELY
                solution_might_exist = search_name == "ehs"
            return (
                up.engines.PlanGenerationResult(status, plan, self.name, metrics),
                solution_might_exist,
                is_any_action_compression_safe,
            )
        except TimeoutError:
            status = up.engines.PlanGenerationResultStatus.TIMEOUT
            return (
                up.engines.PlanGenerationResult(status, None, self.name),
                True,
                is_any_action_compression_safe,
            )
