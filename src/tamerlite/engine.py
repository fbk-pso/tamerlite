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

import logging
import time
import warnings
from collections.abc import Callable, Iterator, MutableMapping
from dataclasses import dataclass
from fractions import Fraction
from functools import partial
from typing import IO, Any, Protocol, cast

import unified_planning as up
import unified_planning.engines
import unified_planning.engines.mixins
from unified_planning.engines.compilers.grounder import Grounder
from unified_planning.engines.compilers.timed_to_sequential import TimedToSequential
from unified_planning.engines.compilers.undefined_initial_numeric_remover import (
    UndefinedInitialNumericRemover,
)
from unified_planning.engines.compilers.utils import get_fresh_name
from unified_planning.engines.plan_validator import (
    SequentialPlanValidator,
    TimeTriggeredPlanValidator,
)
from unified_planning.exceptions import UPStateMissingFluentError
from unified_planning.model import FNode, InterpretedFunction, ProblemKind, StartTiming
from unified_planning.model.state import State
from unified_planning.plans import ActionInstance, PlanKind

from tamerlite.converter import interpreted_function_scope, new_if_cache
from tamerlite.core import (
    HFF,
    Action,
    CustomHeuristic,
    HAdd,
    HMax,
    HMaxExplicit,
    astar_search,
    astar_search_memory_bounded,
    bfs_search,
    dfs_search,
    ehc_search,
    gbfs_search,
    gbfs_search_memory_bounded,
    get_fluent_value,
    multiqueue_search,
    search_space,
    wastar_search,
    wastar_search_memory_bounded,
)
from tamerlite.core.heuristics import Heuristic
from tamerlite.encoder import Encoder

logger = logging.getLogger(__name__)

credits = up.engines.Credits(
    "TamerLite",
    "FBK PSO Unit",
    "pso-tools@fbk.eu",
    "https://github.com/fbk-pso/tamerlite",
    "GPLv3",
    "Heuristic search-based temporal planner.",
    "Heuristic search-based temporal planner designed to address planning "
    "problems with rich temporal dynamics.",
)


class StateWrapper(State):
    def __init__(self, encoder: Encoder, state: search_space.State):
        self.encoder = encoder
        self.state = state
        self.problem = encoder.problem
        self.em = self.problem.environment.expression_manager

    def get_value(self, fluent: FNode) -> FNode:
        try:
            fluent_id = self.encoder.fluent_ids[str(fluent)]
        except KeyError:
            raise UPStateMissingFluentError(
                f"The state {self.state} does not have a value for the fluent {fluent}"
            ) from None
        v = get_fluent_value(fluent_id, self.state)
        if fluent.type.is_bool_type():
            return self.em.Bool(cast(bool, v))
        elif fluent.type.is_int_type():
            return self.em.Int(cast(int, v))
        elif fluent.type.is_real_type():
            # `v` is not necessarily a `Fraction`: both backends normalize an
            # integral real value down to a plain `int`, and `em.Real` requires a
            # `Fraction`.
            return self.em.Real(Fraction(cast("int | Fraction", v)))
        elif fluent.type.is_user_type():
            oid = cast(search_space.ObjectNode, v).object
            return self.em.ObjectExp(
                self.problem.object(self.encoder.object_names[oid])
            )
        else:
            raise NotImplementedError(f"Unknown value type for expression {fluent}")

    @classmethod
    def wrap_heuristic(
        cls,
        encoder: Encoder,
        heuristic: Callable[[State], float | None],
    ) -> Callable[[search_space.State], float | None]:
        """Wrap heuristic so it receives a StateWrapper instead of a State."""
        return lambda state: heuristic(cls(encoder, state))


@dataclass(frozen=True)
class HeuristicParams:
    heuristic: str | None = None
    weight: float | None = None


@dataclass(frozen=True)
class SearchParams(HeuristicParams):
    search: str = "wastar"
    internal_heuristic_cache: bool = True
    inadmissible_numeric_heuristic_variant: bool = False
    early_termination: bool = False
    weak_equality: bool = False
    relevant_equality: bool = True
    symmetry_breaking: bool = True
    compression_safe_actions: bool = True
    relevance_analysis: bool = True
    incomplete_memory_bounded_search: bool = False


@dataclass(frozen=True)
class MultiqueueParams:
    queues: list[HeuristicParams]
    internal_heuristic_cache: bool = True
    inadmissible_numeric_heuristic_variant: bool = False
    early_termination: bool = False
    weak_equality: bool = False
    relevant_equality: bool = True
    symmetry_breaking: bool = True
    compression_safe_actions: bool = True
    relevance_analysis: bool = True


class _SearchCallable(Protocol):
    """Call signature shared by the search functions bound (via `partial`) in
    `TamerLite._get_search`; a plain `Callable[[...], ...]` can't express the
    keyword-only `weak_equality` param, which `dfs`/`bfs` don't accept."""

    def __call__(
        self,
        ss: search_space.SearchSpaceABC,
        timeout: float | None = ...,
        early_termination: bool = ...,
        weak_equality: bool = ...,
    ) -> tuple[list[Action] | None, dict[str, str]]: ...


class _HeuristicCallable(Protocol):
    """Call signature shared by the heuristic constructors in `hh_map`
    (`_get_heuristic`): a mix of plain functions, a class, and `partial`-bound
    variants, which a plain `Callable[[...], ...]` can't unify (their join
    widens to `object`, which isn't callable)."""

    def __call__(
        self,
        actions: list[Action],
        fluent_types: list[str],
        objects: dict[str, list[int]],
        events: dict[Action, list[tuple[search_space.Timing, search_space.Event]]],
        goals: search_space.Expression,
        *,
        internal_caching: bool,
        cache_value_in_state: bool,
        inadmissible_numeric_heuristic_variant: bool,
    ) -> Heuristic: ...


class TamerLite(
    unified_planning.engines.Engine,
    unified_planning.engines.mixins.OneshotPlannerMixin,
    unified_planning.engines.mixins.AnytimePlannerMixin,
):
    def __init__(
        self,
        search: SearchParams | MultiqueueParams = SearchParams(),  # noqa: B008  # frozen (immutable) dataclass, safe as default
    ):
        unified_planning.engines.Engine.__init__(self)
        up.engines.mixins.OneshotPlannerMixin.__init__(self)
        self._params = search

    @property
    def name(self) -> str:
        return "TamerLite"

    @staticmethod
    def get_credits(**kwargs) -> up.engines.Credits | None:
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
        supported_kind.set_expression_duration("REAL_TYPE_DURATIONS")
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
        supported_kind.set_conditions_kind("INTERPRETED_FUNCTIONS_IN_CONDITIONS")
        supported_kind.set_effects_kind("INTERPRETED_FUNCTIONS_IN_BOOLEAN_ASSIGNMENTS")
        supported_kind.set_effects_kind("INTERPRETED_FUNCTIONS_IN_NUMERIC_ASSIGNMENTS")
        supported_kind.set_effects_kind("INTERPRETED_FUNCTIONS_IN_OBJECT_ASSIGNMENTS")
        supported_kind.set_expression_duration("INTERPRETED_FUNCTIONS_IN_DURATIONS")
        supported_kind.set_fluents_type("NUMERIC_FLUENTS")
        supported_kind.set_fluents_type("OBJECT_FLUENTS")
        supported_kind.set_fluents_type("INT_FLUENTS")
        supported_kind.set_fluents_type("REAL_FLUENTS")
        supported_kind.set_quality_metrics("ACTIONS_COST")
        supported_kind.set_quality_metrics("FINAL_VALUE")
        supported_kind.set_quality_metrics("MAKESPAN")
        supported_kind.set_quality_metrics("PLAN_LENGTH")
        supported_kind.set_actions_cost_kind("INT_NUMBERS_IN_ACTIONS_COST")
        supported_kind.set_actions_cost_kind("REAL_NUMBERS_IN_ACTIONS_COST")
        supported_kind.set_actions_cost_kind("STATIC_FLUENTS_IN_ACTIONS_COST")
        supported_kind.set_actions_cost_kind("FLUENTS_IN_ACTIONS_COST")
        supported_kind.set_initial_state("UNDEFINED_INITIAL_NUMERIC")
        return supported_kind

    @staticmethod
    def supports(problem_kind: "up.model.ProblemKind") -> bool:
        return bool(problem_kind <= TamerLite.supported_kind())

    @staticmethod
    def satisfies(optimality_guarantee: up.engines.OptimalityGuarantee) -> bool:
        return bool(optimality_guarantee == up.engines.OptimalityGuarantee.SATISFICING)

    @staticmethod
    def ensures(anytime_guarantee: up.engines.AnytimeGuarantee) -> bool:
        return bool(anytime_guarantee == up.engines.AnytimeGuarantee.INCREASING_QUALITY)

    def _get_heuristic(
        self,
        params: HeuristicParams,
        heuristic: Callable[[State], float | None] | None,
        encoder: Encoder,
        inadmissible_numeric_heuristic_variant: bool,
        internal_heuristic_cache: bool,
        cache_heuristic_in_state: bool = False,
    ) -> tuple[Heuristic, float]:
        assert encoder.goal is not None
        if params.heuristic is None:
            h_name = "custom" if heuristic is not None else "hff"
        else:
            h_name = params.heuristic

        if h_name == "custom":
            assert heuristic is not None
            h: Heuristic = CustomHeuristic(
                StateWrapper.wrap_heuristic(encoder, heuristic),
                cache_heuristic_in_state,
            )
            w = 1.0 if params.weight is None else params.weight

        elif h_name == "blind":
            h = CustomHeuristic(lambda x: 0.0, cache_heuristic_in_state)
            w = 0.0

        else:
            hh_map: dict[str, _HeuristicCallable] = {
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
            h = hh_map[h_name](
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
        search_name: str,
        heuristic: Heuristic,
        weight: float,
        incomplete_memory_bounded_search: bool,
        weak_equality: bool,
        is_temporal: bool,
    ) -> tuple[str, _SearchCallable]:
        if (
            search_name in {"wastar", "astar", "gbfs"}
            and incomplete_memory_bounded_search
            and is_temporal
            and weak_equality
        ):
            warnings.warn(
                "Memory-bounded search does not support weak equality correctly.",
                stacklevel=2,
            )

        if search_name == "wastar":
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
        elif search_name == "ehc":
            search = partial(ehc_search, heuristic=heuristic)
        else:
            raise NotImplementedError(
                f"Unknown search '{search_name}'. "
                "Supported values are: wastar, astar, gbfs, dfs, bfs, ehc."
            )

        return search_name, search

    def _compile_problem(
        self, problem: "up.model.Problem"
    ) -> tuple[
        "up.model.Problem",
        "up.model.Problem",
        Callable[[ActionInstance], ActionInstance | None],
    ]:
        kind = problem.kind
        undefined_map_back_action_instance: Callable[
            [ActionInstance], ActionInstance | None
        ]
        if kind.has_undefined_initial_numeric():
            undefined_initial_numeric_remover = UndefinedInitialNumericRemover()
            compilation_res = undefined_initial_numeric_remover.compile(problem)
            assert compilation_res.map_back_action_instance is not None
            undefined_map_back_action_instance = (
                compilation_res.map_back_action_instance
            )
            problem = cast("up.model.Problem", compilation_res.problem)
        else:

            def undefined_map_back_action_instance(
                ai: ActionInstance,
            ) -> ActionInstance | None:
                return ai

        grounder = Grounder()
        compilation_res = grounder.compile(problem)
        assert compilation_res.map_back_action_instance is not None
        ground_map_back_action_instance = compilation_res.map_back_action_instance
        ground_problem = cast("up.model.Problem", compilation_res.problem)
        lifted_problem = problem

        def map_back_action_instance(
            ai: ActionInstance,
        ) -> ActionInstance | None:
            lifted_ai = ground_map_back_action_instance(ai)
            if lifted_ai is None:
                return None
            return undefined_map_back_action_instance(lifted_ai)

        return lifted_problem, ground_problem, map_back_action_instance

    def _get_solutions_with_params(
        self,
        problem: "up.model.AbstractProblem",
        timeout: float | None = None,
        output_stream: IO[str] | None = None,
        warm_start_plan: up.plans.Plan | None = None,
        **kwargs,
    ) -> Iterator["up.engines.results.PlanGenerationResult"]:
        assert isinstance(problem, up.model.Problem)
        if len(problem.quality_metrics) > 1:
            raise NotImplementedError("Multiple quality metrics are not supported")

        # Brackets this whole run -- including every suspension between
        # `yield`s below, not just the synchronous work in between -- so
        # `clear_interpreted_function_cache` never runs while this generator
        # could still resume and evaluate a `func_id` it registered. See
        # `interpreted_function_scope`'s docstring (converter.py).
        with interpreted_function_scope():
            yield from self._anytime_solutions(
                problem, timeout=timeout, output_stream=output_stream
            )

    def _anytime_solutions(
        self,
        problem: "up.model.Problem",
        timeout: float | None = None,
        output_stream: IO[str] | None = None,
    ) -> Iterator["up.engines.results.PlanGenerationResult"]:
        start_time = time.monotonic()
        em = problem.environment.expression_manager
        tm = problem.environment.type_manager

        lifted_problem, ground_problem, map_back_action_instance = (
            self._compile_problem(problem)
        )
        original_problem = problem

        # Share one interpreted functions cache across every `_solve_ground_problem`
        # call in this anytime run (see `Converter._if_cache`) instead of recomputing
        # from scratch each time.
        if_cache: MutableMapping[tuple[InterpretedFunction, tuple], Any] = (
            new_if_cache()
        )
        # Likewise for the wrapper closures themselves (`Converter._if_wrappers`) --
        # safe because object numbering never changes across this run's re-encodes.
        # Sharing them is what lets the Rust backend's own IF_RESULTS cache persist
        # across re-encodes too, since its `func_id`s are keyed on wrapper identity.
        if_wrappers: dict[InterpretedFunction, Callable] = {}

        # `lifted_problem` is the same object across every `_solve_ground_problem`
        # call this run makes (the loop below only ever mutates/clones the
        # *ground* problem), so compute this once here rather than paying for
        # `Problem.kind`'s from-scratch full-problem scan on every iteration --
        # and skip it entirely when symmetry breaking (its only consumer) is off.
        lifted_problem_kind = (
            lifted_problem.kind if self._params.symmetry_breaking else None
        )

        logger.info(
            "Solving '%s' (anytime): actions=%d fluents=%d",
            ground_problem.name,
            len(list(ground_problem.actions)),
            len(list(ground_problem.fluents)),
        )
        elapsed_time = time.monotonic() - start_time
        res, _, _ = self._solve_ground_problem(
            lifted_problem,
            ground_problem,
            map_back_action_instance,
            timeout=timeout - elapsed_time if timeout is not None else None,
            output_stream=output_stream,
            lifted_problem_kind=lifted_problem_kind,
            is_intermediate_solution=True,
            if_cache=if_cache,
            if_wrappers=if_wrappers,
        )
        if res.plan is not None:
            logger.info(
                "Initial solution found in %.3fs: %s",
                time.monotonic() - start_time,
                res.metrics,
            )
        else:
            logger.info(
                "No initial solution found in %.3fs", time.monotonic() - start_time
            )
        yield res
        if res.plan is None:
            return

        quality_metric: up.model.metrics.PlanQualityMetric | None
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

        validator: SequentialPlanValidator | TimeTriggeredPlanValidator
        if res.plan.kind == PlanKind.SEQUENTIAL_PLAN:
            validator = SequentialPlanValidator()
        elif res.plan.kind == PlanKind.TIME_TRIGGERED_PLAN:
            validator = TimeTriggeredPlanValidator()
        else:
            raise AssertionError(f"Unknown plan type {res.plan.kind}")

        def validate_plan(
            problem: "up.model.Problem", plan: up.plans.Plan
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
        ground_problem_actions = {a.name: a for a in ground_problem.actions}
        while res.status == up.engines.PlanGenerationResultStatus.INTERMEDIATE:
            val_res = validate_plan(original_problem, res.plan)
            assert val_res
            assert (
                val_res.metric_evaluations is not None
                and len(val_res.metric_evaluations) == 1
            ), "Expected metric evaluations for plan validation result"

            problem = cast("up.model.Problem", ground_problem.clone())
            exp = None
            deadline = None
            m, v = next(iter(val_res.metric_evaluations.items()))
            logger.info("Searching for improvement over current quality: %s", v)
            if m.is_minimize_expression_on_final_state():
                exp = em.LT(
                    cast(up.model.metrics.MinimizeExpressionOnFinalState, m).expression,
                    v,
                )
            elif m.is_maximize_expression_on_final_state():
                exp = em.GT(
                    cast(up.model.metrics.MaximizeExpressionOnFinalState, m).expression,
                    v,
                )
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
                            "Only instantaneous actions supported for plan "
                            "length metric"
                        )
                exp = em.LT(plan_length, v)
            elif m.is_minimize_action_costs():
                m = cast(
                    up.model.metrics.MinimizeActionCosts,
                    next(iter(problem.quality_metrics)),
                )
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

            def new_map_back_action_instance(
                ai: ActionInstance,
            ) -> ActionInstance | None:
                action = ground_problem_actions.get(ai.action.name)
                if action is not None:
                    return map_back_action_instance(action())
                return None

            elapsed_time = time.monotonic() - start_time
            res, solution_might_exist, is_any_action_compression_safe = (
                self._solve_ground_problem(
                    lifted_problem,
                    problem,
                    new_map_back_action_instance,
                    timeout=timeout - elapsed_time if timeout is not None else None,
                    output_stream=output_stream,
                    deadline=deadline,
                    is_intermediate_solution=True,
                    if_cache=if_cache,
                    if_wrappers=if_wrappers,
                    lifted_problem_kind=lifted_problem_kind,
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
                logger.info(
                    "No further improvement found, terminating with status: %s",
                    prev_res.status,
                )
                yield prev_res
            elif res.status == up.engines.PlanGenerationResultStatus.TIMEOUT:
                prev_res.status = up.engines.PlanGenerationResultStatus.TIMEOUT
                logger.info("Search timed out during improvement")
                yield prev_res
            else:
                assert res.plan is not None
                logger.info("Improved solution found: %s", res.metrics)
                yield res

            prev_res = up.engines.PlanGenerationResult(
                res.status, res.plan, res.engine_name, res.metrics, res.log_messages
            )

    def _get_solutions(
        self,
        problem: "up.model.AbstractProblem",
        timeout: float | None = None,
        output_stream: IO[str] | None = None,
    ) -> Iterator[up.engines.results.PlanGenerationResult]:
        return self._get_solutions_with_params(problem, timeout, output_stream)

    def _solve(
        self,
        problem: "up.model.AbstractProblem",
        heuristic: Callable[[State], float | None] | None = None,
        timeout: float | None = None,
        output_stream: IO[str] | None = None,
    ) -> "up.engines.results.PlanGenerationResult":
        assert isinstance(problem, up.model.Problem)
        start_time = time.monotonic()
        lifted_problem, ground_problem, map_back_action_instance = (
            self._compile_problem(problem)
        )
        elapsed_time = time.monotonic() - start_time
        logger.info(
            "Solving '%s': actions=%d fluents=%d",
            ground_problem.name,
            len(list(ground_problem.actions)),
            len(list(ground_problem.fluents)),
        )
        # See `interpreted_function_scope`'s docstring (converter.py): brackets
        # this standalone solve so `clear_interpreted_function_cache` never runs
        # while it -- or a concurrently suspended anytime generator -- could
        # still resume and evaluate a `func_id` it registered.
        with interpreted_function_scope():
            res, _, _ = self._solve_ground_problem(
                lifted_problem,
                ground_problem,
                map_back_action_instance,
                heuristic=heuristic,
                timeout=timeout - elapsed_time if timeout is not None else None,
                output_stream=output_stream,
                is_intermediate_solution=False,
            )
        if res.plan is not None:
            logger.info(
                "Solution found in %.3fs: %s",
                time.monotonic() - start_time,
                res.metrics,
            )
        else:
            logger.info(
                "No solution found in %.3fs: %s",
                time.monotonic() - start_time,
                res.metrics,
            )
        return res

    def _solve_ground_problem(
        self,
        problem: "up.model.Problem",
        ground_problem: "up.model.Problem",
        map_back_action_instance: Callable[[ActionInstance], ActionInstance | None],
        heuristic: Callable[[State], float | None] | None = None,
        timeout: float | None = None,
        output_stream: IO[str] | None = None,
        deadline: Fraction | None = None,
        is_intermediate_solution: bool = False,
        if_cache: MutableMapping[tuple[InterpretedFunction, tuple], Any] | None = None,
        if_wrappers: dict[InterpretedFunction, Callable] | None = None,
        lifted_problem_kind: ProblemKind | None = None,
    ) -> tuple["up.engines.results.PlanGenerationResult", bool, bool]:
        # Default to one fresh pair shared by both `Encoder(...)` sites below.
        # This path only ever runs for a standalone oneshot `_solve()` call --
        # `_anytime_solutions` always supplies its own `if_cache`/`if_wrappers`
        # -- so it's the caller's job to have opened an `interpreted_function_scope`
        # around this call (as `_solve` does); a caller that reaches this
        # branch outside any scope gets no reclamation at all, ever.
        if if_cache is None:
            if_cache = new_if_cache()
        if if_wrappers is None:
            if_wrappers = {}
        # `problem` (the lifted problem) never changes across this call's two
        # possible `Encoder(...)` sites below, nor across `_anytime_solutions`'
        # repeated calls for the same solve -- `Problem.kind` is a from-scratch
        # full-problem scan, so compute this once and let callers that already
        # know it (`_anytime_solutions`), or that don't need it at all
        # (symmetry breaking off, `Encoder`'s only consumer), skip it entirely.
        if lifted_problem_kind is None and self._params.symmetry_breaking:
            lifted_problem_kind = problem.kind
        try:
            # Compression-safe-action detection and the TimedToSequential
            # recompile below are unaffected by interpreted functions: they
            # are carried through both as opaque sub-expressions (see UP's
            # TimedToSequential docstring) and TamerLite's own
            # `_compute_compression_safe_actions` only inspects
            # fluents/objects via generic expression-argument traversal.
            # Relevance analysis (`Encoder._compute_relevant_actions`, which
            # runs `HMax` reachability) is interpreted-function-safe too.
            encoder = Encoder(
                ground_problem,
                problem,
                map_back_action_instance,
                self._params.symmetry_breaking,
                self._params.compression_safe_actions,
                self._params.relevance_analysis,
                self._params.relevant_equality,
                self._params.weak_equality,
                deadline=deadline,
                if_cache=if_cache,
                if_wrappers=if_wrappers,
                lifted_problem_kind=lifted_problem_kind,
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
                # Compile a temporal planning problem, where all actions are
                # safe to compress, into an equivalent classical planning problem.
                # `TimedToSequential.supported_kind()` doesn't declare
                # MAKESPAN, but a fully compression-safe problem can still
                # carry a minimize-makespan metric, which would make
                # the kind check reject it.
                t2s_compiler = TimedToSequential()
                t2s_compiler.skip_checks = True
                compilation_res = t2s_compiler.compile(ground_problem)
                ground_problem_actions = {a.name: a for a in ground_problem.actions}

                def new_map_back_action_instance(
                    ai: ActionInstance,
                ) -> ActionInstance | None:
                    action = ground_problem_actions.get(ai.action.name)
                    if action is not None:
                        return map_back_action_instance(action())
                    return None

                encoder = Encoder(
                    cast("up.model.Problem", compilation_res.problem),
                    problem,
                    new_map_back_action_instance,
                    self._params.symmetry_breaking,
                    self._params.compression_safe_actions,
                    self._params.relevance_analysis,
                    self._params.relevant_equality,
                    self._params.weak_equality,
                    deadline=deadline,
                    if_cache=if_cache,
                    if_wrappers=if_wrappers,
                    lifted_problem_kind=lifted_problem_kind,
                )

            if self._params.weak_equality and not encoder.search_space.is_temporal:
                warnings.warn(
                    "weak_equality has no effect on non-temporal problems.",
                    stacklevel=2,
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

                start = time.monotonic()
                path, metrics = multiqueue_search(
                    encoder.search_space,
                    heuristics,
                    timeout,
                    early_termination=self._params.early_termination,
                    weak_equality=self._params.weak_equality,
                )
                if (
                    self._params.weak_equality
                    and encoder.search_space.is_temporal
                    and path is None
                ):
                    updated_timeout = _remaining_timeout(timeout, start)
                    logger.info(
                        "Weak-equality multiqueue search found no plan; retrying "
                        "with weak_equality=False and timeout=%s",
                        updated_timeout,
                    )
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
                    start = time.monotonic()
                    path, metrics = search(
                        encoder.search_space,
                        timeout=timeout,
                        early_termination=self._params.early_termination,
                        weak_equality=True,
                    )
                    if encoder.search_space.is_temporal and path is None:
                        updated_timeout = _remaining_timeout(timeout, start)
                        logger.info(
                            "Weak-equality search found no plan; retrying "
                            "with weak_equality=False and timeout=%s",
                            updated_timeout,
                        )
                        path, metrics = search(
                            encoder.search_space,
                            timeout=updated_timeout,
                            early_termination=self._params.early_termination,
                            weak_equality=False,
                        )
                else:
                    path, metrics = search(
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
                solution_might_exist = search_name == "ehc"
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


def _remaining_timeout(timeout: float | None, start_time: float) -> float | None:
    """Return the non-negative timeout left after a completed search phase."""
    if timeout is None:
        return None
    return max(0.0, timeout - (time.monotonic() - start_time))
