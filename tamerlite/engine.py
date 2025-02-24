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
    enable_heuristic_cache: Optional[bool] = None
    weight: Optional[str] = None
    rl_params: Optional[RLParams] = None
    cache_heuristic: Optional[bool] = None

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


def get_applicable_actions(domain, initial_values, objects, map_back_ai, big_encoder):
    all_objects = domain.all_objects
    map_back_action_instance = map_back_ai
    problem = domain.clone()

    for f, v in initial_values.items():
        problem.set_initial_value(f, v)

    em = problem.environment.expression_manager
    for ut in problem.user_types:
        fname = f"_is_active_{ut.name}"
        f = big_encoder.problem.fluent(fname)
        problem.add_fluent(f, default_initial_value=False)
        for obj in problem.all_objects:
            if obj.type != ut:
                continue
            if obj in objects:
                problem.set_initial_value(f(obj), em.TRUE())
                initial_values[f(obj)] = em.TRUE()
            else:
                initial_values[f(obj)] = em.FALSE()

    inactive_objects = [obj for obj in all_objects if obj not in objects]

    # Compute the potentially applicable actions
    actions = []
    simplifier = Simplifier(problem.environment, problem)
    for action in big_encoder.problem.actions:
        ai = map_back_action_instance(action())
        applicable = True
        for obj in ai.actual_parameters:
            if obj.is_object_exp() and obj.object() in inactive_objects:
                applicable = False
                break
        if applicable:
            if isinstance(action, up.model.InstantaneousAction):
                conditions = action.preconditions
            else:
                conditions = action.conditions.values()
            for lc in conditions:
                nc = simplifier.simplify(em.And(lc))
                if nc.is_false():
                    applicable = False
                    break
        if applicable:
            actions.append(action.name)

    return actions

def simplify_static_fluents(action, events, subs):
    new_events = []
    for i, (t, event) in enumerate(events):
        new_c = simplify(event.conditions, subs)
        new_e = []
        for eff in event.effects:
            new_e.append(Effect(eff.fluent, simplify(eff.value, subs)))
        new_event = Event(action, i, new_c, tuple(), tuple(), tuple(new_e))
        new_events.append((t, new_event))
    return new_events


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
        return supported_kind

    @staticmethod
    def supports(problem_kind: 'up.model.ProblemKind') -> bool:
        return problem_kind <= TamerLite.supported_kind()

    @staticmethod
    def satisfies(optimality_guarantee: up.engines.OptimalityGuarantee) -> bool:
        return optimality_guarantee == up.engines.OptimalityGuarantee.SATISFICING

    def _get_heuristic(self, params, heuristic, encoder, state_encoder, domain, problem):
        if params is not None and params.contains_rl():
            big_encoder, _, map_back_ai = get_encoders(domain)
            initial_values = problem.initial_values
            full_initial_values = domain.initial_values
            full_initial_values.update(initial_values)
            objects = problem.all_objects
            my_goal = big_encoder.goals(problem.goals)
            my_actions = get_applicable_actions(domain, full_initial_values, objects, map_back_ai, big_encoder)
            my_initial_state = big_encoder.initial_state(full_initial_values)
            my_static_fluents = []
            for sf in big_encoder.problem.get_static_fluents():
                my_static_fluents.extend([str(f) for f in get_all_fluent_exp(big_encoder.problem, sf)])
            subs = {k: v for k, v in my_initial_state.items() if k in my_static_fluents}
            my_events = {a: simplify_static_fluents(a, e, subs) for a, e in big_encoder.events.items() if a in my_actions}
        if params is None:
            h = "custom" if heuristic else "hff"
        else:
            h = "custom" if heuristic and params.heuristic is None else params.heuristic if params.heuristic else "hff"
            rl_params = params.rl_params
        if params is None:
            cache_h = False
        else:
            cache_h = False if params.cache_heuristic is None else params.cache_heuristic

        if h == "custom":
            def rewrite_h(search_state):
                return heuristic(StateWrapper(encoder.problem, search_state))
            h = CustomHeuristic(rewrite_h, cache_h)
            w = 1 if params is None or params.weight is None else params.weight
        elif h == "rl_heuristic":
            assert rl_params is not None and rl_params.other_params is not None
            if rl_params.other_params.learning_heuristic == "hadd":
                heuristic_for_residual = HAdd(big_encoder.fluents, big_encoder.objects, my_events, my_goal, cache_enabled=rl_params.other_params.cache_heuristic)
            elif rl_params.other_params.learning_heuristic == "hff":
                heuristic_for_residual = HFF(big_encoder.fluents, big_encoder.objects, my_events, my_goal, cache_enabled=rl_params.other_params.cache_heuristic)
            h = RLHeuristic(state_encoder, rl_params.model, rl_params.model_class, rl_params.other_params, heuristic_for_residual, cache_h)
            w = 0.8 if params is None or params.weight is None else params.weight
        elif h == "rl_rank":
            assert rl_params is not None and rl_params.other_params is not None
            if rl_params.other_params.learning_heuristic == "hadd":
                heuristic_for_residual = HAdd(big_encoder.fluents, big_encoder.objects, my_events, my_goal, cache_enabled=rl_params.other_params.cache_heuristic)
            elif rl_params.other_params.learning_heuristic == "hff":
                heuristic_for_residual = HFF(big_encoder.fluents, big_encoder.objects, my_events, my_goal, cache_enabled=rl_params.other_params.cache_heuristic)
            h = RLRank(state_encoder, rl_params.model, rl_params.model_class, rl_params.other_params, heuristic_for_residual, cache_h)
            w = 1 if params is None or params.weight is None else params.weight
        elif h == "hff":
            enable_heuristic_cache = True if params is None or params.enable_heuristic_cache is None else params.enable_heuristic_cache
            h = HFF(encoder.fluents, encoder.objects, encoder.events, encoder.goal, cache_states=enable_heuristic_cache, cache_enabled=cache_h)
            w = 0.8 if params is None or params.weight is None else params.weight
        elif h == "hadd":
            enable_heuristic_cache = True if params is None or params.enable_heuristic_cache is None else params.enable_heuristic_cache
            h = HAdd(encoder.fluents, encoder.objects, encoder.events, encoder.goal, cache_states=enable_heuristic_cache, cache_enabled=cache_h)
            w = 0.8 if params is None or params.weight is None else params.weight
        elif h == "hmax":
            enable_heuristic_cache = True if params is None or params.enable_heuristic_cache is None else params.enable_heuristic_cache
            h = HMax(encoder.fluents, encoder.objects, encoder.events, encoder.goal, cache_states=enable_heuristic_cache, cache_enabled=cache_h)
            w = 0.8 if params is None or params.weight is None else params.weight
        elif h == "hmax_numeric":
            enable_heuristic_cache = True if params is None or params.enable_heuristic_cache is None else params.enable_heuristic_cache
            h = HMaxNumeric(encoder.fluents, encoder.objects, encoder.events, encoder.goal, cache_states=enable_heuristic_cache, cache_enabled=cache_h)
            w = 0.8 if params is None or params.weight is None else params.weight
        elif h == "blind":
            h = CustomHeuristic(lambda x: 0.0, cache_h)
            w = 0
        else:
            raise NotImplementedError

        return h, w

    def _get_search(self, params, heuristic, encoder, state_encoder, domain, problem):
        if params is None:
            s = "wastar"
        else:
            s = "wastar" if params.search is None else params.search

        h, w = self._get_heuristic(params, heuristic, encoder, state_encoder, domain, problem)

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
                    h, w = self._get_heuristic(p, heuristic, encoder, state_encoder, self._params.domain(), problem)
                    heuristics.append((h, w))
                plan = multiqueue_search(encoder.search_space, heuristics, timeout)
            else:
                search = self._get_search(self._params, heuristic, encoder, state_encoder, self._params.domain() if self._params is not None else None, problem)
                plan = search(encoder.search_space, timeout=timeout)

            if plan:
                plan = encoder.build_plan(plan)
                plan = plan.replace_action_instances(map_back_action_instance)
                status = up.engines.PlanGenerationResultStatus.SOLVED_SATISFICING
            else:
                status = up.engines.PlanGenerationResultStatus.UNSOLVABLE_INCOMPLETELY
            return up.engines.PlanGenerationResult(status, plan, self.name)
        except TimeoutError:
            status = up.engines.PlanGenerationResultStatus.TIMEOUT
            return up.engines.PlanGenerationResult(status, None, self.name)
