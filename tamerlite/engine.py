from dataclasses import dataclass
from functools import partial
import unified_planning as up
import unified_planning.model
import unified_planning.engines
import unified_planning.engines.mixins
from unified_planning.model import ProblemKind, FNode
from unified_planning.model.state import State
from unified_planning.engines.compilers.grounder import GrounderHelper
from typing import IO, Any, Callable, List, Optional, Union
from argparse import Namespace

from tamerlite.core import wastar_search, astar_search, gbfs_search
from tamerlite.core import bfs_search, dfs_search, ehc_search
from tamerlite.core import multiqueue_search
from tamerlite.core import evaluate, make_fluent_node
from tamerlite.core import HFF, HAdd, CustomHeuristic, RLRank, RLHeuristic
from tamerlite.converter import Converter
from tamerlite.encoder import Encoder, get_encoders
from tamerlite.utils import read_macros

import sys
import os
import csv


# def add_row_to_csv(filename, name, number1, number2):

#     with open(filename, mode='a', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow([name, number1, number2])


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
    max_plan_size: Optional[int] = None
    other_params: Optional[Namespace] = None


@dataclass(frozen=True)
class SearchParams:
    search: Optional[str] = None
    heuristic: Optional[str] = None
    weight: Optional[str] = None
    rl_params: Optional[RLParams] = None
    macros: Optional[Union[str, List[str]]] = None
    macros_usage: Optional[str] = None
    plan_length: Optional[str] = None
    max_macros: Optional[str] = None

    def contains_rl(self) -> bool:
        return self.rl_params is not None
    
    def contains_macros(self) -> bool:
        return self.macros is not None
    
    def contains_max_macros(self) -> bool:
        return self.max_macros is not None

    def domain(self):
        return self.rl_params.domain



@dataclass(frozen=True)
class MultiqueueParams:
    queues: List[SearchParams]

    @property
    def macros(self):
        return  self.queues[0].macros
    
    @property
    def macros_usage(self):
        return self.queues[0].macros_usage
    
    @property
    def plan_length(self):
        return self.queues[0].plan_length
    
    @property
    def max_macros(self):
        return self.queues[0].max_macros

    def contains_rl(self) -> bool:
        return any([q.contains_rl() for q in self.queues])
    
    def contains_macros(self) -> bool:
        return any([q.contains_macros() for q in self.queues])

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

    def __init__(self, search: Optional[Union[SearchParams, MultiqueueParams]] = None, heuristic: Optional[str] = None, weight: Optional[str] = None, 
                 macros: Optional[str] = None, macros_usage: Optional[str] = None, max_macros: Optional[str] = None, plan_length: Optional[str] = None):
        unified_planning.engines.Engine.__init__(self)
        up.engines.mixins.OneshotPlannerMixin.__init__(self)
        self._params = search
        if self._params is None and (heuristic is not None or weight is not None or macros is not None or macros_usage is not None or plan_length is not None and max_macros is not None):
            self._params = SearchParams(search=None, heuristic=heuristic, weight=weight, macros=macros, macros_usage=macros_usage, plan_length=plan_length, max_macros=max_macros)  

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

    def _get_heuristic(self, params, heuristic, encoder, state_encoder):
        if params is None:
            h = "custom" if heuristic else "hff"
        else:
            h = "custom" if heuristic and params.heuristic is None else params.heuristic if params.heuristic else "hff"
            rl_params = params.rl_params

        if h == "custom":
            def rewrite_h(search_state):
                return heuristic(StateWrapper(encoder.problem, search_state))
            h = CustomHeuristic(rewrite_h)
            w = 1 if params is None or params.weight is None else params.weight
        elif h == "rl_heuristic":
            assert rl_params is not None and rl_params.max_plan_size is not None and rl_params.other_params is not None
            h = RLHeuristic(state_encoder, rl_params.model, rl_params.model_class, rl_params.max_plan_size, rl_params.other_params)
            w = 0.8 if params is None or params.weight is None else params.weight
        elif h == "rl_rank":
            assert rl_params is not None
            h = RLRank(state_encoder, rl_params.model, rl_params.model_class)
            w = 1 if params is None or params.weight is None else params.weight
        elif h == "hff":
            h = HFF(encoder.fluents, encoder.objects, encoder.events, encoder.goal)
            w = 0.8 if params is None or params.weight is None else params.weight
        elif h == "hadd":
            h = HAdd(encoder.fluents, encoder.objects, encoder.events, encoder.goal)
            w = 0.8 if params is None or params.weight is None else params.weight
        elif h == "blind":
            h = CustomHeuristic(lambda x: 0.0)
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
                if self._params.contains_macros():
                    extracted_macros = read_macros(self._params.macros, self._params.macros_usage, problem, self._params.plan_length, self._params.max_macros)
                    extracted_macros.sort()
                    self._params = SearchParams(
                        search=self._params.search,
                        heuristic=self._params.heuristic,
                        weight=self._params.weight,
                        rl_params=self._params.rl_params,
                        macros=extracted_macros,
                        macros_usage=self._params.macros_usage,
                        plan_length=self._params.plan_length,
                        max_macros=self._params.max_macros
                    )
                    encoder, state_encoder, map_back_action_instance = get_encoders(self._params.domain(), problem, self._params.macros, self._params.macros_usage)
                else:
                    encoder, state_encoder, map_back_action_instance = get_encoders(self._params.domain(), problem)
            else:
                with problem.environment.factory.Compiler(compilation_kind="GROUNDING", problem_kind=problem.kind) as compiler:
                    compilation_res = compiler.compile(problem)
                    map_back_action_instance = compilation_res.map_back_action_instance
                new_problem = compilation_res.problem
                print(f"Number of actions: {len([a for a in new_problem.actions])}")
                sys.stdout.flush()

                if self._params is not None and self._params.contains_macros():
                    grounder_helper = GrounderHelper(
                        problem, compiler._grounding_actions_map, compiler._prune_actions
                    )   
                    #extracted_macros = read_macros(self._params.macros, self._params.macros_usage, problem, self._params.max_macros)
                    extracted_macros = read_macros(self._params.macros, self._params.macros_usage, problem, self._params.plan_length, self._params.max_macros, grounder_helper)
                    extracted_macros.sort()
                    self._params = SearchParams(
                        search=self._params.search,
                        heuristic=self._params.heuristic,
                        weight=self._params.weight,
                        rl_params=self._params.rl_params,
                        macros=extracted_macros,
                        macros_usage=self._params.macros_usage,
                        plan_length=self._params.plan_length,
                        max_macros=self._params.max_macros
                    )

                    # actions = set()  # if use ground_macros
                    # for a in new_problem.actions:
                    #     actions.add(str(a.name))
                    # new_macros = []
                    # for ma in self._params.macros:
                    #     cont = 0
                    #     for a in ma:
                    #         if str(a) in actions:
                    #             cont += 1
                    #     if cont == len(ma):
                    #         new_macros.append(ma)
                    # macros = new_macros
                    # perc = round(len(macros)*100/len(self._params.macros),1)
                    
                    macros = self._params.macros  #if use lifted_macros
                    perc = 100

                    print(f"Useful macros: {len(macros)}\nPercentage of useful macros: {perc}")
                    sys.stdout.flush()

                    #add_row_to_csv('experiments/test-survival-macros-kitting/survived_macros_analysis_kitting_2.csv', problem.name, len(macros), perc)

                    encoder = Encoder(new_problem, macros = macros, macros_usage=self._params.macros_usage)
                else:
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
                plan, metrics, macro_used = search(encoder.search_space, timeout=timeout)
                ma_sel = {}
                for ma in macro_used:
                    #print(f"{ma}:  {self._params.macros.index(ma)}")
                    ma_sel[tuple(ma)] = ma_sel.get(tuple(ma), 0) + 1
                to_print = "|".join("{}:{}".format(key, val) for key, val in ma_sel.items())
                print(f"Macros used: {to_print}")
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
