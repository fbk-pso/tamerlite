# Type stubs for the `rustamer` compiled extension (PEP 561).
#
# `rustamer` is a PyO3 wheel with no inline annotations a type checker can read,
# so the public surface is described here. It mirrors the pure-Python fallback in
# `tamerlite.core` (the two backends expose an identical interface) — keep this in
# sync with `crates/rustamer/src/{lib,search,heuristic}.rs`,
# `crates/rustamer-base/src/`, and `src/tamerlite/core/`.

from collections.abc import Callable
from fractions import Fraction
from typing import Optional, Union

# An expression is a flat, post-order list of nodes (see `make_*_node` builders).
Expression = list["ExpressionNode"]

# (plan, statistics); plan is None when no solution was found.
SearchResult = tuple[Optional[list["Action"]], dict[str, str]]

class ExpressionNode:
    """A single node of an expression. Exactly one accessor is non-None."""

    @property
    def fluent(self) -> Optional[int]: ...
    @property
    def object(self) -> Optional[str]: ...
    @property
    def bool_constant(self) -> Optional[bool]: ...
    @property
    def int_constant(self) -> Optional[int]: ...
    @property
    def real_constant(self) -> Optional[Fraction]: ...
    def __repr__(self) -> str: ...

class Effect:
    def __init__(self, fluent: int, value: Expression) -> None: ...
    @property
    def fluent(self) -> int: ...
    @property
    def value(self) -> Expression: ...
    def __repr__(self) -> str: ...

class Timing:
    def __init__(self, start: bool, delay: Fraction) -> None: ...
    @property
    def delay(self) -> Fraction: ...
    def is_from_start(self) -> bool: ...
    def is_from_end(self) -> bool: ...
    def __repr__(self) -> str: ...

class Action:
    def __init__(self, idx: int) -> None: ...
    @property
    def idx(self) -> int: ...

class Event:
    def __init__(
        self,
        action: Action,
        pos: int,
        conditions: Expression,
        start_conditions: tuple[Expression, ...],
        end_conditions: tuple[Expression, ...],
        effects: tuple[Effect, ...],
    ) -> None: ...
    @property
    def action(self) -> Action: ...
    @property
    def pos(self) -> int: ...
    @property
    def conditions(self) -> Expression: ...
    @property
    def start_conditions(self) -> list[Expression]: ...
    @property
    def end_conditions(self) -> list[Expression]: ...
    @property
    def effects(self) -> list[Effect]: ...
    def __repr__(self) -> str: ...

class State:
    @property
    def g(self) -> float: ...
    @property
    def todo(self) -> dict[Action, tuple[int, int]]: ...
    @property
    def path(self) -> list[tuple[Action, int, int]]: ...
    def get_value(self, fluent: int) -> ExpressionNode: ...

class SearchSpace:
    def __init__(
        self,
        actions_duration: list[Optional[tuple[Expression, Expression, bool, bool]]],
        events: dict[Action, list[tuple[Timing, Event]]],
        actions: list[Action],
        compression_safe_actions: Optional[list[bool]],
        action_objects: Optional[list[list[str]]],
        obj_to_prev_actions_map: Optional[dict[str, set[Action]]],
        initial_state: Optional[Expression] = ...,
        goal: Optional[Expression] = ...,
        relevant_actions: Optional[list[Action]] = ...,
        deadline: Optional[Fraction] = ...,
        epsilon: Optional[Fraction] = ...,
    ) -> None: ...
    @property
    def is_temporal(self) -> bool: ...
    relevant_actions: list[Action]
    def reset(self) -> None: ...
    def initial_state(self, initial_state: Optional[Expression] = ...) -> State: ...
    def get_successor_state(self, state: State, action: Action) -> Optional[State]: ...
    def get_successor_states(self, state: State) -> list[State]: ...
    def goal_reached(self, state: State, goal: Optional[Expression] = ...) -> bool: ...
    def subgoals_sat(
        self, state: State, goal: Optional[Expression] = ...
    ) -> list[Expression]: ...
    # Each scheduled action is (start, action, duration); start/duration are
    # rational numbers serialized as "numer/denom" strings (not Fraction).
    def build_plan(
        self, path: list[Action]
    ) -> Optional[list[tuple[Optional[str], Action, Optional[str]]]]: ...

class Heuristic:
    @staticmethod
    def custom(
        callable: Callable[[State], Optional[float]],
        cache_value_in_state: bool,
    ) -> Heuristic: ...
    @staticmethod
    def hff(
        actions: list[Action],
        fluent_types: list[str],
        objects: dict[str, list[str]],
        events: dict[Action, list[tuple[Timing, Event]]],
        goals: list[ExpressionNode],
        internal_caching: bool,
        cache_value_in_state: bool,
        inadmissible_numeric_heuristic_variant: bool,
        disable_numeric_reasoning: bool = ...,
    ) -> Heuristic: ...
    @staticmethod
    def hadd(
        actions: list[Action],
        fluent_types: list[str],
        objects: dict[str, list[str]],
        events: dict[Action, list[tuple[Timing, Event]]],
        goals: list[ExpressionNode],
        internal_caching: bool,
        cache_value_in_state: bool,
        inadmissible_numeric_heuristic_variant: bool,
        disable_numeric_reasoning: bool = ...,
    ) -> Heuristic: ...
    @staticmethod
    def hmax(
        actions: list[Action],
        fluent_types: list[str],
        objects: dict[str, list[str]],
        events: dict[Action, list[tuple[Timing, Event]]],
        goals: list[ExpressionNode],
        internal_caching: bool,
        cache_value_in_state: bool,
        inadmissible_numeric_heuristic_variant: bool,
        disable_numeric_reasoning: bool = ...,
    ) -> Heuristic: ...
    @staticmethod
    def hmax_explicit(
        actions: list[Action],
        fluent_types: list[str],
        objects: dict[str, list[str]],
        events: dict[Action, list[tuple[Timing, Event]]],
        goals: list[ExpressionNode],
        internal_caching: bool,
        cache_value_in_state: bool,
        inadmissible_numeric_heuristic_variant: bool,
    ) -> Heuristic: ...
    @property
    def name(self) -> str: ...
    def eval(self, state: State, ss: SearchSpace) -> Optional[float]: ...
    def reachable_actions(self, state: State) -> set[Action]: ...

# --- expression builders -------------------------------------------------

def make_operator_node(kind: str, operands: tuple[int, ...]) -> ExpressionNode: ...
def make_bool_constant_node(v: bool) -> ExpressionNode: ...
def make_int_constant_node(v: int) -> ExpressionNode: ...
def make_rational_constant_node(numerator: int, denominator: int) -> ExpressionNode: ...
def make_object_node(name: str) -> ExpressionNode: ...
def make_fluent_node(fluent: int) -> ExpressionNode: ...
def shift_expression(exp: Expression, offset: int) -> Expression: ...
def get_fluents(exp: Expression) -> list[int]: ...
def evaluate(exp: Expression, state: State) -> ExpressionNode: ...
def simplify(exp: Expression, assignments: dict[int, ExpressionNode]) -> Expression: ...

# --- search algorithms ---------------------------------------------------

def bfs_search(
    ss: SearchSpace,
    timeout: Optional[float] = ...,
    early_termination: bool = ...,
) -> SearchResult: ...
def dfs_search(
    ss: SearchSpace,
    timeout: Optional[float] = ...,
    early_termination: bool = ...,
) -> SearchResult: ...
def ehc_search(
    ss: SearchSpace,
    heuristic: Heuristic,
    timeout: Optional[float] = ...,
    early_termination: bool = ...,
    weak_equality: bool = ...,
) -> SearchResult: ...
def wastar_search(
    ss: SearchSpace,
    heuristic: Heuristic,
    weight: float,
    timeout: Optional[float] = ...,
    early_termination: bool = ...,
    weak_equality: bool = ...,
) -> SearchResult: ...
def wastar_search_memory_bounded(
    ss: SearchSpace,
    heuristic: Heuristic,
    weight: float,
    timeout: Optional[float] = ...,
    early_termination: bool = ...,
    weak_equality: bool = ...,
) -> SearchResult: ...
def astar_search(
    ss: SearchSpace,
    heuristic: Heuristic,
    timeout: Optional[float] = ...,
    early_termination: bool = ...,
    weak_equality: bool = ...,
) -> SearchResult: ...
def astar_search_memory_bounded(
    ss: SearchSpace,
    heuristic: Heuristic,
    timeout: Optional[float] = ...,
    early_termination: bool = ...,
    weak_equality: bool = ...,
) -> SearchResult: ...
def gbfs_search(
    ss: SearchSpace,
    heuristic: Heuristic,
    timeout: Optional[float] = ...,
    early_termination: bool = ...,
    weak_equality: bool = ...,
) -> SearchResult: ...
def gbfs_search_memory_bounded(
    ss: SearchSpace,
    heuristic: Heuristic,
    timeout: Optional[float] = ...,
    early_termination: bool = ...,
    weak_equality: bool = ...,
) -> SearchResult: ...
def multiqueue_search(
    ss: SearchSpace,
    heuristics: list[tuple[Heuristic, float]],
    timeout: Optional[float] = ...,
    early_termination: bool = ...,
    weak_equality: bool = ...,
) -> SearchResult: ...
