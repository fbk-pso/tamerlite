# Type stubs for the `rustamer` compiled extension (PEP 561).
#
# `rustamer` is a PyO3 wheel with no inline annotations a type checker can read,
# so the public surface is described here. It mirrors the pure-Python fallback in
# `tamerlite.core` (the two backends expose an identical interface) — keep this in
# sync with `crates/rustamer/src/{lib,search,heuristic}.rs`,
# `crates/rustamer-base/src/`, and `src/tamerlite/core/`.

from collections.abc import Callable
from fractions import Fraction

# An expression is a flat, post-order list of nodes (see `make_*_node` builders).
Expression = list["ExpressionNode"]

# (plan, statistics); plan is None when no solution was found.
SearchResult = tuple[list["Action"] | None, dict[str, str]]

class ExpressionNode:
    """A single node of an expression. Exactly one accessor is non-None."""

    @property
    def fluent(self) -> int | None: ...
    @property
    def object(self) -> int | None: ...
    @property
    def bool_constant(self) -> bool | None: ...
    @property
    def int_constant(self) -> int | None: ...
    @property
    def real_constant(self) -> Fraction | None: ...
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
        actions_duration: list[tuple[Expression, Expression, bool, bool] | None],
        events: dict[Action, list[tuple[Timing, Event]]],
        actions: list[Action],
        compression_safe_actions: list[bool] | None,
        action_objects: list[list[int]] | None,
        obj_to_prev_actions_map: list[set[Action]] | None,
        initial_state: Expression | None = ...,
        goal: Expression | None = ...,
        relevant_actions: list[Action] | None = ...,
        deadline: Fraction | None = ...,
        epsilon: Fraction | None = ...,
    ) -> None: ...
    @property
    def is_temporal(self) -> bool: ...
    relevant_actions: list[Action]
    def reset(self) -> None: ...
    def initial_state(self, initial_state: Expression | None = ...) -> State: ...
    def get_successor_state(self, state: State, action: Action) -> State | None: ...
    def get_successor_states(self, state: State) -> list[State]: ...
    def goal_reached(self, state: State, goal: Expression | None = ...) -> bool: ...
    def subgoals_sat(
        self, state: State, goal: Expression | None = ...
    ) -> list[Expression]: ...
    # Each scheduled action is (start, action, duration); start/duration are
    # rational numbers serialized as "numer/denom" strings (not Fraction).
    def build_plan(
        self, path: list[Action]
    ) -> list[tuple[str | None, Action, str | None]] | None: ...

class Heuristic:
    @staticmethod
    def custom(
        callable: Callable[[State], float | None],
        cache_value_in_state: bool,
    ) -> Heuristic: ...
    @staticmethod
    def hff(
        actions: list[Action],
        fluent_types: list[str],
        objects: dict[str, list[int]],
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
        objects: dict[str, list[int]],
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
        objects: dict[str, list[int]],
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
        objects: dict[str, list[int]],
        events: dict[Action, list[tuple[Timing, Event]]],
        goals: list[ExpressionNode],
        internal_caching: bool,
        cache_value_in_state: bool,
        inadmissible_numeric_heuristic_variant: bool,
    ) -> Heuristic: ...
    @property
    def name(self) -> str: ...
    def eval(self, state: State, ss: SearchSpace) -> float | None: ...
    def reachable_actions(self, state: State) -> set[Action]: ...

# --- expression builders -------------------------------------------------

def make_operator_node(kind: str, operands: tuple[int, ...]) -> ExpressionNode: ...
def make_bool_constant_node(v: bool) -> ExpressionNode: ...
def make_int_constant_node(v: int) -> ExpressionNode: ...
def make_rational_constant_node(numerator: int, denominator: int) -> ExpressionNode: ...
def make_object_node(oid: int) -> ExpressionNode: ...
def make_fluent_node(fluent: int) -> ExpressionNode: ...
def shift_expression(exp: Expression, offset: int) -> Expression: ...
def get_fluents(exp: Expression) -> list[int]: ...
def evaluate(exp: Expression, state: State) -> ExpressionNode: ...
def simplify(exp: Expression, assignments: dict[int, ExpressionNode]) -> Expression: ...

# --- search algorithms ---------------------------------------------------

def bfs_search(
    ss: SearchSpace,
    timeout: float | None = ...,
    early_termination: bool = ...,
) -> SearchResult: ...
def dfs_search(
    ss: SearchSpace,
    timeout: float | None = ...,
    early_termination: bool = ...,
) -> SearchResult: ...
def ehc_search(
    ss: SearchSpace,
    heuristic: Heuristic,
    timeout: float | None = ...,
    early_termination: bool = ...,
    weak_equality: bool = ...,
) -> SearchResult: ...
def wastar_search(
    ss: SearchSpace,
    heuristic: Heuristic,
    weight: float,
    timeout: float | None = ...,
    early_termination: bool = ...,
    weak_equality: bool = ...,
) -> SearchResult: ...
def wastar_search_memory_bounded(
    ss: SearchSpace,
    heuristic: Heuristic,
    weight: float,
    timeout: float | None = ...,
    early_termination: bool = ...,
    weak_equality: bool = ...,
) -> SearchResult: ...
def astar_search(
    ss: SearchSpace,
    heuristic: Heuristic,
    timeout: float | None = ...,
    early_termination: bool = ...,
    weak_equality: bool = ...,
) -> SearchResult: ...
def astar_search_memory_bounded(
    ss: SearchSpace,
    heuristic: Heuristic,
    timeout: float | None = ...,
    early_termination: bool = ...,
    weak_equality: bool = ...,
) -> SearchResult: ...
def gbfs_search(
    ss: SearchSpace,
    heuristic: Heuristic,
    timeout: float | None = ...,
    early_termination: bool = ...,
    weak_equality: bool = ...,
) -> SearchResult: ...
def gbfs_search_memory_bounded(
    ss: SearchSpace,
    heuristic: Heuristic,
    timeout: float | None = ...,
    early_termination: bool = ...,
    weak_equality: bool = ...,
) -> SearchResult: ...
def multiqueue_search(
    ss: SearchSpace,
    heuristics: list[tuple[Heuristic, float]],
    timeout: float | None = ...,
    early_termination: bool = ...,
    weak_equality: bool = ...,
) -> SearchResult: ...
