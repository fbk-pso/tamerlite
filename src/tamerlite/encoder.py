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

from collections.abc import Callable, Iterable, MutableMapping
from fractions import Fraction
from typing import Any, cast

import unified_planning as up
from unified_planning.model import (
    Fluent,
    FNode,
    InterpretedFunction,
    Object,
    Problem,
    TimepointKind,
    Type,
)
from unified_planning.model.types import _UserType
from unified_planning.model.walkers import ExpressionQuantifiersRemover, Nnf
from unified_planning.plans import (
    ActionInstance,
    Plan,
    SequentialPlan,
    TimeTriggeredPlan,
)

from tamerlite.converter import Converter
from tamerlite.core import (
    Action,
    Effect,
    Event,
    Expression,
    HMax,
    SearchSpace,
    Timing,
    get_fluents,
)
from tamerlite.core.search_space import ConstantNode, SearchSpaceABC


def has_interpreted_functions(kind: up.model.ProblemKind) -> bool:
    return bool(
        kind.has_interpreted_functions_in_conditions()
        or kind.has_interpreted_functions_in_boolean_assignments()
        or kind.has_interpreted_functions_in_numeric_assignments()
        or kind.has_interpreted_functions_in_object_assignments()
        or kind.has_interpreted_functions_in_durations()
    )


def extract_objects(exp: FNode) -> Iterable[Object]:
    stack: list[FNode] = [exp]
    while len(stack) > 0:
        exp = stack.pop()
        if exp.is_object_exp():
            yield exp.object()
        else:
            stack.extend(exp.args)


def extract_fluents(exp: FNode) -> Iterable[Fluent]:
    stack: list[FNode] = [exp]
    while len(stack) > 0:
        exp = stack.pop()
        if exp.is_fluent_exp():
            yield exp.fluent()
        else:
            stack.extend(exp.args)


def extract_and_arguments(expressions: list[FNode]) -> Iterable[FNode]:
    stack: list[FNode] = list(expressions)
    while len(stack) > 0:
        exp = stack.pop()
        if exp.is_and():
            stack.extend(exp.args)
        else:
            yield exp


# Value recorded for a fluent appearing in a goal conjunct: the raw constant
# (bool/int/Fraction/Object), or a `(value, False)` pair marking a negated
# fluent-equals-constant comparison.
ConstantValue = bool | int | Fraction | Object
GoalFluentValue = ConstantValue | tuple[ConstantValue, bool]


class Encoder:
    """
    This class takes in input a Problem and builds its search space.
    If full is True, the initial and goal states are already initialized
    in the search space.
    """

    def __init__(
        self,
        problem: Problem,
        lifted_problem: Problem,
        map_back_action_instance: Callable[[ActionInstance], ActionInstance | None],
        symmetry_breaking: bool,
        compression_safe_actions: bool,
        relevance_analysis: bool,
        relevant_equality: bool = True,
        full: bool = True,
        deadline: Fraction | None = None,
        if_cache: MutableMapping[tuple[InterpretedFunction, tuple], Any] | None = None,
        if_wrappers: dict[InterpretedFunction, Callable] | None = None,
        lifted_problem_kind: up.model.ProblemKind | None = None,
    ):
        self._problem = problem
        self._lifted_problem = lifted_problem
        self._map_back_action_instance = map_back_action_instance
        self._lifted_problem_kind = lifted_problem_kind
        if full:
            self._simplifier = up.model.walkers.Simplifier(problem.environment, problem)
        else:
            self._simplifier = problem.environment.simplifier
        self._qrm = ExpressionQuantifiersRemover(problem.environment)
        self._nnf = Nnf(problem.environment)
        self._if_cache = if_cache
        self._if_wrappers = if_wrappers
        # A structural property of `problem`, unaffected by fluent numbering -- computed
        # once and handed to every `Converter` `_encode` builds (one per pass), instead
        # of letting each fresh `Converter` recompute it from scratch.
        self._static_fluents = problem.get_static_fluents()
        # Both caches below persist across `_encode`'s two possible passes (discovery,
        # then compaction) -- populated lazily by `_convert_fluent`/
        # `_normalize_expression`, never cleared or rebuilt by `_encode` itself, since
        # neither result depends on fluent numbering.
        self._fluent_name_cache: dict[FNode, str] = {}
        self._normalized_expression_cache: dict[FNode, FNode] = {}

        self._problem_initial_values = problem.initial_values

        # The object and action universes never change with fluent
        # numbering, so they're built once, up front -- `self._objects` in
        # particular must exist before `_encode`'s first call, since
        # `_compute_relevant_actions` (invoked between the two `_encode`
        # passes below) reads it while building its own `HMax` heuristic.
        self._objects_by_id = sorted(problem.all_objects, key=lambda o: o.name)
        self._object_names: list[str] = [o.name for o in self._objects_by_id]
        self._object_ids = {name: i for i, name in enumerate(self._object_names)}
        self._objects: dict[str, list[int]] = {}
        for ut in problem.user_types:
            self._objects[cast(_UserType, ut).name] = [
                self._object_ids[o.name] for o in problem.objects(ut)
            ]

        self._action_names: list[str] = sorted(
            action.name for action in problem.actions
        )
        self._action_by_name: dict[str, Action] = {
            name: Action(index) for index, name in enumerate(self._action_names)
        }
        self._actions: list[Action] = [
            self._action_by_name[name] for name in self._action_names
        ]

        # Pass 1 (discovery): every fluent gets a slot, exactly like `full`
        # being off would still do below. Needed unconditionally --
        # `_compute_relevant_fluents` (which decides what pass 2, if any,
        # keeps) has to run over *some* numbering, and `full=False` callers
        # (map-back-only encoders) just stop here.
        self._encode(relevant_fluents=None)

        initial_state = None
        self._goal = None
        action_objects = None
        obj_to_prev_actions_map = None
        self._compression_safe_actions = None
        if full:
            initial_state = self.initial_state(self._problem_initial_values)
            self._goal = self.goals(problem.goals)

            # Symmetry breaking and compression-safe detection only ever
            # inspect the raw UP model (`self._problem`/`self._lifted_problem`),
            # never a converted `Expression` -- unaffected by fluent
            # numbering, so computed once here rather than repeated around a
            # possible pass-2 re-encode below.
            if symmetry_breaking:
                action_objects, obj_to_prev_actions_map = (
                    self._compute_obj_to_prev_actions_map()
                )
                if not any(obj_to_prev_actions_map):
                    # Symmetry breaking is not beneficial because there are no
                    # equivalent objects
                    action_objects = None
                    obj_to_prev_actions_map = None

            if compression_safe_actions:
                self._compression_safe_actions = (
                    self._compute_compression_safe_actions()
                )
                if not any(self._compression_safe_actions):
                    # No actions are safe for compression
                    self._compression_safe_actions = None

        # Builds the encoding's `SearchSpace` -- tentatively final, unless
        # compaction below forces a rebuild. Built with pass 1's (possibly
        # uncompacted) `_actions_duration`/`_events`, but already carrying
        # every fluent-numbering-independent setting computed above, so the
        # common case (no compaction, or `relevant_equality=False`) needs
        # only this one construction.
        self._search_space = SearchSpace(
            self._actions_duration,
            self._events,
            self._actions,
            self._compression_safe_actions,
            action_objects,
            obj_to_prev_actions_map,
            initial_state,
            self._goal,
            self._applicable_actions,
            deadline,
            problem.epsilon,
        )

        self._relevant_actions = None
        if full:
            narrow_relevant_actions = False
            if relevance_analysis:
                # Needs a `State` to run `HMax.reachable_actions` against --
                # `self._search_space` above already provides one, regardless
                # of whether compaction ends up rebuilding it below.
                self._relevant_actions = self._compute_relevant_actions()
                narrow_relevant_actions = len(self._relevant_actions) < len(
                    self.applicable_actions
                )
                if narrow_relevant_actions:
                    self._search_space.relevant_actions = self._relevant_actions

            # Compact the encoding to the fluents that can affect search
            # outcome. Seeded from `considered_actions`, not every action in
            # the problem -- `_encode`'s pass 2 (via `_build_events`/
            # `_build_actions_duration`) restricts its own conversion to
            # exactly `considered_actions` too (see those methods), so the
            # seed set and the conversion set always match: a fluent read
            # only by a pruned action's precondition is safe to drop, because
            # that pruned action's precondition is never converted under the
            # new numbering either. Gated on `relevant_equality`: off keeps
            # every fluent, exactly like `main`.
            if relevant_equality:
                relevant_fluents = self._compute_relevant_fluents(
                    self.considered_actions
                )
                if len(relevant_fluents) < len(self._fluents):
                    # Pass 2 (final): re-encode restricted to `relevant_fluents`
                    # (indices in pass 1's numbering). Sound by construction of
                    # `_compute_relevant_fluents`'s fixpoint -- a fluent read by a
                    # condition, a goal, a duration bound, or the RHS of an effect
                    # targeting a kept fluent is always itself in
                    # `relevant_fluents`, so conversion never hits a missing id;
                    # only an effect whose *target* wasn't kept gets silently
                    # dropped (see `_convert_effects`), since nothing reads it
                    # anymore.
                    self._encode(relevant_fluents)
                    initial_state = self.initial_state(self._problem_initial_values)
                    self._goal = self.goals(problem.goals)
                    # Renumbering changed `_actions_duration`/`_events`, so
                    # the mutex/precedence analysis the `SearchSpace` above
                    # ran is now stale over the new numbering and must be
                    # redone -- `_compression_safe_actions`/`action_objects`/
                    # `obj_to_prev_actions_map` don't depend on fluent
                    # numbering at all, so they're reused as-is.
                    self._search_space = SearchSpace(
                        self._actions_duration,
                        self._events,
                        self._actions,
                        self._compression_safe_actions,
                        action_objects,
                        obj_to_prev_actions_map,
                        initial_state,
                        self._goal,
                        self._applicable_actions,
                        deadline,
                        problem.epsilon,
                    )
                    if narrow_relevant_actions:
                        assert self._relevant_actions is not None
                        self._search_space.relevant_actions = self._relevant_actions

    def _encode(self, relevant_fluents: set[int] | None) -> None:
        """(Re)builds everything whose numbering depends on which fluents
        exist: `_fluents`/`_fluent_ids`/`_fluent_types`, the `Converter` (a
        fresh instance every call -- `DagWalker` memoizes conversions per
        `FNode`, so an instance that already saw the previous numbering
        can't be reused), `_actions_duration`, and (via `_build_events`)
        `_events`/`_applicable_actions`.

        `relevant_fluents=None` means every fluent gets a slot -- pass 1
        (discovery), or the whole encoding when compaction never runs.
        Otherwise `relevant_fluents` is a set of *pass 1* fluent indices (as
        returned by `_compute_relevant_fluents`) and only those get a slot;
        see `__init__`'s pass-2 call for why every *read* is guaranteed to
        still resolve.
        """
        self._build_fluents(relevant_fluents)

        self._converter = Converter(
            self._problem,
            self._fluent_ids,
            self._object_ids,
            self._objects_by_id,
            self._if_cache,
            self._if_wrappers,
            self._static_fluents,
        )

        self._build_actions_duration(relevant_fluents)
        self._build_events(
            self.considered_actions if relevant_fluents is not None else None
        )

    def _build_fluents(self, relevant_fluents: set[int] | None) -> None:
        """Sets `_fluents`/`_fluent_ids`/`_fluent_types`. See `_encode` for what
        `relevant_fluents` means.

        Skips a full rescan on pass 2, when nothing renumbering could touch
        would actually change: `relevant_fluents` is always a subset of the
        fluents pass 1 (the only possible prior call) already numbered and
        typed, so pass 2 just filters those results by their pass-1 index
        instead of rescanning `self._problem_initial_values` and redoing FNode
        type inference for names it has already resolved.
        """
        if relevant_fluents is None:
            # Pass 1 (or a full, uncompacted encode): only place that ever needs to
            # scan `self._problem_initial_values`/infer each fluent's type from its
            # UP `FNode`.
            fluent_types = {}
            for f in self._problem_initial_values:
                name = self._convert_fluent(f)
                if f.type.is_bool_type():
                    t = "bool"
                elif f.type.is_int_type():
                    t = "int"
                elif f.type.is_real_type():
                    t = "real"
                elif f.type.is_user_type():
                    t = cast(_UserType, f.type).name
                else:
                    raise NotImplementedError
                fluent_types[name] = t
            self._fluents: list[str] = sorted(fluent_types.keys())
            self._fluent_ids = {f: i for i, f in enumerate(self._fluents)}
            self._fluent_types = [fluent_types[f] for f in self._fluents]
        else:
            # Pass 2: filter by pass-1 index rather than converting indices to
            # names first -- cheaper (int-set membership, no intermediate
            # `set[str]`) and skips the name lookup entirely. `self._fluents`
            # stays sorted since it's a subsequence of a sorted list.
            old_fluent_types = dict(zip(self._fluents, self._fluent_types, strict=True))
            self._fluents = [
                f for i, f in enumerate(self._fluents) if i in relevant_fluents
            ]
            self._fluent_ids = {f: i for i, f in enumerate(self._fluents)}
            self._fluent_types = [old_fluent_types[f] for f in self._fluents]

    def _build_actions_duration(self, relevant_fluents: set[int] | None) -> None:
        """Sets `_actions_duration`/`_is_temporal`. See `_encode` for what
        `relevant_fluents` means.

        Reuses pass 1's `_actions_duration` verbatim on pass 2 whenever no
        duration bound reads a fluent at all: renumbering fluents can't
        possibly change a duration bound's converted `Expression` if there's
        no `FluentNode` in it to renumber. That check only matters for this
        (pass 2) decision, so it's made here, on demand, from pass 1's
        already-built `self._actions_duration` -- not tracked eagerly as an
        attribute during pass 1 (which would pay for it even on a
        `relevant_equality=False` encode, or a `relevant_equality=True` one
        where pass 2 never runs at all, that never consults it).

        Otherwise, pass 2 only reconverts durative actions in
        `self.considered_actions` -- matching `_build_events`'s restriction --
        and reuses pass 1's already-converted (possibly stale-numbered) tuple
        verbatim for a non-considered durative action instead. That's safe
        for the same reason `_build_events`'s restriction is: nothing
        dereferences a non-considered action's `_actions_duration[a.idx]`
        content (only actions with an `_events` entry are ever opened, and
        `_build_events` restricts those to this same `considered_actions`
        set). `_is_temporal` itself stays unrestricted (a cheap `isinstance`
        check, no conversion) so its meaning -- "does *any* action in the
        problem have a duration" -- doesn't shift with which actions happen
        to be considered.
        """
        if relevant_fluents is not None and not any(
            # `next(iter(...), None) is not None` (not a bare truthiness
            # check) because `get_fluents` returns a plain `list[int]` on the
            # Rust backend but an `Iterator[int]` on the Python one, and
            # fluent id `0` is falsy -- a bare `bool(...)`/`any(...)` over the
            # ids themselves would misreport "no fluent read" whenever the
            # only fluent read happens to be id 0.
            entry is not None
            and (
                next(iter(get_fluents(entry[0])), None) is not None
                or next(iter(get_fluents(entry[1])), None) is not None
            )
            for entry in self._actions_duration
        ):
            # Pass 2, and no duration bound (from pass 1) reads a fluent at
            # all -- `_actions_duration`/`_is_temporal` from pass 1 are still
            # exactly correct; skip reconverting every action's duration
            # bounds a second time.
            return

        restrict = relevant_fluents is not None
        considered_names = (
            {self.get_action_name(a) for a in self.considered_actions}
            if restrict
            else None
        )
        actions_duration_map: dict[
            str, tuple[Expression, Expression, bool, bool] | None
        ] = {}
        self._is_temporal = False
        for a in self._problem.actions:
            if isinstance(a, up.model.DurativeAction):
                self._is_temporal = True
                if considered_names is not None and a.name not in considered_names:
                    actions_duration_map[a.name] = self._actions_duration[
                        self.action_by_name[a.name].idx
                    ]
                    continue
                actions_duration_map[a.name] = (
                    self._convert_expression(a.duration.lower),
                    self._convert_expression(a.duration.upper),
                    a.duration.is_left_open(),
                    a.duration.is_right_open(),
                )
            else:
                actions_duration_map[a.name] = None
        self._actions_duration: list[
            tuple[Expression, Expression, bool, bool] | None
        ] = [actions_duration_map[a] for a in self._action_names]

    def initial_state(self, initial_values: dict[FNode, FNode]) -> list[ConstantNode]:
        initial_state_values = {}
        for f, v in initial_values.items():
            initial_state_values[self._convert_fluent(f)] = self._convert_expression(v)[
                0
            ]

        initial_state = [initial_state_values[f] for f in self._fluents]
        # Initial values are always constants (no OperatorNode/FluentNode), so
        # narrowing the wider ExpressionNode element type to ConstantNode is safe.
        return cast(list[ConstantNode], initial_state)

    def _action_read_fluents(self, action: Action) -> set[int]:
        """The fluents an action reads outside of its own effect values: its
        event conditions and its duration bounds.

        `Event.start_conditions` is deliberately not walked. `_build_events`
        pushes an interval condition into the `start_conditions` and
        `end_conditions` buckets unconditionally and as a pair, with the same
        expression, so the `end_conditions` pass below already covers every
        fluent a `start_conditions` pass would -- the two entries sit on
        different events of this same action, and this method unions across
        all of them. If that pairing in `_build_events` ever becomes
        conditional, this method must start walking `start_conditions` too, or
        both relevance analyses below will silently under-approximate.

        Effect *values* are excluded because the two callers route them
        differently: `_compute_relevant_actions` folds them into the same
        dependency set as conditions, while `_compute_relevant_fluents` needs
        them keyed by the fluent the effect writes, to close over "an
        effect's RHS matters only if its target matters".
        """
        fluents: set[int] = set()
        for _, e in self.events[action]:
            fluents.update(get_fluents(e.conditions))
            for c in e.end_conditions:
                fluents.update(get_fluents(c))

        # An action's duration bounds are arbitrary expressions evaluated
        # against the pre-action state (see `SearchSpace._open_action`), so a
        # fluent read only there is still a genuine read.
        duration = self._actions_duration[action.idx]
        if duration is not None:
            fluents.update(get_fluents(duration[0]))
            fluents.update(get_fluents(duration[1]))
        return fluents

    def _compute_relevant_actions(self) -> list[Action]:
        """Computes the actions that are relevant for reaching the goal.

        This is a two-pass over-approximation, used to prune the search space
        to actions that could actually matter:

        1. **Forward reachability.** Builds an `HMax` heuristic over the
        applicable actions and runs its relaxed reachability analysis from
        the initial state. An action is *reachable* only if all of its
        events got a finite cost during the fixpoint (see
        `DeleteRelaxationHeuristic.reachable_actions`) -- i.e. it could ever
        be applied, start to finish, in the delete relaxation.
        2. **Backward goal-dependency walk.** Starting from the fluents in
        the goal, an action is *relevant* if it writes (via an effect) a
        fluent already known to be relevant. Once an action is marked
        relevant, the fluents it *depends on* -- those read by
        `_action_read_fluents` (its own conditions and duration bounds), plus
        those read by its own effect value expressions -- are added to the
        goal-dependency set and the walk continues from there.

        Both passes only ever narrow the search space (an action never
        pruned this way stays available), so this cannot make a solvable
        problem appear unsolvable through under-approximation.

        Returns:
            The subset of `self._actions` that are reachable and relevant;
            actions outside this set can never contribute to a plan.
        """
        assert self.goal is not None
        events = {a: e for a, e in self.events.items() if a in self.applicable_actions}
        heuristic = HMax(
            self.actions,
            self.fluent_types,
            self.objects,
            events,
            self.goal,
            internal_caching=False,
            cache_value_in_state=False,
            inadmissible_numeric_heuristic_variant=False,
        )
        reachable_actions = heuristic.reachable_actions(
            self._search_space.initial_state()
        )

        actions_affecting_fluent: dict[int, set[int]] = {}
        action_to_dependency_fluents: dict[int, set[int]] = {}
        for ra in reachable_actions:
            # `ra` comes from `heuristic.reachable_actions`, which is not
            # guaranteed to be the same object -- nor, under the Rust
            # backend, hash/eq-equal to the same object -- as the canonical
            # `Action` instance keying `events`/`self.events`. Re-fetch the
            # canonical instance by index (a plain list lookup) before using
            # it as a dict key.
            a = self._actions[ra.idx]
            action_to_dependency_fluents[a.idx] = self._action_read_fluents(a)

            for _, e in events[a]:
                for eff in e.effects:
                    actions_affecting_fluent.setdefault(eff.fluent, set()).add(a.idx)

                    # An effect's value expression can read other fluents
                    # (e.g. `g := mid_value`) with no corresponding
                    # precondition tying the two together -- those fluents
                    # must count as a dependency too, or the action that
                    # writes them is never pulled in as relevant.
                    action_to_dependency_fluents[a.idx].update(get_fluents(eff.value))

        checked_fluents = [False] * len(self._fluents)
        stack = list(get_fluents(self.goal))
        for f in stack:
            checked_fluents[f] = True

        relevant_actions: set[int] = set()
        while len(stack) > 0 and len(relevant_actions) < len(
            action_to_dependency_fluents
        ):
            f = stack.pop()
            relevant_actions.update(actions_affecting_fluent.get(f, set()))
            for action_idx in actions_affecting_fluent.get(f, set()):
                for f in action_to_dependency_fluents[action_idx]:
                    if not checked_fluents[f]:
                        checked_fluents[f] = True
                        stack.append(f)

        return [a for a in self._actions if a.idx in relevant_actions]

    def _compute_relevant_fluents(self, actions: list[Action]) -> set[int]:
        """Fluents that can affect search outcome: the least fixpoint of a
        backward slice from what search actually reads, over `actions`.

        Seeds are the fluents read directly by the goal, or by
        `_action_read_fluents` for an action (its conditions -- instantaneous
        or start/end-interval -- and its duration bounds). The closure rule
        is that an effect's RHS only matters if the fluent it writes
        matters: for every effect `f := expr`, once `f` is relevant every
        fluent read by `expr` becomes relevant too. A self-referencing
        assignment (`increase`/`decrease` desugars to e.g. `cost := cost +
        1`) only pulls `cost` in when it's already relevant, so a pure
        bookkeeping fluent read nowhere else, or one that only feeds another
        bookkeeping fluent transitively, is never seeded and never added.

        `__init__` calls this with `self.considered_actions`, matching what
        `_encode`'s pass 2 (`_build_events`/`_build_actions_duration`)
        restricts its own conversion to -- see those methods. Seeding from
        anything narrower than what gets converted would drop a fluent a
        still-converted action's precondition reads, and conversion would
        then fail to resolve it; seeding from anything broader (e.g. every
        action in the problem, including ones pruned from
        `considered_actions`) would keep fluents nothing converted ever
        reads.

        The result -- indices in pass 1's (uncompacted) numbering -- decides
        what `__init__`'s pass 2 keeps: every fluent outside it is dropped
        from the encoding entirely (`_encode`'s `relevant_fluents` parameter),
        so `state.assignments` only ever holds fluents that can affect search
        outcome. There is no separate consumer left to restrict after the
        fact -- dedup and the heuristics both just operate on the
        (potentially already-compacted) state.
        """
        relevant_fluents: set[int] = set(get_fluents(self.goal))  # type: ignore[arg-type]
        # Adjacency for the closure: fluent -> fluents read by the RHS of any
        # effect that writes it.
        written_from: dict[int, set[int]] = {}
        for a in actions:
            relevant_fluents.update(self._action_read_fluents(a))
            for _, e in self.events[a]:
                for eff in e.effects:
                    written_from.setdefault(eff.fluent, set()).update(
                        get_fluents(eff.value)
                    )

        # Closure: propagate relevance backward through effects.
        stack = list(relevant_fluents)
        while stack:
            for f in written_from.get(stack.pop(), ()):
                if f not in relevant_fluents:
                    relevant_fluents.add(f)
                    stack.append(f)
        return relevant_fluents

    def _compute_obj_to_prev_actions_map(
        self,
    ) -> tuple[list[list[int]], list[set[Action]]]:
        """
        This method produces two outputs:
            1. A list of lists of object ids, where each inner list corresponds
                to the objects used as parameters for the action.
            2. A list, indexed by object id, of the set of actions that include
                the previous equivalent object as a parameter (empty set if the
                object has no such constraint).

        Returns:
            Tuple[List[List[int]], List[Set[Action]]]:
                - List of object id lists for each action.
                - List, indexed by object id, of the set of actions.
        """

        equivalent_objects = self._compute_equivalent_objects()
        prev_equivalent_object = {}
        for group in equivalent_objects:
            for i, obj in enumerate(group):
                prev_equivalent_object[obj] = None if i == 0 else group[i - 1]

        obj_to_actions_map: dict[Object, set[Action]] = {}
        action_objects: list[list[int]] = [[] for _ in range(len(self.actions))]
        for action in self._problem.actions:
            ai = self._map_back_action_instance(action())
            assert ai is not None
            objects = [p.object() for p in ai.actual_parameters if p.is_object_exp()]
            action_objects[self.action_by_name[action.name].idx] = [
                self._object_ids[obj.name] for obj in objects
            ]
            for obj in objects:
                if obj not in obj_to_actions_map:
                    obj_to_actions_map[obj] = set()
                obj_to_actions_map[obj].add(self._action_by_name[action.name])

        obj_to_prev_actions_map: list[set[Action]] = [
            set() for _ in range(len(self._object_names))
        ]
        for obj, prev_obj in prev_equivalent_object.items():
            if prev_obj is not None and prev_obj in obj_to_actions_map:
                obj_to_prev_actions_map[self._object_ids[obj.name]] = (
                    obj_to_actions_map[prev_obj]
                )

        return action_objects, obj_to_prev_actions_map

    def _compute_equivalent_objects(self) -> list[list[Object]]:
        """
        Compute groups of equivalent objects in the problem.

        Returns:
            List[List[Object]]: A list of equivalence classes, where each inner
            list contains objects that are equivalent to each other.
        """

        goal_obj_to_fluent_map, goal_tainted_objects = (
            self._extract_goal_obj_to_fluent_map()
        )
        non_equivalent_objects = (
            self._extract_domain_objects()
            | goal_tainted_objects
            | self._extract_interpreted_function_tainted_objects()
        )
        obj_to_init_assignments = self._compute_obj_to_init_assignments_map()

        objects: dict[Type, list[Object]] = {}
        for obj in self._problem.all_objects:
            if obj.type not in objects:
                objects[obj.type] = []
            objects[obj.type].append(obj)

        groups = []
        for objs in objects.values():
            grouped = [False] * len(objs)
            for i, obj1 in enumerate(objs):
                if grouped[i]:
                    continue

                grouped[i] = True
                groups.append([obj1])

                if obj1 in non_equivalent_objects:
                    # treat all domain objects as non-equivalent objects
                    continue

                for j in range(i + 1, len(objs)):
                    if grouped[j]:
                        continue

                    obj2 = objs[j]
                    if obj2 in non_equivalent_objects:
                        continue

                    if self._are_equivalent_objects(
                        obj1,
                        obj2,
                        goal_obj_to_fluent_map,
                        obj_to_init_assignments,
                    ):
                        grouped[j] = True
                        groups[-1].append(obj2)

                groups[-1].sort(key=lambda obj: obj.name)

        return groups

    def _iter_lifted_action_expressions(self) -> Iterable[FNode]:
        """
        Yield every expression appearing in a lifted action's preconditions,
        conditions, effect conditions/fluents/values, or duration bounds.
        """

        for a in self._lifted_problem.actions:
            if isinstance(a, up.model.InstantaneousAction):
                yield from a.preconditions
                for e in a.effects:
                    if e.is_conditional():
                        yield e.condition
                    yield e.fluent
                    yield e.value
            elif isinstance(a, up.model.DurativeAction):
                yield a.duration.lower
                yield a.duration.upper
                for cl in a.conditions.values():
                    yield from cl
                for el in a.effects.values():
                    for e in el:
                        if e.is_conditional():
                            yield e.condition
                        yield e.fluent
                        yield e.value

    def _iter_metric_expressions(self) -> Iterable[FNode]:
        """
        Yield every expression appearing in the problem's quality metrics.
        """

        for qm in self._lifted_problem.quality_metrics:
            if isinstance(
                qm,
                (
                    up.model.metrics.MinimizeExpressionOnFinalState,
                    up.model.metrics.MaximizeExpressionOnFinalState,
                ),
            ):
                yield qm.expression
            elif isinstance(qm, up.model.metrics.MinimizeActionCosts):
                for cost in (*qm.costs.values(), qm.default):
                    if cost is not None:
                        yield cost

    def _extract_domain_objects(self) -> set[Object]:
        """
        Extract all objects that appear in the problem's domain.

        Returns:
            Set[Object]: A set of all objects that appear in the domain.
        """

        return set(self._lifted_problem.domain_constants)

    def _extract_interpreted_function_tainted_objects(self) -> set[Object]:
        """
        Extract objects that an interpreted function (IF) call could observe
        or produce, and which must therefore be excluded from equivalence.

        An IF is opaque: we can't reason about its behavior, only require its
        inputs be swap-invariant. Numeric/boolean values are swap-invariant by
        construction. Object-typed arguments or return values are not and they
        can change under the swap, and the IF is free to react to that
        difference however it wants. So for every IF call reachable from the
        lifted problem, every object compatible with an object-typed parameter
        or the return type (i.e. that type and its subtypes, matching how objects
        could actually be substituted in) is tainted.

        Returns:
            Set[Object]: A set of objects that must be treated as
            non-equivalent because of an interpreted function.
        """

        if not has_interpreted_functions(self.lifted_problem_kind):
            return set()

        ifun_calls: set[FNode] = set()
        extractor = self._lifted_problem.environment.interpreted_functions_extractor
        expressions: list[FNode] = list(self._iter_lifted_action_expressions())
        expressions.extend(self._iter_metric_expressions())
        expressions.extend(self._lifted_problem.goals)
        for goals in self._lifted_problem.timed_goals.values():
            expressions.extend(goals)
        for effects in self._lifted_problem.timed_effects.values():
            for e in effects:
                if e.is_conditional():
                    expressions.append(e.condition)
                expressions.append(e.fluent)
                expressions.append(e.value)
        for (
            fluent_exp,
            value_exp,
        ) in self._lifted_problem.explicit_initial_values.items():
            expressions.append(fluent_exp)
            expressions.append(value_exp)
        for exp in expressions:
            ifun_calls.update(extractor.get(exp))

        tainted_objects: set[Object] = set()
        for call in ifun_calls:
            ifun = call.interpreted_function()
            for param in ifun.signature:
                if param.type.is_user_type():
                    tainted_objects.update(self._problem.objects(param.type))
            if ifun.return_type.is_user_type():
                tainted_objects.update(self._problem.objects(ifun.return_type))
        return tainted_objects

    def _compute_obj_to_init_assignments_map(
        self,
    ) -> dict[Object, list[tuple[FNode, FNode]]]:
        """
        Build a mapping from each object to the initial-value assignments it
        participates in, either as a fluent argument or as the assigned value.

        Uses `initial_values` (the complete grounded initial state, defaults
        included) rather than `explicit_initial_values`, so that an object
        used only as a fluent's default value is checked precisely by
        transposition in `_are_equivalent_objects` instead of needing to be
        conservatively excluded from equivalence altogether.

        Returns:
            Dict[Object, List[Tuple[FNode, FNode]]]: Mapping from objects to
            the list of (fluent expression, value expression) assignments they
            appear in.
        """

        obj_to_assignments: dict[Object, list[tuple[FNode, FNode]]] = {}
        for fluent_exp, value_exp in self._problem_initial_values.items():
            objs = {arg.object() for arg in fluent_exp.args if arg.is_object_exp()}
            if value_exp.is_object_exp():
                objs.add(value_exp.object())
            for obj in objs:
                obj_to_assignments.setdefault(obj, []).append((fluent_exp, value_exp))
        return obj_to_assignments

    def _extract_goal_obj_to_fluent_map(
        self,
    ) -> tuple[
        dict[Object, set[tuple[Fluent, tuple[Object, ...], GoalFluentValue]]],
        set[Object],
    ]:
        """
        Build a mapping from objects to goal fluents they appear in.

        The goal (`problem.goals`, and recursively any nested conjunction) is
        decomposed into individual conjuncts. A conjunct is precisely
        understood only if it has one of 4 recognized shapes: a fluent, a
        negated fluent, a fluent compared to a constant, or its negation.
        Objects appearing in any OTHER conjunct (of unrecognized shape, e.g. a
        disjunction, an implication, or a comparison between two fluents) are
        collected into a separate "tainted" set instead of being registered in
        the map: we don't know how to verify that swapping them preserves that
        conjunct, so they must be excluded from equivalence altogether -- but
        this must not affect objects that only ever appear in recognized
        conjuncts elsewhere in the goal.

        Returns:
            Tuple[Dict[Object, Set[Tuple[Fluent, Tuple[Object, ...], GoalFluentValue]]],
            Set[Object]]:
                - A dictionary mapping each object to the set of associated
                  recognized-conjunct entries.
                - The set of objects appearing in some unrecognized conjunct,
                  who must be excluded from equivalence.
        """

        obj_to_fluent_map: dict[
            Object, set[tuple[Fluent, tuple[Object, ...], GoalFluentValue]]
        ] = {obj: set() for obj in self._problem.all_objects}

        def extract_fluent_equals_constant_exp(
            arg1: FNode, arg2: FNode, is_negated: bool
        ) -> bool:
            fluent_exp = None
            value_exp = None
            if arg1.is_fluent_exp() and arg2.is_constant():
                fluent_exp = arg1
                value_exp = arg2
                v = arg2.constant_value()
            elif arg2.is_fluent_exp() and arg1.is_constant():
                fluent_exp = arg2
                value_exp = arg1
                v = arg1.constant_value()

            if fluent_exp is None:
                return False
            else:
                value = (v, False) if is_negated else v
                fluent = fluent_exp.fluent()
                objs = tuple(
                    arg.object() for arg in fluent_exp.args if arg.is_object_exp()
                )
                entry_objs = set(objs)
                assert value_exp is not None
                if value_exp.is_object_exp():
                    entry_objs.add(value_exp.object())
                for obj in entry_objs:
                    obj_to_fluent_map[obj].add((fluent, objs, value))

                return True

        tainted_objects: set[Object] = set()
        stack: list[FNode] = list(self._problem.goals)
        while len(stack) > 0:
            exp = stack.pop()
            if exp.is_fluent_exp():
                fluent = exp.fluent()
                objs = tuple(arg.object() for arg in exp.args if arg.is_object_exp())
                for obj in objs:
                    obj_to_fluent_map[obj].add((fluent, objs, True))

            elif exp.is_not() and exp.args[0].is_fluent_exp():
                exp = exp.args[0]
                fluent = exp.fluent()
                objs = tuple(arg.object() for arg in exp.args if arg.is_object_exp())
                for obj in objs:
                    obj_to_fluent_map[obj].add((fluent, objs, False))

            elif exp.is_equals():
                arg1, arg2 = exp.args
                if not extract_fluent_equals_constant_exp(arg1, arg2, False):
                    tainted_objects.update(extract_objects(exp))

            elif exp.is_not() and exp.args[0].is_equals():
                arg1, arg2 = exp.args[0].args
                if not extract_fluent_equals_constant_exp(arg1, arg2, True):
                    tainted_objects.update(extract_objects(exp))

            elif exp.is_and():
                stack.extend(exp.args)

            else:
                tainted_objects.update(extract_objects(exp))

        return obj_to_fluent_map, tainted_objects

    def _are_equivalent_objects(
        self,
        obj1: Object,
        obj2: Object,
        goal_obj_to_fluent_map: dict[
            Object, set[tuple[Fluent, tuple[Object, ...], GoalFluentValue]]
        ],
        obj_to_init_assignments: dict[Object, list[tuple[FNode, FNode]]],
    ) -> bool:
        """
        Determine whether two objects are equivalent in the problem, i.e.
        whether swapping them everywhere (as fluent arguments and as
        object-valued fluent values) leaves the goal and initial state
        unchanged.

        Args:
            obj1 (Object): The first object to compare.
            obj2 (Object): The second object to compare.
            goal_obj_to_fluent_map
                (Dict[Object, Set[Tuple[Fluent, Tuple[Object, ...], GoalFluentValue]]]):
                Mapping from objects to the recognized goal fluents they
                appear in (as an argument or as the compared value). Objects
                appearing in an unrecognized goal conjunct are excluded from
                equivalence before reaching this method (see
                `_extract_goal_obj_to_fluent_map`), so this map can be trusted
                to precisely and completely describe every goal constraint
                that could possibly distinguish obj1/obj2.
            obj_to_init_assignments (Dict[Object, List[Tuple[FNode, FNode]]]):
                Mapping from objects to the initial-value assignments
                (explicit or default) they appear in (as an argument or as
                the value).

        Returns:
            bool: True if the objects are equivalent; False otherwise.
        """

        def transpose(x: Object) -> Object:
            return obj2 if x == obj1 else obj1 if x == obj2 else x

        def transpose_constant(c: ConstantValue) -> ConstantValue:
            return transpose(c) if isinstance(c, Object) else c

        def transpose_value(v: GoalFluentValue) -> GoalFluentValue:
            if isinstance(v, tuple):
                return (transpose_constant(v[0]), v[1])
            return transpose_constant(v)

        if len(goal_obj_to_fluent_map[obj1]) != len(goal_obj_to_fluent_map[obj2]):
            # the two objects appear in a different number of goal fluents
            return False

        # for each goal fluent involving obj1, ensure the corresponding
        # fluent (with obj1/obj2 swapped in both the arguments and the
        # compared value) exists for obj2
        for fluent, objs1, v in goal_obj_to_fluent_map[obj1]:
            objs2 = tuple(transpose(obj) for obj in objs1)
            v2 = transpose_value(v)
            if (fluent, objs2, v2) not in goal_obj_to_fluent_map[obj2]:
                return False

        # For each initial-value assignment (explicit or default) involving
        # obj1 or obj2 (as an argument or as the value), swap obj1 and obj2
        # throughout and verify that the resulting assignment still holds.
        obj1_exp = self._problem.environment.expression_manager.ObjectExp(obj1)
        obj2_exp = self._problem.environment.expression_manager.ObjectExp(obj2)

        def swap_exp(exp: FNode) -> FNode:
            if exp == obj1_exp:
                return obj2_exp
            if exp == obj2_exp:
                return obj1_exp
            return exp

        seen_fluent_exps: set[FNode] = set()
        assignments = obj_to_init_assignments.get(
            obj1, []
        ) + obj_to_init_assignments.get(obj2, [])
        for fluent_exp, value_exp in assignments:
            if fluent_exp in seen_fluent_exps:
                continue
            seen_fluent_exps.add(fluent_exp)

            new_fluent_exp = self._problem.environment.expression_manager.FluentExp(
                fluent_exp.fluent(), [swap_exp(arg) for arg in fluent_exp.args]
            )
            if self._problem.initial_value(new_fluent_exp) != swap_exp(value_exp):
                return False

        return True

    def _compute_compression_safe_actions(self) -> list[bool]:
        actions = [False] * len(self.action_names)
        fluent_to_conditions, complex_condition_fluents = self._extract_conditions()
        for action in self._problem.actions:
            if (
                isinstance(action, up.model.DurativeAction)
                and not self._has_intermediate_conditions(action)
                and self._end_conditions_contained_in_overall_conditions(action)
                and not self._effects_interfere_with_conditions(
                    action, fluent_to_conditions, complex_condition_fluents
                )
            ):
                actions[self.action_by_name[action.name].idx] = True

        return actions

    def _extract_conditions(self) -> tuple[dict[Fluent, set[bool]], set[Fluent]]:
        fluent_to_conditions: dict[Fluent, set[bool]] = {}
        complex_condition_fluents: set[Fluent] = set()
        for action in self._problem.actions:
            action_conditions: list[list[FNode]]
            if isinstance(action, up.model.DurativeAction):
                action_conditions = list(action.conditions.values())
            else:
                assert isinstance(action, up.model.InstantaneousAction)
                action_conditions = [action.preconditions]
            for conds in action_conditions:
                for c in extract_and_arguments(conds):
                    f = None
                    if c.is_fluent_exp():
                        f = c.fluent()
                        v = True
                    elif c.is_not() and c.arg(0).is_fluent_exp():
                        f = c.arg(0).fluent()
                        v = False
                    else:
                        complex_condition_fluents.update(extract_fluents(c))

                    if f is not None:
                        if f not in fluent_to_conditions:
                            fluent_to_conditions[f] = set()
                        fluent_to_conditions[f].add(v)

        return fluent_to_conditions, complex_condition_fluents

    def _has_intermediate_conditions(self, action: "up.model.DurativeAction") -> bool:
        return any(
            interval.lower.delay != 0 or interval.upper.delay != 0
            for interval in action.conditions
        )

    def _end_conditions_contained_in_overall_conditions(
        self, action: "up.model.DurativeAction"
    ) -> bool:
        end_conditions: set[FNode] = set()
        overall_conditions: set[FNode] = set()
        for interval, conditions in action.conditions.items():
            if (
                interval.lower == interval.upper
                and interval.lower.timepoint.kind == TimepointKind.END
                and interval.lower.delay == 0
            ):
                end_conditions.update(extract_and_arguments(conditions))

            elif (
                interval.lower.timepoint.kind == TimepointKind.START
                and interval.upper.timepoint.kind == TimepointKind.END
                and interval.lower.delay == 0
                and interval.upper.delay == 0
            ):
                overall_conditions.update(extract_and_arguments(conditions))

        return all(condition in overall_conditions for condition in end_conditions)

    def _effects_interfere_with_conditions(
        self,
        action: "up.model.DurativeAction",
        fluent_to_conditions: dict[Fluent, set[bool]],
        complex_condition_fluents: set[Fluent],
    ) -> bool:
        for timing, effects in action.effects.items():
            if timing.timepoint.kind == TimepointKind.START and timing.delay == 0:
                continue

            for eff in effects:
                f = eff.fluent.fluent()
                if not eff.value.is_bool_constant():
                    return True

                negated_value = not eff.value.bool_constant_value()
                if (
                    f in complex_condition_fluents
                    or negated_value in fluent_to_conditions.get(f, set())
                ):
                    return True

        return False

    def goals(self, goals: list[FNode]) -> Expression:
        return self._convert_expression(
            self._problem.environment.expression_manager.And(goals)
        )

    @property
    def problem(self) -> Problem:
        return self._problem

    @property
    def lifted_problem(self) -> Problem:
        return self._lifted_problem

    @property
    def lifted_problem_kind(self) -> up.model.ProblemKind:
        """
        Lazily resolve and cache `_lifted_problem.kind`. `Problem.kind` is a
        from-scratch full-problem scan, so it's computed on first use here
        (instead of unconditionally in `__init__`) rather than paying for it
        in encoders that never need it -- unless a caller already has it and
        supplied it via the constructor.
        """

        if self._lifted_problem_kind is None:
            self._lifted_problem_kind = self._lifted_problem.kind
        return self._lifted_problem_kind

    @property
    def search_space(self) -> SearchSpaceABC:
        return self._search_space

    @property
    def fluents(self) -> list[str]:
        return self._fluents

    @property
    def fluent_ids(self) -> dict[str, int]:
        return self._fluent_ids

    @property
    def fluent_types(self) -> list[str]:
        return self._fluent_types

    @property
    def objects(self) -> dict[str, list[int]]:
        return self._objects

    @property
    def object_ids(self) -> dict[str, int]:
        return self._object_ids

    @property
    def object_names(self) -> list[str]:
        return self._object_names

    @property
    def events(self) -> dict[Action, list[tuple[Timing, Event]]]:
        return self._events

    @property
    def actions(self) -> list[Action]:
        return self._actions

    @property
    def action_names(self) -> list[str]:
        return self._action_names

    @property
    def action_by_name(self) -> dict[str, Action]:
        return self._action_by_name

    @property
    def applicable_actions(self) -> list[Action]:
        return self._applicable_actions

    @property
    def relevant_actions(self) -> list[Action] | None:
        return self._relevant_actions

    @property
    def considered_actions(self) -> list[Action]:
        """The action set the search will actually expand: `relevant_actions`
        if relevance analysis narrowed it, else `applicable_actions`. Consumed
        by `TamerLite._get_heuristic` to narrow the heuristics' operator set,
        and by `_compute_relevant_fluents`/`_build_events`/
        `_build_actions_duration`, which all restrict themselves to exactly
        this same set (see their docstrings) so the relevance fixpoint's seed
        set always matches what pass 2 actually converts."""
        return (
            self._relevant_actions
            if self._relevant_actions is not None
            else self._applicable_actions
        )

    @property
    def compression_safe_actions(self) -> list[Action]:
        if self._compression_safe_actions is None:
            return []
        return [a for a in self._actions if self._compression_safe_actions[a.idx]]

    @property
    def goal(self) -> Expression | None:
        return self._goal

    def get_action(self, name: str) -> Action:
        return self.action_by_name[name]

    def get_action_name(self, action: Action) -> str:
        return self.action_names[action.idx]

    def are_all_actions_compression_safe(self) -> bool:
        return self._compression_safe_actions is not None and all(
            self._compression_safe_actions
        )

    def is_any_action_compression_safe(self) -> bool:
        return self._compression_safe_actions is not None and any(
            self._compression_safe_actions
        )

    def build_plan(self, path: list[Action]) -> Plan:
        plan = self.search_space.build_plan(path)
        if self._is_temporal:
            actions = []
            for s, a, d in plan:
                assert s is not None
                actions.append((s, self._problem.action(self.get_action_name(a))(), d))
            return TimeTriggeredPlan(actions)
        else:
            return SequentialPlan(
                [self._problem.action(self.get_action_name(a))() for _, a, _ in plan]
            )

    def _convert_fluent(self, fluent_exp: FNode) -> str:
        # Purely structural (`str(FNode)`), independent of fluent numbering, so the
        # same string is reused verbatim across both `_encode` passes and every
        # call site (`_encode`'s fluent-type loop, `_convert_effects`, `initial_state`).
        cached = self._fluent_name_cache.get(fluent_exp)
        if cached is None:
            cached = str(fluent_exp)
            self._fluent_name_cache[fluent_exp] = cached
        return cached

    def _normalize_expression(self, expression: FNode) -> FNode:
        """Quantifier removal + simplification + NNF conversion -- everything
        `_convert_expression` does before handing off to `self._converter`, which is
        the only fluent-numbering-dependent part. Independent of fluent numbering (a
        pure function of `expression` and `self._problem`, neither of which changes
        between `_encode` passes), so cached across both passes instead of re-running
        this from scratch on every action's preconditions/conditions/effects/duration
        bounds a second time when pass 2 (compaction) runs.
        """
        cached = self._normalized_expression_cache.get(expression)
        if cached is None:
            cached = self._qrm.remove_quantifiers(expression, self._problem)
            cached = self._simplifier.simplify(cached)
            cached = self._nnf.get_nnf_expression(cached)
            self._normalized_expression_cache[expression] = cached
        return cached

    def _convert_expression(self, expression: FNode) -> Expression:
        return self._converter.convert(self._normalize_expression(expression))

    def _convert_timing(self, timing: "up.model.Timing") -> Timing:
        return Timing(timing.is_from_start(), Fraction(timing.delay))

    def _convert_effects(self, effects: list["up.model.Effect"]) -> list[Effect]:
        env = self._problem.environment
        em = env.expression_manager
        fluent_to_effects: dict[FNode, list[list[FNode]]] = {}
        for effect in effects:
            if effect.fluent not in fluent_to_effects:
                fluent_to_effects[effect.fluent] = [[], [], []]

            if effect.is_increase():
                fluent_to_effects[effect.fluent][0].append(effect.value)
            elif effect.is_decrease():
                fluent_to_effects[effect.fluent][1].append(effect.value)
            else:
                fluent_to_effects[effect.fluent][2].append(effect.value)

        converted_effects = []
        for fluent, (
            inc_effects,
            dec_effects,
            assign_effects,
        ) in fluent_to_effects.items():
            fluent_id = self.fluent_ids.get(self._convert_fluent(fluent))
            if fluent_id is None:
                # Compaction dropped this fluent as irrelevant -- nothing
                # reads it, so the effect that would write it can't matter
                # either (see `_encode`/`_compute_relevant_fluents`).
                continue

            some_inc_dec_effects = len(inc_effects) > 0 or len(dec_effects) > 0
            some_assign_effects = len(assign_effects) > 0
            is_bool_type = fluent.fluent().type.is_bool_type()
            assert (some_inc_dec_effects and not some_assign_effects) or (
                not some_inc_dec_effects and some_assign_effects
            )

            if some_assign_effects:
                if len(assign_effects) == 1:
                    value = assign_effects[0]
                elif not is_bool_type:
                    # NOTE: If multiple numeric assignment effects are present,
                    # they are assumed to be identical
                    value = assign_effects[0]
                    for v in assign_effects:
                        assert value == v
                else:
                    value = assign_effects[0]
                    non_constant_assignments = 0
                    for v in assign_effects:
                        if v.is_bool_constant():
                            if v.bool_constant_value():
                                value = v
                                non_constant_assignments = 0
                                break
                        else:
                            value = v
                            non_constant_assignments += 1

                    if non_constant_assignments > 1:
                        raise Exception(
                            "TamerLite does not support multiple non-constant "
                            "boolean assignment effects on the same fluent."
                        )
            else:
                if len(inc_effects) > 0:
                    if len(dec_effects) > 0:
                        value = em.Minus(
                            em.Plus([fluent, *inc_effects]), em.Plus(dec_effects)
                        )
                    else:
                        value = em.Plus([fluent, *inc_effects])
                else:
                    value = em.Minus(fluent, em.Plus(dec_effects))

            converted_value = self._convert_expression(value)
            converted_effects.append(Effect(fluent_id, converted_value))

        return converted_effects

    def _build_events(self, considered_actions: list[Action] | None = None) -> None:
        """Sets `_events` (and, when unrestricted, `_applicable_actions`).

        `considered_actions=None` means every action gets an `_events` entry
        (pass 1, or an unrestricted encode), and `_applicable_actions` is
        (re)computed from this same full scan. Otherwise (pass 2) only
        actions in `considered_actions` get converted and get an `_events`
        entry at all -- both `SearchSpace` implementations only ever
        dereference an action's events for actions in
        `SearchSpace.relevant_actions` (which the encoder sets to
        `considered_actions`'s content), so a missing entry for anything else
        is never read. `_applicable_actions` is left untouched in this case:
        it's a public, whole-problem property (`Encoder.applicable_actions`)
        that must not shrink to match a restricted pass -- and since
        applicability never depends on fluent numbering, pass 1's value is
        still exactly correct, so the (otherwise redundant) `simplify()`
        calls that would recompute it are skipped entirely too.
        """
        env = self._problem.environment
        em = env.expression_manager
        self._events: dict[Action, list[tuple[Timing, Event]]] = {}
        restrict = considered_actions is not None
        considered_names = (
            None
            if considered_actions is None
            else {self.get_action_name(a) for a in considered_actions}
        )
        applicable_actions = set()
        for a in self._problem.actions:
            if considered_names is not None and a.name not in considered_names:
                continue
            action = self.get_action(a.name)
            if isinstance(a, up.model.DurativeAction):
                from_start: dict[Any, Any] = {}
                from_end: dict[Any, Any] = {}
                action_events: list[
                    tuple[int | Fraction, up.model.Timing, int, list]
                ] = []
                is_applicable = True
                for i, lc in a.conditions.items():
                    lower = i.lower
                    upper = i.upper
                    if lower == upper:  # conditions
                        action_events.append((lower.delay, lower, 1, lc))
                    else:
                        # lower: start conditions
                        if not i.is_left_open():
                            action_events.append((lower.delay, lower, 1, lc))
                        action_events.append((lower.delay, lower, 2, [em.And(lc)]))
                        # upper: end conditions
                        if not i.is_right_open():
                            action_events.append((upper.delay, upper, 1, lc))
                        action_events.append((upper.delay, upper, 3, [em.And(lc)]))
                    if not restrict:
                        is_applicable = (
                            is_applicable
                            and not self._simplifier.simplify(em.And(lc)).is_false()
                        )
                if not restrict and is_applicable:
                    applicable_actions.add(self.get_action(a.name))

                for t, le in a.effects.items():
                    action_events.append((t.delay, t, 4, le))

                has_ice_from_start = False
                has_ice_from_end = False
                for d, t, p, e in action_events:
                    if t.is_from_start():
                        from_start.setdefault(d, (t, [], [], [], []))
                        from_start[d][p].extend(e)
                        if d > 0:
                            has_ice_from_start = True
                    else:
                        from_end.setdefault(d, (t, [], [], [], []))
                        from_end[d][p].extend(e)
                        if d < 0:
                            has_ice_from_end = True

                if has_ice_from_start and has_ice_from_end:
                    dur_lower, dur_upper = a.duration.lower, a.duration.upper
                    if (
                        dur_lower.is_constant()
                        and dur_upper.is_constant()
                        and dur_lower.constant_value() == dur_upper.constant_value()
                    ):
                        duration = dur_lower.constant_value()
                        for d in from_end:
                            t, lc, lsc, lec, le = from_end[d]
                            d_from_start = duration + d
                            from_start.setdefault(d_from_start, (t, [], [], [], []))
                            from_start[d_from_start][1].extend(lc)
                            from_start[d_from_start][2].extend(lsc)
                            from_start[d_from_start][3].extend(lec)
                            from_start[d_from_start][4].extend(le)
                        from_end.clear()
                    else:
                        raise Exception(
                            "TamerLite does not support ICE from start and from "
                            "end inside the same action!"
                        )

                if from_start and min(from_start) < 0:
                    # Either a directly authored negative `StartTiming`, or an
                    # end-relative timing that the fold above mapped before the
                    # action's own start. Such an event would sort ahead of the
                    # start event below and break the invariant it establishes.
                    raise Exception(
                        "TamerLite does not support conditions or effects placed "
                        f"before the start of a durative action (action `{a.name}`)!"
                    )

                if 0 not in from_start:
                    # Every durative action must own an event at its start
                    # timepoint. The search opens an action when its *first*
                    # event fires and reads the duration bounds from that
                    # state (`SearchSpace._open_action`), so an action whose
                    # first event sits at the end -- or at an intermediate
                    # `start + delay` -- would be sized from the wrong state.
                    # A trivially-true, effect-less event restores the
                    # invariant without changing the action's semantics; it
                    # also covers the degenerate action with no conditions
                    # and no effects at all, which would otherwise end up
                    # with an empty event list.
                    from_start[0] = (up.model.StartTiming(), [], [], [], [])

                events: list[tuple[Timing, Event]] = []
                pos = 0
                for d in sorted(from_start):
                    t, lc, lsc, lec, le = from_start[d]
                    conv_t = self._convert_timing(t)
                    c = self._convert_expression(em.And(lc))
                    tsc = tuple([self._convert_expression(sc) for sc in lsc])
                    tec = tuple([self._convert_expression(ec) for ec in lec])
                    te = tuple(self._convert_effects(le))
                    events.append((conv_t, Event(action, pos, c, tsc, tec, te)))
                    pos += 1
                for d in sorted(from_end):
                    t, lc, lsc, lec, le = from_end[d]
                    conv_t = self._convert_timing(t)
                    c = self._convert_expression(em.And(lc))
                    tsc = tuple([self._convert_expression(sc) for sc in lsc])
                    tec = tuple([self._convert_expression(ec) for ec in lec])
                    te = tuple(self._convert_effects(le))
                    events.append((conv_t, Event(action, pos, c, tsc, tec, te)))
                    pos += 1
                self._events[action] = events
            else:
                assert isinstance(a, up.model.InstantaneousAction)
                conv_t = Timing(True, Fraction(0))
                te = tuple(self._convert_effects(a.effects))
                self._events[action] = [
                    (
                        conv_t,
                        Event(
                            action,
                            0,
                            self._convert_expression(em.And(a.preconditions)),
                            (),
                            (),
                            te,
                        ),
                    )
                ]
                if (
                    not restrict
                    and not self._simplifier.simplify(
                        em.And(a.preconditions)
                    ).is_false()
                ):
                    applicable_actions.add(action)

        if not restrict:
            self._applicable_actions = [
                a for a in self._actions if a in applicable_actions
            ]
