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

"""Partitioned numeric novelty for `tamerlite.core.search.novbfs_search`.

Every action/event condition and the goal is flattened (through nested
``and``/``or`` only) down to leaf subexpressions ("subgoals"). Each subgoal
gets a propositional feature psi (is it satisfied?) and, if it is a numeric
comparison, a distance feature delta (how close is it?). A state's novelty
class (1 = a size-1 conjunction of features is new, 2 = only a size-2
conjunction is new, 3 = nothing is new) is evaluated relative to its parent
state, within a table partitioned by `floor(h)`, where `h` is whatever
heuristic the search was configured with, -- so novelty stays
meaningful deep into a long search instead of diluting against the entire
search history. See `NumericNovelty` below.

This module intentionally does not implement the `Heuristic` ABC
(`tamerlite.core.heuristics`): novelty is not a pure function of a state --
it needs the parent state and mutates persistent per-partition tables as a
side effect, once per generated state, in generation order -- so it is
consumed directly by `tamerlite.core.search.novbfs_search` rather than
through the heuristic dispatch machinery.

Implementation notes -- the structure mirrors `NumericNovelty` in
`crates/rustamer-base/src/novelty.rs` wherever the idiom translates:

- Each leaf is precompiled once, at construction, into a `_PropLeaf` or a
  `_NumLeaf` (Rust's `LeafData::Prop`/`LeafData::Num`), whose operands are
  `_Operand` callables (Rust's `Operand`): a bare fluent reads
  ``state.assignments`` directly and a bare constant returns itself, so
  only a genuinely compound operand goes through `evaluate`.
- Per-partition tables are sparse (``dict``/``set``), allocated lazily on
  first use per partition (see `NumericNovelty._get_partition`), rather than
  dense arrays sized eagerly for every subgoal -- a sparse table never
  allocates what it doesn't touch, so there's no quadratic up-front
  allocation to guard against and no need for a size-based fallback that
  would degrade novelty precision on large problems.
- The partition count is frozen at ``max(1, floor(h(s0)))`` when a
  search starts, and any state whose h exceeds the initial state's is
  clamped into the top partition -- a known sharp edge on problems where
  h is not monotone (temporal problems especially: a durative action's
  very first event can raise h above `h(s0)` more readily than in
  classical planning). A heuristic that is constant (e.g. ``blind``)
  degenerates to a single partition, i.e. unpartitioned novelty.
- Subgoal leaves are deduplicated directly by structural equality:
  `tamerlite.core.search_space.Expression` is already a frozen, hashable,
  structurally-comparable tuple, so no separate id-remapping table is
  needed (see `_leaf_index` below).
- A subgoal's propositional-vs-numeric classification is structural for
  ``<=``/``<`` (always numeric -- only `int`/`Fraction` support ordering).
  ``==`` covers both numeric and object equality, so it is classified
  statically from each operand's `FluentDomain` via
  `tamerlite.core.search_space.is_object_typed_operand`: numeric iff neither
  operand is object-typed.
  A ``not(...)`` leaf is always treated as propositional-only.
- The distance feature (`_compute_sdist`) is computed once per state, in one
  place, alongside satisfaction (avoiding a second, redundant evaluation of
  both operands just to get a boolean already implicit in the distance), in
  an "improves = larger sdist" sign convention -- ``evaluate(rhs) -
  evaluate(lhs)`` for ``<=``/``<`` and ``-abs(evaluate(rhs) -
  evaluate(lhs))`` for ``==`` -- so every caller (Pass A, B, C) reads one
  cached value instead of recomputing and re-normalizing it per use.
- Fluent values here are exact `Fraction`/`int`, so distance-improvement
  comparisons are exact rather than floating-point-precision sensitive. An
  all-`int` difference stays an `int` (Rust's `Sdist::Small`); only a
  `Fraction` operand yields a `Fraction` (Rust's `Sdist::Big`). The two
  compare exactly against each other, so no normalization is needed.
- Same-partition evaluation is sparse. A leaf's value is a pure function of
  the fluents it reads, so a leaf none of whose fluents changed between
  `parent` and `state` ("clean") has exactly its parent's value and can
  never be newly satisfied or improved. `eval` therefore re-evaluates only
  the "dirty" leaves (`_mark_dirty_leaves`, via the `_fluent_to_leaves`
  index), returns 3 straight away if none of them made progress (Pass B and
  C are both driven by `newly_satisfied`/`numeric_improved`, so they would
  be no-ops), and otherwise fills in the clean leaves' contribution from a
  once-per-expansion snapshot of the parent (`_ensure_parent_snapshot`).
  The dirty set may over-approximate (it compares values by identity, see
  `_mark_dirty_leaves`) -- re-evaluating a clean leaf is harmless -- but
  must never miss a changed leaf.
- A propositional leaf is satisfied iff it evaluates to the literal boolean
  `True` -- ``value is True``, not truthiness (`bool(...)`). This matches
  Rust's `is_true` (`matches!(v, ExpressionNode::Bool(true))`), which only
  ever treats `Bool(true)` as satisfied. Not reachable from the encoder
  today -- every propositional leaf here is boolean-valued -- but the two
  backends must agree on `expanded_states`/`goal_depth` exactly, so this is
  stated explicitly rather than left to happen to agree.

Where the Rust core deliberately differs in implementation (never in
outcome): it invalidates its parent caches and snapshot by bumping
generation stamps and reuses its per-call classification buffers (to avoid
reallocating, and deallocating `Sdist::Big`s), where this module simply
rebinds fresh ``dict``/``list``s; it diffs parent and child by comparing
`im::Vector` chunk pointers, where this module compares plain lists
element-wise; and it packs pair keys into a ``u64`` where this module uses
tuples.
"""

import itertools
import math
import operator
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from fractions import Fraction

from tamerlite.core.heuristics import get_event_conditions
from tamerlite.core.search_space import (
    Action,
    ConstantNode,
    Event,
    Expression,
    FluentDomain,
    FluentNode,
    ObjectNode,
    OperatorNode,
    State,
    Timing,
    evaluate,
    extract_sub_expression,
    is_object_typed_operand,
)

# A numeric leaf's distance feature: an exact `int` when both operands are
# integers (Rust's `Sdist::Small`), else an exact `Fraction` (`Sdist::Big`).
_Sdist = int | Fraction

# A leaf operand, precompiled once (Rust's `Operand`): evaluates it in a state.
_Operand = Callable[[State], ConstantNode]


def _make_operand(exp: Expression) -> _Operand:
    """Precompiles `exp` into an `_Operand` (mirrors Rust's
    `Operand::from_vec`): a bare fluent reads `state.assignments` directly
    and a bare constant returns itself, skipping `evaluate`'s per-call work
    for the overwhelmingly common case; anything else goes through
    `evaluate`."""
    if len(exp) == 1:
        node = exp[0]
        if isinstance(node, FluentNode):
            idx = node.fluent.idx
            return lambda state: state.assignments[idx]
        if isinstance(node, (int, Fraction, ObjectNode)):  # bool is an int subclass
            return lambda state: node
    return lambda state: evaluate(exp, state)


def _iter_subgoal_leaves(exp: Expression) -> Iterator[Expression]:
    """Flatten `exp` through nested `and`/`or` down to its leaf
    subexpressions ("subgoals"), iteratively (no recursion-depth risk on
    deeply nested and/or trees). `and`/`or` operands are just indices into
    `exp`, so the walk stays in `exp`'s own index space and only calls
    `extract_sub_expression` once a genuine leaf is reached -- extracting
    at every and/or level instead would re-shift the same leaves once per
    level of nesting."""
    stack = [len(exp) - 1]
    while stack:
        idx = stack.pop()
        node = exp[idx]
        if isinstance(node, OperatorNode) and node.kind in ("and", "or"):
            # Push in reverse so operands are still popped (and thus
            # yielded) in their original left-to-right order.
            stack.extend(reversed(node.operands))
        else:
            yield extract_sub_expression(exp, idx)


@dataclass(frozen=True, slots=True)
class _PropLeaf:
    """A propositional leaf (Rust's `LeafData::Prop`)."""

    operand: _Operand


@dataclass(frozen=True, slots=True)
class _NumLeaf:
    """A numeric leaf ``lhs <kind> rhs``, `kind` one of ``"<="``/``"<"``/
    ``"=="`` (Rust's `LeafData::Num`)."""

    lhs: _Operand
    rhs: _Operand
    kind: str


_Leaf = _PropLeaf | _NumLeaf


def _compute_sdist(leaf: _NumLeaf, state: State) -> tuple[_Sdist, bool]:
    """The distance feature for a numeric leaf -- already in the "improves
    = larger is better" sign convention (see module docstring): satisfied
    inequalities are >= 0, satisfied equalities are exactly 0, and both
    worsen (decrease) as the subgoal moves further from being satisfied --
    paired with its satisfaction boolean, computed from the same operand
    evaluation rather than a second, redundant evaluation of the whole leaf.
    Satisfaction is *not* simply `sdist >= 0`: for strict `<`, the exact
    boundary (`sdist == 0`) is genuinely unsatisfied, so the comparator
    needs to stay kind-aware here too. Mirrors Rust's `compute_sdist`."""
    diff: _Sdist = leaf.rhs(state) - leaf.lhs(state)  # type: ignore[operator]
    kind = leaf.kind
    if kind == "==":
        return (diff if diff <= 0 else -diff), diff == 0  # -abs(diff)
    if kind == "<=":
        return diff, diff >= 0
    return diff, diff > 0


@dataclass
class _PartitionTables:
    """Persistent novelty-tracking state for one `floor(h)` partition.
    One instance per partition, allocated lazily on first use (see
    `NumericNovelty._get_partition`)."""

    # psi: subgoal (leaf id) has been satisfied at least once in this partition.
    psi_seen: set[int] = field(default_factory=set)
    # delta: best (max) sdist ever seen for a numeric subgoal in this partition.
    best_sdist: dict[int, _Sdist] = field(default_factory=dict)
    # psi x psi: unordered pairs of subgoals jointly satisfied at least once.
    psi_pair_seen: set[tuple[int, int]] = field(default_factory=set)
    # psi x delta: best (max) sdist seen for a numeric subgoal conditioned on
    # a propositional subgoal, keyed (prop_leaf_id, numeric_leaf_id) directed.
    best_sdist_with_psi: dict[tuple[int, int], _Sdist] = field(default_factory=dict)


class NumericNovelty:
    """Partitioned numeric novelty over the subgoals of `events`/`goals`.

    Construct once per search (subgoal catalogue only, no state needed), call
    `start(initial_h)` once to fix the partition count, then call
    `begin_expansion()` once per expanded (popped) state followed by
    `eval(...)` once per surviving successor, in generation order, passing
    its parent. Not thread-safe. Reuse across multiple `novbfs_search` calls
    on the same instance (e.g. `TamerLite._solve_ground_problem`'s
    `weak_equality` retry, which calls the same bound `partial` -- hence the
    same instance -- twice) is safe *because* `start` fully resets
    `_partitions`/`_max_partition` and the parent-feature caches; it must be
    called again, before any further `eval()`, whenever reused this way. A
    fresh instance is still constructed per anytime cold-restart iteration
    (see `TamerLite._anytime_solutions`), but that is `TamerLite` starting
    over from scratch, not a requirement this class imposes.

    A parent's own features (psi truth, sdist) are a pure function of
    `(leaf, parent_state)`: every child of one expansion shares the same
    parent, so `eval` fills `_parent_prop_true`/`_parent_numeric` lazily on
    first use per expansion instead of recomputing them for every child,
    and builds the dense `_parent_snapshot` (every leaf satisfied in the
    parent, every numeric leaf unsatisfied in it) at most once per
    expansion. `begin_expansion` drops all three; not calling it before a
    new parent's children would silently reuse a stale parent's features.
    """

    def __init__(
        self,
        events: dict[Action, list[tuple[Timing, Event]]],
        goals: Expression,
        fluent_domains: list[FluentDomain],
    ):
        self._leaves: list[_Leaf] = []
        self._leaf_index: dict[Expression, int] = {}
        leaf_to_fluents: list[list[int]] = []

        def add_leaf(exp: Expression) -> None:
            """Adds `exp` to the leaf catalogue, deduplicated by structural
            equality (mirrors Rust's `add_leaf`). ``<=``/``<`` are always
            numeric; ``==`` is numeric iff neither operand is object-typed
            (see the module docstring); anything else is propositional."""
            if exp in self._leaf_index:
                return
            self._leaf_index[exp] = len(self._leaves)
            leaf_to_fluents.append(
                [n.fluent.idx for n in exp if isinstance(n, FluentNode)]
            )
            root = exp[-1]
            leaf: _Leaf
            if isinstance(root, OperatorNode) and (
                root.kind in ("<=", "<")
                or (
                    root.kind == "=="
                    and not is_object_typed_operand(
                        exp[root.operands[0]], fluent_domains
                    )
                    and not is_object_typed_operand(
                        exp[root.operands[1]], fluent_domains
                    )
                )
            ):
                op1, op2 = root.operands
                leaf = _NumLeaf(
                    _make_operand(extract_sub_expression(exp, op1)),
                    _make_operand(extract_sub_expression(exp, op2)),
                    root.kind,
                )
            else:
                leaf = _PropLeaf(_make_operand(exp))
            self._leaves.append(leaf)

        for event_list in events.values():
            for _, event in event_list:
                for cond in get_event_conditions(event):
                    for leaf in _iter_subgoal_leaves(cond):
                        add_leaf(leaf)
        for leaf in _iter_subgoal_leaves(goals):
            add_leaf(leaf)

        # `_fluent_to_leaves[f]`: every leaf reading fluent `f` (see
        # `_mark_dirty_leaves`). Sized to cover every fluent a leaf reads,
        # so no leaf's fluent can fall outside it.
        n_fluents = max(
            [len(fluent_domains)] + [f + 1 for fs in leaf_to_fluents for f in fs]
        )
        fluent_to_leaves: list[list[int]] = [[] for _ in range(n_fluents)]
        for leaf_id, fluents in enumerate(leaf_to_fluents):
            for f in fluents:
                bucket = fluent_to_leaves[f]
                if not bucket or bucket[-1] != leaf_id:
                    bucket.append(leaf_id)
        self._fluent_to_leaves: list[tuple[int, ...]] = [
            tuple(b) for b in fluent_to_leaves
        ]

        self._partitions: dict[int, _PartitionTables] = {}
        self._max_partition = 1
        # Lazy per-leaf caches of the current expansion's parent state's
        # features, plus the dense parent snapshot -- reset by
        # `begin_expansion`/`start`, filled on first use by `eval`. See the
        # class docstring.
        self._parent_prop_true: dict[int, bool] = {}
        self._parent_numeric: dict[int, tuple[_Sdist, bool]] = {}
        self._parent_snapshot: tuple[list[int], list[int]] | None = None

    def start(self, initial_h: float) -> int:
        """(Re)initializes partition bookkeeping from the initial state's
        heuristic value and clears the parent-feature caches. Must be called
        exactly once, before any `eval()` call. Returns the root's (clamped)
        partition id; the caller is responsible for seeding the tables with
        an explicit `eval()` call on the initial state and then pushing the
        root with novelty hard-coded to 1, regardless of that call's return
        value (see `novbfs_search`)."""
        assert initial_h >= 0, (
            "initial_h must be non-negative (novbfs partitions on "
            "floor(h), so its heuristic must never return a negative value)"
        )
        self._partitions = {}
        self._max_partition = max(1, math.floor(initial_h))
        self.begin_expansion()
        return self.partition_of(initial_h)

    def partition_of(self, h_value: float) -> int:
        """The partition function: `floor(h_value)`, clamped at the top to
        `max_partition`."""
        assert h_value >= 0, (
            "h_value must be non-negative (novbfs partitions on "
            "floor(h), so its heuristic must never return a negative value)"
        )
        return min(math.floor(h_value), self._max_partition)

    def begin_expansion(self) -> None:
        """Resets the lazy parent-feature caches and the parent snapshot.
        Must be called once per expansion, before the first `eval()` call
        for that expansion's children -- not enforced here (the sole caller,
        `novbfs_search`, gets this right by construction)."""
        self._parent_prop_true = {}
        self._parent_numeric = {}
        self._parent_snapshot = None

    def _get_partition(self, partition: int) -> _PartitionTables:
        tables = self._partitions.get(partition)
        if tables is None:
            tables = _PartitionTables()
            self._partitions[partition] = tables
        return tables

    def _parent_prop(self, leaf_id: int, leaf: _PropLeaf, parent: State) -> bool:
        """`leaf`'s truth in `parent`, through the per-expansion cache."""
        p_true = self._parent_prop_true.get(leaf_id)
        if p_true is None:
            p_true = leaf.operand(parent) is True
            self._parent_prop_true[leaf_id] = p_true
        return p_true

    def _parent_num(
        self, leaf_id: int, leaf: _NumLeaf, parent: State
    ) -> tuple[_Sdist, bool]:
        """`leaf`'s `(sdist, satisfied)` in `parent`, through the
        per-expansion cache."""
        cached = self._parent_numeric.get(leaf_id)
        if cached is None:
            cached = _compute_sdist(leaf, parent)
            self._parent_numeric[leaf_id] = cached
        return cached

    def _mark_dirty_leaves(self, parent: State, state: State) -> dict[int, None]:
        """Every leaf reading a fluent whose value may differ between
        `parent` and `state` (mirrors Rust's `mark_dirty_leaves`), as an
        insertion-ordered ``dict`` used as a set. Fluent values are compared
        by identity, all in C: a child is `parent.clone()` plus a handful of
        effect assignments, so an untouched slot still holds the very same
        object. An assignment that happens to write back an equal but
        distinct object is reported dirty too -- a harmless
        over-approximation (a clean leaf re-evaluates to its parent's value)
        -- but a genuinely changed value can never be missed."""
        assert len(parent.assignments) == len(state.assignments)
        fluent_to_leaves = self._fluent_to_leaves
        n_fluents = len(fluent_to_leaves)
        dirty: dict[int, None] = {}
        for f in itertools.compress(
            itertools.count(),
            map(operator.is_not, parent.assignments, state.assignments),
        ):
            if f < n_fluents:
                dirty.update(dict.fromkeys(fluent_to_leaves[f]))
        return dirty

    def _ensure_parent_snapshot(self, parent: State) -> tuple[list[int], list[int]]:
        """The dense parent snapshot `(parent_sat, parent_num_unsat)`: every
        leaf satisfied in `parent`, and every numeric leaf unsatisfied in it
        (mirrors Rust's `ensure_parent_snapshot`). Built at most once per
        expansion, lazily -- only the first time one of its children's
        sparse Pass A finds progress (see `eval`) -- reusing whatever
        per-leaf parent features are already cached."""
        snapshot = self._parent_snapshot
        if snapshot is None:
            parent_sat: list[int] = []
            parent_num_unsat: list[int] = []
            for leaf_id, leaf in enumerate(self._leaves):
                if isinstance(leaf, _PropLeaf):
                    if self._parent_prop(leaf_id, leaf, parent):
                        parent_sat.append(leaf_id)
                elif self._parent_num(leaf_id, leaf, parent)[1]:
                    parent_sat.append(leaf_id)
                else:
                    parent_num_unsat.append(leaf_id)
            snapshot = self._parent_snapshot = (parent_sat, parent_num_unsat)
        return snapshot

    def eval(
        self,
        state: State,
        partition: int,
        parent: State | None,
        parent_partition: int | None,
    ) -> int:
        """Has `state` made progress on something not seen before?

        Every subgoal (leaf, see `__init__`) contributes a propositional
        signal psi (is it satisfied?) and, if numeric, a distance signal
        delta (how close, and did it just beat its best-ever value?).
        Returns `state`'s novelty class relative to its parent, judged
        against the `partition` (`floor(h)`) bucket's own history so
        far: 1 (some single subgoal was newly satisfied, or some single
        numeric subgoal's distance just beat its personal best in this
        partition), 2 (no single subgoal did, but some *pair* of subgoals
        became jointly satisfied -- or a satisfied one paired with an
        improving one -- for the first time), or 3 (nothing new).
        `parent`/`parent_partition` are `state`'s generating state and its
        novelty partition, or both `None` for the root -- like entering a
        fresh partition, that means "nothing seen here yet," so everything
        currently true counts as newly added.

        As a side effect, updates this partition's persistent tables --
        calling this twice on the same state/partition will not return the
        same answer the second time."""

        new_partition = parent is None or parent_partition != partition

        newly_satisfied: list[int] = []
        currently_satisfied: list[int] = []
        persisting_satisfied: list[int] = []
        numeric_improved: list[int] = []
        numeric_unsatisfied: list[int] = []
        # Holds exactly the leaves Pass A evaluated this call (every leaf on
        # a fresh partition, else only the dirty ones); a clean leaf's
        # sdist is its parent's cached one instead -- see C1b.
        sdist_cache: dict[int, _Sdist] = {}

        if new_partition:
            # Pass A, dense: nothing has been generated in this partition
            # yet, so there is no parent to diff against -- every leaf is
            # evaluated, and everything currently true/satisfied counts as
            # newly added.
            for leaf_id, leaf in enumerate(self._leaves):
                if isinstance(leaf, _PropLeaf):
                    # `is True`, not `bool(...)`: only a literal boolean
                    # true satisfies a propositional leaf (matches Rust's
                    # `is_true`; see the module docstring).
                    if leaf.operand(state) is True:
                        currently_satisfied.append(leaf_id)
                        newly_satisfied.append(leaf_id)
                    continue
                sdist, curr_sat = _compute_sdist(leaf, state)
                sdist_cache[leaf_id] = sdist
                if curr_sat:
                    currently_satisfied.append(leaf_id)
                    newly_satisfied.append(leaf_id)
                else:
                    numeric_unsatisfied.append(leaf_id)
                    numeric_improved.append(leaf_id)
        else:
            assert parent is not None
            dirty = self._mark_dirty_leaves(parent, state)

            # Pass A, sparse: only the dirty leaves can differ from the
            # parent, hence only they can be newly satisfied or improved
            # (see the module docstring).
            for leaf_id in dirty:
                leaf = self._leaves[leaf_id]
                if isinstance(leaf, _PropLeaf):
                    p_true = self._parent_prop(leaf_id, leaf, parent)
                    if leaf.operand(state) is True:
                        currently_satisfied.append(leaf_id)
                        if p_true:
                            persisting_satisfied.append(leaf_id)
                        else:
                            newly_satisfied.append(leaf_id)
                    continue
                pdist, parent_sat = self._parent_num(leaf_id, leaf, parent)
                sdist, curr_sat = _compute_sdist(leaf, state)
                sdist_cache[leaf_id] = sdist
                if curr_sat:
                    currently_satisfied.append(leaf_id)
                else:
                    numeric_unsatisfied.append(leaf_id)
                if not parent_sat and sdist > pdist:
                    if curr_sat:
                        newly_satisfied.append(leaf_id)
                    else:
                        numeric_improved.append(leaf_id)
                elif parent_sat and curr_sat:
                    persisting_satisfied.append(leaf_id)

            if not newly_satisfied and not numeric_improved:
                # Nothing dirty made progress, and a clean leaf never can --
                # Pass B and C's outer loops are both driven by these two
                # lists, so both are guaranteed to be no-ops.
                return 3

            # Merge in every clean leaf the sparse loop skipped: its value
            # in `state` is its parent's.
            parent_sat_list, parent_num_unsat_list = self._ensure_parent_snapshot(
                parent
            )
            for leaf_id in parent_sat_list:
                if leaf_id not in dirty:
                    currently_satisfied.append(leaf_id)
                    persisting_satisfied.append(leaf_id)
            numeric_unsatisfied.extend(
                leaf_id for leaf_id in parent_num_unsat_list if leaf_id not in dirty
            )

        tables = self._get_partition(partition)
        novelty = 3

        # Pass B: unary (size-1) novelty.
        for leaf_id in newly_satisfied:
            if leaf_id not in tables.psi_seen:
                tables.psi_seen.add(leaf_id)
                novelty = 1
        for leaf_id in numeric_improved:
            sdist = sdist_cache[leaf_id]
            prev = tables.best_sdist.get(leaf_id)
            if prev is None or sdist > prev:
                tables.best_sdist[leaf_id] = sdist
                novelty = 1

        # Pass C: binary (size-2) novelty. Always runs, even if Pass B
        # already found novelty 1, so the pair tables stay current for
        # future calls.
        #
        # C1: psi x psi -- a newly-added subgoal paired with any currently-
        # true subgoal (including itself-as-numeric-satisfied). On a fresh
        # partition, `start_idx = ax + 1` is only correct because
        # `newly_satisfied` and `currently_satisfied` are appended in
        # lockstep above and are therefore element-identical (every subgoal
        # currently true is also newly satisfied when nothing has been seen
        # in this partition yet, and vice versa): skipping the prefix up to
        # `ax` in `currently_satisfied` is skipping exactly the pairs
        # `(newly_satisfied[0..ax], f)` already enumerated by earlier outer
        # iterations, so each unordered pair is still visited exactly once.
        # If a future edit ever appended to one list and not the other on
        # the `new_partition` branch, this would silently stop visiting some
        # pairs -- no error, just a different novelty class -- hence the
        # assertion.
        assert not new_partition or newly_satisfied == currently_satisfied
        for ax, f in enumerate(newly_satisfied):
            start_idx = ax + 1 if new_partition else 0
            for tid in itertools.islice(currently_satisfied, start_idx, None):
                if f == tid:
                    continue
                pair = (f, tid) if f < tid else (tid, f)
                if pair not in tables.psi_pair_seen:
                    tables.psi_pair_seen.add(pair)
                    novelty = min(novelty, 2)

        # C1b: psi (newly added) x delta (any still-unsatisfied numeric).
        for f in newly_satisfied:
            for tid in numeric_unsatisfied:
                if f == tid:
                    continue
                # `numeric_unsatisfied` (unlike `numeric_improved`) can hold
                # a clean leaf on a same-partition call, which Pass A never
                # evaluated: its sdist is its parent's cached one.
                sdist = (
                    sdist_cache[tid]
                    if tid in sdist_cache
                    else self._parent_numeric[tid][0]
                )
                pair = (f, tid)
                old = tables.best_sdist_with_psi.get(pair)
                if old is None or sdist > old:
                    tables.best_sdist_with_psi[pair] = sdist
                    novelty = min(novelty, 2)

        # C2: psi (persisting, true before and after) x delta (improved but
        # still unsatisfied numeric subgoal). No extra "genuinely
        # unsatisfied" filter here: every leaf in `numeric_improved` is
        # already unsatisfied by construction (Pass A only appends to it
        # under `not curr_sat`), exactly as in C1b above -- an explicit
        # `sdist < 0` check here would additionally exclude a strict `<`
        # leaf sitting exactly at `sdist == 0` (unsatisfied, since strict
        # `<` requires `sdist > 0`), which C1b does process.
        for tid in numeric_improved:
            sdist = sdist_cache[tid]
            for f in persisting_satisfied:
                if f == tid:
                    continue
                pair = (f, tid)
                old = tables.best_sdist_with_psi.get(pair)
                if old is None or sdist > old:
                    tables.best_sdist_with_psi[pair] = sdist
                    novelty = min(novelty, 2)

        return novelty
