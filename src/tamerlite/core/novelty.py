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
state, within a table partitioned by `floor(h^add)` -- so novelty stays
meaningful deep into a long search instead of diluting against the entire
search history. See `NumericNovelty` below.

This module intentionally does not implement the `Heuristic` ABC
(`tamerlite.core.heuristics`): novelty is not a pure function of a state --
it needs the parent state and mutates persistent per-partition tables as a
side effect, once per generated state, in generation order -- so it is
consumed directly by `tamerlite.core.search.novbfs_search` rather than
through the heuristic dispatch machinery.

Implementation notes:

- Per-partition tables are sparse (``dict``/``set``), allocated lazily on
  first use per partition (see `NumericNovelty._get_partition`), rather than
  dense arrays sized eagerly for every subgoal -- a sparse table never
  allocates what it doesn't touch, so there's no quadratic up-front
  allocation to guard against and no need for a size-based fallback that
  would degrade novelty precision on large problems.
- The partition count is frozen at ``max(1, floor(h^add(s0)))`` when a
  search starts, and any state whose h^add exceeds the initial state's is
  clamped into the top partition -- a known sharp edge on problems where
  h^add is not monotone (temporal problems especially: a durative action's
  very first event can raise h^add above `h^add(s0)` more readily than in
  classical planning).
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
- The distance feature (`_sdist_and_sat`) is computed once per state, in one
  place, alongside satisfaction (avoiding a second, redundant evaluation of
  both operands just to get a boolean already implicit in the distance), in
  an "improves = larger sdist" sign convention -- ``evaluate(rhs) -
  evaluate(lhs)`` for ``<=``/``<`` and ``-abs(evaluate(rhs) -
  evaluate(lhs))`` for ``==`` -- so every caller (Pass A, B, C) reads one
  cached value instead of recomputing and re-normalizing it per use.
- Fluent values here are exact `Fraction`/`int`, so distance-improvement
  comparisons are exact rather than floating-point-precision sensitive.
- A propositional leaf is satisfied iff it evaluates to the literal boolean
  `True` -- ``evaluate(leaf, state) is True``, not truthiness (`bool(...)`).
  This matches Rust's `is_true` (`matches!(v, ExpressionNode::Bool(true))`,
  `crates/rustamer-base/src/novelty.rs`), which only ever treats
  `Bool(true)` as satisfied. Not reachable from the encoder today -- every
  propositional leaf here is boolean-valued -- but the two backends must
  agree on `expanded_states`/`goal_depth` exactly, so this is stated
  explicitly rather than left to happen to agree.
"""

import itertools
import math
from collections.abc import Iterator
from dataclasses import dataclass, field
from fractions import Fraction

from tamerlite.core.heuristics import get_event_conditions
from tamerlite.core.search_space import (
    Action,
    Event,
    Expression,
    FluentDomain,
    OperatorNode,
    State,
    Timing,
    evaluate,
    extract_sub_expression,
    is_object_typed_operand,
)


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


@dataclass
class _PartitionTables:
    """Persistent novelty-tracking state for one `floor(h^add)` partition.
    One instance per partition, allocated lazily on first use (see
    `NumericNovelty._get_partition`)."""

    # psi: subgoal (leaf id) has been satisfied at least once in this partition.
    psi_seen: set[int] = field(default_factory=set)
    # delta: best (max) sdist ever seen for a numeric subgoal in this partition.
    best_sdist: dict[int, Fraction] = field(default_factory=dict)
    # psi x psi: unordered pairs of subgoals jointly satisfied at least once.
    psi_pair_seen: set[tuple[int, int]] = field(default_factory=set)
    # psi x delta: best (max) sdist seen for a numeric subgoal conditioned on
    # a propositional subgoal, keyed (prop_leaf_id, numeric_leaf_id) directed.
    best_sdist_with_psi: dict[tuple[int, int], Fraction] = field(default_factory=dict)


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
    first use per expansion instead of recomputing them from scratch for
    every child, as a naive per-call `evaluate(leaf, parent)` would.
    `begin_expansion` drops the cache; not calling it before a new parent's
    children would silently reuse a stale parent's features.
    """

    def __init__(
        self,
        events: dict[Action, list[tuple[Timing, Event]]],
        goals: Expression,
        fluent_domains: list[FluentDomain],
    ):
        self._leaves: list[Expression] = []
        self._leaf_index: dict[Expression, int] = {}
        # (leaf_id, lhs, rhs, kind) for "<="/"<" (always numeric) and "=="
        # (numeric iff neither operand is object-typed).
        self._numeric_leaves: dict[int, tuple[Expression, Expression, str]] = {}

        def add_leaf(exp: Expression) -> None:
            if exp in self._leaf_index:
                return
            idx = len(self._leaves)
            self._leaves.append(exp)
            self._leaf_index[exp] = idx
            root = exp[-1]
            if isinstance(root, OperatorNode) and root.kind in ("<=", "<"):
                lhs = extract_sub_expression(exp, root.operands[0])
                rhs = extract_sub_expression(exp, root.operands[1])
                self._numeric_leaves[idx] = (lhs, rhs, root.kind)
            elif isinstance(root, OperatorNode) and root.kind == "==":
                op1, op2 = root.operands
                if not is_object_typed_operand(
                    exp[op1], fluent_domains
                ) and not is_object_typed_operand(exp[op2], fluent_domains):
                    lhs = extract_sub_expression(exp, op1)
                    rhs = extract_sub_expression(exp, op2)
                    self._numeric_leaves[idx] = (lhs, rhs, "==")

        for event_list in events.values():
            for _, event in event_list:
                for cond in get_event_conditions(event):
                    for leaf in _iter_subgoal_leaves(cond):
                        add_leaf(leaf)
        for leaf in _iter_subgoal_leaves(goals):
            add_leaf(leaf)

        self._partitions: dict[int, _PartitionTables] = {}
        self._max_partition = 1
        # Lazy per-leaf caches of the current expansion's parent state's
        # features -- reset by `begin_expansion`/`start`, filled on first
        # use by `eval`. See the class docstring.
        self._parent_prop_true: dict[int, bool] = {}
        self._parent_numeric: dict[int, tuple[Fraction, bool]] = {}

    def start(self, initial_h: float) -> int:
        """(Re)initializes partition bookkeeping from the initial state's
        h^add value. Must be called exactly once, before any `eval()` call.
        Returns the root's (clamped) partition id; the caller is responsible
        for seeding the tables with an explicit `eval()` call on the initial
        state and then pushing the root with novelty hard-coded to 1,
        regardless of that call's return value (see `novbfs_search`)."""
        assert initial_h >= 0, (
            "initial_h must be non-negative (novbfs always uses h^add, which "
            "never returns a negative value for a reachable state)"
        )
        self._partitions = {}
        self._max_partition = max(1, math.floor(initial_h))
        self._parent_prop_true = {}
        self._parent_numeric = {}
        return self.partition_of(initial_h)

    def partition_of(self, h_value: float) -> int:
        """The partition function: `floor(h_value)`, clamped at the top to
        `max_partition`."""
        assert h_value >= 0, (
            "h_value must be non-negative (novbfs always uses h^add, which "
            "never returns a negative value for a reachable state)"
        )
        return min(math.floor(h_value), self._max_partition)

    def begin_expansion(self) -> None:
        """Resets the lazy parent-feature cache. Must be called once per
        expansion, before the first `eval()` call for that expansion's
        children -- not enforced here (the sole caller, `novbfs_search`,
        gets this right by construction)."""
        self._parent_prop_true = {}
        self._parent_numeric = {}

    def _get_partition(self, partition: int) -> _PartitionTables:
        tables = self._partitions.get(partition)
        if tables is None:
            tables = _PartitionTables()
            self._partitions[partition] = tables
        return tables

    def _sdist_and_sat(self, leaf_id: int, state: State) -> tuple[Fraction, bool]:
        """The distance feature for a numeric leaf -- already in the
        "improves = larger is better" sign convention (see module
        docstring): satisfied inequalities are >= 0, satisfied equalities
        are exactly 0, and both worsen (decrease) as the subgoal moves
        further from being satisfied -- paired with its satisfaction
        boolean, computed from the same `lhs`/`rhs` evaluation rather than
        a second, redundant `evaluate(leaf, state)` over both operands
        again. Satisfaction is *not* simply `sdist >= 0`: for strict `<`,
        the exact boundary (`sdist == 0`) is genuinely unsatisfied, so the
        comparator needs to stay kind-aware here too."""
        lhs, rhs, kind = self._numeric_leaves[leaf_id]
        diff = evaluate(rhs, state) - evaluate(lhs, state)  # type: ignore[operator]
        if kind == "==":
            return -abs(Fraction(diff)), diff == 0
        sdist = Fraction(diff)
        return sdist, (sdist >= 0 if kind == "<=" else sdist > 0)

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
        against the `partition` (`floor(h^add)`) bucket's own history so
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
        tables = self._get_partition(partition)

        newly_satisfied: list[int] = []
        currently_satisfied: list[int] = []
        persisting_satisfied: list[int] = []
        numeric_improved: list[int] = []
        numeric_unsatisfied: list[int] = []
        sdist_cache: dict[int, Fraction] = {}

        # Pass A: classify every subgoal relative to the parent (or, on a
        # fresh partition, relative to "nothing generated here yet" --
        # everything currently true/satisfied counts as newly added).
        for leaf_id, leaf in enumerate(self._leaves):
            if leaf_id not in self._numeric_leaves:
                # `is True`, not `bool(...)`: only a literal boolean true
                # satisfies a propositional leaf (matches Rust's `is_true`;
                # see the module docstring's implementation notes).
                s_true = evaluate(leaf, state) is True
                if s_true:
                    currently_satisfied.append(leaf_id)
                if new_partition:
                    if s_true:
                        newly_satisfied.append(leaf_id)
                else:
                    assert parent is not None
                    p_true = self._parent_prop_true.get(leaf_id)
                    if p_true is None:
                        p_true = evaluate(leaf, parent) is True
                        self._parent_prop_true[leaf_id] = p_true
                    if s_true and not p_true:
                        newly_satisfied.append(leaf_id)
                    elif s_true:
                        persisting_satisfied.append(leaf_id)
                continue

            sdist, curr_sat = self._sdist_and_sat(leaf_id, state)
            sdist_cache[leaf_id] = sdist
            if new_partition:
                if curr_sat:
                    currently_satisfied.append(leaf_id)
                    newly_satisfied.append(leaf_id)
                else:
                    numeric_unsatisfied.append(leaf_id)
                    numeric_improved.append(leaf_id)
            else:
                assert parent is not None
                cached = self._parent_numeric.get(leaf_id)
                if cached is None:
                    cached = self._sdist_and_sat(leaf_id, parent)
                    self._parent_numeric[leaf_id] = cached
                pdist, parent_sat = cached
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
                sdist = sdist_cache[tid]
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
