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
"""Tests for the numeric-novelty search (`search="novbfs_hg"`/`"novbfs_lg"`,
`tamerlite.core.novelty.NumericNovelty` + `tamerlite.core.search.novbfs_search`
in the pure-Python core; `crates/rustamer-base/src/novelty.rs` +
`search.rs::novbfs_search` in the Rust core).

Most end-to-end tests here force `DISABLE_RUSTAMER=1` via `reload_tamerlite`
(`testing_utils`'s helper) purely to keep a single, fixed backend for tests
that aren't about backend parity itself; `test_novbfs_cross_backend_parity`
and `test_novbfs_metrics_regression` are the exception -- they loop both
backends and assert identical `expanded_states`/`goal_depth`
(`testing_utils.check_metrics_equality`), same convention as
`test_engine.py`'s search-algorithm matrix.
"""

import warnings
from collections.abc import Callable

import pytest
from unified_planning.engines import PlanGenerationResult
from unified_planning.engines import PlanGenerationResultStatus as ResultStatus
from unified_planning.shortcuts import OneshotPlanner, PlanValidator

import problems_generator
import testing_utils

reload_tamerlite = testing_utils.reload_tamerlite
check_metrics_equality = testing_utils.check_metrics_equality


def _leaf(*nodes):
    """Builds a self-contained `Expression` from already-built leaf/operator
    nodes appended in order (root = last)."""
    return tuple(nodes)


def _fresh_novelty_test_imports():
    """Imports every `tamerlite.core` name `TestNumericNovelty` needs, fresh,
    right before use. Never hold onto a module-level-imported reference to
    one of these classes: `reload_tamerlite` -- called throughout this file,
    including by tests below this class -- reloads every `tamerlite`
    submodule unconditionally (see `testing_utils.reload_package`'s
    docstring), which replaces `NumericNovelty`/`FluentNode`/`OperatorNode`/
    `State`/`MultiSet`/`ObjectNode` with fresh, distinct class objects; a
    stale reference (e.g. a module-level import) silently stops comparing
    equal/hashing the same as the ones `NumericNovelty` itself uses
    internally the moment any test anywhere in this process reloads."""
    reload_tamerlite(True)  # arbitrary but fixed -- novelty.py doesn't
    # dispatch through the backend switch either way; this call exists to
    # force a fresh set of class objects, not to pick a backend.
    from tamerlite.core.novelty import NumericNovelty
    from tamerlite.core.search_space import (
        FluentDomain,
        FluentKind,
        FluentNode,
        MultiSet,
        ObjectNode,
        OperatorNode,
        State,
    )

    def state(assignments, g=0):
        return State(list(assignments), None, {}, MultiSet(), g, [])

    return (
        NumericNovelty,
        FluentNode,
        OperatorNode,
        ObjectNode,
        FluentDomain,
        FluentKind,
        state,
    )


class TestNumericNovelty:
    """Direct, search-independent unit tests for `NumericNovelty.eval`,
    hand-tracing a small worked example and isolating the novelty=2
    (binary-only) and novelty=3 (nothing novel) cases."""

    def test_worked_example_novelty_one_for_different_reasons(self):
        """Two boolean subgoals `at(loc1)`, `loaded`, one numeric subgoal
        `fuel >= 10` (here: `10 <= fuel`, so `sdist = fuel - 10`). All four
        states are novelty 1, each for a different reason -- this pins the
        *reason* (via the internal tables), not just the return value, which
        is uninformatively 1 throughout."""
        (
            NumericNovelty,
            FluentNode,
            OperatorNode,
            _,
            FluentDomain,
            FluentKind,
            _state,
        ) = _fresh_novelty_test_imports()

        F_AT_LOC1, F_LOADED, F_FUEL = 0, 1, 2
        at_loc1 = _leaf(FluentNode(F_AT_LOC1))
        fuel_leq = _leaf(10, FluentNode(F_FUEL), OperatorNode("<=", (0, 1)))
        goal = _leaf(
            FluentNode(F_AT_LOC1),
            FluentNode(F_LOADED),
            10,
            FluentNode(F_FUEL),
            OperatorNode("<=", (2, 3)),
            OperatorNode("and", (0, 1, 4)),
        )
        fluent_domains = [
            FluentDomain(FluentKind.BOOL),
            FluentDomain(FluentKind.BOOL),
            FluentDomain(FluentKind.REAL),
        ]

        novelty = NumericNovelty({}, goal, fluent_domains)
        fuel_id = novelty._leaf_index[fuel_leq]
        at_loc1_id = novelty._leaf_index[at_loc1]

        # s0 (root): fuel=4, at_loc1=False, loaded=False.
        s0 = _state([False, False, 4])
        root_partition = novelty.start(s0, 0.0)
        assert novelty.eval(s0, root_partition, None, None) == 1
        tables0 = novelty._get_partition(root_partition)
        assert tables0.best_sdist[fuel_id] == 4 - 10
        assert tables0.psi_seen == set()  # neither bool ever true yet

        # Each state below is a *linear* chain -- child of exactly the
        # previous one, no siblings -- so `begin_expansion()` (which resets
        # the parent-feature cache `eval` fills lazily per expansion; see
        # `NumericNovelty`'s class docstring) must be called before every
        # one of these `eval()` calls, exactly as `novbfs_search` calls it
        # once per popped state before scoring that state's children.

        # s1 (child of s0): fuel=8 -- unsatisfied but strictly closer.
        s1 = _state([False, False, 8], g=1)
        novelty.begin_expansion()
        assert novelty.eval(s1, root_partition, s0, root_partition) == 1
        assert tables0.best_sdist[fuel_id] == 8 - 10

        # s2 (child of s1): at_loc1 becomes true; fuel unchanged (8, tied
        # with s1 -- must NOT count as an improvement, strict `>` only).
        s2 = _state([True, False, 8], g=2)
        novelty.begin_expansion()
        assert novelty.eval(s2, root_partition, s1, root_partition) == 1
        assert at_loc1_id in tables0.psi_seen
        assert tables0.best_sdist[fuel_id] == 8 - 10  # unchanged

        # s3 (child of s2): fuel=12 -- now satisfied for the first time.
        s3 = _state([True, False, 12], g=3)
        novelty.begin_expansion()
        assert novelty.eval(s3, root_partition, s2, root_partition) == 1
        assert fuel_id in tables0.psi_seen  # newly-achieved psi, not delta
        assert tables0.best_sdist[fuel_id] == 8 - 10  # B2 untouched by this call

        # s4 (child of s3): nothing changes at all -- nothing left to be
        # novel about in this partition.
        s4 = _state([True, False, 12], g=4)
        novelty.begin_expansion()
        assert novelty.eval(s4, root_partition, s3, root_partition) == 3

    def test_novelty_2_is_a_pure_pair_of_already_seen_features(self):
        """Isolates the size-2-only path (C1, psi x psi): `p` oscillates
        false/true/false/true (so its *own* psi was already seen true, B1
        does not fire again) while `q` stays true from when it first
        appeared -- the first state where both are simultaneously true is
        novel only because the *pair* `(p, q)` has never been jointly true
        before."""
        (
            NumericNovelty,
            FluentNode,
            OperatorNode,
            _,
            FluentDomain,
            FluentKind,
            _state,
        ) = _fresh_novelty_test_imports()

        F_P, F_Q = 0, 1
        p = _leaf(FluentNode(F_P))
        q = _leaf(FluentNode(F_Q))
        goal = _leaf(FluentNode(F_P), FluentNode(F_Q), OperatorNode("and", (0, 1)))
        fluent_domains = [FluentDomain(FluentKind.BOOL), FluentDomain(FluentKind.BOOL)]

        novelty = NumericNovelty({}, goal, fluent_domains)
        p_id = novelty._leaf_index[p]
        q_id = novelty._leaf_index[q]

        s0 = _state([False, False])
        partition = novelty.start(s0, 0.0)
        assert novelty.eval(s0, partition, None, None) == 3  # nothing true yet

        # Linear chain, one child per state -- see the previous test's note
        # on `begin_expansion()`.
        s1 = _state([True, False], g=1)  # p: F -> T (first time)
        novelty.begin_expansion()
        assert novelty.eval(s1, partition, s0, partition) == 1

        s2 = _state([False, True], g=2)  # p: T -> F; q: F -> T (first time)
        novelty.begin_expansion()
        assert novelty.eval(s2, partition, s1, partition) == 1

        s3 = _state([True, True], g=3)  # p: F -> T again; q stays T
        tables = novelty._get_partition(partition)
        assert (min(p_id, q_id), max(p_id, q_id)) not in tables.psi_pair_seen
        novelty.begin_expansion()
        assert novelty.eval(s3, partition, s2, partition) == 2
        assert (min(p_id, q_id), max(p_id, q_id)) in tables.psi_pair_seen

        # And now that the pair has been seen, repeating it is not novel.
        s4 = _state([False, False], g=4)
        s5 = _state([True, True], g=5)
        novelty.begin_expansion()
        novelty.eval(s4, partition, s3, partition)
        novelty.begin_expansion()
        assert novelty.eval(s5, partition, s4, partition) == 3

    def test_begin_expansion_prevents_stale_parent_features(self):
        """A regression guard for `begin_expansion()`'s cache reset: two
        consecutive `eval()` calls with *different* actual parents
        (`parent_a` has `p` true, `parent_b` has `p` false) must each read
        *that* call's own parent's features. If `begin_expansion()` ever
        stopped resetting the lazy parent-feature cache (see
        `NumericNovelty`'s class docstring), the second call below would
        silently reuse `parent_a`'s cached (stale) `p_true=True` instead of
        `parent_b`'s actual `p_true=False`, and wrongly miss `p` becoming
        newly satisfied -- returning 3 instead of 1."""
        (
            NumericNovelty,
            FluentNode,
            _,
            _,
            FluentDomain,
            FluentKind,
            _state,
        ) = _fresh_novelty_test_imports()

        F_P = 0
        goal = _leaf(FluentNode(F_P))
        fluent_domains = [FluentDomain(FluentKind.BOOL)]

        novelty = NumericNovelty({}, goal, fluent_domains)
        partition = novelty.start(_state([True]), 0.0)
        parent_a = _state([True])
        parent_b = _state([False])

        # First call: parent_a (p=True) -- cache fills p_true=True.
        novelty.begin_expansion()
        child_x = _state([True], g=1)
        assert novelty.eval(child_x, partition, parent_a, partition) == 3

        # Second call: a genuinely different parent, parent_b (p=False).
        novelty.begin_expansion()
        child_y = _state([True], g=1)
        assert novelty.eval(child_y, partition, parent_b, partition) == 1

    def test_equality_leaf_classified_by_fluent_domain(self):
        """`==` between two numeric operands gets a distance feature; `==`
        between two object-typed operands does not (falls back to
        propositional-only) -- classified statically from each operand's
        `FluentDomain` at construction time, via
        `search_space.is_object_typed_operand` -- the same classifier
        `DeleteRelaxationHeuristic` uses (see module docstring), not a
        runtime probe. Covers a literal object operand
        (`fluent == object`) and a fluent-vs-fluent object equality
        (`fluent1 == fluent2`, no literal `ObjectNode` anywhere)."""
        (
            NumericNovelty,
            FluentNode,
            OperatorNode,
            ObjectNode,
            FluentDomain,
            FluentKind,
            _,
        ) = _fresh_novelty_test_imports()

        F_NUM, F_OBJ, F_OBJ2 = 0, 1, 2
        numeric_eq = _leaf(FluentNode(F_NUM), 3, OperatorNode("==", (0, 1)))
        object_eq = _leaf(FluentNode(F_OBJ), ObjectNode(0), OperatorNode("==", (0, 1)))
        fluent_vs_fluent_eq = _leaf(
            FluentNode(F_OBJ), FluentNode(F_OBJ2), OperatorNode("==", (0, 1))
        )
        goal = _leaf(
            FluentNode(F_NUM),
            3,
            OperatorNode("==", (0, 1)),
            FluentNode(F_OBJ),
            ObjectNode(0),
            OperatorNode("==", (3, 4)),
            FluentNode(F_OBJ),
            FluentNode(F_OBJ2),
            OperatorNode("==", (6, 7)),
            OperatorNode("and", (2, 5, 8)),
        )
        fluent_domains = [
            FluentDomain(FluentKind.INT),
            FluentDomain(FluentKind.OBJECT, (0, 1)),
            FluentDomain(FluentKind.OBJECT, (0, 1)),
        ]

        novelty = NumericNovelty({}, goal, fluent_domains)
        numeric_eq_id = novelty._leaf_index[numeric_eq]
        object_eq_id = novelty._leaf_index[object_eq]
        fluent_vs_fluent_eq_id = novelty._leaf_index[fluent_vs_fluent_eq]

        assert numeric_eq_id in novelty._numeric_leaves
        assert object_eq_id not in novelty._numeric_leaves
        assert fluent_vs_fluent_eq_id not in novelty._numeric_leaves


NOVBFS_SEARCHES = ["novbfs_hg", "novbfs_lg"]

# Kept small and independent of `up_test_cases`/`PYTHONPATH=up-checkout/...`
# so this file runs standalone -- see `test_engine.py::PROBLEMS` for the
# full cross-product used elsewhere.
NOVBFS_PROBLEMS: list[tuple[str, Callable[[], object]]] = [
    ("numeric", problems_generator.get_problem_numeric),
    ("satellite", problems_generator.get_problem_satellite),
    ("logistics", lambda: problems_generator.get_problem_logistics(1, 1, 4, 2)),
    ("temporal_flight", problems_generator.get_problem_temporal_flight),
]


@pytest.mark.parametrize("search_name", NOVBFS_SEARCHES)
@pytest.mark.parametrize(
    "make_problem", [p for _, p in NOVBFS_PROBLEMS], ids=[n for n, _ in NOVBFS_PROBLEMS]
)
def test_novbfs_solves_and_validates(make_problem, search_name):
    reload_tamerlite(True)
    from tamerlite.engine import SearchParams

    problem = make_problem()
    with OneshotPlanner(
        name="tamerlite", params={"search": SearchParams(search=search_name)}
    ) as planner:
        res = planner.solve(problem, timeout=60)
        assert res.status == ResultStatus.SOLVED_SATISFICING
        with PlanValidator(problem_kind=problem.kind) as validator:
            assert validator.validate(problem, res.plan)


@pytest.mark.parametrize("search_name", NOVBFS_SEARCHES)
@pytest.mark.parametrize(
    "make_problem", [p for _, p in NOVBFS_PROBLEMS], ids=[n for n, _ in NOVBFS_PROBLEMS]
)
def test_novbfs_cross_backend_parity(make_problem, search_name):
    """Both cores must expand exactly the same states -- see
    `testing_utils.check_metrics_equality`, the same check every other
    search in `test_engine.py` is held to."""
    problem = make_problem()
    results = []
    for disable_rustamer in [True, False]:
        reload_tamerlite(disable_rustamer)
        from tamerlite.engine import SearchParams

        with OneshotPlanner(
            name="tamerlite", params={"search": SearchParams(search=search_name)}
        ) as planner:
            res: PlanGenerationResult = planner.solve(problem, timeout=60)
            assert res.status == ResultStatus.SOLVED_SATISFICING
            results.append(res)
            with PlanValidator(problem_kind=problem.kind) as validator:
                assert validator.validate(problem, res.plan)
    check_metrics_equality(results)


def test_novbfs_ignores_configured_heuristic_with_warning():
    reload_tamerlite(True)
    from tamerlite.engine import SearchParams

    problem = problems_generator.get_problem_numeric()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with OneshotPlanner(
            name="tamerlite",
            params={"search": SearchParams(search="novbfs_hg", heuristic="hff")},
        ) as planner:
            res = planner.solve(problem, timeout=30)
    assert res.status == ResultStatus.SOLVED_SATISFICING
    assert any("always use an internal" in str(w.message) for w in caught), (
        "expected a warning that the configured heuristic is ignored"
    )


def test_novbfs_ignores_custom_heuristic_callable_with_warning():
    """A custom heuristic callable passed to `solve()` (as opposed to a
    `SearchParams.heuristic` string) must be flagged too -- it is just as
    silently dropped by novbfs_hg/novbfs_lg."""
    reload_tamerlite(True)
    from tamerlite.engine import SearchParams

    problem = problems_generator.get_problem_numeric()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with OneshotPlanner(
            name="tamerlite", params={"search": SearchParams(search="novbfs_lg")}
        ) as planner:
            res = planner.solve(problem, heuristic=lambda state: 0.0, timeout=30)
    assert res.status == ResultStatus.SOLVED_SATISFICING
    assert any("always use an internal" in str(w.message) for w in caught), (
        "expected a warning that the custom heuristic callable is ignored"
    )


def test_novbfs_rejects_memory_bounded():
    reload_tamerlite(True)
    from tamerlite.engine import SearchParams

    problem = problems_generator.get_problem_numeric()
    params = SearchParams(search="novbfs_hg", incomplete_memory_bounded_search=True)
    with (
        OneshotPlanner(name="tamerlite", params={"search": params}) as planner,
        pytest.raises(NotImplementedError),
    ):
        planner.solve(problem, timeout=10)


@pytest.mark.parametrize("search_name", NOVBFS_SEARCHES)
@pytest.mark.parametrize(
    "make_problem",
    [problems_generator.get_problem_numeric, problems_generator.get_problem_flight],
    ids=["numeric", "flight"],
)
def test_novbfs_metrics_regression(make_problem, search_name, data_regression):
    """Pins `expanded_states`/`goal_depth` for a couple of fixed small
    problems, on *both* backends against the same pinned YAML -- the primary
    check that the Rust core's `novbfs` mirrors the pure-Python one exactly.
    Also validates the plan itself (`test_novbfs_cross_backend_parity`
    doesn't share this problem set): a regression that preserves the metrics
    but silently breaks the plan would otherwise slip through."""
    problem = make_problem()
    metrics = None
    for disable_rustamer in [True, False]:
        reload_tamerlite(disable_rustamer)
        from tamerlite.engine import SearchParams

        with OneshotPlanner(
            name="tamerlite", params={"search": SearchParams(search=search_name)}
        ) as planner:
            res = planner.solve(problem, timeout=60)
            assert res.status == ResultStatus.SOLVED_SATISFICING
            with PlanValidator(problem_kind=problem.kind) as validator:
                assert validator.validate(problem, res.plan)
        if metrics is None:
            metrics = dict(res.metrics)
        else:
            assert dict(res.metrics) == metrics
    data_regression.check(metrics)
