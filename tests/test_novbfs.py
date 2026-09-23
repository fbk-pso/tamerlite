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

import itertools
import warnings
from collections.abc import Callable
from typing import cast

import pytest
from unified_planning.engines import PlanGenerationResult
from unified_planning.engines import PlanGenerationResultStatus as ResultStatus
from unified_planning.shortcuts import AnytimePlanner, OneshotPlanner, PlanValidator

import problems_generator
import testing_utils
from tamerlite.core.search_space import ConstantNode

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


def _fresh_id_types():
    """`Fluent`/`Object` from the same fresh `search_space` module
    `_fresh_novelty_test_imports` just loaded -- call it right after that
    helper, for the same stale-class reason its docstring gives."""
    from tamerlite.core.search_space import Fluent, Object

    return Fluent, Object


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

        Fluent, _ = _fresh_id_types()
        F_AT_LOC1, F_LOADED, F_FUEL = Fluent(0), Fluent(1), Fluent(2)
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
        root_partition = novelty.start(0.0)
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

        Fluent, _ = _fresh_id_types()
        F_P, F_Q = Fluent(0), Fluent(1)
        p = _leaf(FluentNode(F_P))
        q = _leaf(FluentNode(F_Q))
        goal = _leaf(FluentNode(F_P), FluentNode(F_Q), OperatorNode("and", (0, 1)))
        fluent_domains = [FluentDomain(FluentKind.BOOL), FluentDomain(FluentKind.BOOL)]

        novelty = NumericNovelty({}, goal, fluent_domains)
        p_id = novelty._leaf_index[p]
        q_id = novelty._leaf_index[q]

        s0 = _state([False, False])
        partition = novelty.start(0.0)
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

        Fluent, _ = _fresh_id_types()
        F_P = Fluent(0)
        goal = _leaf(FluentNode(F_P))
        fluent_domains = [FluentDomain(FluentKind.BOOL)]

        novelty = NumericNovelty({}, goal, fluent_domains)
        partition = novelty.start(0.0)
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

        Fluent, Object = _fresh_id_types()
        F_NUM, F_OBJ, F_OBJ2 = Fluent(0), Fluent(1), Fluent(2)
        numeric_eq = _leaf(FluentNode(F_NUM), 3, OperatorNode("==", (0, 1)))
        object_eq = _leaf(
            FluentNode(F_OBJ), ObjectNode(Object(0)), OperatorNode("==", (0, 1))
        )
        fluent_vs_fluent_eq = _leaf(
            FluentNode(F_OBJ), FluentNode(F_OBJ2), OperatorNode("==", (0, 1))
        )
        goal = _leaf(
            FluentNode(F_NUM),
            3,
            OperatorNode("==", (0, 1)),
            FluentNode(F_OBJ),
            ObjectNode(Object(0)),
            OperatorNode("==", (3, 4)),
            FluentNode(F_OBJ),
            FluentNode(F_OBJ2),
            OperatorNode("==", (6, 7)),
            OperatorNode("and", (2, 5, 8)),
        )
        fluent_domains = [
            FluentDomain(FluentKind.INT),
            FluentDomain(FluentKind.OBJECT, (Object(0), Object(1))),
            FluentDomain(FluentKind.OBJECT, (Object(0), Object(1))),
        ]

        novelty = NumericNovelty({}, goal, fluent_domains)
        numeric_eq_id = novelty._leaf_index[numeric_eq]
        object_eq_id = novelty._leaf_index[object_eq]
        fluent_vs_fluent_eq_id = novelty._leaf_index[fluent_vs_fluent_eq]

        # Imported here, right after `_fresh_novelty_test_imports()`'s reload,
        # for the same stale-class reason given in its docstring.
        from tamerlite.core.novelty import _NumLeaf, _PropLeaf

        assert isinstance(novelty._leaves[numeric_eq_id], _NumLeaf)
        assert isinstance(novelty._leaves[object_eq_id], _PropLeaf)
        assert isinstance(novelty._leaves[fluent_vs_fluent_eq_id], _PropLeaf)


class TestNumericNoveltyBinaryPassesCrossBackend:
    """Backend-agnostic unit tests for Pass C1b (psi x delta) and Pass C2
    (persisting psi x delta) -- unlike `TestNumericNovelty` above, these
    import through `tamerlite.core` (the backend switch) rather than
    `tamerlite.core.novelty` directly, and assert only the returned novelty
    *class*, never internal tables, so they run against the Rust
    `NumericNovelty` too.

    Motivation: measuring what disabling each pass changes across the full
    end-to-end test set showed Pass C2 entirely inert (`expanded_states`/
    `goal_depth`/novelty-class histogram all unchanged) and Pass C1b's only
    visible effect an unasserted novelty-class histogram shift on one
    problem -- neither pass had any test that would fail if it silently
    broke. Both hand-traced chains below are constructed so the *only* pass
    that can produce novelty 2 on the final state is the one named; every
    other pass is independently shown silent at that point in the trace.

    States are built the same way as `test_engine.py`'s cross-backend
    `SearchSpace` construction (a bare `SearchSpace([], {}, [], None, None,
    None)` plus `initial_state(...)`), since the Rust `State` has no
    constructor of its own -- unlike `TestNumericNovelty` above, which
    constructs Python `State` objects directly and is therefore stuck on
    the Python core.
    """

    @staticmethod
    def _make_novelty_and_states(disable_rustamer, chain):
        """Builds a `NumericNovelty` over the fixed two-leaf catalogue
        (`p`: a bool fluent; `n`: `10 <= fuel`, so `sdist = fuel - 10`) and
        one `State` per `(p, fuel)` raw-value pair in `chain`, using
        whichever backend `disable_rustamer` selects. `reload_tamerlite`
        must run before any `make_*_node` call -- one node built against
        the backend active before this call and mixed into a state/goal
        built after it are different backends' incompatible node types, so
        every builder is imported fresh here, after the reload, rather than
        accepting already-built nodes from the caller."""
        reload_tamerlite(disable_rustamer)
        from tamerlite.core import (
            Fluent,
            NumericNovelty,
            SearchSpace,
            make_bool_constant_node,
            make_fluent_node,
            make_int_constant_node,
            make_operator_node,
        )
        from tamerlite.core.search_space import FluentDomain, FluentKind

        goal = (
            make_fluent_node(Fluent(0)),  # 0: p
            make_int_constant_node(10),  # 1
            make_fluent_node(Fluent(1)),  # 2: fuel
            make_operator_node("<=", (1, 2)),  # 3: n = 10 <= fuel
            make_operator_node("and", (0, 3)),  # 4: goal = p and n
        )
        fluent_domains = [FluentDomain(FluentKind.BOOL), FluentDomain(FluentKind.INT)]
        novelty = NumericNovelty({}, goal, fluent_domains)

        search_space = SearchSpace([], {}, [], None, None, None)
        states = [
            search_space.initial_state(
                cast(
                    "list[ConstantNode]",
                    [make_bool_constant_node(p), make_int_constant_node(fuel)],
                )
            )
            for p, fuel in chain
        ]
        return novelty, states

    @staticmethod
    def _run_chain(novelty, states):
        """Feeds `states` through `novelty` as a linear chain (state `i` is
        the child of state `i - 1`), one shared partition throughout,
        `begin_expansion()` before every `eval()` call as `novbfs_search`
        does -- returns the *last* state's novelty class."""
        partition = novelty.start(0.0)
        novelty.eval(states[0], partition, None, None)
        result = None
        for parent, child in itertools.pairwise(states):
            novelty.begin_expansion()
            result = novelty.eval(child, partition, parent, partition)
        return result

    @pytest.mark.parametrize("disable_rustamer", [True, False])
    def test_pass_c1b_only(self, disable_rustamer):
        """`p` becomes satisfied, is already in `psi_seen` by the final
        state (B1 silent there); `n`'s sdist ties its immediate parent's
        (B2 silent); `currently_satisfied == [p]` at the final state so C1
        only ever sees `f == tid` (silent). Only C1b -- `p` newly
        satisfied x `n` still unsatisfied, `sdist` improving from -10 to
        -2 -- can produce novelty 2."""
        chain = [
            (False, 0),  # root
            (True, 0),  # p: F -> T (first time)
            (False, 8),  # p: T -> F; fuel sdist -10 -> -2 (new, via C1b later)
            (True, 8),  # p: F -> T again (psi_seen already); fuel unchanged
        ]
        novelty, states = self._make_novelty_and_states(disable_rustamer, chain)
        assert self._run_chain(novelty, states) == 2

    @pytest.mark.parametrize("disable_rustamer", [True, False])
    def test_pass_c2_only(self, disable_rustamer):
        """`p` becomes satisfied and then persists; `n`'s best-ever sdist in
        this partition (-1, set while `p` was still false) already beats
        the final state's sdist (-2), so B2 is silent; `newly_satisfied` is
        empty at the final state, so C1/C1b (both keyed off it) cannot fire
        either. Only C2 -- `p` persisting-satisfied x `n` improved-over-its-
        immediate-parent but still unsatisfied -- can produce novelty 2."""
        chain = [
            (False, 0),  # root
            (False, 9),  # fuel sdist -10 -> -1 (new best)
            (False, 12),  # fuel satisfied for the first time
            (True, 12),  # p: F -> T (first time); fuel persists satisfied
            (True, 5),  # fuel regresses to unsatisfied; p persists
            (True, 8),  # fuel sdist -5 -> -2 (beats parent, not the -1 best)
        ]
        novelty, states = self._make_novelty_and_states(disable_rustamer, chain)
        assert self._run_chain(novelty, states) == 2


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
    """`SearchParams.__post_init__` rejects this combination at construction
    time -- both `search` and `incomplete_memory_bounded_search` are known
    there, so there's no need to wait until `_solve_ground_problem` has
    compiled and grounded the problem to reject it."""
    reload_tamerlite(True)
    from tamerlite.engine import SearchParams

    with pytest.raises(NotImplementedError):
        SearchParams(search="novbfs_hg", incomplete_memory_bounded_search=True)


@pytest.mark.parametrize("search_name", NOVBFS_SEARCHES)
@pytest.mark.parametrize(
    "make_problem",
    [
        problems_generator.get_problem_numeric,
        lambda: problems_generator.get_problem_logistics(1, 1, 4, 2),
    ],
    ids=["numeric", "logistics"],
)
def test_novbfs_metrics_regression(make_problem, search_name, data_regression):
    """Pins `expanded_states`/`goal_depth` for a couple of fixed small
    problems, on *both* backends against the same pinned YAML -- the primary
    check that the Rust core's `novbfs` mirrors the pure-Python one exactly.
    Also validates the plan itself (`test_novbfs_cross_backend_parity`
    doesn't share this problem set): a regression that preserves the metrics
    but silently breaks the plan would otherwise slip through.

    `logistics`: `fly_fast` has no
    `connected` precondition (`problems_generator.get_problem_flight`), so
    A->D is a single action and the pinned baseline was `expanded_states: 2`
    / `goal_depth: 1` -- "pop the root, then pop the goal", reproduced by
    almost any scoring function, so it could never detect drift in the
    novelty measure itself. `logistics(1, 1, 4, 2)` is the problem
    `TestNumericNoveltyBinaryPassesCrossBackend`'s motivation measured Pass
    C1b as actually changing the novelty-class histogram on."""
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


NOVBFS_ANYTIME_PROBLEMS = [
    (
        "flight_minimize_plan_length",
        problems_generator.get_problem_flight_minimize_plan_length,
    ),
    (
        "temporal_flight_minimize_makespan",
        problems_generator.get_problem_temporal_flight_minimize_makespan,
    ),
]


@pytest.mark.parametrize("disable_rustamer", [True, False])
@pytest.mark.parametrize("search_name", NOVBFS_SEARCHES)
@pytest.mark.parametrize(
    "make_problem",
    [p for _, p in NOVBFS_ANYTIME_PROBLEMS],
    ids=[n for n, _ in NOVBFS_ANYTIME_PROBLEMS],
)
def test_novbfs_anytime(make_problem, search_name, disable_rustamer):
    """`novbfs_search`'s docstring documents that `TamerLite._anytime_solutions`
    gives each cold-restart iteration a fresh `NumericNovelty` (and so a
    fresh `start()`). Nothing else in the suite drives novbfs through
    `AnytimePlanner` -- `test_engine.py::test_anytime_planner` hard-codes
    `search_kind = "wastar"` and `_anytime_cases()` has no search axis, and
    adding one there would multiply an already large problem x weak x
    symmetry x backend matrix. Covers one classical and one temporal
    metric-carrying problem instead, both search names, both backends.

    Trimmed from `test_anytime_planner`'s assertions: both
    `MinimizeSequentialPlanLength` and `MinimizeMakespan` are minimization
    metrics, so the direction is hard-coded here rather than re-derived."""
    reload_tamerlite(disable_rustamer)
    from tamerlite.engine import SearchParams

    problem = make_problem()
    search = SearchParams(search=search_name, compression_safe_actions=False)
    prev_metric_value = None
    count = 0
    with AnytimePlanner(name="tamerlite", params={"search": search}) as planner:
        for res in planner.get_solutions(problem, timeout=60):
            count += 1
            assert res.status in {
                ResultStatus.INTERMEDIATE,
                ResultStatus.SOLVED_SATISFICING,
                ResultStatus.SOLVED_OPTIMALLY,
            }
            with PlanValidator(problem_kind=problem.kind) as v:
                val_res = v.validate(problem, res.plan)
                assert val_res
                assert (
                    val_res.metric_evaluations is not None
                    and len(val_res.metric_evaluations) == 1
                )
                metric_value = next(iter(val_res.metric_evaluations.values()))
            if prev_metric_value is not None:
                if res.status == ResultStatus.INTERMEDIATE:
                    assert metric_value < prev_metric_value
                else:
                    assert metric_value <= prev_metric_value
            prev_metric_value = metric_value
            if count == 4:
                break
    # >= 2: at least one cold restart happened (the initial solve plus a
    # tightened re-solve), so the "fresh NumericNovelty per cold restart"
    # contract this test exists for is actually exercised, not just the
    # first, single-instance solve.
    assert count >= 2
