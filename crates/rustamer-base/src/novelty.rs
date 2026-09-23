// Copyright (C) 2025 PSO Unit, Fondazione Bruno Kessler
// This file is part of TamerLite.
//
// TamerLite is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// TamerLite is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.
//

//! Rust mirror of `tamerlite.core.novelty.NumericNovelty`
//! (`src/tamerlite/core/novelty.py`); see that module's docstring for the
//! algorithm. This file only notes where the Rust port differs in
//! *implementation* (never in outcome).
//!
//! Leaf numbering need not match the Python core's: the algorithm's result
//! doesn't depend on leaf-id order (Pass A classifies each leaf
//! independently; Pass B/C are order-independent set/map membership), so
//! leaves are catalogued in whatever order `events`' `FxHashMap` iterates,
//! not Python's dict-insertion order.
//!
//! The distance feature (`compute_sdist`) is kept exact, never `f64`, to
//! match Python's exact `int`/`Fraction` arithmetic -- see `Sdist`.

use num::{BigInt, Signed, ToPrimitive, Zero};
use num_rational::BigRational;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use std::borrow::Cow;
use std::cmp::Ordering;
use std::collections::hash_map::Entry;

use super::expressions::{ExpressionNode, PyExpressionNode};
use super::expressions_utils::{
    as_num_ref, internal_evaluate, split_expression, FluentValueTrait, NumRef,
};
use super::heuristics::{
    extract_fluent_domains, extract_sub_expression, is_object_typed, FluentDomain,
};
use super::search_state::State;
use super::structures::{Action, Event, Fluent, Timing};
use super::utils::ArithmeticError;

fn arith_err(e: ArithmeticError) -> PyErr {
    PyException::new_err(format!("{:?}", e))
}

/// An event's own conditions, deduplicated by structural equality.
fn event_conditions(event: &Event) -> PyResult<Vec<Vec<ExpressionNode>>> {
    let mut seen: FxHashSet<Vec<ExpressionNode>> = FxHashSet::default();
    let mut conditions = Vec::new();
    for condition in split_expression(&event.conditions)?
        .into_iter()
        .chain(event.end_conditions.clone())
    {
        if seen.insert(condition.clone()) {
            conditions.push(condition);
        }
    }
    Ok(conditions)
}

/// Flattens `exp` through nested `and`/`or` down to its leaf
/// subexpressions ("subgoals"), lazily. `and`/`or` operands are indices into
/// `exp`, so the walk stays in `exp`'s own index space and only calls
/// `extract_sub_expression` once a genuine leaf is reached.
fn iter_subgoal_leaves(
    exp: &[ExpressionNode],
) -> impl Iterator<Item = PyResult<Vec<ExpressionNode>>> + '_ {
    let mut stack = vec![exp.len() - 1];
    std::iter::from_fn(move || {
        while let Some(idx) = stack.pop() {
            match &exp[idx] {
                ExpressionNode::And(operands) | ExpressionNode::Or(operands) => {
                    // Push in reverse so operands are still popped (and thus
                    // yielded) in their original left-to-right order.
                    stack.extend(operands.iter().rev().copied());
                }
                _ => {
                    return Some(extract_sub_expression(exp, idx).map_err(arith_err));
                }
            }
        }
        None
    })
}

/// A leaf's evaluation shape, precomputed once at construction rather than
/// re-walked as a generic `Vec<ExpressionNode>` on every `eval` -- most real
/// conditions are a bare fluent or a fluent-vs-constant comparison, so this
/// skips `internal_evaluate`'s per-call work for the overwhelmingly common
/// case.
enum Operand {
    Fluent(Fluent),
    Const(ExpressionNode),
    Expr(Vec<ExpressionNode>),
}

impl Operand {
    fn from_vec(exp: Vec<ExpressionNode>) -> Operand {
        if exp.len() == 1 {
            match &exp[0] {
                ExpressionNode::Fluent(f) => return Operand::Fluent(*f),
                ExpressionNode::Bool(_)
                | ExpressionNode::Int(_)
                | ExpressionNode::Rational(_)
                | ExpressionNode::Object(_) => {
                    let mut exp = exp;
                    return Operand::Const(exp.pop().unwrap());
                }
                _ => {}
            }
        }
        Operand::Expr(exp)
    }

    fn eval<'a>(&'a self, state: &'a State) -> PyResult<Cow<'a, ExpressionNode>> {
        match self {
            Operand::Fluent(f) => Ok(Cow::Borrowed(state.get_value(*f))),
            Operand::Const(v) => Ok(Cow::Borrowed(v)),
            Operand::Expr(exp) => Ok(Cow::Owned(internal_evaluate(exp, state)?)),
        }
    }
}

fn is_true(v: &ExpressionNode) -> bool {
    matches!(v, ExpressionNode::Bool(true))
}

#[derive(Clone, Copy)]
enum NumKind {
    Le,
    Lt,
    Eq,
}

/// A leaf's distance feature, kept exact -- never rounded -- while avoiding
/// `BigRational` allocation for the overwhelmingly common case of two
/// small-int operands (see `compute_sdist`). `Small` holds the *exact*
/// integer difference (an i64-i64 subtraction always fits `i128`, so this
/// never overflows); `Big` is the arbitrary-precision fallback for anything
/// involving a `Rational` operand or an integer outside i64 range.
#[derive(Clone)]
enum Sdist {
    Small(i128),
    Big(Box<BigRational>),
}

impl Sdist {
    fn zero() -> Sdist {
        Sdist::Small(0)
    }
}

impl PartialEq for Sdist {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Sdist {}

impl PartialOrd for Sdist {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Sdist {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (Sdist::Small(a), Sdist::Small(b)) => a.cmp(b),
            (Sdist::Big(a), Sdist::Big(b)) => a.as_ref().cmp(b.as_ref()),
            (Sdist::Small(a), Sdist::Big(b)) => (BigInt::from(*a) * b.denom()).cmp(b.numer()),
            (Sdist::Big(a), Sdist::Small(b)) => a.numer().cmp(&(BigInt::from(*b) * a.denom())),
        }
    }
}

fn sdist_small(diff: i128, kind: NumKind) -> (Sdist, bool) {
    match kind {
        NumKind::Eq => (Sdist::Small(-diff.abs()), diff == 0),
        NumKind::Le => (Sdist::Small(diff), diff >= 0),
        NumKind::Lt => (Sdist::Small(diff), diff > 0),
    }
}

fn sdist_big(diff: BigRational, kind: NumKind) -> (Sdist, bool) {
    match kind {
        NumKind::Eq => {
            let sat = diff.is_zero();
            (Sdist::Big(Box::new(-diff.abs())), sat)
        }
        NumKind::Le => {
            let sat = !diff.is_negative();
            (Sdist::Big(Box::new(diff)), sat)
        }
        NumKind::Lt => {
            let sat = diff.is_positive();
            (Sdist::Big(Box::new(diff)), sat)
        }
    }
}

fn num_ref_to_bigrational(n: NumRef) -> BigRational {
    match n {
        NumRef::Int(v) => BigRational::from_integer(v.clone()),
        NumRef::Rational(v) => v.clone(),
    }
}

/// The distance feature paired with satisfaction, in the "improves = larger
/// is better" sign convention -- mirrors `novelty.py::_sdist_and_sat`
/// exactly: satisfied inequalities are `>= 0`, satisfied equalities are
/// exactly `0` (via `-abs(diff)`), and satisfaction for strict `<` is *not*
/// simply `sdist >= 0` (the exact boundary is genuinely unsatisfied).
///
/// Fast path: when both operands are plain integers that fit `i64` --
/// overwhelmingly the common case for a real numeric condition -- the exact
/// difference is a register-width `i128` subtraction with no heap
/// allocation at all (`as_num_ref` borrows rather than clones). Anything
/// else (a `Rational` operand, or an integer literal outside i64 range)
/// falls back to the exact arbitrary-precision path, unchanged in substance
/// from before this type existed.
fn compute_sdist(
    lhs: &Operand,
    rhs: &Operand,
    kind: NumKind,
    state: &State,
) -> PyResult<(Sdist, bool)> {
    let lhs_val = lhs.eval(state)?;
    let rhs_val = rhs.eval(state)?;
    let lhs_ref = as_num_ref(&lhs_val)?;
    let rhs_ref = as_num_ref(&rhs_val)?;
    if let (NumRef::Int(l), NumRef::Int(r)) = (lhs_ref, rhs_ref) {
        if let (Some(l), Some(r)) = (l.to_i64(), r.to_i64()) {
            let diff = r as i128 - l as i128;
            return Ok(sdist_small(diff, kind));
        }
    }
    let diff = num_ref_to_bigrational(rhs_ref) - num_ref_to_bigrational(lhs_ref);
    Ok(sdist_big(diff, kind))
}

enum LeafData {
    Prop {
        operand: Operand,
    },
    Num {
        lhs: Operand,
        rhs: Operand,
        kind: NumKind,
    },
}

fn make_num_leaf(
    exp: &[ExpressionNode],
    op1: usize,
    op2: usize,
    kind: NumKind,
) -> PyResult<LeafData> {
    let lhs = extract_sub_expression(exp, op1).map_err(arith_err)?;
    let rhs = extract_sub_expression(exp, op2).map_err(arith_err)?;
    Ok(LeafData::Num {
        lhs: Operand::from_vec(lhs),
        rhs: Operand::from_vec(rhs),
        kind,
    })
}

/// Adds `exp` to the leaf catalogue, deduplicated by structural equality
/// (mirrors `novelty.py::add_leaf`). Numeric classification is static for
/// `LE`/`LT` (always numeric -- only numbers support ordering); an `Equals`
/// leaf is numeric iff **neither** operand is object-typed
/// (`is_object_typed`, `heuristics.rs`) -- `Equals` covers both numeric and
/// object equality, so the operands' `FluentDomain`s are what tells them apart,
/// exactly the rule `DeleteRelaxationHeuristic`/`is_numeric_leaf_expression` use.
fn add_leaf(
    exp: Vec<ExpressionNode>,
    leaves: &mut Vec<LeafData>,
    leaf_index: &mut FxHashMap<Vec<ExpressionNode>, u32>,
    fluent_domains: &[FluentDomain],
    leaf_to_fluents: &mut Vec<Vec<u32>>,
) -> PyResult<()> {
    if leaf_index.contains_key(&exp) {
        return Ok(());
    }
    let leaf_id = leaves.len() as u32;
    leaf_index.insert(exp.clone(), leaf_id);
    leaf_to_fluents.push(
        exp.iter()
            .filter_map(|n| match n {
                ExpressionNode::Fluent(f) => Some(*f as u32),
                _ => None,
            })
            .collect(),
    );
    let data = match exp.last().expect("leaf expression is non-empty") {
        ExpressionNode::LE(op1, op2) => make_num_leaf(&exp, *op1, *op2, NumKind::Le)?,
        ExpressionNode::LT(op1, op2) => make_num_leaf(&exp, *op1, *op2, NumKind::Lt)?,
        ExpressionNode::Equals(op1, op2)
            if !is_object_typed(&exp[*op1], fluent_domains)
                && !is_object_typed(&exp[*op2], fluent_domains) =>
        {
            make_num_leaf(&exp, *op1, *op2, NumKind::Eq)?
        }
        _ => LeafData::Prop {
            operand: Operand::from_vec(exp),
        },
    };
    leaves.push(data);
    Ok(())
}

/// Persistent novelty-tracking tables for one `floor(h^add)` partition --
/// mirrors `novelty.py::_PartitionTables`. Sparse (never dense arrays)
/// deliberately: the pair tables are up to O(#leaves^2) per partition, and
/// most leaf pairs are never jointly touched.
#[derive(Default)]
struct PartitionTables {
    /// subgoal (leaf id) satisfied at least once in this partition.
    psi_seen: FxHashSet<u32>,
    /// best (max) sdist ever seen for a numeric subgoal in this partition.
    best_sdist: FxHashMap<u32, Sdist>,
    /// unordered pairs of subgoals jointly satisfied at least once, packed
    /// `(min << 32) | max`.
    psi_pair_seen: FxHashSet<u64>,
    /// best (max) sdist for a numeric subgoal conditioned on a
    /// propositional subgoal, keyed `(prop_leaf << 32) | numeric_leaf`
    /// (directed -- shared between Pass C1b and C2, which never race on the
    /// same key since `newly_satisfied`/`persisting_satisfied` are
    /// disjoint).
    best_sdist_with_psi: FxHashMap<u64, Sdist>,
}

fn pack_directed(a: u32, b: u32) -> u64 {
    ((a as u64) << 32) | (b as u64)
}

fn pack_pair(a: u32, b: u32) -> u64 {
    if a < b {
        pack_directed(a, b)
    } else {
        pack_directed(b, a)
    }
}

/// Per-`eval`-call classification of every leaf, reused (never reallocated)
/// across calls: `clear()` only resets the five list lengths to 0, keeping
/// capacity, and `sdist_cache` is never cleared at all -- see its field
/// doc.
struct LeafClassification {
    newly_satisfied: Vec<u32>,
    currently_satisfied: Vec<u32>,
    persisting_satisfied: Vec<u32>,
    numeric_improved: Vec<u32>,
    numeric_unsatisfied: Vec<u32>,
    /// Indexed by leaf id, sized to `n_leaves`. Fresh this call only for a
    /// leaf Pass A evaluated -- every leaf when `new_partition`, else only
    /// `dirty_list` -- so it's never cleared, just stale elsewhere.
    /// `numeric_improved` only ever holds fresh entries (a clean leaf can't
    /// be improved -- see `eval`), so Pass B and C2 index it directly;
    /// `numeric_unsatisfied` can hold a clean leaf too, so C1b guards its
    /// read (`new_partition`/`dirty_stamp`) and falls back to
    /// `parent_numeric` for one.
    sdist_cache: Vec<Sdist>,
}

impl LeafClassification {
    fn new(n_leaves: usize) -> LeafClassification {
        LeafClassification {
            newly_satisfied: Vec::new(),
            currently_satisfied: Vec::new(),
            persisting_satisfied: Vec::new(),
            numeric_improved: Vec::new(),
            numeric_unsatisfied: Vec::new(),
            sdist_cache: vec![Sdist::zero(); n_leaves],
        }
    }

    fn clear(&mut self) {
        self.newly_satisfied.clear();
        self.currently_satisfied.clear();
        self.persisting_satisfied.clear();
        self.numeric_improved.clear();
        self.numeric_unsatisfied.clear();
    }
}

/// Partitioned numeric novelty over the subgoals of `events`/`goals` --
/// Rust mirror of `tamerlite.core.novelty.NumericNovelty`. See the module
/// docstring for how this differs in implementation (never in outcome).
///
/// Construct once per search (subgoal catalogue only, no state needed),
/// call `start(initial_h)` once to fix the partition count, then call
/// `begin_expansion()` once per popped/expanded state followed by `eval()`
/// once per surviving successor, in generation order -- not thread-safe.
/// Reuse across multiple `novbfs_search` calls on the same instance (e.g.
/// the Python engine's `weak_equality` retry, which invokes the same bound
/// `partial` -- hence the same instance -- twice) is safe *because* `start`
/// fully resets `partitions`/`max_partition` and the parent-feature caches;
/// it must be called again, before any further `eval()`, whenever reused
/// this way.
#[pyclass]
pub struct NumericNovelty {
    leaves: Vec<LeafData>,
    partitions: FxHashMap<u64, PartitionTables>,
    max_partition: u64,
    classification: LeafClassification,
    /// Lazy per-leaf parent-feature cache (mirrors `novelty.py`'s
    /// `_parent_prop_true`/`_parent_numeric`). Valid iff stamp ==
    /// `generation`; `begin_expansion`/`start` invalidate everything in
    /// O(1) by bumping `generation`, instead of resetting `n_leaves` slots
    /// (which would also deallocate every cached `Sdist::Big`). Slots are
    /// overwritten in place, never freed, so a hit reads by reference, no
    /// clone.
    parent_prop_stamp: Vec<u64>,
    parent_prop_true: Vec<bool>,
    parent_numeric_stamp: Vec<u64>,
    parent_numeric: Vec<(Sdist, bool)>,
    generation: u64,
    /// `fluent_to_leaves[f]`: leaves reading fluent `f`. `dirty_stamp[leaf]
    /// == eval_generation` marks a leaf dirty for the child being scored;
    /// everything else is clean -- provably identical to the parent, so
    /// `eval` reuses the cached parent value instead of re-evaluating. See
    /// `mark_dirty_leaves`.
    fluent_to_leaves: Vec<Vec<u32>>,
    dirty_stamp: Vec<u64>,
    eval_generation: u64,
    /// Leaf ids marked dirty this `eval` call (built alongside
    /// `dirty_stamp`, refilled every call). `dirty_stamp` only makes a
    /// per-leaf check O(1); this makes the scan itself O(dirty) instead of
    /// O(n_leaves) -- Pass A iterates this, not `self.leaves`, on a
    /// same-partition call.
    dirty_list: Vec<u32>,
    /// Dense "satisfied in the parent" snapshot, valid iff
    /// `parent_snapshot_gen == generation`. Built lazily by
    /// `ensure_parent_snapshot`, at most once per expansion (only when a
    /// child's sparse Pass A finds progress -- see `eval`'s early exit),
    /// not once per child. `parent_sat_list`: every satisfied leaf;
    /// `parent_num_unsat_list`: every unsatisfied numeric leaf. Only
    /// meaningful for a clean leaf; `eval` filters out dirty ones when
    /// merging, since those already got a fresh value from Pass A.
    parent_sat_list: Vec<u32>,
    parent_num_unsat_list: Vec<u32>,
    parent_snapshot_gen: u64,
}

#[pymethods]
impl NumericNovelty {
    #[new]
    fn new(
        events: FxHashMap<Action, Vec<(Timing, Event)>>,
        goals: Vec<PyExpressionNode>,
        #[pyo3(from_py_with = extract_fluent_domains)] fluent_domains: Vec<FluentDomain>,
    ) -> PyResult<Self> {
        let goals: Vec<ExpressionNode> = goals.into_iter().map(|n| n.v).collect();
        let mut leaves: Vec<LeafData> = Vec::new();
        let mut leaf_index: FxHashMap<Vec<ExpressionNode>, u32> = FxHashMap::default();
        let mut leaf_to_fluents: Vec<Vec<u32>> = Vec::new();

        for event_list in events.values() {
            for (_, event) in event_list {
                for cond in event_conditions(event)? {
                    for leaf in iter_subgoal_leaves(&cond) {
                        add_leaf(
                            leaf?,
                            &mut leaves,
                            &mut leaf_index,
                            &fluent_domains,
                            &mut leaf_to_fluents,
                        )?;
                    }
                }
            }
        }
        for leaf in iter_subgoal_leaves(&goals) {
            add_leaf(
                leaf?,
                &mut leaves,
                &mut leaf_index,
                &fluent_domains,
                &mut leaf_to_fluents,
            )?;
        }

        let n = leaves.len();
        let mut fluent_to_leaves: Vec<Vec<u32>> = vec![Vec::new(); fluent_domains.len()];
        for (leaf_id, fluents) in leaf_to_fluents.iter().enumerate() {
            for &f in fluents {
                if let Some(bucket) = fluent_to_leaves.get_mut(f as usize) {
                    bucket.push(leaf_id as u32);
                }
            }
        }

        Ok(NumericNovelty {
            leaves,
            partitions: FxHashMap::default(),
            max_partition: 1,
            classification: LeafClassification::new(n),
            parent_prop_stamp: vec![0; n],
            parent_prop_true: vec![false; n],
            parent_numeric_stamp: vec![0; n],
            parent_numeric: vec![(Sdist::zero(), false); n],
            generation: 0,
            fluent_to_leaves,
            dirty_stamp: vec![0; n],
            eval_generation: 0,
            dirty_list: Vec::new(),
            parent_sat_list: Vec::new(),
            parent_num_unsat_list: Vec::new(),
            parent_snapshot_gen: 0,
        })
    }

    /// (Re)initializes partition bookkeeping from the initial state's
    /// h^add value and clears the parent-feature cache (so a following
    /// `eval` on the root sees "no parent" unconditionally, matching
    /// calling `eval(init, p, None, None)` directly). Must be called before
    /// any `eval()` call -- and again, before any further `eval()`, if this
    /// instance is being reused for another `novbfs_search` call (see the
    /// struct docstring). Returns the root's (clamped) partition id.
    pub fn start(&mut self, initial_h: f64) -> u64 {
        assert!(
            initial_h >= 0.0,
            "initial_h must be non-negative (novbfs always uses h^add, which \
             never returns a negative value for a reachable state)"
        );
        self.partitions.clear();
        self.max_partition = (initial_h.floor() as u64).max(1);
        self.generation += 1;
        self.partition_of(initial_h)
    }

    /// The partition function: `floor(h_value)`, clamped at the top to
    /// `max_partition`. `h_value` is always h^add (see `start`), which is
    /// never negative for a reachable state -- callers already filter out
    /// `None` (unreachable) before calling this, so `partition` is a
    /// genuine, always-non-negative index, never a signed quantity that
    /// happens to stay positive.
    pub fn partition_of(&self, h_value: f64) -> u64 {
        assert!(
            h_value >= 0.0,
            "h_value must be non-negative (novbfs always uses h^add, which \
             never returns a negative value for a reachable state)"
        );
        (h_value.floor() as u64).min(self.max_partition)
    }

    /// Resets the lazy parent-feature cache. Must be called once per
    /// expansion, before the first `eval()` call for that expansion's
    /// children -- not enforced here (the sole caller, `novbfs_search`,
    /// gets this right by construction).
    pub fn begin_expansion(&mut self) {
        self.generation += 1;
    }

    /// Has `state` made progress on something not seen before, relative to
    /// `parent` (the state that generated it)? See
    /// `novelty.py::NumericNovelty.eval`'s docstring for the full
    /// semantics -- this mirrors it exactly. `parent`/`parent_partition`
    /// are both `None` for the root.
    #[pyo3(signature = (state, partition, parent=None, parent_partition=None))]
    pub fn eval(
        &mut self,
        state: &State,
        partition: u64,
        parent: Option<&State>,
        parent_partition: Option<u64>,
    ) -> PyResult<u8> {
        let new_partition = parent.is_none() || parent_partition != Some(partition);

        self.classification.clear();

        if new_partition {
            // Dense: a partition seen for the first time by this child has
            // no parent to diff against, so every leaf is potentially
            // relevant and there is nothing to skip.
            for (idx, leaf) in self.leaves.iter().enumerate() {
                let leaf_id = idx as u32;
                match leaf {
                    LeafData::Prop { operand } => {
                        let s_val = operand.eval(state)?;
                        if is_true(&s_val) {
                            self.classification.currently_satisfied.push(leaf_id);
                            self.classification.newly_satisfied.push(leaf_id);
                        }
                    }
                    LeafData::Num { lhs, rhs, kind } => {
                        let (sdist, curr_sat) = compute_sdist(lhs, rhs, *kind, state)?;
                        if curr_sat {
                            self.classification.currently_satisfied.push(leaf_id);
                            self.classification.newly_satisfied.push(leaf_id);
                        } else {
                            self.classification.numeric_unsatisfied.push(leaf_id);
                            self.classification.numeric_improved.push(leaf_id);
                        }
                        self.classification.sdist_cache[idx] = sdist;
                    }
                }
            }
        } else {
            let parent_state = parent.expect("parent set when !new_partition");
            self.mark_dirty_leaves(parent_state, state);

            // Sparse Pass A: a leaf's value is a pure function of the
            // fluents it reads, so a leaf none of whose
            // fluents changed since the parent has the *exact* same value
            // in `state` as in `parent` -- and therefore can never be
            // `newly_satisfied`/`numeric_improved` (both require differing
            // from the parent). Only `dirty_list` -- built by
            // `mark_dirty_leaves` above -- needs visiting here; every other
            // leaf's contribution is filled in afterwards from the parent
            // snapshot, but only if this loop actually finds progress (see
            // the early exit below).
            for i in 0..self.dirty_list.len() {
                let idx = self.dirty_list[i] as usize;
                let leaf_id = idx as u32;
                match &self.leaves[idx] {
                    LeafData::Prop { operand } => {
                        if self.parent_prop_stamp[idx] != self.generation {
                            let p_val = operand.eval(parent_state)?;
                            self.parent_prop_true[idx] = is_true(&p_val);
                            self.parent_prop_stamp[idx] = self.generation;
                        }
                        let p_true = self.parent_prop_true[idx];
                        let s_val = operand.eval(state)?;
                        let s_true = is_true(&s_val);
                        if s_true {
                            self.classification.currently_satisfied.push(leaf_id);
                        }
                        if s_true && !p_true {
                            self.classification.newly_satisfied.push(leaf_id);
                        } else if s_true {
                            self.classification.persisting_satisfied.push(leaf_id);
                        }
                    }
                    LeafData::Num { lhs, rhs, kind } => {
                        if self.parent_numeric_stamp[idx] != self.generation {
                            self.parent_numeric[idx] =
                                compute_sdist(lhs, rhs, *kind, parent_state)?;
                            self.parent_numeric_stamp[idx] = self.generation;
                        }
                        let (sdist, curr_sat) = compute_sdist(lhs, rhs, *kind, state)?;
                        let (pdist, parent_sat) = &self.parent_numeric[idx];
                        if curr_sat {
                            self.classification.currently_satisfied.push(leaf_id);
                        } else {
                            self.classification.numeric_unsatisfied.push(leaf_id);
                        }
                        if !*parent_sat && sdist.cmp(pdist) == Ordering::Greater {
                            if curr_sat {
                                self.classification.newly_satisfied.push(leaf_id);
                            } else {
                                self.classification.numeric_improved.push(leaf_id);
                            }
                        } else if *parent_sat && curr_sat {
                            self.classification.persisting_satisfied.push(leaf_id);
                        }
                        self.classification.sdist_cache[idx] = sdist;
                    }
                }
            }

            if self.classification.newly_satisfied.is_empty()
                && self.classification.numeric_improved.is_empty()
            {
                // Nothing dirty made progress, and a clean leaf never can
                // (see above) -- Pass B and C's outer loops are both
                // `newly_satisfied`/`numeric_improved`-driven, so both are
                // guaranteed to be no-ops. Skip building the remaining
                // classification lists and the `self.partitions` lookup entirely.
                return Ok(3);
            }

            self.ensure_parent_snapshot(parent_state)?;

            // Merge in every leaf the sparse loop above skipped because
            // it's clean this call: its child value is the parent's cached
            // one, by the same argument.
            for i in 0..self.parent_sat_list.len() {
                let leaf_id = self.parent_sat_list[i];
                if self.dirty_stamp[leaf_id as usize] == self.eval_generation {
                    continue;
                }
                self.classification.currently_satisfied.push(leaf_id);
                self.classification.persisting_satisfied.push(leaf_id);
            }
            for i in 0..self.parent_num_unsat_list.len() {
                let leaf_id = self.parent_num_unsat_list[i];
                if self.dirty_stamp[leaf_id as usize] == self.eval_generation {
                    continue;
                }
                self.classification.numeric_unsatisfied.push(leaf_id);
            }
        }

        let tables = self.partitions.entry(partition).or_default();
        let mut novelty: u8 = 3;

        // Pass B: unary (size-1) novelty.
        for &leaf_id in &self.classification.newly_satisfied {
            if tables.psi_seen.insert(leaf_id) {
                novelty = 1;
            }
        }
        for &leaf_id in &self.classification.numeric_improved {
            let sdist = &self.classification.sdist_cache[leaf_id as usize];
            match tables.best_sdist.entry(leaf_id) {
                Entry::Occupied(mut e) => {
                    if sdist.cmp(e.get()) == Ordering::Greater {
                        e.insert(sdist.clone());
                        novelty = 1;
                    }
                }
                Entry::Vacant(e) => {
                    e.insert(sdist.clone());
                    novelty = 1;
                }
            }
        }

        // Pass C: binary (size-2) novelty. Always runs, even if Pass B
        // already found novelty 1, so the pair tables stay current.
        //
        // C1: psi x psi. On a fresh partition, `start = ax + 1` is only
        // correct because `newly_satisfied` and `currently_satisfied` are
        // pushed in lockstep above and are therefore element-identical --
        // skipping the prefix up to `ax` in `currently_satisfied` is
        // skipping exactly the pairs already enumerated by earlier outer
        // iterations, so each unordered pair is still visited exactly once.
        // If a future edit ever pushed to one list and not the other on the
        // `new_partition` branch, this would silently stop visiting some
        // pairs -- no error, just a different novelty class -- hence the
        // debug assertion (mirrors `novelty.py`'s `eval`).
        debug_assert!(
            !new_partition
                || self.classification.newly_satisfied == self.classification.currently_satisfied
        );
        for (ax, &f) in self.classification.newly_satisfied.iter().enumerate() {
            let start = if new_partition { ax + 1 } else { 0 };
            for &tid in self.classification.currently_satisfied.iter().skip(start) {
                if f == tid {
                    continue;
                }
                let pair = pack_pair(f, tid);
                if tables.psi_pair_seen.insert(pair) {
                    novelty = novelty.min(2);
                }
            }
        }

        // C1b: psi (newly added) x delta (any still-unsatisfied numeric).
        for &f in &self.classification.newly_satisfied {
            for &tid in &self.classification.numeric_unsatisfied {
                if f == tid {
                    continue;
                }
                // `tid` ranges over `numeric_unsatisfied`, which (unlike
                // `numeric_improved`) can hold a clean leaf on a
                // same-partition call -- `sdist_cache[tid]` is only fresh
                // for a leaf Pass A actually evaluated this call (every leaf
                // when `new_partition`, else only `dirty_list`); a clean
                // leaf's exact sdist is its parent's cached one instead
                // (see `sdist_cache`'s field doc).
                let sdist =
                    if new_partition || self.dirty_stamp[tid as usize] == self.eval_generation {
                        &self.classification.sdist_cache[tid as usize]
                    } else {
                        &self.parent_numeric[tid as usize].0
                    };
                let key = pack_directed(f, tid);
                match tables.best_sdist_with_psi.entry(key) {
                    Entry::Occupied(mut e) => {
                        if sdist.cmp(e.get()) == Ordering::Greater {
                            e.insert(sdist.clone());
                            novelty = novelty.min(2);
                        }
                    }
                    Entry::Vacant(e) => {
                        e.insert(sdist.clone());
                        novelty = novelty.min(2);
                    }
                }
            }
        }

        // C2: psi (persisting) x delta (improved but still unsatisfied
        // numeric subgoal). No extra "genuinely unsatisfied" filter here:
        // every leaf in `numeric_improved` is already unsatisfied by
        // construction (Pass A only pushes to it when `!curr_sat`), exactly
        // as in C1b above -- an explicit `is_negative()` check here would
        // additionally exclude a strict `<` leaf sitting exactly at
        // `sdist == 0` (unsatisfied, since strict `<` requires
        // `sdist > 0`), which C1b does process. Mirrored in `novelty.py`.
        for &tid in &self.classification.numeric_improved {
            let sdist = &self.classification.sdist_cache[tid as usize];
            for &f in &self.classification.persisting_satisfied {
                if f == tid {
                    continue;
                }
                let key = pack_directed(f, tid);
                match tables.best_sdist_with_psi.entry(key) {
                    Entry::Occupied(mut e) => {
                        if sdist.cmp(e.get()) == Ordering::Greater {
                            e.insert(sdist.clone());
                            novelty = novelty.min(2);
                        }
                    }
                    Entry::Vacant(e) => {
                        e.insert(sdist.clone());
                        novelty = novelty.min(2);
                    }
                }
            }
        }

        Ok(novelty)
    }
}

impl NumericNovelty {
    /// Marks, via `dirty_stamp`/`eval_generation`, every leaf that reads a
    /// fluent whose value differs between `parent` and `state`. Called once
    /// per `eval` call (not per leaf) whenever a parent is actually
    /// consulted. Bumps `eval_generation` first, so a leaf whose stamp
    /// isn't refreshed below is implicitly clean for this call.
    fn mark_dirty_leaves(&mut self, parent: &State, state: &State) {
        self.eval_generation += 1;
        self.dirty_list.clear();
        self.mark_dirty_by_chunks(parent, state);
    }

    /// The actual diff: `state.assignments`/`parent.assignments` are
    /// `im::Vector`s, and a child is always `parent.clone()` plus a handful
    /// of `Vector::set` calls, which path-copies only the touched leaf
    /// chunks (64 elements each) and leaves every other chunk
    /// pointer-identical to the parent's. So chunk pairs are compared by
    /// pointer first (O(1), no allocation) and only a chunk that actually
    /// differs is scanned element-by-element for the exact fluent ids that
    /// changed. Assumes the two vectors' chunk layouts line up one-to-one --
    /// true by construction (a search's fluent count is fixed once encoded,
    /// and `assignments` is only ever mutated via `set`, which cannot change
    /// chunk topology) -- checked with `debug_assert!` rather than handled
    /// at runtime.
    fn mark_dirty_by_chunks(&mut self, parent: &State, state: &State) {
        debug_assert_eq!(parent.assignments.len(), state.assignments.len());
        let mut fluent_idx: usize = 0;
        for (pc, sc) in parent.assignments.leaves().zip(state.assignments.leaves()) {
            debug_assert_eq!(pc.len(), sc.len());
            if !std::ptr::eq(pc.as_ptr(), sc.as_ptr()) {
                for (i, (pv, sv)) in pc.iter().zip(sc.iter()).enumerate() {
                    if pv != sv {
                        if let Some(leaves) = self.fluent_to_leaves.get(fluent_idx + i) {
                            for &leaf_id in leaves {
                                // Dedup against a leaf reached twice this
                                // call (two of its own fluents both
                                // changed, or the same fluent appears in
                                // more than one differing chunk).
                                if self.dirty_stamp[leaf_id as usize] != self.eval_generation {
                                    self.dirty_stamp[leaf_id as usize] = self.eval_generation;
                                    self.dirty_list.push(leaf_id);
                                }
                            }
                        }
                    }
                }
            }
            fluent_idx += pc.len();
        }
        debug_assert_eq!(fluent_idx, parent.assignments.len());
    }

    /// Builds (once per expansion, lazily -- only the first time a child
    /// actually needs it, see `eval`'s early exit) the dense "is this leaf
    /// satisfied in the parent" snapshot used to fill in the leaves sparse
    /// Pass A skipped. Reuses whatever per-leaf parent caches sparse Pass A
    /// already filled for the leaves it visited this call (`stamp ==
    /// generation`), so this only does fresh work for the other leaves.
    fn ensure_parent_snapshot(&mut self, parent_state: &State) -> PyResult<()> {
        if self.parent_snapshot_gen == self.generation {
            return Ok(());
        }
        self.parent_sat_list.clear();
        self.parent_num_unsat_list.clear();
        for (idx, leaf) in self.leaves.iter().enumerate() {
            let leaf_id = idx as u32;
            match leaf {
                LeafData::Prop { operand } => {
                    if self.parent_prop_stamp[idx] != self.generation {
                        let p_val = operand.eval(parent_state)?;
                        self.parent_prop_true[idx] = is_true(&p_val);
                        self.parent_prop_stamp[idx] = self.generation;
                    }
                    if self.parent_prop_true[idx] {
                        self.parent_sat_list.push(leaf_id);
                    }
                }
                LeafData::Num { lhs, rhs, kind } => {
                    if self.parent_numeric_stamp[idx] != self.generation {
                        self.parent_numeric[idx] = compute_sdist(lhs, rhs, *kind, parent_state)?;
                        self.parent_numeric_stamp[idx] = self.generation;
                    }
                    if self.parent_numeric[idx].1 {
                        self.parent_sat_list.push(leaf_id);
                    } else {
                        self.parent_num_unsat_list.push(leaf_id);
                    }
                }
            }
        }
        self.parent_snapshot_gen = self.generation;
        Ok(())
    }
}
