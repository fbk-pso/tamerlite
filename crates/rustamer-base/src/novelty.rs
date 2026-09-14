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
//! `Sdist` keeps the distance feature exact (`BigInt`/`BigRational`, never
//! `f64`) to match Python's exact `Fraction` arithmetic.

use num::{BigInt, Signed, Zero};
use num_rational::BigRational;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use std::borrow::Cow;
use std::cmp::Ordering;

use super::expressions::{ExpressionNode, PyExpressionNode};
use super::expressions_utils::{
    as_num_ref, internal_evaluate, num_cmp, num_is_zero, split_expression, FluentValueTrait, NumRef,
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

/// The distance feature for a numeric leaf, exact (never `f64`) -- see the
/// module docstring. `Int` is used whenever both operands are exact
/// integers (the common case for resource-counter fluents), promoting to
/// `Rational` only when an operand actually is one, exactly mirroring
/// `internal_evaluate`'s `Minus`/`fold_numeric` promotion rule so this
/// never disagrees with how the rest of the crate does arithmetic.
#[derive(Clone, Debug)]
enum Sdist {
    Int(BigInt),
    Rational(BigRational),
}

impl Sdist {
    /// `a - b`.
    fn sub(a: NumRef, b: NumRef) -> Sdist {
        match (a, b) {
            (NumRef::Int(x), NumRef::Int(y)) => Sdist::Int(x - y),
            (NumRef::Int(x), NumRef::Rational(y)) => {
                Sdist::Rational(BigRational::from_integer(x.clone()) - y)
            }
            (NumRef::Rational(x), NumRef::Int(y)) => {
                Sdist::Rational(x - BigRational::from_integer(y.clone()))
            }
            (NumRef::Rational(x), NumRef::Rational(y)) => Sdist::Rational(x - y),
        }
    }

    fn as_num_ref(&self) -> NumRef<'_> {
        match self {
            Sdist::Int(v) => NumRef::Int(v),
            Sdist::Rational(v) => NumRef::Rational(v),
        }
    }

    fn is_zero(&self) -> bool {
        num_is_zero(self.as_num_ref())
    }

    fn is_negative(&self) -> bool {
        match self {
            Sdist::Int(v) => v.is_negative(),
            Sdist::Rational(v) => v.is_negative(),
        }
    }

    fn is_positive(&self) -> bool {
        match self {
            Sdist::Int(v) => v.is_positive(),
            Sdist::Rational(v) => v.is_positive(),
        }
    }

    fn cmp(&self, other: &Sdist) -> Ordering {
        num_cmp(self.as_num_ref(), other.as_num_ref())
    }

    fn neg_abs(&self) -> Sdist {
        match self {
            Sdist::Int(v) => Sdist::Int(-v.abs()),
            Sdist::Rational(v) => Sdist::Rational(-v.abs()),
        }
    }
}

/// The distance feature paired with satisfaction, in the "improves = larger
/// is better" sign convention -- mirrors `novelty.py::_sdist_and_sat`
/// exactly: satisfied inequalities are `>= 0`, satisfied equalities are
/// exactly `0` (via `-abs(diff)`), and satisfaction for strict `<` is *not*
/// simply `sdist >= 0` (the exact boundary is genuinely unsatisfied).
fn compute_sdist(
    lhs: &Operand,
    rhs: &Operand,
    kind: NumKind,
    state: &State,
) -> PyResult<(Sdist, bool)> {
    let lhs_val = lhs.eval(state)?;
    let rhs_val = rhs.eval(state)?;
    let diff = Sdist::sub(as_num_ref(&rhs_val)?, as_num_ref(&lhs_val)?);
    Ok(match kind {
        NumKind::Eq => {
            let sat = diff.is_zero();
            (diff.neg_abs(), sat)
        }
        NumKind::Le => {
            let sat = !diff.is_negative();
            (diff, sat)
        }
        NumKind::Lt => {
            let sat = diff.is_positive();
            (diff, sat)
        }
    })
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
) -> PyResult<()> {
    if leaf_index.contains_key(&exp) {
        return Ok(());
    }
    let leaf_id = leaves.len() as u32;
    leaf_index.insert(exp.clone(), leaf_id);
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
    /// Indexed by global leaf id, sized to `n_leaves`. Entries are only
    /// ever read for leaf ids present in `numeric_improved`/
    /// `numeric_unsatisfied`, which are rebuilt fresh (and written before
    /// any read) every `eval` call -- so this never needs clearing between
    /// calls, only a valid initial fill at construction.
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
            sdist_cache: vec![Sdist::Int(BigInt::zero()); n_leaves],
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
/// once per surviving successor, in generation order -- not thread-safe /
/// not reusable across searches without calling `start` again.
#[pyclass]
pub struct NumericNovelty {
    leaves: Vec<LeafData>,
    partitions: FxHashMap<u64, PartitionTables>,
    max_partition: u64,
    classification: LeafClassification,
    /// Lazy per-leaf caches of the current expansion's parent state's
    /// features, reset by `begin_expansion` and filled on first use by
    /// `eval` (mirrors `novelty.py`'s `_parent_prop_true`/`_parent_numeric`
    /// -- same caching design, shared with the Python core). `None` where
    /// not yet computed this expansion, or where n/a (root, or a child
    /// landing in a different partition than its parent, never touches
    /// these at all).
    parent_prop_true: Vec<Option<bool>>,
    parent_numeric: Vec<Option<(Sdist, bool)>>,
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

        for event_list in events.values() {
            for (_, event) in event_list {
                for cond in event_conditions(event)? {
                    for leaf in iter_subgoal_leaves(&cond) {
                        add_leaf(leaf?, &mut leaves, &mut leaf_index, &fluent_domains)?;
                    }
                }
            }
        }
        for leaf in iter_subgoal_leaves(&goals) {
            add_leaf(leaf?, &mut leaves, &mut leaf_index, &fluent_domains)?;
        }

        let n = leaves.len();
        Ok(NumericNovelty {
            leaves,
            partitions: FxHashMap::default(),
            max_partition: 1,
            classification: LeafClassification::new(n),
            parent_prop_true: vec![None; n],
            parent_numeric: vec![None; n],
        })
    }

    /// (Re)initializes partition bookkeeping from the initial state's
    /// h^add value and clears the parent-feature cache (so a following
    /// `eval` on the root sees "no parent" unconditionally, matching
    /// calling `eval(init, p, None, None)` directly). Must be called
    /// exactly once, before any `eval()` call. Returns the root's
    /// (clamped) partition id.
    pub fn start(&mut self, initial_h: f64) -> u64 {
        assert!(
            initial_h >= 0.0,
            "initial_h must be non-negative (novbfs always uses h^add, which \
             never returns a negative value for a reachable state)"
        );
        self.partitions.clear();
        self.max_partition = (initial_h.floor() as u64).max(1);
        self.parent_prop_true.iter_mut().for_each(|v| *v = None);
        self.parent_numeric.iter_mut().for_each(|v| *v = None);
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
        self.parent_prop_true.iter_mut().for_each(|v| *v = None);
        self.parent_numeric.iter_mut().for_each(|v| *v = None);
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

        for (idx, leaf) in self.leaves.iter().enumerate() {
            let leaf_id = idx as u32;
            match leaf {
                LeafData::Prop { operand } => {
                    let s_val = operand.eval(state)?;
                    let s_true = is_true(&s_val);
                    if s_true {
                        self.classification.currently_satisfied.push(leaf_id);
                    }
                    if new_partition {
                        if s_true {
                            self.classification.newly_satisfied.push(leaf_id);
                        }
                    } else {
                        let parent_state = parent.expect("parent set when !new_partition");
                        let p_true = match self.parent_prop_true[idx] {
                            Some(v) => v,
                            None => {
                                let p_val = operand.eval(parent_state)?;
                                let v = is_true(&p_val);
                                self.parent_prop_true[idx] = Some(v);
                                v
                            }
                        };
                        if s_true && !p_true {
                            self.classification.newly_satisfied.push(leaf_id);
                        } else if s_true {
                            self.classification.persisting_satisfied.push(leaf_id);
                        }
                    }
                }
                LeafData::Num { lhs, rhs, kind } => {
                    let (sdist, curr_sat) = compute_sdist(lhs, rhs, *kind, state)?;
                    if new_partition {
                        if curr_sat {
                            self.classification.currently_satisfied.push(leaf_id);
                            self.classification.newly_satisfied.push(leaf_id);
                        } else {
                            self.classification.numeric_unsatisfied.push(leaf_id);
                            self.classification.numeric_improved.push(leaf_id);
                        }
                    } else {
                        let parent_state = parent.expect("parent set when !new_partition");
                        let (pdist, parent_sat) = match &self.parent_numeric[idx] {
                            Some(v) => v.clone(),
                            None => {
                                let v = compute_sdist(lhs, rhs, *kind, parent_state)?;
                                self.parent_numeric[idx] = Some(v.clone());
                                v
                            }
                        };
                        if curr_sat {
                            self.classification.currently_satisfied.push(leaf_id);
                        } else {
                            self.classification.numeric_unsatisfied.push(leaf_id);
                        }
                        if !parent_sat && sdist.cmp(&pdist) == Ordering::Greater {
                            if curr_sat {
                                self.classification.newly_satisfied.push(leaf_id);
                            } else {
                                self.classification.numeric_improved.push(leaf_id);
                            }
                        } else if parent_sat && curr_sat {
                            self.classification.persisting_satisfied.push(leaf_id);
                        }
                    }
                    self.classification.sdist_cache[idx] = sdist;
                }
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
            let improved = match tables.best_sdist.get(&leaf_id) {
                None => true,
                Some(prev) => sdist.cmp(prev) == Ordering::Greater,
            };
            if improved {
                tables.best_sdist.insert(leaf_id, sdist.clone());
                novelty = 1;
            }
        }

        // Pass C: binary (size-2) novelty. Always runs, even if Pass B
        // already found novelty 1, so the pair tables stay current.
        //
        // C1: psi x psi.
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
                let sdist = &self.classification.sdist_cache[tid as usize];
                let key = pack_directed(f, tid);
                let improved = match tables.best_sdist_with_psi.get(&key) {
                    None => true,
                    Some(old) => sdist.cmp(old) == Ordering::Greater,
                };
                if improved {
                    tables.best_sdist_with_psi.insert(key, sdist.clone());
                    novelty = novelty.min(2);
                }
            }
        }

        // C2: psi (persisting) x delta (improved but still genuinely
        // unsatisfied numeric subgoal).
        for &tid in &self.classification.numeric_improved {
            let sdist = &self.classification.sdist_cache[tid as usize];
            if !sdist.is_negative() {
                continue;
            }
            for &f in &self.classification.persisting_satisfied {
                if f == tid {
                    continue;
                }
                let key = pack_directed(f, tid);
                let improved = match tables.best_sdist_with_psi.get(&key) {
                    None => true,
                    Some(old) => sdist.cmp(old) == Ordering::Greater,
                };
                if improved {
                    tables.best_sdist_with_psi.insert(key, sdist.clone());
                    novelty = novelty.min(2);
                }
            }
        }

        Ok(novelty)
    }
}
