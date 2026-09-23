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

use log::{debug, info};
use min_max_heap::MinMaxHeap;
use std::collections::VecDeque;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::rc::Rc;
use std::time::SystemTime;
use std::{collections::BinaryHeap, vec::Vec};

use fastbloom::BloomFilter;
use foldhash::fast::{FixedState, RandomState};
use rustc_hash::{FxBuildHasher, FxHashMap, FxHashSet};
use std::hash::BuildHasher;

use pyo3::exceptions::PyTimeoutError;
use pyo3::prelude::*;

use super::heuristics::*;
use super::novelty::NumericNovelty;
use super::search_space::*;
use super::search_state::*;
use super::structures::Action;
use super::utils::PersistentList;

pub type SearchResult = (Option<Vec<Action>>, FxHashMap<&'static str, String>);

/// Mirrors `heuristics.rs`'s `EvalGenItem`/`EvalGenOwnedItem`, generalized
/// over the payload type `StatePayload::eval_gen` bridges between.
type EvalItem<P> = PyResult<(P, Option<f64>)>;

/// A payload an open-list item carries: `Rc<State>` (`wastar_search`,
/// `novbfs_search`) or an owned `State` (`wastar_search_memory_bounded`,
/// which needs ownership to store items inline in a bounded array rather
/// than behind a refcount). Bridges `HeuristicTrait`'s `eval_gen` /
/// `eval_gen_owned` split, which exists for the same reason.
///
/// `'static`: required for `eval_gen`'s boxed return type below to be well
/// formed for a generic `Self` -- both impls satisfy it trivially (`State`
/// is a `#[pyclass]`, hence already `'static`; `Rc<State>: 'static` follows).
trait StatePayload: Sized + 'static {
    fn wrap(s: State) -> Self;
    fn as_state(&self) -> &State;

    /// Evaluates `heuristic` over `states` via whichever of
    /// `HeuristicTrait::eval_gen`/`eval_gen_owned` matches this payload.
    fn eval_gen<'a, H, S, I>(
        heuristic: &'a H,
        states: I,
        ss: &'a S,
    ) -> PyResult<Box<dyn Iterator<Item = EvalItem<Self>> + 'a>>
    where
        H: HeuristicTrait,
        S: SearchSpaceTrait,
        I: Iterator<Item = PyResult<Self>> + 'a;
}

impl StatePayload for Rc<State> {
    fn wrap(s: State) -> Self {
        Rc::new(s)
    }

    fn as_state(&self) -> &State {
        self.as_ref()
    }

    fn eval_gen<'a, H, S, I>(
        heuristic: &'a H,
        states: I,
        ss: &'a S,
    ) -> PyResult<Box<dyn Iterator<Item = EvalItem<Self>> + 'a>>
    where
        H: HeuristicTrait,
        S: SearchSpaceTrait,
        I: Iterator<Item = PyResult<Self>> + 'a,
    {
        heuristic.eval_gen(states, ss)
    }
}

impl StatePayload for State {
    fn wrap(s: State) -> Self {
        s
    }

    fn as_state(&self) -> &State {
        self
    }

    fn eval_gen<'a, H, S, I>(
        heuristic: &'a H,
        states: I,
        ss: &'a S,
    ) -> PyResult<Box<dyn Iterator<Item = EvalItem<Self>> + 'a>>
    where
        H: HeuristicTrait,
        S: SearchSpaceTrait,
        I: Iterator<Item = PyResult<Self>> + 'a,
    {
        heuristic.eval_gen_owned(states, ss)
    }
}

/// The open list `priority_search` drains: `BinaryHeap` (unbounded) or
/// `BoundedPriorityQueue` (memory-bounded). Named distinctly from both
/// types' inherent `push`/`pop`/`len` so a trait call can never silently
/// resolve to the wrong one (inherent methods shadow trait methods of the
/// same name, so an accidental name match here would risk infinite
/// recursion the day an inherent method is refactored away).
trait OpenList<T> {
    fn push_item(&mut self, item: T);
    fn pop_item(&mut self) -> Option<T>;
    fn size(&self) -> usize;
}

impl<T: Ord> OpenList<T> for BinaryHeap<T> {
    fn push_item(&mut self, item: T) {
        self.push(item);
    }

    fn pop_item(&mut self) -> Option<T> {
        self.pop()
    }

    fn size(&self) -> usize {
        self.len()
    }
}

impl<T: Ord> OpenList<T> for BoundedPriorityQueue<T> {
    fn push_item(&mut self, item: T) {
        // Accepted/rejected return value intentionally discarded.
        self.push(item);
    }

    fn pop_item(&mut self) -> Option<T> {
        self.pop()
    }

    fn size(&self) -> usize {
        self.len()
    }
}

/// The visited-state dedup store: a `FxHashSet<WeakEqState>`
/// (`wastar_search`, `novbfs_search`) or a lossy `BloomFilter`
/// (`wastar_search_memory_bounded`), keyed on the same equivalence as
/// `WeakEqState` (`assignments` plus each `todo` action's event index).
/// `insert_new` mirrors `HashSet::insert`'s "newly inserted"
/// polarity regardless of backing store; a disabled store (non-temporal
/// dedup gate off) must always report "new" without recording anything.
trait DedupStore<P> {
    fn insert_new(&mut self, p: &P) -> bool;
}

struct HashSetDedup {
    enabled: bool,
    // `State::heuristic_cache` is a `Mutex`, which is what trips this lint on
    // `Rc<State>` -- but `WeakEqState`'s `Hash`/`PartialEq` impls only ever
    // read `assignments`/`todo`, never `heuristic_cache`, so mutating the
    // cache after insertion can't change a key's hash/eq out from under the
    // set.
    #[allow(clippy::mutable_key_type)]
    seen: FxHashSet<WeakEqState>,
}

impl HashSetDedup {
    fn new<S: SearchSpaceTrait>(ss: &S, weak_equality: bool) -> Self {
        Self {
            enabled: !ss.is_temporal() || weak_equality,
            seen: FxHashSet::with_hasher(FxBuildHasher),
        }
    }
}

impl DedupStore<Rc<State>> for HashSetDedup {
    fn insert_new(&mut self, s: &Rc<State>) -> bool {
        if !self.enabled {
            return true;
        }
        self.seen.insert(WeakEqState {
            state: Rc::clone(s),
        })
    }
}

struct BloomDedup {
    filter: Option<BloomFilter<RandomState>>,
}

impl BloomDedup {
    fn new<S: SearchSpaceTrait>(ss: &S, weak_equality: bool) -> Self {
        const BLOOM_ITEMS: usize = 20_000_000;
        const BLOOM_FP_RATE: f64 = 1e-4;
        let filter = (!ss.is_temporal() || weak_equality).then(|| {
            BloomFilter::with_false_pos(BLOOM_FP_RATE)
                .hasher(RandomState::default())
                .expected_items(BLOOM_ITEMS)
        });
        Self { filter }
    }
}

impl DedupStore<State> for BloomDedup {
    fn insert_new(&mut self, s: &State) -> bool {
        match &mut self.filter {
            None => true,
            // `BloomFilter::insert` returns "may have been previously
            // present" -- the opposite polarity of `HashSet::insert`.
            Some(filter) => {
                // Order-independent digest of `todo`: hash each entry on its
                // own and combine with a commutative `wrapping_add`, so
                // `FxHashMap`'s iteration order doesn't matter -- no
                // allocation, no sort. Map keys are unique, so no entry can
                // cancel another out the way it could under a set-style
                // XOR. The extra collisions this admits are well below the
                // filter's own false-positive rate.
                let todo_digest = s.todo.iter().fold(0u64, |acc, (a, (idx, _))| {
                    acc.wrapping_add(FixedState::default().hash_one((a, idx)))
                });
                !filter.insert(&(&s.assignments, todo_digest))
            }
        }
    }
}

/// Per-search open-list-item construction plus the optional per-expansion
/// hook (`novbfs_search`'s `NumericNovelty::begin_expansion`).
trait SearchStrategy {
    type Payload: StatePayload;
    type Item: Ord;

    /// Associated fn, not a method: taking `&self` here would hold the
    /// strategy borrowed for the whole expansion (the successor iterator
    /// borrows its `&State` argument for that same lifetime), conflicting
    /// with the `&mut self` hooks below.
    fn state_of(item: &Self::Item) -> &Self::Payload;

    fn root(&mut self, s: Self::Payload, h: f64) -> PyResult<Self::Item>;

    fn begin_expansion(&mut self) {}

    fn child(
        &mut self,
        parent: &Self::Item,
        s: Self::Payload,
        h: f64,
        idx: usize,
    ) -> PyResult<Self::Item>;
}

struct PrioritizedItem<T: StatePayload> {
    heuristic: f64,
    state: T,
    idx: usize,
}

impl<T: StatePayload> PartialEq for PrioritizedItem<T> {
    fn eq(&self, _other: &Self) -> bool {
        false
    }
}

impl<T: StatePayload> Eq for PrioritizedItem<T> {}

impl<T: StatePayload> PartialOrd for PrioritizedItem<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: StatePayload> Ord for PrioritizedItem<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // `BinaryHeap` is a max-heap; comparing `other` against `self` (not
        // `self` against `other`) inverts every field so `pop()` returns
        // the item with the lexicographically smallest
        // `(heuristic, todo_len, idx)` tuple first. `f64` isn't `Ord`,
        // hence `total_cmp` in place of a plain tuple comparison.
        other
            .heuristic
            .total_cmp(&self.heuristic)
            .then_with(|| {
                other
                    .state
                    .as_state()
                    .todo
                    .len()
                    .cmp(&self.state.as_state().todo.len())
            })
            .then_with(|| other.idx.cmp(&self.idx))
    }
}

/// `weight`-parameterized strategy shared by `wastar_search` (`P = Rc<State>`)
/// and `wastar_search_memory_bounded` (`P = State`).
struct WAStarStrategy<P> {
    weight: f64,
    _payload: PhantomData<P>,
}

impl<P> WAStarStrategy<P> {
    fn new(weight: f64) -> Self {
        Self {
            weight,
            _payload: PhantomData,
        }
    }
}

impl<P: StatePayload> SearchStrategy for WAStarStrategy<P> {
    type Payload = P;
    type Item = PrioritizedItem<P>;

    fn state_of(item: &Self::Item) -> &P {
        &item.state
    }

    fn root(&mut self, s: P, h: f64) -> PyResult<Self::Item> {
        Ok(PrioritizedItem {
            heuristic: h,
            state: s,
            idx: 0,
        })
    }

    fn child(&mut self, _parent: &Self::Item, s: P, h: f64, idx: usize) -> PyResult<Self::Item> {
        let f = self.weight * h + (1.0 - self.weight) * s.as_state().g;
        Ok(PrioritizedItem {
            heuristic: f,
            state: s,
            idx,
        })
    }
}

pub struct BoundedPriorityQueue<T: Ord> {
    heap: MinMaxHeap<T>,
    bound: usize,
}

impl<T: Ord> BoundedPriorityQueue<T> {
    pub fn with_bound(bound: usize) -> Self {
        assert!(bound > 0, "bound must be positive");
        Self {
            heap: MinMaxHeap::with_capacity(bound),
            bound,
        }
    }

    /// Push an item only if the heap is under capacity, or the item is
    /// better than the current minimum. Returns false if the item was rejected.
    pub fn push(&mut self, item: T) -> bool {
        if self.heap.len() < self.bound {
            self.heap.push(item);
            return true;
        }

        // Heap is full: peek the current minimum
        let min = self.heap.peek_min().unwrap();
        if &item <= min {
            // New item is worse than or equal to the worst in the heap: reject it
            return false;
        }

        // Item is better than the worst: evict the worst and insert the new one
        self.heap.replace_min(item);
        true
    }

    pub fn pop(&mut self) -> Option<T> {
        self.heap.pop_max()
    }

    pub fn len(&self) -> usize {
        self.heap.len()
    }
}

/// Wraps a state so the visited-state dedup set also compares `todo` (durative actions
/// in progress), which is what the temporal `weak_equality` dedup path needs on top of
/// the full `assignments` compare/hash. On the classical (`!is_temporal()`) dedup path
/// `todo` is always empty, so that comparison is a no-op there.
pub struct WeakEqState {
    pub state: Rc<State>,
}

impl PartialEq for WeakEqState {
    fn eq(&self, other: &Self) -> bool {
        weak_eq(&self.state, &other.state)
    }
}

impl Eq for WeakEqState {}

impl Hash for WeakEqState {
    fn hash<H: Hasher>(&self, state: &mut H) {
        Hash::hash(&self.state.assignments, state);
    }
}

pub fn weak_eq(state1: &State, state2: &State) -> bool {
    if state1.todo.len() != state2.todo.len() {
        return false;
    }
    if state1.assignments != state2.assignments {
        return false;
    }
    for (a, (idx, _)) in &state1.todo {
        let idx_id = state2.todo.get(a);
        if idx_id.is_none() || *idx != idx_id.unwrap().0 {
            return false;
        }
    }
    true
}

pub fn extract_path(state: &State) -> Vec<Action> {
    PersistentList::to_vec(&state.path)
        .into_iter()
        .map(|(a, _, _)| *a)
        .collect()
}

/// Shared skeleton for every priority-queue-based search in this module
/// (`wastar_search`, `wastar_search_memory_bounded`, `novbfs_search`).
/// What it actually shares across all three: the
/// metrics map, the `while` loop with its timeout check,
/// `expanded_states`/`generated_states` bookkeeping, the goal checks on
/// both the popped state and each early-terminated successor, and the
/// "generate once, never reopen" dedup contract. What varies between
/// callers is threaded through as parameters: `name` (log prefix), `open`/
/// `dedup` (the queue and dedup-store implementations), and `strategy`
/// (open-list-item construction plus the optional per-expansion hook
/// `novbfs_search` uses for `NumericNovelty::begin_expansion`).
///
/// `strategy`, `open` and `dedup` are three separate `&mut` parameters on
/// purpose: the successor closure below captures `dedup` for the whole
/// expansion, while `strategy`/`open` are mutated inside the loop that
/// consumes it. Bundling them into one `&mut` context struct would rely on
/// disjoint-closure-capture to keep the borrows independent, which breaks
/// the moment a helper method touches more than one field at once.
#[allow(clippy::too_many_arguments)]
fn priority_search<H, S, St, O, D>(
    ss: &S,
    heuristic: &H,
    timeout: Option<f32>,
    early_termination: bool,
    name: &str,
    strategy: &mut St,
    open: &mut O,
    dedup: &mut D,
) -> PyResult<SearchResult>
where
    H: HeuristicTrait,
    S: SearchSpaceTrait,
    St: SearchStrategy,
    O: OpenList<St::Item>,
    D: DedupStore<St::Payload>,
{
    let mut metrics = FxHashMap::with_hasher(FxBuildHasher);
    let start = SystemTime::now();
    let init = <St::Payload as StatePayload>::wrap(ss.initial_state(None)?);
    let mut expanded_states: usize = 0;
    let mut generated_states: usize = 1;

    // Seed the dedup store; return discarded, the root is never re-checked.
    dedup.insert_new(&init);

    if early_termination && ss.goal_reached(init.as_state(), None)? {
        metrics.insert("expanded_states", expanded_states.to_string());
        metrics.insert("goal_depth", init.as_state().g.to_string());
        return Ok((Some(extract_path(init.as_state())), metrics));
    }

    let init_h = match heuristic.eval(init.as_state(), ss)? {
        Some(v) => v,
        None => {
            metrics.insert("expanded_states", 0.to_string());
            return Ok((None, metrics));
        }
    };
    open.push_item(strategy.root(init, init_h)?);

    while let Some(current) = open.pop_item() {
        if let Some(t) = timeout {
            if start.elapsed().unwrap().as_secs_f32() > t {
                return Err(PyTimeoutError::new_err("Timeout"));
            }
        }
        expanded_states += 1;
        if expanded_states.is_multiple_of(10_000) {
            debug!(
                "{}: expanded={} generated={} open={}",
                name,
                expanded_states,
                generated_states,
                open.size()
            );
        }

        // Borrow, never move: `current` must stay whole for
        // `strategy.child` below.
        let state = St::state_of(&current).as_state();
        if !early_termination && ss.goal_reached(state, None)? {
            info!(
                "{}: goal found — expanded={} depth={}",
                name, expanded_states, state.g
            );
            metrics.insert("expanded_states", expanded_states.to_string());
            metrics.insert("goal_depth", state.g.to_string());
            return Ok((Some(extract_path(state)), metrics));
        }

        let successors_iter = ss
            .get_successor_states_iter(state)
            .filter_map(|rs| match rs {
                Ok(s) => {
                    let s = <St::Payload as StatePayload>::wrap(s);
                    dedup.insert_new(&s).then_some(Ok(s))
                }
                Err(e) => Some(Err(e)),
            });

        strategy.begin_expansion();

        for rs in <St::Payload as StatePayload>::eval_gen(heuristic, successors_iter, ss)? {
            let (s, h) = rs?;
            if early_termination && ss.goal_reached(s.as_state(), None)? {
                info!(
                    "{}: goal found — expanded={} depth={}",
                    name,
                    expanded_states,
                    s.as_state().g
                );
                metrics.insert("expanded_states", expanded_states.to_string());
                metrics.insert("goal_depth", s.as_state().g.to_string());
                return Ok((Some(extract_path(s.as_state())), metrics));
            }
            if let Some(v) = h {
                let item = strategy.child(&current, s, v, generated_states)?;
                open.push_item(item);
            }
            generated_states += 1;
        }
    }
    info!("{}: no solution found — expanded={}", name, expanded_states);
    metrics.insert("expanded_states", expanded_states.to_string());
    Ok((None, metrics))
}

pub fn wastar_search<H: HeuristicTrait, S: SearchSpaceTrait>(
    ss: &S,
    heuristic: &H,
    weight: f64,
    timeout: Option<f32>,
    early_termination: bool,
    weak_equality: bool,
) -> PyResult<SearchResult> {
    info!(
        "wastar_search: weight={} timeout={:?} early_termination={} weak_equality={}",
        weight, timeout, early_termination, weak_equality
    );
    let mut strategy = WAStarStrategy::<Rc<State>>::new(weight);
    let mut open: BinaryHeap<PrioritizedItem<Rc<State>>> = BinaryHeap::new();
    let mut dedup = HashSetDedup::new(ss, weak_equality);
    priority_search(
        ss,
        heuristic,
        timeout,
        early_termination,
        "wastar_search",
        &mut strategy,
        &mut open,
        &mut dedup,
    )
}

pub fn wastar_search_memory_bounded<H: HeuristicTrait, S: SearchSpaceTrait>(
    ss: &S,
    heuristic: &H,
    weight: f64,
    timeout: Option<f32>,
    early_termination: bool,
    weak_equality: bool,
) -> PyResult<SearchResult> {
    info!(
        "wastar_search_memory_bounded: weight={} timeout={:?} early_termination={} weak_equality={}",
        weight, timeout, early_termination, weak_equality
    );
    const QUEUE_BOUND: usize = 400_000;
    let mut strategy = WAStarStrategy::<State>::new(weight);
    let mut open: BoundedPriorityQueue<PrioritizedItem<State>> =
        BoundedPriorityQueue::with_bound(QUEUE_BOUND);
    let mut dedup = BloomDedup::new(ss, weak_equality);
    priority_search(
        ss,
        heuristic,
        timeout,
        early_termination,
        "wastar_search_memory_bounded",
        &mut strategy,
        &mut open,
        &mut dedup,
    )
}

pub fn bfs_search<S: SearchSpaceTrait>(
    ss: &S,
    timeout: Option<f32>,
    early_termination: bool,
) -> PyResult<SearchResult> {
    basic_search(ss, true, timeout, early_termination)
}

pub fn dfs_search<S: SearchSpaceTrait>(
    ss: &S,
    timeout: Option<f32>,
    early_termination: bool,
) -> PyResult<SearchResult> {
    basic_search(ss, false, timeout, early_termination)
}

fn basic_search<S: SearchSpaceTrait>(
    ss: &S,
    bfs: bool,
    timeout: Option<f32>,
    early_termination: bool,
) -> PyResult<SearchResult> {
    let name = if bfs { "bfs" } else { "dfs" };
    info!(
        "{}: timeout={:?} early_termination={}",
        name, timeout, early_termination
    );
    let mut metrics = FxHashMap::with_hasher(FxBuildHasher);
    let start = SystemTime::now();
    let init = ss.initial_state(None)?;
    let mut open = VecDeque::new();
    let mut expanded_states = 0;
    let mut generated_states = 1;

    if early_termination && ss.goal_reached(&init, None)? {
        metrics.insert("expanded_states", expanded_states.to_string());
        metrics.insert("goal_depth", init.g.to_string());
        return Ok((Some(extract_path(&init)), metrics));
    }
    open.push_back(init);

    while !open.is_empty() {
        if let Some(t) = timeout {
            if start.elapsed().unwrap().as_secs_f32() > t {
                return Err(PyTimeoutError::new_err("Timeout"));
            }
        }

        let state = if bfs {
            open.pop_front().unwrap()
        } else {
            open.pop_back().unwrap()
        };

        expanded_states += 1;
        if expanded_states % 10_000 == 0 {
            debug!(
                "{}: expanded={} generated={} open={}",
                name,
                expanded_states,
                generated_states,
                open.len()
            );
        }

        if !early_termination && ss.goal_reached(&state, None)? {
            info!(
                "{}: goal found — expanded={} depth={}",
                name, expanded_states, state.g
            );
            metrics.insert("expanded_states", expanded_states.to_string());
            metrics.insert("goal_depth", state.g.to_string());
            return Ok((Some(extract_path(&state)), metrics));
        } else {
            for rs in ss.get_successor_states_iter(&state) {
                let s = rs?;
                if early_termination && ss.goal_reached(&s, None)? {
                    info!(
                        "{}: goal found — expanded={} depth={}",
                        name, expanded_states, s.g
                    );
                    metrics.insert("expanded_states", expanded_states.to_string());
                    metrics.insert("goal_depth", s.g.to_string());
                    return Ok((Some(extract_path(&s)), metrics));
                }
                open.push_back(s);
                generated_states += 1;
            }
        }
    }
    info!("{}: no solution found — expanded={}", name, expanded_states);
    metrics.insert("expanded_states", expanded_states.to_string());
    Ok((None, metrics))
}

pub fn ehc_search<H: HeuristicTrait, S: SearchSpaceTrait>(
    ss: &S,
    heuristic: &H,
    timeout: Option<f32>,
    early_termination: bool,
    weak_equality: bool,
) -> PyResult<SearchResult> {
    info!(
        "ehc_search: timeout={:?} early_termination={} weak_equality={}",
        timeout, early_termination, weak_equality
    );
    let mut metrics = FxHashMap::with_hasher(FxBuildHasher);
    let start = SystemTime::now();
    let init = Rc::new(ss.initial_state(None)?);
    let mut expanded_states = 0;
    let mut generated_states = 1;

    if early_termination && ss.goal_reached(&init, None)? {
        metrics.insert("expanded_states", expanded_states.to_string());
        metrics.insert("goal_depth", init.g.to_string());
        return Ok((Some(extract_path(&init)), metrics));
    }

    let mut best_h = match heuristic.eval(&init, ss)? {
        Some(v) => v,
        None => {
            metrics.insert("expanded_states", 0.to_string());
            return Ok((None, metrics));
        }
    };
    debug!("ehc_search: initial h={:.4}", best_h);
    let mut open = VecDeque::new();
    open.push_back(init);

    let dedup = !ss.is_temporal() || weak_equality;
    // State and WeakEqState contain interior mutability only for heuristic
    // caches. The mutable fields are ignored by Hash/Eq, so using them as HashSet keys is
    // safe.
    #[allow(clippy::mutable_key_type)]
    let mut closed = FxHashSet::with_hasher(FxBuildHasher);
    while let Some(state) = open.pop_front() {
        if let Some(t) = timeout {
            if start.elapsed().unwrap().as_secs_f32() > t {
                return Err(PyTimeoutError::new_err("Timeout"));
            }
        }

        expanded_states += 1;
        if !early_termination && ss.goal_reached(&state, None)? {
            info!(
                "ehc_search: goal found — expanded={} depth={}",
                expanded_states, state.g
            );
            metrics.insert("expanded_states", expanded_states.to_string());
            metrics.insert("goal_depth", state.g.to_string());
            return Ok((Some(extract_path(&state)), metrics));
        } else {
            if dedup {
                closed.insert(WeakEqState {
                    state: Rc::clone(&state),
                });
            }

            let successors_iter = ss
                .get_successor_states_iter(&state)
                .filter_map(|rs| match rs {
                    Ok(s) => {
                        let s = Rc::new(s);
                        if dedup {
                            let weak_eq_state = WeakEqState { state: s };
                            (!closed.contains(&weak_eq_state)).then_some(Ok(weak_eq_state.state))
                        } else {
                            Some(Ok(s))
                        }
                    }
                    Err(e) => Some(Err(e)),
                });

            let mut new_best_found = false;
            for rs in heuristic.eval_gen(successors_iter, ss)? {
                let (s, h) = rs?;
                generated_states += 1;
                if early_termination && ss.goal_reached(&s, None)? {
                    info!(
                        "ehc_search: goal found — expanded={} depth={}",
                        expanded_states, s.g
                    );
                    metrics.insert("expanded_states", expanded_states.to_string());
                    metrics.insert("goal_depth", s.g.to_string());
                    return Ok((Some(extract_path(&s)), metrics));
                }
                if let Some(v) = h {
                    if v < best_h {
                        new_best_found = true;
                        best_h = v;
                        open.clear();
                        open.push_back(s);
                        break;
                    } else {
                        open.push_back(s);
                    }
                }
            }
            if new_best_found {
                debug!(
                    "ehc_search: improved h={:.4} expanded={} generated={}",
                    best_h, expanded_states, generated_states
                );
                closed.clear();
            }
        }
    }
    info!(
        "ehc_search: no solution found — expanded={}",
        expanded_states
    );
    metrics.insert("expanded_states", expanded_states.to_string());
    Ok((None, metrics))
}

/// Open-list entry for `novbfs_search`: the numeric-novelty tie-break chain
/// `(novelty, h^add, ±g)`, then `todo_len` (fewer durative actions in
/// flight first, matching every other search's `PrioritizedItem`), then an
/// `idx` insertion-order tie-break for determinism. `todo_len` is inert on
/// classical problems -- `State::todo` is only ever populated on the
/// temporal path -- so it only breaks otherwise-real ties among temporal
/// states.
struct NovBFSItem {
    novelty: u8,
    h: f64,
    g_key: f64,
    idx: usize,
    state: Rc<State>,
    partition: u64,
}

impl PartialEq for NovBFSItem {
    fn eq(&self, _other: &Self) -> bool {
        false
    }
}

impl Eq for NovBFSItem {}

impl PartialOrd for NovBFSItem {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for NovBFSItem {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // `BinaryHeap` is a max-heap; comparing `other` against `self` (not
        // `self` against `other`) inverts every field so `pop()` returns
        // the item with the lexicographically smallest
        // `(novelty, h, g_key, todo_len, idx)` tuple first -- same
        // convention as `PrioritizedItem` above. `f64` isn't `Ord`, hence
        // `total_cmp` in place of a plain tuple comparison.
        other
            .novelty
            .cmp(&self.novelty)
            .then_with(|| other.h.total_cmp(&self.h))
            .then_with(|| other.g_key.total_cmp(&self.g_key))
            .then_with(|| other.state.todo.len().cmp(&self.state.todo.len()))
            .then_with(|| other.idx.cmp(&self.idx))
    }
}

/// Strategy for `novbfs_search`: numeric-novelty scoring via `NumericNovelty`,
/// plus the `prefer_higher_g` cost-tie-break sign flip.
struct NovBFSStrategy<'a> {
    novelty: &'a mut NumericNovelty,
    prefer_higher_g: bool,
}

impl NovBFSStrategy<'_> {
    fn g_key(&self, g: f64) -> f64 {
        if self.prefer_higher_g {
            -g
        } else {
            g
        }
    }
}

impl SearchStrategy for NovBFSStrategy<'_> {
    type Payload = Rc<State>;
    type Item = NovBFSItem;

    fn state_of(item: &Self::Item) -> &Rc<State> {
        &item.state
    }

    fn root(&mut self, s: Rc<State>, h: f64) -> PyResult<Self::Item> {
        let partition = self.novelty.start(h);
        // Seeds the tables (return discarded); the root's *stored* novelty
        // is hard-coded to 1 below regardless.
        self.novelty.eval(&s, partition, None, None)?;
        let g_key = self.g_key(s.g);
        Ok(NovBFSItem {
            novelty: 1,
            h,
            g_key,
            idx: 0,
            state: s,
            partition,
        })
    }

    fn begin_expansion(&mut self) {
        self.novelty.begin_expansion();
    }

    fn child(
        &mut self,
        parent: &Self::Item,
        s: Rc<State>,
        h: f64,
        idx: usize,
    ) -> PyResult<Self::Item> {
        let partition = self.novelty.partition_of(h);
        let novelty =
            self.novelty
                .eval(&s, partition, Some(&parent.state), Some(parent.partition))?;
        let g_key = self.g_key(s.g);
        Ok(NovBFSItem {
            novelty,
            h,
            g_key,
            idx,
            state: s,
            partition,
        })
    }
}

/// A single open list ordered lexicographically on
/// `(novelty, h^add, ±g, todo_len)` -- numeric novelty first, `h^add` only
/// as a tie-breaker, plan cost `g` next, and the count of durative actions
/// in flight (fewer first) as a final tie-break before insertion order --
/// see `NovBFSItem`. That last key is a no-op on classical
/// problems, where `State::todo` is always empty.
/// `prefer_higher_g=true` is `novbfs_hg` (cost-*maximizing* final
/// tie-break, to dive into longer committed plans and find *a* solution
/// fast); `prefer_higher_g=false` is `novbfs_lg` (cost-*minimizing*).
/// `heuristic` must be an h^add instance -- see
/// `TamerLite._solve_ground_problem`'s novbfs dispatch branch, which always
/// builds one internally regardless of the configured heuristic.
///
/// `novelty` must not have had `start()` called yet -- this function calls
/// it once, on the initial state, and expects a fresh instance per search
/// call (including per anytime cold-restart iteration, which tamerlite
/// already implements generically at the Python `engine.py` level, so no
/// restart logic needs to live here).
///
/// Dedup is the standard tamerlite "generate-once" strategy shared with
/// every other search here (`WeakEqState`/`visited_states`, closed at
/// generation time), not g-based reopening: a state re-reached later via a
/// strictly cheaper path is simply dropped rather than re-queued and
/// re-scored against the novelty tables.
#[allow(clippy::too_many_arguments)]
pub fn novbfs_search<H: HeuristicTrait, S: SearchSpaceTrait>(
    ss: &S,
    heuristic: &H,
    novelty: &mut NumericNovelty,
    prefer_higher_g: bool,
    timeout: Option<f32>,
    early_termination: bool,
    weak_equality: bool,
) -> PyResult<SearchResult> {
    info!(
        "novbfs_search: prefer_higher_g={} timeout={:?} early_termination={} weak_equality={}",
        prefer_higher_g, timeout, early_termination, weak_equality
    );
    let mut strategy = NovBFSStrategy {
        novelty,
        prefer_higher_g,
    };
    let mut open: BinaryHeap<NovBFSItem> = BinaryHeap::new();
    let mut dedup = HashSetDedup::new(ss, weak_equality);
    priority_search(
        ss,
        heuristic,
        timeout,
        early_termination,
        "novbfs_search",
        &mut strategy,
        &mut open,
        &mut dedup,
    )
}
