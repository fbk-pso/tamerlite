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

use std::collections::VecDeque;
use std::hash::{Hash, Hasher};
use std::rc::Rc;
use std::time::SystemTime;
use std::{collections::BinaryHeap, vec::Vec};

use fastbloom::BloomFilter;
use foldhash::fast::RandomState;
use min_max_heap::MinMaxHeap;
use rustc_hash::{FxBuildHasher, FxHashMap, FxHashSet};

use pyo3::exceptions::PyTimeoutError;
use pyo3::prelude::*;

use super::heuristics::*;
use super::search_space::*;
use super::search_state::*;
use super::structures::Action;
use super::utils::PersistentList;

pub type Plan = Vec<(Option<String>, Action, Option<String>)>;

pub struct AnytimeSearchResult {
    pub plan: Option<Plan>,
    pub metrics: FxHashMap<String, String>,
    pub plans: Option<Vec<Vec<Action>>>,
    pub timed_out: bool,
}

#[derive(Debug)]
struct PrioritizedItem {
    heuristic: f64,
    state: Rc<State>,
}

impl PartialEq for PrioritizedItem {
    fn eq(&self, other: &Self) -> bool {
        self.heuristic == other.heuristic && self.state.todo.len() == other.state.todo.len()
    }
}

impl Eq for PrioritizedItem {}

impl PartialOrd for PrioritizedItem {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PrioritizedItem {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        if self.heuristic < other.heuristic {
            std::cmp::Ordering::Greater
        } else if self.heuristic > other.heuristic {
            std::cmp::Ordering::Less
        } else if self.state.todo.len() < other.state.todo.len() {
            std::cmp::Ordering::Greater
        } else {
            std::cmp::Ordering::Less
        }
    }
}

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
    if state1.todo.len() != state2.todo.len() || state1.assignments != state2.assignments {
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

pub fn build_plan<S: SearchSpaceTrait>(
    ss: &S,
    state: &State,
) -> PyResult<Option<Vec<(Option<String>, Action, Option<String>)>>> {
    let plan = ss.build_plan(state)?;
    let mut res = Vec::with_capacity(plan.len());
    for (s, a, d) in plan.into_iter() {
        let mut ss = None;
        let mut ds = None;
        if let Some(start) = s {
            ss = Some(format!(
                "{}/{}",
                start.numer().to_string(),
                start.denom().to_string()
            ));
        }
        if let Some(duration) = d {
            ds = Some(format!(
                "{}/{}",
                duration.numer().to_string(),
                duration.denom().to_string()
            ));
        }
        res.push((ss, a, ds));
    }
    Ok(Some(res))
}

pub fn wastar_search<H: HeuristicTrait, S: SearchSpaceTrait>(
    ss: &S,
    heuristic: &H,
    weight: f64,
    timeout: Option<f32>,
    early_termination: bool,
    weak_equality: bool,
    max_len: Option<f64>,
) -> PyResult<AnytimeSearchResult> {
    let mut metrics = FxHashMap::with_hasher(FxBuildHasher::default());
    let start = SystemTime::now();
    let init = Rc::new(ss.initial_state(None)?);
    let mut expanded_states = 0;
    if early_termination && ss.goal_reached(&init, None)? {
        metrics.insert("expanded_states".to_string(), expanded_states.to_string());
        metrics.insert("goal_depth".to_string(), init.g.to_string());
        return build_plan(ss, &init).map(|plan| AnytimeSearchResult {
            plan,
            metrics,
            plans: None,
            timed_out: false,
        });
    }

    let mut visited_weak_eq_states = FxHashSet::with_hasher(FxBuildHasher::default());
    let mut visited_states = FxHashSet::with_hasher(FxBuildHasher::default());
    if !ss.is_temporal() {
        visited_states.insert(Rc::clone(&init));
    } else if weak_equality {
        visited_weak_eq_states.insert(WeakEqState {
            state: Rc::clone(&init),
        });
    }

    let init_h = match heuristic.eval(&init, ss)? {
        Some(v) => v,
        None => {
            metrics.insert("expanded_states".to_string(), 0.to_string());
            return Ok(AnytimeSearchResult {
                plan: None,
                metrics,
                plans: max_len.map(|_| Vec::new()),
                timed_out: false,
            });
        }
    };
    let mut open = BinaryHeap::new();
    open.push(PrioritizedItem {
        heuristic: init_h,
        state: init,
    });
    let mut plans = Vec::new();
    let mut current_max_len = max_len;
    while let Some(current) = open.pop() {
        if let Some(t) = timeout {
            if start.elapsed().unwrap().as_secs_f32() > t {
                if max_len.is_some() {
                    metrics.insert("expanded_states".to_string(), expanded_states.to_string());
                    return Ok(AnytimeSearchResult {
                        plan: None,
                        metrics,
                        plans: Some(plans),
                        timed_out: true,
                    });
                }
                return Err(PyTimeoutError::new_err("Timeout"));
            }
        }
        let state = current.state;
        expanded_states += 1;
        if !early_termination && ss.goal_reached(&state, None)? {
            if let Some(bound) = current_max_len {
                let path = PersistentList::to_vec_copy(&state.path);
                if (path.len() as f64) <= bound {
                    println!("{}) Found plan of length {}", plans.len(), path.len());
                    plans.push(path.into_iter().map(|(action, _, _)| action).collect());
                    if bound.is_infinite() {
                        current_max_len = Some(plans.last().unwrap().len() as f64);
                    }
                }
            } else {
                metrics.insert("expanded_states".to_string(), expanded_states.to_string());
                metrics.insert("goal_depth".to_string(), state.g.to_string());
                return build_plan(ss, &state).map(|plan| AnytimeSearchResult {
                    plan,
                    metrics,
                    plans: None,
                    timed_out: false,
                });
            }
        }

        {
            let successors_iter = ss
                .get_successor_states_iter(&state)
                .filter_map(|rs| match rs {
                    Ok(s) => {
                        let s = Rc::new(s);
                        let keep = if !ss.is_temporal() {
                            visited_states.insert(Rc::clone(&s))
                        } else if weak_equality {
                            visited_weak_eq_states.insert(WeakEqState {
                                state: Rc::clone(&s),
                            })
                        } else {
                            true
                        };
                        let within_bound = current_max_len.map_or(true, |bound| s.g <= bound);
                        (keep && within_bound).then_some(Ok(s))
                    }
                    Err(e) => Some(Err(e)),
                });

            for rs in heuristic.eval_gen(successors_iter, ss)? {
                let (s, h) = rs?;
                if early_termination && ss.goal_reached(&s, None)? {
                    metrics.insert("expanded_states".to_string(), expanded_states.to_string());
                    metrics.insert("goal_depth".to_string(), s.g.to_string());
                    return build_plan(ss, &s).map(|plan| AnytimeSearchResult {
                        plan,
                        metrics,
                        plans: None,
                        timed_out: false,
                    });
                }
                match h {
                    Some(v) => {
                        let f = weight * v + (1.0 - weight) * s.g;
                        open.push(PrioritizedItem {
                            heuristic: f,
                            state: s,
                        });
                    }
                    None => {}
                }
            }
        }
    }
    metrics.insert("expanded_states".to_string(), expanded_states.to_string());
    Ok(AnytimeSearchResult {
        plan: None,
        metrics,
        plans: max_len.map(|_| plans),
        timed_out: false,
    })
}

/// Fixed capacity of the open list used by the memory-bounded searches.
/// Once full, a new state is only admitted if it is strictly better than the
/// current worst state in the queue, which is then evicted.
const MEMORY_BOUNDED_QUEUE_BOUND: usize = 400_000;
/// Expected number of items for the Bloom filter used to (probabilistically)
/// deduplicate visited states in the memory-bounded searches.
const MEMORY_BOUNDED_BLOOM_ITEMS: usize = 20_000_000;
/// Target false-positive rate for the Bloom filter above.
const MEMORY_BOUNDED_BLOOM_FP_RATE: f64 = 1e-4;

struct BoundedPrioritizedItem {
    heuristic: f64,
    state: State,
    idx: usize,
}

impl PartialEq for BoundedPrioritizedItem {
    fn eq(&self, _other: &Self) -> bool {
        false
    }
}

impl Eq for BoundedPrioritizedItem {}

impl PartialOrd for BoundedPrioritizedItem {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for BoundedPrioritizedItem {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        if self.heuristic < other.heuristic {
            std::cmp::Ordering::Greater
        } else if self.heuristic > other.heuristic {
            std::cmp::Ordering::Less
        } else if self.state.todo.len() < other.state.todo.len() {
            std::cmp::Ordering::Greater
        } else if self.state.todo.len() > other.state.todo.len() {
            std::cmp::Ordering::Less
        } else if self.idx < other.idx {
            std::cmp::Ordering::Greater
        } else {
            std::cmp::Ordering::Less
        }
    }
}

/// A priority queue with a fixed capacity: once full, pushing a new item only
/// succeeds if the item is strictly better than the current worst item in the
/// queue, which is evicted to make room. This bounds the queue's memory usage
/// at the cost of completeness, since worse states are permanently discarded
/// instead of being kept around for later expansion.
struct BoundedPriorityQueue<T: Ord> {
    heap: MinMaxHeap<T>,
    bound: usize,
}

impl<T: Ord> BoundedPriorityQueue<T> {
    fn with_bound(bound: usize) -> Self {
        assert!(bound > 0, "bound must be positive");
        Self {
            heap: MinMaxHeap::with_capacity(bound),
            bound,
        }
    }

    /// Push an item only if the heap is under capacity, or the item is
    /// better than the current minimum. Returns false if the item was rejected.
    fn push(&mut self, item: T) -> bool {
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

    fn pop(&mut self) -> Option<T> {
        self.heap.pop_max()
    }
}

/// Memory-bounded variant of [`wastar_search`], trading completeness for a
/// hard memory ceiling. Two changes from the unbounded search:
/// - the open list is a [`BoundedPriorityQueue`] instead of an unbounded
///   `BinaryHeap`, so once full, only strictly-better-than-worst states are
///   admitted and worse ones are permanently discarded;
/// - the visited-state set (when used) is a probabilistic Bloom filter
///   instead of an exact hash set, so a genuinely new state may rarely be
///   (incorrectly) treated as already visited.
/// Both changes mean this search may miss a solution that the unbounded
/// search would find. It is intended for large/non-temporal problems whose
/// relaxed state space can otherwise grow without bound.
/// Like [`wastar_search`], `max_len` still enables the anytime, multi-plan
/// mode: the search keeps expanding states (including past goal states)
/// looking for further plans of length at most `max_len`, up to `timeout`.
pub fn wastar_search_memory_bounded<H: HeuristicTrait, S: SearchSpaceTrait>(
    ss: &S,
    heuristic: &H,
    weight: f64,
    timeout: Option<f32>,
    early_termination: bool,
    weak_equality: bool,
    max_len: Option<f64>,
) -> PyResult<AnytimeSearchResult> {
    let mut metrics = FxHashMap::with_hasher(FxBuildHasher::default());
    let start = SystemTime::now();
    let init = ss.initial_state(None)?;
    let mut expanded_states = 0;
    let mut generated_states: usize = 1;

    if early_termination && ss.goal_reached(&init, None)? {
        metrics.insert("expanded_states".to_string(), expanded_states.to_string());
        metrics.insert("goal_depth".to_string(), init.g.to_string());
        return build_plan(ss, &init).map(|plan| AnytimeSearchResult {
            plan,
            metrics,
            plans: None,
            timed_out: false,
        });
    }

    let mut visited_states: Option<BloomFilter<RandomState>> =
        if !ss.is_temporal() || weak_equality {
            let mut visited_states = BloomFilter::with_false_pos(MEMORY_BOUNDED_BLOOM_FP_RATE)
                .hasher(RandomState::default())
                .expected_items(MEMORY_BOUNDED_BLOOM_ITEMS);
            visited_states.insert(&init.assignments);
            Some(visited_states)
        } else {
            None
        };

    let init_h = match heuristic.eval(&init, ss)? {
        Some(v) => v,
        None => {
            metrics.insert("expanded_states".to_string(), 0.to_string());
            return Ok(AnytimeSearchResult {
                plan: None,
                metrics,
                plans: max_len.map(|_| Vec::new()),
                timed_out: false,
            });
        }
    };

    let mut open = BoundedPriorityQueue::with_bound(MEMORY_BOUNDED_QUEUE_BOUND);
    open.push(BoundedPrioritizedItem {
        heuristic: init_h,
        state: init,
        idx: 0,
    });

    let mut plans = Vec::new();
    let mut current_max_len = max_len;
    while let Some(current) = open.pop() {
        if let Some(t) = timeout {
            if start.elapsed().unwrap().as_secs_f32() > t {
                if max_len.is_some() {
                    metrics.insert("expanded_states".to_string(), expanded_states.to_string());
                    return Ok(AnytimeSearchResult {
                        plan: None,
                        metrics,
                        plans: Some(plans),
                        timed_out: true,
                    });
                }
                return Err(PyTimeoutError::new_err("Timeout"));
            }
        }
        let state = current.state;
        expanded_states += 1;
        if !early_termination && ss.goal_reached(&state, None)? {
            if let Some(bound) = current_max_len {
                let path = PersistentList::to_vec_copy(&state.path);
                if (path.len() as f64) <= bound {
                    println!("{}) Found plan of length {}", plans.len(), path.len());
                    plans.push(path.into_iter().map(|(action, _, _)| action).collect());
                    if bound.is_infinite() {
                        current_max_len = Some(plans.last().unwrap().len() as f64);
                    }
                }
            } else {
                metrics.insert("expanded_states".to_string(), expanded_states.to_string());
                metrics.insert("goal_depth".to_string(), state.g.to_string());
                return build_plan(ss, &state).map(|plan| AnytimeSearchResult {
                    plan,
                    metrics,
                    plans: None,
                    timed_out: false,
                });
            }
        }

        let successors_iter = ss
            .get_successor_states_iter(&state)
            .filter_map(|rs| match rs {
                Ok(s) => {
                    let keep = if let Some(ref mut visited) = visited_states {
                        !visited.insert(&s.assignments)
                    } else {
                        true
                    };
                    let within_bound = current_max_len.map_or(true, |bound| s.g <= bound);
                    (keep && within_bound).then_some(Ok(s))
                }
                Err(e) => Some(Err(e)),
            });

        for rs in heuristic.eval_gen_owned(successors_iter, ss)? {
            let (s, h) = rs?;
            if early_termination && ss.goal_reached(&s, None)? {
                metrics.insert("expanded_states".to_string(), expanded_states.to_string());
                metrics.insert("goal_depth".to_string(), s.g.to_string());
                return build_plan(ss, &s).map(|plan| AnytimeSearchResult {
                    plan,
                    metrics,
                    plans: None,
                    timed_out: false,
                });
            }
            if let Some(v) = h {
                let f = weight * v + (1.0 - weight) * s.g;
                open.push(BoundedPrioritizedItem {
                    heuristic: f,
                    state: s,
                    idx: generated_states,
                });
            }
            generated_states += 1;
        }
    }
    metrics.insert("expanded_states".to_string(), expanded_states.to_string());
    Ok(AnytimeSearchResult {
        plan: None,
        metrics,
        plans: max_len.map(|_| plans),
        timed_out: false,
    })
}

pub fn bfs_search<S: SearchSpaceTrait>(
    ss: &S,
    timeout: Option<f32>,
    early_termination: bool,
) -> PyResult<(
    Option<Vec<(Option<String>, Action, Option<String>)>>,
    FxHashMap<String, String>,
)> {
    basic_search(ss, true, timeout, early_termination)
}

pub fn dfs_search<S: SearchSpaceTrait>(
    ss: &S,
    timeout: Option<f32>,
    early_termination: bool,
) -> PyResult<(
    Option<Vec<(Option<String>, Action, Option<String>)>>,
    FxHashMap<String, String>,
)> {
    basic_search(ss, false, timeout, early_termination)
}

fn basic_search<S: SearchSpaceTrait>(
    ss: &S,
    bfs: bool,
    timeout: Option<f32>,
    early_termination: bool,
) -> PyResult<(
    Option<Vec<(Option<String>, Action, Option<String>)>>,
    FxHashMap<String, String>,
)> {
    let mut metrics = FxHashMap::with_hasher(FxBuildHasher::default());
    let start = SystemTime::now();
    let init = ss.initial_state(None)?;
    let mut open = VecDeque::new();
    let mut expanded_states = 0;

    if early_termination && ss.goal_reached(&init, None)? {
        metrics.insert("expanded_states".to_string(), expanded_states.to_string());
        metrics.insert("goal_depth".to_string(), init.g.to_string());
        return build_plan(ss, &init).map(|plan| (plan, metrics));
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
        if !early_termination && ss.goal_reached(&state, None)? {
            metrics.insert("expanded_states".to_string(), expanded_states.to_string());
            metrics.insert("goal_depth".to_string(), state.g.to_string());
            return build_plan(ss, &state).map(|plan| (plan, metrics));
        } else {
            for rs in ss.get_successor_states_iter(&state) {
                let s = rs?;
                if early_termination && ss.goal_reached(&s, None)? {
                    metrics.insert("expanded_states".to_string(), expanded_states.to_string());
                    metrics.insert("goal_depth".to_string(), s.g.to_string());
                    return build_plan(ss, &s).map(|plan| (plan, metrics));
                }
                open.push_back(s);
            }
        }
    }
    metrics.insert("expanded_states".to_string(), expanded_states.to_string());
    Ok((None, metrics))
}

/// Checks whether `state` is a goal state and, if so and it is within
/// `current_max_len`, records its path in `plans` (tightening the bound to
/// the plan's length if it started out infinite). Used by [`ehc_search`]'s
/// anytime mode to check every generated state for goal_reached exactly
/// once, right when it is created — EHC restarts by wiping its open/closed
/// sets whenever a strictly better successor is found, so a goal state that
/// is only checked when popped could be discarded by such a restart before
/// ever being tested.
fn log_if_goal<S: SearchSpaceTrait>(
    ss: &S,
    state: &State,
    current_max_len: &mut Option<f64>,
    plans: &mut Vec<Vec<Action>>,
) -> PyResult<()> {
    if ss.goal_reached(state, None)? {
        if let Some(bound) = *current_max_len {
            let path = PersistentList::to_vec_copy(&state.path);
            if (path.len() as f64) <= bound {
                println!("{}) Found plan of length {}", plans.len(), path.len());
                plans.push(path.into_iter().map(|(action, _, _)| action).collect());
                if bound.is_infinite() {
                    *current_max_len = Some(plans.last().unwrap().len() as f64);
                }
            }
        }
    }
    Ok(())
}

pub fn ehc_search<H: HeuristicTrait, S: SearchSpaceTrait>(
    ss: &S,
    heuristic: &H,
    timeout: Option<f32>,
    early_termination: bool,
    weak_equality: bool,
    max_len: Option<f64>,
) -> PyResult<AnytimeSearchResult> {
    let mut metrics = FxHashMap::with_hasher(FxBuildHasher::default());
    let start = SystemTime::now();
    let init = Rc::new(ss.initial_state(None)?);
    let mut expanded_states = 0;

    if early_termination && ss.goal_reached(&init, None)? {
        metrics.insert("expanded_states".to_string(), expanded_states.to_string());
        metrics.insert("goal_depth".to_string(), init.g.to_string());
        return build_plan(ss, &init).map(|plan| AnytimeSearchResult {
            plan,
            metrics,
            plans: None,
            timed_out: false,
        });
    }

    let mut best_h = match heuristic.eval(&init, ss)? {
        Some(v) => v,
        None => {
            metrics.insert("expanded_states".to_string(), 0.to_string());
            return Ok(AnytimeSearchResult {
                plan: None,
                metrics,
                plans: max_len.map(|_| Vec::new()),
                timed_out: false,
            });
        }
    };

    let mut plans = Vec::new();
    let mut current_max_len = max_len;
    if max_len.is_some() {
        if let Some(t) = timeout {
            if start.elapsed().unwrap().as_secs_f32() > t {
                metrics.insert("expanded_states".to_string(), expanded_states.to_string());
                return Ok(AnytimeSearchResult {
                    plan: None,
                    metrics,
                    plans: Some(plans),
                    timed_out: true,
                });
            }
        }
        log_if_goal(ss, &init, &mut current_max_len, &mut plans)?;
    }

    let mut open = VecDeque::new();
    open.push_back(init);
    let mut closed = FxHashSet::with_hasher(FxBuildHasher::default());
    let mut closed_weak_eq = FxHashSet::with_hasher(FxBuildHasher::default());
    while let Some(state) = open.pop_front() {
        if let Some(t) = timeout {
            if start.elapsed().unwrap().as_secs_f32() > t {
                if max_len.is_some() {
                    metrics.insert("expanded_states".to_string(), expanded_states.to_string());
                    return Ok(AnytimeSearchResult {
                        plan: None,
                        metrics,
                        plans: Some(plans),
                        timed_out: true,
                    });
                }
                return Err(PyTimeoutError::new_err("Timeout"));
            }
        }

        expanded_states += 1;
        if max_len.is_none() && !early_termination && ss.goal_reached(&state, None)? {
            metrics.insert("expanded_states".to_string(), expanded_states.to_string());
            metrics.insert("goal_depth".to_string(), state.g.to_string());
            return build_plan(ss, &state).map(|plan| AnytimeSearchResult {
                plan,
                metrics,
                plans: None,
                timed_out: false,
            });
        } else {
            if !ss.is_temporal() {
                closed.insert(Rc::clone(&state));
            } else if weak_equality {
                closed_weak_eq.insert(WeakEqState {
                    state: Rc::clone(&state),
                });
            }

            let successors_iter = ss
                .get_successor_states_iter(&state)
                .filter_map(|rs| match rs {
                    Ok(s) => {
                        let s = Rc::new(s);
                        if !ss.is_temporal() {
                            (!closed.contains(&s)).then_some(Ok(s))
                        } else if weak_equality {
                            let weak_eq_state = WeakEqState { state: s };
                            (!closed_weak_eq.contains(&weak_eq_state))
                                .then_some(Ok(weak_eq_state.state))
                        } else {
                            Some(Ok(s))
                        }
                    }
                    Err(e) => Some(Err(e)),
                });

            let mut new_best_found = false;
            for rs in heuristic.eval_gen(successors_iter, ss)? {
                let (s, h) = rs?;
                if early_termination && ss.goal_reached(&s, None)? {
                    metrics.insert("expanded_states".to_string(), expanded_states.to_string());
                    metrics.insert("goal_depth".to_string(), s.g.to_string());
                    return build_plan(ss, &s).map(|plan| AnytimeSearchResult {
                        plan,
                        metrics,
                        plans: None,
                        timed_out: false,
                    });
                }
                if max_len.is_some() {
                    log_if_goal(ss, &s, &mut current_max_len, &mut plans)?;
                }
                match h {
                    Some(v) => {
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
                    None => {}
                }
            }
            if new_best_found {
                closed.clear();
                closed_weak_eq.clear();
            }
        }
    }
    metrics.insert("expanded_states".to_string(), expanded_states.to_string());
    Ok(AnytimeSearchResult {
        plan: None,
        metrics,
        plans: max_len.map(|_| plans),
        timed_out: false,
    })
}
