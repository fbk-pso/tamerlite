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
use std::rc::Rc;
use std::time::SystemTime;
use std::{collections::BinaryHeap, vec::Vec};

use fastbloom::BloomFilter;
use foldhash::fast::RandomState;
use rustc_hash::{FxBuildHasher, FxHashMap, FxHashSet};

use pyo3::exceptions::PyTimeoutError;
use pyo3::prelude::*;

use super::heuristics::*;
use super::search_space::*;
use super::search_state::*;
use super::structures::Action;
use super::utils::PersistentList;

pub type SearchResult = (Option<Vec<Action>>, FxHashMap<String, String>);

trait HasTodoLen {
    fn todo_len(&self) -> usize;
}

impl HasTodoLen for State {
    fn todo_len(&self) -> usize {
        self.todo.len()
    }
}

impl HasTodoLen for Rc<State> {
    fn todo_len(&self) -> usize {
        self.todo.len()
    }
}

struct PrioritizedItem<T: HasTodoLen> {
    heuristic: f64,
    state: T,
    idx: usize,
}

impl<T: HasTodoLen> PartialEq for PrioritizedItem<T> {
    fn eq(&self, _other: &Self) -> bool {
        false
    }
}

impl<T: HasTodoLen> Eq for PrioritizedItem<T> {}

impl<T: HasTodoLen> PartialOrd for PrioritizedItem<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: HasTodoLen> Ord for PrioritizedItem<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        if self.heuristic < other.heuristic {
            std::cmp::Ordering::Greater
        } else if self.heuristic > other.heuristic {
            std::cmp::Ordering::Less
        } else if self.state.todo_len() < other.state.todo_len() {
            std::cmp::Ordering::Greater
        } else if self.state.todo_len() > other.state.todo_len() {
            std::cmp::Ordering::Less
        } else if self.idx < other.idx {
            std::cmp::Ordering::Greater
        } else {
            std::cmp::Ordering::Less
        }
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

pub fn extract_path(state: &State) -> Vec<Action> {
    PersistentList::to_vec(&state.path)
        .into_iter()
        .map(|(a, _, _)| *a)
        .collect()
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
    let mut metrics = FxHashMap::with_hasher(FxBuildHasher);
    let start = SystemTime::now();
    let init = Rc::new(ss.initial_state(None)?);
    let mut expanded_states = 0;
    let mut generated_states = 1;
    if early_termination && ss.goal_reached(&init, None)? {
        metrics.insert("expanded_states".to_string(), expanded_states.to_string());
        metrics.insert("goal_depth".to_string(), init.g.to_string());
        return Ok((Some(extract_path(&init)), metrics));
    }

    // State and WeakEqState contain interior mutability only for heuristic caches.
    // The mutable fields are ignored by Hash/Eq, so using them as HashSet keys is safe.
    #[allow(clippy::mutable_key_type)]
    let mut visited_weak_eq_states = FxHashSet::with_hasher(FxBuildHasher);
    #[allow(clippy::mutable_key_type)]
    let mut visited_states = FxHashSet::with_hasher(FxBuildHasher);
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
            return Ok((None, metrics));
        }
    };
    let mut open = BinaryHeap::new();
    open.push(PrioritizedItem {
        heuristic: init_h,
        state: init,
        idx: 0,
    });
    while let Some(current) = open.pop() {
        if let Some(t) = timeout {
            if start.elapsed().unwrap().as_secs_f32() > t {
                return Err(PyTimeoutError::new_err("Timeout"));
            }
        }
        let state = current.state;
        expanded_states += 1;
        if expanded_states % 10_000 == 0 {
            debug!(
                "wastar_search: expanded={} generated={} open={}",
                expanded_states,
                generated_states,
                open.len()
            );
        }
        if !early_termination && ss.goal_reached(&state, None)? {
            info!(
                "wastar_search: goal found — expanded={} depth={}",
                expanded_states, state.g
            );
            metrics.insert("expanded_states".to_string(), expanded_states.to_string());
            metrics.insert("goal_depth".to_string(), state.g.to_string());
            return Ok((Some(extract_path(&state)), metrics));
        } else {
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
                        keep.then_some(Ok(s))
                    }
                    Err(e) => Some(Err(e)),
                });

            for rs in heuristic.eval_gen(successors_iter, ss)? {
                let (s, h) = rs?;
                if early_termination && ss.goal_reached(&s, None)? {
                    info!(
                        "wastar_search: goal found — expanded={} depth={}",
                        expanded_states, s.g
                    );
                    metrics.insert("expanded_states".to_string(), expanded_states.to_string());
                    metrics.insert("goal_depth".to_string(), s.g.to_string());
                    return Ok((Some(extract_path(&s)), metrics));
                }
                if let Some(v) = h {
                    let f = weight * v + (1.0 - weight) * s.g;
                    open.push(PrioritizedItem {
                        heuristic: f,
                        state: s,
                        idx: generated_states,
                    });
                }
                generated_states += 1;
            }
        }
    }
    info!(
        "wastar_search: no solution found — expanded={}",
        expanded_states
    );
    metrics.insert("expanded_states".to_string(), expanded_states.to_string());
    Ok((None, metrics))
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
    let mut metrics = FxHashMap::with_hasher(FxBuildHasher);
    let start = SystemTime::now();
    let init = ss.initial_state(None)?;
    let mut expanded_states = 0;
    let mut generated_states = 1;
    if early_termination && ss.goal_reached(&init, None)? {
        metrics.insert("expanded_states".to_string(), expanded_states.to_string());
        metrics.insert("goal_depth".to_string(), init.g.to_string());
        return Ok((Some(extract_path(&init)), metrics));
    }

    let mut visited_states: Option<BloomFilter<RandomState>> = if !ss.is_temporal() || weak_equality
    {
        const BLOOM_ITEMS: usize = 20_000_000;
        const BLOOM_FP_RATE: f64 = 1e-4;
        let mut visited_states = BloomFilter::with_false_pos(BLOOM_FP_RATE)
            .hasher(RandomState::default())
            .expected_items(BLOOM_ITEMS);
        visited_states.insert(&init.assignments);
        Some(visited_states)
    } else {
        None
    };

    let init_h = match heuristic.eval(&init, ss)? {
        Some(v) => v,
        None => {
            metrics.insert("expanded_states".to_string(), 0.to_string());
            return Ok((None, metrics));
        }
    };

    const QUEUE_BOUND: usize = 400_000;
    let mut open = BoundedPriorityQueue::with_bound(QUEUE_BOUND);
    open.push(PrioritizedItem {
        heuristic: init_h,
        state: init,
        idx: 0,
    });
    while let Some(current) = open.pop() {
        if let Some(t) = timeout {
            if start.elapsed().unwrap().as_secs_f32() > t {
                return Err(PyTimeoutError::new_err("Timeout"));
            }
        }
        let state = current.state;
        expanded_states += 1;
        if expanded_states % 10_000 == 0 {
            debug!(
                "wastar_search_memory_bounded: expanded={} generated={} open={}",
                expanded_states,
                generated_states,
                open.len()
            );
        }
        if !early_termination && ss.goal_reached(&state, None)? {
            info!(
                "wastar_search_memory_bounded: goal found — expanded={} depth={}",
                expanded_states, state.g
            );
            metrics.insert("expanded_states".to_string(), expanded_states.to_string());
            metrics.insert("goal_depth".to_string(), state.g.to_string());
            return Ok((Some(extract_path(&state)), metrics));
        } else {
            let successors_iter = ss
                .get_successor_states_iter(&state)
                .filter_map(|rs| match rs {
                    Ok(s) => {
                        let keep = if let Some(ref mut visited) = visited_states {
                            !visited.insert(&s.assignments)
                        } else {
                            true
                        };
                        keep.then_some(Ok(s))
                    }
                    Err(e) => Some(Err(e)),
                });

            for rs in heuristic.eval_gen_owned(successors_iter, ss)? {
                let (s, h) = rs?;
                if early_termination && ss.goal_reached(&s, None)? {
                    info!(
                        "wastar_search_memory_bounded: goal found — expanded={} depth={}",
                        expanded_states, s.g
                    );
                    metrics.insert("expanded_states".to_string(), expanded_states.to_string());
                    metrics.insert("goal_depth".to_string(), s.g.to_string());
                    return Ok((Some(extract_path(&s)), metrics));
                }
                if let Some(v) = h {
                    let f = weight * v + (1.0 - weight) * s.g;
                    open.push(PrioritizedItem {
                        heuristic: f,
                        state: s,
                        idx: generated_states,
                    });
                }
                generated_states += 1;
            }
        }
    }
    info!(
        "wastar_search_memory_bounded: no solution found — expanded={}",
        expanded_states
    );
    metrics.insert("expanded_states".to_string(), expanded_states.to_string());
    Ok((None, metrics))
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
        metrics.insert("expanded_states".to_string(), expanded_states.to_string());
        metrics.insert("goal_depth".to_string(), init.g.to_string());
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
            metrics.insert("expanded_states".to_string(), expanded_states.to_string());
            metrics.insert("goal_depth".to_string(), state.g.to_string());
            return Ok((Some(extract_path(&state)), metrics));
        } else {
            for rs in ss.get_successor_states_iter(&state) {
                let s = rs?;
                if early_termination && ss.goal_reached(&s, None)? {
                    info!(
                        "{}: goal found — expanded={} depth={}",
                        name, expanded_states, s.g
                    );
                    metrics.insert("expanded_states".to_string(), expanded_states.to_string());
                    metrics.insert("goal_depth".to_string(), s.g.to_string());
                    return Ok((Some(extract_path(&s)), metrics));
                }
                open.push_back(s);
                generated_states += 1;
            }
        }
    }
    info!("{}: no solution found — expanded={}", name, expanded_states);
    metrics.insert("expanded_states".to_string(), expanded_states.to_string());
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
        metrics.insert("expanded_states".to_string(), expanded_states.to_string());
        metrics.insert("goal_depth".to_string(), init.g.to_string());
        return Ok((Some(extract_path(&init)), metrics));
    }

    let mut best_h = match heuristic.eval(&init, ss)? {
        Some(v) => v,
        None => {
            metrics.insert("expanded_states".to_string(), 0.to_string());
            return Ok((None, metrics));
        }
    };
    debug!("ehc_search: initial h={:.4}", best_h);
    let mut open = VecDeque::new();
    open.push_back(init);

    // State and WeakEqState contain interior mutability only for heuristic caches.
    // The mutable fields are ignored by Hash/Eq, so using them as HashSet keys is safe.
    #[allow(clippy::mutable_key_type)]
    let mut closed = FxHashSet::with_hasher(FxBuildHasher);
    #[allow(clippy::mutable_key_type)]
    let mut closed_weak_eq = FxHashSet::with_hasher(FxBuildHasher);
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
            metrics.insert("expanded_states".to_string(), expanded_states.to_string());
            metrics.insert("goal_depth".to_string(), state.g.to_string());
            return Ok((Some(extract_path(&state)), metrics));
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
                generated_states += 1;
                if early_termination && ss.goal_reached(&s, None)? {
                    info!(
                        "ehc_search: goal found — expanded={} depth={}",
                        expanded_states, s.g
                    );
                    metrics.insert("expanded_states".to_string(), expanded_states.to_string());
                    metrics.insert("goal_depth".to_string(), s.g.to_string());
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
                closed_weak_eq.clear();
            }
        }
    }
    info!(
        "ehc_search: no solution found — expanded={}",
        expanded_states
    );
    metrics.insert("expanded_states".to_string(), expanded_states.to_string());
    Ok((None, metrics))
}
