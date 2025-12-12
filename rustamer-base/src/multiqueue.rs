// Copyright (C) 2025 PSO Unit, Fondazione Bruno Kessler
// This file is part of TamerLite.
//
// TamerLite is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// TamerLite is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.
//

use std::cell::RefCell;
use std::rc::Rc;
use std::time::SystemTime;
use std::{collections::BinaryHeap, vec::Vec};

use rustc_hash::{FxBuildHasher, FxHashMap, FxHashSet};

use pyo3::exceptions::PyTimeoutError;
use pyo3::prelude::*;

use super::heuristics::*;
use super::search::*;
use super::search_space::*;
use super::search_state::*;

#[derive(Debug, Clone)]
pub struct StateContainer {
    pub state: Rc<State>,
    pub expanded: Rc<RefCell<bool>>,
}

impl StateContainer {
    fn set_expanded(&self, expanded: bool) -> () {
        *self.expanded.borrow_mut() = expanded;
    }
}

#[derive(Debug, Clone)]
pub struct PrioritizedItem {
    pub heuristic: f64,
    pub state_container: StateContainer,
}

impl PartialEq for PrioritizedItem {
    fn eq(&self, other: &Self) -> bool {
        self.heuristic == other.heuristic
            && self.state_container.state.todo.len() == other.state_container.state.todo.len()
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
        } else if self.state_container.state.todo.len() < other.state_container.state.todo.len() {
            std::cmp::Ordering::Greater
        } else {
            std::cmp::Ordering::Less
        }
    }
}

/// Trait for multi-queue switching policies.
pub trait MQSwitchPolicy {
    /// Given the number of expansions done so far, return the index of the next
    /// queue to use
    fn switching_policy(&mut self, i: usize) -> usize;

    /// Notify the policy that an item has been pushed to queue `i`
    fn notify_push(&mut self, i: usize, item: &PrioritizedItem);

    /// Notify the policy that an item has been popped from queue `i` (and
    /// marked as expanded in all other queues)
    fn notify_pop(&mut self, i: usize, item: &PrioritizedItem);
}

/// A simple round-robin switching policy.
struct RoundRobinSwitchPolicy {
    num_queues: usize,
}

impl RoundRobinSwitchPolicy {
    pub fn new(num_queues: usize) -> Self {
        Self { num_queues }
    }
}

impl MQSwitchPolicy for RoundRobinSwitchPolicy {
    fn switching_policy(&mut self, i: usize) -> usize {
        i % self.num_queues
    }

    fn notify_push(&mut self, _i: usize, _item: &PrioritizedItem) {}

    fn notify_pop(&mut self, _i: usize, _item: &PrioritizedItem) {}
}

pub fn multiqueue_search<H: HeuristicTrait, S: SearchSpaceTrait>(
    ss: &S,
    heuristics: Vec<(H, f64)>,
    timeout: Option<f32>,
    early_termination: bool,
    weak_equality: bool,
) -> PyResult<(
    Option<Vec<(Option<String>, String, Option<String>)>>,
    FxHashMap<String, String>,
)> {
    let mut switch_policy = RoundRobinSwitchPolicy::new(heuristics.len());
    _multiqueue_search(
        ss,
        heuristics,
        &mut switch_policy,
        timeout,
        early_termination,
        weak_equality,
    )
}

pub fn _multiqueue_search<T: MQSwitchPolicy, H: HeuristicTrait, S: SearchSpaceTrait>(
    ss: &S,
    heuristics: Vec<(H, f64)>,
    switch_policy: &mut T,
    timeout: Option<f32>,
    early_termination: bool,
    weak_equality: bool,
) -> PyResult<(
    Option<Vec<(Option<String>, String, Option<String>)>>,
    FxHashMap<String, String>,
)> {
    let mut metrics = FxHashMap::with_hasher(FxBuildHasher::default());
    let start = SystemTime::now();
    let init = Rc::new(ss.initial_state(None)?);
    let mut states_expanded = 0;
    if early_termination && ss.goal_reached(&init, None)? {
        metrics.insert("expanded_states".to_string(), states_expanded.to_string());
        metrics.insert("goal_depth".to_string(), init.g.to_string());
        return build_plan(ss, &init).map(|plan| (plan, metrics));
    }

    let item = PrioritizedItem {
        heuristic: 0.0,
        state_container: StateContainer {
            state: init,
            expanded: Rc::new(RefCell::new(false)),
        },
    };

    let mut visited_weak_eq_states = FxHashSet::with_hasher(FxBuildHasher::default());
    let mut visited_states = FxHashSet::with_hasher(FxBuildHasher::default());
    if !ss.is_temporal() {
        visited_states.insert(Rc::clone(&item.state_container.state));
    } else if weak_equality {
        visited_weak_eq_states.insert(VisitedState {
            state: Rc::clone(&item.state_container.state),
        });
    }

    let mut opens = Vec::with_capacity(heuristics.len());
    for (i, _) in heuristics.iter().enumerate() {
        let mut open = BinaryHeap::new();
        open.push(item.clone());
        opens.push(open);
        switch_policy.notify_push(i, &item);
    }

    loop {
        if let Some(t) = timeout {
            if start.elapsed().unwrap().as_secs_f32() > t {
                return Err(PyTimeoutError::new_err("Timeout"));
            }
        }
        // If one of the queues is empty, then all the others are (logically) empty too
        if opens.iter().any(|o| o.is_empty()) {
            break;
        }

        let i = switch_policy.switching_policy(states_expanded);
        let open = &mut opens[i];
        if let Some(current) = open.pop() {
            switch_policy.notify_pop(i, &current);
            if *current.state_container.expanded.borrow() {
                continue;
            }
            current.state_container.set_expanded(true);
            let state = &current.state_container.state;
            states_expanded += 1;
            if !early_termination && ss.goal_reached(&state, None)? {
                metrics.insert("expanded_states".to_string(), states_expanded.to_string());
                metrics.insert("goal_depth".to_string(), state.g.to_string());
                return build_plan(ss, &state).map(|plan| (plan, metrics));
            }

            let mut candidate_containers: Vec<StateContainer> = Vec::new();
            for rs in ss.get_successor_states_iter(&state) {
                let s = rs?;
                if early_termination && ss.goal_reached(&s, None)? {
                    metrics.insert("expanded_states".to_string(), states_expanded.to_string());
                    metrics.insert("goal_depth".to_string(), s.g.to_string());
                    return build_plan(ss, &s).map(|plan| (plan, metrics));
                }
                let s = Rc::new(s);
                let keep = if !ss.is_temporal() {
                    visited_states.insert(Rc::clone(&s))
                } else if weak_equality {
                    visited_weak_eq_states.insert(VisitedState {
                        state: Rc::clone(&s),
                    })
                } else {
                    true
                };
                if keep {
                    let sc = StateContainer {
                        state: s,
                        expanded: Rc::new(RefCell::new(false)),
                    };
                    candidate_containers.push(sc);
                }
            }

            for (i, (heuristic, weight)) in heuristics.iter().enumerate() {
                let candidate_states = candidate_containers.iter().map(|sc| sc.state.as_ref());
                for sh in heuristic.eval_gen(candidate_states, ss)? {
                    let (si, h) = sh?;
                    let g: f64 = candidate_containers[si].state.g;
                    match h {
                        Some(v) => {
                            let f = *weight * v + (1.0 - *weight) * g;
                            let sc = candidate_containers[si].clone();

                            let item = PrioritizedItem {
                                heuristic: f,
                                state_container: sc,
                            };
                            switch_policy.notify_push(i, &item);
                            opens[i].push(item);
                        }
                        None => continue,
                    }
                }
            }
        }
    }
    metrics.insert("expanded_states".to_string(), states_expanded.to_string());
    Ok((None, metrics))
}
