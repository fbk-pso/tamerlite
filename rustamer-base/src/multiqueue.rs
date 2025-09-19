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

use std::cell::RefCell;
use std::rc::Rc;
use std::time::SystemTime;
use std::{collections::BinaryHeap, collections::HashMap, collections::HashSet, vec::Vec};

use pyo3::exceptions::PyTimeoutError;
use pyo3::prelude::*;

use super::heuristics::*;
use super::search::*;
use super::search_space::*;
use super::search_state::*;

#[derive(Debug)]
pub struct StateContainer {
    pub state: State,
    pub expanded: bool,
}

impl StateContainer {
    fn set_expanded(&mut self, expanded: bool) -> () {
        self.expanded = expanded;
    }
}

#[derive(Debug, Clone)]
pub struct PrioritizedItem {
    pub heuristic: f64,
    pub state_container: Rc<RefCell<StateContainer>>,
}

impl PartialEq for PrioritizedItem {
    fn eq(&self, other: &Self) -> bool {
        self.heuristic == other.heuristic
            && self.state_container.borrow().state.todo.len()
                == other.state_container.borrow().state.todo.len()
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
        } else if self.state_container.borrow().state.todo.len()
            < other.state_container.borrow().state.todo.len()
        {
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
    early_termination: bool
) -> PyResult<(
    Option<Vec<(Option<String>, String, Option<String>)>>,
    HashMap<String, String>,
)> {
    let mut switch_policy = RoundRobinSwitchPolicy::new(heuristics.len());
    _multiqueue_search(ss, heuristics, &mut switch_policy, timeout, early_termination)
}

pub fn _multiqueue_search<T: MQSwitchPolicy, H: HeuristicTrait, S: SearchSpaceTrait>(
    ss: &S,
    heuristics: Vec<(H, f64)>,
    switch_policy: &mut T,
    timeout: Option<f32>,
    early_termination: bool
) -> PyResult<(
    Option<Vec<(Option<String>, String, Option<String>)>>,
    HashMap<String, String>,
)> {
    let mut metrics = HashMap::new();
    let start = SystemTime::now();
    let init = ss.initial_state(None)?;

    let mut open_set: HashSet<State> = HashSet::new();
    let mut closed_set: HashSet<State> = HashSet::new();
    if !ss.is_temporal() {
        open_set.insert(init.full_clone());
    }

    let item = PrioritizedItem {
        heuristic: 0.0,
        state_container: Rc::new(RefCell::new(StateContainer {
            state: init,
            expanded: false,
        })),
    };

    let mut opens = Vec::new();
    for (i, _) in heuristics.iter().enumerate() {
        let mut open = BinaryHeap::new();
        open.push(item.clone());
        opens.push(open);
        switch_policy.notify_push(i, &item);
    }

    let mut counter = 0;
    let mut states_expanded = 0;
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
        let i = switch_policy.switching_policy(counter);
        let open = &mut opens[i];
        if let Some(current) = open.pop() {
            switch_policy.notify_pop(i, &current);
            let mut candidate_containers: Vec<Rc<RefCell<StateContainer>>> = Vec::new();
            {
                let sc = &mut (*(current.state_container)).borrow_mut();
                if sc.expanded {
                    continue;
                }
                sc.set_expanded(true);
                let state = &sc.state;
                if !ss.is_temporal() {
                    let opened = open_set.take(state);
                    if let Some(s) = opened {
                        closed_set.insert(s);
                    }
                }
                states_expanded += 1;
                counter += 1;
                if ss.goal_reached(&state, None)? {
                    metrics.insert("expanded_states".to_string(), states_expanded.to_string());
                    metrics.insert("goal_depth".to_string(), state.g.to_string());
                    return build_plan(ss, &state).map(|plan| (plan, metrics));
                }

                for rs in ss.get_successor_states_iter(&state) {
                    let s = rs?;
                    if early_termination && ss.goal_reached(&s, None)? {
                        metrics.insert("expanded_states".to_string(), states_expanded.to_string());
                        metrics.insert("goal_depth".to_string(), s.g.to_string());
                        return build_plan(ss, &s).map(|plan| (plan, metrics));
                    }
                    let sc = StateContainer {
                        state: s,
                        expanded: false,
                    };
                    if !ss.is_temporal() {
                        if closed_set.contains(&sc.state) || open_set.contains(&sc.state) {
                            continue;
                        }
                        open_set.insert(sc.state.full_clone());
                    }
                    candidate_containers.push(Rc::new(RefCell::new(sc)));
                }
            }
            for (i, (heuristic, weight)) in heuristics.iter().enumerate() {
                for sh in heuristic
                    .eval_gen_container(&candidate_containers, ss)?
                {
                    let (si, h) = sh?;
                    let g: f64 = candidate_containers[si].borrow().state.g;
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
