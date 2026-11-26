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
use std::cell::RefCell;
use std::io::{self, Write};
use std::rc::Rc;
use std::time::Instant;
use std::{collections::BinaryHeap, vec::Vec};

use rustc_hash::{FxBuildHasher, FxHashMap};

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
    fn set_expanded(&self, expanded: bool) {
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
        // `BinaryHeap` is a max-heap; comparing `other` against `self` (not
        // `self` against `other`) inverts every field so `pop()` returns
        // the item with the lexicographically smallest
        // `(heuristic, todo_len)` pair first. `f64` isn't `Ord`, hence
        // `total_cmp` in place of a plain tuple comparison.
        other.heuristic.total_cmp(&self.heuristic).then_with(|| {
            other
                .state_container
                .state
                .todo
                .len()
                .cmp(&self.state_container.state.todo.len())
        })
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

/// Draws an expansion progress bar on stderr (`finish_line` ends the line).
fn print_expansion_progress(expanded: usize, limit: usize, finish_line: bool) {
    const WIDTH: usize = 40;
    let filled = (expanded.saturating_mul(WIDTH) / limit).min(WIDTH);
    let bar = format!("{}{}", "#".repeat(filled), "-".repeat(WIDTH - filled));
    let percentage = expanded.saturating_mul(100) / limit;
    eprint!("\rExpanded states: [{bar}] {expanded}/{limit} ({percentage:>3}%)");
    if finish_line {
        eprintln!();
    } else {
        let _ = io::stderr().flush();
    }
}

pub fn multiqueue_search<H: HeuristicTrait, S: SearchSpaceTrait>(
    ss: &S,
    heuristics: Vec<(H, f64)>,
    timeout: Option<f32>,
    max_expanded_states: Option<usize>,
    early_termination: bool,
    weak_equality: bool,
    show_expansion_progress: bool,
) -> PyResult<SearchResult> {
    let mut switch_policy = RoundRobinSwitchPolicy::new(heuristics.len());
    _multiqueue_search(
        ss,
        heuristics,
        &mut switch_policy,
        timeout,
        max_expanded_states,
        early_termination,
        weak_equality,
        show_expansion_progress,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn _multiqueue_search<T: MQSwitchPolicy, H: HeuristicTrait, S: SearchSpaceTrait>(
    ss: &S,
    heuristics: Vec<(H, f64)>,
    switch_policy: &mut T,
    timeout: Option<f32>,
    max_expanded_states: Option<usize>,
    early_termination: bool,
    weak_equality: bool,
    show_expansion_progress: bool,
) -> PyResult<SearchResult> {
    info!(
        "multiqueue_search: queues={} timeout={:?} early_termination={} weak_equality={}",
        heuristics.len(),
        timeout,
        early_termination,
        weak_equality
    );
    let mut metrics = FxHashMap::with_hasher(FxBuildHasher);
    let start = Instant::now();
    let Some(init) = ss.initial_state(None)? else {
        // No valid initial state (see `SearchSpaceTrait::initial_state`).
        metrics.insert("expanded_states", 0.to_string());
        return Ok((None, metrics));
    };
    let init = Rc::new(init);
    let mut expanded_states: usize = 0;
    let mut generated_states = 1;
    // The bar needs a limit to measure against; without one it stays off.
    let progress_limit = max_expanded_states.filter(|_| show_expansion_progress);
    let finish_progress = |expanded: usize| {
        if let Some(limit) = progress_limit {
            print_expansion_progress(expanded, limit, true);
        }
    };
    if let Some(limit) = progress_limit {
        print_expansion_progress(0, limit, false);
    }
    if early_termination && ss.goal_reached(&init, None)? {
        finish_progress(expanded_states);
        metrics.insert("expanded_states", expanded_states.to_string());
        metrics.insert("goal_depth", init.g.to_string());
        return Ok((Some(extract_path(&init)), metrics));
    }

    let item = PrioritizedItem {
        heuristic: 0.0,
        state_container: StateContainer {
            state: init,
            expanded: Rc::new(RefCell::new(false)),
        },
    };

    let mut visited_states = VisitedStates::new(ss, weak_equality);
    visited_states.insert(&item.state_container.state);

    let mut opens = Vec::with_capacity(heuristics.len());
    for (i, _) in heuristics.iter().enumerate() {
        let mut open = BinaryHeap::new();
        open.push(item.clone());
        opens.push(open);
        switch_policy.notify_push(i, &item);
    }

    let mut i = 0;
    loop {
        if let Some(t) = timeout {
            if start.elapsed().as_secs_f32() > t {
                finish_progress(expanded_states);
                return Err(PyTimeoutError::new_err("Timeout"));
            }
        }
        // If one of the queues is empty, then all the others are (logically) empty too
        if opens[i].is_empty() {
            break;
        }

        i = switch_policy.switching_policy(expanded_states);
        if let Some(current) = opens[i].pop() {
            switch_policy.notify_pop(i, &current);
            if *current.state_container.expanded.borrow() {
                continue;
            }
            if let Err(e) = check_expansion_limit(expanded_states, max_expanded_states) {
                finish_progress(expanded_states);
                return Err(e);
            }
            current.state_container.set_expanded(true);
            let state = &current.state_container.state;
            expanded_states += 1;
            if let Some(limit) = progress_limit {
                if expanded_states.is_multiple_of((limit / 100).max(1)) {
                    print_expansion_progress(expanded_states, limit, false);
                }
            }
            if expanded_states.is_multiple_of(10_000) {
                debug!(
                    "multiqueue_search: expanded={} generated={} open={}",
                    expanded_states,
                    generated_states,
                    opens.iter().map(|o| o.len()).sum::<usize>()
                );
            }
            if !early_termination && ss.goal_reached(state, None)? {
                info!(
                    "multiqueue_search: goal found — expanded={} depth={}",
                    expanded_states, state.g
                );
                finish_progress(expanded_states);
                metrics.insert("expanded_states", expanded_states.to_string());
                metrics.insert("goal_depth", state.g.to_string());
                return Ok((Some(extract_path(state)), metrics));
            }

            let mut candidate_containers: Vec<StateContainer> = Vec::new();
            for rs in ss.get_successor_states_iter(state) {
                let s = rs?;
                generated_states += 1;
                if early_termination && ss.goal_reached(&s, None)? {
                    info!(
                        "multiqueue_search: goal found — expanded={} depth={}",
                        expanded_states, s.g
                    );
                    finish_progress(expanded_states);
                    metrics.insert("expanded_states", expanded_states.to_string());
                    metrics.insert("goal_depth", s.g.to_string());
                    return Ok((Some(extract_path(&s)), metrics));
                }
                let s = Rc::new(s);
                if visited_states.insert(&s) {
                    let sc = StateContainer {
                        state: s,
                        expanded: Rc::new(RefCell::new(false)),
                    };
                    candidate_containers.push(sc);
                }
            }

            for (i, (heuristic, weight)) in heuristics.iter().enumerate() {
                for sh in heuristic.eval_gen_container(&candidate_containers, ss)? {
                    let (si, h) = sh?;
                    let g: f64 = candidate_containers[si].state.g;
                    if let Some(h) = h {
                        let f = *weight * h + (1.0 - *weight) * g;
                        let sc = candidate_containers[si].clone();

                        let item = PrioritizedItem {
                            heuristic: f,
                            state_container: sc,
                        };
                        switch_policy.notify_push(i, &item);
                        opens[i].push(item);
                    }
                }
            }
        }
    }
    info!(
        "multiqueue_search: no solution found — expanded={}",
        expanded_states
    );
    finish_progress(expanded_states);
    metrics.insert("expanded_states", expanded_states.to_string());
    Ok((None, metrics))
}
