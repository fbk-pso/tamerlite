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

use std::collections::VecDeque;
use std::hash::{Hash, Hasher};
use std::rc::Rc;
use std::time::SystemTime;
use std::{collections::BinaryHeap, vec::Vec};

use rustc_hash::{FxBuildHasher, FxHashMap, FxHashSet};

use pyo3::exceptions::PyTimeoutError;
use pyo3::prelude::*;

use super::heuristics::*;
use super::search_space::*;
use super::search_state::*;
use super::structures::Action;

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
) -> PyResult<(
    Option<Vec<(Option<String>, Action, Option<String>)>>,
    FxHashMap<String, String>,
)> {
    let mut metrics = FxHashMap::with_hasher(FxBuildHasher::default());
    let start = SystemTime::now();
    let init = Rc::new(ss.initial_state(None)?);
    let mut expanded_states = 0;
    if early_termination && ss.goal_reached(&init, None)? {
        metrics.insert("expanded_states".to_string(), expanded_states.to_string());
        metrics.insert("goal_depth".to_string(), init.g.to_string());
        return build_plan(ss, &init).map(|plan| (plan, metrics));
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
            return Ok((None, metrics));
        }
    };
    let mut open = BinaryHeap::new();
    open.push(PrioritizedItem {
        heuristic: init_h,
        state: init,
    });
    while let Some(current) = open.pop() {
        if let Some(t) = timeout {
            if start.elapsed().unwrap().as_secs_f32() > t {
                return Err(PyTimeoutError::new_err("Timeout"));
            }
        }
        let state = current.state;
        expanded_states += 1;
        if !early_termination && ss.goal_reached(&state, None)? {
            metrics.insert("expanded_states".to_string(), expanded_states.to_string());
            metrics.insert("goal_depth".to_string(), state.g.to_string());
            return build_plan(ss, &state).map(|plan| (plan, metrics));
        } else {
            let mut successors = Vec::new();
            for s in ss.get_successor_states_iter(&state) {
                let s = s?;
                if early_termination && ss.goal_reached(&s, None)? {
                    metrics.insert("expanded_states".to_string(), expanded_states.to_string());
                    metrics.insert("goal_depth".to_string(), s.g.to_string());
                    return build_plan(ss, &s).map(|plan| (plan, metrics));
                }
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
                if keep {
                    successors.push(s);
                }
            }

            let successors_iter = successors.iter().map(|s| &(**s));
            for rs in heuristic.eval_gen(successors_iter, ss)? {
                let (i, h) = rs?;
                let s = &successors[i];
                match h {
                    Some(v) => {
                        let f = weight * v + (1.0 - weight) * s.g;
                        open.push(PrioritizedItem {
                            heuristic: f,
                            state: Rc::clone(s),
                        });
                    }
                    None => {}
                }
            }
        }
    }
    metrics.insert("expanded_states".to_string(), expanded_states.to_string());
    Ok((None, metrics))
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
    let mut counter = 0;

    if early_termination && ss.goal_reached(&init, None)? {
        metrics.insert("expanded_states".to_string(), counter.to_string());
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

        counter += 1;
        if !early_termination && ss.goal_reached(&state, None)? {
            metrics.insert("expanded_states".to_string(), counter.to_string());
            metrics.insert("goal_depth".to_string(), state.g.to_string());
            return build_plan(ss, &state).map(|plan| (plan, metrics));
        } else {
            for rs in ss.get_successor_states_iter(&state) {
                let s = rs?;
                if early_termination && ss.goal_reached(&s, None)? {
                    metrics.insert("expanded_states".to_string(), counter.to_string());
                    metrics.insert("goal_depth".to_string(), s.g.to_string());
                    return build_plan(ss, &s).map(|plan| (plan, metrics));
                }
                open.push_back(s);
            }
        }
    }
    metrics.insert("expanded_states".to_string(), counter.to_string());
    Ok((None, metrics))
}

pub fn ehc_search<H: HeuristicTrait, S: SearchSpaceTrait>(
    ss: &S,
    heuristic: &H,
    timeout: Option<f32>,
    early_termination: bool,
) -> PyResult<(
    Option<Vec<(Option<String>, Action, Option<String>)>>,
    FxHashMap<String, String>,
)> {
    let mut metrics = FxHashMap::with_hasher(FxBuildHasher::default());
    let start = SystemTime::now();
    let init = ss.initial_state(None)?;
    let mut counter = 0;
    if early_termination && ss.goal_reached(&init, None)? {
        metrics.insert("expanded_states".to_string(), counter.to_string());
        metrics.insert("goal_depth".to_string(), init.g.to_string());
        return build_plan(ss, &init).map(|plan| (plan, metrics));
    }
    let mut best_h = match heuristic.eval(&init, ss)? {
        Some(v) => v,
        None => {
            metrics.insert("expanded_states".to_string(), 0.to_string());
            return Ok((None, metrics));
        }
    };
    let mut open = VecDeque::new();
    open.push_back(init);
    while let Some(state) = open.pop_front() {
        if let Some(t) = timeout {
            if start.elapsed().unwrap().as_secs_f32() > t {
                return Err(PyTimeoutError::new_err("Timeout"));
            }
        }

        counter += 1;
        if !early_termination && ss.goal_reached(&state, None)? {
            metrics.insert("expanded_states".to_string(), counter.to_string());
            metrics.insert("goal_depth".to_string(), state.g.to_string());
            return build_plan(ss, &state).map(|plan| (plan, metrics));
        } else {
            for rs in heuristic.eval_gen_owned(ss.get_successor_states_iter(&state), ss)? {
                let (s, h) = rs?;
                if early_termination && ss.goal_reached(&s, None)? {
                    metrics.insert("expanded_states".to_string(), counter.to_string());
                    metrics.insert("goal_depth".to_string(), s.g.to_string());
                    return build_plan(ss, &s).map(|plan| (plan, metrics));
                }
                match h {
                    Some(v) => {
                        if v < best_h {
                            best_h = v;
                            open.clear();
                        }
                        open.push_back(s);
                    }
                    None => continue,
                }
            }
        }
    }
    metrics.insert("expanded_states".to_string(), counter.to_string());
    Ok((None, metrics))
}
