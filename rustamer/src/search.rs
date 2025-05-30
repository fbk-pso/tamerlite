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
use std::sync::Mutex;
use std::time::SystemTime;
use std::{collections::BinaryHeap, collections::HashMap, collections::HashSet, vec::Vec};

use pyo3::exceptions::PyTimeoutError;
use pyo3::prelude::*;

use super::heuristics::*;
use super::search_space::*;
use super::search_state::*;
use super::utils::*;

#[derive(Debug)]
struct PrioritizedItem {
    heuristic: f64,
    state: State,
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

pub fn build_plan(
    ss: &SearchSpace,
    state: &State,
) -> PyResult<Option<Vec<(Option<String>, String, Option<String>)>>> {
    let path = PersistentList::to_vec(&state.path)
        .into_iter()
        .map(|(a, _, _)| a.to_string())
        .collect();
    let plan = ss.build_plan(path)?;
    let mut res = Vec::new();
    for (s, a, d) in plan.iter() {
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
        res.push((ss, a.to_string(), ds));
    }
    Ok(Some(res))
}

#[pyfunction]
#[pyo3(signature = (ss, heuristic, timeout=None))]
pub fn astar_search(
    ss: &SearchSpace,
    heuristic: &Heuristic,
    timeout: Option<f32>,
) -> PyResult<(
    Option<Vec<(Option<String>, String, Option<String>)>>,
    HashMap<String, String>,
)> {
    wastar_search(ss, heuristic, 0.5, timeout)
}

#[pyfunction]
#[pyo3(signature = (ss, heuristic, timeout=None))]
pub fn gbfs_search(
    ss: &SearchSpace,
    heuristic: &Heuristic,
    timeout: Option<f32>,
) -> PyResult<(
    Option<Vec<(Option<String>, String, Option<String>)>>,
    HashMap<String, String>,
)> {
    wastar_search(ss, heuristic, 1.0, timeout)
}

#[pyfunction]
#[pyo3(signature = (ss, heuristic, weight, timeout=None))]
pub fn wastar_search(
    ss: &SearchSpace,
    heuristic: &Heuristic,
    weight: f64,
    timeout: Option<f32>,
) -> PyResult<(
    Option<Vec<(Option<String>, String, Option<String>)>>,
    HashMap<String, String>,
)> {
    let mut metrics = HashMap::new();
    let start = SystemTime::now();
    let init = ss.initial_state(None)?;
    let init_h = match heuristic.eval(&init, ss)? {
        Some(v) => v,
        None => {
            metrics.insert("expanded_states".to_string(), 0.to_string());
            return Ok((None, metrics));
        }
    };
    let mut open = BinaryHeap::new();
    let open_set = Mutex::new(HashSet::new());
    let mut closed_set = HashSet::new();
    if !ss.is_temporal {
        open_set.lock().unwrap().insert(init.full_clone());
    }
    open.push(PrioritizedItem {
        heuristic: init_h,
        state: init,
    });
    let mut counter = 0;
    while let Some(current) = open.pop() {
        if let Some(t) = timeout {
            if start.elapsed().unwrap().as_secs_f32() > t {
                return Err(PyTimeoutError::new_err("Timeout"));
            }
        }
        let state = current.state;
        if !ss.is_temporal {
            let opened = open_set.lock().unwrap().take(&state);
            if let Some(s) = opened {
                closed_set.insert(s);
            }
        }
        // println!("{:?} {:?}", state.path.iter().map(|(ev, _)| &ev.action).collect::<Vec<&String>>(), current.heuristic);
        counter += 1;
        if ss.goal_reached(&state, None)? {
            metrics.insert("expanded_states".to_string(), counter.to_string());
            metrics.insert("goal_depth".to_string(), state.g.to_string());
            return build_plan(ss, &state).map(|plan| (plan, metrics));
        } else {
            let successors_iter =
                ss.get_successor_states_iter(&state)
                    .filter(|sx: &Result<State, PyErr>| match sx {
                        Ok(s) => {
                            ss.is_temporal
                                || (!closed_set.contains(s)
                                    && !open_set.lock().unwrap().contains(s))
                        }
                        Err(_) => return true,
                    });
            for rs in heuristic.eval_gen(successors_iter, ss)? {
                let (s, h) = rs?;
                match h {
                    Some(v) => {
                        let f = weight * v + (1.0 - weight) * s.g;
                        if !ss.is_temporal {
                            open_set.lock().unwrap().insert(s.full_clone());
                        }
                        open.push(PrioritizedItem {
                            heuristic: f,
                            state: s,
                        });
                    }
                    None => continue,
                }
            }
        }
    }
    metrics.insert("expanded_states".to_string(), counter.to_string());
    Ok((None, metrics))
}

#[pyfunction]
#[pyo3(signature = (ss, timeout=None))]
pub fn bfs_search(
    ss: &SearchSpace,
    timeout: Option<f32>,
) -> PyResult<(
    Option<Vec<(Option<String>, String, Option<String>)>>,
    HashMap<String, String>,
)> {
    basic_search(ss, true, timeout)
}

#[pyfunction]
#[pyo3(signature = (ss, timeout=None))]
pub fn dfs_search(
    ss: &SearchSpace,
    timeout: Option<f32>,
) -> PyResult<(
    Option<Vec<(Option<String>, String, Option<String>)>>,
    HashMap<String, String>,
)> {
    basic_search(ss, false, timeout)
}

fn basic_search(
    ss: &SearchSpace,
    bfs: bool,
    timeout: Option<f32>,
) -> PyResult<(
    Option<Vec<(Option<String>, String, Option<String>)>>,
    HashMap<String, String>,
)> {
    let mut metrics = HashMap::new();
    let start = SystemTime::now();
    let init = ss.initial_state(None)?;
    let mut open = VecDeque::new();
    open.push_back(init);
    let mut counter = 0;
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
        if ss.goal_reached(&state, None)? {
            metrics.insert("expanded_states".to_string(), counter.to_string());
            metrics.insert("goal_depth".to_string(), state.g.to_string());
            return build_plan(ss, &state).map(|plan| (plan, metrics));
        } else {
            for rs in ss.get_successor_states_iter(&state) {
                open.push_back(rs?);
            }
        }
    }
    metrics.insert("expanded_states".to_string(), counter.to_string());
    Ok((None, metrics))
}

#[pyfunction]
#[pyo3(signature = (ss, heuristic, timeout=None))]
pub fn ehc_search(
    ss: &SearchSpace,
    heuristic: &Heuristic,
    timeout: Option<f32>,
) -> PyResult<(
    Option<Vec<(Option<String>, String, Option<String>)>>,
    HashMap<String, String>,
)> {
    let mut metrics = HashMap::new();
    let start = SystemTime::now();
    let init = ss.initial_state(None)?;
    let mut best_h = match heuristic.eval(&init, ss)? {
        Some(v) => v,
        None => {
            metrics.insert("expanded_states".to_string(), 0.to_string());
            return Ok((None, metrics));
        }
    };
    let mut open = VecDeque::new();
    open.push_back(init);
    let mut counter = 0;
    while let Some(state) = open.pop_front() {
        if let Some(t) = timeout {
            if start.elapsed().unwrap().as_secs_f32() > t {
                return Err(PyTimeoutError::new_err("Timeout"));
            }
        }

        counter += 1;
        if ss.goal_reached(&state, None)? {
            metrics.insert("expanded_states".to_string(), counter.to_string());
            metrics.insert("goal_depth".to_string(), state.g.to_string());
            return build_plan(ss, &state).map(|plan| (plan, metrics));
        } else {
            for rs in heuristic.eval_gen(ss.get_successor_states_iter(&state), ss)? {
                let (s, h) = rs?;
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
