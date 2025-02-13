use std::collections::VecDeque;
use std::time::SystemTime;
use std::{
    collections::BinaryHeap,
    collections::HashSet,
    vec::Vec
};

use pyo3::exceptions::PyTimeoutError;
use pyo3::prelude::*;

use super::search_space::*;
use super::heuristics::*;
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

pub fn build_plan(ss: &mut SearchSpace, state: &State) -> PyResult<Option<Vec<(Option<String>, String, Option<String>)>>> {
    let path = PersistentList::to_vec(&state.path).into_iter().map(|(a, _, _)| a.to_string()).collect();
    let plan = ss.build_plan(path)?;
    let mut res = Vec::new();
    for (s, a, d) in plan.iter() {
        let mut ss = None;
        let mut ds = None;
        if let Some(start) = s {
            ss = Some(format!("{}/{}", start.numer().to_string(), start.denom().to_string()));
        }
        if let Some(duration) = d {
            ds = Some(format!("{}/{}", duration.numer().to_string(), duration.denom().to_string()));
        }
        res.push((ss, a.to_string(), ds));
    }
    Ok(Some(res))
}

#[pyfunction]
#[pyo3(signature = (ss, heuristic, timeout=None))]
pub fn astar_search(ss: &mut SearchSpace, heuristic: &mut Heuristic, timeout: Option<f32>) -> PyResult<Option<Vec<(Option<String>, String, Option<String>)>>> {
    wastar_search(ss, heuristic, 0.5, timeout)
}

#[pyfunction]
#[pyo3(signature = (ss, heuristic, timeout=None))]
pub fn gbfs_search(ss: &mut SearchSpace, heuristic: &mut Heuristic, timeout: Option<f32>) -> PyResult<Option<Vec<(Option<String>, String, Option<String>)>>> {
    wastar_search(ss, heuristic, 1.0, timeout)
}

#[pyfunction]
#[pyo3(signature = (ss, heuristic, weight, timeout=None))]
pub fn wastar_search(ss: &mut SearchSpace, heuristic: &mut Heuristic, weight: f64, timeout: Option<f32>) -> PyResult<Option<Vec<(Option<String>, String, Option<String>)>>> {
    let start = SystemTime::now();
    let init = ss.initial_state(None)?;
    let init_h = match heuristic.eval(&init, ss)? {
        Some(v) => v,
        None => {
            return Ok(None);
        }
    };
    let mut open = BinaryHeap::new();
    let mut open_set = HashSet::new();
    let mut closed_set = HashSet::new();
    open.push(PrioritizedItem{heuristic: init_h, state: init});
    let mut counter = 0;
    while let Some(current) = open.pop() {
        if let Some(t) = timeout {
            if start.elapsed().unwrap().as_secs_f32() > t {
                return Err(PyTimeoutError::new_err("Timeout"));
            }
        }
        let state = current.state;
        if !ss.is_temporal {
            closed_set.insert(state.clone());
            open_set.remove(&state);
        }
        // println!("{:?} {:?}", state.path.iter().map(|(ev, _)| &ev.action).collect::<Vec<&String>>(), current.heuristic);
        counter += 1;
        if ss.goal_reached(&state, None)? {
            println!("Expanded states: {}", counter);
            return build_plan(ss, &state);
        } else {
            for s in ss.get_successor_states(&state)? {
                if open_set.contains(&s) || closed_set.contains(&s) {
                    continue;
                }
                let h = heuristic.eval(&s, ss)?;
                match h {
                    Some(v) => {
                        let f = weight * v + (1.0 - weight) * s.g;
                        if !ss.is_temporal {
                            open_set.insert(s.clone());
                        }
                        open.push(PrioritizedItem{heuristic: f, state: s});
                    },
                    None => continue,
                }
            }
        }
    }
    Ok(None)
}

#[pyfunction]
#[pyo3(signature = (ss, timeout=None))]
pub fn bfs_search(ss: &mut SearchSpace, timeout: Option<f32>) -> PyResult<Option<Vec<(Option<String>, String, Option<String>)>>> {
    basic_search(ss, true, timeout)
}

#[pyfunction]
#[pyo3(signature = (ss, timeout=None))]
pub fn dfs_search(ss: &mut SearchSpace, timeout: Option<f32>) -> PyResult<Option<Vec<(Option<String>, String, Option<String>)>>> {
    basic_search(ss, false, timeout)
}

fn basic_search(ss: &mut SearchSpace, bfs: bool, timeout: Option<f32>) -> PyResult<Option<Vec<(Option<String>, String, Option<String>)>>> {
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
            println!("Expanded states: {}", counter);
            return build_plan(ss, &state);
        } else {
            for s in ss.get_successor_states(&state)? {
                open.push_back(s);
            }
        }
    }
    Ok(None)
}

#[pyfunction]
#[pyo3(signature = (ss, heuristic, timeout=None))]
pub fn ehc_search(ss: &mut SearchSpace, heuristic: &mut Heuristic, timeout: Option<f32>) -> PyResult<Option<Vec<(Option<String>, String, Option<String>)>>> {
    let start = SystemTime::now();
    let init = ss.initial_state(None)?;
    let mut best_h = match heuristic.eval(&init, ss)? {
        Some(v) => v,
        None => {
            return Ok(None);
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
            println!("Expanded states: {}", counter);
            return build_plan(ss, &state);
        } else {
            for s in ss.get_successor_states(&state)? {
                let h = heuristic.eval(&s, ss)?;
                match h {
                    Some(v) => {
                        if v < best_h {
                            best_h = v;
                            open.clear();
                        }
                        open.push_back(s);
                    },
                    None => continue,
                }
            }
        }
    }
    Ok(None)
}
