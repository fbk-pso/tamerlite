use std::sync::{Arc, Mutex};
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
use super::search::*;


#[derive(Debug, Clone)]
struct StateContainer {
    state: State,
    expanded: bool,
}

#[derive(Debug, Clone)]
struct PrioritizedItem {
    heuristic: f64,
    state_container: Arc<Mutex<StateContainer>>,
}

impl PartialEq for PrioritizedItem {
    fn eq(&self, other: &Self) -> bool {
        self.heuristic == other.heuristic && self.state_container.lock().unwrap().state.todo.len() == other.state_container.lock().unwrap().state.todo.len()
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
        } else if self.state_container.lock().unwrap().state.todo.len() < other.state_container.lock().unwrap().state.todo.len() {
            std::cmp::Ordering::Greater
        } else {
            std::cmp::Ordering::Less
        }
    }
}

#[pyfunction]
pub fn multiqueue_search(ss: &mut SearchSpace, heuristics: Vec<(Heuristic, f64)>, timeout: Option<f32>) -> PyResult<Option<Vec<(Option<String>, String, Option<String>)>>> {
    let start = SystemTime::now();
    let init = ss.initial_state(None)?;
    let item = PrioritizedItem{heuristic: 0.0, state_container: Arc::new(Mutex::new(StateContainer{state: init, expanded: false})) };

    let mut opens = Vec::new();
    for _ in heuristics.iter() {
        let mut open = BinaryHeap::new();
        open.push(item.clone());
        opens.push(open);
    }

    let mut open_set = HashSet::new();
    let mut closed_set = HashSet::new();
    let mut counter = 0;
    let mut states_expanded = 0;
    loop {
        if let Some(t) = timeout {
            if start.elapsed().unwrap().as_secs_f32() > t {
                return Err(PyTimeoutError::new_err("Timeout"));
            }
        }
        if opens.iter().map(|o| o.len()).sum::<usize>() == 0 {
            break;
        }
        let i = counter % opens.len();
        let open = &mut opens[i];
        if open.len() == 0 {
            counter += 1;
            continue;
        }
        if let Some(current) = open.pop() {
            let sc = &mut current.state_container.lock().unwrap();
            if sc.expanded {
                continue;
            }
            sc.expanded = true;
            let state = &sc.state;
            if !ss.is_temporal {
                closed_set.insert(state.clone());
                open_set.remove(state);
            }
            states_expanded += 1;
            counter += 1;
            if ss.goal_reached(&state, None)? {
                println!("Expanded states: {}", states_expanded);
                return build_plan(ss, &state);
            } else {
                for s in ss.get_successor_states(&state)? {
                    if open_set.contains(&s) || closed_set.contains(&s) {
                        continue;
                    }
                    if !ss.is_temporal {
                        open_set.insert(s.clone());
                    }
                    let sc = Arc::new(Mutex::new(StateContainer{state: s.clone(), expanded: false}));
                    for (i, (heuristic, weight)) in heuristics.iter().enumerate() {
                        let h = heuristic.eval(&s, ss)?;
                        match h {
                            Some(v) => {
                                let f = weight * v + (1.0 - weight) * s.g;
                                opens[i].push(PrioritizedItem{heuristic: f, state_container: sc.clone()});
                            },
                            None => continue,
                        }
                    }
                }
            }
        }
    }
    Ok(None)
}
