use std::{
    collections::HashMap, sync::{Arc, Mutex}, vec::Vec
};
use multiset::HashMultiSet;
use std::hash::{Hash, Hasher};
use pyo3::prelude::*;

use super::stn::DeltaSTN;
use super::expressions::*;
use super::utils::*;

#[pyclass(frozen)]
#[derive(Debug)]
pub struct State {
    pub assignments: HashMap<String, ExpressionNode>,
    pub temporal_network: Option<DeltaSTN<u64, f32>>,
    pub todo: HashMap<String, (usize, u32)>,
    pub active_conditions: HashMultiSet<Vec<ExpressionNode>>,
    pub g: f64,
    pub path: Option<Arc<PersistentList<(String, usize, u32)>>>,
    pub heuristic_cache: Mutex<HashMap<String, Option<f64>>>
}

#[pymethods]
impl State {
    #[getter]
    fn g(&self) -> f64 {
        self.g
    }

    #[getter]
    fn todo(&self) -> HashMap<String, (usize, u32)> {
        self.todo.clone()
    }

    #[getter]
    fn path(&self) -> Vec<(String, usize, u32)> {
        PersistentList::to_vec_copy(&self.path)
    }
}

impl State {
    pub fn get_value(&self, fluent: &String) -> &ExpressionNode {
        &self.assignments[fluent]
    }

    /// Clones the current statw, except for the caches
    /// This is useful to create children of this state
    pub fn clone_for_child(&self) -> Self {
        Self { assignments: self.assignments.clone(),
                temporal_network: self.temporal_network.clone(),
                todo: self.todo.clone(),
                active_conditions: self.active_conditions.clone(),
                g: self.g.clone(),
                path: self.path.clone(),
                heuristic_cache: Mutex::new(HashMap::new()) // Cloning erases the cache
             }
    }

    /// Clones the current state, including the caches
    pub fn full_clone(&self) -> Self {
        Self { assignments: self.assignments.clone(),
                temporal_network: self.temporal_network.clone(),
                todo: self.todo.clone(),
                active_conditions: self.active_conditions.clone(),
                g: self.g.clone(),
                path: self.path.clone(),
                heuristic_cache: Mutex::new(self.heuristic_cache.lock().unwrap().clone())
             }
    }
}

impl PartialEq for State {
    fn eq(&self, other: &Self) -> bool {
        if self.temporal_network.is_none() {
            self.assignments == other.assignments
        } else {
            false
        }
    }
}

impl Eq for State {}

impl Hash for State {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let mut pairs: Vec<_> = self.assignments.iter().collect();
        pairs.sort_by_key(|i| i.0);
        Hash::hash(&pairs, state);
    }
}