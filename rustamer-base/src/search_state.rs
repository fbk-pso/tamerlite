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

use im::Vector;
use multiset::HashMultiSet;
use pyo3::{exceptions::PyException, prelude::*};
use rustc_hash::{FxBuildHasher, FxHashMap};
use std::hash::{Hash, Hasher};
use std::{
    sync::{Arc, Mutex},
    vec::Vec,
};

use super::expressions::*;
use super::expressions_utils::*;
use super::stn::DeltaSTN;
use super::structures::Action;
use super::utils::*;

#[pyclass(frozen)]
#[derive(Debug)]
pub struct State {
    pub assignments: Vector<ExpressionNode>,
    pub temporal_network: Option<DeltaSTN<u64, f64>>,
    pub todo: FxHashMap<Action, (usize, u32)>,
    pub active_conditions: HashMultiSet<Vec<ExpressionNode>>,
    pub g: f64,
    pub path: Option<Arc<PersistentList<(Action, usize, u32)>>>,
    pub heuristic_cache: Mutex<FxHashMap<String, Option<f64>>>,
}

#[pymethods]
impl State {
    #[getter]
    fn g(&self) -> f64 {
        self.g
    }

    #[getter]
    fn todo(&self) -> FxHashMap<Action, (usize, u32)> {
        self.todo.clone()
    }

    #[getter]
    fn path(&self) -> Vec<(Action, usize, u32)> {
        PersistentList::to_vec_copy(&self.path)
    }

    fn get_py_value(&self, fluent: usize) -> PyResult<PyExpressionNode> {
        let value = self
            .assignments
            .get(fluent)
            .ok_or_else(|| PyException::new_err("Fluent not found!"))?;
        Ok(PyExpressionNode { v: value.clone() })
    }
}

impl FluentValueTrait for State {
    fn get_value(&self, fluent: usize) -> &ExpressionNode {
        &self.assignments[fluent]
    }
}

impl State {
    /// Clones the current state, except for the caches
    /// This is useful to create children of this state
    pub fn clone_for_child(&self) -> Self {
        Self {
            assignments: self.assignments.clone(),
            temporal_network: self.temporal_network.clone(),
            todo: self.todo.clone(),
            active_conditions: self.active_conditions.clone(),
            g: self.g.clone(),
            path: self.path.clone(),
            heuristic_cache: Mutex::new(FxHashMap::with_hasher(FxBuildHasher::default())), // Cloning erases the cache
        }
    }

    /// Clones the current state, including the caches
    pub fn full_clone(&self) -> Self {
        Self {
            assignments: self.assignments.clone(),
            temporal_network: self.temporal_network.clone(),
            todo: self.todo.clone(),
            active_conditions: self.active_conditions.clone(),
            g: self.g.clone(),
            path: self.path.clone(),
            heuristic_cache: Mutex::new(self.heuristic_cache.lock().unwrap().clone()),
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
        Hash::hash(&self.assignments, state);
    }
}
