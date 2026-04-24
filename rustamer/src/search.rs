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

use super::heuristic::Heuristic;
use pyo3::prelude::*;
use rustamer_base;
use rustamer_base::Action;
use rustc_hash::FxHashMap;

#[pyfunction]
#[pyo3(signature = (ss, timeout=None, max_expanded_states=None, early_termination=false))]
pub fn bfs_search(
    ss: &rustamer_base::SearchSpace,
    timeout: Option<f32>,
    max_expanded_states: Option<usize>,
    early_termination: bool,
) -> PyResult<(Option<Vec<Action>>, FxHashMap<String, String>)> {
    rustamer_base::bfs_search(ss, timeout, max_expanded_states, early_termination)
}

#[pyfunction]
#[pyo3(signature = (ss, timeout=None, max_expanded_states=None, early_termination=false))]
pub fn dfs_search(
    ss: &rustamer_base::SearchSpace,
    timeout: Option<f32>,
    max_expanded_states: Option<usize>,
    early_termination: bool,
) -> PyResult<(Option<Vec<Action>>, FxHashMap<String, String>)> {
    rustamer_base::dfs_search(ss, timeout, max_expanded_states, early_termination)
}

#[pyfunction]
#[pyo3(signature = (ss, heuristic, timeout=None, max_expanded_states=None, early_termination=false, weak_equality=false))]
pub fn ehc_search(
    ss: &rustamer_base::SearchSpace,
    heuristic: &Heuristic,
    timeout: Option<f32>,
    max_expanded_states: Option<usize>,
    early_termination: bool,
    weak_equality: bool,
) -> PyResult<(Option<Vec<Action>>, FxHashMap<String, String>)> {
    rustamer_base::ehc_search(
        ss,
        heuristic,
        timeout,
        max_expanded_states,
        early_termination,
        weak_equality,
    )
}

#[pyfunction]
#[pyo3(signature = (ss, heuristic, weight, timeout=None, max_expanded_states=None, early_termination=false, weak_equality=false))]
pub fn wastar_search(
    ss: &rustamer_base::SearchSpace,
    heuristic: &Heuristic,
    weight: f64,
    timeout: Option<f32>,
    max_expanded_states: Option<usize>,
    early_termination: bool,
    weak_equality: bool,
) -> PyResult<(Option<Vec<Action>>, FxHashMap<String, String>)> {
    rustamer_base::wastar_search(
        ss,
        heuristic,
        weight,
        timeout,
        max_expanded_states,
        early_termination,
        weak_equality,
    )
}

#[pyfunction]
#[pyo3(signature = (ss, heuristic, timeout=None, max_expanded_states=None, early_termination=false, weak_equality=false))]
pub fn astar_search(
    ss: &rustamer_base::SearchSpace,
    heuristic: &Heuristic,
    timeout: Option<f32>,
    max_expanded_states: Option<usize>,
    early_termination: bool,
    weak_equality: bool,
) -> PyResult<(Option<Vec<Action>>, FxHashMap<String, String>)> {
    wastar_search(
        ss,
        heuristic,
        0.5,
        timeout,
        max_expanded_states,
        early_termination,
        weak_equality,
    )
}

#[pyfunction]
#[pyo3(signature = (ss, heuristic, timeout=None, max_expanded_states=None, early_termination=false, weak_equality=false))]
pub fn gbfs_search(
    ss: &rustamer_base::SearchSpace,
    heuristic: &Heuristic,
    timeout: Option<f32>,
    max_expanded_states: Option<usize>,
    early_termination: bool,
    weak_equality: bool,
) -> PyResult<(Option<Vec<Action>>, FxHashMap<String, String>)> {
    wastar_search(
        ss,
        heuristic,
        1.0,
        timeout,
        max_expanded_states,
        early_termination,
        weak_equality,
    )
}

#[pyfunction]
#[pyo3(signature = (ss, heuristics, timeout=None, max_expanded_states=None, early_termination=false, weak_equality=false))]
pub fn multiqueue_search(
    ss: &rustamer_base::SearchSpace,
    heuristics: Vec<(Heuristic, f64)>,
    timeout: Option<f32>,
    max_expanded_states: Option<usize>,
    early_termination: bool,
    weak_equality: bool,
) -> PyResult<(Option<Vec<Action>>, FxHashMap<String, String>)> {
    rustamer_base::multiqueue_search(
        ss,
        heuristics,
        timeout,
        max_expanded_states,
        early_termination,
        weak_equality,
    )
}
