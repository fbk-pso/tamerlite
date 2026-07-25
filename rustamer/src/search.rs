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
use pyo3::exceptions::{PyTimeoutError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rustamer_base;
use rustamer_base::Action;
use rustc_hash::FxHashMap;

#[pyfunction]
#[pyo3(signature = (ss, timeout=None, early_termination=false))]
pub fn bfs_search(
    ss: &rustamer_base::SearchSpace,
    timeout: Option<f32>,
    early_termination: bool,
) -> PyResult<(
    Option<Vec<(Option<String>, Action, Option<String>)>>,
    FxHashMap<String, String>,
)> {
    rustamer_base::bfs_search(ss, timeout, early_termination)
}

#[pyfunction]
#[pyo3(signature = (ss, timeout=None, early_termination=false))]
pub fn dfs_search(
    ss: &rustamer_base::SearchSpace,
    timeout: Option<f32>,
    early_termination: bool,
) -> PyResult<(
    Option<Vec<(Option<String>, Action, Option<String>)>>,
    FxHashMap<String, String>,
)> {
    rustamer_base::dfs_search(ss, timeout, early_termination)
}

#[pyfunction]
#[pyo3(signature = (ss, heuristic, timeout=None, early_termination=false, weak_equality=false, max_len=None))]
pub fn ehc_search(
    py: Python<'_>,
    ss: &rustamer_base::SearchSpace,
    heuristic: &Heuristic,
    timeout: Option<f32>,
    early_termination: bool,
    weak_equality: bool,
    max_len: Option<f64>,
) -> PyResult<(Option<rustamer_base::Plan>, Py<PyDict>)> {
    if max_len.is_some() && timeout.is_none() {
        return Err(PyValueError::new_err(
            "timeout must be provided when max_len is set",
        ));
    }
    if max_len.is_some() && early_termination {
        return Err(PyValueError::new_err(
            "early_termination must be false when max_len is set",
        ));
    }

    let result = rustamer_base::ehc_search(
        ss,
        heuristic,
        timeout,
        early_termination,
        weak_equality,
        max_len,
    )?;
    let metrics = PyDict::new(py);
    for (key, value) in result.metrics {
        metrics.set_item(key, value)?;
    }
    if let Some(plans) = result.plans {
        metrics.set_item("plans", plans)?;
    }
    let metrics = metrics.unbind();
    if result.timed_out {
        Err(PyTimeoutError::new_err(metrics))
    } else {
        Ok((result.plan, metrics))
    }
}

#[pyfunction]
#[pyo3(signature = (ss, heuristic, weight, timeout=None, early_termination=false, weak_equality=false, max_len=None))]
pub fn wastar_search(
    py: Python<'_>,
    ss: &rustamer_base::SearchSpace,
    heuristic: &Heuristic,
    weight: f64,
    timeout: Option<f32>,
    early_termination: bool,
    weak_equality: bool,
    max_len: Option<f64>,
) -> PyResult<(Option<rustamer_base::Plan>, Py<PyDict>)> {
    run_wastar_search(
        py,
        ss,
        heuristic,
        weight,
        timeout,
        early_termination,
        weak_equality,
        max_len,
    )
}

fn run_wastar_search(
    py: Python<'_>,
    ss: &rustamer_base::SearchSpace,
    heuristic: &Heuristic,
    weight: f64,
    timeout: Option<f32>,
    early_termination: bool,
    weak_equality: bool,
    max_len: Option<f64>,
) -> PyResult<(Option<rustamer_base::Plan>, Py<PyDict>)> {
    if max_len.is_some() && timeout.is_none() {
        return Err(PyValueError::new_err(
            "timeout must be provided when max_len is set",
        ));
    }
    if max_len.is_some() && early_termination {
        return Err(PyValueError::new_err(
            "early_termination must be false when max_len is set",
        ));
    }

    let result = rustamer_base::wastar_search(
        ss,
        heuristic,
        weight,
        timeout,
        early_termination,
        weak_equality,
        max_len,
    )?;
    let metrics = PyDict::new(py);
    for (key, value) in result.metrics {
        metrics.set_item(key, value)?;
    }
    if let Some(plans) = result.plans {
        metrics.set_item("plans", plans)?;
    }
    let metrics = metrics.unbind();
    if result.timed_out {
        Err(PyTimeoutError::new_err(metrics))
    } else {
        Ok((result.plan, metrics))
    }
}

#[pyfunction]
#[pyo3(signature = (ss, heuristic, weight, timeout=None, early_termination=false, weak_equality=false, max_len=None))]
pub fn wastar_search_memory_bounded(
    py: Python<'_>,
    ss: &rustamer_base::SearchSpace,
    heuristic: &Heuristic,
    weight: f64,
    timeout: Option<f32>,
    early_termination: bool,
    weak_equality: bool,
    max_len: Option<f64>,
) -> PyResult<(Option<rustamer_base::Plan>, Py<PyDict>)> {
    run_wastar_search_memory_bounded(
        py,
        ss,
        heuristic,
        weight,
        timeout,
        early_termination,
        weak_equality,
        max_len,
    )
}

fn run_wastar_search_memory_bounded(
    py: Python<'_>,
    ss: &rustamer_base::SearchSpace,
    heuristic: &Heuristic,
    weight: f64,
    timeout: Option<f32>,
    early_termination: bool,
    weak_equality: bool,
    max_len: Option<f64>,
) -> PyResult<(Option<rustamer_base::Plan>, Py<PyDict>)> {
    if max_len.is_some() && timeout.is_none() {
        return Err(PyValueError::new_err(
            "timeout must be provided when max_len is set",
        ));
    }
    if max_len.is_some() && early_termination {
        return Err(PyValueError::new_err(
            "early_termination must be false when max_len is set",
        ));
    }

    let result = rustamer_base::wastar_search_memory_bounded(
        ss,
        heuristic,
        weight,
        timeout,
        early_termination,
        weak_equality,
        max_len,
    )?;
    let metrics = PyDict::new(py);
    for (key, value) in result.metrics {
        metrics.set_item(key, value)?;
    }
    if let Some(plans) = result.plans {
        metrics.set_item("plans", plans)?;
    }
    let metrics = metrics.unbind();
    if result.timed_out {
        Err(PyTimeoutError::new_err(metrics))
    } else {
        Ok((result.plan, metrics))
    }
}

#[pyfunction]
#[pyo3(signature = (ss, heuristic, timeout=None, early_termination=false, weak_equality=false))]
pub fn astar_search_memory_bounded(
    py: Python<'_>,
    ss: &rustamer_base::SearchSpace,
    heuristic: &Heuristic,
    timeout: Option<f32>,
    early_termination: bool,
    weak_equality: bool,
) -> PyResult<(Option<rustamer_base::Plan>, Py<PyDict>)> {
    run_wastar_search_memory_bounded(
        py,
        ss,
        heuristic,
        0.5,
        timeout,
        early_termination,
        weak_equality,
        None,
    )
}

#[pyfunction]
#[pyo3(signature = (ss, heuristic, timeout=None, early_termination=false, weak_equality=false))]
pub fn gbfs_search_memory_bounded(
    py: Python<'_>,
    ss: &rustamer_base::SearchSpace,
    heuristic: &Heuristic,
    timeout: Option<f32>,
    early_termination: bool,
    weak_equality: bool,
) -> PyResult<(Option<rustamer_base::Plan>, Py<PyDict>)> {
    run_wastar_search_memory_bounded(
        py,
        ss,
        heuristic,
        1.0,
        timeout,
        early_termination,
        weak_equality,
        None,
    )
}

#[pyfunction]
#[pyo3(signature = (ss, heuristic, timeout=None, early_termination=false, weak_equality=false))]
pub fn astar_search(
    py: Python<'_>,
    ss: &rustamer_base::SearchSpace,
    heuristic: &Heuristic,
    timeout: Option<f32>,
    early_termination: bool,
    weak_equality: bool,
) -> PyResult<(
    Option<Vec<(Option<String>, Action, Option<String>)>>,
    Py<PyDict>,
)> {
    run_wastar_search(
        py,
        ss,
        heuristic,
        0.5,
        timeout,
        early_termination,
        weak_equality,
        None,
    )
}

#[pyfunction]
#[pyo3(signature = (ss, heuristic, timeout=None, early_termination=false, weak_equality=false))]
pub fn gbfs_search(
    py: Python<'_>,
    ss: &rustamer_base::SearchSpace,
    heuristic: &Heuristic,
    timeout: Option<f32>,
    early_termination: bool,
    weak_equality: bool,
) -> PyResult<(
    Option<Vec<(Option<String>, Action, Option<String>)>>,
    Py<PyDict>,
)> {
    run_wastar_search(
        py,
        ss,
        heuristic,
        1.0,
        timeout,
        early_termination,
        weak_equality,
        None,
    )
}

#[pyfunction]
#[pyo3(signature = (ss, heuristics, timeout=None, early_termination=false, weak_equality=false))]
pub fn multiqueue_search(
    ss: &rustamer_base::SearchSpace,
    heuristics: Vec<(Heuristic, f64)>,
    timeout: Option<f32>,
    early_termination: bool,
    weak_equality: bool,
) -> PyResult<(
    Option<Vec<(Option<String>, Action, Option<String>)>>,
    FxHashMap<String, String>,
)> {
    rustamer_base::multiqueue_search(ss, heuristics, timeout, early_termination, weak_equality)
}
