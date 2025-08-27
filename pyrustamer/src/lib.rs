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


use pyo3::prelude::*;
use pyo3::types::PyModule;

use rustamerlib;

/// A Python module implemented in Rust.
#[pymodule]
fn rustamer(_py: Python, m: Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<rustamerlib::PyExpressionNode>()?;
    m.add_class::<rustamerlib::Effect>()?;
    m.add_class::<rustamerlib::Timing>()?;
    m.add_class::<rustamerlib::Event>()?;
    m.add_class::<rustamerlib::SearchSpace>()?;
    m.add_class::<rustamerlib::Heuristic>()?;

    m.add_function(wrap_pyfunction!(rustamerlib::make_operator_node, &m)?)?;
    m.add_function(wrap_pyfunction!(rustamerlib::make_bool_constant_node, &m)?)?;
    m.add_function(wrap_pyfunction!(rustamerlib::make_int_constant_node, &m)?)?;
    m.add_function(wrap_pyfunction!(rustamerlib::make_rational_constant_node, &m)?)?;
    m.add_function(wrap_pyfunction!(rustamerlib::make_object_node, &m)?)?;
    m.add_function(wrap_pyfunction!(rustamerlib::make_fluent_node, &m)?)?;
    m.add_function(wrap_pyfunction!(rustamerlib::shift_expression, &m)?)?;
    m.add_function(wrap_pyfunction!(rustamerlib::wastar_search, &m)?)?;
    m.add_function(wrap_pyfunction!(rustamerlib::multiqueue_search, &m)?)?;
    m.add_function(wrap_pyfunction!(rustamerlib::astar_search, &m)?)?;
    m.add_function(wrap_pyfunction!(rustamerlib::gbfs_search, &m)?)?;
    m.add_function(wrap_pyfunction!(rustamerlib::ehc_search, &m)?)?;
    m.add_function(wrap_pyfunction!(rustamerlib::bfs_search, &m)?)?;
    m.add_function(wrap_pyfunction!(rustamerlib::dfs_search, &m)?)?;
    m.add_function(wrap_pyfunction!(rustamerlib::evaluate, &m)?)?;
    m.add_function(wrap_pyfunction!(rustamerlib::simplify, &m)?)?;

    Ok(())
}
