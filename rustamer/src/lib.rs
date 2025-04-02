mod expressions;
mod expressions_utils;
mod heuristics;
mod multiqueue;
mod search;
mod search_space;
mod search_state;
mod state_encoder;
mod stn;
mod structures;
mod tn_interpreter;
mod utils;

use pyo3::prelude::*;
use pyo3::types::PyModule;

use expressions::*;
use expressions_utils::*;
use heuristics::*;
use multiqueue::*;
use search::*;
use search_space::*;
use state_encoder::CoreStateEncoder;
use structures::*;

/// A Python module implemented in Rust.
#[pymodule]
fn rustamer(_py: Python, m: Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyExpressionNode>()?;
    m.add_class::<Effect>()?;
    m.add_class::<Timing>()?;
    m.add_class::<Event>()?;
    m.add_class::<SearchSpace>()?;
    m.add_class::<Heuristic>()?;
    m.add_class::<CoreStateEncoder>()?;

    m.add_function(wrap_pyfunction!(make_operator_node, &m)?)?;
    m.add_function(wrap_pyfunction!(make_bool_constant_node, &m)?)?;
    m.add_function(wrap_pyfunction!(make_int_constant_node, &m)?)?;
    m.add_function(wrap_pyfunction!(make_rational_constant_node, &m)?)?;
    m.add_function(wrap_pyfunction!(make_object_node, &m)?)?;
    m.add_function(wrap_pyfunction!(make_fluent_node, &m)?)?;
    m.add_function(wrap_pyfunction!(shift_expression, &m)?)?;
    m.add_function(wrap_pyfunction!(wastar_search, &m)?)?;
    m.add_function(wrap_pyfunction!(multiqueue_search, &m)?)?;
    m.add_function(wrap_pyfunction!(astar_search, &m)?)?;
    m.add_function(wrap_pyfunction!(gbfs_search, &m)?)?;
    m.add_function(wrap_pyfunction!(ehc_search, &m)?)?;
    m.add_function(wrap_pyfunction!(bfs_search, &m)?)?;
    m.add_function(wrap_pyfunction!(dfs_search, &m)?)?;
    m.add_function(wrap_pyfunction!(evaluate, &m)?)?;
    m.add_function(wrap_pyfunction!(simplify, &m)?)?;

    Ok(())
}
