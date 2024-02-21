mod stn;
mod utils;
mod expressions;
mod structures;
mod search_space;
mod search;
mod heuristics;
mod state_encoder;

use pyo3::prelude::*;

use heuristics::*;
use search::*;
use search_space::*;
use expressions::*;
use structures::*;
use state_encoder::CoreStateEncoder;


/// A Python module implemented in Rust.
#[pymodule]
fn rustamer(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyExpressionNode>()?;
    m.add_class::<Effect>()?;
    m.add_class::<Timing>()?;
    m.add_class::<Event>()?;
    m.add_class::<SearchSpace>()?;
    m.add_class::<Heuristic>()?;
    m.add_class::<CoreStateEncoder>()?;
    m.add_function(wrap_pyfunction!(make_operator_node, m)?)?;
    m.add_function(wrap_pyfunction!(make_bool_constant_node, m)?)?;
    m.add_function(wrap_pyfunction!(make_int_constant_node, m)?)?;
    m.add_function(wrap_pyfunction!(make_rational_constant_node, m)?)?;
    m.add_function(wrap_pyfunction!(make_object_node, m)?)?;
    m.add_function(wrap_pyfunction!(make_fluent_node, m)?)?;
    m.add_function(wrap_pyfunction!(shift_expression, m)?)?;
    m.add_function(wrap_pyfunction!(wastar_search, m)?)?;
    m.add_function(wrap_pyfunction!(astar_search, m)?)?;
    m.add_function(wrap_pyfunction!(gbfs_search, m)?)?;
    m.add_function(wrap_pyfunction!(ehc_search, m)?)?;
    m.add_function(wrap_pyfunction!(bfs_search, m)?)?;
    m.add_function(wrap_pyfunction!(dfs_search, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate, m)?)?;
    Ok(())
}
