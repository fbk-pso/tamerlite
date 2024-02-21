use num::rational::BigRational;

use pyo3::prelude::*;

use super::expressions::PyExpressionNode;
use super::utils::get_big_rational;


#[pyclass(frozen)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Effect {
    pub fluent: String,
    pub value: Vec<PyExpressionNode>,
}

#[pymethods]
impl Effect {
    #[new]
    fn new(fluent: String, value: Vec<PyExpressionNode>) -> Self {
        Effect { fluent, value }
    }
    
    #[getter]
    fn fluent(&self) -> String {
        self.fluent.to_string()
    }
    
    #[getter]
    fn value(&self) -> Vec<PyExpressionNode> {
        self.value.clone()
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self)
    }
}

#[pyclass(frozen)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Timing {
    start: bool,
    pub delay: BigRational,
}

#[pymethods]
impl Timing {
    #[new]
    fn new(start: bool, #[pyo3(from_py_with = "get_big_rational")] delay: BigRational) -> Self {
        Timing { start, delay }
    }

    pub fn is_from_start(&self) -> bool {
        self.start
    }

    pub fn is_from_end(&self) -> bool {
        !self.start
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self)
    }
}

#[pyclass(frozen)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Event {
    pub action: String,
    pub conditions: Vec<PyExpressionNode>,
    pub start_conditions: Vec<Vec<PyExpressionNode>>,
    pub end_conditions: Vec<Vec<PyExpressionNode>>,
    pub effects: Vec<Effect>,
}

#[pymethods]
impl Event {
    #[new]
    fn new(
        action: String,
        conditions: Vec<PyExpressionNode>,
        start_conditions: Vec<Vec<PyExpressionNode>>,
        end_conditions: Vec<Vec<PyExpressionNode>>,
        effects: Vec<Effect>,
    ) -> Self {
        Event {
            action: action,
            conditions: conditions,
            start_conditions: start_conditions,
            end_conditions: end_conditions,
            effects: effects,
        }
    }

    #[getter]
    fn conditions(&self) -> Vec<PyExpressionNode> {
        self.conditions.clone()
    }

    #[getter]
    fn effects(&self) -> Vec<Effect> {
        self.effects.clone()
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self)
    }
}