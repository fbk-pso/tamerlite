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

use num::BigInt;
use num_rational::BigRational;
use pyo3::{exceptions::PyValueError, prelude::*};
use rustc_hash::{FxBuildHasher, FxHashMap};

use crate::interpreted_functions::{register_interpreted_function, IfReturnType};
use crate::structures::{Fluent, Object};
use crate::utils::{big_rational_to_py_fraction, integer_to_f64, rational_to_f64};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ExpressionNode {
    Bool(bool),
    Int(Box<BigInt>),
    Rational(Box<BigRational>),
    Fluent(Fluent),
    Object(Object),
    And(Vec<usize>),
    Or(Vec<usize>),
    Not(usize),
    Equals(usize, usize),
    LE(usize, usize),
    LT(usize, usize),
    Plus(Vec<usize>),
    Minus(usize, usize),
    Times(Vec<usize>),
    Div(usize, usize),
    /// An interpreted-function call. `func_id` indexes `INTERPRETED_FUNCTIONS` rather
    /// than embedding the callable inline: `ExpressionNode` must stay
    /// `Clone + PartialEq + Eq + Hash`, which a raw `Py<PyAny>` cannot
    /// support without a GIL acquisition on every clone.
    InterpretedFunction {
        func_id: usize,
        return_type: IfReturnType,
        operands: Vec<usize>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Expression {
    id: usize,
}

#[derive(Clone, Debug)]
pub struct ExpressionManager {
    all_expressions: Vec<Vec<ExpressionNode>>,
    expression2id: FxHashMap<Vec<ExpressionNode>, Expression>,
}

impl ExpressionManager {
    pub fn new() -> ExpressionManager {
        ExpressionManager {
            all_expressions: vec![],
            expression2id: FxHashMap::with_hasher(FxBuildHasher),
        }
    }

    pub fn force_get(&self, expr: &Expression) -> &Vec<ExpressionNode> {
        &self.all_expressions[expr.id]
    }

    pub fn put(&mut self, expr: &Vec<ExpressionNode>) -> Expression {
        if let Some(x) = self.expression2id.get(expr) {
            *x
        } else {
            let newid = self.all_expressions.len();
            self.all_expressions.push(expr.clone());
            self.expression2id
                .insert(expr.clone(), Expression { id: newid });
            Expression { id: newid }
        }
    }
}

/// Reads an owned `BigRational` out of `exp`.
/// Only use this when the caller actually needs to *own* the result (store
/// it, mutate it, return it).
pub(crate) fn get_rational_from_expression_node(exp: &ExpressionNode) -> PyResult<BigRational> {
    if let ExpressionNode::Int(v) = exp {
        Ok(BigRational::from_integer((**v).clone()))
    } else if let ExpressionNode::Rational(v) = exp {
        Ok((**v).clone())
    } else {
        Err(PyValueError::new_err("Expected a number!"))
    }
}

/// Reads `exp`'s numeric value straight into `f64`, without ever
/// constructing an owned `BigRational`/`BigInt` -- `integer_to_f64`/
/// `rational_to_f64` (`utils.rs`) already take a borrow. For a caller that
/// only needs the value once (not ownership), this is strictly cheaper
/// than `rational_to_f64(&get_rational_from_expression_node(exp)?)`.
pub(crate) fn expression_node_to_f64(exp: &ExpressionNode) -> PyResult<f64> {
    match exp {
        ExpressionNode::Int(v) => Ok(integer_to_f64(v)),
        ExpressionNode::Rational(v) => Ok(rational_to_f64(v)),
        _ => Err(PyValueError::new_err("Expected a number!")),
    }
}

#[pyclass(frozen, name = "ExpressionNode", from_py_object)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PyExpressionNode {
    pub v: ExpressionNode,
}

#[pymethods]
impl PyExpressionNode {
    #[getter]
    fn fluent(&self) -> Option<Fluent> {
        if let ExpressionNode::Fluent(v) = self.v {
            Some(v)
        } else {
            None
        }
    }

    #[getter]
    fn object(&self) -> Option<Object> {
        if let ExpressionNode::Object(v) = self.v {
            Some(v)
        } else {
            None
        }
    }

    #[getter]
    fn bool_constant(&self) -> Option<bool> {
        if let ExpressionNode::Bool(v) = self.v {
            Some(v)
        } else {
            None
        }
    }

    #[getter]
    fn int_constant(&self) -> Option<BigInt> {
        if let ExpressionNode::Int(v) = &self.v {
            Some((**v).clone())
        } else {
            None
        }
    }

    #[getter]
    fn real_constant<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        if let ExpressionNode::Rational(v) = &self.v {
            Ok(Some(big_rational_to_py_fraction(v, py)?))
        } else {
            Ok(None)
        }
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.v)
    }
}

pub fn make_operator(kind: &str, operands: Vec<usize>) -> PyResult<ExpressionNode> {
    match kind {
        "and" => Ok(ExpressionNode::And(operands)),
        "or" => Ok(ExpressionNode::Or(operands)),
        "not" => Ok(ExpressionNode::Not(operands[0])),
        "==" => Ok(ExpressionNode::Equals(operands[0], operands[1])),
        "<=" => Ok(ExpressionNode::LE(operands[0], operands[1])),
        "<" => Ok(ExpressionNode::LT(operands[0], operands[1])),
        "+" => Ok(ExpressionNode::Plus(operands)),
        "-" => Ok(ExpressionNode::Minus(operands[0], operands[1])),
        "*" => Ok(ExpressionNode::Times(operands)),
        "/" => Ok(ExpressionNode::Div(operands[0], operands[1])),
        _ => Err(PyValueError::new_err(format!("Unknown operator: {kind}"))),
    }
}

#[pyfunction]
pub fn make_operator_node(kind: &str, operands: Vec<usize>) -> PyResult<PyExpressionNode> {
    Ok(PyExpressionNode {
        v: make_operator(kind, operands)?,
    })
}

#[pyfunction]
pub fn make_bool_constant_node(v: bool) -> PyExpressionNode {
    PyExpressionNode {
        v: ExpressionNode::Bool(v),
    }
}

#[pyfunction]
pub fn make_int_constant_node(v: BigInt) -> PyExpressionNode {
    PyExpressionNode {
        v: ExpressionNode::Int(Box::new(v)),
    }
}

#[pyfunction]
pub fn make_rational_constant_node(numerator: BigInt, denominator: BigInt) -> PyExpressionNode {
    PyExpressionNode {
        v: ExpressionNode::Rational(Box::new(BigRational::new(numerator, denominator))),
    }
}

#[pyfunction]
pub fn make_object_node(obj: Object) -> PyExpressionNode {
    PyExpressionNode {
        v: ExpressionNode::Object(obj),
    }
}

#[pyfunction]
pub fn make_fluent_node(fluent: Fluent) -> PyExpressionNode {
    PyExpressionNode {
        v: ExpressionNode::Fluent(fluent),
    }
}

#[pyfunction]
pub fn make_interpreted_function_node(
    function: Py<PyAny>,
    return_type: IfReturnType,
    operands: Vec<usize>,
) -> PyExpressionNode {
    let func_id = register_interpreted_function(function);
    PyExpressionNode {
        v: ExpressionNode::InterpretedFunction {
            func_id,
            return_type,
            operands,
        },
    }
}
