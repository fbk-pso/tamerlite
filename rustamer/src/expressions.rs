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

use num::BigInt;
use num_rational::BigRational;
use pyo3::{exceptions::PyValueError, prelude::*};
use std::collections::HashMap;

use crate::utils::integer_to_i32;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ExpressionNode {
    Bool(bool),
    Int(BigInt),
    Rational(BigRational),
    Fluent(String),
    Object(String),
    And(Vec<usize>),
    Not(usize),
    Equals(usize, usize),
    LE(usize, usize),
    LT(usize, usize),
    Plus(Vec<usize>),
    Minus(usize, usize),
    Times(Vec<usize>),
    Div(usize, usize),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Expression {
    id: usize,
}

#[derive(Clone, Debug)]
pub struct ExpressionManager {
    all_expressions: Vec<Vec<ExpressionNode>>,
    expression2id: HashMap<Vec<ExpressionNode>, Expression>,
}

impl ExpressionManager {
    pub fn new() -> ExpressionManager {
        ExpressionManager {
            all_expressions: vec![],
            expression2id: HashMap::new(),
        }
    }

    // pub fn get(&self, expr: &Expression) -> Option<&Vec<ExpressionNode>> {
    //     if expr.id < self.all_expressions.len() {
    //         Some(&self.all_expressions[expr.id])
    //     }
    //     else {
    //         None
    //     }
    // }

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

pub fn get_rational_from_expression_node(exp: &ExpressionNode) -> PyResult<BigRational> {
    if let ExpressionNode::Int(v) = exp {
        Ok(BigRational::from_integer(v.clone()))
    } else if let ExpressionNode::Rational(v) = exp {
        Ok(v.clone())
    } else {
        Err(PyValueError::new_err("Expected a number!"))
    }
}

#[pyclass(frozen, name = "ExpressionNode")]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PyExpressionNode {
    pub v: ExpressionNode,
}

#[pymethods]
impl PyExpressionNode {
    #[getter]
    fn fluent(&self) -> Option<String> {
        if let ExpressionNode::Fluent(v) = &self.v {
            Some(v.to_string())
        } else {
            None
        }
    }

    #[getter]
    fn object(&self) -> Option<String> {
        if let ExpressionNode::Object(v) = &self.v {
            Some(v.to_string())
        } else {
            None
        }
    }

    #[getter]
    fn bool_constant(&self) -> Option<bool> {
        if let ExpressionNode::Bool(v) = &self.v {
            Some(*v)
        } else {
            None
        }
    }

    #[getter]
    fn int_constant(&self) -> Option<i32> {
        if let ExpressionNode::Int(v) = &self.v {
            Some(integer_to_i32(v))
        } else {
            None
        }
    }

    #[getter]
    fn real_constant(&self) -> Option<(i32, i32)> {
        if let ExpressionNode::Rational(v) = &self.v {
            Some((integer_to_i32(v.numer()), integer_to_i32(v.denom())))
        } else {
            None
        }
    }

    fn __repr__(&self) -> String {
        format!("{:?}", &self.v)
    }
}

pub fn make_operator(kind: String, operands: Vec<usize>) -> PyResult<ExpressionNode> {
    match kind.as_str() {
        "and" => Ok(ExpressionNode::And(operands)),
        "not" => Ok(ExpressionNode::Not(operands[0])),
        "==" => Ok(ExpressionNode::Equals(operands[0], operands[1])),
        "<=" => Ok(ExpressionNode::LE(operands[0], operands[1])),
        "<" => Ok(ExpressionNode::LT(operands[0], operands[1])),
        "+" => Ok(ExpressionNode::Plus(operands)),
        "-" => Ok(ExpressionNode::Minus(operands[0], operands[1])),
        "*" => Ok(ExpressionNode::Times(operands)),
        "/" => Ok(ExpressionNode::Div(operands[0], operands[1])),
        &_ => Err(PyValueError::new_err(
            "Unknown operator: ".to_owned() + kind.as_str(),
        )),
    }
}

#[pyfunction]
pub fn make_operator_node(kind: String, operands: Vec<usize>) -> PyResult<PyExpressionNode> {
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
pub fn make_int_constant_node(v: i32) -> PyExpressionNode {
    PyExpressionNode {
        v: ExpressionNode::Int(super::utils::mk_integer(v)),
    }
}

#[pyfunction]
pub fn make_rational_constant_node(numerator: i32, denominator: i32) -> PyExpressionNode {
    PyExpressionNode {
        v: ExpressionNode::Rational(super::utils::mk_rational(numerator, denominator)),
    }
}

#[pyfunction]
pub fn make_object_node(name: String) -> PyExpressionNode {
    PyExpressionNode {
        v: ExpressionNode::Object(name),
    }
}

#[pyfunction]
pub fn make_fluent_node(name: String) -> PyExpressionNode {
    PyExpressionNode {
        v: ExpressionNode::Fluent(name),
    }
}
