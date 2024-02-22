use num_rational::BigRational;
use num::BigInt;
use pyo3::{prelude::*, exceptions::PyValueError};

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
            Some(integer_to_i32(v.clone()))
        } else {
            None
        }
    }

    #[getter]
    fn real_constant(&self) -> Option<(i32, i32)> {
        if let ExpressionNode::Rational(v) = &self.v {
            Some((integer_to_i32(v.numer().clone()), integer_to_i32(v.denom().clone())))
        } else {
            None
        }
    }

    fn __repr__(&self) -> String {
        format!("{:?}", &self.v)
    }
}

impl PyExpressionNode {
    pub fn to_expression_node(&self) -> ExpressionNode {
        self.v.clone()
    }
}

#[pyfunction]
pub fn make_operator_node(kind: String, operands: Vec<usize>) -> PyResult<PyExpressionNode> {
    let translation = match kind.as_str() {
        "and" => Ok(ExpressionNode::And(operands)),
        "not" => Ok(ExpressionNode::Not(operands[0])),
        "==" => Ok(ExpressionNode::Equals(operands[0], operands[1])),
        "<=" => Ok(ExpressionNode::LE(operands[0], operands[1])),
        "<" => Ok(ExpressionNode::LT(operands[0], operands[1])),
        "+" => Ok(ExpressionNode::Plus(operands)),
        "-" => Ok(ExpressionNode::Minus(operands[0], operands[1])),
        "*" => Ok(ExpressionNode::Times(operands)),
        "/" => Ok(ExpressionNode::Div(operands[0], operands[1])),
        &_ => Err("Unknown operator: ".to_owned() + kind.as_str())
    };

    match translation {
        Ok(val) => Ok(PyExpressionNode{v: val}),
        Err(msg) => Err(PyValueError::new_err(msg))
    }
}

#[pyfunction]
pub fn make_bool_constant_node(v: bool) -> PyExpressionNode {
    PyExpressionNode {
        v: ExpressionNode::Bool(v)
    }
}

#[pyfunction]
pub fn make_int_constant_node(v: i32) -> PyExpressionNode {
    PyExpressionNode {
        v: ExpressionNode::Int(super::utils::mk_integer(v))
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
        v: ExpressionNode::Object(name)
    }
}

#[pyfunction]
pub fn make_fluent_node(name: String) -> PyExpressionNode {
    PyExpressionNode {
        v: ExpressionNode::Fluent(name)
    }
}

fn do_shift(e: &ExpressionNode, offset: usize) -> ExpressionNode {
    match e {
        ExpressionNode::And(v) => {
            ExpressionNode::And(v.iter().map(|&o| o + offset).collect())
        },
        ExpressionNode::Plus(v) => {
            ExpressionNode::Plus(v.iter().map(|&o| o + offset).collect())
        },
        ExpressionNode::Times(v) => {
            ExpressionNode::Times(v.iter().map(|&o| o + offset).collect())
        },
        ExpressionNode::Not(o) => {
            ExpressionNode::Not(o+offset)
        },
        ExpressionNode::Equals(o1, o2) => {
            ExpressionNode::Equals(o1+offset, o2+offset)
        },
        ExpressionNode::LE(o1, o2) => {
            ExpressionNode::LE(o1+offset, o2+offset)
        },
        ExpressionNode::LT(o1, o2) => {
            ExpressionNode::LT(o1+offset, o2+offset)
        },
        ExpressionNode::Minus(o1, o2) => {
            ExpressionNode::Minus(o1+offset, o2+offset)
        },
        ExpressionNode::Div(o1, o2) => {
            ExpressionNode::Div(o1+offset, o2+offset)
        },
        other => {
            other.clone()
        }
    }
}

#[pyfunction]
pub fn shift_expression(exp: Vec<PyExpressionNode>, offset: usize) -> Vec<PyExpressionNode> {
    exp.iter().map(|e| PyExpressionNode{v:do_shift(&e.v, offset)}).collect::<Vec<_>>()
}

pub fn split_expression(exp: &Vec<PyExpressionNode>) -> PyResult<Vec<Vec<PyExpressionNode>>> {
    let mut res = Vec::new();
    if let Some(g) = exp.last() {
        if let ExpressionNode::And(v) = g.to_expression_node() {
            let mut last = 0;
            for i in v.iter() {
                let mut new_exp = Vec::new();
                for e in exp.iter().skip(last).take(*i+1-last) {
                    match e.to_expression_node() {
                        ExpressionNode::And(v) => {
                            let operands = v.iter().map(|&j| j - last).collect();
                            new_exp.push(make_operator_node("and".to_string(), operands)?);
                        },
                        ExpressionNode::Plus(v) => {
                            let operands = v.iter().map(|&j| j - last).collect();
                            new_exp.push(make_operator_node("+".to_string(), operands)?);
                        },
                        ExpressionNode::Times(v) => {
                            let operands = v.iter().map(|&j| j - last).collect();
                            new_exp.push(make_operator_node("*".to_string(), operands)?);
                        },
                        ExpressionNode::Equals(i1, i2) => {
                            new_exp.push(make_operator_node("==".to_string(), vec![i1 - last, i2 - last])?);
                        },
                        ExpressionNode::LE(i1, i2) => {
                            new_exp.push(make_operator_node("<=".to_string(), vec![i1 - last, i2 - last])?);
                        },
                        ExpressionNode::LT(i1, i2) => {
                            new_exp.push(make_operator_node("<".to_string(), vec![i1 - last, i2 - last])?);
                        },
                        ExpressionNode::Minus(i1, i2) => {
                            new_exp.push(make_operator_node("-".to_string(), vec![i1 - last, i2 - last])?);
                        },
                        ExpressionNode::Div(i1, i2) => {
                            new_exp.push(make_operator_node("/".to_string(), vec![i1 - last, i2 - last])?);
                        },
                        ExpressionNode::Not(i) => {
                            new_exp.push(make_operator_node("not".to_string(), vec![i - last])?);
                        },
                        _ => {
                            new_exp.push(e.clone());
                        }
                    }
                }
                res.push(new_exp);
                last = i + 1;
            }
        } else {
            return Ok(vec![exp.clone()]);
        }
    }
    Ok(res)
}
