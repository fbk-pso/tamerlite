use num_rational::BigRational;
use num::BigInt;
use pyo3::{prelude::*, exceptions::PyValueError};
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

#[derive(Clone,Copy,Debug,PartialEq,Eq,Hash)]
pub struct Expression {
    id: usize
}

#[derive(Clone, Debug)]
pub struct ExpressionManager {
    all_expressions: Vec<Vec<ExpressionNode>>,
    expression2id: HashMap<Vec<ExpressionNode>, Expression>
}

impl ExpressionManager {
    pub fn new() -> ExpressionManager {
        ExpressionManager{all_expressions: vec![], expression2id: HashMap::new()}
    }

    pub fn get(&self, expr: &Expression) -> Option<&Vec<ExpressionNode>> {
        if expr.id < self.all_expressions.len() {
            Some(&self.all_expressions[expr.id])
        }
        else {
            None
        }
    }

    pub fn force_get(&self, expr: &Expression) -> &Vec<ExpressionNode> {
        &self.all_expressions[expr.id]
    }

    pub fn put(&mut self, expr:&Vec<ExpressionNode>) -> Expression {
        if let Some(x) = self.expression2id.get(expr) {
            *x
        }
        else {
            let newid = self.all_expressions.len();
            self.all_expressions.push(expr.clone());
            self.expression2id.insert(expr.clone(), Expression{id:newid});
            Expression{id:newid}
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
        &_ => Err(PyValueError::new_err("Unknown operator: ".to_owned() + kind.as_str()))
    }
}

#[pyfunction]
pub fn make_operator_node(kind: String, operands: Vec<usize>) -> PyResult<PyExpressionNode> {
    Ok(PyExpressionNode{v: make_operator(kind, operands)?})
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

pub fn split_expression(exp: &Vec<ExpressionNode>) -> PyResult<Vec<Vec<ExpressionNode>>> {
    let mut res = Vec::new();
    if let Some(g) = exp.last() {
        if let ExpressionNode::And(v) = g {
            let mut last = 0;
            for i in v.iter() {
                let mut new_exp = Vec::new();
                for e in exp.iter().skip(last).take(*i+1-last) {
                    match e {
                        ExpressionNode::And(v) => {
                            let operands = v.iter().map(|&j| j - last).collect();
                            new_exp.push(make_operator("and".to_string(), operands)?);
                        },
                        ExpressionNode::Plus(v) => {
                            let operands = v.iter().map(|&j| j - last).collect();
                            new_exp.push(make_operator("+".to_string(), operands)?);
                        },
                        ExpressionNode::Times(v) => {
                            let operands = v.iter().map(|&j| j - last).collect();
                            new_exp.push(make_operator("*".to_string(), operands)?);
                        },
                        ExpressionNode::Equals(i1, i2) => {
                            new_exp.push(make_operator("==".to_string(), vec![i1 - last, i2 - last])?);
                        },
                        ExpressionNode::LE(i1, i2) => {
                            new_exp.push(make_operator("<=".to_string(), vec![i1 - last, i2 - last])?);
                        },
                        ExpressionNode::LT(i1, i2) => {
                            new_exp.push(make_operator("<".to_string(), vec![i1 - last, i2 - last])?);
                        },
                        ExpressionNode::Minus(i1, i2) => {
                            new_exp.push(make_operator("-".to_string(), vec![i1 - last, i2 - last])?);
                        },
                        ExpressionNode::Div(i1, i2) => {
                            new_exp.push(make_operator("/".to_string(), vec![i1 - last, i2 - last])?);
                        },
                        ExpressionNode::Not(i) => {
                            new_exp.push(make_operator("not".to_string(), vec![i - last])?);
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
