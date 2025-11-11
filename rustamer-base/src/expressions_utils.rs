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

use super::expressions::*;
use super::search_state::*;
use super::utils::*;
use num_rational::BigRational;
use pyo3::{exceptions::PyException, prelude::*};
use rustc_hash::FxHashMap;
use std::vec::Vec;

pub fn do_shift(
    e: &ExpressionNode,
    offset: usize,
    is_negative: bool,
) -> Result<ExpressionNode, ArithmeticError> {
    Ok(match e {
        ExpressionNode::And(v) => ExpressionNode::And(
            v.iter()
                .map(|&o| checked_add_sub(o, offset, is_negative))
                .collect::<Result<_, _>>()?,
        ),
        ExpressionNode::Plus(v) => ExpressionNode::Plus(
            v.iter()
                .map(|&o| checked_add_sub(o, offset, is_negative))
                .collect::<Result<_, _>>()?,
        ),
        ExpressionNode::Times(v) => ExpressionNode::Times(
            v.iter()
                .map(|&o| checked_add_sub(o, offset, is_negative))
                .collect::<Result<_, _>>()?,
        ),
        ExpressionNode::Not(o) => ExpressionNode::Not(checked_add_sub(*o, offset, is_negative)?),
        ExpressionNode::Equals(o1, o2) => ExpressionNode::Equals(
            checked_add_sub(*o1, offset, is_negative)?,
            checked_add_sub(*o2, offset, is_negative)?,
        ),
        ExpressionNode::LE(o1, o2) => ExpressionNode::LE(
            checked_add_sub(*o1, offset, is_negative)?,
            checked_add_sub(*o2, offset, is_negative)?,
        ),
        ExpressionNode::LT(o1, o2) => ExpressionNode::LT(
            checked_add_sub(*o1, offset, is_negative)?,
            checked_add_sub(*o2, offset, is_negative)?,
        ),
        ExpressionNode::Minus(o1, o2) => ExpressionNode::Minus(
            checked_add_sub(*o1, offset, is_negative)?,
            checked_add_sub(*o2, offset, is_negative)?,
        ),
        ExpressionNode::Div(o1, o2) => ExpressionNode::Div(
            checked_add_sub(*o1, offset, is_negative)?,
            checked_add_sub(*o2, offset, is_negative)?,
        ),
        other => other.clone(),
    })
}

#[pyfunction]
pub fn shift_expression(
    exp: Vec<PyExpressionNode>,
    offset: usize,
) -> PyResult<Vec<PyExpressionNode>> {
    let shifted: Vec<ExpressionNode> = exp
        .iter()
        .map(|e| do_shift(&e.v, offset, false))
        .collect::<Result<_, _>>()
        .map_err(|e| PyException::new_err(format!("{:?}", e)))?;
    Ok(shifted
        .into_iter()
        .map(|v| PyExpressionNode { v })
        .collect())
}

pub fn split_expression(exp: &Vec<ExpressionNode>) -> PyResult<Vec<Vec<ExpressionNode>>> {
    let mut res = Vec::new();
    if let Some(g) = exp.last() {
        if let ExpressionNode::And(v) = g {
            let mut last = 0;
            for i in v.iter() {
                let mut new_exp = Vec::new();
                for e in exp.iter().skip(last).take(*i + 1 - last) {
                    match e {
                        ExpressionNode::And(v) => {
                            let operands = v.iter().map(|&j| j - last).collect();
                            new_exp.push(make_operator("and".to_string(), operands)?);
                        }
                        ExpressionNode::Plus(v) => {
                            let operands = v.iter().map(|&j| j - last).collect();
                            new_exp.push(make_operator("+".to_string(), operands)?);
                        }
                        ExpressionNode::Times(v) => {
                            let operands = v.iter().map(|&j| j - last).collect();
                            new_exp.push(make_operator("*".to_string(), operands)?);
                        }
                        ExpressionNode::Equals(i1, i2) => {
                            new_exp
                                .push(make_operator("==".to_string(), vec![i1 - last, i2 - last])?);
                        }
                        ExpressionNode::LE(i1, i2) => {
                            new_exp
                                .push(make_operator("<=".to_string(), vec![i1 - last, i2 - last])?);
                        }
                        ExpressionNode::LT(i1, i2) => {
                            new_exp
                                .push(make_operator("<".to_string(), vec![i1 - last, i2 - last])?);
                        }
                        ExpressionNode::Minus(i1, i2) => {
                            new_exp
                                .push(make_operator("-".to_string(), vec![i1 - last, i2 - last])?);
                        }
                        ExpressionNode::Div(i1, i2) => {
                            new_exp
                                .push(make_operator("/".to_string(), vec![i1 - last, i2 - last])?);
                        }
                        ExpressionNode::Not(i) => {
                            new_exp.push(make_operator("not".to_string(), vec![i - last])?);
                        }
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

#[pyfunction]
pub fn simplify(
    exp: Vec<PyExpressionNode>,
    assignments: FxHashMap<usize, PyExpressionNode>,
) -> PyResult<Vec<PyExpressionNode>> {
    // This function simplifies the given expression using the given assignments

    // We iterate over the expression elements and we store the simplified value in the res vector
    let mut res: Vec<ExpressionNode> = Vec::with_capacity(exp.len());
    for e in exp {
        let value = match e.v {
            ExpressionNode::And(operands) => {
                let mut is_false = false;
                let mut new_operands = Vec::new();
                for i in operands {
                    if let ExpressionNode::Bool(v) = res[i] {
                        if !v {
                            is_false = true;
                            break;
                        }
                    } else {
                        new_operands.push(i);
                    }
                }
                if is_false {
                    ExpressionNode::Bool(false)
                } else {
                    if new_operands.is_empty() {
                        ExpressionNode::Bool(true)
                    } else if new_operands.len() == 1 {
                        res[new_operands[0]].clone()
                    } else {
                        ExpressionNode::And(new_operands)
                    }
                }
            }
            ExpressionNode::Or(operands) => {
                let mut is_true = false;
                let mut new_operands = Vec::new();
                for i in operands {
                    if let ExpressionNode::Bool(v) = res[i] {
                        if v {
                            is_true = true;
                            break;
                        }
                    } else {
                        new_operands.push(i);
                    }
                }
                if is_true {
                    ExpressionNode::Bool(true)
                } else {
                    if new_operands.is_empty() {
                        ExpressionNode::Bool(false)
                    } else if new_operands.len() == 1 {
                        res[new_operands[0]].clone()
                    } else {
                        ExpressionNode::And(new_operands)
                    }
                }
            }
            ExpressionNode::Not(p) => {
                if let ExpressionNode::Bool(v) = res[p] {
                    ExpressionNode::Bool(!v)
                } else {
                    e.v
                }
            }
            ExpressionNode::Equals(p1, p2) => {
                if res[p1] == res[p2] {
                    ExpressionNode::Bool(true)
                } else {
                    let val1 = get_rational_from_expression_node(&res[p1]);
                    let val2 = get_rational_from_expression_node(&res[p2]);
                    if val1.is_ok() && val2.is_ok() {
                        ExpressionNode::Bool(val1.unwrap() == val2.unwrap())
                    } else {
                        e.v
                    }
                }
            }
            ExpressionNode::LE(p1, p2) => {
                let val1 = get_rational_from_expression_node(&res[p1]);
                let val2 = get_rational_from_expression_node(&res[p2]);
                if val1.is_ok() && val2.is_ok() {
                    ExpressionNode::Bool(val1.unwrap() <= val2.unwrap())
                } else {
                    e.v
                }
            }
            ExpressionNode::LT(p1, p2) => {
                let val1 = get_rational_from_expression_node(&res[p1]);
                let val2 = get_rational_from_expression_node(&res[p2]);
                if val1.is_ok() && val2.is_ok() {
                    ExpressionNode::Bool(val1.unwrap() < val2.unwrap())
                } else {
                    e.v
                }
            }
            ExpressionNode::Plus(ref v) => {
                let mut to_simplified = true;
                let mut r = BigRational::from_integer(mk_integer(0));
                for p in v.iter() {
                    let val = get_rational_from_expression_node(&res[*p]);
                    if val.is_ok() {
                        r += val.unwrap();
                    } else {
                        to_simplified = false;
                        break;
                    }
                }
                if to_simplified {
                    if r.is_integer() {
                        ExpressionNode::Int(Box::new(r.to_integer()))
                    } else {
                        ExpressionNode::Rational(Box::new(r))
                    }
                } else {
                    e.v
                }
            }
            ExpressionNode::Minus(p1, p2) => {
                let val1 = get_rational_from_expression_node(&res[p1]);
                let val2 = get_rational_from_expression_node(&res[p2]);
                if val1.is_ok() && val2.is_ok() {
                    let r = val1.unwrap() - val2.unwrap();
                    if r.is_integer() {
                        ExpressionNode::Int(Box::new(r.to_integer()))
                    } else {
                        ExpressionNode::Rational(Box::new(r))
                    }
                } else {
                    e.v
                }
            }
            ExpressionNode::Times(ref v) => {
                let mut to_simplified = true;
                let mut r = BigRational::from_integer(mk_integer(1));
                for p in v.iter() {
                    let val = get_rational_from_expression_node(&res[*p]);
                    if val.is_ok() {
                        r *= val.unwrap();
                    } else {
                        to_simplified = false;
                        break;
                    }
                }
                if to_simplified {
                    if r.is_integer() {
                        ExpressionNode::Int(Box::new(r.to_integer()))
                    } else {
                        ExpressionNode::Rational(Box::new(r))
                    }
                } else {
                    e.v
                }
            }
            ExpressionNode::Div(p1, p2) => {
                let val1 = get_rational_from_expression_node(&res[p1]);
                let val2 = get_rational_from_expression_node(&res[p2]);
                if val1.is_ok() && val2.is_ok() {
                    let r = val1.unwrap() / val2.unwrap();
                    if r.is_integer() {
                        ExpressionNode::Int(Box::new(r.to_integer()))
                    } else {
                        ExpressionNode::Rational(Box::new(r))
                    }
                } else {
                    e.v
                }
            }
            ExpressionNode::Fluent(s) => {
                if let Some(v) = assignments.get(&s) {
                    v.v.clone()
                } else {
                    e.v
                }
            }
            other => other,
        };
        res.push(value);
    }

    // Keep only the nodes reachable from the root using a depth-first search
    let mut final_res = Vec::new();
    let mut stack = vec![(res.len() - 1, false)];
    let mut operands_stack = Vec::new();
    while let Some((idx, processed)) = stack.pop() {
        match &res[idx] {
            ExpressionNode::Bool(_)
            | ExpressionNode::Int(_)
            | ExpressionNode::Rational(_)
            | ExpressionNode::Fluent(_)
            | ExpressionNode::Object(_) => {
                operands_stack.push(final_res.len());
                final_res.push(PyExpressionNode {
                    v: res[idx].clone(),
                });
            }
            ExpressionNode::And(operands)
            | ExpressionNode::Or(operands)
            | ExpressionNode::Plus(operands)
            | ExpressionNode::Times(operands) => {
                if processed {
                    let new_operands = operands
                        .iter()
                        .map(|_| operands_stack.pop().unwrap())
                        .rev()
                        .collect();
                    operands_stack.push(final_res.len());
                    let exp_node = match &res[idx] {
                        ExpressionNode::And(_) => ExpressionNode::And(new_operands),
                        ExpressionNode::Or(_) => ExpressionNode::Or(new_operands),
                        ExpressionNode::Plus(_) => ExpressionNode::Plus(new_operands),
                        ExpressionNode::Times(_) => ExpressionNode::Times(new_operands),
                        _ => unreachable!(),
                    };
                    final_res.push(PyExpressionNode { v: exp_node });
                } else {
                    stack.push((idx, true));
                    for i in operands.iter().rev() {
                        stack.push((*i, false));
                    }
                }
            }
            ExpressionNode::Not(operand) => {
                if processed {
                    let new_operand = operands_stack.pop().unwrap();
                    operands_stack.push(final_res.len());
                    final_res.push(PyExpressionNode {
                        v: ExpressionNode::Not(new_operand),
                    });
                } else {
                    stack.push((idx, true));
                    stack.push((*operand, false));
                }
            }
            ExpressionNode::Equals(op1, op2)
            | ExpressionNode::LE(op1, op2)
            | ExpressionNode::LT(op1, op2)
            | ExpressionNode::Minus(op1, op2)
            | ExpressionNode::Div(op1, op2) => {
                if processed {
                    let new_op2 = operands_stack.pop().unwrap();
                    let new_op1 = operands_stack.pop().unwrap();
                    operands_stack.push(final_res.len());
                    let exp_node = match &res[idx] {
                        ExpressionNode::Equals(_, _) => ExpressionNode::Equals(new_op1, new_op2),
                        ExpressionNode::LE(_, _) => ExpressionNode::LE(new_op1, new_op2),
                        ExpressionNode::LT(_, _) => ExpressionNode::LT(new_op1, new_op2),
                        ExpressionNode::Minus(_, _) => ExpressionNode::Minus(new_op1, new_op2),
                        ExpressionNode::Div(_, _) => ExpressionNode::Div(new_op1, new_op2),
                        _ => unreachable!(),
                    };
                    final_res.push(PyExpressionNode { v: exp_node });
                } else {
                    stack.push((idx, true));
                    stack.push((*op2, false));
                    stack.push((*op1, false));
                }
            }
        }
    }

    Ok(final_res)
}

#[pyfunction]
pub fn evaluate(exp: Vec<PyExpressionNode>, state: &State) -> PyResult<PyExpressionNode> {
    Ok(PyExpressionNode {
        v: internal_evaluate(&exp.into_iter().map(|e| e.v).collect(), state)?,
    })
}

pub trait FluentValueTrait {
    fn get_value(&self, fluent: usize) -> &ExpressionNode;
}

pub fn internal_evaluate(
    exp: &Vec<ExpressionNode>,
    fluent_values: &impl FluentValueTrait,
) -> PyResult<ExpressionNode> {
    let mut res: Vec<ExpressionNode> = vec![];
    for e in exp {
        let value = match &e {
            ExpressionNode::And(v) => {
                let val = v.iter().all(|&p| res[p] == ExpressionNode::Bool(true));
                ExpressionNode::Bool(val)
            }
            ExpressionNode::Or(v) => {
                let val = v.iter().any(|&p| res[p] == ExpressionNode::Bool(true));
                ExpressionNode::Bool(val)
            }
            ExpressionNode::Not(p) => ExpressionNode::Bool(ExpressionNode::Bool(false) == res[*p]),
            ExpressionNode::Equals(p1, p2) => ExpressionNode::Bool(res[*p1] == res[*p2]),
            ExpressionNode::LE(p1, p2) => {
                let val1 = get_rational_from_expression_node(&res[*p1])?;
                let val2 = get_rational_from_expression_node(&res[*p2])?;
                ExpressionNode::Bool(val1 <= val2)
            }
            ExpressionNode::LT(p1, p2) => {
                let val1 = get_rational_from_expression_node(&res[*p1])?;
                let val2 = get_rational_from_expression_node(&res[*p2])?;
                ExpressionNode::Bool(val1 < val2)
            }
            ExpressionNode::Plus(v) => {
                let mut r = get_rational_from_expression_node(&res[v[0]])?;
                for p in v.iter().skip(1) {
                    r += get_rational_from_expression_node(&res[*p])?;
                }
                if r.is_integer() {
                    ExpressionNode::Int(Box::new(r.to_integer()))
                } else {
                    ExpressionNode::Rational(Box::new(r))
                }
            }
            ExpressionNode::Minus(p1, p2) => {
                let val1 = get_rational_from_expression_node(&res[*p1])?;
                let val2 = get_rational_from_expression_node(&res[*p2])?;
                let r = val1 - val2;
                if r.is_integer() {
                    ExpressionNode::Int(Box::new(r.to_integer()))
                } else {
                    ExpressionNode::Rational(Box::new(r))
                }
            }
            ExpressionNode::Times(v) => {
                let mut r = get_rational_from_expression_node(&res[v[0]])?;
                for p in v.iter().skip(1) {
                    r *= get_rational_from_expression_node(&res[*p])?;
                }
                if r.is_integer() {
                    ExpressionNode::Int(Box::new(r.to_integer()))
                } else {
                    ExpressionNode::Rational(Box::new(r))
                }
            }
            ExpressionNode::Div(p1, p2) => {
                let val1 = get_rational_from_expression_node(&res[*p1])?;
                let val2 = get_rational_from_expression_node(&res[*p2])?;
                let r = val1 / val2;
                if r.is_integer() {
                    ExpressionNode::Int(Box::new(r.to_integer()))
                } else {
                    ExpressionNode::Rational(Box::new(r))
                }
            }
            ExpressionNode::Fluent(s) => fluent_values.get_value(*s).clone(),
            other => (*other).clone(),
        };
        if res.len() == exp.len() - 1 {
            return Ok(value);
        } else {
            res.push(value);
        }
    }
    Err(PyException::new_err("Unreachable code"))
}
