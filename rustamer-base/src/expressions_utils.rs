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

use super::expressions::*;
use super::search_state::*;
use super::utils::*;
use num_rational::BigRational;
use pyo3::{exceptions::PyException, prelude::*};
use std::{collections::HashMap, vec::Vec};

fn do_shift(e: &ExpressionNode, offset: usize) -> ExpressionNode {
    match e {
        ExpressionNode::And(v) => ExpressionNode::And(v.iter().map(|&o| o + offset).collect()),
        ExpressionNode::Plus(v) => ExpressionNode::Plus(v.iter().map(|&o| o + offset).collect()),
        ExpressionNode::Times(v) => ExpressionNode::Times(v.iter().map(|&o| o + offset).collect()),
        ExpressionNode::Not(o) => ExpressionNode::Not(o + offset),
        ExpressionNode::Equals(o1, o2) => ExpressionNode::Equals(o1 + offset, o2 + offset),
        ExpressionNode::LE(o1, o2) => ExpressionNode::LE(o1 + offset, o2 + offset),
        ExpressionNode::LT(o1, o2) => ExpressionNode::LT(o1 + offset, o2 + offset),
        ExpressionNode::Minus(o1, o2) => ExpressionNode::Minus(o1 + offset, o2 + offset),
        ExpressionNode::Div(o1, o2) => ExpressionNode::Div(o1 + offset, o2 + offset),
        other => other.clone(),
    }
}

#[pyfunction]
pub fn shift_expression(exp: Vec<PyExpressionNode>, offset: usize) -> Vec<PyExpressionNode> {
    exp.iter()
        .map(|e| PyExpressionNode {
            v: do_shift(&e.v, offset),
        })
        .collect::<Vec<_>>()
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
    assignments: HashMap<String, PyExpressionNode>,
) -> PyResult<Vec<PyExpressionNode>> {
    // This function simplify the given expression using the given assignments

    // We iterate over the expression elements and we store the simplified value in the res vector
    // In the to_remove vector we store the index of the elements that can be removed
    let mut res: Vec<ExpressionNode> = vec![];
    let mut to_remove = vec![];
    for e in exp.iter() {
        let value = match &e.v {
            ExpressionNode::And(v) => {
                let mut val = true;
                let mut unresolved = false;
                let mut true_to_remove = vec![];
                for p in v.iter() {
                    if let ExpressionNode::Bool(pv) = res[*p] {
                        if pv {
                            true_to_remove.push(*p);
                        } else {
                            val = false;
                            break;
                        }
                    } else {
                        unresolved = true;
                    }
                }
                if !unresolved {
                    to_remove.extend(v.iter().clone());
                    ExpressionNode::Bool(val)
                } else {
                    to_remove.extend(true_to_remove);
                    e.v.clone()
                }
            }
            ExpressionNode::Not(p) => {
                if let ExpressionNode::Bool(v) = res[*p] {
                    to_remove.push(*p);
                    ExpressionNode::Bool(!v)
                } else {
                    e.v.clone()
                }
            }
            ExpressionNode::Equals(p1, p2) => {
                if res[*p1] == res[*p2] {
                    to_remove.push(*p1);
                    to_remove.push(*p2);
                    ExpressionNode::Bool(true)
                } else {
                    let val1 = get_rational_from_expression_node(&res[*p1]);
                    let val2 = get_rational_from_expression_node(&res[*p2]);
                    if val1.is_ok() && val2.is_ok() {
                        to_remove.push(*p1);
                        to_remove.push(*p2);
                        ExpressionNode::Bool(val1.unwrap() == val2.unwrap())
                    } else {
                        e.v.clone()
                    }
                }
            }
            ExpressionNode::LE(p1, p2) => {
                let val1 = get_rational_from_expression_node(&res[*p1]);
                let val2 = get_rational_from_expression_node(&res[*p2]);
                if val1.is_ok() && val2.is_ok() {
                    to_remove.push(*p1);
                    to_remove.push(*p2);
                    ExpressionNode::Bool(val1.unwrap() <= val2.unwrap())
                } else {
                    e.v.clone()
                }
            }
            ExpressionNode::LT(p1, p2) => {
                let val1 = get_rational_from_expression_node(&res[*p1]);
                let val2 = get_rational_from_expression_node(&res[*p2]);
                if val1.is_ok() && val2.is_ok() {
                    to_remove.push(*p1);
                    to_remove.push(*p2);
                    ExpressionNode::Bool(val1.unwrap() < val2.unwrap())
                } else {
                    e.v.clone()
                }
            }
            ExpressionNode::Plus(v) => {
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
                    to_remove.extend(v.iter().clone());
                    if r.is_integer() {
                        ExpressionNode::Int(r.to_integer())
                    } else {
                        ExpressionNode::Rational(r)
                    }
                } else {
                    e.v.clone()
                }
            }
            ExpressionNode::Minus(p1, p2) => {
                let val1 = get_rational_from_expression_node(&res[*p1]);
                let val2 = get_rational_from_expression_node(&res[*p2]);
                if val1.is_ok() && val2.is_ok() {
                    to_remove.push(*p1);
                    to_remove.push(*p2);
                    let r = val1.unwrap() - val2.unwrap();
                    if r.is_integer() {
                        ExpressionNode::Int(r.to_integer())
                    } else {
                        ExpressionNode::Rational(r)
                    }
                } else {
                    e.v.clone()
                }
            }
            ExpressionNode::Times(v) => {
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
                    to_remove.extend(v.iter().clone());
                    if r.is_integer() {
                        ExpressionNode::Int(r.to_integer())
                    } else {
                        ExpressionNode::Rational(r)
                    }
                } else {
                    e.v.clone()
                }
            }
            ExpressionNode::Div(p1, p2) => {
                let val1 = get_rational_from_expression_node(&res[*p1]);
                let val2 = get_rational_from_expression_node(&res[*p2]);
                if val1.is_ok() && val2.is_ok() {
                    to_remove.push(*p1);
                    to_remove.push(*p2);
                    let r = val1.unwrap() / val2.unwrap();
                    if r.is_integer() {
                        ExpressionNode::Int(r.to_integer())
                    } else {
                        ExpressionNode::Rational(r)
                    }
                } else {
                    e.v.clone()
                }
            }
            ExpressionNode::Fluent(s) => {
                if let Some(v) = assignments.get(s) {
                    v.v.clone()
                } else {
                    e.v.clone()
                }
            }
            other => (*other).clone(),
        };
        res.push(value);
    }

    // We build the simplified expression iterating over the res elements, removing
    // the ones that are not needed and updating the operands indexes
    let mut final_res: Vec<PyExpressionNode> = Vec::new();
    for (i, e) in res.into_iter().enumerate() {
        if !to_remove.contains(&i) {
            let ne: ExpressionNode = match e {
                ExpressionNode::And(v) => {
                    let mut operands = Vec::new();
                    for o in v {
                        if !to_remove.contains(&o) {
                            let offset = to_remove.iter().filter(|&&x| x < o).count();
                            operands.push(o - offset);
                        }
                    }
                    ExpressionNode::And(operands)
                }
                ExpressionNode::Not(p) => {
                    if !to_remove.contains(&p) {
                        let offset = to_remove.iter().filter(|&&x| x < p).count();
                        ExpressionNode::Not(p - offset)
                    } else {
                        ExpressionNode::Not(p)
                    }
                }
                ExpressionNode::Equals(p1, p2) => {
                    let mut offset1 = 0;
                    if !to_remove.contains(&p1) {
                        offset1 = to_remove.iter().filter(|&&x| x < p1).count();
                    }
                    let mut offset2 = 0;
                    if !to_remove.contains(&p2) {
                        offset2 = to_remove.iter().filter(|&&x| x < p2).count();
                    }
                    ExpressionNode::Equals(p1 - offset1, p2 - offset2)
                }
                ExpressionNode::LE(p1, p2) => {
                    let mut offset1 = 0;
                    if !to_remove.contains(&p1) {
                        offset1 = to_remove.iter().filter(|&&x| x < p1).count();
                    }
                    let mut offset2 = 0;
                    if !to_remove.contains(&p2) {
                        offset2 = to_remove.iter().filter(|&&x| x < p2).count();
                    }
                    ExpressionNode::LE(p1 - offset1, p2 - offset2)
                }
                ExpressionNode::LT(p1, p2) => {
                    let mut offset1 = 0;
                    if !to_remove.contains(&p1) {
                        offset1 = to_remove.iter().filter(|&&x| x < p1).count();
                    }
                    let mut offset2 = 0;
                    if !to_remove.contains(&p2) {
                        offset2 = to_remove.iter().filter(|&&x| x < p2).count();
                    }
                    ExpressionNode::LT(p1 - offset1, p2 - offset2)
                }
                ExpressionNode::Plus(v) => {
                    let mut operands = Vec::new();
                    for o in v {
                        if !to_remove.contains(&o) {
                            let offset = to_remove.iter().filter(|&&x| x < o).count();
                            operands.push(o - offset);
                        }
                    }
                    ExpressionNode::Plus(operands)
                }
                ExpressionNode::Minus(p1, p2) => {
                    let mut offset1 = 0;
                    if !to_remove.contains(&p1) {
                        offset1 = to_remove.iter().filter(|&&x| x < p1).count();
                    }
                    let mut offset2 = 0;
                    if !to_remove.contains(&p2) {
                        offset2 = to_remove.iter().filter(|&&x| x < p2).count();
                    }
                    ExpressionNode::Minus(p1 - offset1, p2 - offset2)
                }
                ExpressionNode::Times(v) => {
                    let mut operands = Vec::new();
                    for o in v {
                        if !to_remove.contains(&o) {
                            let offset = to_remove.iter().filter(|&&x| x < o).count();
                            operands.push(o - offset);
                        }
                    }
                    ExpressionNode::Times(operands)
                }
                ExpressionNode::Div(p1, p2) => {
                    let mut offset1 = 0;
                    if !to_remove.contains(&p1) {
                        offset1 = to_remove.iter().filter(|&&x| x < p1).count();
                    }
                    let mut offset2 = 0;
                    if !to_remove.contains(&p2) {
                        offset2 = to_remove.iter().filter(|&&x| x < p2).count();
                    }
                    ExpressionNode::Div(p1 - offset1, p2 - offset2)
                }
                _ => e,
            };
            final_res.push(PyExpressionNode { v: ne })
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

pub fn internal_evaluate(exp: &Vec<ExpressionNode>, state: &State) -> PyResult<ExpressionNode> {
    let mut res: Vec<ExpressionNode> = vec![];
    for e in exp {
        let value = match &e {
            ExpressionNode::And(v) => {
                let val = v.iter().all(|&p| res[p] == ExpressionNode::Bool(true));
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
                    ExpressionNode::Int(r.to_integer())
                } else {
                    ExpressionNode::Rational(r)
                }
            }
            ExpressionNode::Minus(p1, p2) => {
                let val1 = get_rational_from_expression_node(&res[*p1])?;
                let val2 = get_rational_from_expression_node(&res[*p2])?;
                let r = val1 - val2;
                if r.is_integer() {
                    ExpressionNode::Int(r.to_integer())
                } else {
                    ExpressionNode::Rational(r)
                }
            }
            ExpressionNode::Times(v) => {
                let mut r = get_rational_from_expression_node(&res[v[0]])?;
                for p in v.iter().skip(1) {
                    r *= get_rational_from_expression_node(&res[*p])?;
                }
                if r.is_integer() {
                    ExpressionNode::Int(r.to_integer())
                } else {
                    ExpressionNode::Rational(r)
                }
            }
            ExpressionNode::Div(p1, p2) => {
                let val1 = get_rational_from_expression_node(&res[*p1])?;
                let val2 = get_rational_from_expression_node(&res[*p2])?;
                let r = val1 / val2;
                if r.is_integer() {
                    ExpressionNode::Int(r.to_integer())
                } else {
                    ExpressionNode::Rational(r)
                }
            }
            ExpressionNode::Fluent(s) => state.get_value(&s).clone(),
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
