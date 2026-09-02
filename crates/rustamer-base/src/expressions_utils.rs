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
use num::{BigInt, Zero};
use num_rational::BigRational;
use pyo3::{
    exceptions::PyException, exceptions::PyValueError, exceptions::PyZeroDivisionError, prelude::*,
};
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
        ExpressionNode::Or(v) => ExpressionNode::Or(
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
        ExpressionNode::InterpretedFunction {
            func_id,
            return_type,
            operands,
        } => ExpressionNode::InterpretedFunction {
            func_id: *func_id,
            return_type: *return_type,
            operands: operands
                .iter()
                .map(|&o| checked_add_sub(o, offset, is_negative))
                .collect::<Result<_, _>>()?,
        },
        other => other.clone(),
    })
}

#[pyfunction(name = "shift_expression")]
pub fn py_shift_expression(
    exp: Vec<PyExpressionNode>,
    offset: usize,
) -> PyResult<Vec<PyExpressionNode>> {
    let exp: Vec<ExpressionNode> = exp.into_iter().map(|e| e.v).collect();
    let shifted = shift_expression(&exp, offset, false)
        .map_err(|e| PyException::new_err(format!("{:?}", e)))?;
    Ok(shifted
        .into_iter()
        .map(|v| PyExpressionNode { v })
        .collect())
}

pub fn shift_expression(
    exp: &[ExpressionNode],
    offset: usize,
    is_negative: bool,
) -> Result<Vec<ExpressionNode>, ArithmeticError> {
    exp.iter()
        .map(|e| do_shift(e, offset, is_negative))
        .collect::<Result<_, _>>()
}

pub fn split_expression(exp: &[ExpressionNode]) -> PyResult<Vec<Vec<ExpressionNode>>> {
    if let Some(ExpressionNode::And(operands)) = exp.last() {
        let mut res = Vec::with_capacity(operands.len());
        let mut last = 0;
        for op in operands.iter() {
            let mut new_exp = Vec::with_capacity(op + 1 - last);
            for e in &exp[last..=*op] {
                match e {
                    ExpressionNode::And(v) => {
                        let operands = v.iter().map(|&j| j - last).collect();
                        new_exp.push(make_operator("and".to_string(), operands)?);
                    }
                    ExpressionNode::Or(v) => {
                        let operands = v.iter().map(|&j| j - last).collect();
                        new_exp.push(make_operator("or".to_string(), operands)?);
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
                        new_exp.push(make_operator("==".to_string(), vec![i1 - last, i2 - last])?);
                    }
                    ExpressionNode::LE(i1, i2) => {
                        new_exp.push(make_operator("<=".to_string(), vec![i1 - last, i2 - last])?);
                    }
                    ExpressionNode::LT(i1, i2) => {
                        new_exp.push(make_operator("<".to_string(), vec![i1 - last, i2 - last])?);
                    }
                    ExpressionNode::Minus(i1, i2) => {
                        new_exp.push(make_operator("-".to_string(), vec![i1 - last, i2 - last])?);
                    }
                    ExpressionNode::Div(i1, i2) => {
                        new_exp.push(make_operator("/".to_string(), vec![i1 - last, i2 - last])?);
                    }
                    ExpressionNode::Not(i) => {
                        new_exp.push(make_operator("not".to_string(), vec![i - last])?);
                    }
                    ExpressionNode::InterpretedFunction {
                        func_id,
                        return_type,
                        operands,
                    } => {
                        new_exp.push(ExpressionNode::InterpretedFunction {
                            func_id: *func_id,
                            return_type: *return_type,
                            operands: operands.iter().map(|&j| j - last).collect(),
                        });
                    }
                    ExpressionNode::Bool(_)
                    | ExpressionNode::Int(_)
                    | ExpressionNode::Rational(_)
                    | ExpressionNode::Fluent(_)
                    | ExpressionNode::Object(_) => {
                        new_exp.push(e.clone());
                    }
                }
            }
            res.push(new_exp);
            last = op + 1;
        }
        Ok(res)
    } else {
        Ok(vec![exp.to_owned()])
    }
}

/// A borrowed view of a numeric `ExpressionNode`'s value. Where
/// `get_rational_from_expression_node` always hands back an owned
/// `BigRational` (cloning either the `BigInt` or the `BigRational` behind
/// the node, since ownership is what its other callers -- which store the
/// value or feed it to `rational_to_f64` -- actually need), a comparison or
/// a zero-check needs no ownership at all. Kept local to this module: it's
/// only useful to callers happy to `match` on which variant they got, which
/// `get_rational_from_expression_node`'s callers outside `internal_evaluate`
/// are not.
#[derive(Clone, Copy)]
enum NumRef<'a> {
    Int(&'a BigInt),
    Rational(&'a BigRational),
}

fn as_num_ref(exp: &ExpressionNode) -> PyResult<NumRef<'_>> {
    match exp {
        ExpressionNode::Int(v) => Ok(NumRef::Int(v)),
        ExpressionNode::Rational(v) => Ok(NumRef::Rational(v)),
        _ => Err(PyValueError::new_err("Expected a number!")),
    }
}

fn num_is_zero(n: NumRef) -> bool {
    match n {
        NumRef::Int(v) => v.is_zero(),
        NumRef::Rational(v) => v.is_zero(),
    }
}

/// Three-way numeric ordering with no allocation beyond what a mixed
/// `Int`/`Rational` comparison's cross-multiplication itself requires.
/// `num_rational::BigRational`'s own `PartialOrd` needs no allocation
/// either, but `get_rational_from_expression_node` would still have paid
/// for one converting an `Int` operand into an owned, redundantly-reduced
/// `BigRational` just to throw it away once compared -- comparison is
/// read-only, unlike the arithmetic `fold_numeric` handles, so it never
/// needs to *construct* a `Rational` value, only to reason about one.
/// Cross-multiplication is valid without a sign correction because every
/// `BigRational` in this crate is built through `Ratio::new`/`from_integer`,
/// whose `reduce()` forces a positive denominator.
fn num_cmp(a: NumRef, b: NumRef) -> std::cmp::Ordering {
    match (a, b) {
        (NumRef::Int(a), NumRef::Int(b)) => a.cmp(b),
        (NumRef::Rational(a), NumRef::Rational(b)) => {
            a.partial_cmp(b).expect("rational comparison is total")
        }
        (NumRef::Int(a), NumRef::Rational(b)) => (a * b.denom()).cmp(b.numer()),
        (NumRef::Rational(a), NumRef::Int(b)) => a.numer().cmp(&(b * a.denom())),
    }
}

/// Numeric accumulator shared by `internal_evaluate`'s `fold_numeric` (every
/// operand guaranteed numeric) and `simplify`'s partial `Plus`/`Times` fold
/// (an operand may still be symbolic, so accumulation only starts once one
/// is actually seen -- `seed`/`combine` are split apart, rather than folded
/// into one function, for exactly that reason). Starts on `BigInt` -- the
/// common case, and cheaper: no `BigRational` construction, no
/// gcd-normalization on every op -- and promotes to `BigRational` in place
/// the instant a `Rational` operand is combined in. That promotion happens
/// mid-fold: unlike a "try the all-`Int` path, redo everything in
/// `BigRational` on failure" split, a `Rational` operand encountered after N
/// `Int` ones costs one promotion, not a second pass re-folding those N
/// operands from scratch.
enum Acc {
    Int(BigInt),
    Rational(BigRational),
}

impl Acc {
    fn seed(n: NumRef) -> Self {
        match n {
            NumRef::Int(v) => Acc::Int(v.clone()),
            NumRef::Rational(v) => Acc::Rational(v.clone()),
        }
    }

    fn combine(
        &mut self,
        n: NumRef,
        int_op: &mut impl FnMut(&mut BigInt, &BigInt),
        rational_op: &mut impl FnMut(&mut BigRational, &BigRational),
    ) {
        match (&mut *self, n) {
            (Acc::Int(a), NumRef::Int(b)) => int_op(a, b),
            (Acc::Int(a), NumRef::Rational(b)) => {
                let mut r = BigRational::from_integer(a.clone());
                rational_op(&mut r, b);
                *self = Acc::Rational(r);
            }
            (Acc::Rational(a), NumRef::Int(b)) => {
                rational_op(a, &BigRational::from_integer(b.clone()));
            }
            (Acc::Rational(a), NumRef::Rational(b)) => rational_op(a, b),
        }
    }

    fn into_node(self) -> ExpressionNode {
        match self {
            Acc::Int(v) => ExpressionNode::Int(Box::new(v)),
            Acc::Rational(r) if r.is_integer() => ExpressionNode::Int(Box::new(r.to_integer())),
            Acc::Rational(r) => ExpressionNode::Rational(Box::new(r)),
        }
    }
}

/// Folds `indices` (assumed non-empty -- well-formed `Plus`/`Minus`/`Times`
/// never have zero operands, and every operand is numeric) over
/// `int_op`/`rational_op` via `Acc`, in a single pass.
fn fold_numeric(
    res: &[ExpressionNode],
    indices: &[usize],
    mut int_op: impl FnMut(&mut BigInt, &BigInt),
    mut rational_op: impl FnMut(&mut BigRational, &BigRational),
) -> PyResult<ExpressionNode> {
    let mut acc = Acc::seed(as_num_ref(&res[indices[0]])?);
    for &p in indices.iter().skip(1) {
        acc.combine(as_num_ref(&res[p])?, &mut int_op, &mut rational_op);
    }
    Ok(acc.into_node())
}

/// Divides two already-borrowed numeric operands, producing the normalized
/// `Int`/`Rational` result node. Checks `b` for zero without constructing
/// anything (`NumRef` is a borrow, not an owned value), only cloning once
/// there's a value to actually build -- unlike going through
/// `get_rational_from_expression_node` for both operands unconditionally
/// before ever checking. Shared by `internal_evaluate`'s `Div` (every
/// operand guaranteed numeric) and `simplify`'s `Div` (a non-numeric
/// operand means "not yet foldable", handled by the caller before this is
/// reached).
fn num_div(a: NumRef, b: NumRef) -> PyResult<ExpressionNode> {
    if num_is_zero(b) {
        return Err(PyZeroDivisionError::new_err("division by zero"));
    }
    let r = match (a, b) {
        (NumRef::Int(a), NumRef::Int(b)) => BigRational::new(a.clone(), b.clone()),
        (NumRef::Int(a), NumRef::Rational(b)) => BigRational::from_integer(a.clone()) / b.clone(),
        (NumRef::Rational(a), NumRef::Int(b)) => a.clone() / BigRational::from_integer(b.clone()),
        (NumRef::Rational(a), NumRef::Rational(b)) => a.clone() / b.clone(),
    };
    Ok(if r.is_integer() {
        ExpressionNode::Int(Box::new(r.to_integer()))
    } else {
        ExpressionNode::Rational(Box::new(r))
    })
}

#[pyfunction]
#[pyo3(signature = (exp, assignments, evaluate_interpreted_functions=false))]
pub fn simplify(
    exp: Vec<PyExpressionNode>,
    assignments: FxHashMap<usize, PyExpressionNode>,
    evaluate_interpreted_functions: bool,
) -> PyResult<Vec<PyExpressionNode>> {
    // This function simplifies the given expression using the given assignments.
    //
    // If `evaluate_interpreted_functions` is true, an interpreted function
    // whose operands have all been folded to constants is actually called
    // and replaced by its result; otherwise (the default) it is always
    // re-emitted unchanged.

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
                        ExpressionNode::Or(new_operands)
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
                    match (as_num_ref(&res[p1]), as_num_ref(&res[p2])) {
                        (Ok(v1), Ok(v2)) => ExpressionNode::Bool(num_cmp(v1, v2).is_eq()),
                        _ => e.v,
                    }
                }
            }
            ExpressionNode::LE(p1, p2) => match (as_num_ref(&res[p1]), as_num_ref(&res[p2])) {
                (Ok(v1), Ok(v2)) => ExpressionNode::Bool(num_cmp(v1, v2).is_le()),
                _ => e.v,
            },
            ExpressionNode::LT(p1, p2) => match (as_num_ref(&res[p1]), as_num_ref(&res[p2])) {
                (Ok(v1), Ok(v2)) => ExpressionNode::Bool(num_cmp(v1, v2).is_lt()),
                _ => e.v,
            },
            ExpressionNode::Plus(ref v) => {
                // A partial fold, unlike `internal_evaluate`'s `Plus`: an
                // operand can still be symbolic, so accumulation only
                // starts once a numeric one is actually seen (`acc` stays
                // `None` until then), and every non-numeric operand survives
                // into `operands` unchanged, in its original position.
                let mut acc: Option<Acc> = None;
                let mut first_constant_operand = None;
                let mut operands = Vec::new();
                for &p in v.iter() {
                    match as_num_ref(&res[p]) {
                        Ok(n) => {
                            match &mut acc {
                                Some(a) => a.combine(n, &mut |a, b| *a += b, &mut |a, b| *a += b),
                                None => acc = Some(Acc::seed(n)),
                            }
                            if first_constant_operand.is_none() {
                                first_constant_operand = Some(p);
                                operands.push(p);
                            }
                        }
                        Err(_) => operands.push(p),
                    }
                }

                if let Some(acc) = acc {
                    let new_node = acc.into_node();
                    if operands.len() == 1 {
                        new_node
                    } else {
                        res[first_constant_operand.unwrap()] = new_node;
                        ExpressionNode::Plus(operands)
                    }
                } else {
                    e.v
                }
            }
            ExpressionNode::Minus(p1, p2) => match (as_num_ref(&res[p1]), as_num_ref(&res[p2])) {
                (Ok(a), Ok(b)) => {
                    let mut acc = Acc::seed(a);
                    acc.combine(b, &mut |a, b| *a -= b, &mut |a, b| *a -= b);
                    acc.into_node()
                }
                _ => e.v,
            },
            ExpressionNode::Times(ref v) => {
                let mut acc: Option<Acc> = None;
                let mut first_constant_operand = None;
                let mut operands = Vec::new();
                for &p in v.iter() {
                    match as_num_ref(&res[p]) {
                        Ok(n) => {
                            match &mut acc {
                                Some(a) => a.combine(n, &mut |a, b| *a *= b, &mut |a, b| *a *= b),
                                None => acc = Some(Acc::seed(n)),
                            }
                            if first_constant_operand.is_none() {
                                first_constant_operand = Some(p);
                                operands.push(p);
                            }
                        }
                        Err(_) => operands.push(p),
                    }
                }

                if let Some(acc) = acc {
                    let new_node = acc.into_node();
                    if operands.len() == 1 {
                        new_node
                    } else {
                        res[first_constant_operand.unwrap()] = new_node;
                        ExpressionNode::Times(operands)
                    }
                } else {
                    e.v
                }
            }
            ExpressionNode::Div(p1, p2) => match (as_num_ref(&res[p1]), as_num_ref(&res[p2])) {
                (Ok(a), Ok(b)) => num_div(a, b)?,
                _ => e.v,
            },
            ExpressionNode::Fluent(s) => {
                if let Some(v) = assignments.get(&s) {
                    v.v.clone()
                } else {
                    e.v
                }
            }
            ExpressionNode::InterpretedFunction {
                func_id,
                return_type,
                ref operands,
            } => {
                if evaluate_interpreted_functions
                    && operands.iter().all(|&p| {
                        matches!(
                            &res[p],
                            ExpressionNode::Bool(_)
                                | ExpressionNode::Int(_)
                                | ExpressionNode::Rational(_)
                                | ExpressionNode::Object(_)
                        )
                    })
                {
                    // all operands are constants
                    let operand_values: Vec<&ExpressionNode> =
                        operands.iter().map(|&p| &res[p]).collect();
                    call_interpreted_function(func_id, return_type, &operand_values)?
                } else {
                    e.v
                }
            }
            ExpressionNode::Rational(v) => {
                if v.is_integer() {
                    ExpressionNode::Int(Box::new(v.to_integer()))
                } else {
                    ExpressionNode::Rational(v)
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
            | ExpressionNode::Times(operands)
            | ExpressionNode::InterpretedFunction { operands, .. } => {
                if processed {
                    let new_operands = operands_stack
                        .drain((operands_stack.len() - operands.len())..)
                        .collect();
                    operands_stack.push(final_res.len());
                    let exp_node = match &res[idx] {
                        ExpressionNode::And(_) => ExpressionNode::And(new_operands),
                        ExpressionNode::Or(_) => ExpressionNode::Or(new_operands),
                        ExpressionNode::Plus(_) => ExpressionNode::Plus(new_operands),
                        ExpressionNode::Times(_) => ExpressionNode::Times(new_operands),
                        ExpressionNode::InterpretedFunction {
                            func_id,
                            return_type,
                            ..
                        } => ExpressionNode::InterpretedFunction {
                            func_id: *func_id,
                            return_type: *return_type,
                            operands: new_operands,
                        },
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
    let mut res: Vec<ExpressionNode> = Vec::with_capacity(exp.len() - 1);
    for e in exp {
        let value = match &e {
            ExpressionNode::And(v) => {
                let val = v
                    .iter()
                    .all(|&p| matches!(res[p], ExpressionNode::Bool(true)));
                ExpressionNode::Bool(val)
            }
            ExpressionNode::Or(v) => {
                let val = v
                    .iter()
                    .any(|&p| matches!(res[p], ExpressionNode::Bool(true)));
                ExpressionNode::Bool(val)
            }
            ExpressionNode::Not(p) => {
                ExpressionNode::Bool(matches!(res[*p], ExpressionNode::Bool(false)))
            }
            ExpressionNode::Equals(p1, p2) => {
                // Structural equality first (cheap, and correct for the
                // overwhelmingly common case), falling back to a numeric
                // comparison when it fails and both sides are numbers --
                // an `Int` and a denominator-1 `Rational` holding the same
                // value are legitimately reachable on well-formed input and
                // must still compare equal.
                let val = res[*p1] == res[*p2]
                    || match (as_num_ref(&res[*p1]), as_num_ref(&res[*p2])) {
                        (Ok(v1), Ok(v2)) => num_cmp(v1, v2).is_eq(),
                        _ => false,
                    };
                ExpressionNode::Bool(val)
            }
            ExpressionNode::LE(p1, p2) => {
                let val = num_cmp(as_num_ref(&res[*p1])?, as_num_ref(&res[*p2])?).is_le();
                ExpressionNode::Bool(val)
            }
            ExpressionNode::LT(p1, p2) => {
                let val = num_cmp(as_num_ref(&res[*p1])?, as_num_ref(&res[*p2])?).is_lt();
                ExpressionNode::Bool(val)
            }
            ExpressionNode::Plus(v) => fold_numeric(&res, v, |a, b| *a += b, |a, b| *a += b)?,
            ExpressionNode::Minus(p1, p2) => {
                fold_numeric(&res, &[*p1, *p2], |a, b| *a -= b, |a, b| *a -= b)?
            }
            ExpressionNode::Times(v) => fold_numeric(&res, v, |a, b| *a *= b, |a, b| *a *= b)?,
            ExpressionNode::Div(p1, p2) => num_div(as_num_ref(&res[*p1])?, as_num_ref(&res[*p2])?)?,
            ExpressionNode::Fluent(s) => fluent_values.get_value(*s).clone(),
            ExpressionNode::InterpretedFunction {
                func_id,
                return_type,
                operands,
            } => {
                let args: Vec<&ExpressionNode> = operands.iter().map(|&p| &res[p]).collect();
                call_interpreted_function(*func_id, *return_type, &args)?
            }
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
