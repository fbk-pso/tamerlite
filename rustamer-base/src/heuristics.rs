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

use itertools::Itertools;
use std::hash::{Hash, Hasher};
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use std::vec::Vec;

use rustc_hash::{FxBuildHasher, FxHashMap, FxHashSet};

use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use super::expressions::*;
use super::expressions_utils::*;
use super::multiqueue::StateContainer;
use super::search_space::SearchSpaceTrait;
use super::search_state::State;
use super::structures::*;
use super::utils::*;

pub trait HeuristicTrait {
    fn eval<S: SearchSpaceTrait>(&self, state: &State, ss: &S) -> PyResult<Option<f64>>;

    /// Evaluates the heuristic for a given state, returning an iterator over the results.
    /// This method is used in non-multiqueue search algorithms
    fn eval_gen<'a, I, S: SearchSpaceTrait>(
        &'a self,
        states_iter: I,
        ss: &'a S,
    ) -> PyResult<Box<dyn Iterator<Item = PyResult<(Rc<State>, Option<f64>)>> + 'a>>
    where
        I: Iterator<Item = PyResult<Rc<State>>> + 'a,
    {
        return Ok(Box::new(states_iter.map(|state| {
            let state = state?;
            let h_value = self.eval(&state, ss)?;
            Ok((state, h_value))
        })));
    }

    /// Evaluates the heuristic for a given state, returning an iterator over the results.
    /// This method is used in multiqueue search algorithms
    fn eval_gen_container<'a, S: SearchSpaceTrait>(
        &'a self,
        states: &'a Vec<StateContainer>,
        ss: &'a S,
    ) -> PyResult<Box<dyn Iterator<Item = PyResult<(usize, Option<f64>)>> + 'a>> {
        return Ok(Box::new(states.iter().enumerate().map(|(i, sc)| {
            let h_value = self.eval(&sc.state, ss)?;
            Ok((i, h_value))
        })));
    }
}

#[derive(Clone, Debug)]
pub enum HeuristicKind {
    HFF,
    HADD,
    HMAX,
}

#[derive(Debug)]
pub struct CustomHeuristic {
    callable: Py<PyAny>,
}

impl CustomHeuristic {
    pub fn new(callable: Py<PyAny>) -> PyResult<Self> {
        Ok(CustomHeuristic { callable })
    }

    pub fn eval(&self, state: &State) -> PyResult<Option<f64>> {
        Python::attach(|py| {
            let args = PyTuple::new(py, &[state.full_clone().into_pyobject(py)?])?;
            let r = self.callable.call(py, args, None)?;
            if r.is_none(py) {
                Ok(None)
            } else {
                Ok(Some(r.extract(py)?))
            }
        })
    }

    pub fn name(&self) -> String {
        String::from("custom")
    }
}

impl Clone for CustomHeuristic {
    fn clone(&self) -> Self {
        Python::attach(|py| CustomHeuristic {
            callable: self.callable.clone_ref(py),
        })
    }
}

#[derive(Debug, Clone)]
struct Operator {
    id: OperatorID,
    action: Action,
    conditions: HeuristicExpression,
    effects: Vec<Expression>,
    constant_increase_effects: FxHashMap<usize, f64>,
    constant_assign_effects: FxHashMap<usize, f64>,
    complex_numeric_effects: FxHashMap<usize, Expression>,
    cost: f64,
}

impl PartialEq for Operator {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for Operator {}

impl Hash for Operator {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct OperatorID {
    id: usize,
}

impl OperatorID {
    fn new(id: usize) -> OperatorID {
        OperatorID { id }
    }
}

#[derive(Debug, Clone, PartialEq, Hash)]
enum HeuristicExpressionNode {
    And(usize),
    Or(usize),
    Leaf(Expression),
}

#[derive(Debug, Clone, PartialEq, Hash)]
struct HeuristicExpression {
    expression: Vec<HeuristicExpressionNode>,
    contains_or_node: bool,
}

#[derive(Debug, Clone, PartialEq)]
struct OperatorHmax {
    action: Action,
    conditions: Vec<Vec<ExpressionNode>>,
    condition_expressions: Vec<Expression>,
    effects: Vec<Effect>,
    cost: f64,
}

impl Eq for OperatorHmax {}

impl Hash for OperatorHmax {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.action.hash(state);
        self.conditions.hash(state);
        self.effects.hash(state);
    }
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
struct CacheKey {
    values: Vec<ExpressionNode>,
    todo_values: Vec<usize>,
}

fn get_event_conditions(
    event: &Event,
    expression_manager: &mut ExpressionManager,
) -> PyResult<Vec<Vec<ExpressionNode>>> {
    let mut conditions_set = FxHashSet::with_hasher(FxBuildHasher::default());
    let mut conditions = Vec::new();
    for condition in split_expression(&event.conditions)?
        .into_iter()
        .chain(event.end_conditions.clone())
    {
        if conditions_set.insert(expression_manager.put(&condition)) {
            conditions.push(condition);
        }
    }
    Ok(conditions)
}

/// Build the operator condition as a `HeuristicExpression`.
///
/// This method takes the operator conditions and add the `extra_fluent`.
/// The final result is converted into a `HeuristicExpression`.
///
/// # Arguments
///
/// * `conditions` - The conditions of the operator.
/// * `extra_fluent` - The additional fluent to include in the condition.
/// * `expression_manager` - A mutable reference to the `ExpressionManager`.
///
/// # Returns
///
/// Returns `Some(HeuristicExpression)` if the resulting condition is not explicitly false,
/// otherwise returns `None`.
fn build_operator_condition(
    conditions: &Vec<Vec<ExpressionNode>>,
    extra_fluent: ExpressionNode,
    objects: &FxHashMap<String, Vec<String>>,
    fluent_types: &Vec<String>,
    disable_numeric_reasoning: bool,
    expression_manager: &mut ExpressionManager,
) -> Result<Option<HeuristicExpression>, ArithmeticError> {
    let mut condition_expr = Vec::with_capacity(conditions.len() + 2);
    let mut operands = Vec::with_capacity(conditions.len() + 1);
    for condition in conditions {
        if condition == &vec![ExpressionNode::Bool(false)] {
            // If the condition is explicitly False, the operator is not applicable
            return Ok(None);
        } else if !condition.is_empty() && condition != &vec![ExpressionNode::Bool(true)] {
            condition_expr.extend(shift_expression(condition, condition_expr.len(), false)?);
            operands.push(condition_expr.len() - 1);
        };
    }
    condition_expr.push(extra_fluent);
    operands.push(condition_expr.len() - 1);
    if operands.len() > 1 {
        condition_expr.push(ExpressionNode::And(operands));
    }

    let condition = convert_to_heuristic_expression(&condition_expr, expression_manager)?;
    let condition = simplify_condition(
        &condition,
        objects,
        fluent_types,
        disable_numeric_reasoning,
        expression_manager,
    )?;
    Ok(Some(condition))
}

/// Convert an expression into a `HeuristicExpression`.
///
/// A `HeuristicExpression` represents the input expression in a form where:
/// - Only `AND` and `OR` operations are internal nodes.
/// - All other elements are represented as `Leaf` nodes.
///
/// # Arguments
///
/// * `expr` - A reference to the input expression to convert.
/// * `expression_manager` - A mutable reference to the `ExpressionManager`.
///
/// # Returns
///
/// Returns a `HeuristicExpression` containing:
/// - `expression`: a vector of `HeuristicExpressionNode` representing the converted expression,
///   including `And`, `Or`, and `Leaf` nodes.
/// - `contains_or_node`: a boolean indicating whether the expression contains at least one `Or` node.
fn convert_to_heuristic_expression(
    expr: &Vec<ExpressionNode>,
    expression_manager: &mut ExpressionManager,
) -> Result<HeuristicExpression, ArithmeticError> {
    let mut contains_or_node = false;
    let mut result = Vec::new();
    let mut stack = vec![(expr.len() - 1, false)];

    while let Some((idx, processed)) = stack.pop() {
        match &expr[idx] {
            ExpressionNode::Bool(_)
            | ExpressionNode::Int(_)
            | ExpressionNode::Rational(_)
            | ExpressionNode::Object(_)
            | ExpressionNode::Fluent(_) => result.push(HeuristicExpressionNode::Leaf(
                expression_manager.put(&vec![expr[idx].clone()]),
            )),
            ExpressionNode::And(operands) => {
                if !processed {
                    stack.push((idx, true));
                    for &i in operands {
                        stack.push((i, false));
                    }
                } else {
                    result.push(HeuristicExpressionNode::And(operands.len()));
                }
            }
            ExpressionNode::Or(operands) => {
                if !processed {
                    stack.push((idx, true));
                    for &i in operands {
                        stack.push((i, false));
                    }
                } else {
                    contains_or_node = true;
                    result.push(HeuristicExpressionNode::Or(operands.len()));
                }
            }
            _ => result.push(HeuristicExpressionNode::Leaf(
                expression_manager.put(&extract_sub_expression(&expr, idx)?),
            )),
        }
    }

    Ok(HeuristicExpression {
        expression: result,
        contains_or_node,
    })
}

/// Simplifies leaf expressions in a condition.
///
/// Each leaf node in the condition is rewritten when possible. The following
/// simplifications are applied:
///
/// - Simple numeric leaf expressions containing logical negation (`not`) or
///   equality (`==`) are simplified, unless numeric reasoning is disabled.
/// - Fluent-object inequality expressions (`fluent != object`) are rewritten
///   into an equivalent form.
///
/// Non-leaf nodes or leaf nodes that do not match any simplification rule are
/// left unchanged.
///
/// # Arguments
///
/// * `condition` - The expression to simplify.
/// * `objects` - Mapping from type names to their objects, used for fluent-object inequalities.
/// * `fluent_types` - List of fluent type names.
/// * `disable_numeric_reasoning` - If true, numeric simplifications are skipped.
/// * `expression_manager` - A mutable reference to the `ExpressionManager`.
///
/// # Returns
///
/// Returns a new `HeuristicExpression` with simplified leaf nodes.
///
/// # Errors
///
/// Returns an `ArithmeticError` if a numeric simplification fails due to
/// an arithmetic error.
fn simplify_condition(
    condition: &HeuristicExpression,
    objects: &FxHashMap<String, Vec<String>>,
    fluent_types: &Vec<String>,
    disable_numeric_reasoning: bool,
    expression_manager: &mut ExpressionManager,
) -> Result<HeuristicExpression, ArithmeticError> {
    let mut new_condition = Vec::with_capacity(condition.expression.len());
    let mut contains_or_node = condition.contains_or_node;
    for node in &condition.expression {
        if let HeuristicExpressionNode::Leaf(expr) = node {
            let simplified_expr = if !disable_numeric_reasoning
                && is_numeric_leaf_expression(expression_manager.force_get(expr))
            {
                simplify_numeric_leaf_node(expr, expression_manager)?
            } else {
                simplify_fluent_not_equals_object_expression(
                    expr,
                    objects,
                    fluent_types,
                    expression_manager,
                )
            };
            if let Some(mut simplified_expr) = simplified_expr {
                contains_or_node |= simplified_expr.contains_or_node;
                new_condition.append(&mut simplified_expr.expression);
                continue;
            }
        }
        new_condition.push(node.clone())
    }

    Ok(HeuristicExpression {
        expression: new_condition,
        contains_or_node,
    })
}

/// Simplifies a simple numeric expression.
///
/// This function rewrites numeric expressions containing logical negation (`not`)
/// or equality (`==`) into simpler equivalent expressions suitable for heuristic
/// evaluation. Specifically, it transforms:
///
/// - `a == b` into `a <= b and b <= a`.
/// - `not(a == b)` into `a < b or b < a`
/// - `not(a < b)` into `b <= a`
/// - `not(a <= b)` into `b < a`
///
/// # Arguments
///
/// * `expr` - The numeric expression to simplify.
/// * `expression_manager` - A mutable reference to the `ExpressionManager`.
///
/// # Returns
///
/// Returns `Ok(Some(HeuristicExpression))` if simplification is possible,
/// `Ok(None)` if the expression cannot be simplified.
///
/// # Errors
///
/// Returns an `ArithmeticError` if a numeric simplification fails due to
/// an arithmetic error.
fn simplify_numeric_leaf_node(
    expr: &Expression,
    expression_manager: &mut ExpressionManager,
) -> Result<Option<HeuristicExpression>, ArithmeticError> {
    let expr = expression_manager.force_get(expr).clone();
    if let Some(node) = expr.last() {
        let new_expr = match node {
            ExpressionNode::Equals(op1, op2) => {
                let mut expr1 = expr.clone();
                expr1
                    .last_mut()
                    .map(|last| *last = ExpressionNode::LE(*op1, *op2));
                let expr1 = expression_manager.put(&expr1);

                let expr2 =
                    invert_operands(&expr, *op1, *op2, ExpressionNode::LE, expression_manager)?;

                Some(HeuristicExpression {
                    expression: vec![
                        HeuristicExpressionNode::Leaf(expr1),
                        HeuristicExpressionNode::Leaf(expr2),
                        HeuristicExpressionNode::And(2),
                    ],
                    contains_or_node: false,
                })
            }
            ExpressionNode::Not(op) => {
                let negated = &expr[*op];
                match negated {
                    ExpressionNode::Equals(op1, op2) => {
                        let mut expr1 = expr[0..expr.len() - 2].to_vec();
                        expr1.push(ExpressionNode::LT(*op1, *op2));
                        let expr1 = expression_manager.put(&expr1);

                        let expr2 = invert_operands(
                            &expr,
                            *op1,
                            *op2,
                            ExpressionNode::LT,
                            expression_manager,
                        )?;

                        Some(HeuristicExpression {
                            expression: vec![
                                HeuristicExpressionNode::Leaf(expr1),
                                HeuristicExpressionNode::Leaf(expr2),
                                HeuristicExpressionNode::Or(2),
                            ],
                            contains_or_node: true,
                        })
                    }
                    ExpressionNode::LT(op1, op2) => {
                        let expr1 = invert_operands(
                            &expr,
                            *op1,
                            *op2,
                            ExpressionNode::LE,
                            expression_manager,
                        )?;

                        Some(HeuristicExpression {
                            expression: vec![HeuristicExpressionNode::Leaf(expr1)],
                            contains_or_node: false,
                        })
                    }
                    ExpressionNode::LE(op1, op2) => {
                        let expr1 = invert_operands(
                            &expr,
                            *op1,
                            *op2,
                            ExpressionNode::LT,
                            expression_manager,
                        )?;

                        Some(HeuristicExpression {
                            expression: vec![HeuristicExpressionNode::Leaf(expr1)],
                            contains_or_node: false,
                        })
                    }
                    _ => None,
                }
            }
            _ => None,
        };
        if let Some(new_expr) = new_expr {
            if let Some(HeuristicExpressionNode::Leaf(expr)) = &new_expr.expression.get(0) {
                let expr = expression_manager.force_get(expr);
                let (op1, op2) = match expr.last() {
                    Some(ExpressionNode::LT(op1, op2)) | Some(ExpressionNode::LE(op1, op2)) => {
                        (op1, op2)
                    }
                    _ => return Ok(None),
                };
                let mut polynomial_expr = expr.clone();
                polynomial_expr.pop();
                polynomial_expr.push(ExpressionNode::Minus(*op1, *op2));
                if to_linear_polynomial(&polynomial_expr).is_some() {
                    return Ok(Some(new_expr));
                }
            }
        }
    }

    Ok(None)
}

fn invert_operands<F>(
    expr: &Vec<ExpressionNode>,
    op1: usize,
    op2: usize,
    expression_node_type: F,
    expression_manager: &mut ExpressionManager,
) -> Result<Expression, ArithmeticError>
where
    F: FnOnce(usize, usize) -> ExpressionNode,
{
    let (mut op1_expr, mut op2_expr) = inverted_operands(&expr, op1, op2)?;
    let op1 = op1_expr.len() - 1;
    let op2 = op1_expr.len() + op2_expr.len() - 1;
    let mut new_expr = Vec::with_capacity(op1_expr.len() + op2_expr.len() + 1);
    new_expr.append(&mut op1_expr);
    new_expr.append(&mut op2_expr);
    new_expr.push(expression_node_type(op1, op2));
    let expr1 = expression_manager.put(&new_expr);
    Ok(expr1)
}

fn inverted_operands(
    expr: &Vec<ExpressionNode>,
    op1: usize,
    op2: usize,
) -> Result<(Vec<ExpressionNode>, Vec<ExpressionNode>), ArithmeticError> {
    let op1_expr = expr[0..op1 + 1].to_vec();
    let op2_expr = expr[op1 + 1..op2 + 1].to_vec();
    Ok((
        shift_expression(&op2_expr, op1_expr.len(), true)?,
        shift_expression(&op1_expr, op2_expr.len(), false)?,
    ))
}

/// Simplifies a leaf expression of the form `fluent != object`.
///
/// This function rewrites inequality expressions between a fluent and a specific
/// object into an equivalent disjunction of equalities:
///
/// `fluent != objX` into `fluent == obj1 or fluent == obj2 or ...`
///
/// where `obj1, obj2, ...` are all objects of the fluent's type except `objX`.
///
/// # Arguments
///
/// * `expr` - The expression to simplify.
/// * `objects` - Mapping from type names to their available objects.
/// * `fluent_types` - List of fluent type names.
/// * `expression_manager` - A mutable reference to the `ExpressionManager`.
///
/// # Returns
///
/// Returns `Some(HeuristicExpression)` representing the disjunction of equality
/// expressions if simplification is possible, or `None` if the expression is not
/// of the form `fluent != object`.
fn simplify_fluent_not_equals_object_expression(
    expr: &Expression,
    objects: &FxHashMap<String, Vec<String>>,
    fluent_types: &Vec<String>,
    expression_manager: &mut ExpressionManager,
) -> Option<HeuristicExpression> {
    let expr = expression_manager.force_get(expr).clone();
    let [ExpressionNode::Fluent(f), ExpressionNode::Object(o), ExpressionNode::Equals(_, _), ExpressionNode::Not(_)] =
        expr.as_slice()
    else {
        return None;
    };

    let t = fluent_types.get(*f)?;
    let objs = objects.get(t)?;

    let mut nodes: Vec<_> = objs
        .iter()
        .filter(|obj| *obj != o)
        .map(|obj| {
            let leaf_expr = expression_manager.put(&vec![
                ExpressionNode::Fluent(*f),
                ExpressionNode::Object(obj.clone()),
                ExpressionNode::Equals(0, 1),
            ]);
            HeuristicExpressionNode::Leaf(leaf_expr)
        })
        .collect();

    let res = if nodes.is_empty() {
        let false_expr = vec![ExpressionNode::Bool(false)];
        let false_expr = expression_manager.put(&false_expr);
        HeuristicExpression {
            expression: vec![HeuristicExpressionNode::Leaf(false_expr)],
            contains_or_node: false,
        }
    } else if nodes.len() > 1 {
        nodes.push(HeuristicExpressionNode::Or(nodes.len()));
        HeuristicExpression {
            expression: nodes,
            contains_or_node: true,
        }
    } else {
        HeuristicExpression {
            expression: nodes,
            contains_or_node: false,
        }
    };
    Some(res)
}

/// Processes a numeric effect and categorizes it into one of three types:
///
/// 1. **Constant assignment:** If the effect is a single numeric value, it is
///    stored in `constant_assign_effects`.
/// 2. **Constant increase:** If the effect represents a linear increase of a
///    fluent by a constant amount, it is stored in `constant_increase_effects`.
/// 3. **Complex numeric effect:** If the effect is non-linear or cannot be
///    simplified to a constant increase, it is stored in `complex_numeric_effects`.
///
/// # Arguments
///
/// * `effect` - The numeric effect to process.
/// * `expression_manager` - A mutable reference to the `ExpressionManager`.
/// * `constant_increase_effects` - Mutable mapping from fluents to constant
///   increase values; updated if the effect is a simple increase.
/// * `constant_assign_effects` - Mutable mapping from fluents to constant
///   assignment values; updated if the effect is a constant numeric assignment.
/// * `complex_numeric_effects` - Mutable mapping from fluents to expressions
///   for effects that are complex.
fn update_numeric_effects(
    effect: &Effect,
    expression_manager: &mut ExpressionManager,
    constant_increase_effects: &mut FxHashMap<usize, f64>,
    constant_assign_effects: &mut FxHashMap<usize, f64>,
    complex_numeric_effects: &mut FxHashMap<usize, Expression>,
) {
    if effect.value.len() == 1 {
        let v = match &effect.value[0] {
            ExpressionNode::Int(v) => Some(integer_to_f64(&v)),
            ExpressionNode::Rational(v) => Some(rational_to_f64(&v)),
            _ => None,
        };
        if let Some(v) = v {
            constant_assign_effects.insert(effect.fluent, v);
            return;
        }
    }

    let polynomial = to_linear_polynomial(&effect.value);
    if polynomial.is_some() {
        let mut polynomial = polynomial.unwrap();
        let k = polynomial.remove(&None).unwrap_or(0.0);
        if polynomial.len() == 1 && matches!(polynomial.get(&Some(effect.fluent)), Some(1.0)) {
            constant_increase_effects.insert(effect.fluent, k);
        } else {
            complex_numeric_effects.insert(effect.fluent, expression_manager.put(&effect.value));
        }
    } else {
        complex_numeric_effects.insert(effect.fluent, expression_manager.put(&effect.value));
    }
}

/// Determine if a leaf expression represents a numeric expression.
/// A leaf expression is assumed to contain no `AND` or `OR` nodes.
///
/// # Arguments
///
/// * `expr` - A reference to the leaf expression (`Vec<ExpressionNode>`) to check.
///
/// # Returns
///
/// Returns `true` if the leaf expression is numeric, `false` otherwise.
fn is_numeric_leaf_expression(expr: &Vec<ExpressionNode>) -> bool {
    let idx = match expr.last() {
        Some(ExpressionNode::Not(op)) => *op,
        _ => expr.len() - 1,
    };
    match expr[idx] {
        ExpressionNode::Equals(op1, op2) => {
            !matches!(expr[op1], ExpressionNode::Object(_))
                && !matches!(expr[op2], ExpressionNode::Object(_))
        }
        ExpressionNode::LE(_, _)
        | ExpressionNode::LT(_, _)
        | ExpressionNode::Plus(_)
        | ExpressionNode::Minus(_, _)
        | ExpressionNode::Times(_)
        | ExpressionNode::Div(_, _) => true,
        _ => false,
    }
}

/// Processes a numeric condition and classifies it as simple or complex:
///
/// - If numeric reasoning is disabled, the condition is always treated as complex.
/// - If the condition can be represented as a simple linear numeric expression,
///   it is stored in `simple_numeric_conds` along with its fluents and weights.
/// - Conditions that cannot be simplified are stored in `complex_numeric_conds`.
///
/// # Arguments
///
/// * `numeric_condition` - The numeric condition to process.
/// * `expression_manager` - A mutable reference to the `ExpressionManager`.
/// * `simple_numeric_conds` - Mutable mapping from expressions to tuples of fluents
///   and weights for simple numeric conditions.
/// * `lt_simple_numeric_conds` - Mutable set of expressions that are simple numeric
///   conditions that have the `<` operator.
/// * `complex_numeric_conds` - Mutable set of expressions that are complex and
///   cannot be simplified.
/// * `disable_numeric_reasoning` - If true, numeric simplifications are skipped.
fn update_numeric_conditions(
    numeric_condition: &Expression,
    expression_manager: &ExpressionManager,
    simple_numeric_conds: &mut FxHashMap<Expression, (Vec<usize>, Vec<f64>)>,
    lt_simple_numeric_conds: &mut FxHashSet<Expression>,
    complex_numeric_conds: &mut FxHashSet<Expression>,
    disable_numeric_reasoning: bool,
) {
    if disable_numeric_reasoning {
        complex_numeric_conds.insert(*numeric_condition);
        return;
    }

    let fluents_weights =
        extract_fluents_weights_simple_numeric_condition(numeric_condition, expression_manager);
    if let Some((fluents, weights, is_lt)) = fluents_weights {
        simple_numeric_conds.insert(*numeric_condition, (fluents, weights));
        if is_lt {
            lt_simple_numeric_conds.insert(*numeric_condition);
        }
    } else {
        complex_numeric_conds.insert(*numeric_condition);
    }
}

/// Extracts fluents and weights from a simple numeric condition.
///
/// This function attempts to interpret a numeric condition of the form
/// `linear_expression < constant` or `linear_expression <= constant` as a
/// linear polynomial and extract its components:
///
/// - `fluents`: A vector of fluents appearing in the expression.
/// - `weights`: Corresponding coefficients of the fluents, with the constant
///   term appended as the last element.
/// - `is_lt`: `true` if the original operator was `<`, `false` if `<=`.
///
/// # Arguments
///
/// * `expr` - The expression to analyze.
/// * `expression_manager` - A mutable reference to the `ExpressionManager`.
///
/// # Returns
///
/// Returns `Some((fluents, weights, is_lt))` if the condition is a simple linear
/// numeric condition; otherwise, returns `None`.
fn extract_fluents_weights_simple_numeric_condition(
    expr: &Expression,
    expression_manager: &ExpressionManager,
) -> Option<(Vec<usize>, Vec<f64>, bool)> {
    let expr = expression_manager.force_get(expr);
    let root_node = expr.last()?;
    let (op1, op2) = match root_node {
        ExpressionNode::LT(op1, op2) | ExpressionNode::LE(op1, op2) => (op1, op2),
        _ => return None,
    };

    let mut polynomial_expr = expr.clone();
    polynomial_expr.pop();
    polynomial_expr.push(ExpressionNode::Minus(*op1, *op2));
    let mut polynomial = to_linear_polynomial(&polynomial_expr)?;

    let k = polynomial.remove(&None).unwrap_or(0.0);
    let (fluents, mut weights): (Vec<_>, Vec<_>) =
        polynomial.iter().map(|(f, w)| (f.unwrap(), *w)).unzip();
    weights.push(k);
    Some((
        fluents,
        weights,
        matches!(root_node, ExpressionNode::LT(_, _)),
    ))
}

/// Converts an expression into a linear polynomial representation.
///
/// This function attempts to represent a numeric expression as a linear polynomial of the form:
///
/// ```text
/// w1 * f1 + w2 * f2 + ... + k
/// ```
///
/// where `fi` are fluents, `wi` are their coefficients, and `k` is a constant term.
///
/// Supported operations are `+`, `-`, `*`, and `/`, provided they maintain linearity.
/// If the expression is non-linear (e.g., a product of two fluents or division by a fluent),
/// the function returns `None`.
///
/// # Arguments
///
/// * `expr` - A vector of `ExpressionNode` representing the numeric expression.
///
/// # Returns
///
/// Returns `Some(FxHashMap<Option<usize>, f64>)` mapping fluents to coefficients,
/// with `None` representing the constant term. Returns `None` if the expression is non-linear.
fn to_linear_polynomial(expr: &Vec<ExpressionNode>) -> Option<FxHashMap<Option<usize>, f64>> {
    let mut res = Vec::new();
    for node in expr {
        match node {
            ExpressionNode::Int(v) => {
                let mut p = FxHashMap::with_hasher(FxBuildHasher::default());
                p.insert(None, integer_to_f64(v));
                res.push(p);
            }
            ExpressionNode::Rational(v) => {
                let mut p = FxHashMap::with_hasher(FxBuildHasher::default());
                p.insert(None, rational_to_f64(v));
                res.push(p);
            }
            ExpressionNode::Fluent(f) => {
                let mut p = FxHashMap::with_hasher(FxBuildHasher::default());
                p.insert(Some(*f), 1.0);
                res.push(p);
            }
            ExpressionNode::Minus(_, _) => {
                let p2 = res.pop().unwrap();
                let p1 = res.last_mut().unwrap();
                for (f, w) in p2 {
                    *p1.entry(f).or_insert(0.0) -= w;
                }
            }
            ExpressionNode::Plus(operands) => {
                let mut p = res.pop().unwrap();
                for _ in 1..operands.len() {
                    for (f, w) in res.pop().unwrap() {
                        *p.entry(f).or_insert(0.0) += w;
                    }
                }
                res.push(p);
            }
            ExpressionNode::Div(_, _) => {
                let divisor = res.pop().unwrap();
                let dividend = res.last_mut().unwrap();
                if !is_constant(&divisor) {
                    return None;
                }
                for (f, w) in divisor {
                    *dividend.entry(f).or_insert(0.0) /= w;
                }
            }
            ExpressionNode::Times(operands) => {
                let mut const_multiplier = 1.0;
                let mut polynomial = None;
                for _ in 0..operands.len() {
                    let operand = res.pop().unwrap();
                    if is_constant(&operand) {
                        const_multiplier *= operand.get(&None).unwrap();
                    } else if polynomial.is_some() {
                        return None;
                    } else {
                        polynomial = Some(operand);
                    }
                }

                res.push(polynomial.unwrap_or_else(|| {
                    let mut p = FxHashMap::with_hasher(FxBuildHasher::default());
                    p.insert(None, const_multiplier);
                    p
                }))
            }
            _ => return None,
        }
    }

    Some(res.pop().unwrap())
}

fn is_constant(polynomial: &FxHashMap<Option<usize>, f64>) -> bool {
    polynomial.len() == 1 && polynomial.contains_key(&None)
}

/// Checks whether an operator achieves a given simple numeric condition.
///
/// The check considers:
/// - If the operator has a constant assignment or complex effect on any of the
///   fluents, the condition is considered achieved.
/// - Otherwise, the net effect of the operator on the condition is
///   computed. If the net effect is negative, the condition is considered
///   potentially achieved.
///
/// The `max_net_effect` is updated if the current net effect is the largest
/// negative effect seen so far.
///
/// # Arguments
///
/// * `operator` - The operator whose effects are being evaluated.
/// * `fluents` - Vector of fluents involved in the condition.
/// * `weights` - Corresponding weights for each fluent in the condition.
/// * `max_net_effect` - Mutable reference to the maximum negative net effect
///   seen so far; updated if current net effect is larger.
///
/// # Returns
///
/// Returns `true` if the operator achieves the condition, otherwise `false`.
fn achieves(
    operator: &Operator,
    fluents: &Vec<usize>,
    weights: &Vec<f64>,
    max_net_effect: &mut f64,
) -> bool {
    let mut net_effect = 0.0;
    for (f, w) in fluents.iter().zip(weights) {
        if operator.constant_assign_effects.contains_key(f)
            || operator.complex_numeric_effects.contains_key(f)
        {
            return true;
        }
        if let Some(k) = operator.constant_increase_effects.get(f) {
            net_effect += w * k;
        }
    }
    if net_effect < 0.0 && net_effect > *max_net_effect {
        *max_net_effect = net_effect;
    }
    net_effect < 0.0
}

/// Estimates the number of applications of an operator needed to satisfy a simple numeric condition.
///
/// This function computes the minimum number of times `operator` must be applied
/// to a given `state` for the numeric condition represented by `fluents` and `weights`
/// to become satisfied.
///
/// The computation follows these rules:
/// - If the condition is already satisfied in the state, returns 0.
/// - If the operator has a constant assignment or complex effect on any fluent
///   in the condition, returns 1, assuming one application is sufficient.
/// - Otherwise, computes the net effect of the operator on the condition and
///   return the minimum number of repetitions needed.
///
/// # Arguments
///
/// * `operator` - The operator whose effects are being evaluated.
/// * `fluents` - Vector of fluents involved in the condition.
/// * `weights` - Corresponding weights for each fluent, with the constant term last.
/// * `state` - The state on which the condition is evaluated.
///
/// # Returns
///
/// Returns `Ok(Some(value))` with the minimum number of applications needed to satisfy
/// the condition, `Ok(Some(0.0))` if already satisfied, `Ok(Some(1.0))` if a constant/complex
/// effect applies, or `Ok(None)` if the condition cannot be satisfied.
fn repetitions(
    operator: &Operator,
    fluents: &Vec<usize>,
    weights: &Vec<f64>,
    state: &State,
) -> PyResult<Option<f64>> {
    let mut v = *weights.last().unwrap();
    for (f, w) in fluents.iter().zip(weights) {
        let f_value = rational_to_f64(&get_rational_from_expression_node(state.get_value(*f))?);
        v += *w * f_value;
    }

    if v <= 0.0 {
        // condition satisfied in state
        return Ok(Some(0.0));
    }

    for f in fluents {
        if operator.constant_assign_effects.contains_key(f)
            || operator.complex_numeric_effects.contains_key(f)
        {
            return Ok(Some(1.0));
        }
    }

    let mut net_effect = 0.0;
    for (f, w) in fluents.iter().zip(weights) {
        if let Some(k) = operator.constant_increase_effects.get(f) {
            net_effect += w * k;
        }
    }

    if net_effect >= 0.0 {
        return Ok(None);
    }

    Ok(Some((-v / net_effect).ceil()))
}

/// Extract the sub-expression from a given expression rooted at a specified index.
/// All operands in the extracted sub-expression are re-indexed relative to the
/// start of the sub-expression.
///
/// # Arguments
///
/// * `expr` - A reference to the full expression (`Vec<ExpressionNode>`) from which to extract the sub-expression.
/// * `idx` - The index of the root node of the sub-expression.
///
/// # Returns
///
/// Returns a `Result` containing a `Vec<ExpressionNode>` representing the extracted
/// sub-expression with all operand indices re-indexed relative to the start of
/// the sub-expression, or an `ArithmeticError` if extraction fails.
fn extract_sub_expression(
    expr: &Vec<ExpressionNode>,
    idx: usize,
) -> Result<Vec<ExpressionNode>, ArithmeticError> {
    // find the start index of the sub-expression
    let mut i = idx;
    loop {
        // assumes operand indices are in ascending order
        i = match &expr[i] {
            ExpressionNode::Not(operand) => *operand,
            ExpressionNode::Equals(op1, _)
            | ExpressionNode::LE(op1, _)
            | ExpressionNode::LT(op1, _)
            | ExpressionNode::Minus(op1, _)
            | ExpressionNode::Div(op1, _) => *op1,
            ExpressionNode::And(operands)
            | ExpressionNode::Or(operands)
            | ExpressionNode::Plus(operands)
            | ExpressionNode::Times(operands) => operands[0],
            _ => break,
        };
    }

    shift_expression(&expr[i..(idx + 1)], i, true)
}

#[derive(Clone, Debug)]
pub struct DeleteRelaxationHeuristic {
    actions: Vec<Action>,
    events: FxHashMap<Action, usize>,
    goals: HeuristicExpression,
    extra_fluents: FxHashMap<Action, Vec<Expression>>,
    extra_goals: HeuristicExpression,
    operators: Vec<Operator>,
    precondition_of: FxHashMap<Expression, Vec<OperatorID>>,
    empty_pre_operators: FxHashSet<OperatorID>,
    simple_numeric_conds: FxHashMap<Expression, (Vec<usize>, Vec<f64>)>,
    complex_numeric_conds: FxHashSet<Expression>,
    achieved_simple_numeric_conds: Vec<Vec<Expression>>,
    heuristic_kind: HeuristicKind,
    internal_caching: Arc<Mutex<Option<FxHashMap<CacheKey, Option<f64>>>>>,
    expression_manager: Arc<Mutex<ExpressionManager>>,
    disable_numeric_reasoning: bool,
}

impl DeleteRelaxationHeuristic {
    pub fn new(
        actions: Vec<Action>,
        fluent_types: Vec<String>,
        objects: FxHashMap<String, Vec<String>>,
        events: FxHashMap<Action, Vec<(Timing, Event)>>,
        goals: Vec<PyExpressionNode>,
        heuristic_kind: HeuristicKind,
        internal_caching: bool,
        disable_numeric_reasoning: bool,
    ) -> PyResult<Self> {
        let mut operators = Vec::with_capacity(events.iter().map(|(_, e)| e.len()).sum());
        let mut extra_fluents: FxHashMap<Action, Vec<Expression>> =
            FxHashMap::with_capacity_and_hasher(events.len(), FxBuildHasher::default());
        let mut extra_goals = Vec::with_capacity(events.len() + 1);
        let mut expression_manager = ExpressionManager::new();
        let mut num_fluents = fluent_types.len();
        let map_to_python_exception = |e| PyException::new_err(format!("{:?}", e));

        for a in &actions {
            let Some(le) = events.get(a) else {
                continue;
            };
            let mut a_extra_fluents: Vec<Expression> = Vec::new();
            let f_cond = num_fluents + le.len() - 1;
            let mut cond = ExpressionNode::Fluent(f_cond);
            extra_goals.push(cond.clone());
            for (_, e) in le.iter() {
                let mut effects: Vec<Expression> = Vec::new();
                let mut constant_increase_effects: FxHashMap<usize, f64> =
                    FxHashMap::with_hasher(FxBuildHasher::default());
                let mut constant_assign_effects: FxHashMap<usize, f64> =
                    FxHashMap::with_hasher(FxBuildHasher::default());
                let mut complex_numeric_effects: FxHashMap<usize, Expression> =
                    FxHashMap::with_hasher(FxBuildHasher::default());
                let f = num_fluents;
                num_fluents += 1;
                a_extra_fluents.push(expression_manager.put(&vec![ExpressionNode::Fluent(f)]));
                effects.push(expression_manager.put(&vec![ExpressionNode::Fluent(f)]));
                for eff in e.effects.iter() {
                    let t = fluent_types[eff.fluent].to_string();
                    if t == "bool" {
                        if eff.value.len() == 1 {
                            if let ExpressionNode::Bool(value) = eff.value[0] {
                                if value {
                                    effects.push(
                                        expression_manager
                                            .put(&vec![ExpressionNode::Fluent(eff.fluent)]),
                                    );
                                } else {
                                    effects.push(expression_manager.put(&vec![
                                        ExpressionNode::Fluent(eff.fluent),
                                        make_operator("not".to_string(), vec![0])?,
                                    ]));
                                }
                            } else {
                                effects.push(
                                    expression_manager
                                        .put(&vec![ExpressionNode::Fluent(eff.fluent)]),
                                );
                                effects.push(expression_manager.put(&vec![
                                    ExpressionNode::Fluent(eff.fluent),
                                    make_operator("not".to_string(), vec![0])?,
                                ]));
                            }
                        } else {
                            effects.push(
                                expression_manager.put(&vec![ExpressionNode::Fluent(eff.fluent)]),
                            );
                            effects.push(expression_manager.put(&vec![
                                ExpressionNode::Fluent(eff.fluent),
                                make_operator("not".to_string(), vec![0])?,
                            ]));
                        }
                    } else if t == "real" || t == "int" {
                        assert!(
                            !constant_increase_effects.contains_key(&eff.fluent)
                                && !constant_assign_effects.contains_key(&eff.fluent)
                                && !complex_numeric_effects.contains_key(&eff.fluent)
                        );
                        update_numeric_effects(
                            eff,
                            &mut expression_manager,
                            &mut constant_increase_effects,
                            &mut constant_assign_effects,
                            &mut complex_numeric_effects,
                        );
                    } else {
                        if eff.value.len() == 1 && matches!(eff.value[0], ExpressionNode::Object(_))
                        {
                            effects.push(expression_manager.put(&vec![
                                ExpressionNode::Fluent(eff.fluent),
                                eff.value[0].clone(),
                                make_operator("==".to_string(), vec![0, 1])?,
                            ]));
                        } else {
                            for o in objects[&t].iter() {
                                effects.push(expression_manager.put(&vec![
                                    ExpressionNode::Fluent(eff.fluent),
                                    ExpressionNode::Object(o.to_string()),
                                    make_operator("==".to_string(), vec![0, 1])?,
                                ]));
                            }
                        }
                    }
                }

                if let Some(conditions) = build_operator_condition(
                    &get_event_conditions(e, &mut expression_manager)?,
                    cond.clone(),
                    &objects,
                    &fluent_types,
                    disable_numeric_reasoning,
                    &mut expression_manager,
                )
                .map_err(map_to_python_exception)?
                {
                    operators.push(Operator {
                        id: OperatorID::new(operators.len()),
                        action: *a,
                        conditions,
                        effects,
                        constant_increase_effects,
                        constant_assign_effects,
                        complex_numeric_effects,
                        cost: 1.0,
                    });
                }
                cond = ExpressionNode::Fluent(f);
            }
            extra_fluents.insert(*a, a_extra_fluents);
        }
        operators.sort_by(|a, b| a.action.cmp(&b.action));

        let expr_goals = goals.into_iter().map(|e| e.v).collect();
        let goals = convert_to_heuristic_expression(&expr_goals, &mut expression_manager)
            .map_err(map_to_python_exception)?;
        let goals = simplify_condition(
            &goals,
            &objects,
            &fluent_types,
            disable_numeric_reasoning,
            &mut expression_manager,
        )
        .map_err(map_to_python_exception)?;
        extra_goals.push(ExpressionNode::And((0..extra_goals.len()).collect()));
        let extra_goals = convert_to_heuristic_expression(&extra_goals, &mut expression_manager)
            .map_err(map_to_python_exception)?;

        let mut precondition_of: FxHashMap<Expression, Vec<OperatorID>> =
            FxHashMap::with_hasher(FxBuildHasher::default());
        let mut simple_numeric_conds: FxHashMap<Expression, (Vec<usize>, Vec<f64>)> =
            FxHashMap::with_hasher(FxBuildHasher::default());
        let mut lt_simple_numeric_conds: FxHashSet<Expression> =
            FxHashSet::with_hasher(FxBuildHasher::default());
        let mut complex_numeric_conds: FxHashSet<Expression> =
            FxHashSet::with_hasher(FxBuildHasher::default());
        let mut empty_pre_operators: FxHashSet<OperatorID> =
            FxHashSet::with_hasher(FxBuildHasher::default());
        for o in &operators {
            if o.conditions.expression.is_empty() {
                empty_pre_operators.insert(o.id);
            } else {
                for node in &o.conditions.expression {
                    if let HeuristicExpressionNode::Leaf(e) = node {
                        let expr = expression_manager.force_get(e);
                        if is_numeric_leaf_expression(expr) {
                            update_numeric_conditions(
                                e,
                                &expression_manager,
                                &mut simple_numeric_conds,
                                &mut lt_simple_numeric_conds,
                                &mut complex_numeric_conds,
                                disable_numeric_reasoning,
                            );
                        }

                        precondition_of
                            .entry(*e)
                            .or_insert_with(Vec::new)
                            .push(o.id);
                    }
                }
            }
        }

        for node in goals.expression.iter() {
            if let HeuristicExpressionNode::Leaf(e) = node {
                let expr = expression_manager.force_get(e);
                if is_numeric_leaf_expression(expr) {
                    update_numeric_conditions(
                        e,
                        &expression_manager,
                        &mut simple_numeric_conds,
                        &mut lt_simple_numeric_conds,
                        &mut complex_numeric_conds,
                        disable_numeric_reasoning,
                    );
                }
            }
        }

        let mut max_net_effect = f64::MIN;
        let mut achieved_simple_numeric_conds: Vec<Vec<Expression>> =
            vec![Vec::new(); operators.len()];
        for o in &operators {
            for (c, (fluents, weights)) in &simple_numeric_conds {
                if achieves(o, fluents, weights, &mut max_net_effect) {
                    achieved_simple_numeric_conds[o.id.id].push(*c);
                }
            }
        }

        let epsilon = -max_net_effect / 2.0;
        for simple_cond in lt_simple_numeric_conds {
            if let Some((_, weights)) = simple_numeric_conds.get_mut(&simple_cond) {
                if let Some(k) = weights.last_mut() {
                    *k += epsilon;
                }
            }
        }

        let events_len: FxHashMap<Action, usize> =
            events.into_iter().map(|(a, ev)| (a, ev.len())).collect();

        let internal_caching = if internal_caching {
            Some(FxHashMap::with_hasher(FxBuildHasher::default()))
        } else {
            None
        };

        let res = DeleteRelaxationHeuristic {
            actions,
            events: events_len,
            goals,
            extra_fluents,
            extra_goals,
            operators,
            precondition_of,
            empty_pre_operators,
            simple_numeric_conds,
            complex_numeric_conds,
            achieved_simple_numeric_conds,
            heuristic_kind,
            internal_caching: Arc::new(Mutex::new(internal_caching)),
            expression_manager: Arc::new(Mutex::new(expression_manager)),
            disable_numeric_reasoning,
        };
        Ok(res)
    }

    pub fn eval(&self, state: &State) -> PyResult<Option<f64>> {
        let mut internal_caching = self.internal_caching.lock().unwrap();
        if let Some(internal_caching) = internal_caching.as_mut() {
            let values: Vec<ExpressionNode> = state.assignments.iter().cloned().collect();
            let todo_values: Vec<usize> = self
                .actions
                .iter()
                .map(|action| state.todo.get(action).map(|(j, _)| *j).unwrap_or(0))
                .collect();
            let cache_key = CacheKey {
                values,
                todo_values,
            };
            if let Some(res) = internal_caching.get(&cache_key) {
                return Ok(res.clone());
            }

            let result = self._eval(state);
            if let Ok(v) = result {
                internal_caching.insert(cache_key, v);
            }
            result
        } else {
            self._eval(state)
        }
    }

    /// Computes the heuristic value for a given state.
    ///
    /// This method evaluates the state using the selected delete-relaxation heuristic,
    /// which can be one of `hmax`, `hadd`, or `hff`. The returned value estimates
    /// the cost to reach the goal from the given state.
    ///
    /// # Arguments
    ///
    /// * `state` - The state to evaluate.
    ///
    /// # Returns
    ///
    /// Returns `Ok(Some(value))` with the heuristic value as a floating-point number
    /// if it can be computed, or `Ok(None)` if the heuristic cannot be evaluated.
    fn _eval(&self, state: &State) -> PyResult<Option<f64>> {
        let mut expression_manager = self.expression_manager.lock().unwrap();
        let mut costs: FxHashMap<Expression, f64> = FxHashMap::with_capacity_and_hasher(
            state.assignments.len()
                + self.simple_numeric_conds.len()
                + self.complex_numeric_conds.len()
                + self.events.len(),
            FxBuildHasher::default(),
        );

        for (f, v) in state.assignments.iter().enumerate() {
            let k = match v {
                ExpressionNode::Bool(value) => {
                    if *value {
                        vec![ExpressionNode::Fluent(f)]
                    } else {
                        vec![
                            ExpressionNode::Fluent(f),
                            make_operator("not".to_string(), vec![0])?,
                        ]
                    }
                }
                _ => {
                    vec![
                        ExpressionNode::Fluent(f),
                        v.clone(),
                        make_operator("==".to_string(), vec![0, 1])?,
                    ]
                }
            };
            let k = expression_manager.put(&k);
            costs.insert(k, 0.0);
        }

        for c in self.simple_numeric_conds.keys() {
            if internal_evaluate(expression_manager.force_get(c), state)?
                == ExpressionNode::Bool(true)
            {
                costs.insert(*c, 0.0);
            }
        }

        for c in &self.complex_numeric_conds {
            if internal_evaluate(expression_manager.force_get(c), state)?
                == ExpressionNode::Bool(true)
            {
                costs.insert(*c, 0.0);
            } else {
                costs.insert(*c, 1.0);
            }
        }

        for a in self.events.keys() {
            let v = match state.todo.get(a) {
                Some((j, _)) => self.extra_fluents[a][j - 1],
                None => *self.extra_fluents[a].last().unwrap(),
            };
            costs.insert(v, 0.0);
        }

        let mut lp: Vec<Expression> = costs.keys().copied().collect();
        let mut lo: FxHashSet<OperatorID> = FxHashSet::with_hasher(FxBuildHasher::default());
        let mut reached_by: FxHashMap<Expression, OperatorID> =
            FxHashMap::with_hasher(FxBuildHasher::default());
        let mut operator_cost = vec![None; self.operators.len()];
        let mut new_costs = FxHashMap::with_hasher(FxBuildHasher::default());
        let mut poss = FxHashMap::with_hasher(FxBuildHasher::default());
        while lp.len() > 0 {
            lo.extend(&self.empty_pre_operators);
            for p in lp.iter() {
                if let Some(po) = self.precondition_of.get(p) {
                    lo.extend(po);
                }
            }
            for oid in lo.drain() {
                let o: &Operator = &self.operators[oid.id];
                let c = self.cost(&o.conditions, &costs);
                let op_cost = operator_cost[oid.id];
                if c.is_some() && (op_cost.is_none() || op_cost > c) {
                    operator_cost[oid.id] = c;
                    let c = c.unwrap();

                    let mut achieved_expressions: Vec<_> =
                        o.effects.iter().map(|k| (k, o.cost + c)).collect();

                    for simple_cond in &self.achieved_simple_numeric_conds[oid.id] {
                        if costs.get(simple_cond) == Some(&0.0) {
                            // condition satisfied in state
                            continue;
                        }

                        let (fluents, weights) = &self.simple_numeric_conds[simple_cond];
                        let rep = repetitions(o, fluents, weights, state)?.unwrap();

                        let expr_cost = if matches!(self.heuristic_kind, HeuristicKind::HMAX) {
                            poss.entry(simple_cond)
                                .or_insert_with(|| FxHashSet::with_hasher(FxBuildHasher::default()))
                                .insert(o.id);

                            let min_operator_cost = poss
                                .get(simple_cond)
                                .unwrap()
                                .iter()
                                .map(|oid| {
                                    self.cost(&self.operators[oid.id].conditions, &costs)
                                        .unwrap()
                                })
                                .reduce(f64::min);
                            rep * o.cost + min_operator_cost.unwrap()
                        } else {
                            rep * o.cost + c
                        };

                        achieved_expressions.push((simple_cond, expr_cost));
                    }

                    for (expr, expr_cost) in achieved_expressions {
                        let prev_expr_cost =
                            new_costs.get(expr).or_else(|| costs.get(expr)).copied();
                        if prev_expr_cost.is_none() || expr_cost < prev_expr_cost.unwrap() {
                            if matches!(self.heuristic_kind, HeuristicKind::HFF) {
                                reached_by.insert(*expr, oid);
                            }
                            new_costs.insert(*expr, expr_cost);
                        } else if prev_expr_cost == Some(expr_cost)
                            && matches!(self.heuristic_kind, HeuristicKind::HFF)
                            && oid.id > reached_by[expr].id
                        {
                            reached_by.insert(*expr, oid);
                        }
                    }
                }
            }
            lp.clear();
            lp.extend(new_costs.keys());
            costs.extend(new_costs.drain());
        }

        let h = self.cost(&self.goals, &costs);
        if h.is_none() {
            return Ok(None);
        }

        if matches!(
            self.heuristic_kind,
            HeuristicKind::HADD | HeuristicKind::HMAX
        ) {
            return match self.cost(&self.extra_goals, &costs) {
                Some(v) => {
                    let res = if let HeuristicKind::HMAX = self.heuristic_kind {
                        f64::max(h.unwrap(), v)
                    } else {
                        h.unwrap() + v
                    };
                    Ok(Some(res))
                }
                None => Ok(None),
            };
        }

        let mut res = 0.0;
        for (a, (j, _)) in state.todo.iter() {
            res += (self.events[a] - j) as f64;
        }

        if let Some(hv) = h {
            if hv == 0.0 {
                return Ok(Some(res));
            }
        }

        let mut relaxed_plan = FxHashSet::with_hasher(FxBuildHasher::default());
        let mut tmp_set = FxHashSet::with_hasher(FxBuildHasher::default()); // avoid reallocating the FxHashSet inside hff_leaves
        let mut stack: Vec<Expression> = {
            self.hff_leaves(&self.goals, &costs, &mut tmp_set);
            tmp_set.drain().collect()
        };
        let mut visited_expressions: FxHashSet<Expression> = stack.iter().copied().collect();

        while let Some(g) = stack.pop() {
            if let Some(oid) = reached_by.get(&g) {
                let o = &self.operators[oid.id];
                relaxed_plan.insert(&o.action);

                self.hff_leaves(&o.conditions, &costs, &mut tmp_set);
                for expr in tmp_set.drain() {
                    if visited_expressions.insert(expr) {
                        stack.push(expr);
                    }
                }
            }
        }
        for a in relaxed_plan {
            if !state.todo.contains_key(a) {
                res += self.events[a] as f64;
            }
        }

        Ok(Some(res))
    }

    /// Collect the leaf expressions that contribute to the cost of the expression.
    ///
    /// Leaf expressions are collected according to the type of node:
    /// - AND nodes: all leaf expressions from the operands are included
    /// - OR nodes: only the leaf expressions from the operand with the minimum cost are included
    ///
    /// # Arguments
    ///
    /// * `expr` - The `HeuristicExpression` to evaluate.
    /// * `costs` - A mapping from leaf expressions (`Expression`) to their costs (`f64`).
    /// * `out` - A mutable `FxHashSet` where the contributing leaf expressions will be collected.
    ///
    /// # Notes
    ///
    /// - This method does not return a value; it updates the `out` set in place.
    fn hff_leaves<'a>(
        &'a self,
        expr: &'a HeuristicExpression,
        costs: &'a FxHashMap<Expression, f64>,
        out: &mut FxHashSet<Expression>,
    ) {
        if !expr.contains_or_node {
            for node in &expr.expression {
                if let HeuristicExpressionNode::Leaf(e) = node {
                    out.insert(*e);
                }
            }
            return;
        }

        let mut res: Vec<(Option<f64>, Vec<Expression>)> = Vec::new();
        for node in &expr.expression {
            match node {
                HeuristicExpressionNode::Leaf(e) => res.push((costs.get(e).cloned(), vec![*e])),
                HeuristicExpressionNode::And(num_operands) => {
                    let mut r = 0.0;
                    let mut l = Vec::new();
                    let mut is_none = false;
                    for i in 0..*num_operands {
                        match &res[res.len() - i - 1] {
                            (Some(v), ol) => {
                                if let HeuristicKind::HMAX = self.heuristic_kind {
                                    r = f64::max(r, *v)
                                } else {
                                    l.extend(ol);
                                    r += v;
                                }
                            }
                            (None, _) => {
                                is_none = true;
                                break;
                            }
                        }
                    }
                    res.truncate(res.len() - num_operands);
                    if is_none {
                        res.push((None, vec![]));
                    } else {
                        res.push((Some(r), l));
                    }
                }
                HeuristicExpressionNode::Or(num_operands) => {
                    let mut r = f64::MAX;
                    let mut ml = vec![];
                    for _ in 0..*num_operands {
                        if let (Some(v), ol) = res.pop().unwrap() {
                            if v < r {
                                r = v;
                                ml = ol;
                            }
                        }
                    }
                    if r == f64::MAX {
                        res.push((None, ml));
                    } else {
                        res.push((Some(r), ml));
                    }
                }
            }
        }

        assert!(res.len() == 1);
        out.extend(res.pop().unwrap().1);
    }

    /// Calculate the cost of an expression.
    ///
    /// # Arguments
    ///
    /// * `expr` - The `HeuristicExpression` to evaluate.
    /// * `costs` - A mapping from leaf expressions (`Expression`) to their costs (`f64`).
    ///
    /// # Returns
    ///
    /// Returns the total cost of the expression.
    fn cost(&self, expr: &HeuristicExpression, costs: &FxHashMap<Expression, f64>) -> Option<f64> {
        if let HeuristicExpressionNode::Leaf(e) = expr.expression.last().unwrap() {
            return costs.get(e).cloned();
        }

        let mut res: Vec<Option<f64>> = Vec::new();
        for node in &expr.expression {
            match node {
                HeuristicExpressionNode::Leaf(e) => res.push(costs.get(e).cloned()),
                HeuristicExpressionNode::And(num_operands) => {
                    let mut r = 0.0;
                    let mut is_none = false;
                    for i in 0..*num_operands {
                        match res[res.len() - i - 1] {
                            Some(v) => {
                                if let HeuristicKind::HMAX = self.heuristic_kind {
                                    r = f64::max(r, v)
                                } else {
                                    r += v
                                }
                            }
                            None => {
                                is_none = true;
                                break;
                            }
                        }
                    }
                    res.truncate(res.len() - num_operands);
                    if is_none {
                        res.push(None);
                    } else {
                        res.push(Some(r));
                    }
                }
                HeuristicExpressionNode::Or(num_operands) => {
                    let mut r = f64::MAX;
                    for _ in 0..*num_operands {
                        let operand_value = res.pop().unwrap();
                        match operand_value {
                            Some(v) => r = f64::min(r, v),
                            None => {}
                        }
                    }
                    if r == f64::MAX {
                        res.push(None);
                    } else {
                        res.push(Some(r));
                    }
                }
            }
        }

        assert!(res.len() == 1);
        res.last().copied().flatten()
    }

    pub fn name(&self) -> String {
        let mut name = String::from(match self.heuristic_kind {
            HeuristicKind::HFF => "hff",
            HeuristicKind::HADD => "hadd",
            HeuristicKind::HMAX => "hmax",
        });
        if self.disable_numeric_reasoning {
            name.push_str("_no_numbers");
        }
        name
    }
}

pub struct FluentAssignments<'a> {
    pub assignments: FxHashMap<usize, &'a ExpressionNode>,
}

impl FluentValueTrait for FluentAssignments<'_> {
    fn get_value(&self, fluent: usize) -> &ExpressionNode {
        self.assignments.get(&fluent).unwrap()
    }
}

impl<'a> FluentAssignments<'a> {
    pub fn new(fluents: &Vec<usize>, values: Vec<&'a ExpressionNode>) -> Self {
        let assignments: FxHashMap<usize, &ExpressionNode> =
            fluents.iter().cloned().zip(values.into_iter()).collect();
        FluentAssignments { assignments }
    }
}

#[derive(Clone, Debug)]
pub struct HMaxExplicit {
    actions: Vec<Action>,
    goals: Vec<Vec<ExpressionNode>>,
    goal_expressions: Vec<Expression>,
    extra_fluents: FxHashMap<Action, Vec<Vec<ExpressionNode>>>,
    num_fluents: usize,
    operators: Vec<OperatorHmax>,
    operator_conditions_fluents: Vec<FxHashSet<usize>>,
    operator_effects_fluents: Vec<FxHashSet<usize>>,
    internal_caching: Arc<Mutex<Option<FxHashMap<CacheKey, Option<f64>>>>>,
}

impl HMaxExplicit {
    pub fn new(
        actions: Vec<Action>,
        fluent_types: Vec<String>,
        events: FxHashMap<Action, Vec<(Timing, Event)>>,
        goals: Vec<PyExpressionNode>,
        internal_caching: bool,
    ) -> PyResult<Self> {
        let mut operators = Vec::new();
        let mut extra_fluents = FxHashMap::with_hasher(FxBuildHasher::default());
        let mut extra_goals = Vec::new();
        let mut expression_manager = ExpressionManager::new();
        let mut num_fluents = fluent_types.len();

        for (a, le) in events.iter() {
            let mut a_extra_fluents = Vec::new();
            let f_cond = num_fluents + le.len() - 1;
            let mut cond: Vec<ExpressionNode> = vec![ExpressionNode::Fluent(f_cond)];
            extra_goals.push(cond.clone());
            for (_, e) in le.iter() {
                let mut effects = Vec::new();
                let mut conditions = Vec::new();
                let f = num_fluents;
                num_fluents += 1;
                a_extra_fluents.push(vec![ExpressionNode::Fluent(f)]);
                effects.push(Effect {
                    fluent: f,
                    value: vec![ExpressionNode::Bool(true)],
                });
                for eff in e.effects.iter() {
                    effects.push(eff.clone());
                }
                conditions.push(cond);
                for condition in get_event_conditions(e, &mut expression_manager)? {
                    if condition.len() > 0 && condition != vec![ExpressionNode::Bool(true)] {
                        conditions.extend(split_expression(&condition)?);
                    }
                }
                if !conditions.contains(&vec![ExpressionNode::Bool(false)]) {
                    let condition_expressions: Vec<Expression> = conditions
                        .iter()
                        .map(|cond| expression_manager.put(cond))
                        .collect();
                    operators.push(OperatorHmax {
                        action: *a,
                        conditions,
                        condition_expressions,
                        effects,
                        cost: 1.0,
                    });
                }
                cond = vec![ExpressionNode::Fluent(f)];
            }
            extra_fluents.insert(*a, a_extra_fluents);
        }

        let mut goals = split_expression(&goals.into_iter().map(|e| e.v).collect())?;
        goals.extend(extra_goals);
        let goal_expressions: Vec<Expression> = goals
            .iter()
            .map(|cond| expression_manager.put(cond))
            .collect();

        let mut operator_conditions_fluents = Vec::with_capacity(operators.len());
        for operator in &operators {
            let mut conditions_fluents = FxHashSet::with_hasher(FxBuildHasher::default());
            for cond in &operator.conditions {
                for exp_node in cond {
                    if let ExpressionNode::Fluent(f) = exp_node {
                        conditions_fluents.insert(f.clone());
                    }
                }
            }
            operator_conditions_fluents.push(conditions_fluents);
        }

        let mut operator_effects_fluents = Vec::with_capacity(operators.len());
        for operator in &operators {
            let mut effects_fluents = FxHashSet::with_hasher(FxBuildHasher::default());
            for eff in &operator.effects {
                for exp_node in &eff.value {
                    if let ExpressionNode::Fluent(f) = exp_node {
                        effects_fluents.insert(f.clone());
                    }
                }
            }
            operator_effects_fluents.push(effects_fluents);
        }

        let internal_caching = if internal_caching {
            Some(FxHashMap::with_hasher(FxBuildHasher::default()))
        } else {
            None
        };

        let res = HMaxExplicit {
            actions,
            goals,
            goal_expressions,
            extra_fluents,
            num_fluents,
            operators,
            operator_conditions_fluents,
            operator_effects_fluents,
            internal_caching: Arc::new(Mutex::new(internal_caching)),
        };
        Ok(res)
    }

    fn extract_fluents(&self, exp: &Vec<ExpressionNode>) -> Vec<usize> {
        let mut exp_fluents = Vec::new();
        for exp_node in exp {
            if let ExpressionNode::Fluent(f) = exp_node {
                exp_fluents.push(f.clone());
            }
        }

        exp_fluents
    }

    fn possible_values<'a>(
        &'a self,
        exp: &'a Vec<ExpressionNode>,
        assignments: &'a Vec<FxHashSet<ExpressionNode>>,
        exp_fluents: &'a Vec<usize>,
    ) -> impl Iterator<Item = ExpressionNode> + 'a {
        let values: Vec<&FxHashSet<ExpressionNode>> =
            exp_fluents.iter().map(|&f| &assignments[f]).collect();

        values
            .iter()
            .map(|fluent_values| fluent_values.iter())
            .multi_cartesian_product()
            .map(move |state_values: Vec<&ExpressionNode>| {
                let exp_assignments = FluentAssignments::new(exp_fluents, state_values);
                internal_evaluate(exp, &exp_assignments).unwrap()
            })
    }

    fn exp_can_be_true(
        &self,
        exp: &Vec<ExpressionNode>,
        exp_id: Expression,
        assignments: &Vec<FxHashSet<ExpressionNode>>,
        assignments_changes: &FxHashSet<usize>,
        cache_can_be_true: &mut FxHashMap<Expression, bool>,
    ) -> bool {
        let exp_fluents;
        if cache_can_be_true.contains_key(&exp_id) {
            if cache_can_be_true[&exp_id] {
                return true;
            }

            exp_fluents = self.extract_fluents(exp);
            let exp_fluents_set: FxHashSet<usize> = exp_fluents.iter().copied().collect();
            if exp_fluents_set.is_disjoint(assignments_changes) {
                return false;
            }
        } else {
            exp_fluents = self.extract_fluents(exp);
        }

        let possible_values = self.possible_values(exp, assignments, &exp_fluents);

        for value in possible_values {
            if value == ExpressionNode::Bool(true) {
                cache_can_be_true.insert(exp_id, true);
                return true;
            }
        }
        cache_can_be_true.insert(exp_id, false);
        return false;
    }

    fn can_be_true(
        &self,
        expressions: &Vec<Vec<ExpressionNode>>,
        expression_ids: &Vec<Expression>,
        assignments: &Vec<FxHashSet<ExpressionNode>>,
        assignments_changes: &FxHashSet<usize>,
        cache_can_be_true: &mut FxHashMap<Expression, bool>,
    ) -> bool {
        for (i, exp) in expressions.iter().enumerate() {
            if !self.exp_can_be_true(
                exp,
                expression_ids[i].clone(),
                assignments,
                assignments_changes,
                cache_can_be_true,
            ) {
                return false;
            }
        }
        return true;
    }

    pub fn eval(&self, state: &State) -> PyResult<Option<f64>> {
        let mut internal_caching = self.internal_caching.lock().unwrap();
        if let Some(internal_caching) = internal_caching.as_mut() {
            let values: Vec<ExpressionNode> = state.assignments.iter().cloned().collect();
            let todo_values: Vec<usize> = self
                .actions
                .iter()
                .map(|action| state.todo.get(action).map(|(j, _)| *j).unwrap_or(0))
                .collect();
            let cache_key = CacheKey {
                values,
                todo_values,
            };
            if let Some(res) = internal_caching.get(&cache_key) {
                return Ok(res.clone());
            }

            let result = self._eval(state);
            internal_caching.insert(cache_key, result);
            Ok(result)
        } else {
            Ok(self._eval(state))
        }
    }

    fn _eval(&self, state: &State) -> Option<f64> {
        let mut assignments: Vec<FxHashSet<ExpressionNode>> =
            vec![FxHashSet::with_hasher(FxBuildHasher::default()); self.num_fluents];
        // add state assignments to assignments
        for (f, v) in state.assignments.iter().enumerate() {
            assignments[f] = FxHashSet::from_iter([v.clone()]);
        }
        // add extra fluents to assignments
        for action in &self.actions {
            let r = state.todo.get(action);
            let idx = match r {
                Some((j, _)) => j - 1,
                None => self.extra_fluents[action].len() - 1,
            };

            for (i, f) in self.extra_fluents[action].iter().enumerate() {
                if let ExpressionNode::Fluent(f) = &f[0] {
                    assignments[*f] = FxHashSet::from_iter([ExpressionNode::Bool(i == idx)]);
                }
            }
        }

        let mut cache_can_be_true: FxHashMap<Expression, bool> =
            FxHashMap::with_hasher(FxBuildHasher::default());
        let mut applied_operators = vec![false; self.operators.len()];
        let mut assignments_changes: FxHashSet<usize> = (0..self.num_fluents).collect();
        let mut depth = 0;
        while assignments_changes.len() > 0 {
            if self.can_be_true(
                &self.goals,
                &self.goal_expressions,
                &assignments,
                &assignments_changes,
                &mut cache_can_be_true,
            ) {
                // goal satisfied
                return Some(depth as f64);
            }

            let mut new_assignments: FxHashMap<usize, FxHashSet<ExpressionNode>> =
                FxHashMap::with_hasher(FxBuildHasher::default());
            for (i, operator) in self.operators.iter().enumerate() {
                if applied_operators[i] {
                    // operator already applied
                    let eff_fluents: FxHashSet<usize> =
                        self.operator_effects_fluents[i].iter().copied().collect();
                    if assignments_changes.is_disjoint(&eff_fluents) {
                        // no changes in the effect fluents
                        continue;
                    }
                } else if assignments_changes.is_disjoint(
                    &self.operator_conditions_fluents[i]
                        .iter()
                        .copied()
                        .collect(),
                ) {
                    // operator never applied, but no changes in the condition fluents
                    continue;
                } else if !self.can_be_true(
                    &operator.conditions,
                    &operator.condition_expressions,
                    &assignments,
                    &assignments_changes,
                    &mut cache_can_be_true,
                ) {
                    // operator cannot be applied
                    continue;
                } else {
                    // first time applied
                    applied_operators[i] = true;
                }

                for effect in &operator.effects {
                    let exp_fluents = self.extract_fluents(&effect.value);
                    let possible_values =
                        self.possible_values(&effect.value, &assignments, &exp_fluents);
                    new_assignments
                        .entry(effect.fluent)
                        .or_insert_with(|| FxHashSet::with_hasher(FxBuildHasher::default()))
                        .extend(possible_values);
                }
            }

            // update assignments
            assignments_changes.clear();
            for (fluent, new_vv) in new_assignments {
                let prev_len = assignments[fluent].len();
                for v in new_vv {
                    assignments[fluent].insert(v);
                }
                if assignments[fluent].len() > prev_len {
                    assignments_changes.insert(fluent);
                }
            }

            depth += 1;
        }

        None
    }

    pub fn name(&self) -> String {
        String::from("hmax_explicit")
    }
}
