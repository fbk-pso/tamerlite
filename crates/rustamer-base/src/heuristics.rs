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

use im::Vector;
use num::{BigInt, BigRational, Zero};
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::hash::{Hash, Hasher};
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use std::vec::Vec;

use rustc_hash::{FxBuildHasher, FxHashMap, FxHashSet};

use pyo3::exceptions::{PyException, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use super::expressions::*;
use super::expressions_utils::*;
use super::interpreted_functions::*;
use super::multiqueue::StateContainer;
use super::search_space::SearchSpaceTrait;
use super::search_state::State;
use super::structures::*;
use super::utils::*;

type EvalGenItem = PyResult<(Rc<State>, Option<f64>)>;
type EvalGenOwnedItem = PyResult<(State, Option<f64>)>;
type EvalGenContainerItem = PyResult<(usize, Option<f64>)>;
type HeuristicCache = Arc<Mutex<Option<FxHashMap<CacheKey, Option<f64>>>>>;

pub trait HeuristicTrait {
    fn eval<S: SearchSpaceTrait>(&self, state: &State, ss: &S) -> PyResult<Option<f64>>;

    /// Evaluates the heuristic for a given state, returning an iterator over the results.
    /// This method is used in non-multiqueue search algorithms
    fn eval_gen<'a, I, S: SearchSpaceTrait>(
        &'a self,
        states_iter: I,
        ss: &'a S,
    ) -> PyResult<Box<dyn Iterator<Item = EvalGenItem> + 'a>>
    where
        I: Iterator<Item = PyResult<Rc<State>>> + 'a,
    {
        Ok(Box::new(states_iter.map(|state| {
            let state = state?;
            let h_value = self.eval(&state, ss)?;
            Ok((state, h_value))
        })))
    }

    /// Evaluates the heuristic for a given state, returning an iterator over the results.
    /// This method is used in non-multiqueue search algorithms
    fn eval_gen_owned<'a, I, S: SearchSpaceTrait>(
        &'a self,
        states_iter: I,
        ss: &'a S,
    ) -> PyResult<Box<dyn Iterator<Item = EvalGenOwnedItem> + 'a>>
    where
        I: Iterator<Item = PyResult<State>> + 'a,
    {
        Ok(Box::new(states_iter.map(|state| {
            let state = state?;
            let h_value = self.eval(&state, ss)?;
            Ok((state, h_value))
        })))
    }

    /// Evaluates the heuristic for a given state, returning an iterator over the results.
    /// This method is used in multiqueue search algorithms
    fn eval_gen_container<'a, S: SearchSpaceTrait>(
        &'a self,
        states: &'a [StateContainer],
        ss: &'a S,
    ) -> PyResult<Box<dyn Iterator<Item = EvalGenContainerItem> + 'a>> {
        Ok(Box::new(states.iter().enumerate().map(|(i, sc)| {
            let h_value = self.eval(&sc.state, ss)?;
            Ok((i, h_value))
        })))
    }
}

#[derive(Clone, Debug)]
pub enum HeuristicKind {
    HFF,
    HADD,
    HMAX,
}

/// The set of values one fluent can hold, as the heuristics need it.
///
/// Mirrors `FluentDomain` in `src/tamerlite/core/search_space.py`, which is
/// where the rationale lives: `Encoder` used to hand the heuristics a type
/// *name* plus a name-keyed object map, and the two cores had to re-derive
/// this classification from that name -- which they cannot do
/// unambiguously, since builtin and user type names share one namespace.
/// `Encoder` now decides it once, where it still has the UP type.
///
/// `Objects` carries the fluent's own domain, which may legally be empty
/// (a user type with no objects); the variant, not the emptiness, is what
/// identifies an object-typed fluent.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FluentDomain {
    Bool,
    Int,
    Real,
    Objects(Vec<Object>),
}

/// Wire tag for `FluentDomain`'s kind. Not a `#[pyclass]`: `FluentDomain`
/// (`src/tamerlite/core/search_space.py`) is shared, backend-agnostic data --
/// `Encoder` builds it once and hands it to whichever backend is active --
/// and its own `kind`-identity checks (`__post_init__`, `_object_domain`)
/// always compare against *that module's* `FluentKind`. Swapping this enum
/// in for Python's `FluentKind` the way `IfReturnType` is swapped (see
/// `crates/rustamer-base/src/interpreted_functions.rs`) would make those
/// checks fail for any `FluentDomain` this crate builds, since a Rust-side
/// value would never be identical to `search_space.py`'s own enum member.
/// `IfReturnType` avoids this because its values and comparisons never cross
/// the backend boundary as shared data; `FluentDomain` is exactly that
/// shared data. Values must therefore match `FluentKind`
/// (`src/tamerlite/core/search_space.py`) by discriminant, not by identity --
/// see `extract_fluent_kind` below.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FluentKind {
    Bool = 0,
    Int = 1,
    Real = 2,
    Object = 3,
}

/// Decodes the `kind` a `FluentDomain` (`src/tamerlite/core/search_space.py`)
/// carries from Python's `FluentKind` (an `IntEnum`, so it coerces to `u8`
/// with no conversion on Python's side). The discriminants above must match
/// `search_space.py`'s `FluentKind` member values exactly.
fn extract_fluent_kind(obj: &Bound<'_, PyAny>) -> PyResult<FluentKind> {
    match obj.extract::<u8>()? {
        0 => Ok(FluentKind::Bool),
        1 => Ok(FluentKind::Int),
        2 => Ok(FluentKind::Real),
        3 => Ok(FluentKind::Object),
        k => Err(PyValueError::new_err(format!(
            "unknown FluentKind discriminant: {k}"
        ))),
    }
}

/// Extracts one `FluentDomain` (`src/tamerlite/core/search_space.py`) from
/// its Python side: reads `kind` and, only for the `Object` variant,
/// `objects`.
fn extract_fluent_domain(obj: &Bound<'_, PyAny>) -> PyResult<FluentDomain> {
    Ok(match extract_fluent_kind(&obj.getattr("kind")?)? {
        FluentKind::Bool => FluentDomain::Bool,
        FluentKind::Int => FluentDomain::Int,
        FluentKind::Real => FluentDomain::Real,
        FluentKind::Object => FluentDomain::Objects(obj.getattr("objects")?.extract()?),
    })
}

/// `#[pyo3(from_py_with = ...)]` target for a `fluent_domains: Vec<FluentDomain>`
/// parameter: extracts each element of the Python list via
/// `extract_fluent_domain`.
pub fn extract_fluent_domains(obj: &Bound<'_, PyAny>) -> PyResult<Vec<FluentDomain>> {
    obj.try_iter()?
        .map(|item| extract_fluent_domain(&item?))
        .collect()
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

    pub fn name(&self) -> &'static str {
        "custom"
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
    effects: Vec<Expression>,
    constant_increase_effects: FxHashMap<Fluent, f64>,
    constant_assign_effects: FxHashMap<Fluent, f64>,
    complex_numeric_effects: FxHashMap<Fluent, Expression>,
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
    effect_fluents: Vec<Vec<Fluent>>,
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
    values: Vector<ExpressionNode>,
    todo_values: Vec<usize>,
}

fn get_event_conditions(
    event: &Event,
    expression_manager: &mut ExpressionManager,
) -> PyResult<Vec<Vec<ExpressionNode>>> {
    let mut conditions_set = FxHashSet::with_hasher(FxBuildHasher);
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
    fluent_domains: &[FluentDomain],
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
        fluent_domains,
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
    expr: &[ExpressionNode],
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
                expression_manager.put(&extract_sub_expression(expr, idx)?),
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
/// Each leaf node in the condition is rewritten when possible, via
/// `simplify_leaf` -- see there for the rules and their order. Non-leaf
/// nodes, and leaf nodes no rule matches, are left unchanged.
///
/// # Arguments
///
/// * `condition` - The expression to simplify.
/// * `fluent_domains` - Each fluent's kind and, for object-typed ones,
///   the objects it can hold.
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
    fluent_domains: &[FluentDomain],
    disable_numeric_reasoning: bool,
    expression_manager: &mut ExpressionManager,
) -> Result<HeuristicExpression, ArithmeticError> {
    let mut new_condition = Vec::with_capacity(condition.expression.len());
    let mut contains_or_node = condition.contains_or_node;
    for node in &condition.expression {
        let simplified_expr = match node {
            HeuristicExpressionNode::Leaf(expr) => simplify_leaf(
                expr,
                fluent_domains,
                disable_numeric_reasoning,
                expression_manager,
            )?,
            _ => None,
        };
        if let Some(mut simplified_expr) = simplified_expr {
            contains_or_node |= simplified_expr.contains_or_node;
            new_condition.append(&mut simplified_expr.expression);
        } else {
            new_condition.push(node.clone());
        }
    }

    Ok(HeuristicExpression {
        expression: new_condition,
        contains_or_node,
    })
}

/// Try each leaf-rewrite rule in turn; the first one whose shape matches
/// `expr` wins. Mirrors the Python core's `_simplify_leaf` exactly -- see
/// there for the rules and their order.
///
/// - A leaf containing an interpreted-function call matches no rule -- the
///   callable is opaque, evaluated at search time.
/// - A numeric leaf (equality/`<=`/`<` over a linear expression, or its
///   negation) is simplified, unless numeric reasoning is disabled, in
///   which case it's left as-is. Either way, no other rule is tried: this
///   is what keeps `simplify_object_equality` below from ever firing on a
///   numeric `n1 == n2` leaf, since a bare `Equals` node can't otherwise be
///   told apart from object equality (see `is_object_typed`).
/// - Otherwise, a `fluent != object` expression is rewritten into a
///   disjunction of equalities.
/// - Otherwise, an object-equality expression between two fluents
///   (`fluent1 == fluent2`, `not(fluent1 == fluent2)`) is rewritten into an
///   equivalent disjunction of `fluent == object` facts.
///
/// Returns `Ok(None)` if no rule matches (`expr` should be kept as-is).
fn simplify_leaf(
    expr: &Expression,
    fluent_domains: &[FluentDomain],
    disable_numeric_reasoning: bool,
    expression_manager: &mut ExpressionManager,
) -> Result<Option<HeuristicExpression>, ArithmeticError> {
    let expr_nodes = expression_manager.force_get(expr);
    if has_interpreted_function(expr_nodes) {
        return Ok(None);
    }

    if is_numeric_leaf_expression(expr_nodes, fluent_domains) {
        return if disable_numeric_reasoning {
            Ok(None)
        } else {
            simplify_numeric_leaf_node(expr, expression_manager)
        };
    }

    if let Some(result) =
        simplify_fluent_not_equals_object_expression(expr, fluent_domains, expression_manager)
    {
        return Ok(Some(result));
    }
    Ok(simplify_object_equality(
        expr,
        fluent_domains,
        expression_manager,
    ))
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
                if let Some(last) = expr1.last_mut() {
                    *last = ExpressionNode::LE(*op1, *op2);
                }
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
            if let Some(HeuristicExpressionNode::Leaf(expr)) = &new_expr.expression.first() {
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
    expr: &[ExpressionNode],
    op1: usize,
    op2: usize,
    expression_node_type: F,
    expression_manager: &mut ExpressionManager,
) -> Result<Expression, ArithmeticError>
where
    F: FnOnce(usize, usize) -> ExpressionNode,
{
    let (mut op1_expr, mut op2_expr) = inverted_operands(expr, op1, op2)?;
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
    expr: &[ExpressionNode],
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
/// * `fluent_domains` - Each fluent's kind and, for object-typed ones,
///   the objects it can hold.
/// * `expression_manager` - A mutable reference to the `ExpressionManager`.
///
/// # Returns
///
/// Returns `Some(HeuristicExpression)` representing the disjunction of equality
/// expressions if simplification is possible, or `None` if the expression is not
/// of the form `fluent != object`.
fn simplify_fluent_not_equals_object_expression(
    expr: &Expression,
    fluent_domains: &[FluentDomain],
    expression_manager: &mut ExpressionManager,
) -> Option<HeuristicExpression> {
    let expr = expression_manager.force_get(expr).clone();
    let [ExpressionNode::Fluent(f), ExpressionNode::Object(o), ExpressionNode::Equals(_, _), ExpressionNode::Not(_)] =
        expr.as_slice()
    else {
        return None;
    };

    // `o` is a literal object, and UP's `Equals` requires type-compatible
    // operands, so `f` must be object-typed too.
    let objs = object_domain(*f, fluent_domains)
        .expect("fluent compared to an object must be object-typed");

    let mut nodes: Vec<_> = objs
        .iter()
        .filter(|obj| *obj != o)
        .map(|obj| {
            let leaf_expr = expression_manager.put(&vec![
                ExpressionNode::Fluent(*f),
                ExpressionNode::Object(*obj),
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

/// Simplifies an equality (or its negation) between two object-typed
/// fluents.
///
/// The delete relaxation's cost table only ever holds `fluent == object`
/// facts (seeded from the state and achieved by operator effects, see
/// `DeleteRelaxationHeuristic::_eval`), so a leaf comparing two fluents to
/// each other has nothing to match against and would otherwise dead-end
/// every state that needs it. Both polarities are expanded exactly:
///
/// `fluent1 == fluent2` into
///     `(fluent1 == o and fluent2 == o) or ...`
/// for `o` ranging over the objects both fluents can hold (the intersection
/// of their domains -- hierarchical types mean the two fluents can be
/// declared at different type names while still sharing objects).
///
/// `not(fluent1 == fluent2)` into
///     `(fluent1 == o1 and fluent2 == o2) or ...`
/// for every ordered pair `(o1, o2)` with `o1 != o2`, one from each fluent's
/// domain.
///
/// # Arguments
///
/// * `expr` - The expression to simplify.
/// * `fluent_domains` - Each fluent's kind and, for object-typed ones,
///   the objects it can hold.
/// * `expression_manager` - A mutable reference to the `ExpressionManager`.
///
/// # Returns
///
/// Returns `Some(HeuristicExpression)` representing the expanded disjunction
/// if simplification is possible, or `None` if the expression is not of the
/// form `fluent1 == fluent2` or its negation.
fn simplify_object_equality(
    expr: &Expression,
    fluent_domains: &[FluentDomain],
    expression_manager: &mut ExpressionManager,
) -> Option<HeuristicExpression> {
    let (f1, f2, positive) = match expression_manager.force_get(expr).as_slice() {
        [ExpressionNode::Fluent(f1), ExpressionNode::Fluent(f2), ExpressionNode::Equals(0, 1)] => {
            (*f1, *f2, true)
        }
        [ExpressionNode::Fluent(f1), ExpressionNode::Fluent(f2), ExpressionNode::Equals(0, 1), ExpressionNode::Not(2)] => {
            (*f1, *f2, false)
        }
        _ => return None,
    };

    let objs1 = object_domain(f1, fluent_domains)?;
    let objs2 = object_domain(f2, fluent_domains)?;

    let mut nodes: Vec<HeuristicExpressionNode> = Vec::new();
    let push_conjunct = |o1: Object, o2: Object, expression_manager: &mut ExpressionManager| {
        let leaf1 = expression_manager.put(&vec![
            ExpressionNode::Fluent(f1),
            ExpressionNode::Object(o1),
            ExpressionNode::Equals(0, 1),
        ]);
        let leaf2 = expression_manager.put(&vec![
            ExpressionNode::Fluent(f2),
            ExpressionNode::Object(o2),
            ExpressionNode::Equals(0, 1),
        ]);
        (leaf1, leaf2)
    };

    if positive {
        let objs2_set: FxHashSet<Object> = objs2.iter().copied().collect();
        for &o in objs1.iter() {
            if !objs2_set.contains(&o) {
                continue;
            }
            let (leaf1, leaf2) = push_conjunct(o, o, expression_manager);
            nodes.push(HeuristicExpressionNode::Leaf(leaf1));
            nodes.push(HeuristicExpressionNode::Leaf(leaf2));
            nodes.push(HeuristicExpressionNode::And(2));
        }
    } else {
        for &o1 in objs1.iter() {
            for &o2 in objs2.iter() {
                if o1 == o2 {
                    continue;
                }
                let (leaf1, leaf2) = push_conjunct(o1, o2, expression_manager);
                nodes.push(HeuristicExpressionNode::Leaf(leaf1));
                nodes.push(HeuristicExpressionNode::Leaf(leaf2));
                nodes.push(HeuristicExpressionNode::And(2));
            }
        }
    }

    let num_disjuncts = nodes.len() / 3;
    let res = if num_disjuncts == 0 {
        let false_expr = expression_manager.put(&vec![ExpressionNode::Bool(false)]);
        HeuristicExpression {
            expression: vec![HeuristicExpressionNode::Leaf(false_expr)],
            contains_or_node: false,
        }
    } else if num_disjuncts > 1 {
        nodes.push(HeuristicExpressionNode::Or(num_disjuncts));
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
    constant_increase_effects: &mut FxHashMap<Fluent, f64>,
    constant_assign_effects: &mut FxHashMap<Fluent, f64>,
    complex_numeric_effects: &mut FxHashMap<Fluent, Expression>,
) {
    if effect.value.len() == 1 {
        let v = match &effect.value[0] {
            ExpressionNode::Int(v) => Some(integer_to_f64(v)),
            ExpressionNode::Rational(v) => Some(rational_to_f64(v)),
            _ => None,
        };
        if let Some(v) = v {
            constant_assign_effects.insert(effect.fluent, v);
            return;
        }
    }

    let mut polynomial =
        to_linear_polynomial(&effect.value).unwrap_or(FxHashMap::with_hasher(FxBuildHasher));
    let k = polynomial.remove(&None).unwrap_or(0.0);
    if polynomial.len() == 1 && matches!(polynomial.get(&Some(effect.fluent)), Some(1.0)) {
        constant_increase_effects.insert(effect.fluent, k);
    } else {
        complex_numeric_effects.insert(effect.fluent, expression_manager.put(&effect.value));
    }
}

/// The objects a fluent can hold, or `None` if it isn't object-typed.
///
/// Single oracle for both questions the object-equality handling asks: "is
/// this operand object-typed?" (`is_object_typed`) and "what does it range
/// over?" (`simplify_object_equality`,
/// `simplify_fluent_not_equals_object_expression`). Those must agree
/// exactly, and must match Python's `_object_domain` just as exactly -- see
/// there.
///
/// # Arguments
///
/// * `fluent` - The fluent to resolve.
/// * `fluent_domains` - Each fluent's kind and, for object-typed ones,
///   the objects it can hold.
///
/// # Returns
///
/// Returns the fluent's object domain, or `None` if it is not object-typed.
fn object_domain(fluent: Fluent, fluent_domains: &[FluentDomain]) -> Option<&[Object]> {
    match fluent_domains.get(fluent.idx)? {
        FluentDomain::Objects(objs) => Some(objs),
        _ => None,
    }
}

/// Whether an `==` operand is object-typed rather than numeric.
///
/// `Equals` covers both numeric equality and user-type (object) equality --
/// there is no separate node kind for the two, so the operands' *types* are
/// the only thing that tells them apart. An operand is object-typed if it's
/// a literal object, or a fluent whose `FluentDomain` says so.
///
/// # Arguments
///
/// * `node` - One operand of an `Equals` leaf.
/// * `fluent_domains` - Each fluent's kind and, for object-typed ones,
///   the objects it can hold.
///
/// # Returns
///
/// Returns `true` if `node` is object-typed, `false` if numeric.
fn is_object_typed(node: &ExpressionNode, fluent_domains: &[FluentDomain]) -> bool {
    match node {
        ExpressionNode::Object(_) => true,
        ExpressionNode::Fluent(f) => object_domain(*f, fluent_domains).is_some(),
        _ => false,
    }
}

/// Determine if a leaf expression represents a numeric expression.
/// A leaf expression is assumed to contain no `AND` or `OR` nodes.
///
/// # Arguments
///
/// * `expr` - A reference to the leaf expression (`Vec<ExpressionNode>`) to check.
/// * `fluent_domains` - Each fluent's kind and, for object-typed ones,
///   the objects it can hold.
///
/// # Returns
///
/// Returns `true` if the leaf expression is numeric, `false` otherwise.
fn is_numeric_leaf_expression(expr: &[ExpressionNode], fluent_domains: &[FluentDomain]) -> bool {
    let idx = match expr.last() {
        Some(ExpressionNode::Not(op)) => *op,
        _ => expr.len() - 1,
    };
    match expr[idx] {
        ExpressionNode::Equals(op1, op2) => {
            !is_object_typed(&expr[op1], fluent_domains)
                && !is_object_typed(&expr[op2], fluent_domains)
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
    simple_numeric_conds: &mut FxHashMap<Expression, (Vec<Fluent>, Vec<f64>)>,
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
) -> Option<(Vec<Fluent>, Vec<f64>, bool)> {
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
/// Returns `Some(FxHashMap<Option<Fluent>, f64>)` mapping fluents to coefficients,
/// with `None` representing the constant term. Returns `None` if the expression is non-linear.
fn to_linear_polynomial(expr: &Vec<ExpressionNode>) -> Option<FxHashMap<Option<Fluent>, f64>> {
    let zero = integer_to_rational(BigInt::from(0));
    let one = integer_to_rational(BigInt::from(1));
    let mut res = Vec::new();
    for node in expr {
        match node {
            ExpressionNode::Int(v) => {
                res.push(constant_polynomial(integer_to_rational(*v.clone())));
            }
            ExpressionNode::Rational(v) => {
                res.push(constant_polynomial(*v.clone()));
            }
            ExpressionNode::Fluent(f) => {
                let mut p = FxHashMap::with_hasher(FxBuildHasher);
                p.insert(Some(*f), one.clone());
                res.push(p);
            }
            ExpressionNode::Minus(_, _) => {
                let p2 = res.pop().unwrap();
                let p1 = res.last_mut().unwrap();
                for (f, w) in p2 {
                    *p1.entry(f).or_insert(zero.clone()) -= w;
                }
                simplify_polynomial(p1)
            }
            ExpressionNode::Plus(operands) => {
                let mut p = res.pop().unwrap();
                for _ in 1..operands.len() {
                    for (f, w) in res.pop().unwrap() {
                        *p.entry(f).or_insert(zero.clone()) += w;
                    }
                }
                simplify_polynomial(&mut p);
                res.push(p);
            }
            ExpressionNode::Div(_, _) => {
                let divisor = res.pop().unwrap();
                let dividend = res.last_mut().unwrap();
                if !is_constant_polynomial(&divisor) {
                    return None;
                }
                let divisor = divisor.get(&None).unwrap();
                if divisor.is_zero() {
                    return None;
                }
                for value in dividend.values_mut() {
                    *value /= divisor;
                }
            }
            ExpressionNode::Times(operands) => {
                let mut const_multiplier = one.clone();
                let mut polynomial = None;
                for _ in 0..operands.len() {
                    let operand = res.pop().unwrap();
                    if is_constant_polynomial(&operand) {
                        const_multiplier *= operand.get(&None).unwrap();
                    } else if polynomial.is_some() {
                        return None;
                    } else {
                        polynomial = Some(operand);
                    }
                }

                res.push(match polynomial {
                    Some(mut polynomial) if !const_multiplier.is_zero() => {
                        for w in polynomial.values_mut() {
                            *w *= const_multiplier.clone();
                        }
                        polynomial
                    }
                    _ => constant_polynomial(const_multiplier),
                });
            }
            _ => return None,
        }
    }

    res.pop()
        .unwrap()
        .into_iter()
        .map(|(f, v)| (f, rational_to_f64(&v)))
        .collect::<FxHashMap<Option<Fluent>, f64>>()
        .into()
}

fn constant_polynomial(v: BigRational) -> FxHashMap<Option<Fluent>, BigRational> {
    let mut p = FxHashMap::with_hasher(FxBuildHasher);
    p.insert(None, v);
    p
}

fn is_constant_polynomial(polynomial: &FxHashMap<Option<Fluent>, BigRational>) -> bool {
    polynomial.len() == 1 && polynomial.contains_key(&None)
}

/// Simplifies a polynomial by removing zero-coefficient terms.
///
/// This function iterates over all terms in the polynomial and removes any entry
/// whose coefficient is zero, with the exception of the constant term (`None`),
/// which is always retained.
///
/// # Arguments
///
/// * `polynomial` - A mutable reference to a polynomial represented as a map of
///   fluents to coefficients.
fn simplify_polynomial(polynomial: &mut FxHashMap<Option<Fluent>, BigRational>) {
    polynomial.retain(|key, value| !value.is_zero() || key.is_none());
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
    fluents: &[Fluent],
    weights: &Vec<f64>,
    max_net_effect: &mut f64,
    inadmissible_numeric_heuristic_variant: bool,
) -> bool {
    let mut net_effect = 0.0;
    for (f, w) in fluents.iter().zip(weights) {
        if !inadmissible_numeric_heuristic_variant
            && (operator.constant_assign_effects.contains_key(f)
                || operator.complex_numeric_effects.contains_key(f))
        {
            return true;
        }
        if let Some(k) = operator.constant_increase_effects.get(f) {
            net_effect += w * k;
        } else if inadmissible_numeric_heuristic_variant
            && (operator.constant_assign_effects.contains_key(f)
                || operator.complex_numeric_effects.contains_key(f))
        {
            net_effect -= 1.0;
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
/// * `inadmissible_numeric_heuristic_variant` - Whether to approximate constant
///   or non-linear numeric effects as contributing to the operator's net effect.
///
/// # Returns
///
/// Returns `Ok(Some(value))` with the minimum number of applications needed to satisfy
/// the condition, `Ok(Some(0.0))` if already satisfied, `Ok(Some(1.0))` if a constant/complex
/// effect applies, or `Ok(None)` if the condition cannot be satisfied.
fn repetitions(
    operator: &Operator,
    fluents: &Vec<Fluent>,
    weights: &Vec<f64>,
    state: &State,
    inadmissible_numeric_heuristic_variant: bool,
) -> PyResult<Option<f64>> {
    let mut v = *weights.last().unwrap();
    for (f, w) in fluents.iter().zip(weights) {
        let f_value = expression_node_to_f64(state.get_value(*f))?;
        v += *w * f_value;
    }

    if v <= 0.0 {
        // condition satisfied in state
        return Ok(Some(0.0));
    }

    if !inadmissible_numeric_heuristic_variant {
        for f in fluents {
            if operator.constant_assign_effects.contains_key(f)
                || operator.complex_numeric_effects.contains_key(f)
            {
                return Ok(Some(1.0));
            }
        }
    }

    let mut net_effect = 0.0;
    for (f, w) in fluents.iter().zip(weights) {
        if let Some(k) = operator.constant_increase_effects.get(f) {
            net_effect += w * k;
        } else if inadmissible_numeric_heuristic_variant
            && (operator.constant_assign_effects.contains_key(f)
                || operator.complex_numeric_effects.contains_key(f))
        {
            net_effect -= 1.0;
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
    expr: &[ExpressionNode],
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
            ExpressionNode::InterpretedFunction { operands, .. } if !operands.is_empty() => {
                operands[0]
            }
            _ => break,
        };
    }

    shift_expression(&expr[i..(idx + 1)], i, true)
}

/// A `HeuristicExpression` node with its leaf replaced by a dense condition
/// id ("cid"), so evaluating the expression indexes a `Vec<f64>` instead of
/// hashing an `Expression`.
#[derive(Clone, Copy, Debug)]
enum IndexedHeuristicExpressionNode {
    And(usize),
    Or(usize),
    Leaf(u32),
}

#[derive(Clone, Debug)]
struct IndexedHeuristicExpression {
    nodes: Vec<IndexedHeuristicExpressionNode>,
    contains_or_node: bool,
}

/// Assigns every leaf condition read by some operator or goal a dense cid,
/// and records which operators read it (`cond_to_ops`).
struct CondInterner {
    ids: FxHashMap<Expression, u32>,
    leaves: Vec<Expression>,
    cond_to_ops: Vec<Vec<usize>>,
}

impl CondInterner {
    fn new() -> Self {
        CondInterner {
            ids: FxHashMap::with_hasher(FxBuildHasher),
            leaves: Vec::new(),
            cond_to_ops: Vec::new(),
        }
    }

    fn intern(&mut self, e: Expression) -> u32 {
        *self.ids.entry(e).or_insert_with(|| {
            self.leaves.push(e);
            self.cond_to_ops.push(Vec::new());
            (self.leaves.len() - 1) as u32
        })
    }

    /// Converts `expr` into a `IndexedHeuristicExpression`, registering `owner` (if any) as a
    /// reader of each of its leaves.
    fn index_expression(
        &mut self,
        expr: &HeuristicExpression,
        owner: Option<usize>,
    ) -> IndexedHeuristicExpression {
        let nodes = expr
            .expression
            .iter()
            .map(|n| match n {
                HeuristicExpressionNode::Leaf(e) => {
                    let cid = self.intern(*e);
                    if let Some(op) = owner {
                        self.add_reader(cid, op);
                    }
                    IndexedHeuristicExpressionNode::Leaf(cid)
                }
                HeuristicExpressionNode::And(n) => IndexedHeuristicExpressionNode::And(*n),
                HeuristicExpressionNode::Or(n) => IndexedHeuristicExpressionNode::Or(*n),
            })
            .collect();
        IndexedHeuristicExpression {
            nodes,
            contains_or_node: expr.contains_or_node,
        }
    }

    fn add_reader(&mut self, cid: u32, op: usize) {
        let readers = &mut self.cond_to_ops[cid as usize];
        // A leaf repeated within one expression registers its reader once:
        // all of an expression's leaves are registered consecutively.
        if readers.last() != Some(&op) {
            readers.push(op);
        }
    }
}

/// The conjunction of `parts`, skipping empty ones.
fn conjunction(parts: &[&IndexedHeuristicExpression]) -> IndexedHeuristicExpression {
    let mut nodes = Vec::with_capacity(parts.iter().map(|p| p.nodes.len()).sum::<usize>() + 1);
    let mut num_operands = 0;
    for p in parts.iter().filter(|p| !p.nodes.is_empty()) {
        nodes.extend_from_slice(&p.nodes);
        num_operands += 1;
    }
    if num_operands > 1 {
        nodes.push(IndexedHeuristicExpressionNode::And(num_operands));
    }
    IndexedHeuristicExpression {
        nodes,
        contains_or_node: parts.iter().any(|p| p.contains_or_node),
    }
}

/// Cost of `expr` given the cost of each cid: AND is the sum (`max` for
/// hmax) and OR the minimum of its operands. `f64::INFINITY` means
/// unreached and propagates through both. `stack` is scratch space.
fn expression_cost(
    expr: &IndexedHeuristicExpression,
    costs: &[f64],
    hmax: bool,
    stack: &mut Vec<f64>,
) -> f64 {
    match expr.nodes.last() {
        None => return 0.0,
        Some(IndexedHeuristicExpressionNode::Leaf(cid)) => return costs[*cid as usize],
        _ => {}
    }

    stack.clear();
    for node in &expr.nodes {
        match *node {
            IndexedHeuristicExpressionNode::Leaf(cid) => stack.push(costs[cid as usize]),
            IndexedHeuristicExpressionNode::And(n) => {
                let mut r = 0.0;
                for _ in 0..n {
                    let v = stack.pop().unwrap();
                    if hmax {
                        r = f64::max(r, v);
                    } else {
                        r += v;
                    }
                }
                stack.push(r);
            }
            IndexedHeuristicExpressionNode::Or(n) => {
                let mut r = f64::INFINITY;
                for _ in 0..n {
                    let v = stack.pop().unwrap();
                    if v < r {
                        r = v;
                    }
                }
                stack.push(r);
            }
        }
    }
    debug_assert_eq!(stack.len(), 1);
    stack[0]
}

/// Collects into `out` the cids supporting the hff cost of `expr`, i.e. the
/// subgoals relaxed-plan extraction backchains on: every operand of an AND,
/// and only the cheapest operand of an OR (the first one popped, on ties).
fn supporting_conditions(expr: &IndexedHeuristicExpression, costs: &[f64], out: &mut Vec<u32>) {
    if !expr.contains_or_node {
        out.extend(expr.nodes.iter().filter_map(|n| match n {
            IndexedHeuristicExpressionNode::Leaf(cid) => Some(*cid),
            _ => None,
        }));
        return;
    }

    let mut stack: Vec<(f64, Vec<u32>)> = Vec::new();
    for node in &expr.nodes {
        match *node {
            IndexedHeuristicExpressionNode::Leaf(cid) => {
                stack.push((costs[cid as usize], vec![cid]))
            }
            IndexedHeuristicExpressionNode::And(n) => {
                let mut r = 0.0;
                let mut l = Vec::new();
                for _ in 0..n {
                    let (v, ol) = stack.pop().unwrap();
                    r += v;
                    l.extend(ol);
                }
                stack.push((r, l));
            }
            IndexedHeuristicExpressionNode::Or(n) => {
                let mut r = f64::INFINITY;
                let mut ml = Vec::new();
                for _ in 0..n {
                    let (v, ol) = stack.pop().unwrap();
                    if v < r {
                        r = v;
                        ml = ol;
                    }
                }
                stack.push((r, ml));
            }
        }
    }
    debug_assert_eq!(stack.len(), 1);
    out.extend(stack.pop().unwrap().1);
}

/// Entry of the Dijkstra fix-point's heap: an operator and its precondition
/// cost, packed into one integer key -- the cost's bits in the high 64, the
/// operator id in the low 64 -- so a heap comparison is a single integer
/// comparison. Pushed costs are always finite and non-negative, and for those
/// IEEE-754 bit patterns order exactly like the values (a negative zero is
/// normalized to `+0.0`), so the key orders entries by `(cost, op)`: a total
/// order, so the pop sequence depends only on the entries pushed -- the
/// Python core's `heapq` of `(cost, op)` tuples pops in exactly the same
/// order. `BinaryHeap` is a max-heap, so entries are wrapped in `Reverse`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct QueuedOperator(u128);

impl QueuedOperator {
    fn new(cost: f64, op: usize) -> Self {
        debug_assert!(cost.is_finite() && cost >= 0.0);
        QueuedOperator((u128::from((cost + 0.0).to_bits()) << 64) | op as u128)
    }

    fn cost(self) -> f64 {
        f64::from_bits((self.0 >> 64) as u64)
    }

    fn op(self) -> usize {
        self.0 as usize
    }
}

/// `(cid, fluents, weights)` of a simple numeric condition an operator
/// achieves, `fluents`/`weights` as in `repetitions`.
type NumericAchievement = (u32, Vec<Fluent>, Vec<f64>);

/// Per-evaluation buffers of the Dijkstra fix-point, reused across
/// evaluations to avoid reallocating them.
#[derive(Debug, Default)]
struct FixpointBuffers {
    cond_cost: Vec<f64>,
    op_cost: Vec<f64>,
    closed: Vec<bool>,
    /// hmax only: per cid, the minimum precondition cost among the
    /// operators expanded so far that achieve it numerically.
    min_achiever_pre_cost: Vec<f64>,
    /// hff only: per cid, the operator that achieved its current cost.
    reached_by: Vec<Option<usize>>,
    /// Per operator, how many distinct leaves of its precondition are still
    /// unreached. Reset from `distinct_leaves` by `_eval`.
    unreached: Vec<u32>,
    heap: BinaryHeap<Reverse<QueuedOperator>>,
    stack: Vec<f64>,
}

impl FixpointBuffers {
    fn reset(&mut self, n_conds: usize, n_ops: usize) {
        self.cond_cost.clear();
        self.cond_cost.resize(n_conds, f64::INFINITY);
        self.op_cost.clear();
        self.op_cost.resize(n_ops, f64::INFINITY);
        self.closed.clear();
        self.closed.resize(n_ops, false);
        self.min_achiever_pre_cost.clear();
        self.min_achiever_pre_cost.resize(n_conds, f64::INFINITY);
        self.reached_by.clear();
        self.reached_by.resize(n_conds, None);
        self.heap.clear();
    }
}

#[derive(Clone, Debug)]
pub struct DeleteRelaxationHeuristicConfig {
    pub heuristic_kind: HeuristicKind,
    pub internal_caching: bool,
    pub inadmissible_numeric_heuristic_variant: bool,
    pub disable_numeric_reasoning: bool,
}

/// The delete-relaxation heuristics hmax, hadd and hff, computed by a
/// Dijkstra fix-point over the operators.
///
/// Every leaf condition read by an operator or by the goal gets a dense cid
/// at construction; `eval` initializes the cost of each cid from the state,
/// then pops operators from a min-heap in order of precondition cost,
/// expanding each one exactly once (no reopening) and updating the cost of
/// the conditions it achieves. The goal is a pseudo-operator `goal_op`, and
/// the search stops as soon as it is popped -- except for hmax when some
/// operator achieves a simple numeric condition (`drain_heap`). There, the
/// condition's cost (`rep * cost` plus the cheapest precondition cost among
/// the achievers expanded so far) can be lower than the cost of the operator
/// being expanded, so an operator popped after the goal can still lower the
/// goal's cost, and the heap is drained instead. Without such achievers every
/// relaxed cost exceeds its achiever's, costs pop in non-decreasing order, and
/// stopping at the goal is exact.
///
/// A precondition without OR nodes costs `inf` while any of its leaves is
/// unreached, so its cost is only computed once `unreached` (per operator,
/// counting distinct leaves) drops to 0; a precondition with an OR node is
/// recomputed whenever one of its leaves gets cheaper.
#[derive(Clone, Debug)]
pub struct DeleteRelaxationHeuristic {
    actions: Vec<Action>,
    events: FxHashMap<Action, usize>,
    operators: Vec<Operator>,
    /// Index of the goal pseudo-operator, `operators.len()`. Its
    /// precondition is the goal for hff, and the goal AND the bookkeeping
    /// goal of completing every started durative action for hadd/hmax.
    goal_op: usize,
    /// Precondition of every operator, plus `goal_op`'s.
    op_conditions: Vec<IndexedHeuristicExpression>,
    /// The goal alone: hff's relaxed plan is extracted from it.
    goals: IndexedHeuristicExpression,
    /// cid -> operators (including `goal_op`) whose precondition reads it.
    cond_to_ops: Vec<Vec<usize>>,
    /// Per operator (including `goal_op`), the number of distinct leaves of
    /// its precondition.
    distinct_leaves: Vec<u32>,
    /// Per operator, the cids of its effects that some precondition reads.
    op_effects: Vec<Vec<u32>>,
    /// Per operator, `(cid, fluents, weights)` of every simple numeric
    /// condition it achieves.
    op_numeric_conds: Vec<Vec<NumericAchievement>>,
    /// Whether `eval` drains the heap instead of stopping at `goal_op`: hmax
    /// with some operator achieving a simple numeric condition.
    drain_heap: bool,
    /// Conditions evaluated against the state: `(cid, condition, cost if
    /// false)`. Cost if true is 0.
    evaluated_conds: Vec<(u32, Vec<ExpressionNode>, f64)>,
    /// Conditions `[Fluent(f)]`: cost 0 iff `f` is true in the state.
    true_fluent_conds: Vec<(u32, Fluent)>,
    /// Conditions `[Fluent(f), Not(0)]`: cost 0 iff `f` is false in the state.
    false_fluent_conds: Vec<(u32, Fluent)>,
    /// Conditions `[Fluent(f), v, Equals(0, 1)]`: cost 0 iff `f` holds the
    /// non-bool value `v` in the state.
    equality_conds: Vec<(u32, Fluent, ExpressionNode)>,
    /// Per action with events, the cid of each of its bookkeeping fluents:
    /// the one matching the action's progress in `state.todo` has cost 0.
    extra_fluent_cids: Vec<(Action, Vec<u32>)>,
    buffers: Arc<Mutex<FixpointBuffers>>,
    heuristic_kind: HeuristicKind,
    internal_caching: HeuristicCache,
    inadmissible_numeric_heuristic_variant: bool,
    disable_numeric_reasoning: bool,
}

impl DeleteRelaxationHeuristic {
    pub fn new(
        actions: Vec<Action>,
        fluent_domains: Vec<FluentDomain>,
        events: FxHashMap<Action, Vec<(Timing, Event)>>,
        goals: Vec<PyExpressionNode>,
        config: DeleteRelaxationHeuristicConfig,
    ) -> PyResult<Self> {
        let mut operators = Vec::with_capacity(events.values().map(|e| e.len()).sum());
        // Precondition of each operator, by operator id. Only used here, to
        // build `op_conditions`.
        let mut operator_conditions: Vec<HeuristicExpression> =
            Vec::with_capacity(operators.capacity());
        let mut extra_fluents: Vec<(Action, Vec<Expression>)> = Vec::with_capacity(events.len());
        let mut extra_goals = Vec::with_capacity(events.len() + 1);
        let mut expression_manager = ExpressionManager::new();
        let n_real_fluents = fluent_domains.len();
        let mut num_fluents = fluent_domains.len();
        // Every bookkeeping fluent allocated below is a plain bool flag.
        // Giving them domains up front keeps `object_domain` a *total*
        // lookup, so no caller has to know where the real fluents end.
        // Mirrors `DeleteRelaxationHeuristic.__init__` in the Python core.
        let mut fluent_domains = fluent_domains;
        fluent_domains.resize(
            num_fluents
                + actions
                    .iter()
                    .filter_map(|a| events.get(a))
                    .map(|le| le.len())
                    .sum::<usize>(),
            FluentDomain::Bool,
        );
        let map_to_python_exception = |e| PyException::new_err(format!("{:?}", e));

        for a in &actions {
            let Some(le) = events.get(a) else {
                continue;
            };
            let mut a_extra_fluents: Vec<Expression> = Vec::new();
            let f_cond = Fluent::new(num_fluents + le.len() - 1);
            let mut cond = ExpressionNode::Fluent(f_cond);
            extra_goals.push(cond.clone());
            for (_, e) in le.iter() {
                let mut effects: Vec<Expression> = Vec::new();
                let mut constant_increase_effects: FxHashMap<Fluent, f64> =
                    FxHashMap::with_hasher(FxBuildHasher);
                let mut constant_assign_effects: FxHashMap<Fluent, f64> =
                    FxHashMap::with_hasher(FxBuildHasher);
                let mut complex_numeric_effects: FxHashMap<Fluent, Expression> =
                    FxHashMap::with_hasher(FxBuildHasher);
                let f = Fluent::new(num_fluents);
                num_fluents += 1;
                a_extra_fluents.push(expression_manager.put(&vec![ExpressionNode::Fluent(f)]));
                effects.push(expression_manager.put(&vec![ExpressionNode::Fluent(f)]));
                for eff in e.effects.iter() {
                    // An exhaustive match, unlike the `if t == "bool" / else
                    // if t == "real" || t == "int" / else` chain this replaced:
                    // there, the final `else` silently absorbed any type name
                    // it did not recognise, which is how a `UserType("int")`
                    // used to reach the numeric branch.
                    match &fluent_domains[eff.fluent.idx] {
                        FluentDomain::Bool => {
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
                                            make_operator("not", vec![0])?,
                                        ]));
                                    }
                                } else {
                                    effects.push(
                                        expression_manager
                                            .put(&vec![ExpressionNode::Fluent(eff.fluent)]),
                                    );
                                    effects.push(expression_manager.put(&vec![
                                        ExpressionNode::Fluent(eff.fluent),
                                        make_operator("not", vec![0])?,
                                    ]));
                                }
                            } else {
                                effects.push(
                                    expression_manager
                                        .put(&vec![ExpressionNode::Fluent(eff.fluent)]),
                                );
                                effects.push(expression_manager.put(&vec![
                                    ExpressionNode::Fluent(eff.fluent),
                                    make_operator("not", vec![0])?,
                                ]));
                            }
                        }
                        FluentDomain::Int | FluentDomain::Real => {
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
                        }
                        FluentDomain::Objects(objs) => {
                            if eff.value.len() == 1
                                && matches!(eff.value[0], ExpressionNode::Object(_))
                            {
                                effects.push(expression_manager.put(&vec![
                                    ExpressionNode::Fluent(eff.fluent),
                                    eff.value[0].clone(),
                                    make_operator("==", vec![0, 1])?,
                                ]));
                            } else {
                                for o in objs.iter() {
                                    effects.push(expression_manager.put(&vec![
                                        ExpressionNode::Fluent(eff.fluent),
                                        ExpressionNode::Object(*o),
                                        make_operator("==", vec![0, 1])?,
                                    ]));
                                }
                            }
                        }
                    }
                }

                if let Some(conditions) = build_operator_condition(
                    &get_event_conditions(e, &mut expression_manager)?,
                    cond.clone(),
                    &fluent_domains,
                    config.disable_numeric_reasoning,
                    &mut expression_manager,
                )
                .map_err(map_to_python_exception)?
                {
                    operators.push(Operator {
                        id: OperatorID::new(operators.len()),
                        action: *a,
                        effects,
                        constant_increase_effects,
                        constant_assign_effects,
                        complex_numeric_effects,
                        cost: 1.0,
                    });
                    operator_conditions.push(conditions);
                }
                cond = ExpressionNode::Fluent(f);
            }
            extra_fluents.push((*a, a_extra_fluents));
        }
        debug_assert!(operators.iter().enumerate().all(|(i, o)| o.id.id == i));

        let expr_goals = goals.into_iter().map(|e| e.v).collect::<Vec<_>>();
        let goals = convert_to_heuristic_expression(&expr_goals, &mut expression_manager)
            .map_err(map_to_python_exception)?;
        let goals = simplify_condition(
            &goals,
            &fluent_domains,
            config.disable_numeric_reasoning,
            &mut expression_manager,
        )
        .map_err(map_to_python_exception)?;
        extra_goals.push(ExpressionNode::And((0..extra_goals.len()).collect()));
        let extra_goals = convert_to_heuristic_expression(&extra_goals, &mut expression_manager)
            .map_err(map_to_python_exception)?;

        let mut simple_numeric_conds: FxHashMap<Expression, (Vec<Fluent>, Vec<f64>)> =
            FxHashMap::with_hasher(FxBuildHasher);
        let mut lt_simple_numeric_conds: FxHashSet<Expression> =
            FxHashSet::with_hasher(FxBuildHasher);
        let mut complex_numeric_conds: FxHashSet<Expression> =
            FxHashSet::with_hasher(FxBuildHasher);
        let mut if_conds: FxHashSet<Expression> = FxHashSet::with_hasher(FxBuildHasher);
        for conditions in &operator_conditions {
            for node in &conditions.expression {
                if let HeuristicExpressionNode::Leaf(e) = node {
                    let expr = expression_manager.force_get(e);
                    if has_interpreted_function(expr) {
                        if_conds.insert(*e);
                    } else if is_numeric_leaf_expression(expr, &fluent_domains) {
                        update_numeric_conditions(
                            e,
                            &expression_manager,
                            &mut simple_numeric_conds,
                            &mut lt_simple_numeric_conds,
                            &mut complex_numeric_conds,
                            config.disable_numeric_reasoning,
                        );
                    }
                }
            }
        }

        for node in goals.expression.iter() {
            if let HeuristicExpressionNode::Leaf(e) = node {
                let expr = expression_manager.force_get(e);
                if has_interpreted_function(expr) {
                    if_conds.insert(*e);
                } else if is_numeric_leaf_expression(expr, &fluent_domains) {
                    update_numeric_conditions(
                        e,
                        &expression_manager,
                        &mut simple_numeric_conds,
                        &mut lt_simple_numeric_conds,
                        &mut complex_numeric_conds,
                        config.disable_numeric_reasoning,
                    );
                }
            }
        }

        let mut max_net_effect = f64::MIN;
        let mut achieved_simple_numeric_conds: Vec<Vec<Expression>> =
            vec![Vec::new(); operators.len()];
        for o in &operators {
            for (c, (fluents, weights)) in &simple_numeric_conds {
                if achieves(
                    o,
                    fluents,
                    weights,
                    &mut max_net_effect,
                    config.inadmissible_numeric_heuristic_variant,
                ) {
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

        let mut interner = CondInterner::new();
        let mut op_conditions: Vec<IndexedHeuristicExpression> = operator_conditions
            .iter()
            .enumerate()
            .map(|(op, conditions)| interner.index_expression(conditions, Some(op)))
            .collect();
        let goal_op = operators.len();
        let goals = interner.index_expression(&goals, None);
        let extra_goals = interner.index_expression(&extra_goals, None);
        let goal_condition = if matches!(config.heuristic_kind, HeuristicKind::HFF) {
            goals.clone()
        } else {
            conjunction(&[&goals, &extra_goals])
        };
        for node in &goal_condition.nodes {
            if let IndexedHeuristicExpressionNode::Leaf(cid) = node {
                interner.add_reader(*cid, goal_op);
            }
        }
        op_conditions.push(goal_condition);

        let op_effects: Vec<Vec<u32>> = operators
            .iter()
            .map(|o| {
                o.effects
                    .iter()
                    .filter_map(|e| interner.ids.get(e).copied())
                    .collect()
            })
            .collect();

        let op_numeric_conds: Vec<Vec<NumericAchievement>> = achieved_simple_numeric_conds
            .iter()
            .map(|conds| {
                conds
                    .iter()
                    .map(|c| {
                        let (fluents, weights) = &simple_numeric_conds[c];
                        (interner.ids[c], fluents.clone(), weights.clone())
                    })
                    .collect()
            })
            .collect();

        let extra_fluent_cids: Vec<(Action, Vec<u32>)> = extra_fluents
            .into_iter()
            .map(|(a, fluents)| (a, fluents.into_iter().map(|e| interner.intern(e)).collect()))
            .collect();

        // Classify each cid by how its initial cost is computed from the
        // state. Bookkeeping fluents are initialized from `state.todo`
        // instead, and any other cid is reached only through effects.
        let mut evaluated_conds = Vec::new();
        let mut true_fluent_conds = Vec::new();
        let mut false_fluent_conds = Vec::new();
        let mut equality_conds = Vec::new();
        for (cid, e) in interner.leaves.iter().enumerate() {
            let cid = cid as u32;
            let expr = expression_manager.force_get(e);
            if if_conds.contains(e) || complex_numeric_conds.contains(e) {
                evaluated_conds.push((cid, expr.clone(), 1.0));
            } else if simple_numeric_conds.contains_key(e) {
                evaluated_conds.push((cid, expr.clone(), f64::INFINITY));
            } else {
                match expr.as_slice() {
                    [ExpressionNode::Fluent(f)] if f.idx < n_real_fluents => {
                        true_fluent_conds.push((cid, *f));
                    }
                    [ExpressionNode::Fluent(f), ExpressionNode::Not(0)]
                        if f.idx < n_real_fluents =>
                    {
                        false_fluent_conds.push((cid, *f));
                    }
                    [ExpressionNode::Fluent(f), v, ExpressionNode::Equals(0, 1)]
                        if f.idx < n_real_fluents =>
                    {
                        equality_conds.push((cid, *f, v.clone()));
                    }
                    _ => {}
                }
            }
        }

        let events_len: FxHashMap<Action, usize> =
            events.into_iter().map(|(a, ev)| (a, ev.len())).collect();

        let internal_caching = if config.internal_caching {
            Some(FxHashMap::with_hasher(FxBuildHasher))
        } else {
            None
        };

        let drain_heap = matches!(config.heuristic_kind, HeuristicKind::HMAX)
            && op_numeric_conds.iter().any(|conds| !conds.is_empty());

        // `add_reader` registers each (cid, operator) pair once, so this
        // counts each operator's distinct leaves.
        let mut distinct_leaves = vec![0u32; op_conditions.len()];
        for readers in &interner.cond_to_ops {
            for &op in readers {
                distinct_leaves[op] += 1;
            }
        }

        let res = DeleteRelaxationHeuristic {
            actions,
            events: events_len,
            operators,
            goal_op,
            op_conditions,
            goals,
            cond_to_ops: interner.cond_to_ops,
            distinct_leaves,
            op_effects,
            op_numeric_conds,
            drain_heap,
            evaluated_conds,
            true_fluent_conds,
            false_fluent_conds,
            equality_conds,
            extra_fluent_cids,
            buffers: Arc::new(Mutex::new(FixpointBuffers::default())),
            heuristic_kind: config.heuristic_kind,
            internal_caching: Arc::new(Mutex::new(internal_caching)),
            inadmissible_numeric_heuristic_variant: config.inadmissible_numeric_heuristic_variant,
            disable_numeric_reasoning: config.disable_numeric_reasoning,
        };
        Ok(res)
    }

    pub fn reachable_actions(&self, state: &State) -> PyResult<FxHashSet<Action>> {
        let (_, reachable_operators) = self._eval(state, true)?;

        let mut action_operators = FxHashMap::with_hasher(FxBuildHasher);
        for o in &self.operators {
            *action_operators.entry(o.action).or_insert(0) += 1;
        }

        let mut action_reachable_operators = FxHashMap::with_hasher(FxBuildHasher);
        for operator_idx in reachable_operators.unwrap() {
            *action_reachable_operators
                .entry(&self.operators[operator_idx].action)
                .or_insert(0) += 1;
        }

        let reachable_actions = action_reachable_operators
            .into_iter()
            .filter_map(|(action, reachable_operators)| {
                if reachable_operators == action_operators[action] {
                    Some(*action)
                } else {
                    None
                }
            })
            .collect();

        Ok(reachable_actions)
    }

    pub fn eval(&self, state: &State) -> PyResult<Option<f64>> {
        let mut internal_caching = self.internal_caching.lock().unwrap();
        if let Some(internal_caching) = internal_caching.as_mut() {
            let todo_values: Vec<usize> = self
                .actions
                .iter()
                .map(|action| state.todo.get(action).map(|(j, _)| *j).unwrap_or(0))
                .collect();
            let cache_key = CacheKey {
                values: state.assignments.clone(),
                todo_values,
            };
            if let Some(res) = internal_caching.get(&cache_key) {
                return Ok(*res);
            }

            let (v, _) = self._eval(state, false)?;
            internal_caching.insert(cache_key, v);
            Ok(v)
        } else {
            let (v, _) = self._eval(state, false)?;
            Ok(v)
        }
    }

    /// Computes the heuristic value for a given state.
    ///
    /// This method evaluates the state using the selected delete-relaxation heuristic,
    /// which can be one of `hmax`, `hadd`, or `hff`. The returned value estimates
    /// the cost to reach the goal from the given state.
    ///
    /// If `reachability_analysis` is enabled, the method performs reachability
    /// analysis instead of computing the heuristic value and returns the set of
    /// reachable operators: the heap is then drained instead of stopping at
    /// the goal.
    ///
    /// # Arguments
    ///
    /// * `state` - The state to evaluate.
    /// * `reachability_analysis` - If `true`, perform reachability analysis and return
    ///   the reachable operators instead of the heuristic value.
    ///
    /// # Returns
    ///
    /// A tuple containing:
    /// * `Option<f64>` - The heuristic value, or `None` if not computed.
    /// * `Option<Vec<usize>>` - The indices of reachable operators if
    ///   `reachability_analysis` is `true`; otherwise `None`.
    fn _eval(
        &self,
        state: &State,
        reachability_analysis: bool,
    ) -> PyResult<(Option<f64>, Option<Vec<usize>>)> {
        let hmax = matches!(self.heuristic_kind, HeuristicKind::HMAX);
        let mut buffers = self.buffers.lock().unwrap();
        let buffers = &mut *buffers;
        buffers.reset(self.cond_to_ops.len(), self.goal_op + 1);
        self.init_condition_costs(state, &mut buffers.cond_cost)?;

        buffers.unreached.clear();
        buffers.unreached.extend_from_slice(&self.distinct_leaves);
        for (cid, readers) in self.cond_to_ops.iter().enumerate() {
            if buffers.cond_cost[cid].is_finite() {
                for &reader in readers {
                    buffers.unreached[reader] -= 1;
                }
            }
        }

        for (op, conditions) in self.op_conditions.iter().enumerate() {
            if !conditions.contains_or_node && buffers.unreached[op] > 0 {
                continue;
            }
            let c = expression_cost(conditions, &buffers.cond_cost, hmax, &mut buffers.stack);
            if c.is_finite() {
                buffers.op_cost[op] = c;
                buffers.heap.push(Reverse(QueuedOperator::new(c, op)));
            }
        }

        let drain = reachability_analysis || self.drain_heap;
        while let Some(Reverse(entry)) = buffers.heap.pop() {
            let (popped_cost, op) = (entry.cost(), entry.op());
            if buffers.closed[op] || popped_cost != buffers.op_cost[op] {
                // stale entry, superseded by a cheaper push
                continue;
            }
            buffers.closed[op] = true;
            if op == self.goal_op {
                if drain {
                    continue;
                }
                break;
            }
            self.expand(op, state, buffers)?;
        }

        if reachability_analysis {
            let reachable_operators: Vec<usize> = (0..self.goal_op)
                .filter(|&op| buffers.op_cost[op].is_finite())
                .collect();
            return Ok((None, Some(reachable_operators)));
        }

        let h = expression_cost(
            &self.op_conditions[self.goal_op],
            &buffers.cond_cost,
            hmax,
            &mut buffers.stack,
        );
        if !h.is_finite() {
            return Ok((None, None));
        }

        if !matches!(self.heuristic_kind, HeuristicKind::HFF) {
            return Ok((Some(h), None));
        }

        let mut res = 0.0;
        for (a, (j, _)) in state.todo.iter() {
            res += (self.events[a] - j) as f64;
        }

        if h == 0.0 {
            return Ok((Some(res), None));
        }

        let mut relaxed_plan = FxHashSet::with_hasher(FxBuildHasher);
        let mut stack: Vec<u32> = Vec::new();
        supporting_conditions(&self.goals, &buffers.cond_cost, &mut stack);
        let mut visited: FxHashSet<u32> = stack.iter().copied().collect();
        let mut leaves = Vec::new();
        while let Some(g) = stack.pop() {
            if let Some(op) = buffers.reached_by[g as usize] {
                relaxed_plan.insert(self.operators[op].action);
                supporting_conditions(&self.op_conditions[op], &buffers.cond_cost, &mut leaves);
                for cid in leaves.drain(..) {
                    if visited.insert(cid) {
                        stack.push(cid);
                    }
                }
            }
        }
        for a in relaxed_plan {
            if !state.todo.contains_key(&a) {
                res += self.events[&a] as f64;
            }
        }

        Ok((Some(res), None))
    }

    /// Sets the cost of every cid that holds in `state` (and of every
    /// complex numeric or interpreted-function condition that does not, to
    /// 1); every other cid stays unreached.
    fn init_condition_costs(&self, state: &State, cond_cost: &mut [f64]) -> PyResult<()> {
        for (cid, expr, false_cost) in &self.evaluated_conds {
            cond_cost[*cid as usize] =
                if internal_evaluate(expr, state)? == ExpressionNode::Bool(true) {
                    0.0
                } else {
                    *false_cost
                };
        }
        for (cid, f) in &self.true_fluent_conds {
            if *state.get_value(*f) == ExpressionNode::Bool(true) {
                cond_cost[*cid as usize] = 0.0;
            }
        }
        for (cid, f) in &self.false_fluent_conds {
            if *state.get_value(*f) == ExpressionNode::Bool(false) {
                cond_cost[*cid as usize] = 0.0;
            }
        }
        for (cid, f, v) in &self.equality_conds {
            let value = state.get_value(*f);
            if !matches!(value, ExpressionNode::Bool(_)) && value == v {
                cond_cost[*cid as usize] = 0.0;
            }
        }
        for (a, cids) in &self.extra_fluent_cids {
            let cid = match state.todo.get(a) {
                Some((j, _)) => cids[j - 1],
                None => *cids.last().unwrap(),
            };
            cond_cost[cid as usize] = 0.0;
        }
        Ok(())
    }

    /// Updates the cost of every condition achieved by operator `op`, just
    /// closed: its effects, and the simple numeric conditions it achieves.
    fn expand(&self, op: usize, state: &State, buffers: &mut FixpointBuffers) -> PyResult<()> {
        let o = &self.operators[op];
        let precondition_cost = buffers.op_cost[op];

        for &cid in &self.op_effects[op] {
            self.update_condition_cost(cid, o.cost + precondition_cost, op, buffers);
        }

        for (cid, fluents, weights) in &self.op_numeric_conds[op] {
            if buffers.cond_cost[*cid as usize] == 0.0 {
                // condition satisfied in state
                continue;
            }

            let rep = repetitions(
                o,
                fluents,
                weights,
                state,
                self.inadmissible_numeric_heuristic_variant,
            )?
            .unwrap();

            let cost = if matches!(self.heuristic_kind, HeuristicKind::HMAX) {
                let m = &mut buffers.min_achiever_pre_cost[*cid as usize];
                *m = f64::min(*m, precondition_cost);
                rep * o.cost + *m
            } else {
                rep * o.cost + precondition_cost
            };
            self.update_condition_cost(*cid, cost, op, buffers);
        }
        Ok(())
    }

    /// Lowers the cost of `cid` to `cost`, achieved by operator `op`, if
    /// that improves it, and updates the cost of every operator reading it
    /// that is still open. On a tie, hff's `reached_by` prefers the operator
    /// with the larger id.
    fn update_condition_cost(&self, cid: u32, cost: f64, op: usize, buffers: &mut FixpointBuffers) {
        let hff = matches!(self.heuristic_kind, HeuristicKind::HFF);
        let c = cid as usize;
        if cost < buffers.cond_cost[c] {
            let first_reach = buffers.cond_cost[c] == f64::INFINITY;
            buffers.cond_cost[c] = cost;
            if hff {
                buffers.reached_by[c] = Some(op);
            }
            let hmax = matches!(self.heuristic_kind, HeuristicKind::HMAX);
            for &reader in &self.cond_to_ops[c] {
                if buffers.closed[reader] {
                    continue;
                }
                if first_reach {
                    buffers.unreached[reader] -= 1;
                }
                if !self.op_conditions[reader].contains_or_node && buffers.unreached[reader] > 0 {
                    // still `inf`: some other leaf is unreached
                    continue;
                }
                let reader_cost = expression_cost(
                    &self.op_conditions[reader],
                    &buffers.cond_cost,
                    hmax,
                    &mut buffers.stack,
                );
                if reader_cost < buffers.op_cost[reader] {
                    buffers.op_cost[reader] = reader_cost;
                    buffers
                        .heap
                        .push(Reverse(QueuedOperator::new(reader_cost, reader)));
                }
            }
        } else if hff
            && cost == buffers.cond_cost[c]
            && buffers.reached_by[c].is_none_or(|prev| op > prev)
        {
            buffers.reached_by[c] = Some(op);
        }
    }

    pub fn name(&self) -> &'static str {
        if self.disable_numeric_reasoning {
            match self.heuristic_kind {
                HeuristicKind::HFF => "hff_no_numbers",
                HeuristicKind::HADD => "hadd_no_numbers",
                HeuristicKind::HMAX => "hmax_no_numbers",
            }
        } else {
            match self.heuristic_kind {
                HeuristicKind::HFF => "hff",
                HeuristicKind::HADD => "hadd",
                HeuristicKind::HMAX => "hmax",
            }
        }
    }
}

/// Per-fluent reachable-value tracking for `HMaxExplicit`: `values` is
/// insertion-ordered (so "every value added after size `n`" is the slice
/// `values[n..]`, which is what the semi-naive delta enumeration below walks).
#[derive(Clone, Debug, Default)]
struct ValueSet {
    values: Vec<ExpressionNode>,
    set: FxHashSet<ExpressionNode>,
}

impl ValueSet {
    fn len(&self) -> usize {
        self.values.len()
    }

    /// Inserts `v`, returning whether it was new.
    fn insert(&mut self, v: ExpressionNode) -> bool {
        if self.set.insert(v.clone()) {
            self.values.push(v);
            true
        } else {
            false
        }
    }
}

struct FluentAssignments<'a> {
    fluents: &'a [Fluent],
    values: &'a [&'a ExpressionNode],
}

impl FluentValueTrait for FluentAssignments<'_> {
    fn get_value(&self, fluent: Fluent) -> &ExpressionNode {
        let pos = self
            .fluents
            .iter()
            .position(|&f| f == fluent)
            .expect("fluent must be one of this expression's own fluents");
        self.values[pos]
    }
}

/// Fluent ids referenced by `exp`, deduplicated in first-occurrence order.
/// Dedup also shrinks the cross-product: an expression referencing the same
/// fluent twice (e.g. `x + x`) is a single cross-product dimension, not two.
fn dedup_fluents(exp: &[ExpressionNode]) -> Vec<Fluent> {
    let mut seen: FxHashSet<Fluent> = FxHashSet::with_hasher(FxBuildHasher);
    let mut out = Vec::new();
    for node in exp {
        if let ExpressionNode::Fluent(f) = node {
            if seen.insert(*f) {
                out.push(*f);
            }
        }
    }
    out
}

/// Enumerates the cartesian product of `assignments[fluents[i]].values[lo..hi]`
/// for each `(lo, hi)` in `ranges` (parallel to `fluents`), filling `cur`
/// with one reference per position and invoking `body` for each combination.
/// Any range with `lo >= hi` makes the whole product empty. `body` returns
/// `Ok(false)` to stop early, propagated as `Ok(false)`; a completed
/// enumeration returns `Ok(true)`.
fn for_each_combination<'a>(
    fluents: &[Fluent],
    assignments: &'a [ValueSet],
    ranges: &[(usize, usize)],
    cur: &mut Vec<&'a ExpressionNode>,
    mut body: impl FnMut(&[&'a ExpressionNode]) -> PyResult<bool>,
) -> PyResult<bool> {
    let k = fluents.len();
    if ranges.iter().any(|&(lo, hi)| lo >= hi) {
        return Ok(true);
    }
    let mut idx: Vec<usize> = ranges.iter().map(|&(lo, _)| lo).collect();
    cur.clear();
    cur.resize(k, &assignments[fluents[0].idx].values[idx[0]]);
    loop {
        for i in 0..k {
            cur[i] = &assignments[fluents[i].idx].values[idx[i]];
        }
        if !body(cur)? {
            return Ok(false);
        }
        // Mixed-radix increment, rightmost position first.
        let mut pos = k;
        loop {
            if pos == 0 {
                return Ok(true);
            }
            pos -= 1;
            idx[pos] += 1;
            if idx[pos] < ranges[pos].1 {
                break;
            }
            idx[pos] = ranges[pos].0;
        }
    }
}

/// Semi-naive delta enumeration: evaluates `exp` on every combination of its
/// (already deduplicated) `fluents`' reachable values that includes at
/// least one value added since `old_sizes` was last updated by this same
/// function, then updates `old_sizes` to the current sizes. Reachable-value
/// sets only ever grow and a combination is consumed either as a set
/// (effects) or as an existence check (`exp_can_be_true`), so which round
/// first sees a given combination doesn't matter, only that it's seen
/// exactly once.
///
/// `old_sizes` is intentionally left stale when `visit` stops the
/// enumeration early: the only caller that does that (`exp_can_be_true`,
/// via a `true` result) never re-checks that expression again this eval.
///
/// A nullary expression (no fluents) has exactly one possible value,
/// computed once ever: `old_sizes` conventionally holds `[1]` after that
/// happens, and anything else (including empty) before.
fn for_each_new_value(
    exp: &[ExpressionNode],
    fluents: &[Fluent],
    assignments: &[ValueSet],
    old_sizes: &mut Vec<usize>,
    scratch: &mut Vec<ExpressionNode>,
    mut visit: impl FnMut(ExpressionNode) -> PyResult<bool>,
) -> PyResult<bool> {
    let k = fluents.len();
    if k == 0 {
        if old_sizes.first() == Some(&1) {
            return Ok(true);
        }
        let empty = FluentAssignments {
            fluents: &[],
            values: &[],
        };
        let value = internal_evaluate_into(exp, &empty, scratch)?;
        *old_sizes = vec![1];
        return visit(value);
    }

    let new_sizes: Vec<usize> = fluents.iter().map(|&f| assignments[f.idx].len()).collect();
    if old_sizes.len() != k {
        old_sizes.clear();
        old_sizes.resize(k, 0);
    }
    if *old_sizes == new_sizes {
        return Ok(true);
    }

    let mut cur: Vec<&ExpressionNode> = Vec::new();
    for j in 0..k {
        let ranges: Vec<(usize, usize)> = (0..k)
            .map(|i| {
                if i < j {
                    (0, old_sizes[i])
                } else if i == j {
                    (old_sizes[i], new_sizes[i])
                } else {
                    (0, new_sizes[i])
                }
            })
            .collect();
        let completed =
            for_each_combination(fluents, assignments, &ranges, &mut cur, |cur_slice| {
                let fa = FluentAssignments {
                    fluents,
                    values: cur_slice,
                };
                let value = internal_evaluate_into(exp, &fa, scratch)?;
                visit(value)
            })?;
        if !completed {
            return Ok(false);
        }
    }
    *old_sizes = new_sizes;
    Ok(true)
}

#[derive(Clone, Debug)]
pub struct HMaxExplicit {
    actions: Vec<Action>,
    events: FxHashMap<Action, Vec<(Timing, Event)>>,
    goals: Vec<Vec<ExpressionNode>>,
    goal_expressions: Vec<Expression>,
    expr_fluents: FxHashMap<Expression, Vec<Fluent>>,
    extra_fluents: FxHashMap<Action, Vec<Vec<ExpressionNode>>>,
    num_fluents: usize,
    operators: Vec<OperatorHmax>,
    operator_conditions_fluents: Vec<FxHashSet<Fluent>>,
    operator_effects_fluents: Vec<FxHashSet<Fluent>>,
    internal_caching: HeuristicCache,
}

impl HMaxExplicit {
    pub fn new(
        actions: Vec<Action>,
        fluent_domains: Vec<FluentDomain>,
        events: FxHashMap<Action, Vec<(Timing, Event)>>,
        goals: Vec<PyExpressionNode>,
        internal_caching: bool,
    ) -> PyResult<Self> {
        let mut operators = Vec::new();
        let mut extra_fluents = FxHashMap::with_hasher(FxBuildHasher);
        let mut extra_goals = Vec::new();
        let mut expression_manager = ExpressionManager::new();
        let mut num_fluents = fluent_domains.len();

        for (a, le) in events.iter() {
            let mut a_extra_fluents = Vec::new();
            let f_cond = Fluent::new(num_fluents + le.len() - 1);
            let mut cond: Vec<ExpressionNode> = vec![ExpressionNode::Fluent(f_cond)];
            extra_goals.push(cond.clone());
            for (_, e) in le.iter() {
                let mut effects = Vec::new();
                let mut conditions = Vec::new();
                let f = Fluent::new(num_fluents);
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
                    if !condition.is_empty() && condition != vec![ExpressionNode::Bool(true)] {
                        conditions.extend(split_expression(&condition)?);
                    }
                }
                if !conditions.contains(&vec![ExpressionNode::Bool(false)]) {
                    let condition_expressions: Vec<Expression> = conditions
                        .iter()
                        .map(|cond| expression_manager.put(cond))
                        .collect();
                    let effect_fluents: Vec<Vec<Fluent>> =
                        effects.iter().map(|e| dedup_fluents(&e.value)).collect();
                    operators.push(OperatorHmax {
                        action: *a,
                        conditions,
                        condition_expressions,
                        effects,
                        effect_fluents,
                        cost: 1.0,
                    });
                }
                cond = vec![ExpressionNode::Fluent(f)];
            }
            extra_fluents.insert(*a, a_extra_fluents);
        }

        let mut goals = split_expression(&goals.into_iter().map(|e| e.v).collect::<Vec<_>>())?;
        goals.extend(extra_goals);
        let goal_expressions: Vec<Expression> = goals
            .iter()
            .map(|cond| expression_manager.put(cond))
            .collect();

        let mut expr_fluents: FxHashMap<Expression, Vec<Fluent>> =
            FxHashMap::with_hasher(FxBuildHasher);
        for (cond, &id) in goals.iter().zip(goal_expressions.iter()) {
            expr_fluents
                .entry(id)
                .or_insert_with(|| dedup_fluents(cond));
        }
        for operator in &operators {
            for (cond, &id) in operator
                .conditions
                .iter()
                .zip(operator.condition_expressions.iter())
            {
                expr_fluents
                    .entry(id)
                    .or_insert_with(|| dedup_fluents(cond));
            }
        }

        let mut operator_conditions_fluents = Vec::with_capacity(operators.len());
        for operator in &operators {
            let mut conditions_fluents = FxHashSet::with_hasher(FxBuildHasher);
            for cond in &operator.conditions {
                for exp_node in cond {
                    if let ExpressionNode::Fluent(f) = exp_node {
                        conditions_fluents.insert(*f);
                    }
                }
            }
            operator_conditions_fluents.push(conditions_fluents);
        }

        let mut operator_effects_fluents = Vec::with_capacity(operators.len());
        for operator in &operators {
            let mut effects_fluents = FxHashSet::with_hasher(FxBuildHasher);
            for eff in &operator.effects {
                for exp_node in &eff.value {
                    if let ExpressionNode::Fluent(f) = exp_node {
                        effects_fluents.insert(*f);
                    }
                }
            }
            operator_effects_fluents.push(effects_fluents);
        }

        let internal_caching = if internal_caching {
            Some(FxHashMap::with_hasher(FxBuildHasher))
        } else {
            None
        };

        let res = HMaxExplicit {
            actions,
            events,
            goals,
            goal_expressions,
            expr_fluents,
            extra_fluents,
            num_fluents,
            operators,
            operator_conditions_fluents,
            operator_effects_fluents,
            internal_caching: Arc::new(Mutex::new(internal_caching)),
        };
        Ok(res)
    }

    /// Enumerates only the values `exp` can newly take (`for_each_new_value`)
    /// and reports whether any of them is `true`, short-circuiting on the
    /// first one. Unlike `DeleteRelaxationHeuristic`, which only ever
    /// evaluates an interpreted-function condition against the real,
    /// concrete search state, this cross-product can hand a callable an
    /// argument combination that never jointly occurs in a reachable state.
    /// A partial callable (e.g. a lookup table missing a key) can therefore
    /// raise here in a way it wouldn't under hff/hadd/hmax -- the same class
    /// of hazard as `internal_evaluate`'s own `Div` raising a
    /// `ZeroDivisionError` on relaxed values, just with arbitrary user code
    /// instead of a builtin operator -- so this (and every caller up to
    /// `eval`) propagates `PyResult`.
    fn exp_can_be_true(
        &self,
        exp: &[ExpressionNode],
        exp_id: Expression,
        assignments: &[ValueSet],
        cache_can_be_true: &mut FxHashMap<Expression, bool>,
        old_sizes_by_expr: &mut FxHashMap<Expression, Vec<usize>>,
        scratch: &mut Vec<ExpressionNode>,
    ) -> PyResult<bool> {
        if cache_can_be_true.get(&exp_id) == Some(&true) {
            return Ok(true);
        }

        let fluents = &self.expr_fluents[&exp_id];
        let old_sizes = old_sizes_by_expr.entry(exp_id).or_default();
        let mut found = false;
        for_each_new_value(exp, fluents, assignments, old_sizes, scratch, |value| {
            if value == ExpressionNode::Bool(true) {
                found = true;
                Ok(false)
            } else {
                Ok(true)
            }
        })?;

        if found {
            cache_can_be_true.insert(exp_id, true);
        }
        Ok(found)
    }

    fn can_be_true(
        &self,
        expressions: &[Vec<ExpressionNode>],
        expression_ids: &[Expression],
        assignments: &[ValueSet],
        cache_can_be_true: &mut FxHashMap<Expression, bool>,
        old_sizes_by_expr: &mut FxHashMap<Expression, Vec<usize>>,
        scratch: &mut Vec<ExpressionNode>,
    ) -> PyResult<bool> {
        for (i, exp) in expressions.iter().enumerate() {
            if !self.exp_can_be_true(
                exp,
                expression_ids[i],
                assignments,
                cache_can_be_true,
                old_sizes_by_expr,
                scratch,
            )? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    pub fn eval(&self, state: &State) -> PyResult<Option<f64>> {
        let mut internal_caching = self.internal_caching.lock().unwrap();
        if let Some(internal_caching) = internal_caching.as_mut() {
            let todo_values: Vec<usize> = self
                .actions
                .iter()
                .map(|action| state.todo.get(action).map(|(j, _)| *j).unwrap_or(0))
                .collect();
            let cache_key = CacheKey {
                values: state.assignments.clone(),
                todo_values,
            };
            if let Some(res) = internal_caching.get(&cache_key) {
                return Ok(*res);
            }

            let result = self._eval(state)?;
            internal_caching.insert(cache_key, result);
            Ok(result)
        } else {
            self._eval(state)
        }
    }

    fn _eval(&self, state: &State) -> PyResult<Option<f64>> {
        let mut assignments: Vec<ValueSet> = vec![ValueSet::default(); self.num_fluents];
        // add state assignments to assignments
        for (f, v) in state.assignments.iter().enumerate() {
            assignments[f].insert(v.clone());
        }
        // add extra fluents to assignments
        for action in self.events.keys() {
            let r = state.todo.get(action);
            let idx = match r {
                Some((j, _)) => j - 1,
                None => self.extra_fluents[action].len() - 1,
            };

            for (i, f) in self.extra_fluents[action].iter().enumerate() {
                if let ExpressionNode::Fluent(f) = &f[0] {
                    assignments[f.idx].insert(ExpressionNode::Bool(i == idx));
                }
            }
        }

        let mut cache_can_be_true: FxHashMap<Expression, bool> =
            FxHashMap::with_hasher(FxBuildHasher);
        let mut old_sizes_by_expr: FxHashMap<Expression, Vec<usize>> =
            FxHashMap::with_hasher(FxBuildHasher);
        // Per-(operator, effect) delta progress -- effects aren't interned
        // (only conditions/goals are), so they can't share
        // `old_sizes_by_expr`'s dedup and get their own tracker. Lazily
        // populated, not eagerly shaped from every operator up front.
        let mut eff_old_sizes: FxHashMap<(usize, usize), Vec<usize>> =
            FxHashMap::with_hasher(FxBuildHasher);
        let mut applied_operators = vec![false; self.operators.len()];

        // Dense "did this fluent's value set grow last round" tracking.
        let mut changed: Vec<bool> = vec![true; self.num_fluents];
        let mut changed_list: Vec<usize> = (0..self.num_fluents).collect();

        // Scratch reused across the whole eval by every `internal_evaluate_into`
        // call, instead of allocating a `Vec` per cross-product element.
        let mut scratch: Vec<ExpressionNode> = Vec::new();

        let mut depth = 0;
        while !changed_list.is_empty() {
            if self.can_be_true(
                &self.goals,
                &self.goal_expressions,
                &assignments,
                &mut cache_can_be_true,
                &mut old_sizes_by_expr,
                &mut scratch,
            )? {
                // goal satisfied
                return Ok(Some(depth as f64));
            }

            let mut new_assignments: FxHashMap<Fluent, FxHashSet<ExpressionNode>> =
                FxHashMap::with_hasher(FxBuildHasher);
            for (i, operator) in self.operators.iter().enumerate() {
                if applied_operators[i] {
                    // operator already applied
                    if !self.operator_effects_fluents[i]
                        .iter()
                        .any(|&f| changed[f.idx])
                    {
                        // no changes in the effect fluents
                        continue;
                    }
                } else if !self.operator_conditions_fluents[i]
                    .iter()
                    .any(|&f| changed[f.idx])
                {
                    // operator never applied, but no changes in the condition fluents
                    continue;
                } else if !self.can_be_true(
                    &operator.conditions,
                    &operator.condition_expressions,
                    &assignments,
                    &mut cache_can_be_true,
                    &mut old_sizes_by_expr,
                    &mut scratch,
                )? {
                    // operator cannot be applied
                    continue;
                } else {
                    // first time applied
                    applied_operators[i] = true;
                }

                for (j, effect) in operator.effects.iter().enumerate() {
                    let fluent_new_assignments = new_assignments
                        .entry(effect.fluent)
                        .or_insert_with(|| FxHashSet::with_hasher(FxBuildHasher));
                    let old_sizes = eff_old_sizes.entry((i, j)).or_default();
                    for_each_new_value(
                        &effect.value,
                        &operator.effect_fluents[j],
                        &assignments,
                        old_sizes,
                        &mut scratch,
                        |value| {
                            fluent_new_assignments.insert(value);
                            Ok(true)
                        },
                    )?;
                }
            }

            // update assignments
            // Reset only what `changed_list` actually marked last round --
            // not the whole dense array -- so this stays O(fluents changed
            // last round), rather than O(num_fluents) every round regardless
            // of how few fluents are actually still churning.
            for &f in changed_list.iter() {
                changed[f] = false;
            }
            changed_list.clear();
            for (fluent, new_vv) in new_assignments {
                for v in new_vv {
                    if assignments[fluent.idx].insert(v) && !changed[fluent.idx] {
                        changed[fluent.idx] = true;
                        changed_list.push(fluent.idx);
                    }
                }
            }

            depth += 1;
        }

        Ok(None)
    }

    pub fn name(&self) -> &'static str {
        "hmax_explicit"
    }
}
