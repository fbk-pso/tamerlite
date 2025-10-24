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
use std::cell::RefCell;
use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use std::{collections::HashMap, vec::Vec};

use pyo3::prelude::*;
use pyo3::types::PyTuple;

use super::expressions::*;
use super::expressions_utils::*;
use super::multiqueue::StateContainer;
use super::search_space::SearchSpaceTrait;
use super::search_state::State;
use super::structures::*;

pub trait HeuristicTrait {
    fn eval<S: SearchSpaceTrait>(&self, state: &State, ss: &S) -> PyResult<Option<f64>>;

    /// Evaluates the heuristic for a given state, returning an iterator over the results.
    /// This method is used in non-multiqueue search algorithms
    fn eval_gen<'a, I, S: SearchSpaceTrait>(
        &'a self,
        states_iter: I,
        ss: &'a S,
    ) -> PyResult<Box<dyn Iterator<Item = PyResult<(State, Option<f64>)>> + 'a>>
    where
        I: Iterator<Item = PyResult<State>> + 'a,
    {
        return Ok(Box::new(states_iter.map(|state| match state {
            Ok(state) => {
                let h_value = self.eval(&state, ss);
                match h_value {
                    Ok(x) => Ok((state, x)),
                    Err(e) => Err(e),
                }
            }
            Err(e) => Err(e),
        })));
    }

    /// Evaluates the heuristic for a given state, returning an iterator over the results.
    /// This method is used in multiqueue search algorithms
    fn eval_gen_container<'a, S: SearchSpaceTrait>(
        &'a self,
        states: &'a Vec<Rc<RefCell<StateContainer>>>,
        ss: &'a S,
    ) -> PyResult<Box<dyn Iterator<Item = PyResult<(usize, Option<f64>)>> + 'a>> {
        return Ok(Box::new(states.iter().enumerate().map(|(i, state)| {
            let h_value = self.eval(&state.borrow().state, ss);
            match h_value {
                Ok(x) => Ok((i, x)),
                Err(e) => Err(e),
            }
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

#[derive(Debug, Clone, PartialEq)]
struct Operator {
    action: String,
    conditions: Vec<HeuristicExpressionNode>,
    effects: Vec<Expression>,
    cost: f64,
}

impl Eq for Operator {}

impl Hash for Operator {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.action.hash(state);
        self.conditions.hash(state);
        self.effects.hash(state);
    }
}

#[derive(Debug, Clone, PartialEq, Hash)]
enum HeuristicExpressionNode {
    And(usize),
    Or(usize),
    Leaf(Expression),
}

#[derive(Debug, Clone, PartialEq)]
struct OperatorHmax {
    action: String,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct OperatorID {
    id: usize,
}

impl OperatorID {
    fn new(id: usize) -> OperatorID {
        OperatorID { id }
    }
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
struct CacheKey {
    values: Vec<ExpressionNode>,
    todo_values: Vec<usize>,
}

fn build_operator_conditions(
    conditions: &Vec<ExpressionNode>,
    extra_fluent: ExpressionNode,
    expression_manager: &mut ExpressionManager,
) -> (bool, Vec<HeuristicExpressionNode>) {
    let conditions = if conditions.len() == 0 || conditions == &vec![ExpressionNode::Bool(true)] {
        vec![extra_fluent]
    } else if conditions == &vec![ExpressionNode::Bool(false)] {
        return (false, vec![]);
    } else if matches!(conditions.last().unwrap(), ExpressionNode::And(_)) {
        let mut conditions = conditions.clone();
        let mut and_node = conditions.pop().unwrap();
        if let ExpressionNode::And(ref mut operands) = and_node {
            operands.push(conditions.len());
        }
        conditions.push(extra_fluent);
        conditions.push(and_node);
        conditions
    } else {
        let mut conditions = conditions.clone();
        conditions.push(extra_fluent);
        conditions.push(ExpressionNode::And(vec![
            conditions.len() - 2,
            conditions.len() - 1,
        ]));
        conditions
    };

    (
        true,
        convert_to_heuristic_expression(&conditions, expression_manager),
    )
}

fn convert_to_heuristic_expression(
    expr: &Vec<ExpressionNode>,
    expression_manager: &mut ExpressionManager,
) -> Vec<HeuristicExpressionNode> {
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
                    result.push(HeuristicExpressionNode::Or(operands.len()));
                }
            }
            _ => result.push(HeuristicExpressionNode::Leaf(
                expression_manager.put(&extract_sub_expression(&expr, idx)),
            )),
        }
    }

    result
}

fn is_numeric_leaf_expression(expr: &Vec<ExpressionNode>) -> bool {
    match expr.last() {
        Some(ExpressionNode::Bool(_) | ExpressionNode::Fluent(_)) => false,
        Some(ExpressionNode::Not(i)) => match expr[*i] {
            ExpressionNode::Fluent(_) => false,
            _ => true,
        },
        Some(ExpressionNode::Equals(i1, i2)) => {
            !(matches!(expr[*i1], ExpressionNode::Fluent(_))
                && matches!(expr[*i2], ExpressionNode::Object(_)))
        }
        Some(_) => true,
        None => false,
    }
}

fn extract_sub_expression(expr: &Vec<ExpressionNode>, idx: usize) -> Vec<ExpressionNode> {
    // find the start index of the sub-expression
    let mut i = idx;
    loop {
        let operands = match &expr[i] {
            ExpressionNode::Not(operand) => &vec![*operand],
            ExpressionNode::Equals(op1, op2)
            | ExpressionNode::LE(op1, op2)
            | ExpressionNode::LT(op1, op2)
            | ExpressionNode::Minus(op1, op2)
            | ExpressionNode::Div(op1, op2) => &vec![*op1, *op2],
            ExpressionNode::And(operands)
            | ExpressionNode::Or(operands)
            | ExpressionNode::Plus(operands)
            | ExpressionNode::Times(operands) => operands,
            _ => break,
        };
        i = *operands.iter().min().unwrap();
    }

    let offset = -(i as i32);
    let res: Vec<ExpressionNode> = (i..(idx + 1)).map(|j| do_shift(&expr[j], offset)).collect();
    res
}

#[derive(Clone, Debug)]
pub struct DeleteRelaxationHeuristic {
    events: HashMap<String, usize>,
    goals: Vec<HeuristicExpressionNode>,
    extra_fluents: HashMap<String, Vec<Expression>>,
    extra_goals: Vec<HeuristicExpressionNode>,
    operators: Vec<Operator>,
    precondition_of: HashMap<Expression, Vec<OperatorID>>,
    empty_pre_operators: HashSet<OperatorID>,
    numeric_conds: HashSet<Expression>,
    heuristic_kind: HeuristicKind,
    ordered_actions: Vec<String>,
    internal_caching: Arc<Mutex<Option<HashMap<CacheKey, Option<f64>>>>>,
    expression_manager: Arc<Mutex<ExpressionManager>>,
}

impl DeleteRelaxationHeuristic {
    pub fn new(
        fluent_types: Vec<String>,
        objects: HashMap<String, Vec<String>>,
        events: HashMap<String, Vec<(Timing, Event)>>,
        goal: Vec<PyExpressionNode>,
        heuristic_kind: HeuristicKind,
        internal_caching: bool,
    ) -> PyResult<Self> {
        let mut operators = Vec::new();
        let mut extra_fluents: HashMap<String, Vec<Expression>> = HashMap::new();
        let mut extra_goals = Vec::new();
        let mut expression_manager = ExpressionManager::new();
        let mut num_fluents = fluent_types.len();

        for (a, le) in events.iter() {
            let mut a_extra_fluents: Vec<Expression> = Vec::new();
            let f_cond = num_fluents + le.len() - 1;
            let mut cond = ExpressionNode::Fluent(f_cond);
            extra_goals.push(cond.clone());
            // TODO: handle e.effects.len()==1
            for (_, e) in le.iter() {
                let mut effects: Vec<Expression> = Vec::new();
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
                    } else if t != "real" && t != "int" {
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
                let (is_applicable, conditions) =
                    build_operator_conditions(&e.conditions, cond.clone(), &mut expression_manager);
                if is_applicable {
                    operators.push(Operator {
                        action: a.to_string(),
                        conditions,
                        effects,
                        cost: 1.0,
                    });
                }
                cond = ExpressionNode::Fluent(f);
            }
            extra_fluents.insert(a.to_string(), a_extra_fluents);
        }
        operators.sort_by(|a, b| a.action.cmp(&b.action));

        let expr_goals = goal.into_iter().map(|e| e.v).collect();
        let goals = convert_to_heuristic_expression(&expr_goals, &mut expression_manager);
        let extra_goals = convert_to_heuristic_expression(&extra_goals, &mut expression_manager);

        let mut precondition_of: HashMap<Expression, Vec<OperatorID>> = HashMap::new();
        let mut numeric_conds: HashSet<Expression> = HashSet::new();
        let mut empty_pre_operators: HashSet<OperatorID> = HashSet::new();
        for (idx_o, o) in operators.iter().enumerate() {
            if o.conditions.len() == 0 {
                empty_pre_operators.insert(OperatorID::new(idx_o));
            } else {
                for node in &o.conditions {
                    if let HeuristicExpressionNode::Leaf(e) = node {
                        if is_numeric_leaf_expression(expression_manager.force_get(e)) {
                            numeric_conds.insert(*e);
                        } else {
                            if !precondition_of.contains_key(e) {
                                precondition_of.insert(*e, vec![OperatorID::new(idx_o)]);
                            } else {
                                precondition_of
                                    .get_mut(e)
                                    .unwrap()
                                    .push(OperatorID::new(idx_o));
                            }
                        }
                    }
                }
            }
        }
        for node in goals.iter() {
            if let HeuristicExpressionNode::Leaf(e) = node {
                if is_numeric_leaf_expression(expression_manager.force_get(e)) {
                    numeric_conds.insert(*e);
                }
            }
        }

        let events_len: HashMap<String, usize> = events
            .iter()
            .map(|(a, ev)| (a.to_string(), ev.len()))
            .collect();

        let ordered_actions: Vec<String> = events.keys().map(|action| action.clone()).collect();
        let internal_caching = if internal_caching {
            Some(HashMap::new())
        } else {
            None
        };

        let res = DeleteRelaxationHeuristic {
            events: events_len,
            goals,
            extra_fluents,
            extra_goals,
            operators,
            precondition_of,
            empty_pre_operators,
            numeric_conds,
            heuristic_kind,
            ordered_actions,
            internal_caching: Arc::new(Mutex::new(internal_caching)),
            expression_manager: Arc::new(Mutex::new(expression_manager)),
        };
        Ok(res)
    }

    pub fn eval(&self, state: &State) -> PyResult<Option<f64>> {
        let mut internal_caching = self.internal_caching.lock().unwrap();
        if let Some(internal_caching) = internal_caching.as_mut() {
            let values: Vec<ExpressionNode> = state.assignments.iter().cloned().collect();
            let todo_values: Vec<usize> = self
                .ordered_actions
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

    fn _eval(&self, state: &State) -> PyResult<Option<f64>> {
        let mut expression_manager = self.expression_manager.lock().unwrap();
        let init_capacity = state.assignments.len() + self.numeric_conds.len() + self.events.len();
        let mut costs: HashMap<Expression, f64> = HashMap::with_capacity(init_capacity);
        let mut lp: Vec<Expression> = Vec::with_capacity(init_capacity);

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
            lp.push(k);
            costs.insert(k, 0.0);
        }

        for c in self.numeric_conds.iter() {
            // TODO: lazy eval?
            if internal_evaluate(expression_manager.force_get(c), state)?
                == ExpressionNode::Bool(true)
            {
                costs.insert(*c, 0.0);
            } else {
                costs.insert(*c, 1.0);
            }
            lp.push(*c);
        }

        for a in self.events.keys() {
            let v = match state.todo.get(a) {
                Some((j, _)) => self.extra_fluents.get(a).unwrap().get(j - 1),
                None => self.extra_fluents.get(a).unwrap().last(),
            };
            if let Some(x) = v {
                costs.insert(*x, 0.0);
                lp.push(*x);
            }
        }

        let mut reached_by: HashMap<Expression, OperatorID> = HashMap::new();
        while lp.len() > 0 {
            let mut lo: HashSet<OperatorID> = HashSet::new();
            for x in self.empty_pre_operators.iter() {
                lo.insert(*x);
            }
            for p in lp.iter() {
                if let Some(po) = self.precondition_of.get(p) {
                    for idx_o in po.iter() {
                        lo.insert(*idx_o);
                    }
                }
            }
            lp.clear();
            let mut new_costs = HashMap::new();
            for oid in lo {
                let o: &Operator = &self.operators[oid.id];
                if let Some(c) = self.cost(&o.conditions, &costs) {
                    for k in o.effects.iter() {
                        let new_cost_k = new_costs.get(k);
                        let cost_k = costs.get(k);
                        if (new_cost_k.is_some() && *new_cost_k.unwrap() > c + o.cost)
                            || (new_cost_k.is_none() && cost_k.is_none())
                            || (new_cost_k.is_none() && *cost_k.unwrap() > c + o.cost)
                        {
                            if matches!(self.heuristic_kind, HeuristicKind::HFF) {
                                reached_by.insert(*k, oid);
                            }
                            new_costs.insert(*k, c + o.cost);
                            lp.push(*k);
                        } else if matches!(self.heuristic_kind, HeuristicKind::HFF)
                            && ((new_cost_k.is_some() && *new_cost_k.unwrap() == c + o.cost)
                                || (new_cost_k.is_none() && *cost_k.unwrap() == c + o.cost))
                            && oid.id > reached_by[k].id
                        {
                            reached_by.insert(*k, oid);
                        }
                    }
                }
            }
            costs.extend(new_costs);
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

        // FIXME
        let mut relaxed_plan = HashSet::new();
        let mut stack: Vec<&Expression> = self
            .goals
            .iter()
            .filter_map(|node| match node {
                HeuristicExpressionNode::Leaf(e) => Some(e),
                _ => None,
            })
            .collect();
        while stack.len() > 0 {
            let g = stack.pop().unwrap();
            if let Some(oid) = reached_by.get(g) {
                let o: &Operator = &self.operators[oid.id];
                relaxed_plan.insert(o.action.to_string());
                stack.extend(o.conditions.iter().filter_map(|c| match c {
                    HeuristicExpressionNode::Leaf(e) => Some(e),
                    _ => None,
                }));
            }
        }
        for a in relaxed_plan.iter() {
            if !state.todo.contains_key(a) {
                res += self.events[a] as f64;
            }
        }

        Ok(Some(res))
    }

    fn cost(
        &self,
        expr: &Vec<HeuristicExpressionNode>,
        costs: &HashMap<Expression, f64>,
    ) -> Option<f64> {
        if let HeuristicExpressionNode::Leaf(e) = expr.last().unwrap() {
            return costs.get(e).cloned();
        }

        let mut res: Vec<Option<f64>> = Vec::with_capacity(expr.len());
        for node in expr {
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
                    if is_none {
                        res.push(None);
                    } else {
                        res.push(Some(r));
                    }
                }
                HeuristicExpressionNode::Or(num_operands) => {
                    let mut r = 0.0;
                    let mut is_none = false;
                    for i in 0..*num_operands {
                        match res[res.len() - i - 1] {
                            Some(v) => r = f64::min(r, v),
                            None => {
                                is_none = true;
                                break;
                            }
                        }
                    }
                    if is_none {
                        res.push(None);
                    } else {
                        res.push(Some(r));
                    }
                }
            }
        }

        res.last().copied().flatten()
    }

    pub fn name(&self) -> String {
        match self.heuristic_kind {
            HeuristicKind::HFF => String::from("hff"),
            HeuristicKind::HADD => String::from("hadd"),
            HeuristicKind::HMAX => String::from("hmax"),
        }
    }
}

pub struct FluentAssignments<'a> {
    pub assignments: HashMap<usize, &'a ExpressionNode>,
}

impl FluentValueTrait for FluentAssignments<'_> {
    fn get_value(&self, fluent: usize) -> &ExpressionNode {
        self.assignments.get(&fluent).unwrap()
    }
}

impl<'a> FluentAssignments<'a> {
    pub fn new(fluents: &Vec<usize>, values: Vec<&'a ExpressionNode>) -> Self {
        let assignments: HashMap<usize, &ExpressionNode> =
            fluents.iter().cloned().zip(values.into_iter()).collect();
        FluentAssignments { assignments }
    }
}

#[derive(Clone, Debug)]
pub struct HMaxNumeric {
    goals: Vec<Vec<ExpressionNode>>,
    goal_expressions: Vec<Expression>,
    extra_fluents: HashMap<String, Vec<Vec<ExpressionNode>>>,
    num_fluents: usize,
    operators: Vec<OperatorHmax>,
    operator_conditions_fluents: Vec<HashSet<usize>>,
    operator_effects_fluents: Vec<HashSet<usize>>,
    ordered_actions: Vec<String>,
    internal_caching: Arc<Mutex<Option<HashMap<CacheKey, Option<f64>>>>>,
}

impl HMaxNumeric {
    pub fn new(
        fluent_types: Vec<String>,
        events: HashMap<String, Vec<(Timing, Event)>>,
        goal: Vec<PyExpressionNode>,
        internal_caching: bool,
    ) -> PyResult<Self> {
        let mut operators = Vec::new();
        let mut extra_fluents = HashMap::new();
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
                if e.conditions.len() > 0 && e.conditions != vec![ExpressionNode::Bool(true)] {
                    conditions.extend(split_expression(&e.conditions)?);
                }
                let condition_expressions: Vec<Expression> = conditions
                    .iter()
                    .map(|cond| expression_manager.put(cond))
                    .collect();
                if !conditions.contains(&vec![ExpressionNode::Bool(false)]) {
                    operators.push(OperatorHmax {
                        action: a.to_string(),
                        conditions,
                        condition_expressions,
                        effects,
                        cost: 1.0,
                    });
                }
                cond = vec![ExpressionNode::Fluent(f)];
            }
            extra_fluents.insert(a.to_string(), a_extra_fluents);
        }

        let mut goals = split_expression(&goal.into_iter().map(|e| e.v).collect())?;
        goals.extend(extra_goals);
        let goal_expressions: Vec<Expression> = goals
            .iter()
            .map(|cond| expression_manager.put(cond))
            .collect();

        let mut operator_conditions_fluents = Vec::with_capacity(operators.len());
        for operator in &operators {
            let mut conditions_fluents = HashSet::new();
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
            let mut effects_fluents = HashSet::new();
            for eff in &operator.effects {
                for exp_node in &eff.value {
                    if let ExpressionNode::Fluent(f) = exp_node {
                        effects_fluents.insert(f.clone());
                    }
                }
            }
            operator_effects_fluents.push(effects_fluents);
        }

        let ordered_actions: Vec<String> = events.keys().map(|action| action.clone()).collect();
        let internal_caching = if internal_caching {
            Some(HashMap::new())
        } else {
            None
        };

        let res = HMaxNumeric {
            goals,
            goal_expressions,
            extra_fluents,
            num_fluents,
            operators,
            operator_conditions_fluents,
            operator_effects_fluents,
            ordered_actions,
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
        assignments: &'a Vec<HashSet<ExpressionNode>>,
        exp_fluents: &'a Vec<usize>,
    ) -> impl Iterator<Item = ExpressionNode> + 'a {
        let values: Vec<&HashSet<ExpressionNode>> =
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
        assignments: &Vec<HashSet<ExpressionNode>>,
        assignments_changes: &HashSet<usize>,
        cache_can_be_true: &mut HashMap<Expression, bool>,
    ) -> bool {
        let exp_fluents;
        if cache_can_be_true.contains_key(&exp_id) {
            if cache_can_be_true[&exp_id] {
                return true;
            }

            exp_fluents = self.extract_fluents(exp);
            let exp_fluents_set: HashSet<usize> = exp_fluents.iter().copied().collect();
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
        assignments: &Vec<HashSet<ExpressionNode>>,
        assignments_changes: &HashSet<usize>,
        cache_can_be_true: &mut HashMap<Expression, bool>,
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
                .ordered_actions
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
        let mut assignments: Vec<HashSet<ExpressionNode>> = vec![HashSet::new(); self.num_fluents];
        // add state assignments to assignments
        for (f, v) in state.assignments.iter().enumerate() {
            assignments[f] = HashSet::from([v.clone()]);
        }
        // add extra fluents to assignments
        for action in &self.ordered_actions {
            let r = state.todo.get(action);
            let idx = match r {
                Some((j, _)) => j - 1,
                None => self.extra_fluents[action].len() - 1,
            };

            for (i, f) in self.extra_fluents[action].iter().enumerate() {
                if let ExpressionNode::Fluent(f) = &f[0] {
                    assignments[*f] = HashSet::from([ExpressionNode::Bool(i == idx)]);
                }
            }
        }

        let mut cache_can_be_true: HashMap<Expression, bool> = HashMap::new();
        let mut applied_operators = vec![false; self.operators.len()];
        let mut assignments_changes: HashSet<usize> = (0..self.num_fluents).collect();
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

            let mut new_assignments: HashMap<usize, HashSet<ExpressionNode>> = HashMap::new();
            for (i, operator) in self.operators.iter().enumerate() {
                if applied_operators[i] {
                    // operator already applied
                    let eff_fluents: HashSet<usize> =
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
                        .or_insert_with(HashSet::new)
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
        String::from("hmax_numeric")
    }
}
