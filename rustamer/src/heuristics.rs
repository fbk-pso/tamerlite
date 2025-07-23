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

use itertools::Itertools;
use multiset::HashMultiSet;
use std::cell::RefCell;
use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use std::{collections::HashMap, vec::Vec};

use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::multiqueue::StateContainer;
use crate::state_encoder::CoreStateEncoder;
use crate::{internal_evaluate, SearchSpace};

use super::expressions::*;
use super::expressions_utils::*;
use super::search_state::State;
use super::structures::*;

#[derive(Clone, Debug)]
enum HeuristicKind {
    HFF,
    HADD,
    HMAX,
}

#[pyclass(frozen)]
#[derive(Clone)]
pub struct Heuristic {
    hdr: Option<DeleteRelaxationHeuristic>,
    hmax: Option<HMaxNumeric>,
    hrl: Option<HRL>,
    hcustom: Option<CustomHeuristic>,
    cache_value_in_state: bool,
}

#[pymethods]
impl Heuristic {
    #[staticmethod]
    pub fn custom(callable: PyObject, cache_value_in_state: bool) -> PyResult<Self> {
        Ok(Heuristic {
            hdr: None,
            hmax: None,
            hrl: None,
            hcustom: Some(CustomHeuristic::new(callable)?),
            cache_value_in_state: cache_value_in_state,
        })
    }

    #[staticmethod]
    pub fn hff(
        fluents: HashMap<String, String>,
        objects: HashMap<String, Vec<String>>,
        events: HashMap<String, Vec<(Timing, Event)>>,
        goal: Vec<PyExpressionNode>,
        internal_caching: bool,
        cache_value_in_state: bool,
    ) -> PyResult<Self> {
        Ok(Heuristic {
            hdr: Some(DeleteRelaxationHeuristic::new(
                fluents,
                objects,
                events,
                goal,
                HeuristicKind::HFF,
                internal_caching,
            )?),
            hmax: None,
            hrl: None,
            hcustom: None,
            cache_value_in_state: cache_value_in_state,
        })
    }

    #[staticmethod]
    pub fn hadd(
        fluents: HashMap<String, String>,
        objects: HashMap<String, Vec<String>>,
        events: HashMap<String, Vec<(Timing, Event)>>,
        goal: Vec<PyExpressionNode>,
        internal_caching: bool,
        cache_value_in_state: bool,
    ) -> PyResult<Self> {
        Ok(Heuristic {
            hdr: Some(DeleteRelaxationHeuristic::new(
                fluents,
                objects,
                events,
                goal,
                HeuristicKind::HADD,
                internal_caching,
            )?),
            hmax: None,
            hrl: None,
            hcustom: None,
            cache_value_in_state: cache_value_in_state,
        })
    }

    #[staticmethod]
    pub fn hmax(
        fluents: HashMap<String, String>,
        objects: HashMap<String, Vec<String>>,
        events: HashMap<String, Vec<(Timing, Event)>>,
        goal: Vec<PyExpressionNode>,
        internal_caching: bool,
        cache_value_in_state: bool,
    ) -> PyResult<Self> {
        Ok(Heuristic {
            hdr: Some(DeleteRelaxationHeuristic::new(
                fluents,
                objects,
                events,
                goal,
                HeuristicKind::HMAX,
                internal_caching,
            )?),
            hmax: None,
            hrl: None,
            hcustom: None,
            cache_value_in_state: cache_value_in_state,
        })
    }

    #[staticmethod]
    pub fn hmax_numeric(
        fluents: HashMap<String, String>,
        _objects: HashMap<String, Vec<String>>,
        events: HashMap<String, Vec<(Timing, Event)>>,
        goal: Vec<PyExpressionNode>,
        internal_caching: bool,
        cache_value_in_state: bool,
    ) -> PyResult<Self> {
        Ok(Heuristic {
            hdr: None,
            hmax: Some(HMaxNumeric::new(fluents, events, goal, internal_caching)?),
            hrl: None,
            hcustom: None,
            cache_value_in_state: cache_value_in_state,
        })
    }

    #[staticmethod]
    #[pyo3(signature = (name, ss, goals_vec, constants_vec, callable, h_sym=None, cache_value_in_state=false))]
    pub fn hrl(
        name: &str,
        ss: &CoreStateEncoder,
        goals_vec: Vec<f32>,
        constants_vec: Vec<f32>,
        callable: PyObject,
        h_sym: Option<Heuristic>,
        cache_value_in_state: bool,
    ) -> PyResult<Self> {
        Ok(Heuristic {
            hdr: None,
            hmax: None,
            hrl: Some(HRL::new(
                ss,
                goals_vec,
                constants_vec,
                callable,
                h_sym,
                name,
            )?),
            hcustom: None,
            cache_value_in_state: cache_value_in_state,
        })
    }

    pub fn eval(&self, state: &State, ss: &SearchSpace) -> PyResult<Option<f64>> {
        if self.cache_value_in_state {
            let heuristic_cache = state.heuristic_cache.lock().unwrap();
            if let Some(h_value) = heuristic_cache.get(&self.name()) {
                return Ok(*h_value);
            }
        }
        let h_value = {
            if self.hdr.is_some() {
                let h = self.hdr.as_ref().unwrap();
                h.eval(state)
            } else if self.hmax.is_some() {
                let h = self.hmax.as_ref().unwrap();
                h.eval(state)
            } else if self.hcustom.is_some() {
                let h = self.hcustom.as_ref().unwrap();
                h.eval(state)
            } else if self.hrl.is_some() {
                let h = self.hrl.as_ref().unwrap();
                h.eval(state, ss)
            } else {
                Ok(Some(0.0))
            }
        };
        if self.cache_value_in_state {
            let mut heuristic_cache = state.heuristic_cache.lock().unwrap();
            if let Ok(h_value) = h_value {
                heuristic_cache.insert(self.name().to_string(), h_value);
            }
        }
        return h_value;
    }

    pub fn name(&self) -> String {
        if self.hdr.is_some() {
            let h = self.hdr.as_ref().unwrap();
            h.name()
        } else if self.hmax.is_some() {
            let h = self.hmax.as_ref().unwrap();
            h.name()
        } else if self.hcustom.is_some() {
            let h = self.hcustom.as_ref().unwrap();
            h.name()
        } else if self.hrl.is_some() {
            let h = self.hrl.as_ref().unwrap();
            h.name()
        } else {
            String::from("")
        }
    }
}

// Auxiliary enum used to transform iterators of states into vectors amenable for
// vectorized evaluation
enum StateMode {
    Cached(Option<f64>),
    Error(PyErr),
    ToEval(usize),
    Unknown,
}

// Function to create a vector of StateMode of length l pre-initialized with Unknown
fn mk_unknown_state_mode_vec(l: usize) -> Vec<StateMode> {
    let mut states_mapping: Vec<StateMode> = Vec::new();
    for _i in 0..l {
        states_mapping.push(StateMode::Unknown);
    }
    states_mapping
}

impl Heuristic {
    /// Evaluates the heuristic for a given state, returning an iterator over the results.
    /// This method is used in non-multiqueue search algorithms
    /// If the heuristic is an HRL, it collects states to evaluate in batches.
    pub fn eval_gen<'a, I>(
        &'a self,
        states_iter: I,
        ss: &'a SearchSpace,
    ) -> PyResult<Box<dyn Iterator<Item = PyResult<(State, Option<f64>)>> + 'a>>
    where
        I: Iterator<Item = PyResult<State>> + 'a,
    {
        if self.hrl.is_some() {
            let states = states_iter.collect_vec();
            let mut states_mapping = mk_unknown_state_mode_vec(states.len());
            let mut vectors_to_eval: Vec<Vec<f32>> = Vec::new();
            let mut sym_heuristics_to_eval: Vec<f64> = Vec::new();
            for (i, rstate) in states.iter().enumerate() {
                if let Ok(state) = rstate {
                    states_mapping[i] = self.mk_state_mode(
                        ss,
                        state,
                        &mut vectors_to_eval,
                        &mut sym_heuristics_to_eval,
                    );
                }
                // No need to consider the Err case, it is handled in the final
                // iterator which re-consider the `states` vector
            }
            let computed_heuristics = self
                .hrl
                .as_ref()
                .unwrap()
                .eval_vector(vectors_to_eval, sym_heuristics_to_eval);
            match computed_heuristics {
                Ok(vc) => Ok(Box::new(
                    states
                        .into_iter()
                        .zip_eq(states_mapping.into_iter())
                        .map(move |(rs, case)| match rs {
                            Ok(state) => match case {
                                StateMode::Cached(x) => Ok((state, x)),
                                StateMode::Error(x) => Err(x),
                                StateMode::ToEval(idx) => {
                                    if self.cache_value_in_state {
                                        state
                                            .heuristic_cache
                                            .lock()
                                            .unwrap()
                                            .insert(self.name().to_string(), Some(vc[idx]));
                                    }
                                    Ok((state, Some(vc[idx])))
                                }
                                StateMode::Unknown => {
                                    panic!("This should never happen");
                                }
                            },
                            Err(e) => Err(e),
                        }),
                )),
                Err(e) => Err(e),
            }
        } else {
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
    }

    /// Evaluates the heuristic for a given state, returning an iterator over the results.
    /// This method is used in multiqueue search algorithms
    /// If the heuristic is an HRL, it collects states to evaluate in batches.
    pub fn eval_gen_container<'a>(
        &'a self,
        states: &'a Vec<Rc<RefCell<StateContainer>>>,
        ss: &'a SearchSpace,
    ) -> PyResult<Box<dyn Iterator<Item = PyResult<(usize, Option<f64>)>> + 'a>> {
        if self.hrl.is_some() {
            let mut states_mapping = mk_unknown_state_mode_vec(states.len());
            let mut vectors_to_eval: Vec<Vec<f32>> = Vec::new();
            let mut sym_heuristics_to_eval: Vec<f64> = Vec::new();
            for (i, cstate) in states.iter().enumerate() {
                let state = &(cstate.borrow().state);
                states_mapping[i] = self.mk_state_mode(
                    ss,
                    state,
                    &mut vectors_to_eval,
                    &mut sym_heuristics_to_eval,
                );
            }
            let computed_heuristics = self
                .hrl
                .as_ref()
                .unwrap()
                .eval_vector(vectors_to_eval, sym_heuristics_to_eval);
            match computed_heuristics {
                Ok(vc) => {
                    let mut final_res = Vec::new();
                    for (i, case) in states_mapping.into_iter().enumerate() {
                        match case {
                            StateMode::Cached(x) => final_res.push(Ok((i, x))),
                            StateMode::Error(x) => final_res.push(Err(x)),
                            StateMode::ToEval(idx) => {
                                if self.cache_value_in_state {
                                    states[i]
                                        .borrow()
                                        .state
                                        .heuristic_cache
                                        .lock()
                                        .unwrap()
                                        .insert(self.name().to_string(), Some(vc[idx]));
                                }
                                final_res.push(Ok((i, Some(vc[idx]))))
                            }
                            StateMode::Unknown => {
                                panic!("This should never happen");
                            }
                        }
                    }
                    Ok(Box::new(final_res.into_iter()))
                }
                Err(e) => Err(e),
            }
        } else {
            return Ok(Box::new(states.iter().enumerate().map(|(i, state)| {
                let h_value = self.eval(&state.borrow().state, ss);
                match h_value {
                    Ok(x) => Ok((i, x)),
                    Err(e) => Err(e),
                }
            })));
        }
    }

    /// Creates a StateMode for the given state, possibly extending the
    /// vectors_to_eval and sym_heuristics_to_eval in case an evaluation of the
    /// state is needed
    fn mk_state_mode(
        &self,
        ss: &SearchSpace,
        state: &State,
        vectors_to_eval: &mut Vec<Vec<f32>>,
        sym_heuristics_to_eval: &mut Vec<f64>,
    ) -> StateMode {
        if self.cache_value_in_state {
            let heuristic_cache = state.heuristic_cache.lock().unwrap();
            if let Some(h_value) = heuristic_cache.get(&self.name()) {
                return StateMode::Cached(*h_value);
            }
        }

        match ss.goal_reached(state, None) {
            Ok(true) => {
                if self.cache_value_in_state {
                    state
                        .heuristic_cache
                        .lock()
                        .unwrap()
                        .insert(self.name().to_string(), Some(0.0));
                }
                return StateMode::Cached(Some(0.0));
            }
            Ok(false) => {
                let h_value = self.hrl.as_ref().unwrap().eval_hsym(state, ss);
                match h_value {
                    Ok(x) => {
                        if let Some(hval) = x {
                            let rv = self.hrl.as_ref().unwrap().get_vector(state, ss);
                            match rv {
                                Ok(v) => {
                                    vectors_to_eval.push(v);
                                    sym_heuristics_to_eval.push(hval);
                                    return StateMode::ToEval(vectors_to_eval.len() - 1);
                                }
                                Err(e) => {
                                    return StateMode::Error(e);
                                }
                            }
                        } else {
                            if self.cache_value_in_state {
                                state
                                    .heuristic_cache
                                    .lock()
                                    .unwrap()
                                    .insert(self.name().to_string(), None);
                            }
                            return StateMode::Cached(None);
                        }
                    }
                    Err(e) => {
                        return StateMode::Error(e);
                    }
                }
            }
            Err(e) => {
                return StateMode::Error(e);
            }
        }
    }
}

pub struct HRL {
    ss: CoreStateEncoder,
    goals_vec: Vec<f32>,
    constants_vec: Vec<f32>,
    h_sym: Arc<Option<Heuristic>>,
    callable: PyObject,
    name: String,
}

impl HRL {
    fn new(
        ss: &CoreStateEncoder,
        goals_vec: Vec<f32>,
        constants_vec: Vec<f32>,
        callable: PyObject,
        h_sym: Option<Heuristic>,
        name: &str,
    ) -> PyResult<Self> {
        Ok(HRL {
            ss: ss.clone(),
            goals_vec,
            constants_vec,
            h_sym: Arc::new(h_sym),
            callable,
            name: String::from(name),
        })
    }

    fn eval_hsym(&self, state: &State, ss: &SearchSpace) -> PyResult<Option<f64>> {
        if let Some(h) = self.h_sym.as_ref() {
            h.eval(state, ss)
        } else {
            Ok(Some(-1.0))
        }
    }

    pub fn eval(&self, state: &State, ss: &SearchSpace) -> PyResult<Option<f64>> {
        if ss.goal_reached(&state, None)? {
            return Ok(Some(0.0));
        }
        let enc = self.get_vector(state, ss)?;
        let h_val = match self.eval_hsym(state, ss)? {
            Some(v) => v,
            None => {
                return Ok(None);
            }
        };
        let vec_res = self.eval_vector(vec![enc], vec![h_val])?;
        if vec_res.len() == 1 {
            return Ok(Some(vec_res[0]));
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Expected 1 value, got {}",
                vec_res.len()
            )));
        }
    }

    fn get_vector(&self, state: &State, ss: &SearchSpace) -> Result<Vec<f32>, PyErr> {
        let mut enc: Vec<f32> = Vec::new();
        enc.extend(self.ss.get_fluents_as_vector(state)?);
        enc.extend(self.ss.get_running_actions_as_vector(state)?);
        enc.extend(self.constants_vec.iter());
        enc.extend(self.goals_vec.iter());
        enc.extend(self.ss.get_tn_as_vector(state, ss)?);
        Ok(enc)
    }

    pub fn eval_vector(
        &self,
        vectors_to_eval: Vec<Vec<f32>>,
        sym_heuristics_to_eval: Vec<f64>,
    ) -> PyResult<Vec<f64>> {
        if vectors_to_eval.is_empty() {
            Ok(vec![])
        } else {
            Python::with_gil(|py| {
                let args = PyTuple::new(
                    py,
                    &[
                        vectors_to_eval.into_pyobject(py)?,
                        sym_heuristics_to_eval.into_pyobject(py)?,
                    ],
                )?;
                let r = self.callable.call(py, args, None)?;
                Ok(r.extract(py)?)
            })
        }
    }

    pub fn name(&self) -> String {
        self.name.clone()
    }
}

impl Clone for HRL {
    fn clone(&self) -> Self {
        Python::with_gil(|py| HRL {
            ss: self.ss.clone(),
            goals_vec: self.goals_vec.clone(),
            constants_vec: self.constants_vec.clone(),
            h_sym: self.h_sym.clone(),
            callable: self.callable.clone_ref(py),
            name: self.name.clone(),
        })
    }
}

#[derive(Debug)]
pub struct CustomHeuristic {
    callable: PyObject,
}

impl CustomHeuristic {
    fn new(callable: PyObject) -> PyResult<Self> {
        Ok(CustomHeuristic { callable })
    }

    pub fn eval(&self, state: &State) -> PyResult<Option<f64>> {
        Python::with_gil(|py| {
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
        Python::with_gil(|py| CustomHeuristic {
            callable: self.callable.clone_ref(py),
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
struct Operator {
    action: String,
    conditions: Vec<Expression>,
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

fn is_numeric_condition(cond: &Vec<ExpressionNode>) -> bool {
    if let Some(e) = cond.last() {
        if let ExpressionNode::Bool(_) = e {
            return false;
        } else if let ExpressionNode::Fluent(_) = e {
            return false;
        } else if let ExpressionNode::Not(i) = e {
            if let ExpressionNode::Fluent(_) = cond[*i] {
                return false;
            }
        } else if let ExpressionNode::Equals(i1, i2) = e {
            if let ExpressionNode::Fluent(_) = cond[*i1] {
                if let ExpressionNode::Object(_) = cond[*i2] {
                    return false;
                }
            }
        }
    } else {
        return false;
    }
    true
}

#[derive(Clone, Debug)]
pub struct DeleteRelaxationHeuristic {
    events: HashMap<String, usize>,
    goals: Vec<Expression>,
    extra_fluents: HashMap<String, Vec<Expression>>,
    extra_goals: Vec<Expression>,
    operators: Vec<Operator>,
    precondition_of: HashMap<Expression, Vec<OperatorID>>,
    empty_pre_operators: HashSet<OperatorID>,
    numeric_conds: HashSet<Expression>,
    heuristic_kind: HeuristicKind,
    ordered_fluents: Vec<String>,
    ordered_actions: Vec<String>,
    internal_caching: Arc<Mutex<Option<HashMap<Vec<ExpressionNode>, Option<f64>>>>>,
    expression_manager: Arc<Mutex<ExpressionManager>>,
}

impl DeleteRelaxationHeuristic {
    fn new(
        fluents: HashMap<String, String>,
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

        for (a, le) in events.iter() {
            let mut a_extra_fluents: Vec<Expression> = Vec::new();
            let mut cond: Vec<ExpressionNode> = vec![ExpressionNode::Fluent(format!(
                "__f_{}_{}",
                a,
                le.len() - 1
            ))];
            extra_goals.push(expression_manager.put(&cond));
            for (i, (_, e)) in le.iter().enumerate() {
                let mut effects: Vec<Expression> = Vec::new();
                let mut conditions: Vec<Expression> = Vec::new();
                let f = format!("__f_{}_{}", a, i);
                a_extra_fluents
                    .push(expression_manager.put(&vec![ExpressionNode::Fluent(f.to_string())]));
                effects.push(expression_manager.put(&vec![ExpressionNode::Fluent(f.to_string())]));
                for eff in e.effects.iter() {
                    let t = fluents[&eff.fluent].to_string();
                    if t == "bool" {
                        if eff.value.len() == 1 {
                            if let ExpressionNode::Bool(value) = eff.value[0] {
                                if value {
                                    effects.push(expression_manager.put(&vec![
                                        ExpressionNode::Fluent(eff.fluent.to_string()),
                                    ]));
                                } else {
                                    effects.push(expression_manager.put(&vec![
                                        ExpressionNode::Fluent(eff.fluent.to_string()),
                                        make_operator("not".to_string(), vec![0])?,
                                    ]));
                                }
                            } else {
                                effects
                                    .push(expression_manager.put(&vec![ExpressionNode::Fluent(
                                        eff.fluent.to_string(),
                                    )]));
                                effects.push(expression_manager.put(&vec![
                                    ExpressionNode::Fluent(eff.fluent.to_string()),
                                    make_operator("not".to_string(), vec![0])?,
                                ]));
                            }
                        } else {
                            effects.push(
                                expression_manager
                                    .put(&vec![ExpressionNode::Fluent(eff.fluent.to_string())]),
                            );
                            effects.push(expression_manager.put(&vec![
                                ExpressionNode::Fluent(eff.fluent.to_string()),
                                make_operator("not".to_string(), vec![0])?,
                            ]));
                        }
                    } else if t != "real" && t != "int" {
                        if eff.value.len() == 1 {
                            if let ExpressionNode::Object(_) = eff.value[0] {
                                effects.push(expression_manager.put(&vec![
                                    ExpressionNode::Fluent(eff.fluent.to_string()),
                                    eff.value[0].clone(),
                                    make_operator("==".to_string(), vec![0, 1])?,
                                ]));
                            } else {
                                for o in objects[&t].iter() {
                                    effects.push(expression_manager.put(&vec![
                                        ExpressionNode::Fluent(eff.fluent.to_string()),
                                        ExpressionNode::Object(o.to_string()),
                                        make_operator("==".to_string(), vec![0, 1])?,
                                    ]));
                                }
                            }
                        } else {
                            for o in objects[&t].iter() {
                                effects.push(expression_manager.put(&vec![
                                    ExpressionNode::Fluent(eff.fluent.to_string()),
                                    ExpressionNode::Object(o.to_string()),
                                    make_operator("==".to_string(), vec![0, 1])?,
                                ]));
                            }
                        }
                    }
                }
                conditions.push(expression_manager.put(&cond));
                if e.conditions.len() > 0 && e.conditions != vec![ExpressionNode::Bool(true)] {
                    for sc in split_expression(&e.conditions)? {
                        conditions.push(expression_manager.put(&sc))
                    }
                }
                if !conditions.contains(&expression_manager.put(&vec![ExpressionNode::Bool(false)]))
                {
                    operators.push(Operator {
                        action: a.to_string(),
                        conditions,
                        effects,
                        cost: 1.0,
                    });
                }
                cond = vec![ExpressionNode::Fluent(f.to_string())];
            }
            extra_fluents.insert(a.to_string(), a_extra_fluents);
        }
        operators.sort_by(|a, b| a.action.cmp(&b.action));

        let expr_goals = split_expression(&goal.into_iter().map(|e| e.v).collect())?;
        let goals: Vec<Expression> = expr_goals
            .into_iter()
            .map(|e| expression_manager.put(&e))
            .collect();
        let mut precondition_of: HashMap<Expression, Vec<OperatorID>> = HashMap::new();
        let mut numeric_conds: HashSet<Expression> = HashSet::new();
        let mut empty_pre_operators: HashSet<OperatorID> = HashSet::new();
        for (idx_o, o) in operators.iter().enumerate() {
            if o.conditions.len() == 0
                || o.conditions == vec![expression_manager.put(&vec![ExpressionNode::Bool(true)])]
            {
                empty_pre_operators.insert(OperatorID::new(idx_o));
            }
            for c in o.conditions.iter() {
                if is_numeric_condition(expression_manager.force_get(c)) {
                    numeric_conds.insert(*c);
                } else {
                    if !precondition_of.contains_key(c) {
                        precondition_of.insert(*c, Vec::new());
                    }
                    precondition_of
                        .get_mut(c)
                        .unwrap()
                        .push(OperatorID::new(idx_o));
                }
            }
        }
        for c in goals.iter() {
            if is_numeric_condition(expression_manager.force_get(c)) {
                numeric_conds.insert(*c);
            }
        }

        let events_len: HashMap<String, usize> = events
            .iter()
            .map(|(a, ev)| (a.to_string(), ev.len()))
            .collect();

        let ordered_fluents: Vec<String> = fluents.iter().map(|(f, _)| f.clone()).collect();
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
            ordered_fluents,
            ordered_actions,
            internal_caching: Arc::new(Mutex::new(internal_caching)),
            expression_manager: Arc::new(Mutex::new(expression_manager)),
        };
        Ok(res)
    }

    pub fn eval(&self, state: &State) -> PyResult<Option<f64>> {
        let mut internal_caching = self.internal_caching.lock().unwrap();
        let mut expression_manager = self.expression_manager.lock().unwrap();
        let mut assignments_values: Vec<ExpressionNode> = if internal_caching.is_some() {
            Vec::with_capacity(self.ordered_fluents.len())
        } else {
            Vec::new()
        };
        if internal_caching.is_some() {
            for f in &self.ordered_fluents {
                assignments_values.push(state.assignments[f].clone());
            }
            for action in &self.ordered_actions {
                let r = match state.todo.get(action) {
                    Some((j, _)) => j.clone(),
                    None => 0,
                };
                assignments_values.push(ExpressionNode::Int(r.into()));
            }
            if let Some(res) = internal_caching.as_ref().unwrap().get(&assignments_values) {
                return Ok(res.clone());
            }
        }

        let mut costs: HashMap<Expression, f64> = HashMap::new();
        let mut lp: Vec<Expression> = Vec::new();
        let mut init_lp: Vec<Expression> = Vec::new();

        for (f, v) in state.assignments.iter() {
            let k = match v {
                ExpressionNode::Bool(value) => {
                    if *value {
                        vec![ExpressionNode::Fluent(f.to_string())]
                    } else {
                        vec![
                            ExpressionNode::Fluent(f.to_string()),
                            make_operator("not".to_string(), vec![0])?,
                        ]
                    }
                }
                _ => {
                    vec![
                        ExpressionNode::Fluent(f.to_string()),
                        v.clone(),
                        make_operator("==".to_string(), vec![0, 1])?,
                    ]
                }
            };
            init_lp.push(expression_manager.put(&k));
        }
        for k in init_lp.iter() {
            costs.insert(*k, 0.0);
            lp.push(*k);
        }

        for c in self.numeric_conds.iter() {
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
                            reached_by.insert(*k, oid);
                            new_costs.insert(k, c + o.cost);
                            lp.push(*k);
                        } else if ((new_cost_k.is_some() && *new_cost_k.unwrap() == c + o.cost)
                            || (new_cost_k.is_none() && *cost_k.unwrap() == c + o.cost))
                            && oid.id > reached_by[k].id
                        {
                            reached_by.insert(*k, oid);
                        }
                    }
                }
            }
            for (k, v) in new_costs.iter() {
                costs.insert(**k, *v);
            }
        }

        let h = self.cost(&self.goals, &costs);

        if h.is_none() {
            if internal_caching.is_some() {
                internal_caching
                    .as_mut()
                    .unwrap()
                    .insert(assignments_values, None);
            }
            return Ok(None);
        }

        if matches!(
            self.heuristic_kind,
            HeuristicKind::HADD | HeuristicKind::HMAX
        ) {
            match self.cost(&self.extra_goals, &costs) {
                Some(v) => {
                    let res = if let HeuristicKind::HMAX = self.heuristic_kind {
                        f64::max(h.unwrap(), v)
                    } else {
                        h.unwrap() + v
                    };

                    if internal_caching.is_some() {
                        internal_caching
                            .as_mut()
                            .unwrap()
                            .insert(assignments_values, Some(res));
                    }
                    return Ok(Some(res));
                }
                None => {
                    if internal_caching.is_some() {
                        internal_caching
                            .as_mut()
                            .unwrap()
                            .insert(assignments_values, None);
                    }
                    return Ok(None);
                }
            };
        }

        let mut res = 0.0;
        for (a, (j, _)) in state.todo.iter() {
            res += (self.events[a] - j) as f64;
        }

        if let Some(hv) = h {
            if hv == 0.0 {
                if internal_caching.is_some() {
                    internal_caching
                        .as_mut()
                        .unwrap()
                        .insert(assignments_values, Some(res));
                }
                return Ok(Some(res));
            }
        }

        let mut relaxed_plan = HashSet::new();
        let mut stack: Vec<&Expression> = self.goals.iter().collect();
        while stack.len() > 0 {
            let g = stack.pop().unwrap();
            if let Some(oid) = reached_by.get(g) {
                let o: &Operator = &self.operators[oid.id];
                relaxed_plan.insert(o.action.to_string());
                for c in o.conditions.iter() {
                    stack.push(c);
                }
            }
        }
        for a in relaxed_plan.iter() {
            if !state.todo.contains_key(a) {
                res += self.events[a] as f64;
            }
        }

        if internal_caching.is_some() {
            internal_caching
                .as_mut()
                .unwrap()
                .insert(assignments_values, Some(res));
        }
        Ok(Some(res))
    }

    fn cost(&self, exp: &Vec<Expression>, costs: &HashMap<Expression, f64>) -> Option<f64> {
        let mut res = 0.0;
        for g in exp.iter() {
            let c = costs.get(g);
            if let Some(cost) = c {
                if let HeuristicKind::HMAX = self.heuristic_kind {
                    res = f64::max(res, *cost);
                } else {
                    res += cost;
                }
            } else {
                return None;
            }
        }
        Some(res)
    }

    fn name(&self) -> String {
        match self.heuristic_kind {
            HeuristicKind::HFF => String::from("hff"),
            HeuristicKind::HADD => String::from("hadd"),
            HeuristicKind::HMAX => String::from("hmax"),
        }
    }
}

#[derive(Clone, Debug)]
pub struct HMaxNumeric {
    goals: Vec<Vec<ExpressionNode>>,
    goal_expressions: Vec<Expression>,
    extra_fluents: HashMap<String, Vec<Vec<ExpressionNode>>>,
    all_fluents: Vec<String>,
    operators: Vec<OperatorHmax>,
    operator_conditions_fluents: Vec<HashSet<String>>,
    operator_effects_fluents: Vec<HashSet<String>>,
    ordered_fluents: Vec<String>,
    ordered_actions: Vec<String>,
    internal_caching: Arc<Mutex<Option<HashMap<Vec<ExpressionNode>, Option<f64>>>>>,
}

impl HMaxNumeric {
    fn new(
        fluents: HashMap<String, String>,
        events: HashMap<String, Vec<(Timing, Event)>>,
        goal: Vec<PyExpressionNode>,
        internal_caching: bool,
    ) -> PyResult<Self> {
        let mut operators = Vec::new();
        let mut extra_fluents = HashMap::new();
        let mut all_fluents = Vec::new();
        let mut extra_goals = Vec::new();
        let mut expression_manager = ExpressionManager::new();

        for (a, le) in events.iter() {
            let mut a_extra_fluents = Vec::new();
            let mut cond: Vec<ExpressionNode> = vec![ExpressionNode::Fluent(format!(
                "__f_{}_{}",
                a,
                le.len() - 1
            ))];
            extra_goals.push(cond.clone());
            for (i, (_, e)) in le.iter().enumerate() {
                let mut effects = Vec::new();
                let mut conditions = Vec::new();
                let f = format!("__f_{}_{}", a, i);
                all_fluents.push(f.clone());
                a_extra_fluents.push(vec![ExpressionNode::Fluent(f.to_string())]);
                effects.push(Effect {
                    fluent: f.clone(),
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
                cond = vec![ExpressionNode::Fluent(f.to_string())];
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

        for (f, _v) in &fluents {
            all_fluents.push(f.clone());
        }
        let ordered_fluents: Vec<String> = fluents.iter().map(|(f, _)| f.clone()).collect();
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
            all_fluents,
            operators,
            operator_conditions_fluents,
            operator_effects_fluents,
            ordered_fluents,
            ordered_actions,
            internal_caching: Arc::new(Mutex::new(internal_caching)),
        };
        Ok(res)
    }

    fn extract_fluents<'a>(&self, exp: &'a Vec<ExpressionNode>) -> Vec<&'a String> {
        let mut exp_fluents = Vec::new();
        for exp_node in exp {
            if let ExpressionNode::Fluent(f) = exp_node {
                exp_fluents.push(f);
            }
        }

        exp_fluents
    }

    fn possible_values(
        &self,
        exp: &Vec<ExpressionNode>,
        assignments: &HashMap<&String, HashSet<ExpressionNode>>,
        exp_fluents: Option<&Vec<&String>>,
    ) -> Vec<ExpressionNode> {
        let exp_fluents_extracted;
        let exp_fluents = match exp_fluents {
            Some(fluents) => fluents,
            None => {
                exp_fluents_extracted = self.extract_fluents(exp);
                &exp_fluents_extracted
            }
        };
        let mut values = Vec::new();
        for f in exp_fluents {
            values.push(&assignments[f]);
        }

        let mut possible_values = Vec::new();
        let mut tmp_state = State {
            assignments: HashMap::with_capacity(exp_fluents.len()),
            temporal_network: None,
            todo: HashMap::new(),
            active_conditions: HashMultiSet::new(),
            g: 0.0,
            path: None,
            heuristic_cache: Mutex::new(HashMap::new()),
        };
        for state_values in values
            .iter()
            .map(|fluent_values| fluent_values.iter())
            .multi_cartesian_product()
        {
            tmp_state.assignments.clear();
            for (i, f) in exp_fluents.iter().enumerate() {
                tmp_state
                    .assignments
                    .insert(f.to_string(), (*state_values[i]).clone());
            }

            possible_values.push(internal_evaluate(exp, &tmp_state).unwrap());
        }

        return possible_values;
    }

    fn exp_can_be_true(
        &self,
        exp: &Vec<ExpressionNode>,
        exp_id: Expression,
        assignments: &HashMap<&String, HashSet<ExpressionNode>>,
        assignments_changes: &HashSet<&String>,
        cache_can_be_true: &mut HashMap<Expression, bool>,
    ) -> bool {
        let mut exp_fluents = Vec::new();
        if cache_can_be_true.contains_key(&exp_id) {
            if cache_can_be_true[&exp_id] {
                return true;
            }

            exp_fluents = self.extract_fluents(exp);
            let exp_fluents_set: HashSet<&String> = exp_fluents.iter().copied().collect();
            if exp_fluents_set.is_disjoint(assignments_changes) {
                return false;
            }
        }

        let possible_values = if exp_fluents.len() > 0 {
            self.possible_values(exp, assignments, Some(&exp_fluents))
        } else {
            self.possible_values(exp, assignments, None)
        };

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
        assignments: &HashMap<&String, HashSet<ExpressionNode>>,
        assignments_changes: &HashSet<&String>,
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

    fn eval(&self, state: &State) -> PyResult<Option<f64>> {
        let mut internal_caching = self.internal_caching.lock().unwrap();
        let mut assignments_values: Vec<ExpressionNode> = if internal_caching.is_some() {
            Vec::with_capacity(self.ordered_fluents.len())
        } else {
            Vec::new()
        };
        if internal_caching.is_some() {
            for f in &self.ordered_fluents {
                assignments_values.push(state.assignments[f].clone());
            }
            for action in &self.ordered_actions {
                let r = match state.todo.get(action) {
                    Some((j, _)) => j.clone(),
                    None => 0,
                };
                assignments_values.push(ExpressionNode::Int(r.into()));
            }
            if let Some(res) = internal_caching.as_ref().unwrap().get(&assignments_values) {
                return Ok(res.clone());
            }
        }

        let mut assignments: HashMap<&String, HashSet<ExpressionNode>> = HashMap::new();
        // add state assignments to assignments
        for (f, v) in &state.assignments {
            assignments.insert(f, HashSet::from([v.clone()]));
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
                    assignments.insert(f, HashSet::from([ExpressionNode::Bool(i == idx)]));
                }
            }
        }

        let mut cache_can_be_true: HashMap<Expression, bool> = HashMap::new();
        let mut applied_operators = vec![false; self.operators.len()];
        let mut assignments_changes: HashSet<&String> = self.all_fluents.iter().collect();
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
                if internal_caching.is_some() {
                    internal_caching
                        .as_mut()
                        .unwrap()
                        .insert(assignments_values, Some(depth as f64));
                }
                return Ok(Some(depth as f64));
            }

            let mut new_assignments: HashMap<&String, HashSet<ExpressionNode>> = HashMap::new();
            for (i, operator) in self.operators.iter().enumerate() {
                if applied_operators[i] {
                    // operator already applied
                    let eff_fluents: HashSet<&String> =
                        self.operator_effects_fluents[i].iter().collect();
                    if assignments_changes.is_disjoint(&eff_fluents) {
                        // no changes in the effect fluents
                        continue;
                    }
                } else if assignments_changes.is_disjoint(
                    &self.operator_conditions_fluents[i]
                        .iter()
                        .collect::<HashSet<&String>>(),
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
                    let possible_values = self.possible_values(&effect.value, &assignments, None);
                    new_assignments
                        .entry(&effect.fluent)
                        .or_insert_with(HashSet::new)
                        .extend(possible_values);
                }
            }

            // update assignments
            assignments_changes.clear();
            for (fluent, new_vv) in new_assignments {
                let vv = assignments.entry(fluent).or_insert_with(HashSet::new);
                let prev_len = vv.len();
                for v in new_vv {
                    vv.insert(v);
                }
                if vv.len() > prev_len {
                    assignments_changes.insert(fluent);
                }
            }

            depth += 1;
        }

        if internal_caching.is_some() {
            internal_caching
                .as_mut()
                .unwrap()
                .insert(assignments_values, None);
        }
        Ok(None)
    }

    pub fn name(&self) -> String {
        String::from("hmax_numeric")
    }
}
