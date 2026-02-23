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

use im::Vector;
use multiset::HashMultiSet;
use num_rational::BigRational;
use pyo3::{exceptions::PyException, prelude::*};
use rustc_hash::{FxBuildHasher, FxHashMap, FxHashSet};
use std::{sync::Mutex, vec::Vec};

use super::expressions::*;
use super::expressions_utils::*;
use super::search_state::*;
use super::stn::DeltaSTN;
use super::structures::*;
use super::tn_interpreter::TNInterpreter;
use super::utils::*;

pub trait SearchSpaceTrait {
    fn is_temporal(&self) -> bool;
    fn tn_interpreter(&self) -> &TNInterpreter;
    fn initial_state(&self, initial_state: Option<Vec<PyExpressionNode>>) -> PyResult<State>;
    fn get_successor_state(&self, state: &State, action: Action) -> PyResult<Option<State>>;
    fn get_successor_states_iter<'a>(
        &'a self,
        state: &'a State,
    ) -> impl Iterator<Item = PyResult<State>> + 'a;
    fn get_successor_states(&self, state: &State) -> PyResult<Vec<State>> {
        let mut res = Vec::new();
        for rs in self.get_successor_states_iter(state) {
            res.push(rs?);
        }
        Ok(res)
    }
    fn reset(&self);
    fn goal_reached(&self, state: &State, goal: Option<Vec<PyExpressionNode>>) -> PyResult<bool>;
    fn subgoals_sat(
        &self,
        state: &State,
        goal: Option<Vec<PyExpressionNode>>,
    ) -> PyResult<Vec<Vec<PyExpressionNode>>>;
    fn build_plan(
        &self,
        state: &State,
    ) -> PyResult<Vec<(Option<BigRational>, Action, Option<BigRational>)>>;
}

#[derive(Debug)]
struct MutexChecker {
    mutex: Mutex<FxHashMap<((Action, usize), (Action, usize)), bool>>,
}

impl MutexChecker {
    fn new() -> Self {
        MutexChecker {
            mutex: Mutex::new(FxHashMap::with_hasher(FxBuildHasher::default())),
        }
    }

    fn check(
        &self,
        events_pair: &((Action, usize), (Action, usize)),
        event_fluents: &Vec<
            Vec<(
                FxHashSet<usize>,
                FxHashSet<usize>,
                FxHashSet<usize>,
                FxHashSet<usize>,
                FxHashSet<usize>,
            )>,
        >,
    ) -> bool {
        let ((a1, i1), (a2, i2)) = events_pair;
        if a1 == a2 {
            return true;
        }

        let mut mutex = self.mutex.lock().unwrap();
        if let Some(are_mutex) = mutex.get(&events_pair) {
            return *are_mutex;
        }

        let (_, a_e, a_pe, _, _) = &event_fluents[a1.idx][*i1];
        let (b_p, b_e, _, _, _) = &event_fluents[a2.idx][*i2];
        let are_mutex = !(b_p.is_disjoint(a_e) && a_pe.is_disjoint(b_e));
        mutex.insert(*events_pair, are_mutex);
        are_mutex
    }
}

#[derive(Debug)]
struct PrecedenceChecker {
    precedence: Mutex<FxHashMap<((Action, usize), (Action, usize)), bool>>,
}

impl PrecedenceChecker {
    fn new() -> Self {
        PrecedenceChecker {
            precedence: Mutex::new(FxHashMap::with_hasher(FxBuildHasher::default())),
        }
    }

    fn check(
        &self,
        events_pair: &((Action, usize), (Action, usize)),
        event_fluents: &Vec<
            Vec<(
                FxHashSet<usize>,
                FxHashSet<usize>,
                FxHashSet<usize>,
                FxHashSet<usize>,
                FxHashSet<usize>,
            )>,
        >,
    ) -> bool {
        let ((a1, i1), (a2, i2)) = events_pair;
        if a1 == a2 {
            return true;
        }

        let mut precedence = self.precedence.lock().unwrap();
        if let Some(res) = precedence.get(&events_pair) {
            return *res;
        }

        let (_, a_e, _, _, a_ec) = &event_fluents[a1.idx][*i1];
        let (_, b_e, _, b_sc, _) = &event_fluents[a2.idx][*i2];
        let res = !(a_e.is_disjoint(b_sc) && b_e.is_disjoint(a_ec));
        precedence.insert(*events_pair, res);
        res
    }
}

fn get_fluents<'a>(expr: &'a Vec<ExpressionNode>) -> impl Iterator<Item = usize> + 'a {
    expr.iter().filter_map(|node| match node {
        ExpressionNode::Fluent(fluent) => Some(*fluent),
        _ => None,
    })
}

#[pyclass(name = "SearchSpace", frozen)]
#[derive(Debug)]
pub struct SearchSpace {
    actions_duration: Vec<Option<(Vec<ExpressionNode>, Vec<ExpressionNode>, bool, bool)>>,
    events: FxHashMap<Action, Vec<(Timing, Event)>>,
    actions: Vec<Action>,
    compression_safe_actions: Option<Vec<bool>>,
    event_fluents: Vec<
        Vec<(
            FxHashSet<usize>,
            FxHashSet<usize>,
            FxHashSet<usize>,
            FxHashSet<usize>,
            FxHashSet<usize>,
        )>,
    >,
    mutex: MutexChecker,
    precedence: PrecedenceChecker,
    action_objects: Option<Vec<Vec<String>>>,
    obj_to_prev_actions_map: Option<FxHashMap<String, FxHashSet<Action>>>,
    initial_state: Option<Vec<ExpressionNode>>,
    goal: Option<Vec<ExpressionNode>>,
    tn_interpreter: TNInterpreter,
    epsilon: f32,
    epsilon_rational: BigRational,
    is_temporal: bool,
    counter: Mutex<u32>,
}

#[pymethods]
impl SearchSpace {
    #[new]
    #[pyo3(signature = (actions_duration, events, actions, compression_safe_actions, action_objects, obj_to_prev_actions_map, initial_state=None, goal=None, epsilon=None))]
    fn new(
        actions_duration: Vec<Option<(Vec<PyExpressionNode>, Vec<PyExpressionNode>, bool, bool)>>,
        events: FxHashMap<Action, Vec<(Timing, Event)>>,
        actions: Vec<Action>,
        compression_safe_actions: Option<Vec<bool>>,
        action_objects: Option<Vec<Vec<String>>>,
        obj_to_prev_actions_map: Option<FxHashMap<String, FxHashSet<Action>>>,
        initial_state: Option<Vec<PyExpressionNode>>,
        goal: Option<Vec<PyExpressionNode>>,
        #[pyo3(from_py_with = get_option_big_rational)] epsilon: Option<BigRational>,
    ) -> PyResult<Self> {
        let is_temporal = actions_duration.iter().any(|value| !value.is_none());
        let converted_actions_duration: Vec<
            Option<(Vec<ExpressionNode>, Vec<ExpressionNode>, bool, bool)>,
        > = actions_duration
            .into_iter()
            .map(|value| {
                value.map(|(vec1, vec2, b1, b2)| {
                    (
                        vec1.into_iter().map(|e| e.v).collect(),
                        vec2.into_iter().map(|e| e.v).collect(),
                        b1,
                        b2,
                    )
                })
            })
            .collect();

        let mut event_fluents = vec![Vec::new(); actions.len()];
        for (a, le) in &events {
            for (_, e) in le {
                let mut a_p: FxHashSet<usize> = get_fluents(&e.conditions).collect();
                a_p.extend(e.effects.iter().flat_map(|eff| get_fluents(&eff.value)));
                let a_e: FxHashSet<usize> = e.effects.iter().map(|eff| eff.fluent).collect();
                let a_pe: FxHashSet<usize> = a_p.union(&a_e).copied().collect();
                let a_sc: FxHashSet<usize> = e
                    .start_conditions
                    .iter()
                    .flat_map(|c| get_fluents(c))
                    .collect();
                let a_ec: FxHashSet<usize> = e
                    .end_conditions
                    .iter()
                    .flat_map(|c| get_fluents(c))
                    .collect();
                event_fluents[a.idx].push((a_p, a_e, a_pe, a_sc, a_ec));
            }
        }

        let tn_interpreter = TNInterpreter::new(&actions, &events);

        let res = SearchSpace {
            actions_duration: converted_actions_duration,
            events: events,
            actions: actions,
            compression_safe_actions: compression_safe_actions,
            event_fluents: event_fluents,
            mutex: MutexChecker::new(),
            precedence: PrecedenceChecker::new(),
            action_objects: action_objects,
            obj_to_prev_actions_map: obj_to_prev_actions_map,
            initial_state: initial_state
                .map(|inner_vec| inner_vec.into_iter().map(|v| v.v).collect()),
            goal: goal.map(|inner_vec| inner_vec.into_iter().map(|e| e.v).collect()),
            tn_interpreter: tn_interpreter,
            epsilon: match &epsilon {
                Some(x) => rational_to_f32(x),
                None => 0.01,
            },
            epsilon_rational: match epsilon {
                Some(x) => x,
                None => mk_rational(1, 100),
            },
            is_temporal: is_temporal,
            counter: Mutex::new(0),
        };
        Ok(res)
    }

    #[getter]
    #[pyo3(name = "is_temporal")]
    fn py_is_temporal(&self) -> bool {
        self.is_temporal()
    }

    #[pyo3(name = "reset")]
    fn py_reset(&self) {
        self.reset();
    }

    #[pyo3(name = "initial_state", signature = (initial_state=None))]
    pub fn py_initial_state(
        &self,
        initial_state: Option<Vec<PyExpressionNode>>,
    ) -> PyResult<State> {
        self.initial_state(initial_state)
    }

    #[pyo3(name = "get_successor_states")]
    pub fn py_get_successor_states(&self, state: &State) -> PyResult<Vec<State>> {
        self.get_successor_states(state)
    }

    #[pyo3(name = "get_successor_state")]
    pub fn py_get_successor_state(&self, state: &State, action: Action) -> PyResult<Option<State>> {
        self.get_successor_state(state, action)
    }

    #[pyo3(name = "goal_reached", signature = (state, goal=None))]
    pub fn py_goal_reached(
        &self,
        state: &State,
        goal: Option<Vec<PyExpressionNode>>,
    ) -> PyResult<bool> {
        self.goal_reached(state, goal)
    }

    #[pyo3(name = "subgoals_sat", signature = (state, goal=None))]
    pub fn py_subgoals_sat(
        &self,
        state: &State,
        goal: Option<Vec<PyExpressionNode>>,
    ) -> PyResult<Vec<Vec<PyExpressionNode>>> {
        self.subgoals_sat(state, goal)
    }
}

impl SearchSpace {
    fn get_successor_state_with_compression(
        &self,
        state: &State,
        action: Action,
        enable_compression_safe_actions: bool,
    ) -> PyResult<Option<State>> {
        if let Some(events) = self.events.get(&action) {
            if let Some((index, id)) = state.todo.get(&action) {
                if let Some((_, e)) = events.get(*index) {
                    // Check if the event is applicable before creating the new state
                    if !self.is_sat(&e.conditions, state)? {
                        return Ok(None);
                    }

                    let mut new_state = state.clone_for_child();
                    new_state.g += 1.0;

                    if index + 1 >= events.len() {
                        new_state.todo.remove(&action);
                    } else {
                        new_state.todo.insert(action, (index + 1, id + 1));
                    }
                    if self.expand_event(state, &mut new_state, &e, index, id)? {
                        return Ok(Some(new_state));
                    }
                }
            } else {
                // Check if action is applicable before creating the new state
                if !self.is_sat(&events[0].1.conditions, state)? {
                    return Ok(None);
                }

                let mut new_state = state.clone_for_child();
                new_state.g += 1.0;
                if !self.open_action(state, &mut new_state, action, &events)? {
                    return Ok(None);
                }

                if enable_compression_safe_actions
                    && self
                        .compression_safe_actions
                        .as_ref()
                        .map_or(false, |is_compression_safe| is_compression_safe[action.idx])
                {
                    let mut id = new_state.todo.remove(&action).unwrap().1;
                    for index in 1..events.len() {
                        new_state.g += 1.0;
                        if !self.expand_event(
                            state,
                            &mut new_state,
                            &events[index].1,
                            &index,
                            &id,
                        )? {
                            return Ok(None);
                        }

                        id += 1;
                    }
                }
                return Ok(Some(new_state));
            }
        }
        Ok(None)
    }

    fn is_sat(&self, conditions: &Vec<ExpressionNode>, state: &State) -> PyResult<bool> {
        let sat = match internal_evaluate(conditions, state)? {
            ExpressionNode::Bool(v) => v,
            _ => {
                return Err(PyException::new_err(
                    "An action condition is not a boolean expression!",
                ))
            }
        };
        Ok(sat)
    }

    fn expand_event(
        &self,
        state: &State,
        new_state: &mut State,
        e: &Event,
        index: &usize,
        id: &u32,
    ) -> PyResult<bool> {
        new_state.path = PersistentList::append((e.action, e.pos, *id), &new_state.path);

        // check conditions is done before calling this method

        // check active conditions
        for c in new_state.active_conditions.iter() {
            let sat = match internal_evaluate(&c, state)? {
                ExpressionNode::Bool(v) => v,
                _ => {
                    return Err(PyException::new_err(
                        "An action condition is not a boolean expression!",
                    ))
                }
            };
            if !sat {
                return Ok(false);
            }
        }

        // remove end conditions
        for c in e.end_conditions.iter() {
            new_state.active_conditions.remove(&c);
        }

        // insert start conditions
        for c in e.start_conditions.iter() {
            new_state.active_conditions.insert(c.to_vec());
        }

        // apply effects
        for eff in e.effects.iter() {
            new_state.assignments[eff.fluent] = internal_evaluate(&eff.value, state)?;
        }

        // check active conditions
        for c in new_state.active_conditions.iter() {
            let sat = match internal_evaluate(&c, new_state)? {
                ExpressionNode::Bool(v) => v,
                _ => {
                    return Err(PyException::new_err(
                        "An action condition is not a boolean expression!",
                    ))
                }
            };
            if !sat {
                return Ok(false);
            }
        }

        if self.is_temporal {
            // Add temporal constraints between past or todo events and the current one
            let tn = new_state.temporal_network.as_mut().unwrap();
            let ev = self.tn_interpreter.get_event_id(e.action, e.pos, *id);
            for e2 in PersistentList::iter_rev(&state.path) {
                let ev2 = self.tn_interpreter.get_event_id(e2.0, e2.1, e2.2);
                let e_id = (e.action, *index);
                let e2_id = (e2.0, e2.1);
                if self.mutex.check(&(e_id, e2_id), &self.event_fluents) {
                    let b: f32 = -self.epsilon;
                    tn.add(&ev2, &ev, &b);
                } else {
                    tn.add(&ev2, &ev, &0.0);
                }
            }
            for (a, i) in new_state.todo.iter() {
                let mut id2 = i.1;
                for (j, (_, e2)) in self.events[a].iter().skip(i.0).enumerate() {
                    let e_id = (e.action, *index);
                    let e2_id = (a.clone(), j + i.0);
                    let ev2 = self.tn_interpreter.get_event_id(e2.action, e2.pos, id2);
                    if self.mutex.check(&(e_id, e2_id), &self.event_fluents) {
                        let b: f32 = -self.epsilon;
                        tn.add(&ev, &ev2, &b);
                    } else {
                        tn.add(&ev, &ev2, &0.0);
                    }
                    id2 += 1;
                }
            }
            if !tn.check() {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn open_action(
        &self,
        state: &State,
        new_state: &mut State,
        action: Action,
        events: &Vec<(Timing, Event)>,
    ) -> PyResult<bool> {
        if let (Some(action_objects), Some(obj_to_prev_actions_map)) =
            (&self.action_objects, &self.obj_to_prev_actions_map)
        {
            for obj in &action_objects[action.idx] {
                let prev_actions = match obj_to_prev_actions_map.get(obj) {
                    Some(actions) => actions,
                    None => continue,
                };

                if prev_actions.contains(&action) {
                    continue;
                }

                if !PersistentList::iter_rev(&state.path).any(|(a, _, _)| prev_actions.contains(a))
                {
                    return Ok(false);
                }
            }
        }

        let mut counter = self.counter.lock().unwrap();
        let mut id = counter.clone();
        if self.is_temporal {
            // Add temporal constraints between events of the action
            let tn = new_state.temporal_network.as_mut().unwrap();
            let start = self.tn_interpreter.get_action_id(action, true, *counter);
            let end = self.tn_interpreter.get_action_id(action, false, *counter);
            *counter += 1;
            let duration = self.actions_duration[action.idx].as_ref();
            let mut lb: f32 = 0.0;
            let mut ub: f32 = 0.0;
            if duration.is_some() {
                let d = duration.unwrap();
                lb = -rational_to_f32(&get_rational_from_expression_node(&internal_evaluate(
                    &d.0, state,
                )?)?);
                ub = rational_to_f32(&get_rational_from_expression_node(&internal_evaluate(
                    &d.1, state,
                )?)?);
                if d.2 {
                    lb -= self.epsilon;
                }
                if d.3 {
                    ub -= self.epsilon;
                }
            }
            tn.add(&start, &end, &lb);
            tn.add(&end, &start, &ub);
            id = *counter;
            for (t, e) in events.iter() {
                let ev = self.tn_interpreter.get_event_id(e.action, e.pos, *counter);
                let b1 = -rational_to_f32(&t.delay);
                let b2 = rational_to_f32(&t.delay);
                if t.is_from_start() {
                    tn.add(&start, &ev, &b1);
                    tn.add(&ev, &start, &b2);
                } else {
                    tn.add(&end, &ev, &b1);
                    tn.add(&ev, &end, &b2);
                }
                *counter += 1;
            }
            if events.len() > 1 {
                new_state.todo.insert(action, (1, id + 1));
            }
        }
        self.expand_event(state, new_state, &events[0].1, &0, &id)
    }
}

impl SearchSpaceTrait for SearchSpace {
    fn is_temporal(&self) -> bool {
        self.is_temporal
    }

    fn tn_interpreter(&self) -> &TNInterpreter {
        &self.tn_interpreter
    }

    fn reset(&self) {
        // DO nothing :)
    }

    fn initial_state(&self, initial_state: Option<Vec<PyExpressionNode>>) -> PyResult<State> {
        let init: Vector<ExpressionNode> = match initial_state {
            Some(v) => v.iter().map(|v| v.v.clone()).collect(),
            None => match &self.initial_state {
                Some(v) => Vector::from(v),
                None => {
                    return Err(PyException::new_err(
                        "The initial state must be defined somewhere!",
                    ));
                }
            },
        };
        let tn: Option<DeltaSTN<u64, f32>> = match self.is_temporal {
            true => Some(DeltaSTN::new(self.epsilon / 1000.0)),
            false => None,
        };
        Ok(State {
            assignments: init,
            temporal_network: tn,
            todo: FxHashMap::with_hasher(FxBuildHasher::default()),
            active_conditions: HashMultiSet::new(),
            g: 0.0,
            path: PersistentList::new(),
            heuristic_cache: Mutex::new(FxHashMap::with_hasher(FxBuildHasher::default())),
        })
    }

    fn get_successor_states_iter<'a>(
        &'a self,
        state: &'a State,
    ) -> impl Iterator<Item = PyResult<State>> + 'a {
        self.actions
            .iter()
            .filter_map(|action| self.get_successor_state(state, *action).transpose())
    }

    fn get_successor_state(&self, state: &State, action: Action) -> PyResult<Option<State>> {
        self.get_successor_state_with_compression(state, action, true)
    }

    fn goal_reached(&self, state: &State, goal: Option<Vec<PyExpressionNode>>) -> PyResult<bool> {
        if !state.todo.is_empty() {
            return Ok(false);
        }
        let goal = goal.map(|g| g.into_iter().map(|e| e.v).collect());
        let g = match &goal {
            Some(v) => v,
            None => match &self.goal {
                Some(v) => v,
                None => {
                    return Err(PyException::new_err("The goal must be defined somewhere!"));
                }
            },
        };
        match internal_evaluate(&g, state)? {
            ExpressionNode::Bool(v) => Ok(v),
            _ => Err(PyException::new_err(
                "The goal is not a boolean expression!",
            )),
        }
    }

    fn subgoals_sat(
        &self,
        state: &State,
        goal: Option<Vec<PyExpressionNode>>,
    ) -> PyResult<Vec<Vec<PyExpressionNode>>> {
        let goals = match goal {
            Some(v) => split_expression(&v.into_iter().map(|e| e.v).collect())?,
            None => match &self.goal {
                Some(v) => split_expression(&v)?,
                None => {
                    return Err(PyException::new_err("The goal must be defined somewhere!"));
                }
            },
        };
        let mut res: FxHashSet<_> = FxHashSet::with_hasher(FxBuildHasher::default());
        for g in goals {
            if internal_evaluate(&g, state)? == ExpressionNode::Bool(true) {
                res.insert(g.into_iter().map(|v| PyExpressionNode { v }).collect());
            }
        }
        Ok(res.into_iter().collect())
    }

    fn build_plan(
        &self,
        state: &State,
    ) -> PyResult<Vec<(Option<BigRational>, Action, Option<BigRational>)>> {
        let all_path = PersistentList::to_vec(&state.path)
            .into_iter()
            .map(|(a, _, _)| a);

        if !self.is_temporal {
            return Ok(all_path.map(|a| (None, a.clone(), None)).collect());
        }

        let mut tn = DeltaSTN::new(mk_rational(0, 1));
        let mut todo: FxHashMap<Action, (usize, u32)> =
            FxHashMap::with_hasher(FxBuildHasher::default());
        let mut path: Vec<(Event, u32)> = Vec::new();
        let mut counter = 0;
        let mut state = self.initial_state(None)?;
        for action in all_path {
            state = self
                .get_successor_state_with_compression(&state, *action, false)?
                .unwrap();
            if let Some(events) = self.events.get(action).cloned() {
                if let Some((index, id)) = todo.get(action).cloned() {
                    if let Some((_, e)) = events.get(index) {
                        if index + 1 >= events.len() {
                            todo.remove(action);
                        } else {
                            todo.insert(*action, (index + 1, id + 1));
                        }
                        let ev = self.tn_interpreter.get_event_id(e.action, e.pos, id);
                        for (e2, id2) in path.iter() {
                            let e_id = (e.action, index);
                            let e2_id = (e2.action, e2.pos);
                            let ev2 = self.tn_interpreter.get_event_id(e2.action, e2.pos, *id2);
                            if self.mutex.check(&(e_id, e2_id), &self.event_fluents) {
                                let b = -self.epsilon_rational.clone();
                                tn.add(&ev2, &ev, &b);
                            } else if self.precedence.check(&(e2_id, e_id), &self.event_fluents) {
                                tn.add(&ev2, &ev, &mk_rational(0, 1));
                            } else {
                                // tn.add(&ev2, &ev, &mk_rational(0, 1));
                            }
                        }
                        for (a, i) in todo.iter() {
                            let mut id2 = i.1;
                            for (j, (_, e2)) in self.events[a].iter().skip(i.0).enumerate() {
                                let e_id = (e.action, index);
                                let e2_id = (a.clone(), j + i.0);
                                let ev2 = self.tn_interpreter.get_event_id(e2.action, e2.pos, id2);
                                if self.mutex.check(&(e_id, e2_id), &self.event_fluents) {
                                    let b = -self.epsilon_rational.clone();
                                    tn.add(&ev, &ev2, &b);
                                } else {
                                    // tn.add(&ev, &ev2, &mk_rational(0, 1));
                                }
                                id2 += 1;
                            }
                        }
                        path.push((e.clone(), id));
                    }
                } else {
                    let start = self.tn_interpreter.get_action_id(*action, true, counter);
                    let end = self.tn_interpreter.get_action_id(*action, false, counter);
                    counter += 1;
                    let duration = self.actions_duration[action.idx].as_ref();
                    let (lb, ub) = match duration {
                        Some(d) => {
                            let mut lb = -get_rational_from_expression_node(&internal_evaluate(
                                &d.0, &state,
                            )?)?;
                            let mut ub = get_rational_from_expression_node(&internal_evaluate(
                                &d.1, &state,
                            )?)?;
                            if d.2 {
                                lb -= self.epsilon_rational.clone();
                            }
                            if d.3 {
                                ub -= self.epsilon_rational.clone();
                            }
                            (lb, ub)
                        }
                        None => (mk_rational(0, 1), mk_rational(0, 1)),
                    };
                    tn.add(&start, &end, &lb);
                    tn.add(&end, &start, &ub);
                    let id = counter;
                    for (t, e) in events.iter() {
                        let ev = self.tn_interpreter.get_event_id(e.action, e.pos, counter);
                        let b1 = -t.delay.clone();
                        let b2 = t.delay.clone();
                        if t.is_from_start() {
                            tn.add(&start, &ev, &b1);
                            tn.add(&ev, &start, &b2);
                        } else {
                            tn.add(&end, &ev, &b1);
                            tn.add(&ev, &end, &b2);
                        }
                        counter += 1;
                    }
                    let e = events[0].1.clone();
                    let ev = self.tn_interpreter.get_event_id(e.action, e.pos, id);
                    for (e2, id2) in path.iter() {
                        let e_id = (e.action, 0);
                        let e2_id = (e2.action, e2.pos);
                        let ev2 = self.tn_interpreter.get_event_id(e2.action, e2.pos, *id2);
                        if self.mutex.check(&(e_id, e2_id), &self.event_fluents) {
                            let b = -self.epsilon_rational.clone();
                            tn.add(&ev2, &ev, &b);
                        } else if self.precedence.check(&(e2_id, e_id), &self.event_fluents) {
                            tn.add(&ev2, &ev, &mk_rational(0, 1));
                        } else {
                            // tn.add(&ev2, &ev, &mk_rational(0, 1));
                        }
                    }
                    for (a, i) in todo.iter() {
                        let mut id2 = i.1;
                        for (j, (_, e2)) in self.events[a].iter().skip(i.0).enumerate() {
                            let e_id = (e.action, 0);
                            let e2_id = (a.clone(), j + i.0);
                            let ev2 = self.tn_interpreter.get_event_id(e2.action, e2.pos, id2);
                            if self.mutex.check(&(e_id, e2_id), &self.event_fluents) {
                                let b = -self.epsilon_rational.clone();
                                tn.add(&ev, &ev2, &b);
                            } else {
                                // tn.add(&ev, &ev2, &mk_rational(0, 1));
                            }
                            id2 += 1;
                        }
                    }
                    path.push((e.clone(), id));
                    if events.len() > 1 {
                        todo.insert(*action, (1, id + 1));
                    }
                }
            }
        }

        let mut res = Vec::new();
        let mut start_time: FxHashMap<(Action, u32), BigRational> =
            FxHashMap::with_hasher(FxBuildHasher::default());
        let mut end_time: FxHashMap<(Action, u32), BigRational> =
            FxHashMap::with_hasher(FxBuildHasher::default());
        for (a, t) in self.tn_interpreter.get_actions_timings(&tn).iter() {
            if a.1 {
                start_time.insert((a.0, a.2), t.clone());
            } else {
                end_time.insert((a.0, a.2), t.clone());
            }
        }
        for (a, st) in start_time {
            let et = &end_time[&a];
            let d = et - st.clone();
            let d: Option<BigRational> = if d == mk_rational(0, 1) {
                None
            } else {
                Some(d)
            };
            res.push((Some(st), a.0, d));
        }
        res.sort();
        Ok(res)
    }
}
