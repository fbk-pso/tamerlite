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

use multiset::HashMultiSet;
use num_rational::BigRational;
use pyo3::{exceptions::PyException, prelude::*};
use std::{
    collections::{HashMap, HashSet},
    sync::Mutex,
    vec::Vec,
};

use super::expressions::*;
use super::expressions_utils::*;
use super::search_state::*;
use super::stn::DeltaSTN;
use super::structures::*;
use super::tn_interpreter::TNInterpreter;
use super::utils::*;

#[pyclass(name = "SearchSpace", frozen)]
#[derive(Debug)]
pub struct SearchSpace {
    actions_duration:
        HashMap<String, Option<(Vec<ExpressionNode>, Vec<ExpressionNode>, bool, bool)>>,
    events: HashMap<String, Vec<(Timing, Event)>>,
    actions: Vec<String>,
    mutex: HashSet<((String, usize), (String, usize))>,
    initial_state: Option<HashMap<String, ExpressionNode>>,
    goal: Option<Vec<ExpressionNode>>,
    pub tn_interpreter: TNInterpreter,
    epsilon: f32,
    epsilon_rational: BigRational,
    pub is_temporal: bool,
    counter: Mutex<u32>,
}

#[pymethods]
impl SearchSpace {
    #[new]
    #[pyo3(signature = (actions_duration, events, mutex, initial_state=None, goal=None, epsilon=None))]
    fn new(
        actions_duration: HashMap<
            String,
            Option<(Vec<PyExpressionNode>, Vec<PyExpressionNode>, bool, bool)>,
        >,
        events: HashMap<String, Vec<(Timing, Event)>>,
        mutex: HashSet<((String, usize), (String, usize))>,
        initial_state: Option<HashMap<String, PyExpressionNode>>,
        goal: Option<Vec<PyExpressionNode>>,
        #[pyo3(from_py_with = "get_option_big_rational")] epsilon: Option<BigRational>,
    ) -> PyResult<Self> {
        let is_temporal = actions_duration.values().any(|value| !value.is_none());
        let mut actions: Vec<String> = events.keys().cloned().collect();
        actions.sort();
        let converted_actions_duration: HashMap<
            String,
            Option<(Vec<ExpressionNode>, Vec<ExpressionNode>, bool, bool)>,
        > = actions_duration
            .into_iter()
            .map(|(key, value)| {
                let converted_value = match value {
                    Some((vec1, vec2, b1, b2)) => Some((
                        vec1.into_iter().map(|e| e.v).collect(),
                        vec2.into_iter().map(|e| e.v).collect(),
                        b1,
                        b2,
                    )),
                    None => None,
                };
                (key, converted_value)
            })
            .collect();

        let tn_interpreter = TNInterpreter::new(&actions, &events);

        let res = SearchSpace {
            actions_duration: converted_actions_duration,
            events: events,
            actions: actions,
            mutex: mutex,
            initial_state: initial_state
                .map(|inner_map| inner_map.into_iter().map(|(k, v)| (k, v.v)).collect()),
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

    pub fn reset(&self) {
        // DO nothing :)
    }

    #[pyo3(signature = (initial_state=None))]
    pub fn initial_state(
        &self,
        initial_state: Option<HashMap<String, PyExpressionNode>>,
    ) -> PyResult<State> {
        let init = match initial_state {
            Some(v) => v.into_iter().map(|(k, v)| (k, v.v)).collect(),
            None => match &self.initial_state {
                Some(v) => v.clone(),
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
            todo: HashMap::new(),
            active_conditions: HashMultiSet::new(),
            g: 0.0,
            path: PersistentList::new(),
            heuristic_cache: Mutex::new(HashMap::new()),
        })
    }

    pub fn get_successor_states(&self, state: &State) -> PyResult<Vec<State>> {
        let mut res = Vec::new();
        for rs in self.get_successor_states_iter(state) {
            res.push(rs?);
        }
        Ok(res)
    }

    pub fn get_successor_state(&self, state: &State, action: &str) -> PyResult<Option<State>> {
        if let Some(events) = self.events.get(action) {
            if let Some((index, id)) = state.todo.get(action) {
                if let Some((_, e)) = events.get(*index) {
                    // Check if the event is applicable before creating the new state
                    if !self.is_sat(&e.conditions, state)? {
                        return Ok(None);
                    }

                    let mut new_state = state.clone_for_child();
                    new_state.g += 1.0;

                    if index + 1 >= events.len() {
                        new_state.todo.remove(action);
                    } else {
                        new_state
                            .todo
                            .insert(action.to_string(), (index + 1, id + 1));
                    }
                    if self.expand_event(state, &mut new_state, &e.clone(), index, id)? {
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

                if self.open_action(state, &mut new_state, action, &events.clone())? {
                    return Ok(Some(new_state));
                }
            }
        }
        Ok(None)
    }

    #[pyo3(signature = (state, goal=None))]
    pub fn goal_reached(
        &self,
        state: &State,
        goal: Option<Vec<PyExpressionNode>>,
    ) -> PyResult<bool> {
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
            _ => {
                return Err(PyException::new_err(
                    "The goal is not a boolean expression!",
                ))
            }
        }
    }

    #[pyo3(signature = (state, goal=None))]
    pub fn subgoals_sat(
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
        let mut res: HashSet<_> = HashSet::new();
        for g in goals {
            if internal_evaluate(&g, state)? == ExpressionNode::Bool(true) {
                res.insert(g.into_iter().map(|v| PyExpressionNode { v }).collect());
            }
        }
        Ok(res.into_iter().collect())
    }
}

impl SearchSpace {
    pub fn get_successor_states_iter<'a>(
        &'a self,
        state: &'a State,
    ) -> impl Iterator<Item = PyResult<State>> + 'a {
        return self
            .actions
            .iter()
            .map(|action| self.get_successor_state(state, action).transpose())
            .filter(|x| x.is_some())
            .map(|x| x.unwrap());
    }

    pub fn build_plan(
        &self,
        all_path: Vec<String>,
    ) -> PyResult<Vec<(Option<BigRational>, String, Option<BigRational>)>> {
        let mut tn = DeltaSTN::new(mk_rational(0, 1));
        let mut todo: HashMap<String, (usize, u32)> = HashMap::new();
        let mut path: Vec<(Event, u32)> = Vec::new();
        let mut counter = 0;
        let mut state = self.initial_state(None)?;
        for action in all_path.iter() {
            state = self.get_successor_state(&state, action)?.unwrap();
            if let Some(events) = self.events.get(action).cloned() {
                if let Some((index, id)) = todo.get(action).cloned() {
                    if let Some((_, e)) = events.get(index) {
                        if index + 1 >= events.len() {
                            todo.remove(action);
                        } else {
                            todo.insert(action.to_string(), (index + 1, id + 1));
                        }
                        let ev = self.tn_interpreter.get_event_id(&e.action, e.pos, id);
                        for (e2, id2) in path.iter() {
                            let e_id = (e.action.to_string(), index);
                            let e2_id = (e2.action.to_string(), e2.pos);
                            let ev2 = self.tn_interpreter.get_event_id(&e2.action, e2.pos, *id2);
                            if self.mutex.contains(&(e_id, e2_id)) {
                                let b = -self.epsilon_rational.clone();
                                tn.add(&ev2, &ev, &b);
                            } else {
                                // tn.add(&ev2, &ev, &mk_rational(0, 1));
                            }
                        }
                        for (a, i) in todo.iter() {
                            let mut id2 = i.1;
                            for (j, (_, e2)) in self.events[a].iter().skip(i.0).enumerate() {
                                let e_id = (e.action.to_string(), index);
                                let e2_id = (a.to_string(), j + i.0);
                                let ev2 = self.tn_interpreter.get_event_id(&e2.action, e2.pos, id2);
                                if self.mutex.contains(&(e_id, e2_id)) {
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
                    let start = self.tn_interpreter.get_action_id(action, true, counter);
                    let end = self.tn_interpreter.get_action_id(action, false, counter);
                    counter += 1;
                    let duration = self.actions_duration[action].as_ref();
                    let mut lb = mk_rational(0, 1);
                    let mut ub = mk_rational(0, 1);
                    if duration.is_some() {
                        let d = duration.unwrap();
                        lb = -get_rational_from_expression_node(&internal_evaluate(&d.0, &state)?)?;
                        ub = get_rational_from_expression_node(&internal_evaluate(&d.1, &state)?)?;
                        if d.2 {
                            lb -= self.epsilon_rational.clone();
                        }
                        if d.3 {
                            ub -= self.epsilon_rational.clone();
                        }
                    }
                    tn.add(&start, &end, &lb);
                    tn.add(&end, &start, &ub);
                    let id = counter;
                    for (t, e) in events.iter() {
                        let ev = self.tn_interpreter.get_event_id(&e.action, e.pos, counter);
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
                    let ev = self.tn_interpreter.get_event_id(&e.action, e.pos, id);
                    for (e2, id2) in path.iter() {
                        let e_id = (e.action.to_string(), 0);
                        let e2_id = (e2.action.to_string(), e2.pos);
                        let ev2 = self.tn_interpreter.get_event_id(&e2.action, e2.pos, *id2);
                        if self.mutex.contains(&(e_id, e2_id)) {
                            let b = -self.epsilon_rational.clone();
                            tn.add(&ev2, &ev, &b);
                        } else {
                            // tn.add(&ev2, &ev, &mk_rational(0, 1));
                        }
                    }
                    for (a, i) in todo.iter() {
                        let mut id2 = i.1;
                        for (j, (_, e2)) in self.events[a].iter().skip(i.0).enumerate() {
                            let e_id = (e.action.to_string(), 0);
                            let e2_id = (a.to_string(), j + i.0);
                            let ev2 = self.tn_interpreter.get_event_id(&e2.action, e2.pos, id2);
                            if self.mutex.contains(&(e_id, e2_id)) {
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
                        todo.insert(action.to_string(), (1, id + 1));
                    }
                }
            }
        }

        let mut res = Vec::new();
        let mut start_time: HashMap<(String, u32), BigRational> = HashMap::new();
        let mut end_time: HashMap<(String, u32), BigRational> = HashMap::new();
        for (a, t) in self.tn_interpreter.get_actions_timings(&tn).iter() {
            if a.1 {
                start_time.insert((a.0.to_string(), a.2), t.clone());
            } else {
                end_time.insert((a.0.to_string(), a.2), t.clone());
            }
        }
        for (a, st) in start_time.iter() {
            let et = &end_time[a];
            let d: Option<BigRational> = if et - st == mk_rational(0, 1) {
                None
            } else {
                Some((et - st).clone())
            };
            res.push((Some(st.clone()), a.0.to_string(), d));
        }
        res.sort();
        Ok(res)
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
        new_state.path =
            PersistentList::append((e.action.to_string(), e.pos, *id), &new_state.path);

        // check conditions is done before calling this method

        // remove end conditions
        for c in e.end_conditions.iter() {
            new_state.active_conditions.remove(&c);
        }

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

        // insert start conditions
        for c in e.start_conditions.iter() {
            new_state.active_conditions.insert(c.to_vec());
        }

        // apply effects
        for eff in e.effects.iter() {
            new_state.assignments.insert(
                eff.fluent.to_string(),
                internal_evaluate(&eff.value, state)?,
            );
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
            let ev = self.tn_interpreter.get_event_id(&e.action, e.pos, *id);
            for e2 in PersistentList::to_vec(&state.path) {
                let ev2 = self.tn_interpreter.get_event_id(&e2.0, e2.1, e2.2);
                let e_id = (e.action.to_string(), *index);
                let e2_id = (e2.0.to_string(), e2.1);
                if self.mutex.contains(&(e_id, e2_id)) {
                    let b: f32 = -self.epsilon;
                    tn.add(&ev2, &ev, &b);
                } else {
                    tn.add(&ev2, &ev, &0.0);
                }
            }
            for (a, i) in new_state.todo.iter() {
                let mut id2 = i.1;
                for (j, (_, e2)) in self.events[a].iter().skip(i.0).enumerate() {
                    let e_id = (e.action.to_string(), *index);
                    let e2_id = (a.to_string(), j + i.0);
                    let ev2 = self.tn_interpreter.get_event_id(&e2.action, e2.pos, id2);
                    if self.mutex.contains(&(e_id, e2_id)) {
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
        action: &str,
        events: &Vec<(Timing, Event)>,
    ) -> PyResult<bool> {
        let mut counter = self.counter.lock().unwrap();
        let mut id = counter.clone();
        if self.is_temporal {
            // Add temporal constraints between events of the action
            let tn = new_state.temporal_network.as_mut().unwrap();
            let start = self.tn_interpreter.get_action_id(action, true, *counter);
            let end = self.tn_interpreter.get_action_id(action, false, *counter);
            *counter += 1;
            let duration = self.actions_duration[action].as_ref();
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
                let ev = self.tn_interpreter.get_event_id(&e.action, e.pos, *counter);
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
                new_state.todo.insert(action.to_string(), (1, id + 1));
            }
        }
        self.expand_event(state, new_state, &events[0].1, &0, &id)
    }
}
