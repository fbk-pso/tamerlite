use std::{
    collections::{HashMap, HashSet}, sync::Arc, vec::Vec
};
use multiset::HashMultiSet;
use std::hash::{Hash, Hasher};
use num_rational::BigRational;
use pyo3::{exceptions::PyException, prelude::*};

use super::stn::DeltaSTN;
use super::expressions::*;
use super::structures::*;
use super::utils::*;

#[pyclass]
#[derive(Debug, Clone)]
pub struct State {
    pub assignments: HashMap<String, ExpressionNode>,
    pub temporal_network: Option<DeltaSTN<f32>>,
    pub todo: HashMap<String, (usize, usize)>,
    pub active_conditions: HashMultiSet<Vec<ExpressionNode>>,
    pub g: f64,
    pub path: Option<Arc<PersistentList<(String, usize, usize)>>>
}

#[pymethods]
impl State {
    #[getter]
    fn g(&self) -> f64 {
        self.g
    }

    #[getter]
    fn todo(&self) -> HashMap<String, (usize, usize)> {
        self.todo.clone()
    }
}

impl State {
    pub fn get_value(&self, fluent: &String) -> &ExpressionNode {
        &self.assignments[fluent]
    }
}

impl PartialEq for State {
    fn eq(&self, other: &Self) -> bool {
        if self.temporal_network.is_none() {
            self.assignments == other.assignments
        } else {
            false
        }
    }
}

impl Eq for State {}

impl Hash for State {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let mut pairs: Vec<_> = self.assignments.iter().collect();
        pairs.sort_by_key(|i| i.0);
        Hash::hash(&pairs, state);
    }
}

#[derive(Debug)]
pub struct TNInterpreter {
    actions_ids: HashMap<(String, bool, usize), i32>,
    events_ids: HashMap<(String, usize, usize), i32>,
    actions_ids_map_back: HashMap<i32, (String, bool, usize)>,
    events_ids_map_back: HashMap<i32, (String, usize, usize)>,
    counter: i32,
}

impl TNInterpreter {
    fn new() -> Self {
        TNInterpreter { actions_ids: HashMap::new(), events_ids: HashMap::new(), actions_ids_map_back: HashMap::new(), events_ids_map_back: HashMap::new(), counter: 0 }
    }

    pub fn clear(&mut self) {
        self.actions_ids.clear();
        self.events_ids.clear();
        self.actions_ids_map_back.clear();
        self.events_ids_map_back.clear();
        self.counter = 0;
    }

    pub fn get_action_id(&mut self, action: (String, bool, usize)) -> i32 {
        if let Some(id) = self.actions_ids.get(&action) {
            *id
        } else {
            let id = self.counter;
            self.actions_ids.insert(action.clone(), id);
            self.actions_ids_map_back.insert(id, action);
            self.counter += 1;
            id
        }
    }

    pub fn get_event_id(&mut self, event: (String, usize, usize)) -> i32 {
        if let Some(id) = self.events_ids.get(&event) {
            *id
        } else {
            let id = self.counter;
            self.events_ids.insert(event.clone(), id);
            self.events_ids_map_back.insert(id, event);
            self.counter += 1;
            id
        }
    }

    pub fn get_action_timing<Q>(&self, tn: &DeltaSTN<Q>, action: &(String, bool, usize)) -> Option<Q>
    where Q: num_traits::Num + std::ops::Neg<Output=Q> + PartialOrd + Clone {
        let id = self.actions_ids.get(action)?;
        tn.get_model_value(id)
    }

    pub fn get_event_timing<Q>(&self, tn: &DeltaSTN<Q>, event: &(String, usize, usize)) -> Option<Q>
    where Q: num_traits::Num + std::ops::Neg<Output=Q> + PartialOrd + Clone {
        let id = self.events_ids.get(event)?;
        tn.get_model_value(id)
    }

    pub fn get_actions_timings<Q>(&self, tn: &DeltaSTN<Q>) -> Vec<(&(String, bool, usize), Q)>
    where Q: num_traits::Num + std::ops::Neg<Output=Q> + PartialOrd + Clone {
        let mut res: Vec<(&(String, bool, usize), Q)> = Vec::new();
        for (id, v) in tn.distances.iter() {
            let a = self.actions_ids_map_back.get(id);
            if a.is_some() {
                res.push((a.unwrap(), v.clone() * (- Q::one())));
            }
        }
        res.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        res
    }

    pub fn get_events_timings<Q>(&self, tn: &DeltaSTN<Q>) -> Vec<(&(String, usize, usize), Q)>
    where Q: num_traits::Num + std::ops::Neg<Output=Q> + PartialOrd + Clone {
        let mut res = Vec::new();
        for (id, v) in tn.distances.iter() {
            let a = self.events_ids_map_back.get(id);
            if a.is_some() {
                res.push((a.unwrap(), v.clone() * (- Q::one())));
            }
        }
        res.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        res
    }

}

#[pyclass(name = "SearchSpace")]
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
    counter: usize,
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
        #[pyo3(from_py_with = "get_option_big_rational")]
        epsilon: Option<BigRational>,
    ) -> PyResult<Self> {
        let is_temporal = actions_duration.values().any(|value| !value.is_none());
        let mut actions: Vec<String> = events.keys().cloned().collect();
        actions.sort();
        let converted_actions_duration: HashMap<String, Option<(Vec<ExpressionNode>, Vec<ExpressionNode>, bool, bool)>> = actions_duration
            .into_iter()
            .map(|(key, value)| {
                let converted_value = match value {
                    Some((vec1, vec2, b1, b2)) => {
                        Some((
                            vec1.into_iter().map(|e| e.v).collect(),
                            vec2.into_iter().map(|e| e.v).collect(),
                            b1,
                            b2,
                        ))
                    }
                    None => None,
                };
                (key, converted_value)
            })
            .collect();
        let res = SearchSpace {
            actions_duration: converted_actions_duration,
            events: events,
            actions: actions,
            mutex: mutex,
            initial_state: initial_state.map(|inner_map| inner_map.into_iter().map(|(k, v)| (k, v.v)).collect()),
            goal: goal.map(|inner_vec| inner_vec.into_iter().map(|e| e.v).collect()),
            tn_interpreter: TNInterpreter::new(),
            epsilon: match &epsilon {
                Some(x) => rational_to_f32(x),
                None => 0.01,
            },
            epsilon_rational: match epsilon {
                Some(x) => x,
                None => mk_rational(1, 100),
            },
            is_temporal: is_temporal,
            counter: 0,
        };
        Ok(res)
    }

    pub fn reset(&mut self) {
        self.tn_interpreter.clear();
    }

    #[pyo3(signature = (initial_state=None))]
    pub fn initial_state(&self, initial_state: Option<HashMap<String, PyExpressionNode>>) -> PyResult<State> {
        let init = match initial_state {
            Some(v) => v.into_iter().map(|(k, v)| (k, v.v)).collect(),
            None => {
                match &self.initial_state {
                    Some(v) => v.clone(),
                    None => {
                        return Err(PyException::new_err("The initial state must be defined somewhere!"));
                    },
                }
            }
        };
        let tn: Option<DeltaSTN<f32>> = match self.is_temporal {
            true => Some(DeltaSTN::new(self.epsilon/1000.0)),
            false => None,
        };
        Ok(State {
            assignments: init,
            temporal_network: tn,
            todo: HashMap::new(),
            active_conditions: HashMultiSet::new(),
            g: 0.0,
            path: PersistentList::new(),
        })
    }

    pub fn get_successor_states(&mut self, state: &State) -> PyResult<Vec<State>> {
        let mut res = Vec::new();
        let actions: Vec<String> = self.actions.iter().map(|a|a.to_string()).collect();
        for action in actions.iter() {
            match self.get_successor_state(state, action)? {
                Some(s) => res.push(s),
                None => continue,
            }
        }
        Ok(res)
    }

    pub fn get_successor_state(&mut self, state: &State, action: &str) -> PyResult<Option<State>> {
        if let Some(events) = self.events.get(action) {
            if let Some((index, id)) = state.todo.get(action) {
                if let Some((_, e)) = events.get(*index) {
                    // Check if the event is applicable before creating the new state
                    if !self.is_sat(&e.conditions, state)? { return Ok(None); }

                    let mut new_state = state.clone();
                    new_state.g += 1.0;

                    if index + 1 >= events.len() {
                        new_state.todo.remove(action);
                    } else {
                        new_state.todo.insert(action.to_string(), (index + 1, id + 1));
                    }
                    if self.expand_event(state, &mut new_state, &e.clone(), index, id)? {
                        return Ok(Some(new_state));
                    }
                }
            } else {
                // Check if action is applicable before creating the new state
                if !self.is_sat(&events[0].1.conditions, state)? { return Ok(None); }

                let mut new_state = state.clone();
                new_state.g += 1.0;

                if self.open_action(state, &mut new_state, action, &events.clone())? {
                    return Ok(Some(new_state));
                }
            }
        }
        Ok(None)
    }

    #[pyo3(signature = (state, goal=None))]
    pub fn goal_reached(&self, state: &State, goal: Option<Vec<PyExpressionNode>>) -> PyResult<bool> {
        if ! state.todo.is_empty() {
            return Ok(false);
        }
        let goal = goal.map(|g| g.into_iter().map(|e| e.v).collect());
        let g = match &goal {
            Some(v) => v,
            None => {
                match &self.goal {
                    Some(v) => v,
                    None => {
                        return Err(PyException::new_err("The goal must be defined somewhere!"));
                    },
                }
            }
        };
        match internal_evaluate(&g, state)? {
            ExpressionNode::Bool(v) => Ok(v),
            _ => return Err(PyException::new_err("The goal is not a boolean expression!")),
        }
    }

    #[pyo3(signature = (state, goal=None))]
    pub fn subgoals_sat(&self, state: &State, goal: Option<Vec<PyExpressionNode>>) -> PyResult<Vec<Vec<PyExpressionNode>>> {
        let goals = match goal {
            Some(v) => split_expression(&v.into_iter().map(|e| e.v).collect())?,
            None => {
                match &self.goal {
                    Some(v) => split_expression(&v)?,
                    None => {
                        return Err(PyException::new_err("The goal must be defined somewhere!"));
                    },
                }
            }
        };
        let mut res: HashSet<_> = HashSet::new();
        for g in goals {
            if internal_evaluate(&g, state)? == ExpressionNode::Bool(true) {
                res.insert(g.into_iter().map(|v| PyExpressionNode {v}).collect() );
            }
        }
        Ok(res.into_iter().collect())
    }

}

impl SearchSpace {

    pub fn build_plan(&mut self, all_path: Vec<String>) -> PyResult<Vec<(Option<BigRational>, String, Option<BigRational>)>> {
        let mut tn = DeltaSTN::new(mk_rational(0, 1));
        let mut todo: HashMap<String, (usize, usize)> = HashMap::new();
        let mut path: Vec<(Event, usize)> = Vec::new();
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
                        let ev = self.tn_interpreter.get_event_id((e.action.to_string(), e.pos, id));
                        for (e2, id2) in path.iter() {
                            let e_id = (e.action.to_string(), index);
                            let e2_id = (e2.action.to_string(), e2.pos);
                            let ev2 = self.tn_interpreter.get_event_id((e2.action.to_string(), e2.pos, *id2));
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
                                let ev2 = self.tn_interpreter.get_event_id((e2.action.to_string(), e2.pos, id2));
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
                    let start = self.tn_interpreter.get_action_id((action.to_string(), true, counter));
                    let end = self.tn_interpreter.get_action_id((action.to_string(), false, counter));
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
                        let ev = self.tn_interpreter.get_event_id((e.action.to_string(), e.pos, counter));
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
                    let ev = self.tn_interpreter.get_event_id((e.action.to_string(), e.pos, id));
                    for (e2, id2) in path.iter() {
                        let e_id = (e.action.to_string(), 0);
                        let e2_id = (e2.action.to_string(), e2.pos);
                        let ev2 = self.tn_interpreter.get_event_id((e2.action.to_string(), e2.pos, *id2));
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
                            let ev2 = self.tn_interpreter.get_event_id((e2.action.to_string(), e2.pos, id2));
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
                        todo.insert(action.to_string(), (1, id+1));
                    }
                }
            }
        }

        let mut res = Vec::new();
        let mut start_time: HashMap<(String, usize), BigRational> = HashMap::new();
        let mut end_time: HashMap<(String, usize), BigRational> = HashMap::new();
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
            _ => return Err(PyException::new_err("An action condition is not a boolean expression!")),
        };
        Ok(sat)
    }

    fn expand_event(&mut self, state: &State, new_state: &mut State, e: &Event, index: &usize, id: &usize) -> PyResult<bool> {
        new_state.path = PersistentList::append((e.action.to_string(), e.pos, *id), &new_state.path);

        // check conditions is done before calling this method

        // remove end conditions
        for c in e.end_conditions.iter() {
            new_state.active_conditions.remove(&c);
        }

        // check active conditions
        for c in new_state.active_conditions.iter() {
            let sat = match internal_evaluate(&c, state)? {
                ExpressionNode::Bool(v) => v,
                _ => return Err(PyException::new_err("An action condition is not a boolean expression!")),
            };
            if !sat { return Ok(false); }
        }

        // insert start conditions
        for c in e.start_conditions.iter() {
            new_state.active_conditions.insert(c.to_vec());
        }

        // apply effects
        for eff in e.effects.iter() {
            new_state.assignments.insert(eff.fluent.to_string(), internal_evaluate(&eff.value, state)?);
        }

        // check active conditions
        for c in new_state.active_conditions.iter() {
            let sat = match internal_evaluate(&c, new_state)? {
                ExpressionNode::Bool(v) => v,
                _ => return Err(PyException::new_err("An action condition is not a boolean expression!")),
            };
            if !sat { return Ok(false); }
        }

        if self.is_temporal { // Add temporal constraints between past or todo events and the current one
            let tn = new_state.temporal_network.as_mut().unwrap();
            let ev = self.tn_interpreter.get_event_id((e.action.to_string(), e.pos, *id));
            for e2 in PersistentList::to_vec(&state.path) {
                let ev2 = self.tn_interpreter.get_event_id(e2.clone());
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
                    let ev2 = self.tn_interpreter.get_event_id((e2.action.to_string(), e2.pos, id2));
                    if self.mutex.contains(&(e_id, e2_id)) {
                        let b: f32 = -self.epsilon;
                        tn.add(&ev, &ev2, &b);
                    } else {
                        tn.add(&ev, &ev2, &0.0);
                    }
                    id2 += 1;
                }
            }
            if ! tn.check() {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn open_action(&mut self, state: &State, new_state: &mut State, action: &str, events: &Vec<(Timing, Event)>) -> PyResult<bool> {
        let mut id = self.counter;
        if self.is_temporal { // Add temporal constraints between events of the action
            let tn = new_state.temporal_network.as_mut().unwrap();
            let start = self.tn_interpreter.get_action_id((action.to_string(), true, self.counter));
            let end = self.tn_interpreter.get_action_id((action.to_string(), false, self.counter));
            self.counter += 1;
            let duration = self.actions_duration[action].as_ref();
            let mut lb: f32 = 0.0;
            let mut ub: f32 = 0.0;
            if duration.is_some() {
                let d = duration.unwrap();
                lb = -rational_to_f32(&get_rational_from_expression_node(&internal_evaluate(&d.0, state)?)?);
                ub = rational_to_f32(&get_rational_from_expression_node(&internal_evaluate(&d.1, state)?)?);
                if d.2 {
                    lb -= self.epsilon;
                }
                if d.3 {
                    ub -= self.epsilon;
                }
            }
            tn.add(&start, &end, &lb);
            tn.add(&end, &start, &ub);
            id = self.counter;
            for (t, e) in events.iter() {
                let ev = self.tn_interpreter.get_event_id((e.action.to_string(), e.pos, self.counter));
                let b1 = -rational_to_f32(&t.delay);
                let b2 = rational_to_f32(&t.delay);
                if t.is_from_start() {
                    tn.add(&start, &ev, &b1);
                    tn.add(&ev, &start, &b2);
                } else {
                    tn.add(&end, &ev, &b1);
                    tn.add(&ev, &end, &b2);
                }
                self.counter += 1;
            }
            if events.len() > 1 {
                new_state.todo.insert(action.to_string(), (1, id+1));
            }
        }
        self.expand_event(state, new_state, &events[0].1, &0, &id)
    }

}

#[pyfunction]
pub fn simplify(exp: Vec<PyExpressionNode>, assignments: HashMap<String, PyExpressionNode>) -> PyResult<Vec<PyExpressionNode>> {
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
                if ! unresolved {
                    to_remove.extend(v.iter().clone());
                    ExpressionNode::Bool(val)
                } else {
                    to_remove.extend(true_to_remove);
                    e.v.clone()
                }
            },
            ExpressionNode::Not(p) => {
                if let ExpressionNode::Bool(v) = res[*p] {
                    to_remove.push(*p);
                    ExpressionNode::Bool(!v)
                } else {
                    e.v.clone()
                }
            },
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
            },
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
            },
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
            },
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
                    }
                    else {
                        ExpressionNode::Rational(r)
                    }
                } else {
                    e.v.clone()
                }
            },
            ExpressionNode::Minus(p1, p2) => {
                let val1 = get_rational_from_expression_node(&res[*p1]);
                let val2 = get_rational_from_expression_node(&res[*p2]);
                if val1.is_ok() && val2.is_ok() {
                    to_remove.push(*p1);
                    to_remove.push(*p2);
                    let r = val1.unwrap() - val2.unwrap();
                    if r.is_integer() {
                        ExpressionNode::Int(r.to_integer())
                    }
                    else {
                        ExpressionNode::Rational(r)
                    }
                } else {
                    e.v.clone()
                }
            },
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
                    }
                    else {
                        ExpressionNode::Rational(r)
                    }
                } else {
                    e.v.clone()
                }
            },
            ExpressionNode::Div(p1, p2) => {
                let val1 = get_rational_from_expression_node(&res[*p1]);
                let val2 = get_rational_from_expression_node(&res[*p2]);
                if val1.is_ok() && val2.is_ok() {
                    to_remove.push(*p1);
                    to_remove.push(*p2);
                    let r = val1.unwrap() / val2.unwrap();
                    if r.is_integer() {
                        ExpressionNode::Int(r.to_integer())
                    }
                    else {
                        ExpressionNode::Rational(r)
                    }
                } else {
                    e.v.clone()
                }
            },
            ExpressionNode::Fluent(s) => {
                if let Some(v) = assignments.get(s) {
                    v.v.clone()
                } else {
                    e.v.clone()
                }
            }
            other => {
                (*other).clone()
            }
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
                },
                ExpressionNode::Not(p) => {
                    if !to_remove.contains(&p) {
                        let offset = to_remove.iter().filter(|&&x| x < p).count();
                        ExpressionNode::Not(p - offset)
                    } else {
                        ExpressionNode::Not(p)
                    }
                },
                ExpressionNode::Equals(p1, p2) => {
                    let mut offset1 = 0;
                    if !to_remove.contains(&p1) {
                        offset1 = to_remove.iter().filter(|&&x| x < p1).count();
                    }
                    let mut offset2 = 0;
                    if !to_remove.contains(&p2) {
                        offset2 = to_remove.iter().filter(|&&x| x < p2).count();
                    }
                    ExpressionNode::Equals(p1-offset1, p2-offset2)
                },
                ExpressionNode::LE(p1, p2) => {
                    let mut offset1 = 0;
                    if !to_remove.contains(&p1) {
                        offset1 = to_remove.iter().filter(|&&x| x < p1).count();
                    }
                    let mut offset2 = 0;
                    if !to_remove.contains(&p2) {
                        offset2 = to_remove.iter().filter(|&&x| x < p2).count();
                    }
                    ExpressionNode::LE(p1-offset1, p2-offset2)
                },
                ExpressionNode::LT(p1, p2) => {
                    let mut offset1 = 0;
                    if !to_remove.contains(&p1) {
                        offset1 = to_remove.iter().filter(|&&x| x < p1).count();
                    }
                    let mut offset2 = 0;
                    if !to_remove.contains(&p2) {
                        offset2 = to_remove.iter().filter(|&&x| x < p2).count();
                    }
                    ExpressionNode::LT(p1-offset1, p2-offset2)
                },
                ExpressionNode::Plus(v) => {
                    let mut operands = Vec::new();
                    for o in v {
                        if !to_remove.contains(&o) {
                            let offset = to_remove.iter().filter(|&&x| x < o).count();
                            operands.push(o - offset);
                        }
                    }
                    ExpressionNode::Plus(operands)
                },
                ExpressionNode::Minus(p1, p2) => {
                    let mut offset1 = 0;
                    if !to_remove.contains(&p1) {
                        offset1 = to_remove.iter().filter(|&&x| x < p1).count();
                    }
                    let mut offset2 = 0;
                    if !to_remove.contains(&p2) {
                        offset2 = to_remove.iter().filter(|&&x| x < p2).count();
                    }
                    ExpressionNode::Minus(p1-offset1, p2-offset2)
                },
                ExpressionNode::Times(v) => {
                    let mut operands = Vec::new();
                    for o in v {
                        if !to_remove.contains(&o) {
                            let offset = to_remove.iter().filter(|&&x| x < o).count();
                            operands.push(o - offset);
                        }
                    }
                    ExpressionNode::Times(operands)
                },
                ExpressionNode::Div(p1, p2) => {
                    let mut offset1 = 0;
                    if !to_remove.contains(&p1) {
                        offset1 = to_remove.iter().filter(|&&x| x < p1).count();
                    }
                    let mut offset2 = 0;
                    if !to_remove.contains(&p2) {
                        offset2 = to_remove.iter().filter(|&&x| x < p2).count();
                    }
                    ExpressionNode::Div(p1-offset1, p2-offset2)
                },
                _ => {
                    e
                }
            };
            final_res.push(PyExpressionNode{v: ne})
        }
    }

    Ok(final_res)
}

#[pyfunction]
pub fn evaluate(exp: Vec<PyExpressionNode>, state: &State) -> PyResult<PyExpressionNode> {
    Ok(PyExpressionNode {v: internal_evaluate(&exp.into_iter().map(|e| e.v).collect(), state)? })
}

pub fn internal_evaluate(exp: &Vec<ExpressionNode>, state: &State) -> PyResult<ExpressionNode> {
    let mut res: Vec<ExpressionNode> = vec![];
    for e in exp {
        let value = match &e {
            ExpressionNode::And(v) => {
                let val = v.iter().all(|&p| res[p] == ExpressionNode::Bool(true));
                ExpressionNode::Bool(val)
            },
            ExpressionNode::Not(p) => {
                ExpressionNode::Bool(ExpressionNode::Bool(false) == res[*p])
            },
            ExpressionNode::Equals(p1, p2) => {
                ExpressionNode::Bool(res[*p1] == res[*p2])
            },
            ExpressionNode::LE(p1, p2) => {
                let val1 = get_rational_from_expression_node(&res[*p1])?;
                let val2 = get_rational_from_expression_node(&res[*p2])?;
                ExpressionNode::Bool(val1 <= val2)
            },
            ExpressionNode::LT(p1, p2) => {
                let val1 = get_rational_from_expression_node(&res[*p1])?;
                let val2 = get_rational_from_expression_node(&res[*p2])?;
                ExpressionNode::Bool(val1 < val2)
            },
            ExpressionNode::Plus(v) => {
                let mut r = get_rational_from_expression_node(&res[v[0]])?;
                for p in v.iter().skip(1) {
                    r += get_rational_from_expression_node(&res[*p])?;
                }
                if r.is_integer() {
                    ExpressionNode::Int(r.to_integer())
                }
                else {
                    ExpressionNode::Rational(r)
                }
            },
            ExpressionNode::Minus(p1, p2) => {
                let val1 = get_rational_from_expression_node(&res[*p1])?;
                let val2 = get_rational_from_expression_node(&res[*p2])?;
                let r = val1 - val2;
                if r.is_integer() {
                    ExpressionNode::Int(r.to_integer())
                }
                else {
                    ExpressionNode::Rational(r)
                }
            },
            ExpressionNode::Times(v) => {
                let mut r = get_rational_from_expression_node(&res[v[0]])?;
                for p in v.iter().skip(1) {
                    r *= get_rational_from_expression_node(&res[*p])?;
                }
                if r.is_integer() {
                    ExpressionNode::Int(r.to_integer())
                }
                else {
                    ExpressionNode::Rational(r)
                }
            },
            ExpressionNode::Div(p1, p2) => {
                let val1 = get_rational_from_expression_node(&res[*p1])?;
                let val2 = get_rational_from_expression_node(&res[*p2])?;
                let r = val1 / val2;
                if r.is_integer() {
                    ExpressionNode::Int(r.to_integer())
                }
                else {
                    ExpressionNode::Rational(r)
                }
            },
            ExpressionNode::Fluent(s) => {
                state.get_value(&s).clone()
            }
            other => {
                (*other).clone()
            }
        };
        if res.len() == exp.len() - 1 {
            return Ok(value)
        } else {
            res.push(value);
        }
    }
    Err(PyException::new_err("Unreachable code"))
}
