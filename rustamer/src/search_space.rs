use std::{
    collections::{HashSet, HashMap},
    vec::Vec
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
    pub assignments: HashMap<String, PyExpressionNode>,
    pub temporal_network: Option<DeltaSTN<f32>>,
    pub todo: HashMap<String, (usize, usize)>,
    pub active_conditions: HashMultiSet<Vec<PyExpressionNode>>,
    pub g: f64,
    pub path: Vec<(Event, usize)>,
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
    pub fn get_value(&self, fluent: &String) -> PyExpressionNode {
        self.assignments[fluent].clone()
    }

    pub fn extract_solution(&self) -> Vec<(Option<f32>, String, Option<f32>)> {
        let mut res = Vec::new();
        if self.temporal_network.is_some() {
            let tn = self.temporal_network.as_ref().unwrap();
            let mut start_time: HashMap<(String, usize), f32> = HashMap::new();
            let mut end_time: HashMap<(String, usize), f32> = HashMap::new();
            for (a, t) in tn.get_actions_timings().iter() {
                if a.1 {
                    start_time.insert((a.0.to_string(), a.2), *t);
                } else {
                    end_time.insert((a.0.to_string(), a.2), *t);
                }
            }
            for (a, st) in start_time.iter() {
                let et = end_time[a];
                let mut d: Option<f32> = Some(et - st);
                if d.unwrap() < tn.tolerance && d.unwrap() > -tn.tolerance {
                    d = None;
                }
                res.push((Some(*st), a.0.to_string(), d));
            }
        } else {
            for (e, _) in self.path.iter() {
                res.push((None, e.action.clone(), None));
            }
        }
        res
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

#[pyclass(name = "SearchSpace")]
#[derive(Debug)]
pub struct SearchSpace {
    actions_duration:
        HashMap<String, Option<(Vec<PyExpressionNode>, Vec<PyExpressionNode>, bool, bool)>>,
    events: HashMap<String, Vec<(Timing, Event)>>,
    actions: Vec<String>,
    mutex: HashSet<((String, usize), (String, usize))>,
    initial_state: Option<HashMap<String, PyExpressionNode>>,
    goal: Option<Vec<PyExpressionNode>>,
    epsilon: f32,
    epsilon_rational: BigRational,
    pub is_temporal: bool,
    counter: usize,
}

#[pymethods]
impl SearchSpace {
    #[new]
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
        let is_temporal = actions_duration.values().all(|value| !value.is_none());
        let mut actions: Vec<String> = events.keys().cloned().collect();
        actions.sort();
        let res = SearchSpace {
            actions_duration: actions_duration,
            events: events,
            actions: actions,
            mutex: mutex,
            initial_state: initial_state,
            goal: goal,
            epsilon: match epsilon.clone() {
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

    pub fn initial_state(&self, initial_state: Option<HashMap<String, PyExpressionNode>>) -> PyResult<State> {
        let init = match initial_state {
            Some(v) => v,
            None => {
                match self.initial_state.clone() {
                    Some(v) => v,
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
            path: vec![],
        })
    }

    pub fn get_successor_states(&mut self, state: &State) -> PyResult<Vec<State>> {
        let mut res = Vec::new();
        let mut actions: Vec<String> = Vec::new();
        for a in self.actions.iter() {
            actions.push(a.clone());
        }
        for action in  actions {
            match self.get_successor_state(state, &action)? {
                Some(s) => res.push(s),
                None => continue,
            }
        }
        Ok(res)
    }

    pub fn get_successor_state(&mut self, state: &State, action: &str) -> PyResult<Option<State>> {
        if let Some(events) = self.events.get(action).cloned() {
            let mut new_state = state.clone();
            new_state.g += 1.0;

            if let Some((index, id)) = state.todo.get(action) {
                if let Some((_, e)) = events.get(*index) {
                    if index + 1 >= events.len() {
                        new_state.todo.remove(action);
                    } else {
                        new_state.todo.insert(action.to_string(), (index + 1, id + 1));
                    }
                    if self.expand_event(state, &mut new_state, &e, index, id)? {
                        return Ok(Some(new_state));
                    }
                }
            } else {
                if self.open_action(state, &mut new_state, action, &events)? {
                    return Ok(Some(new_state));
                }
            }
        }
        Ok(None)
    }

    pub fn goal_reached(&self, state: &State, goal: Option<Vec<PyExpressionNode>>) -> PyResult<bool> {
        if ! state.todo.is_empty() {
            return Ok(false);
        }
        let g = match goal {
            Some(v) => v,
            None => {
                match self.goal.clone() {
                    Some(v) => v,
                    None => {
                        return Err(PyException::new_err("The goal must be defined somewhere!"));
                    },
                }
            }
        };
        match evaluate(g, state)?.v {
            ExpressionNode::Bool(v) => Ok(v),
            _ => return Err(PyException::new_err("The goal is not a boolean expression!")),
        }
    }

    pub fn subgoals_sat(&self, state: &State, goal: Option<Vec<PyExpressionNode>>) -> PyResult<Vec<Vec<PyExpressionNode>>> {
        let goals = match goal {
            Some(v) => split_expression(&v)?,
            None => {
                match self.goal.clone() {
                    Some(v) => split_expression(&v)?,
                    None => {
                        return Err(PyException::new_err("The goal must be defined somewhere!"));
                    },
                }
            }
        };
        let mut res = HashSet::new();
        for g in goals.iter() {
            if evaluate(g.clone(), state)?.to_expression_node() == ExpressionNode::Bool(true) {
                res.insert(g.clone());
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
                        let ev = tn.get_event_id((e.clone(), id));
                        for (e2, id2) in path.iter() {
                            let e_id = (e.action.to_string(), index);
                            let e2_id = (
                                e2.action.to_string(),
                                self.events[&e2.action].iter().enumerate().find_map(|(j, (_, ev))| {
                                    if *ev == *e2 {
                                        Some(j)
                                    } else {
                                        None
                                    }
                                }).unwrap());
                            let ev2 = tn.get_event_id((e2.clone(), *id2));
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
                                let ev2 = tn.get_event_id((e2.clone(), id2));
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
                    let start = tn.get_action_id((action.to_string(), true, counter));
                    let end = tn.get_action_id((action.to_string(), false, counter));
                    counter += 1;
                    let duration = self.actions_duration[action].as_ref();
                    let mut lb = mk_rational(0, 1);
                    let mut ub = mk_rational(0, 1);
                    if duration.is_some() {
                        let d = duration.unwrap();
                        lb = -get_rational_from_expression_node(&evaluate(d.0.clone(), &state)?.v)?;
                        ub = get_rational_from_expression_node(&evaluate(d.1.clone(), &state)?.v)?;
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
                        let ev = tn.get_event_id((e.clone(), counter));
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
                    let ev = tn.get_event_id((e.clone(), id));
                    for (e2, id2) in path.iter() {
                        let e_id = (e.action.to_string(), 0);
                        let e2_id = (
                            e2.action.to_string(),
                            self.events[&e2.action].iter().enumerate().find_map(|(j, (_, ev))| {
                                if *ev == *e2 {
                                    Some(j)
                                } else {
                                    None
                                }
                            }).unwrap());
                        let ev2 = tn.get_event_id((e2.clone(), *id2));
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
                            let ev2 = tn.get_event_id((e2.clone(), id2));
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
        for (a, t) in tn.get_actions_timings().iter() {
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

    fn expand_event(&self, state: &State, new_state: &mut State, e: &Event, index: &usize, id: &usize) -> PyResult<bool> {
        new_state.path.push((e.clone(), *id));

        // check conditions
        let sat = match evaluate(e.conditions.clone(), state)?.v {
            ExpressionNode::Bool(v) => v,
            _ => return Err(PyException::new_err("An action condition is not a boolean expression!")),
        };
        if !sat { return Ok(false); }

        // remove end conditions
        for c in e.end_conditions.iter() {
            new_state.active_conditions.remove(&c);
        }

        // check active conditions
        for c in new_state.active_conditions.iter() {
            let sat = match evaluate(c.to_vec(), state)?.v {
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
            new_state.assignments.insert(eff.fluent.clone(), evaluate(eff.value.clone(), state)?);
        }

        // check active conditions
        for c in new_state.active_conditions.iter() {
            let sat = match evaluate(c.to_vec(), new_state)?.v {
                ExpressionNode::Bool(v) => v,
                _ => return Err(PyException::new_err("An action condition is not a boolean expression!")),
            };
            if !sat { return Ok(false); }
        }

        if self.is_temporal {
            let tn = new_state.temporal_network.as_mut().unwrap();
            let ev = tn.get_event_id((e.clone(), *id));
            for (e2, id2) in state.path.iter() {
                let ev2 = tn.get_event_id((e2.clone(), *id2));
                let e_id = (e.action.to_string(), *index);
                let e2_id = (
                    e2.action.to_string(),
                    self.events[&e2.action].iter().enumerate().find_map(|(j, (_, ev))| {
                        if *ev == *e2 {
                            Some(j)
                        } else {
                            None
                        }
                    }).unwrap());
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
                    let ev2 = tn.get_event_id((e2.clone(), id2));
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
        if self.is_temporal {
            let tn = new_state.temporal_network.as_mut().unwrap();
            let start = tn.get_action_id((action.to_string(), true, self.counter));
            let end = tn.get_action_id((action.to_string(), false, self.counter));
            self.counter += 1;
            let duration = self.actions_duration[action].as_ref();
            let mut lb: f32 = 0.0;
            let mut ub: f32 = 0.0;
            if duration.is_some() {
                let d = duration.unwrap();
                lb = -rational_to_f32(get_rational_from_expression_node(&evaluate(d.0.clone(), state)?.v)?);
                ub = rational_to_f32(get_rational_from_expression_node(&evaluate(d.1.clone(), state)?.v)?);
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
                let ev = tn.get_event_id((e.clone(), self.counter));
                let b1 = -rational_to_f32(t.delay.clone());
                let b2 = rational_to_f32(t.delay.clone());
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
pub fn evaluate(exp: Vec<PyExpressionNode>, state: &State) -> PyResult<PyExpressionNode> {
    let mut res: Vec<ExpressionNode> = vec![];
    for e in exp {
        match e.to_expression_node() {
            ExpressionNode::And(v) => {
                let mut val = true;
                for p in v {
                    if ExpressionNode::Bool(false) == res[p] {
                        val = false;
                        break;
                    }
                }
                res.push(ExpressionNode::Bool(val));
            },
            ExpressionNode::Not(p) => {
                if ExpressionNode::Bool(false) == res[p] {
                    res.push(ExpressionNode::Bool(true));
                } else {
                    res.push(ExpressionNode::Bool(false));
                }
            },
            ExpressionNode::Equals(p1, p2) => {
                if res[p1] == res[p2] {
                    res.push(ExpressionNode::Bool(true));
                } else {
                    res.push(ExpressionNode::Bool(false));
                }
            },
            ExpressionNode::LE(p1, p2) => {
                let val1 = get_rational_from_expression_node(&res[p1])?;
                let val2 = get_rational_from_expression_node(&res[p2])?;
                if val1 <= val2 {
                    res.push(ExpressionNode::Bool(true));
                } else {
                    res.push(ExpressionNode::Bool(false));
                }
            },
            ExpressionNode::LT(p1, p2) => {
                let val1 = get_rational_from_expression_node(&res[p1])?;
                let val2 = get_rational_from_expression_node(&res[p2])?;
                if val1 < val2 {
                    res.push(ExpressionNode::Bool(true));
                } else {
                    res.push(ExpressionNode::Bool(false));
                }
            },
            ExpressionNode::Plus(v) => {
                let mut r = get_rational_from_expression_node(&res[v[0]])?;
                for p in v.iter().skip(1) {
                    r += get_rational_from_expression_node(&res[*p])?;
                }
                if r.is_integer() {
                    res.push(ExpressionNode::Int(r.to_integer()));
                }
                else {
                    res.push(ExpressionNode::Rational(r));
                }
            },
            ExpressionNode::Minus(p1, p2) => {
                let val1 = get_rational_from_expression_node(&res[p1])?;
                let val2 = get_rational_from_expression_node(&res[p2])?;
                let r = val1 - val2;
                if r.is_integer() {
                    res.push(ExpressionNode::Int(r.to_integer()));
                }
                else {
                    res.push(ExpressionNode::Rational(r));
                }
            },
            ExpressionNode::Times(v) => {
                let mut r = get_rational_from_expression_node(&res[v[0]])?;
                for p in v.iter().skip(1) {
                    r *= get_rational_from_expression_node(&res[*p])?;
                }
                if r.is_integer() {
                    res.push(ExpressionNode::Int(r.to_integer()));
                }
                else {
                    res.push(ExpressionNode::Rational(r));
                }
            },
            ExpressionNode::Div(p1, p2) => {
                let val1 = get_rational_from_expression_node(&res[p1])?;
                let val2 = get_rational_from_expression_node(&res[p2])?;
                let r = val1 / val2;
                if r.is_integer() {
                    res.push(ExpressionNode::Int(r.to_integer()));
                }
                else {
                    res.push(ExpressionNode::Rational(r));
                }
            },
            ExpressionNode::Fluent(s) => {
                res.push(state.get_value(&s).to_expression_node());
            }
            _ => {
                res.push(e.to_expression_node());
            }
        }

    }
    Ok(PyExpressionNode { v: res.last().unwrap().clone() })
}
