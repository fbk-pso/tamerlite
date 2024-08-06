use std::collections::HashSet;
use std::{
    collections::HashMap,
    vec::Vec
};
use std::hash::{Hash, Hasher};

use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::{internal_evaluate, SearchSpace};
use crate::state_encoder::CoreStateEncoder;

use super::search_space::State;
use super::expressions::*;
use super::structures::*;


#[pyclass(frozen)]
#[derive(Clone)]
pub struct Heuristic {
    hff: Option<HFF>,
    hrl: Option<HRL>,
    hcustom: Option<CustomHeuristic>,
}

#[pymethods]
impl Heuristic {

    #[staticmethod]
    pub fn custom(callable: PyObject) -> PyResult<Self> {
        Ok(Heuristic {hff: None, hrl: None, hcustom: Some(CustomHeuristic::new(callable)?)})
    }

    #[staticmethod]
    pub fn hff(fluents: HashMap<String, String>, objects: HashMap<String, Vec<String>>, events: HashMap<String, Vec<(Timing, Event)>>, goal: Vec<PyExpressionNode>) -> PyResult<Self> {
        Ok(Heuristic {hff: Some(HFF::new(fluents, objects, events, goal, false)?), hrl: None, hcustom: None})
    }

    #[staticmethod]
    pub fn hadd(fluents: HashMap<String, String>, objects: HashMap<String, Vec<String>>, events: HashMap<String, Vec<(Timing, Event)>>, goal: Vec<PyExpressionNode>) -> PyResult<Self> {
        Ok(Heuristic {hff: Some(HFF::new(fluents, objects, events, goal, true)?), hrl: None, hcustom: None})
    }

    #[staticmethod]
    pub fn hrl(ss: &CoreStateEncoder, goals_vec: Vec<f32>, constants_vec: Vec<f32>, callable: PyObject) -> PyResult<Self> {
        Ok(Heuristic {hff: None, hrl: Some(HRL::new(ss, goals_vec, constants_vec, callable)?), hcustom: None})
    }

    pub fn eval(&self, state: &State, ss: &SearchSpace) -> PyResult<Option<f64>> {
        if self.hff.is_some() {
            let h = self.hff.as_ref().unwrap();
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
    }

}

#[derive(Clone)]
pub struct HRL {
    ss: CoreStateEncoder,
    goals_vec: Vec<f32>,
    constants_vec: Vec<f32>,
    callable: PyObject,
}

impl HRL {
    fn new(ss: &CoreStateEncoder, goals_vec: Vec<f32>, constants_vec: Vec<f32>, callable: PyObject) -> PyResult<Self> {
        Ok(HRL { ss: ss.clone(), goals_vec, constants_vec, callable })
    }

    pub fn eval(&self, state: &State, ss: &SearchSpace) -> PyResult<Option<f64>> {
        let mut enc: Vec<f32> = Vec::new();
        enc.extend(self.ss.get_fluents_as_vector(state)?);
        enc.extend(self.ss.get_running_actions_as_vector(state)?);
        enc.extend(self.constants_vec.iter());
        enc.extend(self.goals_vec.iter());
        enc.extend(self.ss.get_tn_as_vector(state, ss)?);
        Python::with_gil(|py| {
            let args = PyTuple::new(py, &[enc.into_py(py)]);
            let r = self.callable.call(py, args, None)?;
            if r.is_none(py) {
                Ok(None)
            } else {
                Ok(Some(r.extract(py)?))
            }
        })
    }

}

#[derive(Clone, Debug)]
pub struct CustomHeuristic {
    callable: PyObject
}

impl CustomHeuristic {
    fn new(callable: PyObject) -> PyResult<Self> {
        Ok(CustomHeuristic { callable })
    }

    pub fn eval(&self, state: &State) -> PyResult<Option<f64>> {
        Python::with_gil(|py| {
            let args = PyTuple::new(py, &[state.clone().into_py(py)]);
            let r = self.callable.call(py, args, None)?;
            if r.is_none(py) {
                Ok(None)
            } else {
                Ok(Some(r.extract(py)?))
            }
        })
    }
}


#[derive(Debug, Clone, PartialEq)]
struct Operator {
    action: String,
    conditions: Vec<Vec<ExpressionNode>>,
    effects: Vec<Vec<ExpressionNode>>,
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

fn is_numeric_condition(cond: &Vec<ExpressionNode>) -> bool {
    if let Some(e) = cond.last() {
        if let ExpressionNode::Fluent(_) = e {
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

fn cost(exp: &Vec<Vec<ExpressionNode>>, costs: &HashMap<&Vec<ExpressionNode>, f64>) -> Option<f64> {
    let mut res = 0.0;
    for g in exp.iter() {
        let c = costs.get(g);
        if let Some(cost) = c {
            res += cost;
        } else {
            return None;
        }
    }
    Some(res)
}

#[derive(Clone, Debug)]
pub struct HFF {
    events: HashMap<String, usize>,
    goals: Vec<Vec<ExpressionNode>>,
    extra_fluents: HashMap<String, Vec<Vec<ExpressionNode>>>,
    extra_goals: Vec<Vec<ExpressionNode>>,
    operators : Vec<Operator>,
    precondition_of: HashMap<Vec<ExpressionNode>, Vec<usize>>,
    empty_pre_operators: HashSet<usize>,
    numeric_conds: HashSet<Vec<ExpressionNode>>,
    return_hadd: bool,
}

impl HFF {
    fn new(
        fluents: HashMap<String, String>,
        objects: HashMap<String, Vec<String>>,
        events: HashMap<String, Vec<(Timing, Event)>>,
        goal: Vec<PyExpressionNode>,
        return_hadd: bool,
    ) -> PyResult<Self> {
        let mut operators = Vec::new();
        let mut extra_fluents = HashMap::new();
        let mut extra_goals = Vec::new();
        for (a, le) in events.iter() {
            let mut a_extra_fluents = Vec::new();
            let mut cond: Vec<ExpressionNode> = vec![ExpressionNode::Fluent(format!("__f_{}_{}", a, le.len()-1))];
            extra_goals.push(cond.clone());
            for (i, (_, e)) in le.iter().enumerate() {
                let mut effects = Vec::new();
                let mut conditions = Vec::new();
                let f = format!("__f_{}_{}", a, i);
                a_extra_fluents.push(vec![ExpressionNode::Fluent(f.to_string())]);
                effects.push(vec![ExpressionNode::Fluent(f.to_string())]);
                for eff in e.effects.iter() {
                    let t = fluents[&eff.fluent].to_string();
                    if t == "bool" {
                        if eff.value.len() == 1 {
                            if let ExpressionNode::Bool(value) = eff.value[0] {
                                if value {
                                    effects.push(vec![ExpressionNode::Fluent(eff.fluent.to_string())]);
                                } else {
                                    effects.push(vec![ExpressionNode::Fluent(eff.fluent.to_string()), make_operator("not".to_string(), vec![0])?]);
                                }
                            }
                            else {
                                effects.push(vec![ExpressionNode::Fluent(eff.fluent.to_string())]);
                                effects.push(vec![ExpressionNode::Fluent(eff.fluent.to_string()), make_operator("not".to_string(), vec![0])?]);
                            }
                        } else {
                            effects.push(vec![ExpressionNode::Fluent(eff.fluent.to_string())]);
                            effects.push(vec![ExpressionNode::Fluent(eff.fluent.to_string()), make_operator("not".to_string(), vec![0])?]);
                        }
                    } else if t != "real" && t != "int" {
                        if eff.value.len() == 1 {
                            if let ExpressionNode::Object(_) = eff.value[0] {
                                effects.push(vec![ExpressionNode::Fluent(eff.fluent.to_string()), eff.value[0].clone(), make_operator("==".to_string(), vec![0, 1])?]);
                            }
                            else {
                                for o in objects[&t].iter() {
                                    effects.push(vec![ExpressionNode::Fluent(eff.fluent.to_string()), ExpressionNode::Object(o.to_string()), make_operator("==".to_string(), vec![0, 1])?]);
                                }
                            }
                        } else {
                            for o in objects[&t].iter() {
                                effects.push(vec![ExpressionNode::Fluent(eff.fluent.to_string()), ExpressionNode::Object(o.to_string()), make_operator("==".to_string(), vec![0, 1])?]);
                            }
                        }
                    }
                }
                conditions.push(cond);
                if e.conditions.len() > 0 && e.conditions != vec![ExpressionNode::Bool(true)] {
                    conditions.extend(split_expression(&e.conditions)?);
                }
                operators.push(Operator { action: a.to_string(), conditions, effects, cost: 1.0 } );
                cond = vec![ExpressionNode::Fluent(f.to_string())];
            }
            extra_fluents.insert(a.to_string(), a_extra_fluents);
        }
        operators.sort_by(|a, b| a.action.cmp(&b.action));

        let goals = split_expression(&goal.into_iter().map(|e| e.v).collect())?;
        let mut precondition_of: HashMap<Vec<ExpressionNode>, Vec<usize>> = HashMap::new();
        let mut numeric_conds: HashSet<Vec<ExpressionNode>> = HashSet::new();
        let mut empty_pre_operators: HashSet<usize> = HashSet::new();
        for (idx_o, o) in operators.iter().enumerate() {
            if o.conditions.len() == 0 || o.conditions == vec![vec![ExpressionNode::Bool(true)]] {
                empty_pre_operators.insert(idx_o);
            }
            for c in o.conditions.iter() {
                if is_numeric_condition(c) {
                    numeric_conds.insert(c.to_vec());
                } else {
                    if ! precondition_of.contains_key(c) {
                        precondition_of.insert(c.to_vec(), Vec::new());
                    }
                    precondition_of.get_mut(c).unwrap().push(idx_o);
                }
            }
        }
        for c in goals.iter() {
            if is_numeric_condition(c) {
                numeric_conds.insert(c.to_vec());
            }
        }

        let events_len: HashMap<String, usize> = events.iter().map(|(a, ev)| (a.to_string(), ev.len())).collect();
        let res = HFF {
            events: events_len,
            goals,
            extra_fluents,
            extra_goals,
            operators,
            precondition_of,
            empty_pre_operators,
            numeric_conds,
            return_hadd
        };
        Ok(res)
    }

    pub fn eval(&self, state: &State) -> PyResult<Option<f64>> {
        let mut costs : HashMap<&Vec<ExpressionNode>, f64> = HashMap::new();
        let mut lp : Vec<&Vec<ExpressionNode>> = Vec::new();
        let mut init_lp : Vec<Vec<ExpressionNode>> = Vec::new();

        for (f, v) in state.assignments.iter() {
            let k = match v {
                ExpressionNode::Bool(value) => {
                    if *value {
                        vec![ExpressionNode::Fluent(f.to_string())]
                    } else {
                        vec![ExpressionNode::Fluent(f.to_string()), make_operator("not".to_string(), vec![0])?]
                    }
                }
                _ => {
                    vec![ExpressionNode::Fluent(f.to_string()), v.clone(), make_operator("==".to_string(), vec![0, 1])?]
                }
            };
            init_lp.push(k);
        }
        for k in init_lp.iter() {
            costs.insert(k, 0.0);
            lp.push(k);
        }

        for c in self.numeric_conds.iter() {
            if internal_evaluate(c, state)? == ExpressionNode::Bool(true) {
                costs.insert(c, 0.0);
            } else {
                costs.insert(c, 1.0);
            }
            lp.push(c);
        }

        for a in self.events.keys() {
            let v = match state.todo.get(a) {
                Some((j, _)) => self.extra_fluents.get(a).unwrap().get(j-1),
                None => self.extra_fluents.get(a).unwrap().last(),
            };
            if let Some(x) = v {
                costs.insert(x, 0.0);
                lp.push(x);
            }
        }

        let mut reached_by : HashMap<&Vec<ExpressionNode>, usize> = HashMap::new();
        while lp.len() > 0 {
            let mut lo: HashSet<usize> = HashSet::new();
            for x in self.empty_pre_operators.iter() {
                lo.insert(*x);
            }
            for p in lp.iter() {
                if let Some(po) = self.precondition_of.get(*p) {
                    for idx_o in po.iter() {
                        lo.insert(*idx_o);
                    }
                }
            }
            lp.clear();
            let mut new_costs = HashMap::new();
            for idx_o in lo {
                let o: &Operator = &self.operators[idx_o];
                if let Some(c) = cost(&o.conditions, &costs) {
                    for k in o.effects.iter() {
                        let new_cost_k = new_costs.get(k);
                        let cost_k = costs.get(k);
                        if (new_cost_k.is_some() && *new_cost_k.unwrap() > c + o.cost) ||
                        (new_cost_k.is_none() && cost_k.is_none()) ||
                        (new_cost_k.is_none() && *cost_k.unwrap() > c + o.cost) {
                            reached_by.insert(k, idx_o);
                            new_costs.insert(k, c + o.cost);
                            lp.push(k);
                        } else if ((new_cost_k.is_some() && *new_cost_k.unwrap() == c + o.cost) ||
                        (new_cost_k.is_none() && *cost_k.unwrap() == c + o.cost)) && idx_o > reached_by[k] {
                            reached_by.insert(k, idx_o);
                        }
                    }
                }
            }
            for (k, v) in new_costs.iter() {
                costs.insert(*k, *v);
            }
        }

        let h = cost(&self.goals, &costs);

        if h.is_none() {
            return Ok(None);
        }

        if self.return_hadd {
            match cost(&self.extra_goals, &costs) {
                Some(v) => {
                    return Ok(Some(h.unwrap() + v));
                },
                None => {
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
                return Ok(Some(res));
            }
        }

        let mut relaxed_plan = HashSet::new();
        let mut stack: Vec<&Vec<ExpressionNode>> = self.goals.iter().collect();
        while stack.len() > 0 {
            let g = stack.pop().unwrap();
            if let Some(idx_o) = reached_by.get(g) {
                let o: &Operator = &self.operators[*idx_o];
                relaxed_plan.insert(o.action.to_string());
                for c in o.conditions.iter() {
                    stack.push(c);
                }
            }
        }
        for a in relaxed_plan.iter() {
            if ! state.todo.contains_key(a) {
                res += self.events[a] as f64;
            }
        }

        Ok(Some(res))
    }
}
