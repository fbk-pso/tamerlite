use std::collections::HashSet;
use std::{
    collections::HashMap,
    vec::Vec
};
use std::hash::{Hash, Hasher};

use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::evaluate;

use super::search_space::State;
use super::expressions::*;
use super::structures::*;


#[pyclass]
pub struct Heuristic {
    hff: Option<HFF>,
    hcustom: Option<CustomHeuristic>,
}

#[pymethods]
impl Heuristic {

    #[staticmethod]
    pub fn custom(callable: PyObject) -> PyResult<Self> {
        Ok(Heuristic {hff: None, hcustom: Some(CustomHeuristic::new(callable)?)})
    }

    #[staticmethod]
    pub fn hff(fluents: HashMap<String, String>, objects: HashMap<String, Vec<String>>, events: HashMap<String, Vec<(Timing, Event)>>, goal: Vec<PyExpressionNode>) -> PyResult<Self> {
        Ok(Heuristic {hff: Some(HFF::new(fluents, objects, events, goal, false)?), hcustom: None})
    }

    #[staticmethod]
    pub fn hadd(fluents: HashMap<String, String>, objects: HashMap<String, Vec<String>>, events: HashMap<String, Vec<(Timing, Event)>>, goal: Vec<PyExpressionNode>) -> PyResult<Self> {
        Ok(Heuristic {hff: Some(HFF::new(fluents, objects, events, goal, true)?), hcustom: None})
    }

    pub fn eval(&self, state: &State) -> PyResult<Option<f32>> {
        if self.hff.is_some() {
            let h = self.hff.as_ref().unwrap();
            h.eval(state)
        } else if self.hcustom.is_some() {
            let h = self.hcustom.as_ref().unwrap();
            h.eval(state)
        } else {
            Ok(Some(0.0))
        }
    }

}

pub struct CustomHeuristic {
    callable: PyObject
}

impl CustomHeuristic {
    fn new(callable: PyObject) -> PyResult<Self> {
        Ok(CustomHeuristic { callable })
    }

    pub fn eval(&self, state: &State) -> PyResult<Option<f32>> {
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
    conditions: Vec<Vec<PyExpressionNode>>,
    effects: Vec<(String, PyExpressionNode)>,
    cost: f32,
}

impl Eq for Operator {}

impl Hash for Operator {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.action.hash(state);
        self.conditions.hash(state);
        self.effects.hash(state);
    }
}

fn is_numeric_condition(cond: &Vec<PyExpressionNode>) -> bool {
    if let Some(e) = cond.last() {
        if let ExpressionNode::Fluent(_) = e.to_expression_node() {
            return false;
        } else if let ExpressionNode::Not(i) = e.to_expression_node() {
            if let ExpressionNode::Fluent(_) = cond[i].to_expression_node() {
                return false;
            }
        } else if let ExpressionNode::Equals(i1, i2) = e.to_expression_node() {
            if let ExpressionNode::Fluent(_) = cond[i1].to_expression_node() {
                if let ExpressionNode::Object(_) = cond[i2].to_expression_node() {
                    return false;
                }
            }
        }
    } else {
        return false;
    }
    true
}

fn cost(exp: &Vec<Vec<PyExpressionNode>>, costs: &HashMap<Vec<PyExpressionNode>, f32>) -> Option<f32> {
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

pub struct HFF {
    events: HashMap<String, usize>,
    goals: Vec<Vec<PyExpressionNode>>,
    extra_fluents: HashMap<String, Vec<Vec<PyExpressionNode>>>,
    extra_goals: Vec<Vec<PyExpressionNode>>,
    precondition_of: HashMap<Vec<PyExpressionNode>, Vec<Operator>>,
    empty_pre_operators: Vec<Operator>,
    numeric_conds: HashSet<Vec<PyExpressionNode>>,
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
            let mut cond: Vec<PyExpressionNode> = vec![make_fluent_node(format!("__f_{}_{}", a, le.len()-1))];
            extra_goals.push(cond.clone());
            for (i, (_, e)) in le.iter().enumerate() {
                let mut effects = Vec::new();
                let mut conditions = Vec::new();
                let f = format!("__f_{}_{}", a, i);
                a_extra_fluents.push(vec![make_fluent_node(f.to_string())]);
                effects.push((f.to_string(), make_bool_constant_node(true)));
                for eff in e.effects.iter() {
                    let t = fluents[&eff.fluent].clone();
                    if t == "bool" {
                        if eff.value.len() == 1 {
                            if let ExpressionNode::Bool(_) = eff.value[0].to_expression_node() {
                                effects.push((eff.fluent.to_string(), eff.value[0].clone()));
                            }
                            else {
                                effects.push((eff.fluent.to_string(), make_bool_constant_node(true)));
                                effects.push((eff.fluent.to_string(), make_bool_constant_node(false)));
                            }
                        } else {
                            effects.push((eff.fluent.to_string(), make_bool_constant_node(true)));
                            effects.push((eff.fluent.to_string(), make_bool_constant_node(false)));
                        }
                    } else if t != "real" && t != "int" {
                        if eff.value.len() == 1 {
                            if let ExpressionNode::Object(_) = eff.value[0].to_expression_node() {
                                effects.push((eff.fluent.to_string(), eff.value[0].clone()));
                            }
                            else {
                                for o in objects[&t].iter() {
                                    effects.push((eff.fluent.to_string(), make_object_node(o.to_string())));
                                }
                            }
                        } else {
                            for o in objects[&t].iter() {
                                effects.push((eff.fluent.to_string(), make_object_node(o.to_string())));
                            }
                        }
                    }
                }
                conditions.push(cond);
                if e.conditions.len() > 0 && e.conditions != vec![make_bool_constant_node(true)] {
                    conditions.extend(split_expression(&e.conditions)?);
                }
                operators.push(Operator { action: a.to_string(), conditions, effects, cost: 1.0 } );
                cond = vec![make_fluent_node(f.to_string())];
            }
            extra_fluents.insert(a.to_string(), a_extra_fluents);
        }

        let goals = split_expression(&goal)?;
        let mut precondition_of: HashMap<Vec<PyExpressionNode>, Vec<Operator>> = HashMap::new();
        let mut numeric_conds: HashSet<Vec<PyExpressionNode>> = HashSet::new();
        let mut empty_pre_operators: Vec<Operator> = Vec::new();
        for o in operators.iter() {
            if o.conditions.len() == 0 || o.conditions == vec![vec![make_bool_constant_node(true)]] {
                empty_pre_operators.push(o.clone());
            }
            for c in o.conditions.iter() {
                if is_numeric_condition(c) {
                    numeric_conds.insert(c.to_vec());
                } else {
                    if ! precondition_of.contains_key(c) {
                        precondition_of.insert(c.to_vec(), Vec::new());
                    }
                    precondition_of.get_mut(c).unwrap().push(o.clone());
                }
            }
        }

        let events_len: HashMap<String, usize> = events.iter().map(|(a, ev)| (a.to_string(), ev.len())).collect();
        let res = HFF {
            events: events_len,
            goals,
            extra_fluents,
            extra_goals,
            precondition_of,
            empty_pre_operators,
            numeric_conds,
            return_hadd
        };
        Ok(res)
    }

    pub fn eval(&self, state: &State) -> PyResult<Option<f32>> {
        let mut costs = HashMap::new();
        let mut lp = Vec::new();

        for (f, v) in state.assignments.iter() {
            let k = match v.v {
                ExpressionNode::Bool(value) => {
                    if value {
                        vec![make_fluent_node(f.to_string())]
                    } else {
                        vec![make_fluent_node(f.to_string()), make_operator_node("not".to_string(), vec![0])?]
                    }
                }
                _ => {
                    vec![make_fluent_node(f.to_string()), v.clone(), make_operator_node("==".to_string(), vec![0, 1])?]
                }
            };
            costs.insert(k.to_vec(), 0.0);
            lp.push(k.to_vec());
        }

        for c in self.numeric_conds.iter() {
            if evaluate(c.to_vec(), state)?.v == ExpressionNode::Bool(true) {
                costs.insert(c.to_vec(), 0.0);
            }
            else {
                costs.insert(c.to_vec(), 1.0);
            }
            lp.push(c.to_vec());
        }

        for a in self.events.keys() {
            let v = match state.todo.get(a) {
                Some((j, _)) => self.extra_fluents.get(a).unwrap().get(j-1),
                None => self.extra_fluents.get(a).unwrap().last(),
            };
            if let Some(x) = v {
                costs.insert(x.to_vec(), 0.0);
                lp.push(x.to_vec());
            }
        }

        let mut reached_by = HashMap::new();
        while lp.len() > 0 {
            let mut lo = self.empty_pre_operators.clone();
            for p in lp.iter() {
                if let Some(po) = self.precondition_of.get(p) {
                    lo.extend(po.to_vec());
                }
            }
            lp.clear();
            let mut new_costs = HashMap::new();
            let lo_iter: std::collections::HashSet<_> = lo.into_iter().collect();
            for o in lo_iter.iter() {
                if let Some(c) = cost(&o.conditions, &costs) {
                    for (f, v) in o.effects.iter() {
                        let k = match v.v {
                            ExpressionNode::Bool(value) => {
                                if value {
                                    vec![make_fluent_node(f.to_string())]
                                } else {
                                    vec![make_fluent_node(f.to_string()), make_operator_node("not".to_string(), vec![0])?]
                                }
                            }
                            _ => {
                                vec![make_fluent_node(f.to_string()), v.clone(), make_operator_node("==".to_string(), vec![0, 1])?]
                            }
                        };
                        let new_cost_k = new_costs.get(&k);
                        let cost_k = costs.get(&k);
                        if (new_cost_k.is_some() && *new_cost_k.unwrap() > c + o.cost) ||
                        (new_cost_k.is_none() && cost_k.is_none()) ||
                        (new_cost_k.is_none() && *cost_k.unwrap() > c + o.cost) {
                            reached_by.insert(k.clone(), o.clone());
                            new_costs.insert(k.clone(), c + o.cost);
                            lp.push(k.clone());
                        }
                    }
                }
            }
            for (k, v) in new_costs.iter() {
                costs.insert(k.to_vec(), *v);
            }
        }

        let h = cost(&self.goals, &costs);

        if h.is_none() {
            return Ok(None);
        }

        if self.return_hadd {
            match cost(&self.extra_goals, &costs) {
                Some(v) => {
                    return Ok(Some(h.unwrap() + 2.0*v));
                },
                None => {
                    return Ok(None);
                }
            };
        }

        let mut res = 0.0;
        for (a, (j, _)) in state.todo.iter() {
            res += (self.events[a] - j) as f32;
        }

        if let Some(hv) = h {
            if hv == 0.0 {
                return Ok(Some(res));
            }
        }

        let mut relaxed_plan = HashSet::new();
        let mut stack = self.goals.clone();
        while stack.len() > 0 {
            let g = stack.pop().unwrap();
            if let Some(o) = reached_by.get(&g) {
                relaxed_plan.insert(o.action.to_string());
                for c in o.conditions.iter() {
                    stack.push(c.clone());
                }
            }
        }
        for a in relaxed_plan.iter() {
            if ! state.todo.contains_key(a) {
                res += self.events[a] as f32;
            }
        }

        Ok(Some(res))
    }
}
