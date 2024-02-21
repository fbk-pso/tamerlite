use std::{
    collections::HashMap,
    vec::Vec
};
use pyo3::exceptions::PyException;
use pyo3::prelude::*;

use crate::utils::{integer_to_f32, rational_to_f32, usize_to_f32};
use crate::ExpressionNode;

use super::search_space::State;
use super::structures::*;

#[pyclass]
pub struct CoreStateEncoder {
    num_actions: usize,
    tn_size: usize,
    fluents: Vec<(String, bool, (Option<f32>, Option<f32>))>,
    actions_pos: HashMap<String, usize>,
    tn_actions_pos: HashMap<String, usize>,
    objects: HashMap<String, f32>,
    events: HashMap<String, Vec<(Timing, Event)>>,
}

#[pymethods]
impl CoreStateEncoder {
    #[new]
    fn new(
        num_actions: usize,
        tn_size: usize,
        fluents: Vec<(String, bool, (Option<f32>, Option<f32>))>,
        actions_pos: HashMap<String, usize>,
        tn_actions_pos: HashMap<String, usize>,
        objects: HashMap<String, f32>,
        events: HashMap<String, Vec<(Timing, Event)>>,
    ) -> PyResult<Self> {
        let res = CoreStateEncoder {
            num_actions: num_actions,
            tn_size: tn_size,
            fluents: fluents,
            actions_pos: actions_pos,
            tn_actions_pos: tn_actions_pos,
            objects: objects,
            events: events,
        };
        Ok(res)
    }

    pub fn get_fluents_as_vector(&self, state: &State) -> PyResult<Vec<f32>> {
        let mut res = Vec::new();
        for (sfe, _, (lb, ub)) in self.fluents.iter() {
            let v = state.get_value(sfe);
            match v.to_expression_node() {
                ExpressionNode::Bool(v) => {
                    if v {
                        res.push(1.0);
                    }
                    else {
                        res.push(0.0);
                    }
                },
                ExpressionNode::Int(v) => {
                    let f = integer_to_f32(v);
                    if lb.is_some() && ub.is_some() {
                        res.push((f - lb.unwrap()) / (ub.unwrap() - lb.unwrap()));
                    } else {
                        res.push(f);
                    }
                },
                ExpressionNode::Rational(v) => {
                    let f = rational_to_f32(v);
                    if lb.is_some() && ub.is_some() {
                        res.push((f - lb.unwrap()) / (ub.unwrap() - lb.unwrap()));
                    } else {
                        res.push(f);
                    }
                },
                ExpressionNode::Object(v) => {
                    res.push(self.objects[&v]);
                },
                _ => {
                    return Err(PyException::new_err("State assignment is not a constant!"));
                },
            }
        }
        Ok(res)
    }

    pub fn get_running_actions_as_vector(&self, state: &State) -> PyResult<Vec<f32>> {
        let mut res = vec![0.0; self.num_actions];
        for (a, i) in self.actions_pos.iter() {
            let v = match state.todo.get(a) {
                Some((x, _)) => {
                    self.events[a].len() - x
                },
                None => 0,
            };
            res[*i] = usize_to_f32(v);
        }
        Ok(res)
    }

    pub fn get_tn_as_vector(&self, state: &State) -> PyResult<Vec<f32>> {
        let mut res = vec![0.0; self.tn_size];
        if state.temporal_network.is_some() {
            let tn = state.temporal_network.as_ref().unwrap();
            let mut last = -1.0;
            if !state.path.is_empty() {
                let ev = tn.get_option_event_id(state.path.last().unwrap().clone()).unwrap();
                last = tn.get_model_value(&ev).unwrap();
            }
            let mut vec = Vec::new();
            for (ev, t) in tn.get_actions_timings() {
                if t < last {
                    vec.push((t, ev.1, ev.0));
                }
            }
            vec.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

            let mut m: HashMap<Event, f32> = HashMap::new();
            for (ev, t) in tn.get_events_timings() {
                if t < last {
                    match m.get(&ev.0) {
                        Some(v) => {
                            if t > *v {
                                m.insert(ev.0, t);
                            }
                        },
                        None => {
                            m.insert(ev.0, t);
                        }
                    }
                }
            }

            let mut t_safe = 0.0;
            let mut t_last = 0.0;
            let mut c = 0;
            let mut nea = 0;
            let mut nsa = 0;
            let mut actions: Vec<String> = Vec::new();
            for (t, is_start, action) in vec.iter() {
                if tn.equals_with_tolerance(t, &last) {
                    break;
                }
                if !tn.equals_with_tolerance(t, &t_last) {
                    c -= nea;
                    if c == 0 {
                        t_safe = t_last;
                        actions.clear()
                    }
                    c += nsa;
                    nsa = 0;
                    nea = 0;
                }
                if *is_start {
                    nsa += 1;
                    actions.push(action.to_string());
                } else {
                    nea += 1;
                }
                t_last = *t;
            }

            for a in actions.iter() {
                let p = self.tn_actions_pos[a];
                for (i, (_, e)) in self.events[a].iter().enumerate() {
                    let v = match m.get(e) {
                        Some(x) => {
                            if x - t_safe <= tn.tolerance {
                                continue
                            }
                            x - t_safe + 1.0
                        },
                        None => continue,
                    };
                    res[p+i] = v;
                }
            }
        }
        Ok(res)
    }

}
