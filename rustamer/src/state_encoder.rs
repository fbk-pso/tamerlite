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

use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use std::{collections::HashMap, vec::Vec};

use crate::utils::{integer_to_f32, rational_to_f32, usize_to_f32};
use crate::{ExpressionNode, SearchSpace};

use super::search_state::State;
use super::structures::*;

#[pyclass]
#[derive(Clone)]
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
            match &v {
                ExpressionNode::Bool(v) => {
                    if *v {
                        res.push(1.0);
                    } else {
                        res.push(0.0);
                    }
                }
                ExpressionNode::Int(v) => {
                    let f = integer_to_f32(v);
                    if lb.is_some() && ub.is_some() {
                        res.push((f - lb.unwrap()) / (ub.unwrap() - lb.unwrap()));
                    } else {
                        res.push(f);
                    }
                }
                ExpressionNode::Rational(v) => {
                    let f = rational_to_f32(v);
                    if lb.is_some() && ub.is_some() {
                        res.push((f - lb.unwrap()) / (ub.unwrap() - lb.unwrap()));
                    } else {
                        res.push(f);
                    }
                }
                ExpressionNode::Object(v) => {
                    res.push(self.objects[v]);
                }
                _ => {
                    return Err(PyException::new_err("State assignment is not a constant!"));
                }
            }
        }
        Ok(res)
    }

    pub fn get_running_actions_as_vector(&self, state: &State) -> PyResult<Vec<f32>> {
        let mut res = vec![0.0; self.num_actions];
        for (a, i) in self.actions_pos.iter() {
            let v = match state.todo.get(a) {
                Some((x, _)) => self.events[a].len() - x,
                None => 0,
            };
            res[*i] = usize_to_f32(v);
        }
        Ok(res)
    }

    pub fn get_tn_as_vector(
        &self,
        state: &State,
        search_space: &SearchSpace,
    ) -> PyResult<Vec<f32>> {
        let mut res = vec![0.0; self.tn_size];
        if state.temporal_network.is_some() {
            let tn = state.temporal_network.as_ref().unwrap();
            let mut last = -1.0;
            if state.path.is_some() {
                let payload = &state.path.as_ref().unwrap().payload;
                last = search_space
                    .tn_interpreter
                    .get_event_timing(tn, &payload.0, payload.1, payload.2)
                    .unwrap();
            }

            let mut m = HashMap::new();
            for (ev, t) in search_space.tn_interpreter.get_events_timings(tn) {
                if t - last >= tn.tolerance {
                    break;
                }
                m.insert(ev, t);
            }

            let mut t_safe = 0.0;
            let mut t_last = 0.0;
            let mut c = 0;
            let mut nea = 0;
            let mut nsa = 0;
            for (ev, t) in search_space.tn_interpreter.get_actions_timings(tn).iter() {
                if !tn.equals_with_tolerance(t, &t_last) {
                    c -= nea;
                    if c == 0 {
                        t_safe = t_last;
                    }
                    c += nsa;
                    nsa = 0;
                    nea = 0;
                }
                if !tn.equals_with_tolerance(
                    &search_space
                        .tn_interpreter
                        .get_action_timing(tn, &ev.0, !ev.1, ev.2)
                        .unwrap(),
                    t,
                ) {
                    if ev.1 {
                        nsa += 1;
                    } else {
                        nea += 1;
                    }
                }
                t_last = *t;
                if t - last >= -tn.tolerance {
                    break;
                }
            }

            for (e, t) in m.iter() {
                if *t - t_safe > tn.tolerance {
                    let a = &e.0;
                    let p = self.tn_actions_pos[a];
                    let v = *t - t_safe + 1.0;
                    if v > res[p + e.1] {
                        res[p + e.1] = v;
                    }
                }
            }
        }
        Ok(res)
    }
}
