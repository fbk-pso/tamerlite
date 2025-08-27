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

use std::{collections::HashMap, vec::Vec};
use pyo3::prelude::*;
use rustamerlib::*;


#[pyclass(frozen)]
#[derive(Clone)]
pub struct Heuristic {
    hdr: Option<DeleteRelaxationHeuristic>,
    hmax: Option<HMaxNumeric>,
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
            hcustom: None,
            cache_value_in_state: cache_value_in_state,
        })
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
        } else {
            String::from("")
        }
    }
}

impl HeuristicTrait for Heuristic {
    fn eval(&self, state: &State, _ss: &SearchSpace) -> PyResult<Option<f64>> {
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
}