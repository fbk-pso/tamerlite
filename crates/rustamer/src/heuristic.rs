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

use pyo3::exceptions::PyNotImplementedError;
use pyo3::prelude::*;
use rustamer_base::*;
use rustc_hash::{FxHashMap, FxHashSet};
use std::vec::Vec;

#[pyclass(frozen)]
#[derive(Clone)]
pub struct Heuristic {
    hdr: Option<DeleteRelaxationHeuristic>,
    hmax_explicit: Option<HMaxExplicit>,
    hcustom: Option<CustomHeuristic>,
    cache_value_in_state: bool,
}

#[pymethods]
impl Heuristic {
    #[staticmethod]
    pub fn custom(callable: Py<PyAny>, cache_value_in_state: bool) -> PyResult<Self> {
        Ok(Heuristic {
            hdr: None,
            hmax_explicit: None,
            hcustom: Some(CustomHeuristic::new(callable)?),
            cache_value_in_state,
        })
    }

    #[staticmethod]
    #[pyo3(signature = (actions, fluent_types, objects, events, goals, internal_caching, cache_value_in_state, inadmissible_numeric_heuristic_variant, disable_numeric_reasoning=false))]
    pub fn hff(
        actions: Vec<Action>,
        fluent_types: Vec<String>,
        objects: FxHashMap<String, Vec<String>>,
        events: FxHashMap<Action, Vec<(Timing, Event)>>,
        goals: Vec<PyExpressionNode>,
        internal_caching: bool,
        cache_value_in_state: bool,
        inadmissible_numeric_heuristic_variant: bool,
        disable_numeric_reasoning: bool,
    ) -> PyResult<Self> {
        Ok(Heuristic {
            hdr: Some(DeleteRelaxationHeuristic::new(
                actions,
                fluent_types,
                objects,
                events,
                goals,
                HeuristicKind::HFF,
                internal_caching,
                inadmissible_numeric_heuristic_variant,
                disable_numeric_reasoning,
            )?),
            hmax_explicit: None,
            hcustom: None,
            cache_value_in_state,
        })
    }

    #[staticmethod]
    #[pyo3(signature = (actions, fluent_types, objects, events, goals, internal_caching, cache_value_in_state, inadmissible_numeric_heuristic_variant, disable_numeric_reasoning=false))]
    pub fn hadd(
        actions: Vec<Action>,
        fluent_types: Vec<String>,
        objects: FxHashMap<String, Vec<String>>,
        events: FxHashMap<Action, Vec<(Timing, Event)>>,
        goals: Vec<PyExpressionNode>,
        internal_caching: bool,
        cache_value_in_state: bool,
        inadmissible_numeric_heuristic_variant: bool,
        disable_numeric_reasoning: bool,
    ) -> PyResult<Self> {
        Ok(Heuristic {
            hdr: Some(DeleteRelaxationHeuristic::new(
                actions,
                fluent_types,
                objects,
                events,
                goals,
                HeuristicKind::HADD,
                internal_caching,
                inadmissible_numeric_heuristic_variant,
                disable_numeric_reasoning,
            )?),
            hmax_explicit: None,
            hcustom: None,
            cache_value_in_state,
        })
    }

    #[staticmethod]
    #[pyo3(signature = (actions, fluent_types, objects, events, goals, internal_caching, cache_value_in_state, inadmissible_numeric_heuristic_variant, disable_numeric_reasoning=false))]
    pub fn hmax(
        actions: Vec<Action>,
        fluent_types: Vec<String>,
        objects: FxHashMap<String, Vec<String>>,
        events: FxHashMap<Action, Vec<(Timing, Event)>>,
        goals: Vec<PyExpressionNode>,
        internal_caching: bool,
        cache_value_in_state: bool,
        inadmissible_numeric_heuristic_variant: bool,
        disable_numeric_reasoning: bool,
    ) -> PyResult<Self> {
        Ok(Heuristic {
            hdr: Some(DeleteRelaxationHeuristic::new(
                actions,
                fluent_types,
                objects,
                events,
                goals,
                HeuristicKind::HMAX,
                internal_caching,
                inadmissible_numeric_heuristic_variant,
                disable_numeric_reasoning,
            )?),
            hmax_explicit: None,
            hcustom: None,
            cache_value_in_state,
        })
    }

    #[staticmethod]
    #[allow(unused_variables)]
    pub fn hmax_explicit(
        actions: Vec<Action>,
        fluent_types: Vec<String>,
        objects: FxHashMap<String, Vec<String>>,
        events: FxHashMap<Action, Vec<(Timing, Event)>>,
        goals: Vec<PyExpressionNode>,
        internal_caching: bool,
        cache_value_in_state: bool,
        inadmissible_numeric_heuristic_variant: bool,
    ) -> PyResult<Self> {
        Ok(Heuristic {
            hdr: None,
            hmax_explicit: Some(HMaxExplicit::new(
                actions,
                fluent_types,
                events,
                goals,
                internal_caching,
            )?),
            hcustom: None,
            cache_value_in_state,
        })
    }

    #[getter]
    pub fn name(&self) -> &'static str {
        if let Some(h) = &self.hdr {
            h.name()
        } else if let Some(h) = &self.hmax_explicit {
            h.name()
        } else if let Some(h) = &self.hcustom {
            h.name()
        } else {
            unreachable!("One of hdr, hmax_explicit, or hcustom must be set")
        }
    }

    #[pyo3(name = "eval")]
    pub fn py_eval(&self, state: &State, ss: &SearchSpace) -> PyResult<Option<f64>> {
        self.eval(state, ss)
    }

    pub fn reachable_actions(&self, state: &State) -> PyResult<FxHashSet<Action>> {
        if let Some(h) = &self.hdr {
            h.reachable_actions(state)
        } else {
            Err(PyNotImplementedError::new_err(
                "reachable_actions is not available: hdr is None",
            ))
        }
    }
}

impl HeuristicTrait for Heuristic {
    fn eval<S: SearchSpaceTrait>(&self, state: &State, _ss: &S) -> PyResult<Option<f64>> {
        if self.cache_value_in_state {
            let heuristic_cache = state.heuristic_cache.lock().unwrap();
            if let Some(h_value) = heuristic_cache.get(&self.name()) {
                return Ok(*h_value);
            }
        }
        let h_value = {
            if let Some(h) = &self.hdr {
                h.eval(state)
            } else if let Some(h) = &self.hmax_explicit {
                h.eval(state)
            } else if let Some(h) = &self.hcustom {
                h.eval(state)
            } else {
                unreachable!("One of hdr, hmax_explicit, or hcustom must be set")
            }
        };
        if self.cache_value_in_state {
            let mut heuristic_cache = state.heuristic_cache.lock().unwrap();
            if let Ok(h_value) = h_value {
                heuristic_cache.insert(self.name(), h_value);
            }
        }
        h_value
    }
}
