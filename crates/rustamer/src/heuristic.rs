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

#[derive(Clone)]
enum HeuristicVariant {
    DeleteRelaxation(DeleteRelaxationHeuristic),
    HMaxExplicit(HMaxExplicit),
    Custom(CustomHeuristic),
}

#[pyclass(frozen, from_py_object)]
#[derive(Clone)]
pub struct Heuristic {
    variant: HeuristicVariant,
    cache_value_in_state: bool,
}

#[pymethods]
impl Heuristic {
    #[staticmethod]
    pub fn custom(callable: Py<PyAny>, cache_value_in_state: bool) -> PyResult<Self> {
        Ok(Heuristic {
            variant: HeuristicVariant::Custom(CustomHeuristic::new(callable)?),
            cache_value_in_state,
        })
    }

    #[staticmethod]
    #[pyo3(signature = (actions, fluent_types, objects, events, goals, internal_caching, cache_value_in_state, inadmissible_numeric_heuristic_variant, disable_numeric_reasoning=false))]
    #[allow(clippy::too_many_arguments)]
    pub fn hff(
        actions: Vec<Action>,
        fluent_types: Vec<String>,
        objects: FxHashMap<String, Vec<usize>>,
        events: FxHashMap<Action, Vec<(Timing, Event)>>,
        goals: Vec<PyExpressionNode>,
        internal_caching: bool,
        cache_value_in_state: bool,
        inadmissible_numeric_heuristic_variant: bool,
        disable_numeric_reasoning: bool,
    ) -> PyResult<Self> {
        Ok(Heuristic {
            variant: HeuristicVariant::DeleteRelaxation(DeleteRelaxationHeuristic::new(
                actions,
                fluent_types,
                objects,
                events,
                goals,
                DeleteRelaxationHeuristicConfig {
                    heuristic_kind: HeuristicKind::HFF,
                    internal_caching,
                    inadmissible_numeric_heuristic_variant,
                    disable_numeric_reasoning,
                },
            )?),
            cache_value_in_state,
        })
    }

    #[staticmethod]
    #[pyo3(signature = (actions, fluent_types, objects, events, goals, internal_caching, cache_value_in_state, inadmissible_numeric_heuristic_variant, disable_numeric_reasoning=false))]
    #[allow(clippy::too_many_arguments)]
    pub fn hadd(
        actions: Vec<Action>,
        fluent_types: Vec<String>,
        objects: FxHashMap<String, Vec<usize>>,
        events: FxHashMap<Action, Vec<(Timing, Event)>>,
        goals: Vec<PyExpressionNode>,
        internal_caching: bool,
        cache_value_in_state: bool,
        inadmissible_numeric_heuristic_variant: bool,
        disable_numeric_reasoning: bool,
    ) -> PyResult<Self> {
        Ok(Heuristic {
            variant: HeuristicVariant::DeleteRelaxation(DeleteRelaxationHeuristic::new(
                actions,
                fluent_types,
                objects,
                events,
                goals,
                DeleteRelaxationHeuristicConfig {
                    heuristic_kind: HeuristicKind::HADD,
                    internal_caching,
                    inadmissible_numeric_heuristic_variant,
                    disable_numeric_reasoning,
                },
            )?),
            cache_value_in_state,
        })
    }

    #[staticmethod]
    #[pyo3(signature = (actions, fluent_types, objects, events, goals, internal_caching, cache_value_in_state, inadmissible_numeric_heuristic_variant, disable_numeric_reasoning=false))]
    #[allow(clippy::too_many_arguments)]
    pub fn hmax(
        actions: Vec<Action>,
        fluent_types: Vec<String>,
        objects: FxHashMap<String, Vec<usize>>,
        events: FxHashMap<Action, Vec<(Timing, Event)>>,
        goals: Vec<PyExpressionNode>,
        internal_caching: bool,
        cache_value_in_state: bool,
        inadmissible_numeric_heuristic_variant: bool,
        disable_numeric_reasoning: bool,
    ) -> PyResult<Self> {
        Ok(Heuristic {
            variant: HeuristicVariant::DeleteRelaxation(DeleteRelaxationHeuristic::new(
                actions,
                fluent_types,
                objects,
                events,
                goals,
                DeleteRelaxationHeuristicConfig {
                    heuristic_kind: HeuristicKind::HMAX,
                    internal_caching,
                    inadmissible_numeric_heuristic_variant,
                    disable_numeric_reasoning,
                },
            )?),
            cache_value_in_state,
        })
    }

    #[staticmethod]
    #[allow(clippy::too_many_arguments, unused_variables)]
    pub fn hmax_explicit(
        actions: Vec<Action>,
        fluent_types: Vec<String>,
        objects: FxHashMap<String, Vec<usize>>,
        events: FxHashMap<Action, Vec<(Timing, Event)>>,
        goals: Vec<PyExpressionNode>,
        internal_caching: bool,
        cache_value_in_state: bool,
        inadmissible_numeric_heuristic_variant: bool,
    ) -> PyResult<Self> {
        Ok(Heuristic {
            variant: HeuristicVariant::HMaxExplicit(HMaxExplicit::new(
                actions,
                fluent_types,
                events,
                goals,
                internal_caching,
            )?),
            cache_value_in_state,
        })
    }

    #[getter]
    pub fn name(&self) -> &'static str {
        match &self.variant {
            HeuristicVariant::DeleteRelaxation(h) => h.name(),
            HeuristicVariant::HMaxExplicit(h) => h.name(),
            HeuristicVariant::Custom(h) => h.name(),
        }
    }

    #[pyo3(name = "eval")]
    pub fn py_eval(&self, state: &State, ss: &SearchSpace) -> PyResult<Option<f64>> {
        self.eval(state, ss)
    }

    pub fn reachable_actions(&self, state: &State) -> PyResult<FxHashSet<Action>> {
        match &self.variant {
            HeuristicVariant::DeleteRelaxation(h) => h.reachable_actions(state),
            _ => Err(PyNotImplementedError::new_err(
                "reachable_actions is only available for delete-relaxation heuristics",
            )),
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
        let h_value = match &self.variant {
            HeuristicVariant::DeleteRelaxation(h) => h.eval(state),
            HeuristicVariant::HMaxExplicit(h) => h.eval(state),
            HeuristicVariant::Custom(h) => h.eval(state),
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
