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

use num::{rational::BigRational, BigInt, ToPrimitive};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::sync::Arc;

pub fn is_fraction(obj: &pyo3::Bound<'_, PyAny>) -> PyResult<bool> {
    let py = obj.py();
    let fractions = PyModule::import(py, "fractions")?;
    let fraction_type = fractions.getattr("Fraction")?;
    obj.is_instance(&fraction_type)
}

pub fn get_big_rational(obj: &pyo3::Bound<'_, PyAny>) -> PyResult<BigRational> {
    if let Ok(int_n) = obj.extract::<i32>() {
        return Ok(BigRational::from_integer(BigInt::from(int_n)));
    }

    if is_fraction(obj).unwrap_or(false) {
        if let (Ok(numerator), Ok(denominator)) = (
            obj.getattr("numerator").and_then(|n| n.extract::<i32>()),
            obj.getattr("denominator").and_then(|d| d.extract::<i32>()),
        ) {
            return Ok(mk_rational(numerator, denominator));
        }
    }

    Err(PyValueError::new_err("Unable to parse Rational number"))
}

pub fn get_option_big_rational(obj: &pyo3::Bound<'_, PyAny>) -> PyResult<Option<BigRational>> {
    if obj.is_none() {
        Ok(None)
    } else {
        get_big_rational(obj).map(Some)
    }
}

pub fn mk_rational(n: i32, d: i32) -> BigRational {
    BigRational::new(BigInt::from(n), BigInt::from(d))
}

pub fn mk_integer(n: i32) -> BigInt {
    BigInt::from(n)
}

pub fn rational_to_f32(n: &BigRational) -> f32 {
    n.to_f32().unwrap()
}

pub fn integer_to_f32(n: &BigInt) -> f32 {
    n.to_f32().unwrap()
}

pub fn integer_to_i32(n: &BigInt) -> i32 {
    n.to_i32().unwrap()
}

pub fn usize_to_f32(n: usize) -> f32 {
    n.to_f32().unwrap()
}

#[derive(Debug, Clone)]
pub struct PersistentList<Q> {
    pub payload: Q,
    previous: Option<Arc<PersistentList<Q>>>,
}

impl<Q> PersistentList<Q>
where
    Q: Clone,
{
    pub fn new() -> Option<Arc<Self>> {
        None
    }

    pub fn append(payload: Q, previous: &Option<Arc<Self>>) -> Option<Arc<Self>> {
        Some(Arc::new(PersistentList {
            payload: payload,
            previous: previous.clone(),
        }))
    }

    pub fn to_vec(list: &Option<Arc<Self>>) -> Vec<&Q> {
        let mut result = vec![];
        let mut current_node = list;

        while let Some(node) = current_node {
            result.push(&node.payload);
            current_node = &node.previous;
        }

        result.reverse();
        result
    }

    pub fn to_vec_copy(list: &Option<Arc<Self>>) -> Vec<Q> {
        let mut result = vec![];
        let mut current_node = list;

        while let Some(node) = current_node {
            result.push(node.payload.clone());
            current_node = &node.previous;
        }

        result.reverse();
        result
    }
}
