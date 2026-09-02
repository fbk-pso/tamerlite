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

use num::{rational::BigRational, BigInt, ToPrimitive};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::cell::RefCell;
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

/// Same as `get_big_rational`, but via arbitrary-precision `BigInt`
/// numerator/denominator instead of `i32`.
pub fn get_big_rational_bigint(obj: &pyo3::Bound<'_, PyAny>) -> PyResult<BigRational> {
    if let Ok(v) = obj.extract::<BigInt>() {
        return Ok(BigRational::from_integer(v));
    }

    if is_fraction(obj).unwrap_or(false) {
        if let (Ok(numerator), Ok(denominator)) = (
            obj.getattr("numerator").and_then(|n| n.extract::<BigInt>()),
            obj.getattr("denominator")
                .and_then(|d| d.extract::<BigInt>()),
        ) {
            return Ok(BigRational::new(numerator, denominator));
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

// Cached `fractions.Fraction` type object, resolved via `import fractions`
// once per thread instead of on every call that needs it (interpreted
// function real-typed args/returns, plan reconstruction, `Event.delay`).
thread_local! {
    static FRACTION_TYPE: RefCell<Option<Py<PyAny>>> = const { RefCell::new(None) };
}

pub fn get_fraction_type(py: Python<'_>) -> PyResult<Py<PyAny>> {
    if let Some(cached) = FRACTION_TYPE.with_borrow(|cell| cell.as_ref().map(|f| f.clone_ref(py))) {
        return Ok(cached);
    }
    let fraction_type = PyModule::import(py, "fractions")?
        .getattr("Fraction")?
        .unbind();
    FRACTION_TYPE.with_borrow_mut(|cell| *cell = Some(fraction_type.clone_ref(py)));
    Ok(fraction_type)
}

pub fn big_rational_to_py_fraction<'py>(
    n: &BigRational,
    py: Python<'py>,
) -> PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
    get_fraction_type(py)?
        .bind(py)
        .call1((n.numer(), n.denom()))
}

pub fn mk_rational(n: i32, d: i32) -> BigRational {
    BigRational::new(BigInt::from(n), BigInt::from(d))
}

pub fn mk_integer(n: i32) -> BigInt {
    BigInt::from(n)
}

pub fn rational_to_f64(n: &BigRational) -> f64 {
    n.to_f64().unwrap()
}

pub fn integer_to_f64(n: &BigInt) -> f64 {
    n.to_f64().unwrap()
}

pub fn integer_to_rational(n: BigInt) -> BigRational {
    BigRational::from_integer(n)
}

#[derive(Debug)]
pub enum ArithmeticError {
    Overflow,
}

pub fn checked_add_sub(
    lhs: usize,
    rhs: usize,
    is_subtraction: bool,
) -> Result<usize, ArithmeticError> {
    if is_subtraction {
        lhs.checked_sub(rhs).ok_or(ArithmeticError::Overflow)
    } else {
        lhs.checked_add(rhs).ok_or(ArithmeticError::Overflow)
    }
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
            payload,
            previous: previous.clone(),
        }))
    }

    pub fn to_vec(list: &Option<Arc<Self>>) -> Vec<&Q> {
        let mut result: Vec<&Q> = PersistentList::iter_rev(list).collect();
        result.reverse();
        result
    }

    pub fn to_vec_copy(list: &Option<Arc<Self>>) -> Vec<Q> {
        let mut result: Vec<Q> = PersistentList::iter_rev(list).cloned().collect();
        result.reverse();
        result
    }

    /// Iterate from newest to oldest (reverse order)
    pub fn iter_rev(list: &Option<Arc<Self>>) -> impl Iterator<Item = &Q> {
        struct Iter<'a, Q> {
            current: Option<&'a Arc<PersistentList<Q>>>,
        }

        impl<'a, Q> Iterator for Iter<'a, Q> {
            type Item = &'a Q;

            fn next(&mut self) -> Option<Self::Item> {
                if let Some(node) = self.current {
                    self.current = node.previous.as_ref();
                    Some(&node.payload)
                } else {
                    None
                }
            }
        }

        Iter {
            current: list.as_ref(),
        }
    }
}
