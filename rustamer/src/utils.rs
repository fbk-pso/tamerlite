use num::{rational::BigRational, BigInt, FromPrimitive};
use pyo3::{exceptions::PyValueError, PyResult, PyAny};
use pyo3::prelude::*;
use std::sync::Arc;


pub fn get_big_rational(obj: &pyo3::Bound<'_, PyAny>) -> PyResult<BigRational> {
    if let Ok(int_n) = obj.extract::<i64>() {
        return Ok(BigRational::from_integer(BigInt::from(int_n)));
    }

    if let Ok(float_n) = obj.extract::<f64>() {
        if let Some(n) = BigRational::from_float(float_n) {
            return Ok(n);
        }
    }

    Err(PyValueError::new_err("Unable to parse Rational number"))
}

pub fn get_option_big_rational(obj: &pyo3::Bound<'_, PyAny>) -> PyResult<Option<BigRational>> {
    if let Ok(int_n) = obj.extract::<i64>() {
        return Ok(Some(BigRational::from_integer(BigInt::from(int_n))));
    }

    if let Ok(float_n) = obj.extract::<f64>() {
        if let Some(n) = BigRational::from_float(float_n) {
            return Ok(Some(n));
        }
    }

    Ok(None)
}

pub fn mk_rational(n: i32, d: i32) -> BigRational {
    BigRational::from_i32(n).unwrap() / BigRational::from_i32(d).unwrap()
}

pub fn mk_integer(n: i32) -> BigInt {
    BigInt::from_i32(n).unwrap()
}

pub fn rational_to_f32(n: &BigRational) -> f32 {
    n.to_string().parse::<f32>().unwrap()
}

pub fn integer_to_f32(n: &BigInt) -> f32 {
    n.to_string().parse::<f32>().unwrap()
}

pub fn integer_to_i32(n: &BigInt) -> i32 {
    n.to_string().parse::<i32>().unwrap()
}

pub fn usize_to_f32(n: usize) -> f32 {
    n.to_string().parse::<f32>().unwrap()
}


#[derive(Debug, Clone)]
pub struct PersistentList<Q> {
    pub payload: Q,
    previous: Option<Arc<PersistentList<Q>>>,
}

impl<Q> PersistentList<Q>
where Q: Clone
{
    pub fn new() -> Option<Arc<Self>> {
        None
    }

    pub fn append(payload: Q, previous: &Option<Arc<Self>>) -> Option<Arc<Self>> {
        Some(Arc::new(PersistentList { payload:payload, previous:previous.clone() }))
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
}
