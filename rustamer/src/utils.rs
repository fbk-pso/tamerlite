use num::{rational::BigRational, BigInt, FromPrimitive};
use pyo3::{exceptions::PyValueError, PyResult, PyAny};

pub fn get_big_rational(obj: &PyAny) -> PyResult<BigRational> {
    match obj.to_string().parse::<BigRational>() {
        Ok(n) => Ok(n),
        Err(_) => Err(PyValueError::new_err("Unable to parse Rational number")),
    }
}

pub fn get_option_big_rational(obj: &PyAny) -> PyResult<Option<BigRational>> {
    match obj.to_string().parse::<BigRational>() {
        Ok(n) => Ok(Some(n)),
        Err(_) => Ok(None),
    }
}

pub fn mk_rational(n: i32, d: i32) -> BigRational {
    BigRational::from_i32(n).unwrap() / BigRational::from_i32(d).unwrap()
}

pub fn mk_integer(n: i32) -> BigInt {
    BigInt::from_i32(n).unwrap()
}

pub fn rational_to_f32(n: BigRational) -> f32 {
    n.to_string().parse::<f32>().unwrap()
}

pub fn integer_to_f32(n: BigInt) -> f32 {
    n.to_string().parse::<f32>().unwrap()
}

pub fn integer_to_i32(n: BigInt) -> i32 {
    n.to_string().parse::<i32>().unwrap()
}

pub fn usize_to_f32(n: usize) -> f32 {
    n.to_string().parse::<f32>().unwrap()
}