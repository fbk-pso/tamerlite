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

use std::cell::RefCell;
use std::num::NonZeroUsize;

use lru::LruCache;
use num::BigInt;
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
    types::{PyBool, PyInt, PyTuple},
};
use rustc_hash::{FxBuildHasher, FxHashMap};

use crate::expressions::{ExpressionNode, PyExpressionNode};
use crate::utils::{big_rational_to_py_fraction, get_big_rational_bigint, get_fraction_type};

#[pyclass(eq, eq_int, frozen, hash, from_py_object)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum IfReturnType {
    #[pyo3(name = "BOOL")]
    Bool,
    #[pyo3(name = "INT")]
    Int,
    #[pyo3(name = "REAL")]
    Real,
    #[pyo3(name = "OBJECT")]
    Object,
}

// Process-global registry mapping a `func_id` to the Python callable it was
// built from. Populated by `make_interpreted_function_node` (called from
// `Converter.walk_interpreted_function_exp`) and read by
// `call_interpreted_function`. `IF_IDS_BY_PTR` dedups registrations by pointer
// identity so the single, memoizing wrapper callable
// `Converter._get_interpreted_function_wrapper` builds per
// `InterpretedFunction` always maps back to the same `func_id`.
//
// `thread_local!` + `RefCell`: TamerLite's Rust search is single-threaded,
// so there is no concurrent access to guard against. `thread_local!` is what
// makes that legal: a plain `static` requires its type to be `Sync`, which
// `RefCell` deliberately isn't, so this would need `unsafe` (`static mut`
// or a raw `UnsafeCell`) without it.
thread_local! {
    static INTERPRETED_FUNCTIONS: RefCell<Vec<Py<PyAny>>> = const { RefCell::new(Vec::new()) };
    static IF_IDS_BY_PTR: RefCell<FxHashMap<usize, usize>> =
        const { RefCell::new(FxHashMap::with_hasher(FxBuildHasher)) };
}

pub(crate) fn register_interpreted_function(function: Py<PyAny>) -> usize {
    let ptr = function.as_ptr() as usize;
    if let Some(id) = IF_IDS_BY_PTR.with_borrow(|by_ptr| by_ptr.get(&ptr).copied()) {
        return id;
    }

    let id = INTERPRETED_FUNCTIONS.with_borrow_mut(|registry| {
        registry.push(function);
        registry.len() - 1
    });
    IF_IDS_BY_PTR.with_borrow_mut(|by_ptr| by_ptr.insert(ptr, id));
    id
}

/// Resolves `func_id` to its registered callable, or a `PyRuntimeError` if
/// it isn't (or is no longer) registered.
fn get_interpreted_function(py: Python<'_>, func_id: usize) -> PyResult<Py<PyAny>> {
    let found = INTERPRETED_FUNCTIONS
        .with_borrow(|registry| registry.get(func_id).map(|f| f.clone_ref(py)));
    found.ok_or_else(|| {
        PyRuntimeError::new_err(format!(
            "interpreted function {func_id} is no longer registered: the \
             expression referencing it outlived the solve that created it"
        ))
    })
}

/// Converts an evaluated `ExpressionNode` argument into the Python value an
/// interpreted function's real callable expects.
fn interpreted_function_arg<'py>(
    node: &ExpressionNode,
    py: Python<'py>,
) -> PyResult<Bound<'py, PyAny>> {
    match node {
        ExpressionNode::Bool(v) => Ok(v.into_pyobject(py)?.to_owned().into_any()),
        ExpressionNode::Int(v) => Ok(v.as_ref().into_pyobject(py)?.into_any()),
        ExpressionNode::Rational(v) => big_rational_to_py_fraction(v, py),
        ExpressionNode::Object(oid) => Ok(Bound::new(
            py,
            PyExpressionNode {
                v: ExpressionNode::Object(*oid),
            },
        )?
        .into_any()),
        other => Err(PyValueError::new_err(format!(
            "Cannot pass {:?} as an interpreted-function argument",
            other
        ))),
    }
}

/// Converts an interpreted function's raw Python return value into an
/// `ExpressionNode`, coercing it to the declared `return_type` -- the raw
/// callable is free to return any Python-native type (e.g. a plain `float`
/// for a "real" function), so this normalizes it to the exact type the rest
/// of the search space expects.
fn interpreted_function_result(
    result: &Bound<'_, PyAny>,
    return_type: IfReturnType,
) -> PyResult<ExpressionNode> {
    let py = result.py();
    match return_type {
        IfReturnType::Bool => {
            let b: bool = py.get_type::<PyBool>().call1((result,))?.extract()?;
            Ok(ExpressionNode::Bool(b))
        }
        IfReturnType::Int => {
            let v: BigInt = py.get_type::<PyInt>().call1((result,))?.extract()?;
            Ok(ExpressionNode::Int(Box::new(v)))
        }
        IfReturnType::Real => {
            let fraction = get_fraction_type(py)?.bind(py).call1((result,))?;
            let v = get_big_rational_bigint(&fraction)?;
            Ok(ExpressionNode::Rational(Box::new(v)))
        }
        IfReturnType::Object => {
            let node: PyExpressionNode = result.extract()?;
            match node.v {
                ExpressionNode::Object(oid) => Ok(ExpressionNode::Object(oid)),
                other => Err(PyValueError::new_err(format!(
                    "An interpreted function with an object return type must \
                     return an ObjectNode, got {:?}",
                    other
                ))),
            }
        }
    }
}

fn call_interpreted_function_uncached(
    func_id: usize,
    return_type: IfReturnType,
    args: &[&ExpressionNode],
) -> PyResult<ExpressionNode> {
    Python::attach(|py| {
        let callable = get_interpreted_function(py, func_id)?;
        let mut py_args = Vec::with_capacity(args.len());
        for &a in args {
            py_args.push(interpreted_function_arg(a, py)?);
        }
        let result = callable.call1(py, PyTuple::new(py, py_args)?)?;
        interpreted_function_result(result.bind(py), return_type)
    })
}

/// Memoized result of one interpreted-function call: `func_id` (which
/// callable), `return_type` (see below), and the already-evaluated argument
/// nodes.
///
/// `return_type` is part of the key even though a given `func_id` is
/// normally called with a single, fixed `return_type`: `register_interpreted_function`
/// dedups purely by `Py` pointer, and `make_interpreted_function_node` is a
/// public `#[pyfunction]` that can be (and in tests is) called directly with
/// a raw, non-wrapper callable. Two registrations of the *same* raw callable
/// under two different `IfReturnType`s would otherwise share one `func_id`
/// and collide in this cache. `IfReturnType` is `Copy + Hash`, so including
/// it costs nothing on the wrapper path where it never varies.
///
/// `args` is `Box<[ExpressionNode]>` rather than `Vec` -- exactly-sized and
/// smaller (16 vs 24 bytes) for a key that lives as long as the cache entry.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct IfCallKey {
    func_id: usize,
    return_type: IfReturnType,
    args: Box<[ExpressionNode]>,
}

// Result cache for interpreted-function calls, keyed on `IfCallKey`. Sound to key
// on `func_id` alone (modulo `return_type`) because `register_interpreted_function`
// dedups by pointer identity and, between resets, never drops a registered
// callable, so a given `func_id` always resolves to the same wrapper closure --
// and therefore the same object numbering -- for as long as it stays registered.
// `_if_wrappers` (Python side) may be shared across every `Converter` built within
// one `TamerLite._solve`/`_get_solutions_with_params` call, which is what lets
// entries here persist across anytime re-encodes instead of starting cold. Not
// shared across unrelated problems/solves -- but also not reset *between* two
// interleaved ones any more: `clear_interpreted_function_cache` below now only
// runs once no solve is in flight (`tamerlite.converter.interpreted_function_scope`
// tracks that with a live-scope counter on the Python side), so a solve that is
// merely suspended (e.g. a not-yet-exhausted anytime generator) keeps its entries
// warm across whatever else runs while it's suspended, rather than losing them to
// an unrelated solve's start-of-run reset the way it used to.
//
// Errors are never cached (see the `?` before the insert below): a replayed
// `PyErr` carries a stale traceback and could poison a callable that raises once
// and later succeeds; the one path that expects errors (`HMaxExplicit::
// possible_values` probing a partial callable) aborts the whole solve on the first
// one anyway, so there's nothing to amortize.
//
// Capped at `IF_RESULTS_CAPACITY` via a real LRU: entries now persist across a
// whole solve rather than one encoding, so an unbounded map could grow across an
// arbitrarily long anytime run.
const IF_RESULTS_CAPACITY: usize = 65_536;

thread_local! {
    static IF_RESULTS: RefCell<LruCache<IfCallKey, ExpressionNode>> =
        RefCell::new(LruCache::new(NonZeroUsize::new(IF_RESULTS_CAPACITY).unwrap()));
}

/// Calls the interpreted function registered under `func_id` with the given
/// already-evaluated argument nodes, returning its result coerced to
/// `return_type`. Memoized in `IF_RESULTS`: a hit returns the previously
/// coerced result without touching Python at all.
pub fn call_interpreted_function(
    func_id: usize,
    return_type: IfReturnType,
    args: &[&ExpressionNode],
) -> PyResult<ExpressionNode> {
    let key = IfCallKey {
        func_id,
        return_type,
        args: args.iter().map(|&a| a.clone()).collect(),
    };

    if let Some(hit) = IF_RESULTS.with_borrow_mut(|cache| cache.get(&key).cloned()) {
        return Ok(hit);
    }

    let value = call_interpreted_function_uncached(func_id, return_type, args)?;
    IF_RESULTS.with_borrow_mut(|cache| {
        cache.put(key, value.clone());
    });
    Ok(value)
}

/// Drops every memoized interpreted-function result and every registered
/// callable, resetting `func_id` allocation back to zero.
///
/// Callers are expected to only call this when no solve is in flight (see
/// `tamerlite.converter.interpreted_function_scope`, which tracks that on
/// the Python side via a live-scope counter and calls this once the count
/// returns to zero). A `func_id` still referenced after this runs -- e.g. by
/// a still-suspended anytime generator despite that expectation -- resolves
/// against whatever the next epoch registers under the same, recycled id;
/// `get_interpreted_function` only catches the out-of-range case, so this is
/// still a caller-behavior assumption the crate cannot itself verify.
///
/// `INTERPRETED_FUNCTIONS.clear()` below drops every registered `Py<PyAny>`
/// while still holding that `RefCell`'s mutable borrow. Assumed sound: no
/// registered interpreted-function callable, nor anything its closure
/// captures, defines a `__del__` or other finalizer that re-enters this module
/// -- if one did, that reentrant call would try to borrow `INTERPRETED_FUNCTIONS`
/// (or `IF_IDS_BY_PTR`) again while this borrow is still outstanding and panic
/// with "already borrowed". Not enforced by this crate.
#[pyfunction]
pub fn clear_interpreted_function_cache() {
    IF_RESULTS.with_borrow_mut(|cache| cache.clear());
    INTERPRETED_FUNCTIONS.with_borrow_mut(|registry| registry.clear());
    IF_IDS_BY_PTR.with_borrow_mut(|by_ptr| by_ptr.clear());
}

/// True if `expr` contains an interpreted-function call.
pub fn has_interpreted_function(expr: &[ExpressionNode]) -> bool {
    expr.iter()
        .any(|e| matches!(e, ExpressionNode::InterpretedFunction { .. }))
}
