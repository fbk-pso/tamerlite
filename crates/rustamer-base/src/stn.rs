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

use log::warn;
use std::cmp::Ordering;
use std::collections::VecDeque;
use std::fs::read_to_string;
use std::ops::{Add, Neg, Sub};
use std::sync::Arc;

use regex::Regex;
use rustc_hash::{FxBuildHasher, FxHashMap};

/// A time `c + k·δ`, where `δ` is a positive infinitesimal.
///
/// A strict separation between two events is the bound `(0, 1)`: "later by
/// some positive amount", which is what UP means when `Problem.epsilon` is
/// unset. A bound with `k = 0` is an ordinary number, so a fixed separation
/// `ε` is `(ε, 0)`. Times compare lexicographically (`c` first, then `k`),
/// which is their order for every small enough `δ`: the derived `PartialOrd`
/// is that exact order, and `STNWeight::cmp_tol` the one that allows a
/// tolerance on `c`. Shortest paths and negative cycles over these pairs are
/// those of the network for every small enough `δ`, so the `DeltaSTN`
/// algorithm needs no change; a concrete `δ` is chosen only when a plan is
/// built (`SearchSpace::build_plan`).
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct DeltaTime<R> {
    pub c: R,
    pub k: i64,
}

impl<R> DeltaTime<R> {
    pub fn new(c: R, k: i64) -> Self {
        DeltaTime { c, k }
    }

    /// The ordinary number `c`, with no infinitesimal part.
    pub fn standard(c: R) -> Self {
        DeltaTime { c, k: 0 }
    }
}

impl<R: Add<Output = R>> Add for DeltaTime<R> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        DeltaTime {
            c: self.c + other.c,
            k: self.k + other.k,
        }
    }
}

impl<R: Sub<Output = R>> Sub for DeltaTime<R> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        DeltaTime {
            c: self.c - other.c,
            k: self.k - other.k,
        }
    }
}

impl<R: Neg<Output = R>> Neg for DeltaTime<R> {
    type Output = Self;

    fn neg(self) -> Self {
        DeltaTime {
            c: -self.c,
            k: -self.k,
        }
    }
}

/// The type of a `DeltaSTN`'s weights: its bounds, distances and times.
pub trait STNWeight: Clone + Add<Output = Self> + Sub<Output = Self> + Neg<Output = Self> {
    type Tolerance: Clone;

    fn zero() -> Self;

    /// Compares `self` with `other`, taking values within `tol` of each other as equal.
    fn cmp_tol(&self, other: &Self, tol: &Self::Tolerance) -> Ordering;
}

impl<R> STNWeight for DeltaTime<R>
where
    R: Clone + num_traits::Signed + PartialOrd,
{
    type Tolerance = R;

    fn zero() -> Self {
        DeltaTime::standard(R::zero())
    }

    /// The tolerance applies to the standard part `c` only: within it, the
    /// infinitesimal parts decide, exactly. So a cycle whose standard parts
    /// cancel out up to rounding but that holds a strict separation is still
    /// negative. With `k = 0` on both sides this is the plain comparison
    /// with tolerance.
    fn cmp_tol(&self, other: &Self, tol: &R) -> Ordering {
        let d = self.c.clone() - other.c.clone();
        if d.abs() <= *tol {
            self.k.cmp(&other.k)
        } else if d.is_negative() {
            Ordering::Less
        } else {
            Ordering::Greater
        }
    }
}

#[derive(Debug)]
struct DeltaNeighbors<T, Q> {
    dst: T,
    bound: Q,
    next: Option<Arc<DeltaNeighbors<T, Q>>>,
}

/// Frees the chain iteratively: a node collecting one edge per action on a long
/// temporal plan would otherwise overflow the stack.
impl<T, Q> Drop for DeltaNeighbors<T, Q> {
    fn drop(&mut self) {
        let mut next = self.next.take();
        while let Some(node) = next {
            next = Arc::into_inner(node).and_then(|mut inner| inner.next.take());
        }
    }
}

impl<T, Q> DeltaNeighbors<T, Q>
where
    Q: Clone,
    T: Copy,
{
    fn mk_empty() -> Option<Arc<Self>> {
        None
    }

    fn add(dst: &T, bound: &Q, next: &Option<Arc<Self>>) -> Option<Arc<Self>> {
        Some(Arc::new(DeltaNeighbors {
            dst: *dst,
            bound: bound.clone(),
            next: next.clone(),
        }))
    }
}

#[derive(Debug, Clone)]
pub struct DeltaSTN<T, Q: STNWeight> {
    constraints: FxHashMap<T, Option<Arc<DeltaNeighbors<T, Q>>>>,
    pub distances: FxHashMap<T, Q>,
    is_sat: bool,
    pub tolerance: Q::Tolerance,
}

impl<T, Q> DeltaSTN<T, Q>
where
    Q: STNWeight,
    T: std::hash::Hash + Eq + Clone + Copy,
{
    pub fn new(tolerance: Q::Tolerance) -> Self {
        DeltaSTN {
            constraints: FxHashMap::with_hasher(FxBuildHasher),
            distances: FxHashMap::with_hasher(FxBuildHasher),
            is_sat: true,
            tolerance,
        }
    }

    pub fn add(&mut self, x: &T, y: &T, b: &Q) {
        if self.is_sat {
            if !self.distances.contains_key(x) {
                self.distances.insert(*x, Q::zero());
                self.constraints.insert(*x, DeltaNeighbors::mk_empty());
            }
            if !self.distances.contains_key(y) {
                self.distances.insert(*y, Q::zero());
                self.constraints.insert(*y, DeltaNeighbors::mk_empty());
            }
            if !self.is_subsumed(x, y, b) {
                let old_x = self.constraints.get(x).unwrap();
                self.constraints
                    .insert(*x, DeltaNeighbors::add(y, b, old_x));
            }
            self.is_sat = self.inc_check(x, y, b);
        }
    }

    pub fn check(&self) -> bool {
        self.is_sat
    }

    pub fn get_model_value(&self, x: &T) -> Option<Q> {
        self.distances.get(x).map(|v| -v.clone())
    }

    fn is_subsumed(&self, x: &T, y: &T, b: &Q) -> bool {
        let mut neighbors: &Option<Arc<DeltaNeighbors<T, Q>>> = self.constraints.get(x).unwrap();
        while neighbors.is_some() {
            let n: &Arc<DeltaNeighbors<T, Q>> = neighbors.as_ref().unwrap();
            if n.dst == *y {
                return n.bound.cmp_tol(b, &self.tolerance) != Ordering::Greater;
            }
            neighbors = &n.next
        }
        false
    }

    pub fn equals_with_tolerance(&self, b1: &Q, b2: &Q) -> bool {
        b1.cmp_tol(b2, &self.tolerance) == Ordering::Equal
    }

    fn inc_check(&mut self, x: &T, y: &T, b: &Q) -> bool {
        let x_plus_b = self.distances[x].clone() + b.clone();
        if x_plus_b.cmp_tol(&self.distances[y], &self.tolerance) == Ordering::Less {
            self.distances.insert(*y, x_plus_b);
        } else {
            return true;
        }

        let mut q: VecDeque<&T> = VecDeque::from([y]);
        while !q.is_empty() {
            let c: &T = q.pop_front().unwrap();
            let mut neighbors: &Option<Arc<DeltaNeighbors<T, Q>>> =
                self.constraints.get(c).unwrap();
            while neighbors.is_some() {
                let n: &Arc<DeltaNeighbors<T, Q>> = neighbors.as_ref().unwrap();
                let val = self.distances[c].clone() + n.bound.clone();
                if val.cmp_tol(&self.distances[&n.dst], &self.tolerance) == Ordering::Less {
                    if n.dst == *y && self.equals_with_tolerance(&n.bound, b) {
                        return false; // Cycle detected
                    } else {
                        self.distances.insert(n.dst, val);
                        q.push_back(&n.dst);
                    }
                }
                neighbors = &n.next
            }
        }
        true
    }
}

pub fn _tnsolve(fname: String) {
    let re_new_tn = Regex::new(r#"^NewTN\("([a-z0-9]+)"\);$"#).unwrap();
    let re_check = Regex::new(r#"^Check\("([a-z0-9]+)"\);$"#).unwrap();
    let re_destroy_tn = Regex::new(r#"^DestroyTN\("([a-z0-9]+)"\);$"#).unwrap();
    let re_copy_tn = Regex::new(r#"^CopyTN\("([a-z0-9]+)",\s*"([a-z0-9]+)"\);$"#).unwrap();
    let re_add = Regex::new(
        r#"^Add\("([a-z0-9]+)",\s*([0-9]+),\s*([0-9]+),\s*((-?)(0|([1-9][0-9]*))(\.[0-9]+)?)\);$"#,
    )
    .unwrap();

    let mut tn_map = FxHashMap::<String, DeltaSTN<u32, DeltaTime<f64>>>::with_hasher(FxBuildHasher);

    for line in read_to_string(fname).unwrap().lines() {
        if let Some(new_tn) = re_new_tn.captures(line) {
            tn_map.insert(new_tn[1].to_owned(), DeltaSTN::new(0.00000001));
            continue;
        }

        if let Some(check) = re_check.captures(line) {
            print!(
                "{} ",
                if tn_map[&check[1].to_owned()].check() {
                    "1"
                } else {
                    "0"
                }
            );
            continue;
        }

        if let Some(destroy_tn) = re_destroy_tn.captures(line) {
            tn_map.remove(&destroy_tn[1]);
            continue;
        }

        if let Some(copy_tn) = re_copy_tn.captures(line) {
            let map = &mut tn_map;
            let new = map[&copy_tn[1].to_owned()].clone();
            map.insert(copy_tn[2].to_owned(), new);
            continue;
        }

        if let Some(add) = re_add.captures(line) {
            let x = add[2].parse::<u32>().unwrap();
            let y = add[3].parse::<u32>().unwrap();
            let b = add[4].parse::<f64>().unwrap();
            tn_map
                .get_mut(&add[1])
                .unwrap()
                .add(&x, &y, &DeltaTime::standard(b));
            continue;
        }

        warn!("Unmatched line: {}", line)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stn() -> DeltaSTN<u32, DeltaTime<f64>> {
        DeltaSTN::new(1e-9)
    }

    #[test]
    fn cmp_tol_decides_by_k_within_tolerance() {
        let a = DeltaTime::new(1.0, 0);
        let b = DeltaTime::new(1.0 + 1e-12, 1);
        assert_eq!(a.cmp_tol(&b, &1e-9), Ordering::Less);
        assert_eq!(b.cmp_tol(&a, &1e-9), Ordering::Greater);
        assert_eq!(a.cmp_tol(&a.clone(), &1e-9), Ordering::Equal);
        // Outside the tolerance the standard part decides, whatever `k`.
        let c = DeltaTime::new(0.5, 100);
        assert_eq!(c.cmp_tol(&a, &1e-9), Ordering::Less);
    }

    #[test]
    fn strict_cycle_is_inconsistent() {
        // t1 >= t0 + δ and t0 >= t1: no schedule.
        let mut tn = stn();
        tn.add(&0, &1, &DeltaTime::new(0.0, -1));
        assert!(tn.check());
        tn.add(&1, &0, &DeltaTime::new(0.0, 0));
        assert!(!tn.check());
    }

    #[test]
    fn non_strict_zero_cycle_is_consistent() {
        // t1 >= t0 and t0 >= t1: t0 = t1.
        let mut tn = stn();
        tn.add(&0, &1, &DeltaTime::new(0.0, 0));
        tn.add(&1, &0, &DeltaTime::new(0.0, 0));
        assert!(tn.check());
    }

    #[test]
    fn strict_separations_fit_in_any_positive_window() {
        // 1000 events, each strictly after the previous, all within [0, 1e-6]:
        // inconsistent for any fixed epsilon above 1e-9, fine for δ.
        let mut tn = stn();
        for i in 0..1000 {
            tn.add(&i, &(i + 1), &DeltaTime::new(0.0, -1));
        }
        tn.add(&1000, &0, &DeltaTime::new(1e-6, 0));
        assert!(tn.check());
        assert_eq!(tn.get_model_value(&1000), Some(DeltaTime::new(0.0, 1000)));
    }
}
