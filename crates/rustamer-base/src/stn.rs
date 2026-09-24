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

use std::collections::VecDeque;
use std::sync::Arc;

use rustc_hash::{FxBuildHasher, FxHashMap};

#[derive(Debug)]
struct DeltaNeighbors<T, Q> {
    dst: T,
    bound: Q,
    next: Option<Arc<DeltaNeighbors<T, Q>>>,
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
pub struct DeltaSTN<T, Q> {
    constraints: FxHashMap<T, Option<Arc<DeltaNeighbors<T, Q>>>>,
    pub distances: FxHashMap<T, Q>,
    is_sat: bool,
    pub tolerance: Q,
    subsumption: bool,
}

impl<T, Q> DeltaSTN<T, Q>
where
    Q: num_traits::Num + std::ops::Neg<Output = Q> + PartialOrd + Clone,
    T: std::hash::Hash + Eq + Clone + Copy,
{
    pub fn new(tolerance: Q) -> Self {
        DeltaSTN {
            constraints: FxHashMap::with_hasher(FxBuildHasher),
            distances: FxHashMap::with_hasher(FxBuildHasher),
            is_sat: true,
            tolerance,
            subsumption: true,
        }
    }

    /// A network whose `add` never checks for subsumption and always prepends
    /// the new edge. Meant for a network where each (x, y) pair is added about
    /// once, like `SearchSpace::build_plan`'s, so the walk almost always
    /// misses. Skipping it changes neither the verdict nor the schedule, since an
    /// implied edge never lowers a distance; it only leaves a redundant edge
    /// in the out-list.
    pub fn new_without_subsumption(tolerance: Q) -> Self {
        DeltaSTN {
            subsumption: false,
            ..Self::new(tolerance)
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
            if !self.subsumption || !self.is_subsumed(x, y, b) {
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

    fn is_subsumed(&self, x: &T, y: &T, b: &Q) -> bool {
        let mut neighbors: &Option<Arc<DeltaNeighbors<T, Q>>> = self.constraints.get(x).unwrap();
        while neighbors.is_some() {
            let n: &Arc<DeltaNeighbors<T, Q>> = neighbors.as_ref().unwrap();
            if n.dst == *y {
                return n.bound <= b.clone() + self.tolerance.clone();
            }
            neighbors = &n.next
        }
        false
    }

    pub fn equals_with_tolerance(&self, b1: &Q, b2: &Q) -> bool {
        b1.clone() - b2.clone() <= self.tolerance
            && b1.clone() - b2.clone() >= -self.tolerance.clone()
    }

    fn inc_check(&mut self, x: &T, y: &T, b: &Q) -> bool {
        if self.distances[x].clone() + b.clone()
            < self.distances[y].clone() - self.tolerance.clone()
        {
            self.distances
                .insert(*y, self.distances[x].clone() + b.clone());
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
                if val < self.distances[&n.dst].clone() - self.tolerance.clone() {
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
