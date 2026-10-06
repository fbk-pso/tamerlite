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
    /// A timepoint whose earliest time must not exceed the given bound: the
    /// network is inconsistent as soon as propagation pushes it later
    deadline: Option<(T, Q)>,
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
            deadline: None,
        }
    }

    /// A network in which the earliest time of `node` must not exceed
    /// `deadline` (within the tolerance). Distances only ever decrease and
    /// all start at 0, so the earliest schedule has every timepoint at >= 0
    /// and this is checked directly where `node`'s distance is lowered,
    /// instead of through edges to a plan-start timepoint, which would close
    /// a cycle through every timepoint and catch a late `node` only after
    /// propagating all the way round it.
    pub fn with_deadline(tolerance: Q, node: T, deadline: Q) -> Self {
        DeltaSTN {
            deadline: Some((node, deadline)),
            ..Self::new(tolerance)
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

    /// Whether giving `node` the distance `distance` (its earliest time,
    /// negated) makes it miss the deadline
    fn misses_deadline(&self, node: &T, distance: &Q) -> bool {
        match &self.deadline {
            Some((n, d)) => n == node && *distance < -d.clone() - self.tolerance.clone(),
            None => false,
        }
    }

    pub fn equals_with_tolerance(&self, b1: &Q, b2: &Q) -> bool {
        b1.clone() - b2.clone() <= self.tolerance
            && b1.clone() - b2.clone() >= -self.tolerance.clone()
    }

    fn inc_check(&mut self, x: &T, y: &T, b: &Q) -> bool {
        if self.distances[x].clone() + b.clone()
            < self.distances[y].clone() - self.tolerance.clone()
        {
            let d = self.distances[x].clone() + b.clone();
            if self.misses_deadline(y, &d) {
                return false;
            }
            self.distances.insert(*y, d);
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
                    } else if self.misses_deadline(&n.dst, &val) {
                        return false;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deadline_bounds_the_earliest_time_of_its_node() {
        let tol = 0.01 / 1000.0;
        // Plan end (0) must be at most 10; action ends (1, 2) are at least
        // 0.01 before it
        let mut tn: DeltaSTN<u64, f64> = DeltaSTN::with_deadline(tol, 0, 10.0);
        tn.add(&1, &0, &-0.01);
        tn.add(&2, &0, &-0.01);

        // End 1 at 9.99 puts plan end exactly on the deadline
        tn.add(&3, &1, &-9.99);
        assert!(tn.check());

        // A copy keeps the bound: end 2 at 9.99 + tol / 2 is within the
        // tolerance, at 10 it misses the deadline, through propagation
        let mut copy = tn.clone();
        copy.add(&3, &2, &(-9.99 - tol / 2.0));
        assert!(copy.check());
        copy.add(&3, &2, &-10.0);
        assert!(!copy.check());
        assert!(tn.check());

        // A deadline node lowered directly (it is `y` of the edge) is caught too
        tn.add(&4, &0, &-10.5);
        assert!(!tn.check());

        // Without a deadline, nothing bounds plan end
        let mut free: DeltaSTN<u64, f64> = DeltaSTN::new(tol);
        free.add(&4, &0, &-10.5);
        assert!(free.check());
    }
}
