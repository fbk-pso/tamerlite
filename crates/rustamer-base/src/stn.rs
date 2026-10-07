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

/// A timepoint of a `DeltaSTN`: its position in creation order. Timepoints
/// are only ever appended (`add_timepoints`), so a network and its copies
/// agree on the number of every timepoint they share
pub type Timepoint = u32;

#[derive(Debug)]
struct DeltaNeighbors<Q> {
    dst: Timepoint,
    bound: Q,
    next: Option<Arc<DeltaNeighbors<Q>>>,
}

impl<Q> DeltaNeighbors<Q>
where
    Q: Clone,
{
    fn add(dst: Timepoint, bound: &Q, next: &Option<Arc<Self>>) -> Option<Arc<Self>> {
        Some(Arc::new(DeltaNeighbors {
            dst,
            bound: bound.clone(),
            next: next.clone(),
        }))
    }
}

#[derive(Debug, Clone)]
pub struct DeltaSTN<Q> {
    /// The out-list of each timepoint, indexed by timepoint
    constraints: Vec<Option<Arc<DeltaNeighbors<Q>>>>,
    /// The distance of each timepoint (its earliest time, negated), indexed
    /// by timepoint
    distances: Vec<Q>,
    is_sat: bool,
    pub tolerance: Q,
    subsumption: bool,
    /// A timepoint whose earliest time must not exceed the given bound: the
    /// network is inconsistent as soon as propagation pushes it later
    deadline: Option<(Timepoint, Q)>,
}

impl<Q> DeltaSTN<Q>
where
    Q: num_traits::Num + std::ops::Neg<Output = Q> + PartialOrd + Clone,
{
    pub fn new(tolerance: Q) -> Self {
        DeltaSTN {
            constraints: Vec::new(),
            distances: Vec::new(),
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
    pub fn with_deadline(tolerance: Q, node: Timepoint, deadline: Q) -> Self {
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

    /// Appends `n` unconstrained timepoints and returns the first of them;
    /// the others follow it consecutively
    pub fn add_timepoints(&mut self, n: usize) -> Timepoint {
        let first = self.distances.len() as Timepoint;
        self.distances.resize(self.distances.len() + n, Q::zero());
        self.constraints.resize(self.constraints.len() + n, None);
        first
    }

    pub fn num_timepoints(&self) -> usize {
        self.distances.len()
    }

    /// The earliest time of `t` in the earliest schedule
    pub fn earliest_time(&self, t: Timepoint) -> Q {
        -self.distances[t as usize].clone()
    }

    /// A copy with room for `extra` more timepoints. `clone` allocates
    /// exactly the current length, so the copy's next `add_timepoints` would
    /// double its capacity, and a stored state would keep that slack.
    pub fn clone_reserving(&self, extra: usize) -> Self {
        let mut constraints = Vec::with_capacity(self.constraints.len() + extra);
        constraints.extend_from_slice(&self.constraints);
        let mut distances = Vec::with_capacity(self.distances.len() + extra);
        distances.extend_from_slice(&self.distances);
        DeltaSTN {
            constraints,
            distances,
            is_sat: self.is_sat,
            tolerance: self.tolerance.clone(),
            subsumption: self.subsumption,
            deadline: self.deadline.clone(),
        }
    }

    /// Adds `t(x) - t(y) <= b`. Both timepoints must already exist
    pub fn add(&mut self, x: Timepoint, y: Timepoint, b: &Q) {
        debug_assert!((x as usize) < self.distances.len() && (y as usize) < self.distances.len());
        if self.is_sat {
            if !self.subsumption || !self.is_subsumed(x, y, b) {
                let old_x = &self.constraints[x as usize];
                self.constraints[x as usize] = DeltaNeighbors::add(y, b, old_x);
            }
            self.is_sat = self.inc_check(x, y, b);
        }
    }

    pub fn check(&self) -> bool {
        self.is_sat
    }

    fn is_subsumed(&self, x: Timepoint, y: Timepoint, b: &Q) -> bool {
        let mut neighbors = &self.constraints[x as usize];
        while let Some(n) = neighbors {
            if n.dst == y {
                return n.bound <= b.clone() + self.tolerance.clone();
            }
            neighbors = &n.next
        }
        false
    }

    /// Whether giving `node` the distance `distance` (its earliest time,
    /// negated) makes it miss the deadline
    fn misses_deadline(&self, node: Timepoint, distance: &Q) -> bool {
        match &self.deadline {
            Some((n, d)) => *n == node && *distance < -d.clone() - self.tolerance.clone(),
            None => false,
        }
    }

    pub fn equals_with_tolerance(&self, b1: &Q, b2: &Q) -> bool {
        b1.clone() - b2.clone() <= self.tolerance
            && b1.clone() - b2.clone() >= -self.tolerance.clone()
    }

    fn inc_check(&mut self, x: Timepoint, y: Timepoint, b: &Q) -> bool {
        let d = self.distances[x as usize].clone() + b.clone();
        if d < self.distances[y as usize].clone() - self.tolerance.clone() {
            if self.misses_deadline(y, &d) {
                return false;
            }
            self.distances[y as usize] = d;
        } else {
            return true;
        }

        let mut q: VecDeque<Timepoint> = VecDeque::from([y]);
        while let Some(c) = q.pop_front() {
            let mut neighbors = &self.constraints[c as usize];
            while let Some(n) = neighbors {
                let val = self.distances[c as usize].clone() + n.bound.clone();
                if val < self.distances[n.dst as usize].clone() - self.tolerance.clone() {
                    if n.dst == y && self.equals_with_tolerance(&n.bound, b) {
                        return false; // Cycle detected
                    } else if self.misses_deadline(n.dst, &val) {
                        return false;
                    } else {
                        self.distances[n.dst as usize] = val;
                        q.push_back(n.dst);
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
        let mut tn: DeltaSTN<f64> = DeltaSTN::with_deadline(tol, 0, 10.0);
        assert_eq!(tn.add_timepoints(5), 0);
        tn.add(1, 0, &-0.01);
        tn.add(2, 0, &-0.01);

        // End 1 at 9.99 puts plan end exactly on the deadline
        tn.add(3, 1, &-9.99);
        assert!(tn.check());

        // A copy keeps the bound: end 2 at 9.99 + tol / 2 is within the
        // tolerance, at 10 it misses the deadline, through propagation
        let mut copy = tn.clone();
        copy.add(3, 2, &(-9.99 - tol / 2.0));
        assert!(copy.check());
        copy.add(3, 2, &-10.0);
        assert!(!copy.check());
        assert!(tn.check());

        // A deadline node lowered directly (it is `y` of the edge) is caught too
        tn.add(4, 0, &-10.5);
        assert!(!tn.check());

        // Without a deadline, nothing bounds plan end
        let mut free: DeltaSTN<f64> = DeltaSTN::new(tol);
        free.add_timepoints(5);
        free.add(4, 0, &-10.5);
        assert!(free.check());
    }

    #[test]
    fn timepoints_are_numbered_in_creation_order() {
        let mut tn: DeltaSTN<f64> = DeltaSTN::new(0.0);
        assert_eq!(tn.add_timepoints(1), 0);
        assert_eq!(tn.add_timepoints(2), 1);
        assert_eq!(tn.add_timepoints(2), 3);
        assert_eq!(tn.num_timepoints(), 5);
        // 2 is at least 3 after 1
        tn.add(1, 2, &-3.0);
        assert_eq!(tn.earliest_time(1), 0.0);
        assert_eq!(tn.earliest_time(2), 3.0);
    }

    #[test]
    fn clone_reserving_copies_into_a_larger_allocation() {
        let mut tn: DeltaSTN<f64> = DeltaSTN::new(0.0);
        tn.add_timepoints(3);
        tn.add(0, 1, &-1.0);
        tn.add(1, 2, &-2.0);

        let mut copy = tn.clone_reserving(2);
        assert!(copy.distances.capacity() >= 5);
        assert!(copy.constraints.capacity() >= 5);
        assert_eq!(copy.num_timepoints(), 3);
        assert_eq!(copy.earliest_time(2), 3.0);

        // The copy keeps the edges of `tn` and extends independently of it
        assert_eq!(copy.add_timepoints(2), 3);
        copy.add(2, 3, &-1.0);
        assert_eq!(copy.earliest_time(3), 4.0);
        copy.add(0, 2, &-5.0);
        assert_eq!(copy.earliest_time(3), 6.0);
        copy.add(3, 0, &5.0);
        assert!(!copy.check());
        assert_eq!(tn.num_timepoints(), 3);
        assert_eq!(tn.earliest_time(2), 3.0);
        assert!(tn.check());
    }
}
