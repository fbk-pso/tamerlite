// Copyright (C) 2026 PSO Unit, Fondazione Bruno Kessler
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

use rustc_hash::{FxBuildHasher, FxHashMap};
use std::hash::Hash;

/// A minimal multiset tracking occurrence counts.
#[derive(Debug, Clone)]
pub struct HashMultiSet<T: Eq + Hash> {
    counts: FxHashMap<T, usize>,
}

impl<T: Eq + Hash> HashMultiSet<T> {
    pub fn new() -> Self {
        Self {
            counts: FxHashMap::with_hasher(FxBuildHasher),
        }
    }

    /// Insert one occurrence of `value`.
    pub fn insert(&mut self, value: T) {
        *self.counts.entry(value).or_insert(0) += 1;
    }

    /// Remove one occurrence of `value`, dropping the key once its count
    /// reaches zero. Returns `true` if an occurrence was present.
    pub fn remove(&mut self, value: &T) -> bool {
        if let Some(c) = self.counts.get_mut(value) {
            *c -= 1;
            if *c == 0 {
                self.counts.remove(value);
            }
            true
        } else {
            false
        }
    }

    /// Iterate over the distinct elements currently in the multiset (each
    /// yielded once, regardless of its multiplicity).
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.counts.keys()
    }
}

impl<T: Eq + Hash> Default for HashMultiSet<T> {
    fn default() -> Self {
        Self::new()
    }
}
