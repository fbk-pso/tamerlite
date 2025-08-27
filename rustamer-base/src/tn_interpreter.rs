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

use super::stn::DeltaSTN;
use super::structures::*;
use std::{collections::HashMap, vec::Vec};

#[derive(Debug)]
pub struct TNInterpreter {
    actions_ids: HashMap<(String, bool), u32>,
    events_ids: HashMap<(String, usize), u32>,
    actions_ids_map_back: HashMap<u32, (String, bool)>,
    events_ids_map_back: HashMap<u32, (String, usize)>,
}

impl TNInterpreter {
    pub fn new(actions: &Vec<String>, events: &HashMap<String, Vec<(Timing, Event)>>) -> Self {
        let mut actions_ids = HashMap::new();
        let mut actions_ids_map_back = HashMap::new();

        let mut next_id = 1;
        for a in actions {
            for b in [true, false] {
                actions_ids.insert((a.clone(), b), next_id);
                actions_ids_map_back.insert(next_id, (a.clone(), b));
                next_id += 1;
            }
        }

        let mut events_ids = HashMap::new();
        let mut events_ids_map_back = HashMap::new();

        for (action, events) in events {
            for (_t, e) in events {
                events_ids.insert((action.clone(), e.pos), next_id);
                events_ids_map_back.insert(next_id, (action.clone(), e.pos));
                next_id += 1;
            }
        }

        TNInterpreter {
            actions_ids: actions_ids,
            events_ids: events_ids,
            actions_ids_map_back: actions_ids_map_back,
            events_ids_map_back: events_ids_map_back,
        }
    }

    fn pack_u32(&self, a: u32, b: u32) -> u64 {
        ((a as u64) << 32) | (b as u64)
    }

    fn unpack_u64(&self, x: u64) -> (u32, u32) {
        ((x >> 32) as u32, (x & 0xFFFFFFFF) as u32)
    }

    pub fn get_action_id(&self, action: &str, is_start: bool, id: u32) -> u64 {
        if let Some(aid) = self.actions_ids.get(&(action.to_string(), is_start)) {
            // Concatenate the action id and the instance id using the
            // lower and higher parts of the u64 binary representation
            return self.pack_u32(*aid, id);
        }
        panic!("Action not found in the TNInterpreter");
        //return 0;
    }

    pub fn get_event_id(&self, action: &str, pos: usize, id: u32) -> u64 {
        if let Some(eid) = self.events_ids.get(&(action.to_string(), pos)) {
            return self.pack_u32(*eid, id);
        }
        panic!("Event not found in the TNInterpreter");
        //return 0;
    }

    pub fn get_action_timing<Q>(
        &self,
        tn: &DeltaSTN<u64, Q>,
        action: &str,
        is_start: bool,
        id: u32,
    ) -> Option<Q>
    where
        Q: num_traits::Num + std::ops::Neg<Output = Q> + PartialOrd + Clone,
    {
        let id = self.get_action_id(action, is_start, id);
        tn.get_model_value(&id)
    }

    pub fn get_event_timing<Q>(
        &self,
        tn: &DeltaSTN<u64, Q>,
        action: &str,
        pos: usize,
        id: u32,
    ) -> Option<Q>
    where
        Q: num_traits::Num + std::ops::Neg<Output = Q> + PartialOrd + Clone,
    {
        let id = self.get_event_id(action, pos, id);
        tn.get_model_value(&id)
    }

    pub fn get_actions_timings<Q>(&self, tn: &DeltaSTN<u64, Q>) -> Vec<((String, bool, u32), Q)>
    where
        Q: num_traits::Num + std::ops::Neg<Output = Q> + PartialOrd + Clone,
    {
        let mut res: Vec<((String, bool, u32), Q)> = Vec::new();
        for (id, v) in tn.distances.iter() {
            let (action_id, outer_id) = self.unpack_u64(*id);
            let a = self.actions_ids_map_back.get(&action_id);
            if let Some((action, is_start)) = a {
                res.push((
                    (action.clone(), *is_start, outer_id),
                    v.clone() * (-Q::one()),
                ));
            }
        }
        res.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        res
    }

    pub fn get_events_timings<Q>(&self, tn: &DeltaSTN<u64, Q>) -> Vec<((String, usize, u32), Q)>
    where
        Q: num_traits::Num + std::ops::Neg<Output = Q> + PartialOrd + Clone,
    {
        let mut res = Vec::new();
        for (id, v) in tn.distances.iter() {
            let (event_id, outer_id) = self.unpack_u64(*id);
            let a = self.events_ids_map_back.get(&event_id);
            if let Some((action, pos)) = a {
                res.push((
                    (action.clone(), pos.clone(), outer_id),
                    v.clone() * (-Q::one()),
                ));
            }
        }
        res.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        res
    }
}
