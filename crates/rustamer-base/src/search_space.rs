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

use im::Vector;
use num_rational::BigRational;
use pyo3::{exceptions::PyException, prelude::*};
use rustc_hash::{FxBuildHasher, FxHashMap, FxHashSet};
use std::{sync::Mutex, vec::Vec};

use super::expressions::*;
use super::expressions_utils::*;
use super::multiset::HashMultiSet;
use super::search_state::*;
use super::stn::{DeltaSTN, Timepoint};
use super::structures::*;
use super::utils::*;

type ScheduledAction = (Option<BigRational>, Action, Option<BigRational>);
type PyScheduledAction<'py> = (Option<Bound<'py, PyAny>>, Action, Option<Bound<'py, PyAny>>);
type MutexCache = Mutex<FxHashMap<((Action, usize), (Action, usize)), bool>>;
type PrecedenceCache = Mutex<FxHashMap<((Action, usize), (Action, usize)), bool>>;
type EventFluents = Vec<
    Vec<(
        FxHashSet<Fluent>,
        FxHashSet<Fluent>,
        FxHashSet<Fluent>,
        FxHashSet<Fluent>,
        FxHashSet<Fluent>,
    )>,
>;
type DurationInterval = (Vec<ExpressionNode>, Vec<ExpressionNode>, bool, bool);
type PyDurationInterval = (Vec<PyExpressionNode>, Vec<PyExpressionNode>, bool, bool);

pub trait SearchSpaceTrait {
    fn is_temporal(&self) -> bool;
    fn initial_state(&self, initial_state: Option<Vec<PyExpressionNode>>) -> PyResult<State>;
    fn get_successor_state(&self, state: &State, action: Action) -> PyResult<Option<State>>;
    fn get_successor_states_iter<'a>(
        &'a self,
        state: &'a State,
    ) -> impl Iterator<Item = PyResult<State>> + 'a;
    fn get_successor_states(&self, state: &State) -> PyResult<Vec<State>> {
        let mut res = Vec::new();
        for rs in self.get_successor_states_iter(state) {
            res.push(rs?);
        }
        Ok(res)
    }
    fn reset(&self);
    fn goal_reached(&self, state: &State, goal: Option<Vec<PyExpressionNode>>) -> PyResult<bool>;
    fn subgoals_sat(
        &self,
        state: &State,
        goal: Option<Vec<PyExpressionNode>>,
    ) -> PyResult<Vec<Vec<PyExpressionNode>>>;
    fn build_plan(&self, path: &[Action]) -> PyResult<Vec<ScheduledAction>>;
}

#[derive(Debug)]
struct MutexChecker {
    cache: MutexCache,
}

impl MutexChecker {
    fn new() -> Self {
        MutexChecker {
            cache: Mutex::new(FxHashMap::with_hasher(FxBuildHasher)),
        }
    }

    fn check(
        &self,
        events_pair: &((Action, usize), (Action, usize)),
        event_fluents: &EventFluents,
    ) -> bool {
        let ((a1, i1), (a2, i2)) = events_pair;
        if a1 == a2 {
            return true;
        }

        let mut cache = self.cache.lock().unwrap();
        if let Some(are_mutex) = cache.get(events_pair) {
            return *are_mutex;
        }

        let (_, a_writes, a_read_writes, _, _) = &event_fluents[a1.idx][*i1];
        let (b_reads, b_writes, _, _, _) = &event_fluents[a2.idx][*i2];
        let are_mutex = !(b_reads.is_disjoint(a_writes) && a_read_writes.is_disjoint(b_writes));
        cache.insert(*events_pair, are_mutex);
        are_mutex
    }
}

#[derive(Debug)]
struct PrecedenceChecker {
    cache: PrecedenceCache,
}

impl PrecedenceChecker {
    fn new() -> Self {
        PrecedenceChecker {
            cache: Mutex::new(FxHashMap::with_hasher(FxBuildHasher)),
        }
    }

    fn check(
        &self,
        events_pair: &((Action, usize), (Action, usize)),
        event_fluents: &EventFluents,
    ) -> bool {
        let ((a1, i1), (a2, i2)) = events_pair;
        if a1 == a2 {
            return true;
        }

        let mut cache = self.cache.lock().unwrap();
        if let Some(res) = cache.get(events_pair) {
            return *res;
        }

        let (_, a_writes, _, _, a_end_cond_reads) = &event_fluents[a1.idx][*i1];
        let (_, b_writes, _, b_start_cond_reads, _) = &event_fluents[a2.idx][*i2];
        let res =
            !(a_writes.is_disjoint(b_start_cond_reads) && b_writes.is_disjoint(a_end_cond_reads));
        cache.insert(*events_pair, res);
        res
    }
}

#[pyfunction(name = "get_fluents")]
pub fn py_get_fluents(expr: Vec<PyExpressionNode>) -> Vec<Fluent> {
    expr.iter()
        .filter_map(|node| match node.v {
            ExpressionNode::Fluent(fluent) => Some(fluent),
            _ => None,
        })
        .collect()
}

fn get_fluents<'a>(expr: &'a [ExpressionNode]) -> impl Iterator<Item = Fluent> + 'a {
    expr.iter().filter_map(|node| match node {
        ExpressionNode::Fluent(fluent) => Some(*fluent),
        _ => None,
    })
}

/// An action being opened by `open_action`, whose constraints `expand_event`
/// adds: the action and the timepoint its start gets (its end gets the next)
type PendingOpening = (Action, Timepoint);

/// Plan end's timepoint in every search network: the first one created. Only
/// a deadline reads it
const PLAN_END: Timepoint = 0;

/// Adds `t(u) - t(v) <= b` for two events `u`, `v`, each given as the
/// timepoint of its anchor (its action instance's start or end) and its
/// constant delay from it. Events are rigidly `t(anchor) + delay`, so they get
/// no timepoint of their own: the constraint becomes one on the anchors,
/// `t(a_u) - t(a_v) <= b - delay_u + delay_v`, or a constant when both share
/// an anchor. Returns false if that constant is violated, i.e. the network is
/// inconsistent; `tn.add` records every other inconsistency in `tn.check()`.
fn add_event_constraint<Q>(
    tn: &mut DeltaSTN<Q>,
    u: (Timepoint, &Q),
    v: (Timepoint, &Q),
    b: Q,
) -> bool
where
    Q: num_traits::Num + std::ops::Neg<Output = Q> + PartialOrd + Clone,
{
    let b = b - u.1.clone() + v.1.clone();
    if u.0 == v.0 {
        return b >= -tn.tolerance.clone();
    }
    tn.add(u.0, v.0, &b);
    true
}

#[pyclass(name = "SearchSpace")]
#[derive(Debug)]
pub struct SearchSpace {
    actions_duration: Vec<Option<DurationInterval>>,
    events: FxHashMap<Action, Vec<(Timing, Event)>>,
    relevant_actions: Vec<Action>,
    compression_safe_actions: Option<Vec<bool>>,
    event_fluents: EventFluents,
    /// For event `k` of action `a`, at `[a.idx][k]`: whether it is anchored at
    /// the action's start (else its end) and its delay from that anchor
    event_anchors: Vec<Vec<(bool, f64)>>,
    effect_independent_start_conditions: Vec<Vec<Box<[usize]>>>,
    mutex: MutexChecker,
    precedence: PrecedenceChecker,
    action_objects: Option<Vec<Vec<Object>>>,
    obj_to_prev_actions_map: Option<Vec<FxHashSet<Action>>>,
    initial_state: Option<Vec<ExpressionNode>>,
    goal: Option<Vec<ExpressionNode>>,
    deadline: Option<f64>,
    epsilon: f64,
    epsilon_rational: BigRational,
    is_temporal: bool,
}

#[pymethods]
impl SearchSpace {
    #[new]
    #[pyo3(signature = (actions_duration, events, actions, compression_safe_actions, action_objects, obj_to_prev_actions_map, initial_state=None, goal=None, relevant_actions=None, deadline=None, epsilon=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        actions_duration: Vec<Option<PyDurationInterval>>,
        events: FxHashMap<Action, Vec<(Timing, Event)>>,
        actions: Vec<Action>,
        compression_safe_actions: Option<Vec<bool>>,
        action_objects: Option<Vec<Vec<Object>>>,
        obj_to_prev_actions_map: Option<Vec<FxHashSet<Action>>>,
        initial_state: Option<Vec<PyExpressionNode>>,
        goal: Option<Vec<PyExpressionNode>>,
        relevant_actions: Option<Vec<Action>>,
        #[pyo3(from_py_with = get_option_big_rational)] deadline: Option<BigRational>,
        #[pyo3(from_py_with = get_option_big_rational)] epsilon: Option<BigRational>,
    ) -> PyResult<Self> {
        let relevant_actions = if let Some(relevant_actions) = relevant_actions {
            relevant_actions
        } else {
            actions.clone()
        };
        // Every action the search can expand must have an entry in `events`.
        // A missing one is not a benign no-op here: `get_successor_state` and
        // `build_plan` both look events up with `events.get(...)`, so the
        // action would silently report as inapplicable -- and silently vanish
        // from a reconstructed plan -- instead of failing. `Encoder` restricts
        // `events` and `relevant_actions` to the same `considered_actions` set
        // when it compacts the encoding (and `relevant_actions` is only ever
        // narrowed further afterwards, through the setter), so a mismatch is an
        // encoder bug; catch it here rather than as a wrong plan much later.
        if let Some(a) = relevant_actions.iter().find(|a| !events.contains_key(a)) {
            return Err(PyException::new_err(format!(
                "Action {} is expandable but has no events entry",
                a.idx
            )));
        }
        let is_temporal = actions_duration.iter().any(|value| !value.is_none());
        let converted_actions_duration: Vec<Option<DurationInterval>> = actions_duration
            .into_iter()
            .map(|value| {
                value.map(|(vec1, vec2, b1, b2)| {
                    (
                        vec1.into_iter().map(|e| e.v).collect(),
                        vec2.into_iter().map(|e| e.v).collect(),
                        b1,
                        b2,
                    )
                })
            })
            .collect();

        let mut event_fluents = vec![Vec::new(); actions.len()];
        let mut effect_independent_start_conditions = vec![Vec::new(); actions.len()];
        let mut event_anchors = vec![Vec::new(); actions.len()];
        for (a, le) in &events {
            let duration = &converted_actions_duration[a.idx];
            event_anchors[a.idx] = le
                .iter()
                .map(|(t, _)| (t.is_from_start(), rational_to_f64(&t.delay)))
                .collect();
            for (i, (_, e)) in le.iter().enumerate() {
                let mut reads: FxHashSet<Fluent> = get_fluents(&e.conditions).collect();
                reads.extend(e.effects.iter().flat_map(|eff| get_fluents(&eff.value)));
                if i == 0 {
                    // The duration bounds are read when the action is opened,
                    // i.e. at its first (start) event, so that event reads the
                    // fluents they mention just like a condition would. Without
                    // this, nothing orders the start against the events writing
                    // those fluents and `build_plan` is free to schedule the
                    // action where its duration does not hold.
                    if let Some((lower, upper, _, _)) = duration {
                        reads.extend(get_fluents(lower));
                        reads.extend(get_fluents(upper));
                    }
                }
                let writes: FxHashSet<Fluent> = e.effects.iter().map(|eff| eff.fluent).collect();
                effect_independent_start_conditions[a.idx].push(
                    e.start_conditions
                        .iter()
                        .enumerate()
                        .filter(|(_, c)| get_fluents(c).all(|f| !writes.contains(&f)))
                        .map(|(k, _)| k)
                        .collect(),
                );
                let read_writes: FxHashSet<Fluent> = reads.union(&writes).copied().collect();
                let start_cond_reads: FxHashSet<Fluent> = e
                    .start_conditions
                    .iter()
                    .flat_map(|c| get_fluents(c))
                    .collect();
                let end_cond_reads: FxHashSet<Fluent> = e
                    .end_conditions
                    .iter()
                    .flat_map(|c| get_fluents(c))
                    .collect();
                event_fluents[a.idx].push((
                    reads,
                    writes,
                    read_writes,
                    start_cond_reads,
                    end_cond_reads,
                ));
            }
        }

        let res = SearchSpace {
            actions_duration: converted_actions_duration,
            events,
            relevant_actions,
            compression_safe_actions,
            event_fluents,
            event_anchors,
            effect_independent_start_conditions,
            mutex: MutexChecker::new(),
            precedence: PrecedenceChecker::new(),
            action_objects,
            obj_to_prev_actions_map,
            initial_state: initial_state
                .map(|inner_vec| inner_vec.into_iter().map(|v| v.v).collect()),
            goal: goal.map(|inner_vec| inner_vec.into_iter().map(|e| e.v).collect()),
            deadline: deadline.map(|v| rational_to_f64(&v)),
            epsilon: match &epsilon {
                Some(x) => rational_to_f64(x),
                None => 0.01,
            },
            epsilon_rational: match epsilon {
                Some(x) => x,
                None => mk_rational(1, 100),
            },
            is_temporal,
        };
        Ok(res)
    }

    #[getter]
    #[pyo3(name = "is_temporal")]
    fn py_is_temporal(&self) -> bool {
        self.is_temporal()
    }

    #[getter]
    #[pyo3(name = "relevant_actions")]
    fn py_relevant_actions(&self) -> Vec<Action> {
        self.relevant_actions.clone()
    }

    #[setter]
    #[pyo3(name = "relevant_actions")]
    fn py_set_relevant_actions(&mut self, relevant_actions: Vec<Action>) {
        self.relevant_actions = relevant_actions;
    }

    #[pyo3(name = "reset")]
    fn py_reset(&self) {
        self.reset();
    }

    #[pyo3(name = "initial_state", signature = (initial_state=None))]
    pub fn py_initial_state(
        &self,
        initial_state: Option<Vec<PyExpressionNode>>,
    ) -> PyResult<State> {
        self.initial_state(initial_state)
    }

    #[pyo3(name = "get_successor_states")]
    pub fn py_get_successor_states(&self, state: &State) -> PyResult<Vec<State>> {
        self.get_successor_states(state)
    }

    #[pyo3(name = "get_successor_state")]
    pub fn py_get_successor_state(&self, state: &State, action: Action) -> PyResult<Option<State>> {
        self.get_successor_state(state, action)
    }

    #[pyo3(name = "goal_reached", signature = (state, goal=None))]
    pub fn py_goal_reached(
        &self,
        state: &State,
        goal: Option<Vec<PyExpressionNode>>,
    ) -> PyResult<bool> {
        self.goal_reached(state, goal)
    }

    #[pyo3(name = "subgoals_sat", signature = (state, goal=None))]
    pub fn py_subgoals_sat(
        &self,
        state: &State,
        goal: Option<Vec<PyExpressionNode>>,
    ) -> PyResult<Vec<Vec<PyExpressionNode>>> {
        self.subgoals_sat(state, goal)
    }

    #[pyo3(name = "build_plan", signature = (path))]
    fn py_build_plan<'py>(
        &self,
        py: Python<'py>,
        path: Vec<Action>,
    ) -> PyResult<Option<Vec<PyScheduledAction<'py>>>> {
        let plan = self.build_plan(&path)?;
        let mut res = Vec::with_capacity(plan.len());
        for (start, action, duration) in plan.into_iter() {
            let start = match start {
                Some(start) => Some(big_rational_to_py_fraction(&start, py)?),
                None => None,
            };
            let duration = match duration {
                Some(duration) => Some(big_rational_to_py_fraction(&duration, py)?),
                None => None,
            };
            res.push((start, action, duration));
        }
        Ok(Some(res))
    }
}

impl SearchSpace {
    fn get_successor_state_with_compression(
        &self,
        state: &State,
        action: Action,
        enable_compression_safe_actions: bool,
    ) -> PyResult<Option<State>> {
        if let Some(events) = self.events.get(&action) {
            if let Some((index, id)) = state.todo.get(&action) {
                if let Some((_, e)) = events.get(*index) {
                    // Check if the event is applicable before creating the new state
                    if !self.is_sat(&e.conditions, state)?
                        || !self
                            .effect_independent_start_conditions_hold(action, *index, e, state)?
                    {
                        return Ok(None);
                    }

                    let mut new_state = state.clone_for_child_without_tn();
                    new_state.g += 1.0;

                    if index + 1 >= events.len() {
                        new_state.todo.remove(&action);
                    } else {
                        new_state.todo.insert(action, (index + 1, *id));
                    }
                    if self.expand_event(state, &mut new_state, e, index, id, None)? {
                        return Ok(Some(new_state));
                    }
                }
            } else {
                // Check if action is applicable before creating the new state
                if !self.is_sat(&events[0].1.conditions, state)?
                    || !self.effect_independent_start_conditions_hold(
                        action,
                        0,
                        &events[0].1,
                        state,
                    )?
                {
                    return Ok(None);
                }
                if !self.symmetry_allows_opening(state, action) {
                    return Ok(None);
                }

                let mut new_state = state.clone_for_child_without_tn();
                new_state.g += 1.0;
                if !self.open_action(state, &mut new_state, action, events)? {
                    return Ok(None);
                }

                if enable_compression_safe_actions
                    && self
                        .compression_safe_actions
                        .as_ref()
                        .is_some_and(|is_compression_safe| is_compression_safe[action.idx])
                    && events.len() > 1
                {
                    let start = new_state.todo.remove(&action).unwrap().1;
                    for (index, event) in events.iter().enumerate().skip(1) {
                        // `expand_event` relies on this check having been made
                        if !self.effect_independent_start_conditions_hold(
                            action, index, &event.1, &new_state,
                        )? {
                            return Ok(None);
                        }
                        let state = new_state.clone_for_child_without_tn();
                        new_state.g += 1.0;
                        if !self.expand_event(
                            &state,
                            &mut new_state,
                            &event.1,
                            &index,
                            &start,
                            None,
                        )? {
                            return Ok(None);
                        }
                    }
                }
                return Ok(Some(new_state));
            }
        }
        Ok(None)
    }

    fn is_sat(&self, conditions: &[ExpressionNode], state: &State) -> PyResult<bool> {
        let sat = match internal_evaluate(conditions, state)? {
            ExpressionNode::Bool(v) => v,
            _ => {
                return Err(PyException::new_err(
                    "An action condition is not a boolean expression!",
                ))
            }
        };
        Ok(sat)
    }

    /// Whether event `index`'s effect-independent start conditions hold on the
    /// parent `state`. They can't change value in the child, so this rejects
    /// exactly the successors `expand_event`'s post-effect check would, but
    /// before cloning the state or opening the action.
    fn effect_independent_start_conditions_hold(
        &self,
        action: Action,
        index: usize,
        e: &Event,
        state: &State,
    ) -> PyResult<bool> {
        for &k in self.effect_independent_start_conditions[action.idx][index].iter() {
            if !self.is_sat(&e.start_conditions[k], state)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn expand_event(
        &self,
        state: &State,
        new_state: &mut State,
        e: &Event,
        index: &usize,
        id: &Timepoint,
        pending_opening: Option<PendingOpening>,
    ) -> PyResult<bool> {
        new_state.path = PersistentList::append((e.action, e.pos, *id), &new_state.path);

        // check conditions is done before calling this method

        // The inherited active conditions need no check here, on the parent:
        // every state this search space returns passed the post-effect check
        // below on its own assignments, and the initial state has none (a
        // `State` can't be built from Python), so they all hold on `state`

        // remove end conditions
        for c in e.end_conditions.iter() {
            new_state.active_conditions.remove(c);
        }

        // insert start conditions
        for c in e.start_conditions.iter() {
            new_state.active_conditions.insert(c.to_vec());
        }

        // apply effects
        for eff in e.effects.iter() {
            new_state.assignments[eff.fluent.idx] = internal_evaluate(&eff.value, state)?;
        }

        // check active conditions. Without effects the child's assignments are
        // the parent's: the inherited conditions hold on them (see above), and
        // the new start conditions, all effect-independent, were checked by
        // the caller with `effect_independent_start_conditions_hold`
        if !e.effects.is_empty() {
            for c in new_state.active_conditions.iter() {
                let sat = match internal_evaluate(c, new_state)? {
                    ExpressionNode::Bool(v) => v,
                    _ => {
                        return Err(PyException::new_err(
                            "An action condition is not a boolean expression!",
                        ))
                    }
                };
                if !sat {
                    return Ok(false);
                }
            }
        }

        if self.is_temporal {
            // Only now, with the non-temporal checks passed, copy the parent's
            // network (a compression-safe chain already owns one after its
            // first event), with room for the timepoints an opening appends
            if new_state.temporal_network.is_none() {
                let extra = if pending_opening.is_some() { 2 } else { 0 };
                new_state.temporal_network = state
                    .temporal_network
                    .as_ref()
                    .map(|tn| tn.clone_reserving(extra));
            }
            let tn = new_state.temporal_network.as_mut().unwrap();
            if let Some(pending_opening) = pending_opening {
                self.add_opening_constraints(state, tn, pending_opening)?;
            }
            // Add temporal constraints between past or todo events and the current one
            let (ev, ev_delay) = self.event_timepoint(e.action, e.pos, *id);

            // Only two of the edges from past events are needed. The edge from
            // the immediate predecessor is always added, so the path is a
            // chain in which each event is no later than the next: a 0-edge
            // from any older event is already implied. Likewise, once the
            // most recent mutex predecessor e_m gets its -epsilon edge, every
            // older event e_j satisfies t(e_j) <= t(e_m) <= t(e) - epsilon, so
            // the scan stops there.
            let e_id = (e.action, *index);
            let mut is_predecessor = true;
            let ev = (ev, &ev_delay);
            for e2 in PersistentList::iter_rev(&state.path) {
                let (ev2, ev2_delay) = self.event_timepoint(e2.0, e2.1, e2.2);
                let ev2 = (ev2, &ev2_delay);
                let e2_id = (e2.0, e2.1);
                if self.mutex.check(&(e_id, e2_id), &self.event_fluents) {
                    if !add_event_constraint(tn, ev2, ev, -self.epsilon) {
                        return Ok(false);
                    }
                    break;
                }
                if is_predecessor {
                    if !add_event_constraint(tn, ev2, ev, 0.0) {
                        return Ok(false);
                    }
                    is_predecessor = false;
                }
            }
            for (a, i) in new_state.todo.iter() {
                for (j, (_, e2)) in self.events[a].iter().enumerate().skip(i.0) {
                    let e_id = (e.action, *index);
                    let e2_id = (*a, j);
                    let (ev2, ev2_delay) = self.event_timepoint(e2.action, e2.pos, i.1);
                    let b = if self.mutex.check(&(e_id, e2_id), &self.event_fluents) {
                        -self.epsilon
                    } else {
                        0.0
                    };
                    if !add_event_constraint(tn, ev, (ev2, &ev2_delay), b) {
                        return Ok(false);
                    }
                }
            }
            if !tn.check() {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Whether symmetry breaking lets `action` be opened after the parent
    /// `state`. It reads only the parent's path, so it runs before cloning the
    /// state for the child.
    fn symmetry_allows_opening(&self, state: &State, action: Action) -> bool {
        if let (Some(action_objects), Some(obj_to_prev_actions_map)) =
            (&self.action_objects, &self.obj_to_prev_actions_map)
        {
            for obj in &action_objects[action.idx] {
                let prev_actions = &obj_to_prev_actions_map[obj.idx];

                if prev_actions.is_empty() || prev_actions.contains(&action) {
                    continue;
                }

                if !PersistentList::iter_rev(&state.path).any(|(a, _, _)| prev_actions.contains(a))
                {
                    return false;
                }
            }
        }
        true
    }

    fn open_action(
        &self,
        state: &State,
        new_state: &mut State,
        action: Action,
        events: &[(Timing, Event)],
    ) -> PyResult<bool> {
        // The instance's start and end are the next two timepoints of the
        // parent's network, which the child's copies. `expand_event` appends
        // them and adds the opening constraints, after the checks that reject
        // most successors
        let mut start = 0;
        let mut pending_opening = None;
        if self.is_temporal {
            start = state
                .temporal_network
                .as_ref()
                .expect("a temporal state has a network")
                .num_timepoints() as Timepoint;
            pending_opening = Some((action, start));
            if events.len() > 1 {
                new_state.todo.insert(action, (1, start));
            }
        }
        self.expand_event(state, new_state, &events[0].1, &0, &start, pending_opening)
    }

    /// Appends the start and end timepoints of an action opened by
    /// `open_action` and adds its constraints: the duration bounds and, under
    /// a deadline, the edge to plan end. Events have no timepoints of their
    /// own: `event_timepoint` maps them onto start/end.
    fn add_opening_constraints(
        &self,
        state: &State,
        tn: &mut DeltaSTN<f64>,
        (action, start): PendingOpening,
    ) -> PyResult<()> {
        let appended = tn.add_timepoints(2);
        debug_assert_eq!(appended, start);
        let end = start + 1;
        let duration = self.actions_duration[action.idx].as_ref();
        let mut lb: f64 = 0.0;
        let mut ub: f64 = 0.0;
        if let Some(duration) = duration {
            let d = duration;
            lb = -expression_node_to_f64(&internal_evaluate(&d.0, state)?)?;
            ub = expression_node_to_f64(&internal_evaluate(&d.1, state)?)?;
            if d.2 {
                lb -= self.epsilon;
            }
            if d.3 {
                ub -= self.epsilon;
            }
        }
        tn.add(start, end, &lb);
        tn.add(end, start, &ub);
        // Under a deadline, plan end follows the latest action end (plus
        // epsilon) and the network bounds its earliest time by the deadline
        // (see `DeltaSTN::with_deadline`). Without one, plan end would be a
        // sink nothing reads, so the edge is skipped
        if self.deadline.is_some() {
            tn.add(end, PLAN_END, &-self.epsilon);
        }
        Ok(())
    }

    /// Adds `build_plan`'s ordering constraints between event `e` (of the
    /// action instance starting at `id`) and the events already on the plan or
    /// still to come. The plan comes from the search, whose networks were
    /// consistent, so a violated constant between events of one action
    /// instance can't happen here
    fn add_plan_event_constraints(
        &self,
        tn: &mut DeltaSTN<BigRational>,
        e: &Event,
        id: Timepoint,
        event_path: &[(Event, Timepoint)],
        todo: &FxHashMap<Action, (usize, Timepoint)>,
    ) {
        let e_id = (e.action, e.pos);
        let ev = self.event_timepoint_rational(e.action, e.pos, id);
        for (e2, id2) in event_path.iter() {
            let e2_id = (e2.action, e2.pos);
            let ev2 = self.event_timepoint_rational(e2.action, e2.pos, *id2);
            if self.mutex.check(&(e_id, e2_id), &self.event_fluents) {
                add_event_constraint(tn, ev2, ev, -self.epsilon_rational.clone());
            } else if self.precedence.check(&(e2_id, e_id), &self.event_fluents) {
                add_event_constraint(tn, ev2, ev, mk_rational(0, 1));
            }
        }
        for (a, i) in todo.iter() {
            for j in i.0..self.events[a].len() {
                let e2_id = (*a, j);
                if self.mutex.check(&(e_id, e2_id), &self.event_fluents) {
                    let ev2 = self.event_timepoint_rational(*a, j, i.1);
                    add_event_constraint(tn, ev, ev2, -self.epsilon_rational.clone());
                }
            }
        }
    }

    /// Where event `index` of `action` sits in the STN, given the start
    /// timepoint `start` of its action instance.
    ///
    /// Events have no timepoint of their own: each one is fixed at its action
    /// instance's start or end timepoint plus a constant delay. Returns that
    /// (anchor timepoint, delay) pair. The end is the timepoint right after
    /// the start (see `add_opening_constraints`).
    fn event_timepoint(&self, action: Action, index: usize, start: Timepoint) -> (Timepoint, f64) {
        let (is_start, delay) = self.event_anchors[action.idx][index];
        (start + !is_start as Timepoint, delay)
    }

    /// `event_timepoint` with the exact delay, for `build_plan`
    fn event_timepoint_rational(
        &self,
        action: Action,
        index: usize,
        start: Timepoint,
    ) -> (Timepoint, &BigRational) {
        let t = &self.events[&action][index].0;
        (start + !t.is_from_start() as Timepoint, &t.delay)
    }
}

impl SearchSpaceTrait for SearchSpace {
    fn is_temporal(&self) -> bool {
        self.is_temporal
    }

    fn reset(&self) {
        // DO nothing :)
    }

    fn initial_state(&self, initial_state: Option<Vec<PyExpressionNode>>) -> PyResult<State> {
        let init: Vector<ExpressionNode> = match initial_state {
            Some(v) => v.iter().map(|v| v.v.clone()).collect(),
            None => match &self.initial_state {
                Some(v) => Vector::from(v),
                None => {
                    return Err(PyException::new_err(
                        "The initial state must be defined somewhere!",
                    ));
                }
            },
        };
        let tn: Option<DeltaSTN<f64>> = if self.is_temporal {
            let tolerance = self.epsilon / 1000.0;
            let mut tn = match self.deadline {
                Some(deadline) => DeltaSTN::with_deadline(tolerance, PLAN_END, deadline),
                None => DeltaSTN::new(tolerance),
            };
            let plan_end = tn.add_timepoints(1);
            debug_assert_eq!(plan_end, PLAN_END);
            Some(tn)
        } else {
            None
        };
        Ok(State {
            assignments: init,
            temporal_network: tn,
            todo: FxHashMap::with_hasher(FxBuildHasher),
            active_conditions: HashMultiSet::new(),
            g: 0.0,
            path: PersistentList::new(),
            heuristic_cache: Mutex::new(FxHashMap::with_hasher(FxBuildHasher)),
        })
    }

    fn get_successor_states_iter<'a>(
        &'a self,
        state: &'a State,
    ) -> impl Iterator<Item = PyResult<State>> + 'a {
        self.relevant_actions
            .iter()
            .filter_map(|action| self.get_successor_state(state, *action).transpose())
    }

    fn get_successor_state(&self, state: &State, action: Action) -> PyResult<Option<State>> {
        self.get_successor_state_with_compression(state, action, true)
    }

    fn goal_reached(&self, state: &State, goal: Option<Vec<PyExpressionNode>>) -> PyResult<bool> {
        if !state.todo.is_empty() {
            return Ok(false);
        }
        let goal = goal.map(|g| g.into_iter().map(|e| e.v).collect());
        let g = match &goal {
            Some(v) => v,
            None => match &self.goal {
                Some(v) => v,
                None => {
                    return Err(PyException::new_err("The goal must be defined somewhere!"));
                }
            },
        };
        match internal_evaluate(g, state)? {
            ExpressionNode::Bool(v) => Ok(v),
            _ => Err(PyException::new_err(
                "The goal is not a boolean expression!",
            )),
        }
    }

    fn subgoals_sat(
        &self,
        state: &State,
        goal: Option<Vec<PyExpressionNode>>,
    ) -> PyResult<Vec<Vec<PyExpressionNode>>> {
        let goals = match goal {
            Some(v) => split_expression(&v.into_iter().map(|e| e.v).collect::<Vec<_>>())?,
            None => match &self.goal {
                Some(v) => split_expression(v)?,
                None => {
                    return Err(PyException::new_err("The goal must be defined somewhere!"));
                }
            },
        };
        let mut res: FxHashSet<_> = FxHashSet::with_hasher(FxBuildHasher);
        for g in goals {
            if internal_evaluate(&g, state)? == ExpressionNode::Bool(true) {
                res.insert(g.into_iter().map(|v| PyExpressionNode { v }).collect());
            }
        }
        Ok(res.into_iter().collect())
    }

    fn build_plan(&self, path: &[Action]) -> PyResult<Vec<ScheduledAction>> {
        if !self.is_temporal {
            return Ok(path.iter().map(|a| (None, *a, None)).collect());
        }

        let mut tn: DeltaSTN<BigRational> = DeltaSTN::new_without_subsumption(mk_rational(0, 1));
        let mut todo: FxHashMap<Action, (usize, Timepoint)> = FxHashMap::with_hasher(FxBuildHasher);
        let mut event_path: Vec<(Event, Timepoint)> = Vec::new();
        // Each action instance and its start timepoint (its end is the next)
        let mut openings: Vec<(Action, Timepoint)> = Vec::new();
        let mut state = self.initial_state(None)?;
        for action in path {
            if let Some(events) = self.events.get(action).cloned() {
                if let Some((index, id)) = todo.get(action).cloned() {
                    if let Some((_, e)) = events.get(index) {
                        if index + 1 >= events.len() {
                            todo.remove(action);
                        } else {
                            todo.insert(*action, (index + 1, id));
                        }
                        self.add_plan_event_constraints(&mut tn, e, id, &event_path, &todo);
                        event_path.push((e.clone(), id));
                    }
                } else {
                    let start = tn.add_timepoints(2);
                    let end = start + 1;
                    openings.push((*action, start));
                    let duration = self.actions_duration[action.idx].as_ref();
                    let (lb, ub) = match duration {
                        Some(d) => {
                            let mut lb = -get_rational_from_expression_node(&internal_evaluate(
                                &d.0, &state,
                            )?)?;
                            let mut ub = get_rational_from_expression_node(&internal_evaluate(
                                &d.1, &state,
                            )?)?;
                            if d.2 {
                                lb -= self.epsilon_rational.clone();
                            }
                            if d.3 {
                                ub -= self.epsilon_rational.clone();
                            }
                            (lb, ub)
                        }
                        None => (mk_rational(0, 1), mk_rational(0, 1)),
                    };
                    tn.add(start, end, &lb);
                    tn.add(end, start, &ub);
                    let e = &events[0].1;
                    self.add_plan_event_constraints(&mut tn, e, start, &event_path, &todo);
                    event_path.push((e.clone(), start));
                    if events.len() > 1 {
                        todo.insert(*action, (1, start));
                    }
                }
            }
            // Advance to the successor state only after evaluating the action's
            // duration bounds above, so they are evaluated against the pre-action state
            state = self
                .get_successor_state_with_compression(&state, *action, false)?
                .unwrap();
        }

        let mut res = Vec::with_capacity(openings.len());
        for (action, start) in openings {
            let st = tn.earliest_time(start);
            let d = tn.earliest_time(start + 1) - st.clone();
            let d: Option<BigRational> = if d == mk_rational(0, 1) {
                None
            } else {
                Some(d)
            };
            res.push((Some(st), action, d));
        }
        res.sort();
        Ok(res)
    }
}
