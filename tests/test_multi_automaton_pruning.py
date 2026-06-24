import json
from pathlib import Path

from aalpy.automata.Dfa import Dfa, DfaState
from tamerlite.core.search import WeakEqState
from tamerlite.core.search_space import Action, Event, MultiSet, SearchSpace, State, Timing
from tamerlite.pruning_automata import DfaPruningModel, MultiAutomatonPruningModel


def _build_dfa(bad_token: str) -> Dfa:
    safe = DfaState("safe", is_accepting=False)
    bad = DfaState("bad", is_accepting=True)
    safe.transitions[bad_token] = bad
    bad.transitions[bad_token] = bad
    return Dfa(safe, [safe, bad])


def _instant_event(action: Action) -> tuple[Timing, Event]:
    return (
        Timing(True, 0),
        Event(
            action=action,
            pos=0,
            conditions=(True,),
            start_conditions=(),
            end_conditions=(),
            effects=(),
        ),
    )


class _FakeFluent:
    def __init__(self, name: str):
        self.name = name


class _FakeAtom:
    def __init__(self, name: str, args: list[str]):
        self._name = name
        self.args = args

    def is_fluent_exp(self):
        return True

    def fluent(self):
        return _FakeFluent(self._name)


class _FakeGoal(_FakeAtom):
    def __init__(self, name: str, args: list[str], negated: bool = False):
        super().__init__(name, args)
        self._negated = negated

    def is_not(self):
        return self._negated


class _FakeObjectRef:
    def __init__(self, name: str):
        self._name = name

    def __str__(self):
        return self._name


class _FakeValue:
    def __init__(self, value, kind: str):
        self._value = value
        self._kind = kind

    def is_bool_constant(self):
        return self._kind == "bool"

    def bool_constant_value(self):
        return self._value

    def is_int_constant(self):
        return self._kind == "int"

    def int_constant_value(self):
        return self._value

    def is_real_constant(self):
        return self._kind == "real"

    def is_object_exp(self):
        return self._kind == "object"

    def object(self):
        return _FakeObjectRef(self._value)

    def __str__(self):
        return str(self._value)


def test_multi_automaton_progress_is_part_of_state_identity():
    state_a = State([], None, {}, MultiSet(), 0, [])
    state_b = State([], None, {}, MultiSet(), 0, [])
    state_a.pruning_state = type("Progress", (), {"object_states": (("robot", "r0", "safe"),)})()
    state_b.pruning_state = type("Progress", (), {"object_states": (("robot", "r0", "bad"),)})()

    assert state_a != state_b
    assert WeakEqState(state_a) != WeakEqState(state_b)


def test_single_dfa_plain_uses_exact_grounded_action_tokens():
    inspect_r0 = Action(0)

    pruning_model = DfaPruningModel(_build_dfa("inspect_r0"))
    pruning_model.bind_to_planner(
        action_by_name={"inspect_r0": inspect_r0},
        objects_by_type={"robot": ["r0"]},
    )

    progress = pruning_model.initial_state
    next_progress = pruning_model.advance(progress, inspect_r0)
    assert pruning_model.is_prunable(next_progress)
    assert pruning_model.pruning_labels(next_progress) == ("single-dfa-plain",)


def test_single_dfa_abstract_maps_grounded_actions_to_lifted_action_names(tmp_path: Path):
    automaton_dir = tmp_path / "single_dfa_abstract"
    automaton_dir.mkdir(parents=True)

    dfa = _build_dfa("inspect")
    dfa.save(str(automaton_dir / "automaton"), file_type="dot")
    (automaton_dir / "automaton.metadata.json").write_text(
        json.dumps(
            {
                "format": "automaton_metadata/v1",
                "metadata": {
                    "split_ground_actions": False,
                    "abstract_ground_actions": True,
                },
            }
        ),
        encoding="utf-8",
    )

    pruning_model = DfaPruningModel.from_automaton_files(automaton_dir / "automaton.dot")
    inspect_r0 = Action(0)
    pruning_model.bind_to_planner(
        action_by_name={"inspect_r0": inspect_r0},
        objects_by_type={"robot": ["r0"]},
    )

    progress = pruning_model.initial_state
    next_progress = pruning_model.advance(progress, inspect_r0)
    assert pruning_model.is_prunable(next_progress)
    assert pruning_model.pruning_labels(next_progress) == ("single-dfa-abstract",)


def test_single_dfa_split_consumes_action_name_and_parameter_tokens(tmp_path: Path):
    automaton_dir = tmp_path / "single_dfa_split"
    automaton_dir.mkdir(parents=True)

    safe = DfaState("safe", is_accepting=False)
    mid = DfaState("mid", is_accepting=False)
    bad = DfaState("bad", is_accepting=True)
    safe.transitions["inspect"] = mid
    mid.transitions["r0"] = bad
    dfa = Dfa(safe, [safe, mid, bad])
    dfa.save(str(automaton_dir / "automaton"), file_type="dot")
    (automaton_dir / "automaton.metadata.json").write_text(
        json.dumps(
            {
                "format": "automaton_metadata/v1",
                "metadata": {
                    "split_ground_actions": True,
                    "abstract_ground_actions": False,
                },
            }
        ),
        encoding="utf-8",
    )

    pruning_model = DfaPruningModel.from_automaton_files(automaton_dir / "automaton.dot")
    inspect_r0 = Action(0)
    pruning_model.bind_to_planner(
        action_by_name={"inspect_r0": inspect_r0},
        objects_by_type={"robot": ["r0"]},
    )

    progress = pruning_model.initial_state
    next_progress = pruning_model.advance(progress, inspect_r0)
    assert pruning_model.is_prunable(next_progress)
    assert pruning_model.pruning_labels(next_progress) == ("single-dfa-split",)


def test_multi_automaton_prunes_only_the_object_that_reaches_accepting():
    inspect_r0 = Action(0)
    inspect_r1 = Action(1)
    wait = Action(2)
    actions = [inspect_r0, inspect_r1, wait]
    events = {
        inspect_r0: [_instant_event(inspect_r0)],
        inspect_r1: [_instant_event(inspect_r1)],
        wait: [_instant_event(wait)],
    }

    pruning_model = MultiAutomatonPruningModel(
        action_parameter_types={"inspect": ["robot"], "wait": []},
        placeholders_by_type={"robot": "r"},
        automata={"robot": type("Spec", (), {"placeholder": "r", "dfa": _build_dfa("inspect(*r*)")})()},
        drop_wildcards=True,
        abstract_other_objects=False,
    )
    pruning_model.bind_to_planner(
        action_by_name={"inspect_r0": inspect_r0, "inspect_r1": inspect_r1, "wait": wait},
        objects_by_type={"robot": ["r0", "r1"]},
    )

    ss = SearchSpace(
        actions_duration=[None, None, None],
        events=events,
        actions=actions,
        action_objects=None,
        obj_to_prev_actions_map=None,
        initial_state=[],
        goal=(True,),
        dfa=pruning_model,
    )

    init = ss.initial_state()
    assert init.pruning_state.object_states == ()

    r0_state = ss.get_successor_state(init, inspect_r0)
    assert r0_state is not None
    assert pruning_model.is_prunable(r0_state.pruning_state)
    assert r0_state.pruning_state.object_states == (("robot", "r0", "bad"),)

    r1_state = ss.get_successor_state(init, inspect_r1)
    assert r1_state is not None
    assert pruning_model.is_prunable(r1_state.pruning_state)
    assert r1_state.pruning_state.object_states == (("robot", "r1", "bad"),)

    wait_state = ss.get_successor_state(init, wait)
    assert wait_state is not None
    assert wait_state.pruning_state.object_states == ()

    successors = list(ss.get_successor_states(init))
    assert successors == [wait_state]


def test_multi_automaton_loader_reads_summary_signature(tmp_path: Path):
    automaton_dir = tmp_path / "robot"
    automaton_dir.mkdir(parents=True)

    dfa = _build_dfa("inspect(*r*)")
    dfa.save(str(automaton_dir / "automaton"), file_type="dot")

    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        """
{
  "signature": {
    "action_parameter_types": {"inspect": ["robot"]},
    "placeholders_by_type": {"robot": "r"},
    "drop_wildcards": true,
    "abstract_other_objects": false
  },
  "automata": {
    "robot": {
      "dot_path": "robot/automaton.dot",
      "placeholder": "r",
      "signature": {
        "placeholder": "r"
      }
    }
  }
}
""".strip(),
        encoding="utf-8",
    )

    pruning_model = MultiAutomatonPruningModel.from_summary_file(summary_path)
    inspect_r0 = Action(0)
    pruning_model.bind_to_planner(
        action_by_name={"inspect_r0": inspect_r0},
        objects_by_type={"robot": ["r0"]},
    )

    progress = pruning_model.initial_state
    next_progress = pruning_model.advance(progress, inspect_r0)
    assert pruning_model.is_prunable(next_progress)


def test_multi_automaton_bind_and_advance_support_hyphenated_action_names():
    to_tray = Action(0)

    pruning_model = MultiAutomatonPruningModel(
        action_parameter_types={"to-tray": ["item", "arm", "bot"]},
        placeholders_by_type={"item": "i", "arm": "a", "bot": "b"},
        automata={
            "item": type(
                "Spec",
                (),
                {"placeholder": "i", "dfa": _build_dfa("to-tray(*i*,right2,bot2)")},
            )(),
        },
        drop_wildcards=True,
        abstract_other_objects=False,
    )
    pruning_model.bind_to_planner(
        action_by_name={"to-tray_item3_right2_bot2": to_tray},
        objects_by_type={"item": ["item3"], "arm": ["right2"], "bot": ["bot2"]},
    )

    progress = pruning_model.initial_state
    next_progress = pruning_model.advance(progress, to_tray)
    assert pruning_model.is_prunable(next_progress)
    assert pruning_model.pruning_labels(next_progress) == ("item",)


def test_multi_automaton_abstracts_other_objects_for_robot_focus():
    treat_r0 = Action(0)
    treat_r1 = Action(1)

    pruning_model = MultiAutomatonPruningModel(
        action_parameter_types={"make_treatment": ["robot", "pallet", "position"]},
        placeholders_by_type={"robot": "r", "pallet": "b", "position": "p"},
        automata={
            "robot": type("Spec", (), {"placeholder": "r", "dfa": _build_dfa("make_treatment(*r*,b,p)")})(),
        },
        drop_wildcards=True,
        abstract_other_objects=True,
    )
    pruning_model.bind_to_planner(
        action_by_name={
            "make_treatment_r0_b1_p5": treat_r0,
            "make_treatment_r1_b2_p3": treat_r1,
        },
        objects_by_type={"robot": ["r0", "r1"]},
    )

    progress = pruning_model.initial_state
    next_progress = pruning_model.advance(progress, treat_r0)
    assert pruning_model.is_prunable(next_progress)
    assert next_progress.object_states == (("robot", "r0", "bad"),)

    next_progress = pruning_model.advance(progress, treat_r1)
    assert pruning_model.is_prunable(next_progress)
    assert next_progress.object_states == (("robot", "r1", "bad"),)


def test_multi_automaton_abstracts_other_objects_for_position_focus():
    move_p5 = Action(0)
    move_depot = Action(1)

    pruning_model = MultiAutomatonPruningModel(
        action_parameter_types={"move": ["robot", "position"]},
        placeholders_by_type={"robot": "r", "position": "p"},
        automata={
            "position": type("Spec", (), {"placeholder": "p", "dfa": _build_dfa("move(r,*p*)")})(),
        },
        drop_wildcards=True,
        abstract_other_objects=True,
    )
    pruning_model.bind_to_planner(
        action_by_name={
            "move_r0_p5": move_p5,
            "move_r1_DEPOT": move_depot,
        },
        objects_by_type={"position": ["p5", "DEPOT"]},
    )

    progress = pruning_model.initial_state
    next_progress = pruning_model.advance(progress, move_p5)
    assert pruning_model.is_prunable(next_progress)
    assert next_progress.object_states == (("position", "p5", "bad"),)

    next_progress = pruning_model.advance(progress, move_depot)
    assert pruning_model.is_prunable(next_progress)
    assert next_progress.object_states == (("position", "DEPOT", "bad"),)


def test_multi_automaton_marks_same_type_non_focus_objects_with_tilde_placeholder():
    move = Action(0)

    pruning_model = MultiAutomatonPruningModel(
        action_parameter_types={"move": ["robot", "position", "position"]},
        placeholders_by_type={"robot": "r", "position": "p"},
        automata={
            "position__position": type(
                "Spec",
                (),
                {
                    "profile_types": ("position", "position"),
                    "placeholder": "p1",
                    "slot_placeholders": ("p1", "p2"),
                    "dfa": _build_dfa("move(r,*p1*,~p)"),
                },
            )(),
        },
        drop_wildcards=True,
        abstract_other_objects=True,
    )
    pruning_model.bind_to_planner(
        action_by_name={"move_r0_p1_p3": move},
        objects_by_type={"position": ["p1", "p2", "p3"]},
    )

    progress = pruning_model.initial_state
    next_progress = pruning_model.advance(progress, move)
    assert pruning_model.is_prunable(next_progress)
    assert ("position__position", "p1", "p2", "bad") in next_progress.object_states


def test_multi_automaton_legacy_plain_same_type_placeholder_still_matches():
    move = Action(0)

    pruning_model = MultiAutomatonPruningModel(
        action_parameter_types={"move": ["robot", "position", "position"]},
        placeholders_by_type={"robot": "r", "position": "p"},
        automata={
            "position__position": type(
                "Spec",
                (),
                {
                    "profile_types": ("position", "position"),
                    "placeholder": "p1",
                    "slot_placeholders": ("p1", "p2"),
                    "dfa": _build_dfa("move(r,*p1*,p)"),
                },
            )(),
        },
        drop_wildcards=True,
        abstract_other_objects=True,
    )
    pruning_model.bind_to_planner(
        action_by_name={"move_r0_p1_p3": move},
        objects_by_type={"position": ["p1", "p2", "p3"]},
    )

    progress = pruning_model.initial_state
    next_progress = pruning_model.advance(progress, move)
    assert pruning_model.is_prunable(next_progress)
    assert ("position__position", "p1", "p2", "bad") in next_progress.object_states


def test_multi_automaton_matches_numeric_parameters_via_int_placeholder():
    initialize_drawer = Action(0)

    pruning_model = MultiAutomatonPruningModel(
        action_parameter_types={"initializeDrawer": ["drawer", "cardboardtype", "integer"]},
        placeholders_by_type={"drawer": "d", "cardboardtype": "c"},
        automata={
            "drawer": type("Spec", (), {"placeholder": "d", "dfa": _build_dfa("initializeDrawer(*d*,c,INT)")})(),
        },
        drop_wildcards=True,
        abstract_other_objects=True,
    )
    pruning_model.bind_to_planner(
        action_by_name={
            "initializeDrawer_drawer_0_cardboard_type_3_6": initialize_drawer,
        },
        objects_by_type={
            "drawer": ["drawer_0"],
            "cardboardtype": ["cardboard_type_3"],
        },
    )

    progress = pruning_model.initial_state
    next_progress = pruning_model.advance(progress, initialize_drawer)
    assert pruning_model.is_prunable(next_progress)
    assert next_progress.object_states == (("drawer", "drawer_0", "bad"),)


def test_multi_automaton_plain_keeps_concrete_numeric_parameters():
    initialize_drawer = Action(0)

    pruning_model = MultiAutomatonPruningModel(
        action_parameter_types={"initializeDrawer": ["drawer", "cardboardtype", "integer"]},
        placeholders_by_type={"drawer": "d", "cardboardtype": "c"},
        automata={
            "drawer": type("Spec", (), {"placeholder": "d", "dfa": _build_dfa("initializeDrawer(*d*,cardboard_type_3,6)")})(),
        },
        drop_wildcards=True,
        abstract_other_objects=False,
    )
    pruning_model.bind_to_planner(
        action_by_name={
            "initializeDrawer_drawer_0_cardboard_type_3_6": initialize_drawer,
        },
        objects_by_type={
            "drawer": ["drawer_0"],
            "cardboardtype": ["cardboard_type_3"],
        },
    )

    progress = pruning_model.initial_state
    next_progress = pruning_model.advance(progress, initialize_drawer)
    assert pruning_model.is_prunable(next_progress)
    assert next_progress.object_states == (("drawer", "drawer_0", "bad"),)


def test_multi_automaton_supports_ordered_tuple_profiles_of_the_same_type():
    move_forward = Action(0)
    move_reverse = Action(1)

    pruning_model = MultiAutomatonPruningModel(
        action_parameter_types={"move": ["robot", "position", "position"]},
        placeholders_by_type={"robot": "r", "position": "p"},
        automata={
            "position__position": type(
                "Spec",
                (),
                {
                    "profile_types": ("position", "position"),
                    "placeholder": "p1",
                    "slot_placeholders": ("p1", "p2"),
                    "dfa": _build_dfa("move(r,*p1*,*p2*)"),
                },
            )(),
        },
        drop_wildcards=True,
        abstract_other_objects=True,
    )
    pruning_model.bind_to_planner(
        action_by_name={
            "move_r0_p1_p2": move_forward,
            "move_r0_p2_p1": move_reverse,
        },
        objects_by_type={"position": ["p1", "p2"]},
    )

    progress = pruning_model.initial_state
    next_progress = pruning_model.advance(progress, move_forward)
    assert pruning_model.is_prunable(next_progress)
    assert ("position__position", "p1", "p2", "bad") in next_progress.object_states
    assert ("position__position", "p2", "p1", "safe") not in next_progress.object_states

    next_progress = pruning_model.advance(progress, move_reverse)
    assert pruning_model.is_prunable(next_progress)
    assert ("position__position", "p1", "p2", "safe") not in next_progress.object_states
    assert ("position__position", "p2", "p1", "bad") in next_progress.object_states


def test_multi_automaton_creates_tuple_states_lazily_and_preserves_existing_progress():
    move_forward = Action(0)
    move_reverse = Action(1)

    pruning_model = MultiAutomatonPruningModel(
        action_parameter_types={"move": ["robot", "position", "position"]},
        placeholders_by_type={"robot": "r", "position": "p"},
        automata={
            "position__position": type(
                "Spec",
                (),
                {
                    "profile_types": ("position", "position"),
                    "placeholder": "p1",
                    "slot_placeholders": ("p1", "p2"),
                    "dfa": _build_dfa("move(r,*p1*,*p2*)"),
                },
            )(),
        },
        drop_wildcards=True,
        abstract_other_objects=True,
    )
    pruning_model.bind_to_planner(
        action_by_name={
            "move_r0_p1_p2": move_forward,
            "move_r0_p2_p3": move_reverse,
        },
        objects_by_type={"position": ["p1", "p2", "p3"]},
    )

    progress = pruning_model.initial_state
    assert progress.object_states == ()

    after_first = pruning_model.advance(progress, move_forward)
    assert ("position__position", "p1", "p2", "bad") in after_first.object_states

    after_second = pruning_model.advance(after_first, move_reverse)
    assert ("position__position", "p1", "p2", "bad") in after_second.object_states
    assert ("position__position", "p2", "p3", "bad") in after_second.object_states


def test_multi_automaton_initializes_lazy_monitor_with_init_and_goal_prefixes():
    load = Action(0)

    start = DfaState("start", is_accepting=False)
    after_init_marker = DfaState("after_init_marker", is_accepting=False)
    after_init = DfaState("after_init", is_accepting=False)
    after_goal_marker = DfaState("after_goal_marker", is_accepting=False)
    after_goal = DfaState("after_goal", is_accepting=False)
    after_trace_marker = DfaState("after_trace_marker", is_accepting=False)
    bad = DfaState("bad", is_accepting=True)

    start.transitions["<INIT>"] = after_init_marker
    after_init_marker.transitions["components_on_kit(*k*,0)=c"] = after_init
    after_init.transitions["<GOAL>"] = after_goal_marker
    after_goal_marker.transitions["completed(0,*k*)"] = after_goal
    after_goal.transitions["<TRACE>"] = after_trace_marker
    after_trace_marker.transitions["load(r,l,c,*k*,INT)"] = bad

    pruning_model = MultiAutomatonPruningModel(
        action_parameter_types={"load": ["robot", "location", "component", "kit", "integer"]},
        placeholders_by_type={"robot": "r", "location": "l", "component": "c", "kit": "k"},
        automata={
            "kit": type("Spec", (), {"placeholder": "k", "dfa": Dfa(start, [start, after_init_marker, after_init, after_goal_marker, after_goal, after_trace_marker, bad])})(),
        },
        drop_wildcards=True,
        abstract_other_objects=True,
        include_goal_prefixes=True,
        include_init_prefixes=True,
    )
    pruning_model.bind_to_planner(
        action_by_name={"load_r0_l1_c1_k1_0": load},
        objects_by_type={
            "robot": ["r0"],
            "location": ["l1"],
            "component": ["c1"],
            "kit": ["k1"],
        },
        initial_values={
            _FakeAtom("components_on_kit", ["k1", "0"]): _FakeValue("c1", "object"),
        },
        goals=[_FakeGoal("completed", ["0", "k1"])],
    )

    progress = pruning_model.initial_state
    next_progress = pruning_model.advance(progress, load)
    assert pruning_model.is_prunable(next_progress)
    assert next_progress.object_states == (("kit", "k1", "bad"),)
