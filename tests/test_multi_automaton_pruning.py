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
    (automaton_dir / "automaton.dataset.json").write_text(
        json.dumps(
            {
                "format": "flat_trace_dataset/v1",
                "positive": [["inspect"]],
                "negative": [],
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
    (automaton_dir / "automaton.dataset.json").write_text(
        json.dumps(
            {
                "format": "flat_trace_dataset/v1",
                "positive": [["inspect", "r0"]],
                "negative": [],
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
    assert init.pruning_state.object_states == (("robot", "r0", "safe"), ("robot", "r1", "safe"))

    r0_state = ss.get_successor_state(init, inspect_r0)
    assert r0_state is not None
    assert pruning_model.is_prunable(r0_state.pruning_state)

    r1_state = ss.get_successor_state(init, inspect_r1)
    assert r1_state is not None
    assert pruning_model.is_prunable(r1_state.pruning_state)

    wait_state = ss.get_successor_state(init, wait)
    assert wait_state is not None
    assert wait_state.pruning_state.object_states == (("robot", "r0", "safe"), ("robot", "r1", "safe"))

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
    assert next_progress.object_states == (("robot", "r0", "bad"), ("robot", "r1", "safe"))

    next_progress = pruning_model.advance(progress, treat_r1)
    assert pruning_model.is_prunable(next_progress)
    assert next_progress.object_states == (("robot", "r0", "safe"), ("robot", "r1", "bad"))


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
    assert next_progress.object_states == (("position", "DEPOT", "safe"), ("position", "p5", "bad"))

    next_progress = pruning_model.advance(progress, move_depot)
    assert pruning_model.is_prunable(next_progress)
    assert next_progress.object_states == (("position", "DEPOT", "bad"), ("position", "p5", "safe"))


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
