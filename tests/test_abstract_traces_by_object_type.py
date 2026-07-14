import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src" / "trace_preprocessing"))

from abstract_traces_by_object_type import abstract_trace_for_profile


def test_parameterless_actions_are_kept_in_multi_abstraction():
    trace = ("wait", "move_r0_l0_l1")

    abstracted = abstract_trace_for_profile(
        trace,
        profile_types=("robot",),
        focus_tuple=("r0",),
        slot_placeholders=("r",),
        placeholders_by_type={"robot": "r", "location": "l"},
        action_parameter_types={"wait": [], "move": ["robot", "location", "location"]},
        object_type_by_name={"r0": "robot", "l0": "location", "l1": "location"},
        drop_wildcards=True,
        abstract_other_objects=True,
    )

    assert abstracted == ["wait", "move(*r*,l,l)"]


def test_integer_only_actions_are_kept_as_global_tokens_without_integer_monitor():
    trace = ("prepare_unload_0", "move_r0_l0_l1")

    abstracted = abstract_trace_for_profile(
        trace,
        profile_types=("robot",),
        focus_tuple=("r0",),
        slot_placeholders=("r",),
        placeholders_by_type={"robot": "r", "location": "l", "integer": "i"},
        action_parameter_types={
            "prepare_unload": ["integer"],
            "move": ["robot", "location", "location"],
        },
        object_type_by_name={"r0": "robot", "l0": "location", "l1": "location"},
        drop_wildcards=True,
        abstract_other_objects=True,
    )

    assert abstracted == ["prepare_unload(INT)", "move(*r*,l,l)"]
