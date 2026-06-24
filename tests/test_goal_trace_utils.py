import sys
from pathlib import Path
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src" / "trace_preprocessing"))

from goal_trace_utils import (
    INIT_START_TOKEN,
    GOAL_START_TOKEN,
    TRACE_START_TOKEN,
    add_state_trace_markers,
    add_goal_trace_markers,
    init_tokens_for_profile,
    goal_tokens_for_profile,
    goal_tokens_for_single_abstract,
)


class FakeFluent:
    def __init__(self, name: str):
        self.name = name


class FakeGoal:
    def __init__(self, name: str, args: list[str], negated: bool = False, conjunction: bool = False):
        self._name = name
        self.args = args
        self._negated = negated
        self._conjunction = conjunction

    def is_not(self):
        return self._negated

    def is_fluent_exp(self):
        return not self._negated and not self._conjunction

    def is_and(self):
        return self._conjunction

    def fluent(self):
        return FakeFluent(self._name)

    def __str__(self):
        if self._conjunction:
            return f"({' and '.join(str(argument) for argument in self.args)})"
        if self._negated:
            return f"not({self.args[0]})"
        return f"{self._name}({','.join(self.args)})"


class FakeEqualsGoal:
    def __init__(self, lhs, rhs):
        self.args = [lhs, rhs]

    def is_not(self):
        return False

    def is_fluent_exp(self):
        return False

    def is_and(self):
        return False

    def is_equals(self):
        return True

    def __str__(self):
        return f"({self.args[0]} == {self.args[1]})"


class FakeObjectRef:
    def __init__(self, name: str):
        self._name = name

    def __str__(self):
        return self._name


class FakeValue:
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
        return FakeObjectRef(self._value)

    def __str__(self):
        return str(self._value)


def build_negated_goal(name: str, args: list[str]):
    atom = FakeGoal(name, args, negated=False)
    wrapper = FakeGoal(name, [], negated=True)
    wrapper.args = [atom]
    return wrapper


def build_conjunction_goal(*children):
    return FakeGoal("and", list(children), conjunction=True)


class FakeUnsupportedGoal:
    def __init__(self, label: str, args: list[object] | None = None):
        self._label = label
        self.args = args or []

    def is_not(self):
        return False

    def is_fluent_exp(self):
        return False

    def is_and(self):
        return False

    def __str__(self):
        return self._label


def test_single_abstract_goal_tokens_keep_only_predicates():
    goals = [
        FakeGoal("treated", ["b1", "p1"]),
        build_negated_goal("ready", ["b2", "p2"]),
    ]

    assert goal_tokens_for_single_abstract(goals) == ["not_ready", "treated"]


def test_single_abstract_goal_tokens_flatten_conjunctions():
    goals = [
        build_conjunction_goal(
            FakeGoal("treated", ["b1", "p1"]),
            build_negated_goal("ready", ["b2", "p2"]),
        )
    ]

    assert goal_tokens_for_single_abstract(goals) == ["not_ready", "treated"]


def test_single_abstract_goal_tokens_support_fluent_equalities():
    goals = [
        FakeEqualsGoal(FakeGoal("goal_progress", []), FakeValue(5, "int")),
    ]

    assert goal_tokens_for_single_abstract(goals) == ["goal_progress=5"]


def test_profile_goal_tokens_focus_and_abstract_other_objects():
    goals = [
        FakeGoal("treated", ["b1", "p1"]),
        FakeGoal("treated", ["b2", "p2"]),
        FakeGoal("ready", ["b3", "p1"]),
    ]

    tokens = goal_tokens_for_profile(
        goals,
        profile_types=("pallet",),
        focus_tuple=("b1",),
        slot_placeholders=("b",),
        placeholders_by_type={"pallet": "b", "position": "p"},
        object_type_by_name={"b1": "pallet", "b2": "pallet", "b3": "pallet", "p1": "position", "p2": "position"},
        abstract_other_objects=True,
    )

    assert tokens == ["treated(*b*,p)"]


def test_profile_goal_tokens_support_multi_type_profiles_and_other_same_type_slots():
    goals = [
        FakeGoal("connected", ["p1", "p2"]),
        FakeGoal("connected", ["p2", "p3"]),
        FakeGoal("treated", ["b1", "p2"]),
    ]

    tokens = goal_tokens_for_profile(
        goals,
        profile_types=("position", "position"),
        focus_tuple=("p1", "p2"),
        slot_placeholders=("p1", "p2"),
        placeholders_by_type={"position": "p", "pallet": "b"},
        object_type_by_name={"p1": "position", "p2": "position", "p3": "position", "b1": "pallet"},
        abstract_other_objects=True,
    )

    assert tokens == ["connected(*p1*,*p2*)", "connected(*p2*,~p)", "treated(b,*p2*)"]


def test_profile_goal_tokens_include_zero_arity_global_equalities():
    goals = [
        FakeEqualsGoal(FakeGoal("goal_progress", []), FakeValue(5, "int")),
    ]

    tokens = goal_tokens_for_profile(
        goals,
        profile_types=("drawer",),
        focus_tuple=("drawer_1",),
        slot_placeholders=("d",),
        placeholders_by_type={"drawer": "d"},
        object_type_by_name={"drawer_1": "drawer"},
        abstract_other_objects=True,
    )

    assert tokens == ["goal_progress=INT"]


def test_goal_token_extraction_rejects_unsupported_goal_forms():
    with pytest.raises(ValueError, match="Unsupported goal expression"):
        goal_tokens_for_single_abstract([FakeUnsupportedGoal("or(served(p1),served(p2))")])


def test_add_goal_trace_markers_wraps_goal_prefix_and_trace():
    assert add_goal_trace_markers(["treated", "ready"], ["move(*b*)"]) == (
        GOAL_START_TOKEN,
        "treated",
        "ready",
        TRACE_START_TOKEN,
        "move(*b*)",
    )


def test_add_goal_trace_markers_leaves_trace_unchanged_without_goals():
    assert add_goal_trace_markers([], ["move(*b*)"]) == ("move(*b*)",)


def test_add_state_trace_markers_wraps_init_goal_and_trace():
    assert add_state_trace_markers(["robot_at(*r*,l)"], ["treated(*b*)"], ["move(*r*)"]) == (
        INIT_START_TOKEN,
        "robot_at(*r*,l)",
        GOAL_START_TOKEN,
        "treated(*b*)",
        TRACE_START_TOKEN,
        "move(*r*)",
    )


def test_init_tokens_for_profile_abstracts_state_assignments():
    initial_values = {
        FakeGoal("robot_at", ["r1", "l0"]): FakeValue(True, "bool"),
        FakeGoal("robot_cnt", ["r1"]): FakeValue(2, "int"),
        FakeGoal("assigned_kit", ["r2"]): FakeValue("k1", "object"),
        FakeGoal("robot_at", ["r2", "l1"]): FakeValue(True, "bool"),
    }

    tokens = init_tokens_for_profile(
        initial_values,
        profile_types=("robot",),
        focus_tuple=("r1",),
        slot_placeholders=("r",),
        placeholders_by_type={"robot": "r", "location": "l", "kit": "k"},
        object_type_by_name={"r1": "robot", "r2": "robot", "l0": "location", "l1": "location", "k1": "kit"},
        abstract_other_objects=True,
    )

    assert tokens == ["robot_at(*r*,l)", "robot_cnt(*r*)=INT"]
