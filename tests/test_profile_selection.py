from profile_selection import (
    build_selection_context,
    parse_explicit_profiles,
    select_action_signature_profiles,
    select_repeated_type_cores_with_unary_types,
    select_profiles,
    select_unary_profiles,
)


def test_unary_selector_returns_one_profile_per_type():
    context = build_selection_context(
        object_type_labels=["robot", "pallet", "position"],
        action_parameter_types={},
        object_type_by_name={},
    )

    assert select_unary_profiles(context) == [
        ("robot",),
        ("pallet",),
        ("position",),
    ]


def test_action_signature_selector_uses_non_numeric_action_signatures():
    context = build_selection_context(
        object_type_labels=["robot", "pallet", "position"],
        action_parameter_types={
            "move": ["robot", "pallet", "position"],
            "initialize": ["position", "integer"],
            "repeat_move": ["robot", "pallet", "position"],
        },
        object_type_by_name={},
    )

    assert select_action_signature_profiles(context) == [
        ("robot", "pallet", "position"),
        ("position",),
    ]


def test_explicit_profiles_and_selectors_are_merged_without_duplicates():
    profiles = select_profiles(
        object_type_labels=["robot", "pallet", "position"],
        action_parameter_types={"move": ["robot", "pallet", "position"]},
        object_type_by_name={},
        selector_names=["unary_types", "action_signatures"],
        explicit_profiles=["position,position", "robot,pallet,position"],
    )

    assert profiles == [
        ("robot",),
        ("pallet",),
        ("position",),
        ("robot", "pallet", "position"),
        ("position", "position"),
    ]


def test_repeated_type_cores_with_unary_types_include_all_unaries_and_repeated_cores():
    context = build_selection_context(
        object_type_labels=["robot", "pallet", "position", "lander"],
        action_parameter_types={
            "move": ["robot", "position", "position"],
            "communicate": ["robot", "lander", "position", "position", "position"],
            "transfer": ["robot", "pallet", "position"],
        },
        object_type_by_name={},
    )

    assert select_repeated_type_cores_with_unary_types(context) == [
        ("robot",),
        ("pallet",),
        ("position",),
        ("lander",),
        ("position", "position"),
        ("position", "position", "position"),
    ]


def test_parse_explicit_profiles_rejects_unknown_types():
    try:
        parse_explicit_profiles(["robot,unknown"], ["robot", "position"])
    except ValueError as exc:
        assert "Unknown object type" in str(exc)
        return

    raise AssertionError("Expected ValueError for unknown explicit profile type.")
