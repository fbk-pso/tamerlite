import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src" / "trace_preprocessing"))

from trace_conversion_utils import read_domain_symbols, split_ground_action, split_trace


def test_read_domain_symbols_accepts_hyphenated_action_names(tmp_path: Path):
    domain_path = tmp_path / "domain.anml"
    domain_path.write_text(
        """
type item;
type arm;
type bot;
action to-tray(item i, arm a, bot b) {
   [ start ] true;
};
action from-tray(item i, arm a, bot b) {
   [ start ] true;
};
instance item item3;
instance arm right2;
instance bot bot2;
""".strip(),
        encoding="utf-8",
    )

    _, _, object_type_by_name, action_parameter_types = read_domain_symbols(domain_path)

    assert action_parameter_types["to-tray"] == ["item", "arm", "bot"]
    assert action_parameter_types["from-tray"] == ["item", "arm", "bot"]
    assert object_type_by_name["right2"] == "arm"

    assert split_ground_action(
        "to-tray_item3_right2_bot2",
        action_parameter_types,
        object_type_by_name,
    ) == ("to-tray", [("item", "item3"), ("arm", "right2"), ("bot", "bot2")])


def test_split_trace_accepts_pipe_and_tuple_formats():
    assert split_trace("a|b|c") == ("a", "b", "c")
    assert split_trace("(a, b, c)") == ("a", "b", "c")
    assert split_trace("(move(x,y), to-tray_item3_right2_bot2)") == (
        "move(x,y)",
        "to-tray_item3_right2_bot2",
    )
