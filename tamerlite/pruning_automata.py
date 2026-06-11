from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence
import sys

from aalpy.automata.Dfa import Dfa
from aalpy.utils import load_automaton_from_file

try:
    from trace_conversion_utils import _is_numeric_parameter_type, split_ground_action
except ModuleNotFoundError:
    REPO_ROOT = Path(__file__).resolve().parents[2]
    for candidate in (
        REPO_ROOT / "src",
        REPO_ROOT / "src" / "trace_preprocessing",
    ):
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
    from trace_conversion_utils import _is_numeric_parameter_type, split_ground_action


@dataclass(frozen=True)
class SingleAutomatonProgress:
    state_id: str


@dataclass(frozen=True)
class MultiAutomatonProgress:
    object_states: tuple[tuple[str, str, str], ...]


@dataclass(frozen=True)
class MultiAutomatonSpec:
    focus_type: str
    placeholder: str
    dfa: Dfa


@dataclass(frozen=True)
class SingleDfaVariantSpec:
    name: str
    split_ground_actions: bool
    abstract_ground_actions: bool

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, object] | None) -> "SingleDfaVariantSpec":
        payload = metadata or {}
        split_ground_actions = bool(payload.get("split_ground_actions", False))
        abstract_ground_actions = bool(payload.get("abstract_ground_actions", False))
        if abstract_ground_actions:
            return cls(
                name="single-dfa-abstract",
                split_ground_actions=False,
                abstract_ground_actions=True,
            )
        if split_ground_actions:
            return cls(
                name="single-dfa-split",
                split_ground_actions=True,
                abstract_ground_actions=False,
            )
        return cls(
            name="single-dfa-plain",
            split_ground_actions=False,
            abstract_ground_actions=False,
        )


def _normalize_transition_token(token: object) -> str:
    token_str = str(token)
    if len(token_str) >= 2 and token_str[0] == token_str[-1] and token_str[0] in {"'", '"'}:
        return token_str[1:-1]
    return token_str


class DfaPruningModel:
    def __init__(self, dfa: Dfa, *, variant: SingleDfaVariantSpec | None = None):
        self._dfa = dfa
        self._variant = variant or SingleDfaVariantSpec(
            name="single-dfa-plain",
            split_ground_actions=False,
            abstract_ground_actions=False,
        )
        self._action_by_name: Dict[str, object] = {}
        self._tokens_by_action: Dict[object, tuple[str, ...]] = {}
        self._state_by_id = {str(state.state_id): state for state in dfa.states}
        self._known_action_labels = self._collect_action_labels()

    @classmethod
    def from_automaton_files(
        cls,
        dot_path: str | Path,
        *,
        dataset_path: str | Path | None = None,
    ) -> "DfaPruningModel":
        dot_file = Path(dot_path)
        payload = None
        if dataset_path is None:
            candidate = dot_file.with_suffix(".dataset.json")
            if candidate.exists():
                dataset_path = candidate
        if dataset_path is not None:
            payload = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
        variant = SingleDfaVariantSpec.from_metadata((payload or {}).get("metadata"))
        dfa = load_automaton_from_file(str(dot_file), automaton_type="dfa")
        return cls(dfa, variant=variant)

    @property
    def variant_name(self) -> str:
        return self._variant.name

    def clone(self) -> "DfaPruningModel":
        return DfaPruningModel(copy.deepcopy(self._dfa), variant=self._variant)

    def bind_to_planner(
        self,
        action_by_name: Mapping[str, object],
        objects_by_type: Mapping[str, Sequence[str]],
    ) -> None:
        del objects_by_type
        self._action_by_name = dict(action_by_name)
        self._tokens_by_action = {}
        for action_name, action in self._action_by_name.items():
            tokens = self._planner_tokens_for_action_name(action_name)
            if tokens:
                self._tokens_by_action[action] = tokens

        for state in self._dfa.states:
            new_transitions = {}
            for input_symbol, destination_state in state.transitions.items():
                new_transitions[_normalize_transition_token(input_symbol)] = destination_state
            state.transitions = new_transitions

    @property
    def initial_state(self) -> SingleAutomatonProgress:
        return SingleAutomatonProgress(str(self._dfa.initial_state.state_id))

    def advance(self, progress: SingleAutomatonProgress, action: object) -> SingleAutomatonProgress | None:
        state = self._state_by_id[progress.state_id]
        tokens = self._tokens_by_action.get(action)
        if not tokens:
            return None

        current_state = state
        for token in tokens:
            next_state = current_state.transitions.get(token)
            if next_state is None:
                return None
            current_state = next_state
        return SingleAutomatonProgress(str(current_state.state_id))

    def is_prunable(self, progress: SingleAutomatonProgress | None) -> bool:
        if progress is None:
            return False
        return self._state_by_id[progress.state_id].is_accepting

    def pruning_labels(self, progress: SingleAutomatonProgress | None) -> tuple[str, ...]:
        if not self.is_prunable(progress):
            return ()
        return (self._variant.name,)

    def progress_key(self, progress: SingleAutomatonProgress | None):
        return None if progress is None else progress.state_id

    def _collect_action_labels(self) -> tuple[str, ...]:
        labels = set()
        for state in self._dfa.states:
            for token in state.transitions:
                token_str = _normalize_transition_token(token)
                if not token_str:
                    continue
                labels.add(token_str)
        return tuple(sorted(labels, key=len, reverse=True))

    def _planner_tokens_for_action_name(self, action_name: str) -> tuple[str, ...]:
        if self._variant.name == "single-dfa-plain":
            return (action_name,)

        label = self._infer_action_label(action_name)
        if label is None:
            return ()

        if self._variant.name == "single-dfa-abstract":
            return (label,)

        suffix = action_name[len(label):]
        if suffix.startswith("_"):
            suffix = suffix[1:]
        parameters = tuple(part for part in suffix.split("_") if part)
        return (label, *parameters)

    def _infer_action_label(self, action_name: str) -> str | None:
        for candidate in self._known_action_labels:
            if action_name == candidate:
                return candidate
            if action_name.startswith(f"{candidate}_"):
                return candidate
        return None


class MultiAutomatonPruningModel:
    def __init__(
        self,
        action_parameter_types: Mapping[str, Sequence[str]],
        placeholders_by_type: Mapping[str, str],
        automata: Mapping[str, MultiAutomatonSpec],
        *,
        drop_wildcards: bool,
        abstract_other_objects: bool,
    ):
        self._action_parameter_types = {
            action_name: list(parameter_types)
            for action_name, parameter_types in action_parameter_types.items()
        }
        self._placeholders_by_type = dict(placeholders_by_type)
        self._automata = dict(automata)
        self._drop_wildcards = drop_wildcards
        self._abstract_other_objects = abstract_other_objects
        self._action_by_name: Dict[str, object] = {}
        self._planner_action_details: Dict[object, tuple[str, list[tuple[str, str]]]] = {}
        self._state_by_type_and_id = {
            focus_type: {str(state.state_id): state for state in spec.dfa.states}
            for focus_type, spec in self._automata.items()
        }
        self._objects_by_type: Dict[str, tuple[str, ...]] = {}

    @classmethod
    def from_summary_file(cls, summary_path: str | Path) -> "MultiAutomatonPruningModel":
        summary_file = Path(summary_path)
        payload = json.loads(summary_file.read_text(encoding="utf-8"))
        signature = payload.get("signature") or {}
        action_parameter_types = signature.get("action_parameter_types") or {}
        placeholders_by_type = signature.get("placeholders_by_type") or {}
        drop_wildcards = bool(signature.get("drop_wildcards", True))
        abstract_other_objects = bool(signature.get("abstract_other_objects", False))

        automata = {}
        for focus_type, entry in (payload.get("automata") or {}).items():
            raw_dot_path = Path(entry["dot_path"])
            if raw_dot_path.is_absolute():
                dot_path = raw_dot_path
            else:
                candidates = [summary_file.parent / raw_dot_path]
                candidates.extend(parent / raw_dot_path for parent in summary_file.parents)
                candidates.append(summary_file.parent / focus_type / raw_dot_path.name)
                dot_path = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
            dfa = load_automaton_from_file(str(dot_path), automaton_type="dfa")
            entry_signature = entry.get("signature") or {}
            automata[focus_type] = MultiAutomatonSpec(
                focus_type=focus_type,
                placeholder=entry_signature.get("placeholder", entry["placeholder"]),
                dfa=dfa,
            )

        return cls(
            action_parameter_types=action_parameter_types,
            placeholders_by_type=placeholders_by_type,
            automata=automata,
            drop_wildcards=drop_wildcards,
            abstract_other_objects=abstract_other_objects,
        )

    def bind_to_planner(
        self,
        action_by_name: Mapping[str, object],
        objects_by_type: Mapping[str, Sequence[str]],
    ) -> None:
        self._action_by_name = dict(action_by_name)
        self._objects_by_type = {
            str(object_type).lower(): tuple(sorted(object_names))
            for object_type, object_names in objects_by_type.items()
        }
        object_type_by_name = {
            object_name: object_type
            for object_type, object_names in self._objects_by_type.items()
            for object_name in object_names
        }

        for action_name, action in self._action_by_name.items():
            parsed_name, typed_parameters = split_ground_action(
                action_name,
                self._action_parameter_types,
                object_type_by_name=object_type_by_name,
            )
            self._planner_action_details[action] = (parsed_name, typed_parameters)

        for spec in self._automata.values():
            for state in spec.dfa.states:
                new_transitions = {}
                for token, destination_state in state.transitions.items():
                    new_transitions[str(token)] = destination_state
                state.transitions = new_transitions

    @property
    def initial_state(self) -> MultiAutomatonProgress:
        object_states = []
        for focus_type, spec in sorted(self._automata.items()):
            for object_name in self._objects_by_type.get(focus_type, ()):
                object_states.append((focus_type, object_name, str(spec.dfa.initial_state.state_id)))
        return MultiAutomatonProgress(tuple(object_states))

    def advance(self, progress: MultiAutomatonProgress, action: object) -> MultiAutomatonProgress:
        parsed = self._planner_action_details.get(action)
        if parsed is None:
            return progress

        action_name, typed_parameters = parsed
        parameter_names = [parameter_name for _, parameter_name in typed_parameters]
        updated_states = {
            (focus_type, object_name): state_id
            for focus_type, object_name, state_id in progress.object_states
        }

        for focus_type, spec in self._automata.items():
            for progress_focus_type, object_name, state_id in progress.object_states:
                if progress_focus_type != focus_type or object_name not in parameter_names:
                    continue
                if object_name not in self._objects_by_type.get(focus_type, ()):
                    continue

                token = self._render_token(action_name, typed_parameters, focus_type, object_name)
                if token is None:
                    continue

                current_state = self._state_by_type_and_id[focus_type][state_id]
                next_state = current_state.transitions.get(token)
                if next_state is None:
                    continue
                updated_states[(focus_type, object_name)] = str(next_state.state_id)

        return MultiAutomatonProgress(
            tuple(
                sorted(
                    (focus_type, object_name, state_id)
                    for (focus_type, object_name), state_id in updated_states.items()
                )
            )
        )

    def is_prunable(self, progress: MultiAutomatonProgress | None) -> bool:
        return bool(self.pruning_labels(progress))

    def pruning_labels(self, progress: MultiAutomatonProgress | None) -> tuple[str, ...]:
        if progress is None:
            return ()

        accepting_focus_types = set()
        for focus_type, spec in self._automata.items():
            valid_objects = set(self._objects_by_type.get(focus_type, ()))
            for progress_focus_type, object_name, state_id in progress.object_states:
                if progress_focus_type != focus_type or object_name not in valid_objects:
                    continue
                if self._state_by_type_and_id[focus_type][state_id].is_accepting:
                    accepting_focus_types.add(focus_type)
                    break
        return tuple(sorted(accepting_focus_types))

    def progress_key(self, progress: MultiAutomatonProgress | None):
        return None if progress is None else progress.object_states

    def _render_token(
        self,
        action_name: str,
        typed_parameters: Iterable[tuple[str, str]],
        focus_type: str,
        focus_object: str,
    ) -> str | None:
        parameter_names = [parameter_name for _, parameter_name in typed_parameters]
        if focus_object not in parameter_names:
            return None if self._drop_wildcards else "*"

        rendered_parameters = []
        for parameter_type, parameter_name in typed_parameters:
            if _is_numeric_parameter_type(parameter_type):
                rendered_parameters.append("INT")
                continue

            normalized_type = str(parameter_type).lower()
            placeholder = self._placeholders_by_type.get(normalized_type, normalized_type[:1])
            if parameter_name == focus_object:
                rendered_parameters.append(f"*{placeholder}*")
            elif self._abstract_other_objects:
                rendered_parameters.append(placeholder)
            else:
                rendered_parameters.append(parameter_name)
        return f"{action_name}({','.join(rendered_parameters)})"
