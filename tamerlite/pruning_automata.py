from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

from aalpy.automata.Dfa import Dfa
from aalpy.utils import load_automaton_from_file

from trace_conversion_utils import split_ground_action


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


class DfaPruningModel:
    def __init__(self, dfa: Dfa):
        self._dfa = dfa
        self._action_by_name: Dict[str, object] = {}
        self._state_by_id = {str(state.state_id): state for state in dfa.states}

    def bind_to_planner(
        self,
        action_by_name: Mapping[str, object],
        objects_by_type: Mapping[str, Sequence[str]],
    ) -> None:
        del objects_by_type
        self._action_by_name = dict(action_by_name)
        for state in self._dfa.states:
            new_transitions = {}
            for input_symbol, destination_state in state.transitions.items():
                planner_action = self._action_by_name.get(str(input_symbol))
                if planner_action is not None:
                    new_transitions[planner_action] = destination_state
            state.transitions = new_transitions

    @property
    def initial_state(self) -> SingleAutomatonProgress:
        return SingleAutomatonProgress(str(self._dfa.initial_state.state_id))

    def advance(self, progress: SingleAutomatonProgress, action: object) -> SingleAutomatonProgress | None:
        state = self._state_by_id[progress.state_id]
        next_state = state.transitions.get(action)
        if next_state is None:
            return None
        return SingleAutomatonProgress(str(next_state.state_id))

    def is_prunable(self, progress: SingleAutomatonProgress | None) -> bool:
        if progress is None:
            return False
        return self._state_by_id[progress.state_id].is_accepting

    def progress_key(self, progress: SingleAutomatonProgress | None):
        return None if progress is None else progress.state_id


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

        for action_name, action in self._action_by_name.items():
            parsed_name, typed_parameters = split_ground_action(
                action_name,
                self._action_parameter_types,
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
        if progress is None:
            return False

        for focus_type, spec in self._automata.items():
            valid_objects = set(self._objects_by_type.get(focus_type, ()))
            for progress_focus_type, object_name, state_id in progress.object_states:
                if progress_focus_type != focus_type or object_name not in valid_objects:
                    continue
                if self._state_by_type_and_id[focus_type][state_id].is_accepting:
                    return True
        return False

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
            rendered_parameters.append(
                self._render_parameter_token(
                    parameter_type=parameter_type,
                    parameter_name=parameter_name,
                    focus_type=focus_type,
                    focus_object=focus_object,
                )
            )

        if not rendered_parameters:
            return action_name
        return f"{action_name}({','.join(rendered_parameters)})"

    def _render_parameter_token(
        self,
        *,
        parameter_type: str,
        parameter_name: str,
        focus_type: str,
        focus_object: str,
    ) -> str:
        if parameter_name == focus_object:
            placeholder = self._automata[focus_type].placeholder
            return f"*{placeholder}*"
        if self._abstract_other_objects:
            return self._placeholders_by_type[parameter_type]
        return parameter_name
