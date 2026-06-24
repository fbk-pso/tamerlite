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
    from trace_conversion_utils import _is_numeric_parameter_type, canonicalize_identifier, split_ground_action
    from goal_trace_utils import add_state_trace_markers, goal_tokens_for_profile, init_tokens_for_profile
except ModuleNotFoundError:
    REPO_ROOT = Path(__file__).resolve().parents[2]
    for candidate in (
        REPO_ROOT / "src",
        REPO_ROOT / "src" / "trace_preprocessing",
    ):
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
    from trace_conversion_utils import _is_numeric_parameter_type, canonicalize_identifier, split_ground_action
    from goal_trace_utils import add_state_trace_markers, goal_tokens_for_profile, init_tokens_for_profile


@dataclass(frozen=True)
class SingleAutomatonProgress:
    state_id: str


@dataclass(frozen=True)
class MultiAutomatonProgress:
    object_states: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class MultiAutomatonSpec:
    focus_label: str
    profile_types: tuple[str, ...]
    placeholder: str
    slot_placeholders: tuple[str, ...]
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

    @classmethod
    def from_variant_name(cls, variant_name: str | None) -> "SingleDfaVariantSpec":
        normalized = (variant_name or "").strip()
        if normalized in {"single_dfa_abstract", "single_abstract"}:
            return cls(
                name="single-dfa-abstract",
                split_ground_actions=False,
                abstract_ground_actions=True,
            )
        if normalized in {"single_dfa_split", "single_split"}:
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
        token_str = token_str[1:-1]
    return canonicalize_identifier(token_str)


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
        metadata_path: str | Path | None = None,
        dataset_path: str | Path | None = None,
    ) -> "DfaPruningModel":
        dot_file = Path(dot_path)
        payload = None
        if metadata_path is None:
            metadata_candidate = dot_file.with_suffix(".metadata.json")
            if metadata_candidate.exists():
                metadata_path = metadata_candidate
        if metadata_path is not None:
            payload = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
        if payload is None and dataset_path is None:
            candidate = dot_file.with_suffix(".dataset.json")
            if candidate.exists():
                dataset_path = candidate
        if payload is None and dataset_path is not None:
            payload = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
        if payload is not None:
            variant = SingleDfaVariantSpec.from_metadata((payload or {}).get("metadata") or payload)
        else:
            variant = SingleDfaVariantSpec.from_variant_name(dot_file.parent.name)
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
        initial_values: Mapping[object, object] | None = None,
        goals: Sequence[object] | None = None,
    ) -> None:
        del initial_values
        del goals
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
        include_goal_prefixes: bool = False,
        include_init_prefixes: bool = False,
    ):
        self._action_parameter_types = {
            action_name: list(parameter_types)
            for action_name, parameter_types in action_parameter_types.items()
        }
        self._placeholders_by_type = dict(placeholders_by_type)
        self._automata = {
            focus_label: self._coerce_spec(focus_label, spec)
            for focus_label, spec in automata.items()
        }
        self._drop_wildcards = drop_wildcards
        self._abstract_other_objects = abstract_other_objects
        self._include_goal_prefixes = include_goal_prefixes
        self._include_init_prefixes = include_init_prefixes
        self._action_by_name: Dict[str, object] = {}
        self._planner_action_details: Dict[object, tuple[str, list[tuple[str, str]]]] = {}
        self._state_by_type_and_id = {
            focus_label: {str(state.state_id): state for state in spec.dfa.states}
            for focus_label, spec in self._automata.items()
        }
        self._objects_by_type: Dict[str, tuple[str, ...]] = {}
        self._object_type_by_name: Dict[str, str] = {}
        self._problem_initial_values: Dict[object, object] = {}
        self._problem_goals: tuple[object, ...] = tuple()
        self._prefixed_initial_states: Dict[tuple[str, ...], str] = {}

    def _coerce_spec(self, focus_label: str, spec: MultiAutomatonSpec | object) -> MultiAutomatonSpec:
        if isinstance(spec, MultiAutomatonSpec):
            return spec

        profile_types = getattr(spec, "profile_types", None)
        if profile_types is None:
            profile_types = (focus_label,)
        else:
            profile_types = tuple(str(current) for current in profile_types)

        slot_placeholders = getattr(spec, "slot_placeholders", None)
        if slot_placeholders is None:
            placeholder = str(getattr(spec, "placeholder", focus_label))
            slot_placeholders = (placeholder,) if len(profile_types) == 1 else tuple(
                f"{placeholder}{index + 1}" for index in range(len(profile_types))
            )
        else:
            slot_placeholders = tuple(str(current) for current in slot_placeholders)

        return MultiAutomatonSpec(
            focus_label=focus_label,
            profile_types=profile_types,
            placeholder=str(getattr(spec, "placeholder", slot_placeholders[0])),
            slot_placeholders=slot_placeholders,
            dfa=getattr(spec, "dfa"),
        )

    @classmethod
    def from_summary_file(cls, summary_path: str | Path) -> "MultiAutomatonPruningModel":
        summary_file = Path(summary_path)
        payload = json.loads(summary_file.read_text(encoding="utf-8"))
        signature = payload.get("signature") or {}
        action_parameter_types = signature.get("action_parameter_types") or {}
        placeholders_by_type = signature.get("placeholders_by_type") or {}
        drop_wildcards = bool(signature.get("drop_wildcards", True))
        abstract_other_objects = bool(signature.get("abstract_other_objects", False))
        include_goal_prefixes = bool(payload.get("goal_prefixes_included", False))
        include_init_prefixes = bool(payload.get("init_prefixes_included", False))

        automata = {}
        for focus_label, entry in (payload.get("automata") or {}).items():
            raw_dot_path = Path(entry["dot_path"])
            if raw_dot_path.is_absolute():
                dot_path = raw_dot_path
            else:
                candidates = [summary_file.parent / raw_dot_path]
                candidates.extend(parent / raw_dot_path for parent in summary_file.parents)
                candidates.append(summary_file.parent / focus_label / raw_dot_path.name)
                dot_path = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
            dfa = load_automaton_from_file(str(dot_path), automaton_type="dfa")
            entry_signature = entry.get("signature") or {}
            focus_profile = entry_signature.get("focus_profile") or entry.get("profile_types") or [focus_label]
            slot_placeholders = (
                entry_signature.get("slot_placeholders")
                or entry.get("slot_placeholders")
                or [entry_signature.get("placeholder", entry["placeholder"])]
            )
            automata[focus_label] = MultiAutomatonSpec(
                focus_label=focus_label,
                profile_types=tuple(str(current) for current in focus_profile),
                placeholder=entry_signature.get("placeholder", entry["placeholder"]),
                slot_placeholders=tuple(str(current) for current in slot_placeholders),
                dfa=dfa,
            )

        return cls(
            action_parameter_types=action_parameter_types,
            placeholders_by_type=placeholders_by_type,
            automata=automata,
            drop_wildcards=drop_wildcards,
            abstract_other_objects=abstract_other_objects,
            include_goal_prefixes=include_goal_prefixes,
            include_init_prefixes=include_init_prefixes,
        )

    @classmethod
    def from_automata_root(cls, automata_root: str | Path) -> "MultiAutomatonPruningModel":
        root = Path(automata_root)
        if not root.is_dir():
            raise FileNotFoundError(f"Multi automata root does not exist: {root}")

        metadata_entries: list[tuple[str, Path, dict[str, object]]] = []
        for metadata_path in sorted(root.rglob("automaton.metadata.json")):
            dot_path = metadata_path.with_name("automaton.dot")
            if not dot_path.exists():
                continue
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata = payload.get("metadata") or {}
            focus_label = str(metadata.get("focus_type") or metadata_path.parent.name)
            metadata_entries.append((focus_label, dot_path, metadata))

        if not metadata_entries:
            raise FileNotFoundError(f"No multi automata with metadata found under: {root}")

        first_signature = (metadata_entries[0][2].get("signature") or {})
        action_parameter_types = first_signature.get("action_parameter_types") or {}
        placeholders_by_type = first_signature.get("placeholders_by_type") or {}
        drop_wildcards = bool(first_signature.get("drop_wildcards", metadata_entries[0][2].get("drop_wildcards", True)))
        abstract_other_objects = bool(
            first_signature.get("abstract_other_objects", metadata_entries[0][2].get("abstract_other_objects", False))
        )
        include_goal_prefixes = bool(
            first_signature.get("goal_prefixes_included", metadata_entries[0][2].get("goal_prefixes_included", False))
        )
        include_init_prefixes = bool(
            first_signature.get("init_prefixes_included", metadata_entries[0][2].get("init_prefixes_included", False))
        )

        automata = {}
        for focus_label, dot_path, metadata in metadata_entries:
            signature = metadata.get("signature") or {}
            focus_profile = signature.get("focus_profile") or metadata.get("focus_profile") or [focus_label]
            slot_placeholders = (
                signature.get("slot_placeholders")
                or metadata.get("slot_placeholders")
                or [signature.get("placeholder", metadata.get("placeholder", focus_label))]
            )
            dfa = load_automaton_from_file(str(dot_path), automaton_type="dfa")
            automata[focus_label] = MultiAutomatonSpec(
                focus_label=focus_label,
                profile_types=tuple(str(current) for current in focus_profile),
                placeholder=str(signature.get("placeholder", metadata.get("placeholder", focus_label))),
                slot_placeholders=tuple(str(current) for current in slot_placeholders),
                dfa=dfa,
            )

        return cls(
            action_parameter_types=action_parameter_types,
            placeholders_by_type=placeholders_by_type,
            automata=automata,
            drop_wildcards=drop_wildcards,
            abstract_other_objects=abstract_other_objects,
            include_goal_prefixes=include_goal_prefixes,
            include_init_prefixes=include_init_prefixes,
        )

    def bind_to_planner(
        self,
        action_by_name: Mapping[str, object],
        objects_by_type: Mapping[str, Sequence[str]],
        initial_values: Mapping[object, object] | None = None,
        goals: Sequence[object] | None = None,
    ) -> None:
        self._action_by_name = dict(action_by_name)
        self._objects_by_type = {
            str(object_type).lower(): tuple(sorted(object_names))
            for object_type, object_names in objects_by_type.items()
        }
        self._object_type_by_name = {
            object_name: object_type
            for object_type, object_names in self._objects_by_type.items()
            for object_name in object_names
        }
        self._problem_initial_values = dict(initial_values or {})
        self._problem_goals = tuple(goals or ())
        self._prefixed_initial_states = {}

        for action_name, action in self._action_by_name.items():
            parsed_name, typed_parameters = split_ground_action(
                action_name,
                self._action_parameter_types,
                object_type_by_name=self._object_type_by_name,
            )
            self._planner_action_details[action] = (parsed_name, typed_parameters)

        for spec in self._automata.values():
            for state in spec.dfa.states:
                new_transitions = {}
                for token, destination_state in state.transitions.items():
                    new_transitions[_normalize_transition_token(token)] = destination_state
                state.transitions = new_transitions

    @property
    def initial_state(self) -> MultiAutomatonProgress:
        return MultiAutomatonProgress(tuple())

    def advance(self, progress: MultiAutomatonProgress, action: object) -> MultiAutomatonProgress:
        parsed = self._planner_action_details.get(action)
        if parsed is None:
            return progress

        action_name, typed_parameters = parsed
        updated_states = {
            tuple(progress_entry[:-1]): progress_entry[-1]
            for progress_entry in progress.object_states
        }

        for focus_label, spec in self._automata.items():
            candidate_focus_tuples = self._candidate_focus_tuples(typed_parameters, spec)
            if not candidate_focus_tuples:
                continue
            for focus_tuple in candidate_focus_tuples:
                entry_key = (focus_label, *focus_tuple)
                state_id = updated_states.get(entry_key)
                if state_id is None:
                    state_id = self._prefixed_initial_state_id(focus_label, spec, focus_tuple)
                token = self._render_token(action_name, typed_parameters, spec, focus_tuple)
                if token is None:
                    continue

                current_state = self._state_by_type_and_id[focus_label][state_id]
                next_state = current_state.transitions.get(token)
                if next_state is None:
                    legacy_token = self._legacy_same_type_token(token, spec)
                    if legacy_token is not None:
                        next_state = current_state.transitions.get(legacy_token)
                if next_state is None:
                    continue
                updated_states[entry_key] = str(next_state.state_id)

        return MultiAutomatonProgress(
            tuple(
                sorted(
                    (*entry_key, state_id)
                    for entry_key, state_id in updated_states.items()
                )
            )
        )

    def is_prunable(self, progress: MultiAutomatonProgress | None) -> bool:
        return bool(self.pruning_labels(progress))

    def pruning_labels(self, progress: MultiAutomatonProgress | None) -> tuple[str, ...]:
        if progress is None:
            return ()

        accepting_focus_types = set()
        for focus_label, _ in self._automata.items():
            for progress_entry in progress.object_states:
                if not progress_entry or progress_entry[0] != focus_label:
                    continue
                state_id = progress_entry[-1]
                if self._state_by_type_and_id[focus_label][state_id].is_accepting:
                    accepting_focus_types.add(focus_label)
                    break
        return tuple(sorted(accepting_focus_types))

    def progress_key(self, progress: MultiAutomatonProgress | None):
        return None if progress is None else progress.object_states

    def _render_token(
        self,
        action_name: str,
        typed_parameters: Iterable[tuple[str, str]],
        spec: MultiAutomatonSpec,
        focus_tuple: tuple[str, ...],
    ) -> str | None:
        focus_parameter_to_slot = {
            focus_object: slot_placeholder
            for focus_object, slot_placeholder in zip(focus_tuple, spec.slot_placeholders)
        }
        focus_types = set(spec.profile_types)
        parameter_names = [parameter_name for _, parameter_name in typed_parameters]
        if not any(focus_object in parameter_names for focus_object in focus_tuple):
            return None if self._drop_wildcards else "*"

        rendered_parameters = []
        for parameter_type, parameter_name in typed_parameters:
            if _is_numeric_parameter_type(parameter_type):
                rendered_parameters.append("INT" if self._abstract_other_objects else parameter_name)
                continue

            normalized_type = str(parameter_type).lower()
            placeholder = self._placeholders_by_type.get(normalized_type, normalized_type[:1])
            if parameter_name in focus_parameter_to_slot:
                rendered_parameters.append(f"*{focus_parameter_to_slot[parameter_name]}*")
            elif self._abstract_other_objects and normalized_type in focus_types:
                rendered_parameters.append(f"~{placeholder}")
            elif self._abstract_other_objects:
                rendered_parameters.append(placeholder)
            else:
                rendered_parameters.append(parameter_name)
        return canonicalize_identifier(f"{action_name}({','.join(rendered_parameters)})")

    def _legacy_same_type_token(
        self,
        token: str | None,
        spec: MultiAutomatonSpec,
    ) -> str | None:
        if token is None or not self._abstract_other_objects:
            return None

        legacy_token = token
        changed = False
        for focus_type in set(spec.profile_types):
            normalized_type = str(focus_type).lower()
            placeholder = self._placeholders_by_type.get(normalized_type, normalized_type[:1])
            legacy_marker = f"~{placeholder}"
            if legacy_marker in legacy_token:
                legacy_token = legacy_token.replace(legacy_marker, placeholder)
                changed = True
        if not changed:
            return None
        return legacy_token

    def _prefixed_initial_state_id(
        self,
        focus_label: str,
        spec: MultiAutomatonSpec,
        focus_tuple: tuple[str, ...],
    ) -> str:
        entry_key = (focus_label, *focus_tuple)
        cached_state_id = self._prefixed_initial_states.get(entry_key)
        if cached_state_id is not None:
            return cached_state_id

        init_tokens = []
        if self._include_init_prefixes:
            init_tokens = init_tokens_for_profile(
                self._problem_initial_values,
                profile_types=spec.profile_types,
                focus_tuple=focus_tuple,
                slot_placeholders=spec.slot_placeholders,
                placeholders_by_type=self._placeholders_by_type,
                object_type_by_name=self._object_type_by_name,
                abstract_other_objects=self._abstract_other_objects,
            )
        goal_tokens = []
        if self._include_goal_prefixes:
            goal_tokens = goal_tokens_for_profile(
                self._problem_goals,
                profile_types=spec.profile_types,
                focus_tuple=focus_tuple,
                slot_placeholders=spec.slot_placeholders,
                placeholders_by_type=self._placeholders_by_type,
                object_type_by_name=self._object_type_by_name,
                abstract_other_objects=self._abstract_other_objects,
            )

        current_state = spec.dfa.initial_state
        for token in add_state_trace_markers(init_tokens, goal_tokens, ()):
            normalized_token = _normalize_transition_token(token)
            next_state = current_state.transitions.get(normalized_token)
            if next_state is None:
                legacy_token = self._legacy_same_type_token(normalized_token, spec)
                if legacy_token is not None:
                    next_state = current_state.transitions.get(legacy_token)
            if next_state is None:
                break
            current_state = next_state

        state_id = str(current_state.state_id)
        self._prefixed_initial_states[entry_key] = state_id
        return state_id

    def _candidate_focus_tuples(
        self,
        typed_parameters: Sequence[tuple[str, str]],
        spec: MultiAutomatonSpec,
    ) -> tuple[tuple[str, ...], ...]:
        action_object_names = {
            parameter_name
            for parameter_type, parameter_name in typed_parameters
            if not _is_numeric_parameter_type(parameter_type)
        }
        if not action_object_names:
            return ()

        domains_by_slot: list[tuple[str, ...]] = []
        for profile_type in spec.profile_types:
            slot_candidates = self._objects_by_type.get(str(profile_type).lower(), ())
            if not slot_candidates:
                return ()
            domains_by_slot.append(slot_candidates)

        tuples: list[tuple[str, ...]] = []

        def backtrack(slot_index: int, current: list[str]) -> None:
            if slot_index == len(domains_by_slot):
                if not action_object_names.intersection(current):
                    return
                tuples.append(tuple(current))
                return
            for candidate_name in domains_by_slot[slot_index]:
                if candidate_name in current:
                    continue
                current.append(candidate_name)
                backtrack(slot_index + 1, current)
                current.pop()

        backtrack(0, [])
        return tuple(tuples)
