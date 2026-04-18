from __future__ import annotations

import json
import re
from typing import Any

from ..types import (
    ClarificationOption,
    UserPreferenceProfile,
)
from .errors import AgentHarnessError

JSON_FENCE_PATTERN = re.compile(r"```(?:json)?\s*([\s\S]*?)```", flags=re.IGNORECASE)
JSON_LINE_START_PATTERN = re.compile(r"(?m)^[ \t]*([\{\[])")


def normalize_choices(choice_payload: Any) -> list[ClarificationOption]:
    if not isinstance(choice_payload, list):
        raise AgentHarnessError("suggested_choices must be a list.")

    choices: list[ClarificationOption] = []
    for index, item in enumerate(choice_payload):
        if not isinstance(item, dict):
            raise AgentHarnessError("Each suggested choice must be an object.")
        label = str(item.get("label", "")).strip()
        if not label:
            raise AgentHarnessError("Each suggested choice requires a label.")
        option_id = str(item.get("id", "")).strip() or slugify(label)
        choices.append(
            ClarificationOption(
                id=option_id,
                label=label,
                description=str(item.get("description", "")).strip() or f"Option {index + 1}",
            )
        )
    return choices


def parse_json_payload(payload: str) -> dict[str, Any]:
    parsed = parse_json_value(payload)
    if not isinstance(parsed, dict):
        raise AgentHarnessError("Expected the agent payload to be a JSON object.")
    return parsed


def parse_json_value(payload: str) -> Any:
    stripped = payload.strip()
    decoder = json.JSONDecoder()
    last_error: json.JSONDecodeError | None = None

    for candidate in iter_json_candidates(stripped, decoder):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc

    if last_error is not None:
        raise last_error
    return json.loads(stripped)


def iter_json_candidates(payload: str, decoder: json.JSONDecoder) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()

    def add(candidate: str) -> None:
        normalized = candidate.strip()
        if not normalized or normalized in seen:
            return
        seen.add(normalized)
        candidates.append(normalized)

    add(payload)

    fenced_match = re.fullmatch(JSON_FENCE_PATTERN, payload)
    if fenced_match:
        add(fenced_match.group(1))

    for match in JSON_FENCE_PATTERN.finditer(payload):
        add(match.group(1))

    for match in JSON_LINE_START_PATTERN.finditer(payload):
        start = match.start(1)
        try:
            _, end = decoder.raw_decode(payload[start:])
        except json.JSONDecodeError:
            continue
        add(payload[start : start + end])

    return candidates


def profile_from_payload(query: str, payload: Any) -> UserPreferenceProfile:
    if not isinstance(payload, dict):
        return UserPreferenceProfile(raw_query=query)
    clarified_answers = payload.get("clarified_answers", {})
    inferred_requirements = payload.get("inferred_requirements", [])
    uncertainty_notes = payload.get("uncertainty_notes", [])
    return UserPreferenceProfile(
        raw_query=str(payload.get("raw_query", query)),
        clarified_answers={
            str(key): str(value) for key, value in clarified_answers.items() if isinstance(key, str)
        }
        if isinstance(clarified_answers, dict)
        else {},
        inferred_requirements=[str(item) for item in inferred_requirements if item is not None]
        if isinstance(inferred_requirements, list)
        else [],
        uncertainty_notes=[str(item) for item in uncertainty_notes if item is not None]
        if isinstance(uncertainty_notes, list)
        else [],
    )


def format_json_value(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=True)


def slugify(value: str) -> str:
    lowered = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return lowered or "clarification"


def response_total_tokens(response: Any) -> int:
    usage = None
    if isinstance(response, dict):
        usage = response.get("usage")
        if isinstance(usage, dict):
            return _coerce_non_negative_int(usage.get("total_tokens"))
        return 0

    usage = getattr(response, "usage", None)
    if usage is None:
        return 0
    if isinstance(usage, dict):
        return _coerce_non_negative_int(usage.get("total_tokens"))
    return _coerce_non_negative_int(getattr(usage, "total_tokens", 0))


def _coerce_non_negative_int(value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return parsed if parsed > 0 else 0
