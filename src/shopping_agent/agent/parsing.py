from __future__ import annotations

import json
import re
from typing import Any

from ..types import (
    ClarificationOption,
    UserPreferenceProfile,
)
from .errors import AgentHarnessError


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


def assistant_message_for_history(message: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "role": "assistant",
        "content": message.get("content") or "",
    }
    tool_calls = message.get("tool_calls") or []
    if tool_calls:
        payload["tool_calls"] = tool_calls
    return payload


def parse_json_payload(payload: str) -> dict[str, Any]:
    parsed = parse_json_value(payload)
    if not isinstance(parsed, dict):
        raise AgentHarnessError("Expected the agent payload to be a JSON object.")
    return parsed


def parse_json_value(payload: str) -> Any:
    stripped = payload.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    return json.loads(stripped)


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


def flatten_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text")
                if isinstance(text, str):
                    text_parts.append(text)
            elif hasattr(item, "text") and isinstance(item.text, str):  # type: ignore
                text_parts.append(item.text)  # type: ignore
        return "\n".join(part.strip() for part in text_parts if part.strip())
    return str(content).strip()


def slugify(value: str) -> str:
    lowered = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return lowered or "clarification"
