from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from typing import Any

from openai import AsyncOpenAI

from ..config import AppConfig
from .errors import AgentHarnessError

type ProgressCallback = Callable[[str], Awaitable[None] | None]


def build_client(config: AppConfig) -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=config.agent.openai_api_key,
        base_url=config.agent.openai_base_url,
    )


async def default_response_runner(client: Any, **kwargs: Any) -> Any:
    if not hasattr(client, "chat") or not hasattr(client.chat, "completions"):
        raise AgentHarnessError("Configured OpenAI client does not expose chat completions.")
    return await client.chat.completions.create(**kwargs)


def extract_choice(response: Any) -> tuple[dict[str, Any], str | None]:
    if isinstance(response, dict):
        choices = response.get("choices") or []
        first_choice = choices[0] if choices else None
        message = first_choice.get("message") if isinstance(first_choice, dict) else None
        if not isinstance(message, dict):
            raise AgentHarnessError("Response payload is missing a message.")
        finish_reason = (
            first_choice.get("finish_reason") if isinstance(first_choice, dict) else None
        )
        if finish_reason is None:
            finish_reason = "tool_calls" if message.get("tool_calls") else "stop"
        return message, finish_reason

    choices = getattr(response, "choices", None)
    if not choices:
        raise AgentHarnessError("Response object is missing choices.")
    first_choice = choices[0]
    message = getattr(first_choice, "message", None)
    if message is None:
        raise AgentHarnessError("Response choice is missing a message.")

    tool_calls = [
        {
            "id": getattr(call, "id", ""),
            "type": getattr(call, "type", "function"),
            "function": {
                "name": getattr(getattr(call, "function", None), "name", ""),
                "arguments": getattr(getattr(call, "function", None), "arguments", "{}"),
            },
        }
        for call in getattr(message, "tool_calls", []) or []
    ]
    finish_reason = getattr(first_choice, "finish_reason", None)
    if finish_reason is None:
        finish_reason = "tool_calls" if tool_calls else "stop"

    return (
        {
            "role": getattr(message, "role", "assistant"),
            "content": getattr(message, "content", None),
            "tool_calls": tool_calls,
        },
        finish_reason,
    )


def assistant_message_for_history(message: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "role": "assistant",
        "content": message.get("content") or "",
    }
    tool_calls = message.get("tool_calls") or []
    if tool_calls:
        payload["tool_calls"] = tool_calls
    return payload


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


async def emit_progress(
    progress: ProgressCallback | None,
    message: str,
) -> None:
    if progress is None:
        return
    maybe_awaitable = progress(message)
    if inspect.isawaitable(maybe_awaitable):
        await maybe_awaitable


async def emit_visible_content(
    progress: ProgressCallback | None,
    assistant_message: dict[str, Any],
    parse_json_value: Callable[[str], Any],
) -> None:
    content = flatten_content(assistant_message.get("content"))
    if not content:
        return
    try:
        parse_json_value(content)
    except Exception:
        await emit_progress(progress, content)


async def emit_final_payload(
    progress: ProgressCallback | None,
    agent_name: str,
    payload: dict[str, Any],
    format_json_value: Callable[[Any], str],
) -> None:
    await emit_progress(progress, format_json_value(payload))
    await emit_progress(progress, f"[success] {agent_name} has completed")
