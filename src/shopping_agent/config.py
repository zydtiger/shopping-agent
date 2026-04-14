from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True, frozen=True)
class OpenAIConfig:
    openai_base_url: str
    openai_model_id: str
    openai_api_key: str

    @classmethod
    def from_mapping(cls, name: str, raw_config: object) -> OpenAIConfig:
        if not isinstance(raw_config, dict):
            raise ValueError(f"Expected '{name}' to be a mapping in the config file.")

        return cls(
            openai_base_url=_require_string(raw_config, name, "openai_base_url"),
            openai_model_id=_require_string(raw_config, name, "openai_model_id"),
            openai_api_key=_require_string(raw_config, name, "openai_api_key"),
        )


@dataclass(slots=True, frozen=True)
class AppConfig:
    agent: OpenAIConfig
    embedding: OpenAIConfig

    @classmethod
    def from_file(cls, path: str | Path) -> AppConfig:
        config_path = Path(path).expanduser()
        with config_path.open("r", encoding="utf-8") as handle:
            raw_config = yaml.safe_load(handle) or {}

        if not isinstance(raw_config, dict):
            raise ValueError("Expected the config file root to be a mapping.")

        return cls(
            agent=OpenAIConfig.from_mapping("agent", raw_config.get("agent")),
            embedding=OpenAIConfig.from_mapping(
                "embedding", raw_config.get("embedding")
            ),
        )


def _require_string(raw_config: dict[str, Any], section: str, key: str) -> str:
    value = raw_config.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Expected '{section}.{key}' to be a non-empty string.")
    return value
