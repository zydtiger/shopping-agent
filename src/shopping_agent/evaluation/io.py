from __future__ import annotations

import json
import re
from pathlib import Path

from .models import EvaluationCase

SUMMARY_KEYS = ("item_summary", "summary", "query")
_SLUG_PATTERN = re.compile(r"[^a-z0-9]+")


def read_jsonl_cases(path: Path) -> list[EvaluationCase]:
    cases: list[EvaluationCase] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            item_summary = _extract_summary(line)
            if not item_summary:
                raise ValueError(f"Unable to parse an item summary from {path} line {line_number}.")
            cases.append(EvaluationCase(item_summary=item_summary, source_line=line_number))
    return cases


def make_summary_slug(value: str) -> str:
    normalized = _SLUG_PATTERN.sub("-", value.strip().lower()).strip("-")
    return normalized or "item"


def build_output_path(
    out_dir: Path,
    prompt: str,
    *,
    reserved_paths: set[Path] | None = None,
) -> Path:
    reserved_paths = reserved_paths or set()
    out_dir.mkdir(parents=True, exist_ok=True)

    base_name = make_summary_slug(prompt)
    candidate = out_dir / f"{base_name}.json"
    suffix = 2
    while candidate.exists() or candidate in reserved_paths:
        candidate = out_dir / f"{base_name}-{suffix}.json"
        suffix += 1
    return candidate


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
        handle.write("\n")


def _extract_summary(raw_line: str) -> str:
    try:
        payload = json.loads(raw_line)
    except json.JSONDecodeError:
        return raw_line

    if isinstance(payload, str):
        return payload.strip()
    if isinstance(payload, dict):
        for key in SUMMARY_KEYS:
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""
