from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable
from typing import Any

from ..agent.api import extract_choice, flatten_content
from ..agent.errors import AgentHarnessError
from ..agent.parsing import parse_json_payload, parse_json_value
from ..types import ClarificationQuestion, RankedProduct

type ResponseRunner = Callable[..., Awaitable[Any]]


class EvaluationLLMSteps:
    def __init__(self, *, model_id: str, response_runner: ResponseRunner) -> None:
        self.model_id = model_id
        self.response_runner = response_runner

    async def draft_detailed_item(self, item_summary: str) -> str:
        payload = await self._complete_json(
            system_prompt=(
                "You write hidden evaluator shopping briefs. Expand a short item summary "
                "into a precise, highly detailed item requirement draft."
            ),
            user_payload={"item_summary": item_summary},
            output_contract={"detailed_item_draft": "string"},
            json_schema={
                "type": "object",
                "properties": {
                    "detailed_item_draft": {"type": "string"},
                },
                "required": ["detailed_item_draft"],
                "additionalProperties": False,
            },
        )
        draft = str(payload.get("detailed_item_draft", "")).strip()
        if not draft:
            raise AgentHarnessError("Draft generation returned an empty detailed_item_draft.")
        return draft

    async def compress_prompt(self, detailed_item_draft: str) -> str:
        payload = await self._complete_json(
            system_prompt=(
                "Convert a detailed shopping draft into one intentionally basic query of at most "
                "three words."
            ),
            user_payload={"detailed_item_draft": detailed_item_draft},
            output_contract={"compressed_prompt": "string up to 3 words"},
            json_schema={
                "type": "object",
                "properties": {
                    "compressed_prompt": {"type": "string"},
                },
                "required": ["compressed_prompt"],
                "additionalProperties": False,
            },
        )
        compressed = str(payload.get("compressed_prompt", "")).strip()
        if not compressed:
            compressed = self._fallback_compressed_prompt(detailed_item_draft)

        words = compressed.split()
        if len(words) > 3:
            compressed = " ".join(words[:3])
        return compressed

    async def answer_clarification(
        self,
        detailed_item_draft: str,
        question: ClarificationQuestion,
    ) -> dict[str, Any]:
        payload = await self._complete_json(
            system_prompt=(
                "Answer a shopping clarification question using only the provided hidden draft. "
                "Prefer one suggested option when possible and keep answer concise."
            ),
            user_payload={
                "detailed_item_draft": detailed_item_draft,
                "question": {
                    "id": question.id,
                    "prompt": question.prompt,
                    "reason": question.reason,
                    "options": [
                        {
                            "id": option.id,
                            "label": option.label,
                            "description": option.description,
                        }
                        for option in question.options
                    ],
                },
            },
            output_contract={
                "answer": "string",
                "selected_choice_id": "string|null",
                "selected_choice_label": "string|null",
                "source": "suggested_choice|custom",
            },
            json_schema={
                "type": "object",
                "properties": {
                    "answer": {"type": "string"},
                    "selected_choice_id": {"type": ["string", "null"]},
                    "selected_choice_label": {"type": ["string", "null"]},
                    "source": {"type": "string"},
                },
                "required": ["answer"],
                "additionalProperties": False,
            },
        )

        answer = str(payload.get("answer", "")).strip()
        if not answer:
            raise AgentHarnessError("Clarification answering returned an empty answer.")

        option_by_id = {option.id: option for option in question.options}
        selected_choice_id = payload.get("selected_choice_id")
        selected_choice_label = payload.get("selected_choice_label")
        source = str(payload.get("source", "")).strip() or "custom"

        normalized_choice_id = str(selected_choice_id).strip() if selected_choice_id else ""
        normalized_choice_label = (
            str(selected_choice_label).strip() if selected_choice_label else ""
        )

        if normalized_choice_id in option_by_id:
            matched = option_by_id[normalized_choice_id]
            return {
                "answer": matched.label,
                "selected_choice_id": matched.id,
                "selected_choice_label": matched.label,
                "source": "suggested_choice",
            }

        if normalized_choice_label:
            for option in question.options:
                if option.label.casefold() == normalized_choice_label.casefold():
                    return {
                        "answer": option.label,
                        "selected_choice_id": option.id,
                        "selected_choice_label": option.label,
                        "source": "suggested_choice",
                    }

        return {
            "answer": answer,
            "selected_choice_id": None,
            "selected_choice_label": None,
            "source": source,
        }

    async def judge_recommendations(
        self,
        detailed_item_draft: str,
        ranked_products: list[RankedProduct],
    ) -> list[dict[str, Any]]:
        payload = await self._complete_json(
            system_prompt=(
                "Evaluate each recommended product against the hidden detailed shopping draft. "
                "Return one score per item in [0, 100] and a short rationale."
            ),
            user_payload={
                "detailed_item_draft": detailed_item_draft,
                "recommendations": [
                    {
                        "rank": item.rank,
                        "agent_score": item.score,
                        "agent_rationale": item.rationale,
                        "product": item.product.to_dict(),
                    }
                    for item in ranked_products
                ],
            },
            output_contract={
                "evaluations": [
                    {
                        "product_url": "string optional",
                        "title": "string optional",
                        "eval_score": "integer 0-100",
                        "eval_rationale": "string",
                    }
                ]
            },
            json_schema={
                "type": "object",
                "properties": {
                    "evaluations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "product_url": {"type": "string"},
                                "title": {"type": "string"},
                                "eval_score": {"type": "integer"},
                                "eval_rationale": {"type": "string"},
                            },
                            "required": ["eval_score", "eval_rationale"],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["evaluations"],
                "additionalProperties": False,
            },
        )

        evaluations = payload.get("evaluations")
        if not isinstance(evaluations, list):
            raise AgentHarnessError("Judge response must include an evaluations list.")
        normalized: list[dict[str, Any]] = []
        for item in evaluations:
            if not isinstance(item, dict):
                continue
            try:
                score = max(0, min(100, int(item.get("eval_score", 0))))
            except (TypeError, ValueError):
                score = 0
            normalized.append(
                {
                    "product_url": str(item.get("product_url", "")).strip(),
                    "title": str(item.get("title", "")).strip(),
                    "eval_score": score,
                    "eval_rationale": str(item.get("eval_rationale", "")).strip(),
                }
            )
        return normalized

    async def _complete_json(
        self,
        *,
        system_prompt: str,
        user_payload: dict[str, Any],
        output_contract: dict[str, Any],
        json_schema: dict[str, Any],
    ) -> dict[str, Any]:
        prompt = "\n\n".join(
            [
                system_prompt,
                "Return one JSON object only.",
                "Strict JSON schema to follow:",
                json.dumps(json_schema, ensure_ascii=True),
            ]
        )

        last_error: Exception | None = None
        for _ in range(2):
            response = await self.response_runner(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": prompt},
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "input": user_payload,
                                "output_contract": output_contract,
                                "return_only_valid_json": True,
                            },
                            ensure_ascii=True,
                        ),
                    },
                ],
            )

            assistant_message, _ = extract_choice(response)
            content = flatten_content(assistant_message.get("content"))
            if not content:
                last_error = AgentHarnessError("Evaluation LLM step returned no payload.")
                continue

            try:
                parsed = parse_json_value(content)
                if not isinstance(parsed, dict):
                    raise AgentHarnessError("Evaluation LLM step must return a JSON object.")
                return parse_json_payload(json.dumps(parsed, ensure_ascii=True))
            except Exception as exc:
                last_error = exc
                continue

        raise AgentHarnessError(
            f"Evaluation LLM step failed to return valid structured JSON: {last_error}"
        )

    def _fallback_compressed_prompt(self, detailed_item_draft: str) -> str:
        tokens = re.findall(r"[A-Za-z0-9]+", detailed_item_draft)
        if not tokens:
            raise AgentHarnessError("Compression returned an empty compressed_prompt.")
        return " ".join(tokens[:3])
