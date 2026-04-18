from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from ..types import Product, RankedProduct, RankingDesign, UserPreferenceProfile
from .api import (
    ProgressCallback,
    assistant_message_for_history,
    emit_final_payload,
    emit_progress,
    emit_visible_content,
    extract_choice,
    flatten_content,
)
from .errors import AgentHarnessError
from .parsing import (
    format_json_value,
    parse_json_payload,
    parse_json_value,
    response_total_tokens,
)
from .sql import ProductSQLStore
from .system_prompt import build_ranking_system_prompt
from .tools import build_ranking_sql_tool_specs, run_product_store_query

type ResponseRunner = Callable[..., Awaitable[Any]]


@dataclass(slots=True)
class RankingOutcome:
    ranked_products: list[RankedProduct]
    status_message: str
    debug_notes: list[str]
    total_tokens: int = 0


class RankingAgent:
    def __init__(
        self,
        *,
        model_id: str,
        response_runner: ResponseRunner,
    ) -> None:
        self.model_id = model_id
        self.response_runner = response_runner

    async def run(
        self,
        *,
        design: RankingDesign,
        profile: UserPreferenceProfile,
        product_store: ProductSQLStore,
        progress: ProgressCallback | None,
    ) -> RankingOutcome:
        if design is RankingDesign.SQL:
            return await self._run_sql(
                profile=profile,
                product_store=product_store,
                progress=progress,
            )
        return await self._run_direct_json(
            profile=profile,
            product_store=product_store,
            progress=progress,
        )

    async def _run_direct_json(
        self,
        *,
        profile: UserPreferenceProfile,
        product_store: ProductSQLStore,
        progress: ProgressCallback | None,
    ) -> RankingOutcome:
        await emit_progress(
            progress,
            "[plan] RankingAgent is evaluating the full normalized product payload directly.",
        )
        messages = [
            {
                "role": "system",
                "content": build_ranking_system_prompt(RankingDesign.DIRECT_JSON),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "profile": profile.to_dict(),
                        "products": [product.to_dict() for product in product_store.all_products()],
                    },
                    ensure_ascii=True,
                ),
            },
        ]
        response = await self.response_runner(
            model=self.model_id,
            messages=messages,
        )
        total_tokens = response_total_tokens(response)
        assistant_message, finish_reason = extract_choice(response)
        if finish_reason == "tool_calls":
            raise AgentHarnessError("Direct JSON RankingAgent must not request tools.")

        await emit_visible_content(progress, assistant_message, parse_json_value)
        content = flatten_content(assistant_message.get("content"))
        if not content:
            raise AgentHarnessError("RankingAgent returned no final payload.")

        final_payload = parse_json_payload(content)
        await emit_final_payload(
            progress,
            "RankingAgent",
            final_payload,
            format_json_value,
        )
        return RankingOutcome(
            ranked_products=self._parse_ranked_products(
                final_payload.get("recommendations"),
                product_store,
            ),
            status_message=str(
                final_payload.get(
                    "status_message",
                    "RankingAgent prepared recommendations from the injected JSON payload.",
                )
            ),
            debug_notes=[str(note) for note in final_payload.get("debug_notes", [])]
            if isinstance(final_payload.get("debug_notes"), list)
            else [],
            total_tokens=total_tokens,
        )

    async def _run_sql(
        self,
        *,
        profile: UserPreferenceProfile,
        product_store: ProductSQLStore,
        progress: ProgressCallback | None,
    ) -> RankingOutcome:
        await emit_progress(
            progress,
            "[plan] RankingAgent is querying the shared SQLite product "
            "store to shortlist candidates.",
        )
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": build_ranking_system_prompt(RankingDesign.SQL),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "profile": profile.to_dict(),
                        "product_store": product_store.summary(),
                    },
                    ensure_ascii=True,
                ),
            },
        ]

        final_payload: dict[str, Any] | None = None
        total_tokens = 0
        while True:
            response = await self.response_runner(
                model=self.model_id,
                messages=messages,
                tools=build_ranking_sql_tool_specs(),
            )
            total_tokens += response_total_tokens(response)
            assistant_message, finish_reason = extract_choice(response)
            await emit_visible_content(progress, assistant_message, parse_json_value)

            tool_calls = assistant_message.get("tool_calls") or []
            if finish_reason == "tool_calls":
                if not tool_calls:
                    raise AgentHarnessError(
                        "RankingAgent requested tool use but no SQL tool call was provided."
                    )
                messages.append(assistant_message_for_history(assistant_message))
                for tool_call in tool_calls:
                    tool_result = await self._execute_tool_call(
                        tool_call=tool_call,
                        product_store=product_store,
                        progress=progress,
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "content": json.dumps(tool_result, ensure_ascii=True),
                        }
                    )
                continue

            content = flatten_content(assistant_message.get("content"))
            if not content:
                raise AgentHarnessError(
                    "RankingAgent returned no final payload. "
                    f"finish_reason={finish_reason} "
                    f"assistant_message={json.dumps(assistant_message, ensure_ascii=True)}"
                )

            final_payload = parse_json_payload(content)
            await emit_final_payload(
                progress,
                "RankingAgent",
                final_payload,
                format_json_value,
            )
            break

        return RankingOutcome(
            ranked_products=self._parse_ranked_products(
                final_payload.get("recommendations"),
                product_store,
            ),
            status_message=str(
                final_payload.get(
                    "status_message",
                    "RankingAgent prepared recommendations from the SQL shortlist.",
                )
            ),
            debug_notes=[str(note) for note in final_payload.get("debug_notes", [])]
            if isinstance(final_payload.get("debug_notes"), list)
            else [],
            total_tokens=total_tokens,
        )

    async def _execute_tool_call(
        self,
        *,
        tool_call: dict[str, Any],
        product_store: ProductSQLStore,
        progress: ProgressCallback | None,
    ) -> dict[str, Any]:
        function = tool_call.get("function") or {}
        tool_name = function.get("name")
        raw_arguments = function.get("arguments", "{}")
        arguments = parse_json_payload(raw_arguments)

        if tool_name != "query_product_store":
            raise AgentHarnessError(f"Unsupported ranking tool call: {tool_name}")

        sql = str(arguments.get("sql", "")).strip()
        await emit_progress(progress, f"[plan] RankingAgent SQL query: {sql}")
        try:
            result = run_product_store_query(arguments=arguments, store=product_store)
        except ValueError as exc:
            raise AgentHarnessError(str(exc)) from exc

        await emit_progress(
            progress,
            f"[action] query_product_store returned {result['row_count']} row(s).",
        )
        return result

    def _parse_ranked_products(
        self,
        payload: Any,
        product_store: ProductSQLStore,
    ) -> list[RankedProduct]:
        if not isinstance(payload, list):
            raise AgentHarnessError(
                "RankingAgent final payload must contain a recommendations list."
            )

        deduped: dict[str, RankedProduct] = {}
        for item in payload:
            if not isinstance(item, dict):
                raise AgentHarnessError("Each recommendation must be a JSON object.")
            product_payload = item.get("product")
            if not isinstance(product_payload, dict):
                raise AgentHarnessError("Each recommendation must include a product object.")

            parsed_product = Product.from_dict(product_payload)
            canonical_product = (
                product_store.get_product(parsed_product.product_url) or parsed_product
            )
            score = max(0, min(100, int(item.get("score", 0))))
            ranked = RankedProduct(
                rank=0,
                score=score,
                rationale=str(item.get("rationale", "")),
                product=canonical_product,
            )
            existing = deduped.get(canonical_product.product_url)
            if existing is None or ranked.score > existing.score:
                deduped[canonical_product.product_url] = ranked

        normalized = sorted(deduped.values(), key=lambda item: item.score, reverse=True)[:10]
        return [
            RankedProduct(
                rank=index + 1,
                score=item.score,
                rationale=item.rationale,
                product=item.product,
            )
            for index, item in enumerate(normalized)
        ]
