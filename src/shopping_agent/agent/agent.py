from __future__ import annotations

import inspect
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from openai import AsyncOpenAI

from ..config import AppConfig
from ..retrieval import AmazonAdapter, EbayAdapter, NeweggAdapter
from ..types import (
    ClarificationQuestion,
    Product,
    RankedProduct,
    RankingDesign,
    RetrievalBatch,
    SearchResponse,
    UserPreferenceProfile,
)
from .errors import AgentHarnessError
from .parsing import (
    assistant_message_for_history,
    flatten_content,
    normalize_choices,
    parse_json_payload,
    parse_json_value,
    profile_from_payload,
    slugify,
)
from .sql import ProductSQLStore
from .system_prompt import build_ranking_system_prompt, build_retrieval_system_prompt
from .tools import (
    build_ranking_sql_tool_specs,
    build_retrieval_tool_specs,
    handle_search_amazon,
    handle_search_ebay,
    handle_search_newegg,
    run_product_store_query,
)

type ProgressCallback = Callable[[str], Awaitable[None] | None]
type ClarificationCallback = Callable[
    [ClarificationQuestion], Awaitable[dict[str, Any]] | dict[str, Any]
]


@dataclass(slots=True)
class RetrievalOutcome:
    profile: UserPreferenceProfile
    status_message: str
    retrieval_batches: list[RetrievalBatch]
    debug_notes: list[str]


@dataclass(slots=True)
class RankingOutcome:
    ranked_products: list[RankedProduct]
    status_message: str
    debug_notes: list[str]


class ShoppingAgent:
    def __init__(self, config: AppConfig, result_limit: int = 50) -> None:
        self.config = config
        self.result_limit = max(1, result_limit)
        self.amazon_adapter = AmazonAdapter()
        self.ebay_adapter = EbayAdapter()
        self.newegg_adapter = NeweggAdapter()
        self.client = self._build_client(config)
        self.response_runner = self._default_response_runner

    async def run_search(
        self,
        query: str,
        design: RankingDesign,
        progress: ProgressCallback | None = None,
        ask_user: ClarificationCallback | None = None,
    ) -> SearchResponse:
        if ask_user is None:
            raise AgentHarnessError(
                "The harness needs a clarification callback to handle ask_clarification."
            )

        product_store = ProductSQLStore()
        retrieval = await self._run_retrieval_agent(
            query=query,
            progress=progress,
            ask_user=ask_user,
            product_store=product_store,
        )

        debug_notes = [f"Selected ranking design: {design.label}.", *retrieval.debug_notes]

        if not product_store.all_products():
            status_message = (
                "Retrieval finished but no products were collected, so ranking was skipped."
            )
            await self._emit_progress(progress, f"[action] {status_message}")
            return SearchResponse(
                query=query,
                design=design,
                profile=retrieval.profile,
                stage="results",
                status_message=status_message,
                retrieval_batches=retrieval.retrieval_batches,
                ranked_products=[],
                debug_notes=debug_notes,
            )

        await self._emit_progress(
            progress,
            "[plan] RetrievalAgent finished. Handing the shared product store to "
            f"RankingAgent using {design.label}.",
        )
        ranking = await self._run_ranking_agent(
            design=design,
            profile=retrieval.profile,
            product_store=product_store,
            progress=progress,
        )
        debug_notes.extend(ranking.debug_notes)

        await self._emit_progress(
            progress,
            "[action] Final ranking prepared with "
            f"{len(ranking.ranked_products)} recommendation(s).",
        )
        return SearchResponse(
            query=query,
            design=design,
            profile=retrieval.profile,
            stage="results",
            status_message=ranking.status_message or retrieval.status_message,
            retrieval_batches=retrieval.retrieval_batches,
            ranked_products=ranking.ranked_products,
            debug_notes=debug_notes,
        )

    async def _run_retrieval_agent(
        self,
        *,
        query: str,
        progress: ProgressCallback | None,
        ask_user: ClarificationCallback,
        product_store: ProductSQLStore,
    ) -> RetrievalOutcome:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": build_retrieval_system_prompt()},
            {"role": "user", "content": query},
        ]
        clarified_answers: dict[str, str] = {}
        retrieved_products: dict[str, Product] = {}
        retrieval_batches: list[RetrievalBatch] = []
        debug_notes: list[str] = []

        await self._emit_progress(
            progress,
            "[plan] RetrievalAgent is analyzing the raw shopping request and deciding "
            "whether clarification is needed.",
        )

        final_payload: dict[str, Any] | None = None
        while True:
            response = await self.response_runner(
                model=self.config.agent.openai_model_id,
                messages=messages,
                tools=build_retrieval_tool_specs(self.result_limit),
            )
            assistant_message, finish_reason = self._extract_choice(response)
            await self._emit_visible_content(progress, assistant_message)

            tool_calls = assistant_message.get("tool_calls") or []
            if finish_reason == "tool_calls":
                if not tool_calls:
                    raise AgentHarnessError(
                        "Response indicated tool calls but did not include any tool calls."
                    )
                messages.append(assistant_message_for_history(assistant_message))
                for tool_call in tool_calls:
                    tool_result = await self._execute_retrieval_tool_call(
                        tool_call=tool_call,
                        ask_user=ask_user,
                        progress=progress,
                        clarified_answers=clarified_answers,
                        retrieved_products=retrieved_products,
                        retrieval_batches=retrieval_batches,
                        product_store=product_store,
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
                raise AgentHarnessError("RetrievalAgent returned no final payload.")
            final_payload = parse_json_payload(content)
            break

        profile = profile_from_payload(query, final_payload.get("profile"))
        if clarified_answers:
            profile.clarified_answers = {**profile.clarified_answers, **clarified_answers}

        raw_debug_notes = final_payload.get("debug_notes", [])
        if isinstance(raw_debug_notes, list):
            debug_notes.extend(str(note) for note in raw_debug_notes)

        status_message = str(
            final_payload.get(
                "status_message",
                f"RetrievalAgent collected {len(product_store.all_products())} products.",
            )
        )
        return RetrievalOutcome(
            profile=profile,
            status_message=status_message,
            retrieval_batches=retrieval_batches,
            debug_notes=debug_notes,
        )

    async def _run_ranking_agent(
        self,
        *,
        design: RankingDesign,
        profile: UserPreferenceProfile,
        product_store: ProductSQLStore,
        progress: ProgressCallback | None,
    ) -> RankingOutcome:
        if design is RankingDesign.SQL:
            return await self._run_sql_ranking_agent(
                profile=profile,
                product_store=product_store,
                progress=progress,
            )
        return await self._run_direct_json_ranking_agent(
            profile=profile,
            product_store=product_store,
            progress=progress,
        )

    async def _run_direct_json_ranking_agent(
        self,
        *,
        profile: UserPreferenceProfile,
        product_store: ProductSQLStore,
        progress: ProgressCallback | None,
    ) -> RankingOutcome:
        await self._emit_progress(
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
            model=self.config.agent.openai_model_id,
            messages=messages,
        )
        assistant_message, finish_reason = self._extract_choice(response)
        if finish_reason == "tool_calls":
            raise AgentHarnessError("Direct JSON RankingAgent must not request tools.")

        await self._emit_visible_content(progress, assistant_message)
        content = flatten_content(assistant_message.get("content"))
        if not content:
            raise AgentHarnessError("RankingAgent returned no final payload.")
        final_payload = parse_json_payload(content)
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
        )

    async def _run_sql_ranking_agent(
        self,
        *,
        profile: UserPreferenceProfile,
        product_store: ProductSQLStore,
        progress: ProgressCallback | None,
    ) -> RankingOutcome:
        await self._emit_progress(
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
        while True:
            response = await self.response_runner(
                model=self.config.agent.openai_model_id,
                messages=messages,
                tools=build_ranking_sql_tool_specs(),
            )
            assistant_message, finish_reason = self._extract_choice(response)
            await self._emit_visible_content(progress, assistant_message)

            tool_calls = assistant_message.get("tool_calls") or []
            if finish_reason == "tool_calls":
                if not tool_calls:
                    raise AgentHarnessError(
                        "RankingAgent requested tool use but no SQL tool call was provided."
                    )
                messages.append(assistant_message_for_history(assistant_message))
                for tool_call in tool_calls:
                    tool_result = await self._execute_ranking_tool_call(
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
                raise AgentHarnessError("RankingAgent returned no final payload.")
            final_payload = parse_json_payload(content)
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
        )

    async def _execute_retrieval_tool_call(
        self,
        *,
        tool_call: dict[str, Any],
        ask_user: ClarificationCallback,
        progress: ProgressCallback | None,
        clarified_answers: dict[str, str],
        retrieved_products: dict[str, Product],
        retrieval_batches: list[RetrievalBatch],
        product_store: ProductSQLStore,
    ) -> dict[str, Any]:
        function = tool_call.get("function") or {}
        tool_name = function.get("name")
        raw_arguments = function.get("arguments", "{}")
        arguments = parse_json_payload(raw_arguments)

        if tool_name == "ask_clarification":
            return await self._handle_ask_clarification(
                arguments,
                ask_user,
                progress,
                clarified_answers,
            )

        previous_batch_count = len(retrieval_batches)
        if tool_name == "search_amazon":
            result = await handle_search_amazon(
                arguments=arguments,
                adapter=self.amazon_adapter,
                result_limit=self.result_limit,
                emit_progress=self._emit_progress,
                progress=progress,
                retrieved_products=retrieved_products,
                retrieval_batches=retrieval_batches,
            )
        elif tool_name == "search_ebay":
            result = await handle_search_ebay(
                arguments=arguments,
                adapter=self.ebay_adapter,
                result_limit=self.result_limit,
                emit_progress=self._emit_progress,
                progress=progress,
                retrieved_products=retrieved_products,
                retrieval_batches=retrieval_batches,
            )
        elif tool_name == "search_newegg":
            result = await handle_search_newegg(
                arguments=arguments,
                adapter=self.newegg_adapter,
                result_limit=self.result_limit,
                emit_progress=self._emit_progress,
                progress=progress,
                retrieved_products=retrieved_products,
                retrieval_batches=retrieval_batches,
            )
        else:
            raise AgentHarnessError(f"Unsupported tool call: {tool_name}")

        if len(retrieval_batches) > previous_batch_count:
            inserted = product_store.add_products(retrieval_batches[-1].products)
            result["inserted_count"] = inserted
            result["shared_store_total"] = len(product_store.all_products())
        return result

    async def _execute_ranking_tool_call(
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
        await self._emit_progress(progress, f"[plan] RankingAgent SQL query: {sql}")
        try:
            result = run_product_store_query(arguments=arguments, store=product_store)
        except ValueError as exc:
            raise AgentHarnessError(str(exc)) from exc
        await self._emit_progress(
            progress,
            f"[action] query_product_store returned {result['row_count']} row(s).",
        )
        return result

    async def _handle_ask_clarification(
        self,
        arguments: dict[str, Any],
        ask_user: ClarificationCallback,
        progress: ProgressCallback | None,
        clarified_answers: dict[str, str],
    ) -> dict[str, Any]:
        prompt = str(arguments.get("question", "")).strip()
        if not prompt:
            raise AgentHarnessError("ask_clarification requires a non-empty question.")

        preference_dimension = str(arguments.get("preference_dimension", "")).strip()
        suggested_choices = normalize_choices(arguments.get("suggested_choices", []))
        if len(suggested_choices) < 3 or len(suggested_choices) > 5:
            raise AgentHarnessError(
                "ask_clarification must provide between 3 and 5 suggested choices."
            )

        question = ClarificationQuestion(
            id=preference_dimension or slugify(prompt),
            prompt=prompt,
            options=suggested_choices,
            reason=(
                f"Preference dimension: {preference_dimension}."
                if preference_dimension
                else "The agent needs one more preference to sharpen retrieval."
            ),
            metadata={
                "preference_dimension": preference_dimension or None,
                "tool_name": "ask_clarification",
            },
        )
        await self._emit_progress(progress, f"[action] Clarification requested: {prompt}")

        answer_payload = ask_user(question)
        if inspect.isawaitable(answer_payload):
            answer_payload = await answer_payload
        if not isinstance(answer_payload, dict):
            raise AgentHarnessError(
                "Clarification callback must return a dictionary with answer details."
            )

        answer_text = str(answer_payload.get("answer", "")).strip()
        if not answer_text:
            raise AgentHarnessError("Clarification callback returned an empty answer.")

        answer_dimension = preference_dimension or question.id
        clarified_answers[answer_dimension] = answer_text
        result = {
            "question": prompt,
            "preference_dimension": answer_dimension,
            "answer": answer_text,
            "selected_choice_id": answer_payload.get("selected_choice_id"),
            "selected_choice_label": answer_payload.get("selected_choice_label"),
            "source": answer_payload.get("source", "custom"),
        }
        await self._emit_progress(
            progress,
            f"[action] Clarification captured for {result['preference_dimension']}: {answer_text}",
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

    def _build_client(self, config: AppConfig) -> AsyncOpenAI:
        return AsyncOpenAI(
            api_key=config.agent.openai_api_key,
            base_url=config.agent.openai_base_url,
        )

    async def _default_response_runner(self, **kwargs: Any) -> Any:
        if not hasattr(self.client, "chat") or not hasattr(self.client.chat, "completions"):
            raise AgentHarnessError("Configured OpenAI client does not expose chat completions.")
        return await self.client.chat.completions.create(**kwargs)

    def _extract_choice(self, response: Any) -> tuple[dict[str, Any], str | None]:
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
                    "arguments": getattr(
                        getattr(call, "function", None),
                        "arguments",
                        "{}",
                    ),
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

    async def _emit_visible_content(
        self,
        progress: ProgressCallback | None,
        assistant_message: dict[str, Any],
    ) -> None:
        content = flatten_content(assistant_message.get("content"))
        if not content:
            return
        try:
            parse_json_value(content)
        except Exception:
            await self._emit_progress(progress, content)

    async def _emit_progress(
        self,
        progress: ProgressCallback | None,
        message: str,
    ) -> None:
        if progress is None:
            return
        maybe_awaitable = progress(message)
        if inspect.isawaitable(maybe_awaitable):
            await maybe_awaitable
