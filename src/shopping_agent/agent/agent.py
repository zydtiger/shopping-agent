from __future__ import annotations

import inspect
import json
from collections.abc import Awaitable, Callable
from time import perf_counter
from typing import Any

from openai import AsyncOpenAI

from ..config import AppConfig
from ..event_log import EventLogger
from ..retrieval import AmazonAdapter, EbayAdapter, ProductSourceAdapter
from ..types import (
    ClarificationQuestion,
    Product,
    RankedProduct,
    RankingDesign,
    RetrievalBatch,
    SearchResponse,
)
from .errors import AgentHarnessError
from .parsing import (
    assistant_message_for_history,
    flatten_content,
    normalize_choices,
    parse_json_payload,
    profile_from_payload,
    slugify,
)
from .system_prompt import build_system_prompt

type ProgressCallback = Callable[[str], Awaitable[None] | None]
type ClarificationCallback = Callable[
    [ClarificationQuestion], Awaitable[dict[str, Any]] | dict[str, Any]
]
type ResponseRunner = Callable[..., Awaitable[Any]]


class ShoppingAgent:
    def __init__(
        self,
        logger: EventLogger,
        config: AppConfig | None = None,
        client: Any | None = None,
        response_runner: ResponseRunner | None = None,
        max_turns: int = 8,
    ) -> None:
        self.logger = logger
        self.config = config
        self.amazon_adapter = AmazonAdapter()
        self.ebay_adapter = EbayAdapter()
        self.client = client or self._build_client(config)
        self.response_runner = response_runner or self._default_response_runner
        self.max_turns = max_turns

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

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt(design)},
            {"role": "user", "content": query},
        ]
        retrieved_products: dict[str, Product] = {}
        retrieval_batches: list[RetrievalBatch] = []
        debug_notes: list[str] = [f"Selected ranking design: {design.label}."]

        self.logger.log_event(
            "agent_search_started",
            {"query": query, "design": design},
        )
        await self._emit_progress(
            progress,
            "[plan] Analyze the raw shopping request and decide whether clarification is needed.",
        )

        final_payload: dict[str, Any] | None = None
        for _ in range(self.max_turns):
            response = await self.response_runner(
                model=self._require_model(),
                messages=messages,
                tools=self._tool_specs(),
            )
            assistant_message = self._extract_message(response)
            content = flatten_content(assistant_message.get("content"))
            if content and not content.lstrip().startswith("{"):
                await self._emit_progress(progress, content)

            tool_calls = assistant_message.get("tool_calls") or []
            if tool_calls:
                messages.append(assistant_message_for_history(assistant_message))
                for tool_call in tool_calls:
                    tool_result = await self._execute_tool_call(
                        tool_call=tool_call,
                        ask_user=ask_user,
                        progress=progress,
                        retrieved_products=retrieved_products,
                        retrieval_batches=retrieval_batches,
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "content": json.dumps(tool_result, ensure_ascii=True),
                        }
                    )
                continue

            if not content:
                raise AgentHarnessError(
                    "Agent response was empty and contained no tool calls."
                )

            final_payload = parse_json_payload(content)
            break

        if final_payload is None:
            raise AgentHarnessError("Agent loop reached the maximum number of turns.")

        response = self._build_search_response(
            query=query,
            design=design,
            final_payload=final_payload,
            retrieval_batches=retrieval_batches,
            debug_notes=debug_notes,
            retrieved_products=retrieved_products,
        )
        self.logger.log_event(
            "agent_search_completed",
            {
                "query": query,
                "design": design,
                "profile": response.profile,
                "retrieval_batches": retrieval_batches,
                "ranked_products": response.ranked_products,
                "debug_notes": response.debug_notes,
            },
        )
        await self._emit_progress(
            progress,
            "[action] Final ranking prepared with "
            f"{len(response.ranked_products)} recommendation(s).",
        )
        return response

    async def _execute_tool_call(
        self,
        tool_call: dict[str, Any],
        ask_user: ClarificationCallback,
        progress: ProgressCallback | None,
        retrieved_products: dict[str, Product],
        retrieval_batches: list[RetrievalBatch],
    ) -> dict[str, Any]:
        function = tool_call.get("function") or {}
        tool_name = function.get("name")
        raw_arguments = function.get("arguments", "{}")
        arguments = parse_json_payload(raw_arguments)

        self.logger.log_event(
            "tool_called",
            {"tool_name": tool_name, "arguments": arguments},
        )

        if tool_name == "ask_clarification":
            return await self._handle_ask_clarification(arguments, ask_user, progress)
        if tool_name == "search_amazon":
            return await self._handle_search_amazon(
                arguments,
                progress,
                retrieved_products,
                retrieval_batches,
            )
        if tool_name == "search_ebay":
            return await self._handle_search_ebay(
                arguments,
                progress,
                retrieved_products,
                retrieval_batches,
            )
        raise AgentHarnessError(f"Unsupported tool call: {tool_name}")

    async def _handle_ask_clarification(
        self,
        arguments: dict[str, Any],
        ask_user: ClarificationCallback,
        progress: ProgressCallback | None,
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
                else "The agent needs one more preference to sharpen ranking."
            ),
            metadata={
                "preference_dimension": preference_dimension or None,
                "tool_name": "ask_clarification",
            },
        )
        await self._emit_progress(
            progress, f"[action] Clarification requested: {prompt}"
        )

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

        result = {
            "question": prompt,
            "preference_dimension": preference_dimension or question.id,
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

    async def _handle_search_amazon(
        self,
        arguments: dict[str, Any],
        progress: ProgressCallback | None,
        retrieved_products: dict[str, Product],
        retrieval_batches: list[RetrievalBatch],
    ) -> dict[str, Any]:
        query = str(arguments.get("query", "")).strip()
        if not query:
            raise AgentHarnessError("search_amazon requires a non-empty query.")

        return await self._run_search_tool(
            adapter=self.amazon_adapter,
            query=query,
            tool_name="search_amazon",
            progress=progress,
            retrieved_products=retrieved_products,
            retrieval_batches=retrieval_batches,
        )

    async def _handle_search_ebay(
        self,
        arguments: dict[str, Any],
        progress: ProgressCallback | None,
        retrieved_products: dict[str, Product],
        retrieval_batches: list[RetrievalBatch],
    ) -> dict[str, Any]:
        query = str(arguments.get("query", "")).strip()
        if not query:
            raise AgentHarnessError("search_ebay requires a non-empty query.")

        return await self._run_search_tool(
            adapter=self.ebay_adapter,
            query=query,
            tool_name="search_ebay",
            progress=progress,
            retrieved_products=retrieved_products,
            retrieval_batches=retrieval_batches,
        )

    async def _run_search_tool(
        self,
        adapter: ProductSourceAdapter,
        query: str,
        tool_name: str,
        progress: ProgressCallback | None,
        retrieved_products: dict[str, Product],
        retrieval_batches: list[RetrievalBatch],
    ) -> dict[str, Any]:
        await self._emit_progress(
            progress,
            f"[plan] Search {adapter.source_name} for: {query}",
        )
        started = perf_counter()
        products = await adapter.search(query, limit=50)
        latency_ms = int((perf_counter() - started) * 1000)
        retrieval_batches.append(
            RetrievalBatch(
                source=adapter.source_name,
                products=products,
                latency_ms=latency_ms,
            )
        )
        for product in products:
            retrieved_products[product.product_url] = product
        await self._emit_progress(
            progress,
            f"[action] {tool_name} returned {len(products)} product(s) in {latency_ms} ms.",
        )
        return {
            "query": query,
            "source": adapter.source_name,
            "result_count": len(products),
            "products": [product.to_dict() for product in products],
        }

    def _build_search_response(
        self,
        query: str,
        design: RankingDesign,
        final_payload: dict[str, Any],
        retrieval_batches: list[RetrievalBatch],
        debug_notes: list[str],
        retrieved_products: dict[str, Product],
    ) -> SearchResponse:
        profile = profile_from_payload(query, final_payload.get("profile"))
        recommendations_payload = final_payload.get("recommendations", [])
        ranked_products: list[RankedProduct] = []
        if isinstance(recommendations_payload, list):
            for item in recommendations_payload[:10]:
                if not isinstance(item, dict):
                    continue
                try:
                    ranked = RankedProduct.from_dict(item)
                except (TypeError, ValueError):
                    continue
                canonical = retrieved_products.get(ranked.product.product_url)
                if canonical is not None:
                    ranked = RankedProduct(
                        rank=ranked.rank,
                        score=ranked.score,
                        rationale=ranked.rationale,
                        product=canonical,
                    )
                ranked_products.append(ranked)

        raw_debug_notes = final_payload.get("debug_notes", [])
        if isinstance(raw_debug_notes, list):
            debug_notes.extend(str(note) for note in raw_debug_notes)

        status_message = str(
            final_payload.get(
                "status_message",
                f"Prepared {len(ranked_products)} recommendations from the agent harness.",
            )
        )
        return SearchResponse(
            query=query,
            design=design,
            profile=profile,
            stage="results",
            status_message=status_message,
            retrieval_batches=retrieval_batches,
            ranked_products=ranked_products,
            debug_notes=debug_notes,
        )

    def _system_prompt(self, design: RankingDesign) -> str:
        return build_system_prompt(design=design)

    def _tool_specs(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "ask_clarification",
                    "description": (
                        "Ask the user a popup clarification question with 3 to 5 "
                        "choices and an optional preference dimension."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": "The user-facing clarification question.",
                            },
                            "suggested_choices": {
                                "type": "array",
                                "description": "Three to five suggested options for the popup.",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "id": {"type": "string"},
                                        "label": {"type": "string"},
                                        "description": {"type": "string"},
                                    },
                                    "required": ["id", "label", "description"],
                                    "additionalProperties": False,
                                },
                                "minItems": 3,
                                "maxItems": 5,
                            },
                            "preference_dimension": {
                                "type": "string",
                                "description": (
                                    "Optional metadata name for the preference "
                                    "being resolved."
                                ),
                            },
                        },
                        "required": ["question", "suggested_choices"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_amazon",
                    "description": (
                        "Search Amazon with one query string and return up to 50 "
                        "normalized product results."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The Amazon search query generated by the agent.",
                            }
                        },
                        "required": ["query"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_ebay",
                    "description": (
                        "Search eBay with one query string and return up to 50 "
                        "normalized product results."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The eBay search query generated by the agent.",
                            }
                        },
                        "required": ["query"],
                        "additionalProperties": False,
                    },
                },
            },
        ]

    def _build_client(self, config: AppConfig | None) -> Any | None:
        if config is None:
            return None
        return AsyncOpenAI(
            api_key=config.agent.openai_api_key,
            base_url=config.agent.openai_base_url,
        )

    async def _default_response_runner(self, **kwargs: Any) -> Any:
        if self.client is None:
            raise AgentHarnessError(
                "No OpenAI client is configured. Provide AppConfig or inject a "
                "response runner for tests."
            )
        if not hasattr(self.client, "chat") or not hasattr(
            self.client.chat, "completions"
        ):
            raise AgentHarnessError(
                "Configured OpenAI client does not expose chat completions."
            )
        return await self.client.chat.completions.create(**kwargs)

    def _extract_message(self, response: Any) -> dict[str, Any]:
        if isinstance(response, dict):
            choices = response.get("choices") or []
            message = choices[0].get("message") if choices else None
            if not isinstance(message, dict):
                raise AgentHarnessError("Response payload is missing a message.")
            return message

        choices = getattr(response, "choices", None)
        if not choices:
            raise AgentHarnessError("Response object is missing choices.")
        message = getattr(choices[0], "message", None)
        if message is None:
            raise AgentHarnessError("Response choice is missing a message.")
        return {
            "role": getattr(message, "role", "assistant"),
            "content": getattr(message, "content", None),
            "tool_calls": [
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
            ],
        }

    def _require_model(self) -> str:
        if self.config is None:
            raise AgentHarnessError(
                "AppConfig with agent.openai_model_id is required to run the harness."
            )
        return self.config.agent.openai_model_id

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


def build_default_agent(
    log_dir: str = "logs",
    config: AppConfig | None = None,
    client: Any | None = None,
    response_runner: ResponseRunner | None = None,
) -> ShoppingAgent:
    return ShoppingAgent(
        logger=EventLogger(log_dir),
        config=config,
        client=client,
        response_runner=response_runner,
    )
