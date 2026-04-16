from __future__ import annotations

import inspect
import json
from collections.abc import Awaitable, Callable
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
from .tools import (
    build_tool_specs,
    handle_search_amazon,
    handle_search_ebay,
    handle_search_newegg,
)

type ProgressCallback = Callable[[str], Awaitable[None] | None]
type ClarificationCallback = Callable[
    [ClarificationQuestion], Awaitable[dict[str, Any]] | dict[str, Any]
]


class ShoppingAgent:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
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

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt(design)},
            {"role": "user", "content": query},
        ]
        retrieved_products: dict[str, Product] = {}
        retrieval_batches: list[RetrievalBatch] = []
        debug_notes: list[str] = [f"Selected ranking design: {design.label}."]

        await self._emit_progress(
            progress,
            "[plan] Analyze the raw shopping request and decide whether clarification is needed.",
        )

        final_payload: dict[str, Any] | None = None
        while True:
            response = await self.response_runner(
                model=self.config.agent.openai_model_id,
                messages=messages,
                tools=build_tool_specs(),
            )
            assistant_message, finish_reason = self._extract_choice(response)
            content = flatten_content(assistant_message.get("content"))
            if content and not content.lstrip().startswith("{"):
                await self._emit_progress(progress, content)

            tool_calls = assistant_message.get("tool_calls") or []
            if finish_reason == "tool_calls":
                if not tool_calls:
                    raise AgentHarnessError(
                        "Response indicated tool calls but did not include any tool calls."
                    )
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
                raise AgentHarnessError("Agent response was empty and contained no tool calls.")

            final_payload = parse_json_payload(content)
            break

        response = self._build_search_response(
            query=query,
            design=design,
            final_payload=final_payload,
            retrieval_batches=retrieval_batches,
            debug_notes=debug_notes,
            retrieved_products=retrieved_products,
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

        if tool_name == "ask_clarification":
            return await self._handle_ask_clarification(arguments, ask_user, progress)
        if tool_name == "search_amazon":
            return await handle_search_amazon(
                arguments=arguments,
                adapter=self.amazon_adapter,
                emit_progress=self._emit_progress,
                progress=progress,
                retrieved_products=retrieved_products,
                retrieval_batches=retrieval_batches,
            )
        if tool_name == "search_ebay":
            return await handle_search_ebay(
                arguments=arguments,
                adapter=self.ebay_adapter,
                emit_progress=self._emit_progress,
                progress=progress,
                retrieved_products=retrieved_products,
                retrieval_batches=retrieval_batches,
            )
        if tool_name == "search_newegg":
            return await handle_search_newegg(
                arguments=arguments,
                adapter=self.newegg_adapter,
                emit_progress=self._emit_progress,
                progress=progress,
                retrieved_products=retrieved_products,
                retrieval_batches=retrieval_batches,
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
