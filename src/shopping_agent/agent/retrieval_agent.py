from __future__ import annotations

import inspect
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from ..retrieval import AmazonAdapter, EbayAdapter, NeweggAdapter, ProductSourceAdapter
from ..types import (
    ClarificationQuestion,
    Product,
    RetrievalBatch,
    UserPreferenceProfile,
)
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
    normalize_choices,
    parse_json_payload,
    parse_json_value,
    profile_from_payload,
    response_total_tokens,
    slugify,
)
from .sql import ProductSQLStore
from .system_prompt import build_retrieval_system_prompt
from .tools import (
    build_retrieval_tool_specs,
    handle_search_amazon,
    handle_search_ebay,
    handle_search_newegg,
)

type ResponseRunner = Callable[..., Awaitable[Any]]
type ClarificationCallback = Callable[
    [ClarificationQuestion], Awaitable[dict[str, Any]] | dict[str, Any]
]


@dataclass(slots=True)
class RetrievalOutcome:
    profile: UserPreferenceProfile
    status_message: str
    retrieval_batches: list[RetrievalBatch]
    debug_notes: list[str]
    total_tokens: int = 0


class RetrievalAgent:
    def __init__(
        self,
        *,
        model_id: str,
        result_limit: int,
        response_runner: ResponseRunner,
        amazon_adapter: ProductSourceAdapter | None = None,
        ebay_adapter: ProductSourceAdapter | None = None,
        newegg_adapter: ProductSourceAdapter | None = None,
    ) -> None:
        self.model_id = model_id
        self.result_limit = max(1, result_limit)
        self.response_runner = response_runner
        self.amazon_adapter = amazon_adapter or AmazonAdapter()
        self.ebay_adapter = ebay_adapter or EbayAdapter()
        self.newegg_adapter = newegg_adapter or NeweggAdapter()

    async def run(
        self,
        *,
        query: str,
        ask_user: ClarificationCallback,
        progress: ProgressCallback | None,
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
        total_tokens = 0

        await emit_progress(
            progress,
            "[plan] RetrievalAgent is analyzing the raw shopping request and deciding "
            "whether clarification is needed.",
        )

        final_payload: dict[str, Any] | None = None
        while True:
            response = await self.response_runner(
                model=self.model_id,
                messages=messages,
                tools=build_retrieval_tool_specs(self.result_limit),
            )
            total_tokens += response_total_tokens(response)
            assistant_message, finish_reason = extract_choice(response)
            await emit_visible_content(progress, assistant_message, parse_json_value)

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
            await emit_final_payload(
                progress,
                "RetrievalAgent",
                final_payload,
                format_json_value,
            )
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
            total_tokens=total_tokens,
        )

    async def _execute_tool_call(
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
                emit_progress=emit_progress,
                progress=progress,
                retrieved_products=retrieved_products,
                retrieval_batches=retrieval_batches,
            )
        elif tool_name == "search_ebay":
            result = await handle_search_ebay(
                arguments=arguments,
                adapter=self.ebay_adapter,
                result_limit=self.result_limit,
                emit_progress=emit_progress,
                progress=progress,
                retrieved_products=retrieved_products,
                retrieval_batches=retrieval_batches,
            )
        elif tool_name == "search_newegg":
            result = await handle_search_newegg(
                arguments=arguments,
                adapter=self.newegg_adapter,
                result_limit=self.result_limit,
                emit_progress=emit_progress,
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
        await emit_progress(progress, f"[action] Clarification requested: {prompt}")

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
        await emit_progress(
            progress,
            f"[action] Clarification captured for {result['preference_dimension']}: {answer_text}",
        )
        return result
