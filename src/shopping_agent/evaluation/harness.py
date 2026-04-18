from __future__ import annotations

import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..agent import ShoppingAgent
from ..agent.errors import AgentHarnessError
from ..config import AppConfig
from ..types import ClarificationQuestion, RankedProduct, RankingDesign, SearchResponse
from .io import build_output_path, read_jsonl_cases, write_json
from .llm_steps import EvaluationLLMSteps
from .models import (
    ClarificationTurn,
    EvaluationArtifact,
    EvaluationCase,
    FinalRecommendationRecord,
    InteractionLogs,
    ShoppingAgentRunRecord,
)


class EvaluationHarness:
    def __init__(
        self,
        *,
        config: AppConfig,
        design: RankingDesign = RankingDesign.DIRECT_JSON,
        result_limit: int = 50,
        shopping_agent: ShoppingAgent | None = None,
        llm_steps: EvaluationLLMSteps | None = None,
    ) -> None:
        self.design = design
        self.result_limit = max(1, result_limit)
        self.shopping_agent = shopping_agent or ShoppingAgent(
            config=config,
            result_limit=self.result_limit,
        )
        self.llm_steps = llm_steps or EvaluationLLMSteps(
            model_id=config.agent.openai_model_id,
            response_runner=self.shopping_agent.response_runner,
        )

    async def evaluate_jsonl(
        self,
        *,
        input_path: Path,
        out_dir: Path,
    ) -> list[tuple[Path, bool]]:
        cases = read_jsonl_cases(input_path)
        outputs: list[tuple[Path, bool]] = []
        reserved_paths: set[Path] = set()

        total = len(cases)
        for index, case in enumerate(cases, start=1):
            self._print_batch_progress(index, total, f"start: {case.item_summary}")
            try:
                artifact = await self.evaluate_case(
                    case,
                    source_file=input_path,
                )
                output_path = build_output_path(
                    out_dir=out_dir,
                    prompt=artifact.compressed_prompt,
                    reserved_paths=reserved_paths,
                )
                reserved_paths.add(output_path)
                write_json(output_path, artifact.to_dict())
                outputs.append((output_path, True))
                self._print_batch_progress(index, total, f"ok: {output_path.name}")
            except Exception as exc:
                error_artifact = self._build_error_artifact(
                    case,
                    error=str(exc),
                    source_file=input_path,
                )
                output_path = build_output_path(
                    out_dir=out_dir,
                    prompt=case.item_summary,
                    reserved_paths=reserved_paths,
                )
                reserved_paths.add(output_path)
                write_json(output_path, error_artifact.to_dict())
                outputs.append((output_path, False))
                self._print_batch_progress(index, total, f"error: {output_path.name}")
        return outputs

    async def evaluate_case(
        self,
        case: EvaluationCase,
        *,
        source_file: Path | None,
    ) -> EvaluationArtifact:
        started_at = _utc_now_iso()
        interaction_logs = InteractionLogs()

        def record_progress(message: str) -> None:
            interaction_logs.progress_logs.append(message)
            for line in message.split("\n"):
                print(f"  {line}", file=sys.stdout)

        record_progress("[plan] Preparing hidden detailed item draft.")
        detailed_item_draft = await self.llm_steps.draft_detailed_item(case.item_summary)
        record_progress("[plan] Generating compressed prompt from detailed draft.")
        compressed_prompt = await self.llm_steps.compress_prompt(detailed_item_draft)
        record_progress(f"[action] Compressed prompt ready: {compressed_prompt}")

        async def ask_user(question: ClarificationQuestion) -> dict[str, Any]:
            answer_payload = await self.llm_steps.answer_clarification(
                detailed_item_draft=detailed_item_draft,
                question=question,
            )
            interaction_logs.clarification_turns.append(
                ClarificationTurn(
                    question_id=question.id,
                    prompt=question.prompt,
                    reason=question.reason,
                    options=[
                        {
                            "id": option.id,
                            "label": option.label,
                            "description": option.description,
                        }
                        for option in question.options
                    ],
                    answer=str(answer_payload.get("answer", "")),
                    selected_choice_id=_optional_string(answer_payload.get("selected_choice_id")),
                    selected_choice_label=_optional_string(
                        answer_payload.get("selected_choice_label")
                    ),
                    source=str(answer_payload.get("source", "custom")),
                )
            )
            return answer_payload

        response = await self._run_search_with_retry(
            compressed_prompt=compressed_prompt,
            progress_callback=record_progress,
            ask_user=ask_user,
        )

        judged_rows = await self.llm_steps.judge_recommendations(
            detailed_item_draft=detailed_item_draft,
            ranked_products=response.ranked_products,
        )

        final_recommendations = self._merge_judged_recommendations(
            ranked_products=response.ranked_products,
            judged_rows=judged_rows,
        )

        shopping_agent_run = _serialize_search_response(response)
        completed_at = _utc_now_iso()
        return EvaluationArtifact(
            item_summary=case.item_summary,
            detailed_item_draft=detailed_item_draft,
            compressed_prompt=compressed_prompt,
            shopping_agent_run=shopping_agent_run,
            interaction_logs=interaction_logs,
            final_recommendations=final_recommendations,
            timestamps={"started_at": started_at, "completed_at": completed_at},
            meta={
                "design": self.design.value,
                "model_id": self.shopping_agent.config.agent.openai_model_id,
                "result_limit": self.result_limit,
                "source_file": str(source_file) if source_file else None,
                "source_line": case.source_line,
            },
        )

    def _build_error_artifact(
        self,
        case: EvaluationCase,
        *,
        error: str,
        source_file: Path | None,
    ) -> EvaluationArtifact:
        now = _utc_now_iso()
        return EvaluationArtifact(
            item_summary=case.item_summary,
            detailed_item_draft="",
            compressed_prompt="",
            shopping_agent_run=None,
            interaction_logs=InteractionLogs(),
            final_recommendations=[],
            timestamps={"started_at": now, "completed_at": now},
            meta={
                "design": self.design.value,
                "model_id": self.shopping_agent.config.agent.openai_model_id,
                "result_limit": self.result_limit,
                "source_file": str(source_file) if source_file else None,
                "source_line": case.source_line,
            },
            error=error,
        )

    def _merge_judged_recommendations(
        self,
        *,
        ranked_products: list[RankedProduct],
        judged_rows: list[dict[str, Any]],
    ) -> list[FinalRecommendationRecord]:
        by_url: dict[str, dict[str, Any]] = {}
        by_title: dict[str, dict[str, Any]] = {}
        for row in judged_rows:
            url = str(row.get("product_url", "")).strip()
            title = str(row.get("title", "")).strip().casefold()
            if url and url not in by_url:
                by_url[url] = row
            if title and title not in by_title:
                by_title[title] = row

        results: list[FinalRecommendationRecord] = []
        for item in ranked_products:
            matched = by_url.get(item.product.product_url)
            if matched is None:
                matched = by_title.get(item.product.title.casefold())
            eval_score = int(matched.get("eval_score", 0)) if matched else 0
            eval_rationale = str(matched.get("eval_rationale", "")) if matched else ""
            results.append(
                FinalRecommendationRecord(
                    rank=item.rank,
                    agent_score=item.score,
                    agent_rationale=item.rationale,
                    product=item.product.to_dict(),
                    eval_score=max(0, min(100, eval_score)),
                    eval_rationale=eval_rationale,
                )
            )
        return results

    def _print_batch_progress(self, index: int, total: int, message: str) -> None:
        print(f"[{index}/{total}] {message}", file=sys.stdout)

    async def _run_search_with_retry(
        self,
        *,
        compressed_prompt: str,
        progress_callback: Any,
        ask_user: Any,
    ) -> SearchResponse:
        max_attempts = 2
        for attempt in range(1, max_attempts + 1):
            try:
                return await self.shopping_agent.run_search(
                    compressed_prompt,
                    self.design,
                    progress=progress_callback,
                    ask_user=ask_user,
                )
            except AgentHarnessError as exc:
                message = str(exc)
                retryable = "returned no final payload" in message
                if not retryable or attempt >= max_attempts:
                    raise
                progress_callback(
                    f"[action] Retrying shopping agent run after transient payload issue "
                    f"(attempt {attempt + 1}/{max_attempts})."
                )
        raise AgentHarnessError("Shopping agent run failed without an explicit error.")


def _serialize_search_response(response: SearchResponse) -> ShoppingAgentRunRecord:
    retrieval_batches = [
        {
            "source": batch.source,
            "latency_ms": batch.latency_ms,
            "products": [product.to_dict() for product in batch.products],
        }
        for batch in response.retrieval_batches
    ]
    ranked_products = [
        {
            "rank": item.rank,
            "score": item.score,
            "rationale": item.rationale,
            "product": item.product.to_dict(),
        }
        for item in response.ranked_products
    ]
    return ShoppingAgentRunRecord(
        design=response.design.value,
        status_message=response.status_message,
        profile=asdict(response.profile),
        retrieval_batches=retrieval_batches,
        ranked_products=ranked_products,
        debug_notes=response.debug_notes,
        total_tokens=response.total_tokens,
    )


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _utc_now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()
