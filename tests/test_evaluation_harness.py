from __future__ import annotations

import asyncio
import pathlib
import sys
import tempfile
import unittest
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from conftest import build_config

from shopping_agent.agent.errors import AgentHarnessError
from shopping_agent.evaluation.harness import EvaluationHarness
from shopping_agent.evaluation.models import EvaluationCase
from shopping_agent.types import (
    ClarificationOption,
    ClarificationQuestion,
    Product,
    RankedProduct,
    RankingDesign,
    RetrievalBatch,
    SearchResponse,
    UserPreferenceProfile,
)


class FakeLLMSteps:
    def __init__(self, *, fail_summary: str | None = None) -> None:
        self.fail_summary = fail_summary

    async def draft_detailed_item(self, item_summary: str) -> str:
        if self.fail_summary and item_summary == self.fail_summary:
            raise RuntimeError("synthetic llm failure")
        return f"Detailed draft for {item_summary} with constraints and preferences."

    async def compress_prompt(self, detailed_item_draft: str) -> str:
        return "basic item"

    async def answer_clarification(
        self,
        detailed_item_draft: str,
        question: ClarificationQuestion,
    ) -> dict[str, Any]:
        option = question.options[1]
        return {
            "answer": option.label,
            "selected_choice_id": option.id,
            "selected_choice_label": option.label,
            "source": "suggested_choice",
        }

    async def judge_recommendations(
        self,
        detailed_item_draft: str,
        ranked_products: list[RankedProduct],
    ) -> list[dict[str, Any]]:
        return [
            {
                "product_url": item.product.product_url,
                "title": item.product.title,
                "eval_score": 88,
                "eval_rationale": "Strong fit.",
            }
            for item in ranked_products
        ]


class FakeShoppingAgent:
    def __init__(self) -> None:
        self.config = build_config()

    async def run_search(
        self,
        query: str,
        design: RankingDesign,
        progress: Any,
        ask_user: Any,
    ) -> SearchResponse:
        progress("[plan] Start retrieval")
        question = ClarificationQuestion(
            id="budget",
            prompt="What budget tier fits best?",
            reason="Need budget signal",
            options=[
                ClarificationOption(id="budget", label="Budget", description="Lower cost"),
                ClarificationOption(
                    id="mid_range",
                    label="Mid-range",
                    description="Balanced quality",
                ),
                ClarificationOption(id="premium", label="Premium", description="Top quality"),
            ],
        )
        await ask_user(question)
        progress("[action] Clarification received")

        product = Product(
            title="Atlas Travel Backpack 35L",
            price=129.99,
            source_site="Amazon",
            product_url="https://example.com/backpack",
            rating=4.8,
        )
        ranked = RankedProduct(
            rank=1,
            score=93,
            rationale="Best fit for commute and durability.",
            product=product,
        )
        batch = RetrievalBatch(source="Amazon", products=[product], latency_ms=120)
        return SearchResponse(
            query=query,
            design=design,
            profile=UserPreferenceProfile(
                raw_query=query,
                clarified_answers={"budget": "Mid-range"},
                inferred_requirements=["durable", "laptop"],
                uncertainty_notes=[],
            ),
            stage="results",
            status_message="Completed run.",
            retrieval_batches=[batch],
            ranked_products=[ranked],
            debug_notes=["fake debug note"],
        )


class _RetryingShoppingAgent(FakeShoppingAgent):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    async def run_search(
        self,
        query: str,
        design: RankingDesign,
        progress: Any,
        ask_user: Any,
    ) -> SearchResponse:
        self.calls += 1
        if self.calls == 1:
            raise AgentHarnessError("RetrievalAgent returned no final payload.")
        return await super().run_search(query, design, progress, ask_user)


class EvaluationHarnessTests(unittest.TestCase):
    def test_evaluate_case_runs_full_pipeline_and_scores_results(self) -> None:
        harness = EvaluationHarness(
            config=build_config(),
            design=RankingDesign.DIRECT_JSON,
            result_limit=10,
            shopping_agent=FakeShoppingAgent(),
            llm_steps=FakeLLMSteps(),
        )

        artifact = asyncio.run(
            harness.evaluate_case(
                EvaluationCase(item_summary="commuter backpack", source_line=3),
                source_file=pathlib.Path("/tmp/items.jsonl"),
            )
        )

        payload = artifact.to_dict()
        self.assertEqual(payload["item_summary"], "commuter backpack")
        self.assertEqual(payload["compressed_prompt"], "basic item")
        self.assertGreaterEqual(len(payload["interaction_logs"]["progress_logs"]), 5)
        self.assertIn(
            "[plan] Preparing hidden detailed item draft.",
            payload["interaction_logs"]["progress_logs"],
        )
        self.assertIn(
            "[plan] Generating compressed prompt from detailed draft.",
            payload["interaction_logs"]["progress_logs"],
        )
        self.assertEqual(len(payload["interaction_logs"]["clarification_turns"]), 1)
        self.assertEqual(len(payload["final_recommendations"]), 1)
        self.assertEqual(payload["final_recommendations"][0]["eval_score"], 88)
        self.assertIn("shopping_agent_run", payload)

    def test_evaluate_jsonl_continues_after_case_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = pathlib.Path(tmp_dir) / "items.jsonl"
            out_dir = pathlib.Path(tmp_dir) / "eval"
            input_path.write_text(
                "\n".join(
                    [
                        '{"item_summary": "valid item"}',
                        '{"item_summary": "broken item"}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            harness = EvaluationHarness(
                config=build_config(),
                design=RankingDesign.SQL,
                shopping_agent=FakeShoppingAgent(),
                llm_steps=FakeLLMSteps(fail_summary="broken item"),
            )

            outcomes = asyncio.run(
                harness.evaluate_jsonl(
                    input_path=input_path,
                    out_dir=out_dir,
                )
            )

            self.assertEqual(len(outcomes), 2)
            self.assertEqual(sum(1 for _, ok in outcomes if ok), 1)
            self.assertEqual(sum(1 for _, ok in outcomes if not ok), 1)
            for path, _ in outcomes:
                self.assertTrue(path.exists())

    def test_evaluate_case_retries_once_on_transient_payload_failure(self) -> None:
        harness = EvaluationHarness(
            config=build_config(),
            design=RankingDesign.DIRECT_JSON,
            shopping_agent=_RetryingShoppingAgent(),
            llm_steps=FakeLLMSteps(),
        )

        artifact = asyncio.run(
            harness.evaluate_case(
                EvaluationCase(item_summary="retry item", source_line=1),
                source_file=pathlib.Path("/tmp/items.jsonl"),
            )
        )

        payload = artifact.to_dict()
        self.assertEqual(payload["item_summary"], "retry item")
        self.assertTrue(
            any(
                "Retrying shopping agent run after transient payload issue" in entry
                for entry in payload["interaction_logs"]["progress_logs"]
            )
        )


if __name__ == "__main__":
    unittest.main()
