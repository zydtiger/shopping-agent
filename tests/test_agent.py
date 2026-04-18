from __future__ import annotations

import asyncio
import copy
import json
import pathlib
import sys
import unittest
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from conftest import (
    FakeOpenAIClient,
    MockAmazonProvider,
    SequenceRunner,
    build_amazon_products,
    build_config,
    build_mock_amazon_catalog,
    build_two_agent_direct_json_responses,
    build_two_agent_sql_responses,
)

from shopping_agent.agent import ShoppingAgent
from shopping_agent.types import RankingDesign


class ShoppingAgentHarnessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.amazon_products = build_amazon_products()

    def test_agent_runs_two_agent_direct_json_pipeline(self) -> None:
        responses = copy.deepcopy(build_two_agent_direct_json_responses())
        token_budget = [111, 222, 333, 444]
        for index, item in enumerate(responses):
            item["usage"] = {"total_tokens": token_budget[index]}
        runner = SequenceRunner(responses)
        progress_messages: list[str] = []
        clarification_calls: list[str] = []

        async def ask_user(question: Any) -> dict[str, object]:
            clarification_calls.append(question.prompt)
            return {
                "answer": "Mid-range",
                "selected_choice_id": "mid_range",
                "selected_choice_label": "Mid-range",
                "source": "suggested_choice",
            }

        agent = ShoppingAgent(config=build_config())
        agent.client = FakeOpenAIClient(runner)
        agent.amazon_adapter = MockAmazonProvider("Amazon", self.amazon_products)

        response = asyncio.run(
            agent.run_search(
                "good backpack",
                RankingDesign.DIRECT_JSON,
                progress=progress_messages.append,
                ask_user=ask_user,
            )
        )

        self.assertEqual(response.stage, "results")
        self.assertEqual(response.profile.clarified_answers["budget"], "Mid-range")
        self.assertEqual(len(response.retrieval_batches), 1)
        self.assertEqual(len(response.ranked_products), 2)
        self.assertEqual(response.ranked_products[0].product.title, "Atlas Travel Backpack 35L")
        self.assertGreater(response.ranked_products[0].score, response.ranked_products[1].score)
        self.assertEqual(response.ranked_products[0].rank, 1)
        self.assertEqual(response.total_tokens, sum(token_budget))
        self.assertEqual(clarification_calls, ["What budget tier fits best?"])
        self.assertTrue(
            any("search_amazon returned 2 product(s)" in item for item in progress_messages)
        )
        self.assertTrue(
            any(
                "RankingAgent is evaluating the full normalized product payload directly." in item
                for item in progress_messages
            )
        )
        self.assertEqual(runner.calls[0]["model"], "gpt-agent")
        self.assertEqual(len(runner.calls), 4)
        ranking_payload = json.loads(runner.calls[3]["messages"][1]["content"])
        self.assertEqual(ranking_payload["profile"]["clarified_answers"]["budget"], "Mid-range")

    def test_agent_runs_sql_ranking_via_product_store_tool(self) -> None:
        responses = copy.deepcopy(build_two_agent_sql_responses())
        token_budget = [120, 130, 140, 150]
        for index, item in enumerate(responses):
            item["usage"] = {"total_tokens": token_budget[index]}
        runner = SequenceRunner(responses)
        agent = ShoppingAgent(config=build_config())
        agent.client = FakeOpenAIClient(runner)
        agent.amazon_adapter = MockAmazonProvider("Amazon", build_mock_amazon_catalog())

        response = asyncio.run(
            agent.run_search(
                "travel backpack",
                RankingDesign.SQL,
                ask_user=lambda _: {"answer": "unused"},
            )
        )

        self.assertEqual(response.stage, "results")
        self.assertEqual(response.retrieval_batches[0].source, "Amazon")
        self.assertGreaterEqual(len(response.retrieval_batches[0].products), 1)
        self.assertIn("Travel Backpack", response.ranked_products[0].product.title)
        self.assertGreaterEqual(
            response.ranked_products[0].score,
            response.ranked_products[1].score,
        )
        self.assertEqual(response.ranked_products[0].rank, 1)
        self.assertEqual(response.design, RankingDesign.SQL)
        self.assertEqual(response.total_tokens, sum(token_budget))
        self.assertTrue(any(call.get("tools") for call in runner.calls[2:3]))
        ranking_payload = json.loads(runner.calls[2]["messages"][1]["content"])
        self.assertEqual(ranking_payload["profile"]["raw_query"], "travel backpack")

    def test_agent_uses_configured_result_limit_for_scrapers(self) -> None:
        runner = SequenceRunner(build_two_agent_sql_responses())
        agent = ShoppingAgent(config=build_config(), result_limit=1)
        agent.client = FakeOpenAIClient(runner)
        agent.amazon_adapter = MockAmazonProvider("Amazon", build_mock_amazon_catalog())

        response = asyncio.run(
            agent.run_search(
                "travel backpack",
                RankingDesign.SQL,
                ask_user=lambda _: {"answer": "unused"},
            )
        )

        self.assertEqual(len(response.retrieval_batches), 1)
        self.assertEqual(len(response.retrieval_batches[0].products), 1)
        search_amazon_spec = next(
            spec for spec in runner.calls[0]["tools"] if spec["function"]["name"] == "search_amazon"
        )
        self.assertIn(
            "up to 1 normalized product results",
            search_amazon_spec["function"]["description"],
        )

    def test_agent_parses_fenced_retrieval_json_and_logs_success(self) -> None:
        responses = copy.deepcopy(build_two_agent_direct_json_responses())
        retrieval_payload = json.loads(responses[2]["choices"][0]["message"]["content"])
        responses[2]["choices"][0]["message"]["content"] = "\n".join(
            [
                "[plan] Will stop retrieval and return final JSON profile.",
                "",
                "```json",
                json.dumps(retrieval_payload, indent=2, ensure_ascii=True),
                "```",
            ]
        )

        runner = SequenceRunner(responses)
        progress_messages: list[str] = []

        agent = ShoppingAgent(config=build_config())
        agent.client = FakeOpenAIClient(runner)
        agent.amazon_adapter = MockAmazonProvider("Amazon", self.amazon_products)

        response = asyncio.run(
            agent.run_search(
                "good backpack",
                RankingDesign.DIRECT_JSON,
                progress=progress_messages.append,
                ask_user=lambda _: {
                    "answer": "Mid-range",
                    "selected_choice_id": "mid_range",
                    "selected_choice_label": "Mid-range",
                    "source": "suggested_choice",
                },
            )
        )

        self.assertEqual(response.profile.clarified_answers["budget"], "Mid-range")
        self.assertTrue(
            any(
                '"status_message": "RetrievalAgent collected a usable backpack pool."' in item
                for item in progress_messages
            )
        )
        self.assertIn("[success] RetrievalAgent has completed", progress_messages)


if __name__ == "__main__":
    unittest.main()
