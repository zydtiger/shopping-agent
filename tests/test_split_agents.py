from __future__ import annotations

import asyncio
import json
import pathlib
import sys
import unittest

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

from shopping_agent.agent.errors import AgentHarnessError
from shopping_agent.agent.ranking_agent import RankingAgent
from shopping_agent.agent.retrieval_agent import RetrievalAgent
from shopping_agent.agent.shopping_agent import ShoppingAgent
from shopping_agent.agent.sql import ProductSQLStore
from shopping_agent.types import RankingDesign, UserPreferenceProfile


class SplitAgentTests(unittest.TestCase):
    def test_retrieval_agent_handles_clarification_search_and_profile(self) -> None:
        runner = SequenceRunner(build_two_agent_direct_json_responses()[:3])
        store = ProductSQLStore()
        retrieval_agent = RetrievalAgent(
            model_id="gpt-agent",
            result_limit=50,
            response_runner=runner,
            amazon_adapter=MockAmazonProvider("Amazon", build_amazon_products()),
        )

        outcome = asyncio.run(
            retrieval_agent.run(
                query="good backpack",
                ask_user=lambda _: {
                    "answer": "Mid-range",
                    "selected_choice_id": "mid_range",
                    "selected_choice_label": "Mid-range",
                    "source": "suggested_choice",
                },
                progress=None,
                product_store=store,
            )
        )

        self.assertEqual(outcome.profile.clarified_answers["budget"], "Mid-range")
        self.assertEqual(len(outcome.retrieval_batches), 1)
        self.assertEqual(len(store.all_products()), 2)
        self.assertIn("usable backpack pool", outcome.status_message)

    def test_ranking_agent_direct_json_rejects_tool_calls(self) -> None:
        runner = SequenceRunner(
            [
                {
                    "choices": [
                        {
                            "finish_reason": "tool_calls",
                            "message": {
                                "role": "assistant",
                                "content": "[plan] attempting tool call",
                                "tool_calls": [
                                    {
                                        "id": "call-1",
                                        "type": "function",
                                        "function": {
                                            "name": "query_product_store",
                                            "arguments": json.dumps(
                                                {"sql": "SELECT * FROM products"}
                                            ),
                                        },
                                    }
                                ],
                            },
                        }
                    ]
                }
            ]
        )
        store = ProductSQLStore()
        store.add_products(build_amazon_products())
        ranking_agent = RankingAgent(model_id="gpt-agent", response_runner=runner)

        with self.assertRaisesRegex(AgentHarnessError, "must not request tools"):
            asyncio.run(
                ranking_agent.run(
                    design=RankingDesign.DIRECT_JSON,
                    profile=UserPreferenceProfile(raw_query="good backpack"),
                    product_store=store,
                    progress=None,
                )
            )

    def test_ranking_agent_sql_executes_store_query_and_parses_results(self) -> None:
        runner = SequenceRunner(build_two_agent_sql_responses()[2:])
        store = ProductSQLStore()
        store.add_products(build_mock_amazon_catalog())
        ranking_agent = RankingAgent(model_id="gpt-agent", response_runner=runner)

        outcome = asyncio.run(
            ranking_agent.run(
                design=RankingDesign.SQL,
                profile=UserPreferenceProfile(raw_query="travel backpack"),
                product_store=store,
                progress=None,
            )
        )

        self.assertEqual(len(outcome.ranked_products), 2)
        self.assertGreaterEqual(outcome.ranked_products[0].score, outcome.ranked_products[1].score)
        self.assertEqual(runner.calls[0]["tools"][0]["function"]["name"], "query_product_store")

    def test_shopping_agent_short_circuits_when_retrieval_returns_no_products(self) -> None:
        runner = SequenceRunner(
            [
                {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "role": "assistant",
                                "content": json.dumps(
                                    {
                                        "status_message": "No products available.",
                                        "profile": {
                                            "raw_query": "rare specialty item",
                                            "clarified_answers": {},
                                            "inferred_requirements": ["rare"],
                                            "uncertainty_notes": ["No sources queried"],
                                        },
                                        "debug_notes": ["Stopped early."],
                                    }
                                ),
                            },
                        }
                    ]
                }
            ]
        )
        agent = ShoppingAgent(config=build_config())
        agent.client = FakeOpenAIClient(runner)

        response = asyncio.run(
            agent.run_search(
                "rare specialty item",
                RankingDesign.SQL,
                ask_user=lambda _: {"answer": "unused"},
            )
        )

        self.assertEqual(response.stage, "results")
        self.assertEqual(response.ranked_products, [])
        self.assertIn("ranking was skipped", response.status_message)
        self.assertEqual(len(runner.calls), 1)


if __name__ == "__main__":
    unittest.main()
