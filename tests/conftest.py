from __future__ import annotations

import json
from dataclasses import dataclass
from types import SimpleNamespace

from shopping_agent.config import AppConfig, OpenAIConfig
from shopping_agent.types import Product

TEST_AGENT_BASE_URL = "https://agent.example.com/v1"
TEST_AGENT_MODEL = "gpt-agent"


@dataclass(slots=True)
class MockAmazonProvider:
    source_name: str
    products: list[Product]

    async def search(self, query: str, limit: int = 50) -> list[Product]:
        return self.products[:limit]


class SequenceRunner:
    def __init__(self, responses: list[dict[str, object]]) -> None:
        self.responses = responses
        self.calls: list[dict[str, object]] = []

    async def __call__(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("Runner was called more times than expected.")
        return self.responses.pop(0)


class FakeOpenAIClient:
    def __init__(self, runner: SequenceRunner) -> None:
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(
                create=runner,
            )
        )


def build_config() -> AppConfig:
    return AppConfig(
        agent=OpenAIConfig(
            openai_base_url=TEST_AGENT_BASE_URL,
            openai_model_id=TEST_AGENT_MODEL,
            openai_api_key="agent-key",
        ),
    )


def build_config_yaml() -> str:
    return "\n".join(
        [
            "agent:",
            f'  openai_base_url: "{TEST_AGENT_BASE_URL}"',
            f'  openai_model_id: "{TEST_AGENT_MODEL}"',
            '  openai_api_key: "agent-key"',
        ]
    )


def _amazon_product(title: str, price: float, product_url: str, rating: float | None) -> Product:
    return Product(
        title=title,
        price=price,
        source_site="Amazon",
        product_url=product_url,
        rating=rating,
    )


def build_amazon_products() -> list[Product]:
    return [
        _amazon_product(
            title="Atlas Travel Backpack 35L",
            price=129.99,
            product_url="https://example.com/amazon/backpack-travel",
            rating=4.8,
        ),
        _amazon_product(
            title="Atlas Daily Carry Backpack",
            price=74.0,
            product_url="https://example.com/amazon/backpack-daily",
            rating=4.4,
        ),
    ]


def build_mock_amazon_catalog() -> list[Product]:
    return [
        _amazon_product(
            title="Atlas Commuter Laptop Backpack",
            price=89.99,
            product_url="https://example.com/amazon/backpack-commuter",
            rating=4.6,
        ),
        _amazon_product(
            title="Atlas Travel Backpack 35L",
            price=129.99,
            product_url="https://example.com/amazon/backpack-travel",
            rating=4.8,
        ),
        _amazon_product(
            title="North Coast Leather Wallet",
            price=54.5,
            product_url="https://example.com/amazon/wallet-leather",
            rating=4.5,
        ),
        _amazon_product(
            title="Trail Peak Insulated Water Bottle",
            price=28.0,
            product_url="https://example.com/amazon/bottle-insulated",
            rating=4.7,
        ),
        _amazon_product(
            title="Studio Ergonomic Desk Chair",
            price=239.0,
            product_url="https://example.com/amazon/chair-ergonomic",
            rating=4.4,
        ),
        _amazon_product(
            title="Harbor Waxed Canvas Messenger Bag",
            price=118.0,
            product_url="https://example.com/amazon/bag-canvas",
            rating=4.3,
        ),
        _amazon_product(
            title="Arbor Solid Wood Writing Desk",
            price=349.0,
            product_url="https://example.com/amazon/desk-wood",
            rating=4.5,
        ),
        _amazon_product(
            title="Fjord Lightweight Rain Jacket",
            price=142.0,
            product_url="https://example.com/amazon/jacket-rain",
            rating=4.6,
        ),
        _amazon_product(
            title="Atlas Daily Carry Backpack",
            price=74.0,
            product_url="https://example.com/amazon/backpack-daily",
            rating=4.4,
        ),
        _amazon_product(
            title="Everfield Premium Travel Backpack",
            price=189.0,
            product_url="https://example.com/amazon/backpack-premium",
            rating=4.9,
        ),
        _amazon_product(
            title="Cedar Minimal Desk Lamp",
            price=67.0,
            product_url="https://example.com/amazon/lamp-desk",
            rating=4.2,
        ),
        _amazon_product(
            title="Harbor Packable Duffel",
            price=39.0,
            product_url="https://example.com/amazon/duffel-packable",
            rating=4.1,
        ),
    ]


def build_two_agent_direct_json_responses() -> list[dict[str, object]]:
    return [
        {
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": "[plan] The request is still ambiguous on budget.",
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {
                                    "name": "ask_clarification",
                                    "arguments": json.dumps(
                                        {
                                            "question": "What budget tier fits best?",
                                            "suggested_choices": [
                                                {
                                                    "id": "budget",
                                                    "label": "Budget",
                                                    "description": "Keep cost low.",
                                                },
                                                {
                                                    "id": "mid_range",
                                                    "label": "Mid-range",
                                                    "description": "Balance cost and quality.",
                                                },
                                                {
                                                    "id": "premium",
                                                    "label": "Premium",
                                                    "description": "Prioritize build quality.",
                                                },
                                            ],
                                            "preference_dimension": "budget",
                                        }
                                    ),
                                },
                            }
                        ],
                    },
                }
            ]
        },
        {
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": "[action] Clarification received. Searching Amazon next.",
                        "tool_calls": [
                            {
                                "id": "call-2",
                                "type": "function",
                                "function": {
                                    "name": "search_amazon",
                                    "arguments": json.dumps(
                                        {"query": "durable travel laptop backpack"}
                                    ),
                                },
                            }
                        ],
                    },
                }
            ]
        },
        {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(
                            {
                                "status_message": (
                                    "RetrievalAgent collected a usable backpack pool."
                                ),
                                "profile": {
                                    "raw_query": "good backpack",
                                    "clarified_answers": {"budget": "Mid-range"},
                                    "inferred_requirements": [
                                        "durable build",
                                        "travel-friendly",
                                        "laptop protection",
                                    ],
                                    "uncertainty_notes": [],
                                },
                                "debug_notes": [
                                    "RetrievalAgent stopped after the first "
                                    "successful Amazon search."
                                ],
                            }
                        ),
                    },
                }
            ]
        },
        {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(
                            {
                                "status_message": (
                                    "RankingAgent prepared two backpack recommendations."
                                ),
                                "recommendations": [
                                    {
                                        "score": 91,
                                        "rationale": (
                                            "Best travel-focused fit for the clarified brief."
                                        ),
                                        "product": build_amazon_products()[0].to_dict(),
                                    },
                                    {
                                        "score": 78,
                                        "rationale": "Good value fallback with weaker travel cues.",
                                        "product": build_amazon_products()[1].to_dict(),
                                    },
                                ],
                                "debug_notes": [
                                    "RankingAgent used direct JSON injection over the shared store."
                                ],
                            }
                        ),
                    },
                }
            ]
        },
    ]


def build_two_agent_sql_responses() -> list[dict[str, object]]:
    catalog = build_mock_amazon_catalog()
    return [
        {
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": "[action] Searching Amazon with the raw backpack request.",
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {
                                    "name": "search_amazon",
                                    "arguments": json.dumps({"query": "travel backpack"}),
                                },
                            }
                        ],
                    },
                }
            ]
        },
        {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(
                            {
                                "status_message": (
                                    "RetrievalAgent prepared a backpack-oriented product pool."
                                ),
                                "profile": {
                                    "raw_query": "travel backpack",
                                    "clarified_answers": {},
                                    "inferred_requirements": ["travel use"],
                                    "uncertainty_notes": [],
                                },
                                "debug_notes": [],
                            }
                        ),
                    },
                }
            ]
        },
        {
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": "[plan] Querying the SQL product store for backpack candidates.",
                        "tool_calls": [
                            {
                                "id": "call-2",
                                "type": "function",
                                "function": {
                                    "name": "query_product_store",
                                    "arguments": json.dumps(
                                        {
                                            "sql": (
                                                "SELECT title, price, "
                                                "source_site, product_url, "
                                                "rating "
                                                "FROM products "
                                                "WHERE lower(title) LIKE '%backpack%' "
                                                "ORDER BY rating DESC, price ASC "
                                                "LIMIT 5"
                                            )
                                        }
                                    ),
                                },
                            }
                        ],
                    },
                }
            ]
        },
        {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(
                            {
                                "status_message": (
                                    "RankingAgent prepared a shortlist from the SQL store."
                                ),
                                "recommendations": [
                                    {
                                        "score": 94,
                                        "rationale": "Strong travel alignment plus top rating.",
                                        "product": catalog[1].to_dict(),
                                    },
                                    {
                                        "score": 88,
                                        "rationale": "Balanced commuter-friendly fallback.",
                                        "product": catalog[0].to_dict(),
                                    },
                                ],
                                "debug_notes": [
                                    "RankingAgent used query_product_store to inspect candidates."
                                ],
                            }
                        ),
                    },
                }
            ]
        },
    ]
