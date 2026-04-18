from __future__ import annotations

import asyncio
import pathlib
import sys
import unittest
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from shopping_agent.evaluation.llm_steps import EvaluationLLMSteps


class _SequenceRunner:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("runner called too many times")
        return self.responses.pop(0)


class EvaluationLLMStepsTests(unittest.TestCase):
    def test_complete_json_retries_after_invalid_json(self) -> None:
        runner = _SequenceRunner(
            [
                {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": "not json"},
                        }
                    ]
                },
                {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "role": "assistant",
                                "content": '{"compressed_prompt": "monitor"}',
                            },
                        }
                    ]
                },
            ]
        )
        steps = EvaluationLLMSteps(model_id="gpt-agent", response_runner=runner)

        compressed = asyncio.run(steps.compress_prompt("27 inch usb c monitor"))

        self.assertEqual(compressed, "monitor")
        self.assertEqual(len(runner.calls), 2)
        self.assertIn("Strict JSON schema to follow", runner.calls[0]["messages"][0]["content"])

    def test_compress_prompt_uses_fallback_when_empty(self) -> None:
        runner = _SequenceRunner(
            [
                {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "role": "assistant",
                                "content": '{"compressed_prompt": ""}',
                            },
                        }
                    ]
                }
            ]
        )
        steps = EvaluationLLMSteps(model_id="gpt-agent", response_runner=runner)

        compressed = asyncio.run(
            steps.compress_prompt(
                "Portable 24000mAh USB C power bank with 100W output for laptop charging"
            )
        )

        self.assertEqual(compressed, "Portable 24000mAh USB")


if __name__ == "__main__":
    unittest.main()
