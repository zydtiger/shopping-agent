from __future__ import annotations

import asyncio
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from shopping_agent.agent.tools import run_search_tool


class _FailingAdapter:
    source_name = "Amazon"

    async def search(self, query: str, limit: int = 50) -> list[object]:
        raise TimeoutError("synthetic timeout")


class RetrievalToolsResilienceTests(unittest.TestCase):
    def test_run_search_tool_returns_source_error_instead_of_raising(self) -> None:
        logs: list[str] = []
        retrieved_products: dict[str, object] = {}
        retrieval_batches: list[object] = []

        result = asyncio.run(
            run_search_tool(
                adapter=_FailingAdapter(),
                query="desk frame",
                tool_name="search_amazon",
                result_limit=10,
                emit_progress=lambda progress, message: _emit(progress, message),
                progress=logs.append,
                retrieved_products=retrieved_products,
                retrieval_batches=retrieval_batches,
            )
        )

        self.assertEqual(result["status"], "source_error")
        self.assertEqual(result["result_count"], 0)
        self.assertEqual(len(retrieval_batches), 0)
        self.assertTrue(any("failed" in message for message in logs))


async def _emit(progress: object, message: str) -> None:
    if progress is None:
        return
    progress(message)


if __name__ == "__main__":
    unittest.main()
