from __future__ import annotations

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from shopping_agent.agent.tools import build_retrieval_tool_specs


class AgentToolWordingTests(unittest.TestCase):
    def test_clarification_tool_description_is_channel_agnostic(self) -> None:
        specs = build_retrieval_tool_specs()
        clarification = next(
            spec for spec in specs if spec["function"]["name"] == "ask_clarification"
        )
        description = clarification["function"]["description"].lower()
        choices_description = clarification["function"]["parameters"]["properties"][
            "suggested_choices"
        ]["description"].lower()

        self.assertIn("clarification question", description)
        self.assertNotIn("popup", description)
        self.assertNotIn("popup", choices_description)


if __name__ == "__main__":
    unittest.main()
