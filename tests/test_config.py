from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest
from unittest.mock import ANY, patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from conftest import TEST_AGENT_BASE_URL, TEST_AGENT_MODEL, build_config_yaml
from typer.testing import CliRunner

from shopping_agent.agent import ShoppingAgent
from shopping_agent.config import AppConfig
from shopping_agent.main import app
from shopping_agent.types import RankingDesign


class AppConfigTests(unittest.TestCase):
    def test_from_file_loads_agent_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = pathlib.Path(temp_dir) / "config.yaml"
            config_path.write_text(build_config_yaml(), encoding="utf-8")

            config = AppConfig.from_file(config_path)

        self.assertEqual(config.agent.openai_base_url, TEST_AGENT_BASE_URL)
        self.assertEqual(config.agent.openai_model_id, TEST_AGENT_MODEL)

    def test_cli_requires_config_path(self) -> None:
        result = CliRunner().invoke(app, [])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("Missing option", result.output)

    def test_cli_uses_headless_browser_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = pathlib.Path(temp_dir) / "config.yaml"
            config_path.write_text(build_config_yaml(), encoding="utf-8")

            with (
                patch("shopping_agent.main.set_browser_headless") as set_headless,
                patch("shopping_agent.main.ShoppingAgent") as shopping_agent_cls,
                patch("shopping_agent.ui.app.run_app") as run_app,
            ):
                result = CliRunner().invoke(app, ["-c", str(config_path)])

        self.assertEqual(result.exit_code, 0, msg=result.output)
        set_headless.assert_called_once_with(True)
        shopping_agent_cls.assert_called_once_with(config=ANY, result_limit=10)
        run_app.assert_called_once()
        self.assertEqual(run_app.call_args.kwargs["design"], RankingDesign.DIRECT_JSON)
        self.assertIs(run_app.call_args.kwargs["agent"], shopping_agent_cls.return_value)

    def test_cli_head_flag_opens_visible_browser(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = pathlib.Path(temp_dir) / "config.yaml"
            config_path.write_text(build_config_yaml(), encoding="utf-8")

            with (
                patch("shopping_agent.main.set_browser_headless") as set_headless,
                patch("shopping_agent.main.ShoppingAgent"),
                patch("shopping_agent.ui.app.run_app"),
            ):
                result = CliRunner().invoke(app, ["-c", str(config_path), "--head", "--sql"])

        self.assertEqual(result.exit_code, 0, msg=result.output)
        set_headless.assert_called_once_with(False)

    def test_cli_passes_limit_to_agent(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = pathlib.Path(temp_dir) / "config.yaml"
            config_path.write_text(build_config_yaml(), encoding="utf-8")

            with (
                patch("shopping_agent.main.set_browser_headless"),
                patch("shopping_agent.main.ShoppingAgent") as shopping_agent_cls,
                patch("shopping_agent.ui.app.run_app"),
            ):
                result = CliRunner().invoke(app, ["-c", str(config_path), "--limit", "7"])

        self.assertEqual(result.exit_code, 0, msg=result.output)
        shopping_agent_cls.assert_called_once_with(config=ANY, result_limit=7)

    def test_agent_preserves_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = pathlib.Path(temp_dir) / "config.yaml"
            config_path.write_text(build_config_yaml(), encoding="utf-8")
            config = AppConfig.from_file(config_path)
            agent = ShoppingAgent(config=config)

        self.assertEqual(agent.config.agent.openai_model_id, "gpt-agent")


if __name__ == "__main__":
    unittest.main()
