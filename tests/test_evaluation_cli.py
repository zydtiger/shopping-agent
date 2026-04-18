from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from conftest import build_config_yaml

from shopping_agent.evaluation import cli


class _FakeHarness:
    def __init__(self, **_: object) -> None:
        return None

    async def evaluate_jsonl(
        self,
        *,
        input_path: pathlib.Path,
        out_dir: pathlib.Path,
    ) -> list[tuple[pathlib.Path, bool]]:
        out_dir.mkdir(parents=True, exist_ok=True)
        output = out_dir / "sample-item.json"
        output.write_text('{"ok": true}\n', encoding="utf-8")
        return [(output, True)]


class EvaluationCLITests(unittest.TestCase):
    def test_cli_smoke_writes_output_in_eval_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = pathlib.Path(tmp_dir)
            config_path = tmp_path / "config.yaml"
            input_path = tmp_path / "items.jsonl"
            out_dir = tmp_path / "eval"

            config_path.write_text(build_config_yaml() + "\n", encoding="utf-8")
            input_path.write_text('{"item_summary": "sample item"}\n', encoding="utf-8")

            with patch("shopping_agent.evaluation.cli.EvaluationHarness", _FakeHarness):
                with self.assertRaises(SystemExit) as raised:
                    cli.run(
                        [
                            "--config",
                            str(config_path),
                            "--input",
                            str(input_path),
                            "--out-dir",
                            str(out_dir),
                        ]
                    )
                self.assertEqual(raised.exception.code, 0)

            self.assertTrue((out_dir / "sample-item.json").exists())


if __name__ == "__main__":
    unittest.main()
