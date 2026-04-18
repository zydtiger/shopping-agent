from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from shopping_agent.evaluation.io import build_output_path, make_summary_slug, read_jsonl_cases


class EvaluationIOTests(unittest.TestCase):
    def test_read_jsonl_cases_supports_aliases_and_plain_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = pathlib.Path(tmp_dir) / "input.jsonl"
            path.write_text(
                "\n".join(
                    [
                        '{"item_summary": "ultralight hiking tent"}',
                        '{"summary": "wireless mouse"}',
                        '{"query": "desk lamp"}',
                        "running shoes",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            cases = read_jsonl_cases(path)

        self.assertEqual(len(cases), 4)
        self.assertEqual(cases[0].item_summary, "ultralight hiking tent")
        self.assertEqual(cases[1].item_summary, "wireless mouse")
        self.assertEqual(cases[2].item_summary, "desk lamp")
        self.assertEqual(cases[3].item_summary, "running shoes")

    def test_slug_and_output_path_collision_handling(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = pathlib.Path(tmp_dir) / "eval"
            out_dir.mkdir(parents=True, exist_ok=True)
            first = out_dir / "gaming-laptop.json"
            first.write_text("{}\n", encoding="utf-8")

            reserved = {out_dir / "gaming-laptop-2.json"}
            output = build_output_path(
                out_dir=out_dir,
                prompt="Gaming Laptop",
                reserved_paths=reserved,
            )

        self.assertEqual(make_summary_slug("  Gaming   Laptop!! "), "gaming-laptop")
        self.assertTrue(output.name.endswith("gaming-laptop-3.json"))


if __name__ == "__main__":
    unittest.main()
