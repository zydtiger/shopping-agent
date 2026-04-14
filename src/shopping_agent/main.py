from __future__ import annotations

import argparse
from collections.abc import Sequence

from shopping_agent.domain import RankingDesign


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="shopping-agent",
        description="Terminal shopping agent scaffold.",
    )
    parser.add_argument(
        "--rag",
        action="store_true",
        help="Use the RAG candidate-retrieval ranking path instead of direct JSON ranking.",
    )
    return parser.parse_args(argv)


def run(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    design = RankingDesign.RAG if args.rag else RankingDesign.DIRECT_JSON
    from shopping_agent.ui.app import run_app

    run_app(design=design)


if __name__ == "__main__":
    run()
