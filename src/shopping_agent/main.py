from __future__ import annotations

import argparse
from collections.abc import Sequence

from .agent import ShoppingAgent
from .config import AppConfig
from .types import RankingDesign


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
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        metavar="PATH",
        help="Path to the YAML config file.",
    )
    return parser.parse_args(argv)


def run(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    design = RankingDesign.RAG if args.rag else RankingDesign.DIRECT_JSON
    config = AppConfig.from_file(args.config)
    agent = ShoppingAgent(config=config)
    from .ui.app import run_app

    run_app(design=design, agent=agent)


if __name__ == "__main__":
    run()
