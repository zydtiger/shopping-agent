from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from .agent import ShoppingAgent
from .config import AppConfig
from .retrieval import set_browser_headless
from .types import RankingDesign

app = typer.Typer(add_completion=False, invoke_without_command=True)


@app.callback()
def main(
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            "-c",
            exists=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Path to the YAML config file.",
        ),
    ],
    sql: Annotated[
        bool,
        typer.Option(help="Use the SQL-backed ranking path instead of direct JSON ranking."),
    ] = False,
    head: Annotated[
        bool,
        typer.Option(help="Open the browser window instead of running Playwright headless."),
    ] = False,
    limit: Annotated[
        int,
        typer.Option(
            "--limit",
            min=1,
            help="Maximum number of products each scraper should return per search.",
        ),
    ] = 10,
) -> None:
    design = RankingDesign.SQL if sql else RankingDesign.DIRECT_JSON
    set_browser_headless(not head)
    config_obj = AppConfig.from_file(config)
    agent = ShoppingAgent(config=config_obj, result_limit=limit)
    from .ui.app import run_app

    run_app(design=design, agent=agent)


def run(argv: list[str] | None = None) -> None:
    app(prog_name="shopping-agent", args=argv)


if __name__ == "__main__":
    app()
