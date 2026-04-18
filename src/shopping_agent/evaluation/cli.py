from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

import typer

from ..config import AppConfig
from ..retrieval import set_browser_headless
from ..types import RankingDesign
from .harness import EvaluationHarness

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
    input_path: Annotated[
        Path,
        typer.Option(
            "--input",
            "-i",
            exists=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Path to a JSONL file with item_summary rows.",
        ),
    ],
    design: Annotated[
        RankingDesign,
        typer.Option(help="Ranking design to run for shopping-agent retrieval+ranking."),
    ] = RankingDesign.DIRECT_JSON,
    limit: Annotated[
        int,
        typer.Option(min=1, help="Result limit per source search call."),
    ] = 10,
    head: Annotated[
        bool,
        typer.Option(help="Open the browser window instead of running Playwright headless."),
    ] = False,
    out_dir: Annotated[
        Path,
        typer.Option(help="Directory where evaluation JSON artifacts are written."),
    ] = Path("./eval"),
) -> None:
    set_browser_headless(not head)
    config_obj = AppConfig.from_file(config)
    harness = EvaluationHarness(
        config=config_obj,
        design=design,
        result_limit=limit,
    )

    outcomes = asyncio.run(
        harness.evaluate_jsonl(
            input_path=input_path,
            out_dir=out_dir,
        )
    )

    success_count = sum(1 for _, success in outcomes if success)
    failure_count = len(outcomes) - success_count

    typer.echo(f"Evaluated {len(outcomes)} case(s).")
    typer.echo(f"Succeeded: {success_count}")
    typer.echo(f"Failed: {failure_count}")
    for path, success in outcomes:
        status = "ok" if success else "error"
        typer.echo(f"- [{status}] {path}")


def run(argv: list[str] | None = None) -> None:
    app(prog_name="shopping-agent-eval", args=argv)


if __name__ == "__main__":
    app()
