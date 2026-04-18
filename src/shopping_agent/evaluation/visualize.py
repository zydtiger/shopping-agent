from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(add_completion=False, invoke_without_command=True)


def _score_at_k(recommendations: list[dict[str, Any]], k: int) -> float:
    topk_scores = [float(item.get("eval_score", 0.0)) for item in recommendations[:k]]
    if not topk_scores:
        return 0.0
    return max(topk_scores)


def _print_pretty_table(rows: list[list[str]]) -> None:
    table = Table(show_header=True, header_style="bold cyan")
    for column in rows[0]:
        table.add_column(column)
    for row in rows[1:]:
        table.add_row(*row)
    Console().print(table)


@app.callback()
def main(
    input_dir: Annotated[
        Path,
        typer.Option(
            "--input-dir",
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            resolve_path=True,
            help="Directory containing evaluation JSON files.",
        ),
    ],
    pretty: Annotated[
        bool,
        typer.Option(
            "--pretty",
            help="Render output as an aligned table instead of TSV.",
        ),
    ] = False,
) -> None:
    json_paths = sorted(input_dir.glob("*.json"))
    if not json_paths:
        typer.echo(f"No .json files found in {input_dir}")
        raise typer.Exit(code=1)

    header = ["file", "score@1", "score@5", "score@10", "total_tokens"]
    rows: list[list[str]] = [header]

    for json_path in json_paths:
        with json_path.open("r", encoding="utf-8") as f:
            json_obj = json.load(f)

        recommendations = json_obj.get("final_recommendations", [])
        first_score = float(recommendations[0].get("eval_score", 0.0)) if recommendations else 0.0
        score_at_5 = _score_at_k(recommendations, 5)
        score_at_10 = _score_at_k(recommendations, 10)
        total_tokens = json_obj.get("shopping_agent_run", {}).get("total_tokens", 0)

        rows.append(
            [
                json_path.name,
                str(first_score),
                str(score_at_5),
                str(score_at_10),
                str(total_tokens),
            ]
        )

    if pretty:
        _print_pretty_table(rows)
        return

    typer.echo("\t".join(header))
    for row in rows[1:]:
        typer.echo("\t".join(row))


def run(argv: list[str] | None = None) -> None:
    app(prog_name="shopping-agent-eval-vis", args=argv)


if __name__ == "__main__":
    app()
