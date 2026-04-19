from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import matplotlib.pyplot as plt
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


def _plot_scores(
    k_values: list[int], series: list[tuple[Path, int, list[float]]], output_path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for file_path, total_tokens, scores in series:
        label = f"{file_path.stem.replace('-', ' ')} (tokens={total_tokens})"
        ax.plot(k_values, scores, marker="o", linewidth=2, label=label)

    ax.set_xlabel("K")
    ax.set_ylabel("Score@K")
    ax.set_ylim(0, 100)
    ax.set_title("Evaluation Scores by File")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.show()
    plt.close(fig)


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
    plot: Annotated[
        Path | None,
        typer.Option(
            "--plot",
            resolve_path=True,
            help="Save a matplotlib line chart for score@k to this file path.",
        ),
    ] = None,
) -> None:
    k_values = list(range(1, 11))
    json_paths = sorted(input_dir.glob("*.json"))
    if not json_paths:
        typer.echo(f"No .json files found in {input_dir}")
        raise typer.Exit(code=1)

    header = ["file"] + [f"score@{k}" for k in k_values] + ["total_tokens"]
    rows: list[list[str]] = [header]
    plot_series: list[tuple[Path, int, list[float]]] = []

    for json_path in json_paths:
        with json_path.open("r", encoding="utf-8") as f:
            json_obj = json.load(f)

        recommendations = json_obj.get("final_recommendations", [])
        scores = [_score_at_k(recommendations, k) for k in k_values]
        total_tokens = int(json_obj.get("shopping_agent_run", {}).get("total_tokens", 0))
        rows.append([json_path.name, *[str(score) for score in scores], str(total_tokens)])
        plot_series.append((json_path, total_tokens, scores))

    if pretty:
        _print_pretty_table(rows)
    else:
        typer.echo("\t".join(header))
        for row in rows[1:]:
            typer.echo("\t".join(row))

    if plot:
        _plot_scores(k_values, plot_series, plot)


def run(argv: list[str] | None = None) -> None:
    app(prog_name="shopping-agent-eval-vis", args=argv)


if __name__ == "__main__":
    app()
