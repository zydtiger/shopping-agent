# Shopping Agent Backbone

Terminal-first scaffold for an interactive shopping agent project. This repository includes the UI flow, domain model, mock retrieval/ranking pipelines, structured logging, and an evaluation harness outline. It intentionally excludes real LLM calls and live product scraping.

## Included

- Textual terminal UI with request entry, loading state, follow-up questions, ranked results, and open-link actions
- Shared product schema and preference-profile models
- Mock adapters for Amazon, eBay, and Shopify-style sources
- Two ranking design stubs matching the project plan:
  - direct JSON style ranking
  - RAG-style candidate retrieval plus final reranking
- JSONL logging for retrieval and ranking runs
- Evaluation harness skeleton for synthetic demand generation and system comparison

## Not Included

- Real scraping or API integration
- Real LLM prompting, embeddings, or evaluator models
- Production persistence or vector databases

## UV Workflow

```bash
uv sync
uv run shopping-agent
```

Use the alternate ranking path with:

```bash
uv run shopping-agent --rag
```

## Test

```bash
uv run python -m unittest discover -s tests
```

## Notes

- Dependencies are managed through `pyproject.toml` and resolved with `uv`.
- `uv sync` creates the local environment and installs the package plus the default `dev` group.
- If you want to avoid installing dev dependencies, run `uv sync --no-dev`.
- Direct JSON ranking is the default launch mode. Pass `--rag` to use the RAG path.

## Project Layout

```text
src/shopping_agent/
  clarification.py
  domain.py
  event_log.py
  service.py
  main.py
  evaluation/
  ranking/
  retrieval/
  ui/
tests/
```
