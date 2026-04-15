# Shopping Agent Backbone

Terminal-first scaffold for an interactive shopping agent project. This repository now centers the end-to-end flow inside an OpenAI SDK agent harness: the model receives the raw user query, decides when to clarify, calls local retrieval tools, and returns final ranked recommendations for the Textual TUI.

## Included

- Textual terminal UI with request entry, loading state, follow-up questions, ranked results, and open-link actions
- OpenAI Python SDK harness with runtime tool registration for `ask_clarification` and `search_amazon`
- Shared product schema and preference-profile models
- Retrieval-layer provider scaffolds for future marketplace adapters such as Amazon and eBay
- Two ranking design stubs matching the project plan:
  - direct JSON style ranking
  - RAG-style candidate retrieval plus final reranking
- Evaluation harness skeleton for synthetic demand generation and system comparison

## Not Included

- Real Amazon scraping or API integration
- Real embedding generation or evaluator models
- Production persistence or vector databases

## UV Workflow

```bash
uv sync
uv run shopping-agent --config config.yaml
```

Use the alternate ranking path with:

```bash
uv run shopping-agent --config config.yaml --rag
```

The config file must define the SDK connection for the main agent model and the placeholder embedding model:

```yaml
agent:
  openai_base_url: "https://api.openai.com/v1"
  openai_model_id: "gpt-4.1"
  openai_api_key: "sk-..."
embedding:
  openai_base_url: "https://api.openai.com/v1"
  openai_model_id: "text-embedding-3-large"
  openai_api_key: "sk-..."
```

## Test

```bash
uv run python -m unittest discover -s tests
```

## Notes

- Dependencies are managed through `pyproject.toml` and resolved with `uv`.
- `uv sync` creates the local environment and installs the package plus the default `dev` group.
- If you want to avoid installing dev dependencies, run `uv sync --no-dev`.
- The TUI popup flow is driven by the agent's `ask_clarification` tool call rather than a hard-coded question phase.
- Marketplace-specific search providers should be added under `src/shopping_agent/retrieval/`, not under the agent harness package.
- Direct JSON ranking is the default launch mode. Pass `--rag` to switch the agent guidance to the RAG comparison path.

## Project Layout

```text
src/shopping_agent/
  agent/
  domain.py
  main.py
  evaluation/
  ranking/
  retrieval/
  ui/
tests/
```
