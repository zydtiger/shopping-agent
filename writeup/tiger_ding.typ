= Shopping Agent Backbone: Terminal Multi-Agent Product Retrieval and Ranking

Team members: Tiger Ding \
Section: 466 \
Email: zding27\@jh.edu \
Project summary: I built a terminal-first shopping agent that takes vague user requests, asks clarifying questions through tool calls, retrieves products from multiple e-commerce sources, and returns a ranked top-10 list. The system compares two ranking designs: direct JSON context injection and SQL-backed candidate querying. \
Repository URL: #underline[https://github.com/zydtiger/shopping-agent.git]

= How to run the project

Prerequisites:
- Python 3.11+
- `uv` package manager
- OpenAI-compatible API endpoint and key

Setup:
- Copy `config.example.yaml` to `config.yaml`
- Fill in:
  - `agent.openai_base_url`
  - `agent.openai_model_id`
  - `agent.openai_api_key`

Install and run:
```bash
uv sync
uv run shopping-agent --config config.yaml
```

Useful options:
- Limit per-source retrieval volume:
```bash
uv run shopping-agent --config config.yaml --limit 10
```
- Switch ranking design to SQL-backed mode:
```bash
uv run shopping-agent --config config.yaml --sql
```
- Use visible browser (non-headless):
```bash
uv run shopping-agent --config config.yaml --head
```

Evaluation commands:
```bash
uv run shopping-agent-eval --config config.yaml --input eval.jsonl --out-dir ./eval
uv run shopping-agent-eval-vis --input-dir ./eval --pretty
```

Tests:
```bash
uv run pytest
```

= Project achievements, strengths, and scope

- Built a complete Textual TUI workflow:
  - launch screen with centered query input
  - transition to main layout with top query box and bottom tabbed area
  - `Recommendations` tab with rank / name / price / site columns
  - `Logs` tab for retrieval and ranking lifecycle traces
  - browser-openable product rows

- Implemented a two-agent architecture with clear responsibility boundaries:
  - `RetrievalAgent`: clarification and source-specific search planning
  - `RankingAgent`: final product scoring and top-10 generation

- Enforced tool-first control flow rather than hard-coded UI questioning:
  - clarification UI is driven by runtime `ask_clarification` tool calls
  - retrieval tools are registered and executed by the harness

- Integrated multiple retrieval sources with one canonical schema:
  - Amazon, eBay, and Newegg adapters
  - normalized `Product` fields shared across tools, in-memory store, ranking input, and SQL rows

- Implemented and compared two ranking designs:
  - direct JSON payload injection
  - in-memory SQLite filtering/querying path

- Added an evaluation harness:
  - generates hidden detailed demand drafts from concise prompts
  - runs the interactive agent flow
  - scores top recommendations with an LLM judge
  - writes reproducible JSON artifacts

Complexity and success assessment:
- Engineering complexity: medium-high (multi-agent orchestration + TUI + retrieval adapters + evaluation pipeline)
- Scope completion: high for the planned backbone architecture
- Stability: strong in both modes; direct JSON is more straightforward operationally because the ranking agent only needs to produce one final response

= Limitations

- Scraping reliability and anti-bot defenses vary by source; some searches can fail or return sparse/noisy listings.
- Product metadata quality is uneven across sites (e.g., missing ratings on eBay/Newegg).
- Ranking quality is still tied to LLM heuristic judgment and prompt behavior; deterministic reranking is limited.
- Evaluation judge scores are useful but still noisy and model-dependent.

= Worthwhile extensions

- Add more stable source connectors (official APIs where possible) and retry/backoff policies.
- Improve deterministic pre-ranking (hard filters + feature extraction) before final LLM scoring.
- Further improve SQL mode with stronger schema constraints and query optimization.
- Add richer product signals (shipping, return policy, review count, seller quality).
- Expand evaluation to larger benchmark sets and calibrated multi-judge agreement.

= Methods and architecture

User interaction flow:
1. User enters a vague shopping query in the terminal.
2. `RetrievalAgent` decides whether clarification is needed.
3. If needed, it calls `ask_clarification` with 3--5 choices + freeform fallback.
4. Harness executes retrieval tools (`search_amazon`, `search_ebay`, `search_newegg`) and stores normalized `Product` objects.
5. Retrieval agent receives only compact execution status strings, not raw product payloads.
6. Harness invokes `RankingAgent` with either:
  - full JSON product list (direct JSON design), or
  - SQL query interface over in-memory SQLite (SQL design).
7. `RankingAgent` returns strict top-10 JSON objects of shape `{ "score": int, "product": Product }`.

#figure(
  image("./user_flow.png", height: 7in),
  caption: [Shopping agent user interaction and multi-agent/tool orchestration flow (`./user_flow.png`).],
)

= Empirical evaluation

Evaluation setup used in this project:
- Input: a JSONL list of shopping intents (`eval.jsonl`)
- Process:
  - produce detailed hidden demand draft
  - compress to a fuzzy short user prompt
  - run full shopping-agent pipeline
  - judge recommendation relevance against hidden detailed demand
- Outputs: per-case JSON artifacts in `eval-json/` (direct JSON design) and `eval-sql/` (SQL design)

#figure(
  image("./llm_eval.png", height: 5in),
  caption: [LLM-based evaluation workflow from concise prompt generation to judged recommendation outputs (`./llm_eval.png`).],
)

Observed quantitative summary from current artifacts:

#table(
  columns: 5,
  align: center + horizon,
  stroke: 0.5pt,
  inset: 6pt,
  [Mode], [Success], [Avg score\@1], [Avg score\@5], [Avg score\@10],
  [Direct JSON (`eval-json/`)], [10/10], [68.8], [77.8], [81.5],
  [SQL (`eval-sql/`)], [10/10], [82.1], [83.9], [85.9],
)

#table(
  columns: 2,
  align: center + horizon,
  stroke: 0.5pt,
  inset: 6pt,
  [Mode], [Avg total tokens],
  [Direct JSON (`eval-json/`)], [18,417.5],
  [SQL (`eval-sql/`)], [23,561.0],
)

#figure(
  image("./eval-json.png", width: 80%),
  caption: [Direct JSON evaluation plot (`./eval-json.png`).],
)

#figure(
  image("./eval-sql.png", width: 80%),
  caption: [SQL evaluation plot (`./eval-sql.png`).],
)

Interpretation:
- In this run, SQL outperforms Direct JSON on average ranking quality at \@1/\@5/\@10 (+13.3 / +6.1 / +4.4 points).
- Direct JSON is more token-efficient, using about 5,143.5 fewer tokens per task on average.
- Both modes completed all 10/10 cases in this snapshot.
- Direct JSON shows a clearer upward trend as `k` increases, suggesting relevant items are often present but not always placed at the very top initially.
- SQL curves are comparatively flatter across `score\@k`, which indicates the best guess is usually already captured at \@1 and later ranks add less incremental gain.
- Error distribution is mode-specific: SQL is very strong on most tasks but weak on `smart-desk-lamp` (55.0 at score\@10), while Direct JSON is weaker on `acoustic-privacy-panel` (60.0 at score\@10).

= Outside components and reuse disclosure

- Third-party libraries are used normally (Textual, Typer, OpenAI SDK, Playwright, SQLite, etc.).
- No teammate-external custom project code was copied into this repository beyond standard dependency usage.
- This submission represents original project implementation for this course.
