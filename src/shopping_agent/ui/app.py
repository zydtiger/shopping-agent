from __future__ import annotations

import webbrowser

from textual import events, work
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    DataTable,
    Input,
    LoadingIndicator,
    RichLog,
    Static,
    TabbedContent,
    TabPane,
)
from textual.widgets.data_table import ColumnKey

from ..agent import AgentHarnessError, ShoppingAgent
from ..types import ClarificationQuestion, RankedProduct, RankingDesign


class ClarificationDialog(ModalScreen[str]):
    def __init__(self, question: ClarificationQuestion) -> None:
        super().__init__()
        self.question = question

    def compose(self) -> ComposeResult:
        with Container(id="dialog-card"):
            yield Static(self.question.prompt, classes="dialog-title")
            yield Static(self.question.reason, classes="dialog-reason")
            with Vertical(classes="dialog-options"):
                for option in self.question.options:
                    yield Button(
                        option.label,
                        id=f"dialog-choice-{option.id}",
                        classes="dialog-choice",
                    )
            yield Static("Custom answer", classes="dialog-label")
            with Horizontal(id="dialog-custom-row"):
                yield Input(
                    placeholder="Type a custom answer",
                    id="dialog-custom-input",
                )
                yield Button("Submit", id="dialog-custom-submit", variant="primary")

    def on_mount(self) -> None:
        self.query_one("#dialog-custom-input", Input).focus()

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""
        if button_id.startswith("dialog-choice-"):
            self.dismiss(button_id.removeprefix("dialog-choice-"))
            return
        if button_id == "dialog-custom-submit":
            value = self.query_one("#dialog-custom-input", Input).value.strip()
            if value:
                self.dismiss(value)

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "dialog-custom-input":
            return
        value = event.value.strip()
        if value:
            self.dismiss(value)


class ShoppingAgentApp(App[None]):
    CSS_PATH = "app.tcss"
    TITLE = "Shopping Agent"
    SUB_TITLE = "Interactive retrieval and ranking scaffold"

    def __init__(
        self,
        agent: ShoppingAgent,
        design: RankingDesign = RankingDesign.DIRECT_JSON,
    ) -> None:
        super().__init__()
        self.agent = agent
        self.selected_design = design
        self.pending_query = ""
        self.selected_answers: dict[str, str] = {}
        self.result_urls: dict[str, str] = {}
        self.search_in_flight = False
        self.recommendation_columns: dict[str, ColumnKey] = {}
        self.compact_column_widths: dict[str, int] = {
            "rank": len("RANK"),
            "price": len("PRICE"),
            "site": len("SITE"),
        }

    def compose(self) -> ComposeResult:
        with Container(id="launch-view"):
            yield Input(
                placeholder="Describe what you want to shop for and press Enter",
                id="launch-query-input",
            )

        with Vertical(id="main-view"):
            with Container(id="search-panel", classes="panel"):
                with Horizontal(id="search-row"):
                    yield Input(
                        placeholder="Enter a new shopping search and press Enter",
                        id="active-query-input",
                    )
                    yield LoadingIndicator(id="loading")

            with Container(id="status-panel", classes="panel"):
                yield Static("Active Prompt", classes="panel-title")
                yield Static("No search submitted yet.", id="active-prompt")
                yield Static(
                    f"Pipeline: {self.selected_design.label}",
                    id="design-status",
                )
                yield Static("Ready for a shopping query.", id="status-copy")

            with Container(id="bottom-panel", classes="panel"):
                with TabbedContent(id="workspace-tabs"):
                    with TabPane("Recommendations", id="recommendations-tab"):
                        yield DataTable(id="recommendations-table")
                        yield Static(
                            "Select a recommendation row to open the listing in your browser.",
                            id="table-hint",
                        )
                    with TabPane("Logs", id="logs-tab"):
                        yield RichLog(
                            id="logs-view",
                            markup=False,
                            highlight=False,
                            wrap=True,
                        )

    def on_mount(self) -> None:
        self.query_one("#main-view").display = False
        self.query_one("#loading").display = False
        self.query_one("#launch-query-input", Input).focus()

        table = self.query_one("#recommendations-table", DataTable)
        table.cursor_type = "row"
        table.zebra_stripes = True
        self.recommendation_columns = {
            "rank": table.add_column("RANK", key="rank", width=len("RANK")),
            "name": table.add_column("NAME", key="name", width=len("NAME")),
            "price": table.add_column("PRICE", key="price", width=len("PRICE")),
            "site": table.add_column("SITE", key="site", width=len("SITE")),
        }
        self.call_after_refresh(self._resize_recommendations_columns)

        self.query_one("#workspace-tabs", TabbedContent).active = "logs-tab"
        self._append_log("[plan] Waiting for the first user prompt.")

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id not in {"launch-query-input", "active-query-input"}:
            return
        query = event.value.strip()
        if not query or self.search_in_flight:
            return
        self.search_in_flight = True
        self._run_search(query)

    async def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        row_key = str(event.row_key.value)
        url = self.result_urls.get(row_key)
        if not url:
            return
        webbrowser.open(url)
        self.query_one("#status-copy", Static).update(
            "Opened the selected product in the default browser."
        )
        self._append_log("[action] Opened the selected recommendation in the browser.")

    @work(exclusive=True)
    async def _run_search(self, query: str) -> None:
        self.search_in_flight = True
        self.pending_query = query
        self.selected_answers = {}

        self._enter_main_layout(query)
        self._reset_search_views()
        self._append_log(f"[action] Received user prompt: {query}")
        self._set_loading(True, "Analyzing prompt and preparing the agent workflow...")

        try:
            response = await self.agent.run_search(
                query,
                self.selected_design,
                progress=self._handle_progress,
                ask_user=self._ask_clarification,
            )
        except AgentHarnessError as exc:
            self._append_log(f"[action] Agent harness failed: {exc}")
            self._set_loading(False, str(exc))
            self.search_in_flight = False
            return
        except Exception as exc:
            self._append_log(f"[action] Unexpected search failure: {exc}")
            self._set_loading(False, "The search failed before completion.")
            self.search_in_flight = False
            return

        await self._render_results(response.ranked_products)
        for note in response.debug_notes:
            self._append_log(f"[action] {note}")

        self._set_loading(False, response.status_message)
        self.query_one("#workspace-tabs", TabbedContent).active = "recommendations-tab"
        self.search_in_flight = False

    async def _ask_clarification(self, question: ClarificationQuestion) -> dict[str, str | None]:
        self._set_loading(False, "Waiting for clarification input...")
        self._append_log(f"[action] Opening clarification popup: {question.prompt}")
        selected_value = await self.push_screen_wait(ClarificationDialog(question))
        if not selected_value:
            selected_value = ""
        selected_option = next(
            (option for option in question.options if option.id == selected_value),
            None,
        )
        if selected_option is None:
            answer = selected_value.strip()
            source = "custom"
        else:
            answer = selected_option.label
            source = "suggested_choice"
        self.selected_answers[question.id] = answer
        self._set_loading(True, "Clarification received. Resuming the agent...")
        return {
            "answer": answer,
            "selected_choice_id": selected_option.id if selected_option else None,
            "selected_choice_label": selected_option.label if selected_option else None,
            "source": source,
        }

    async def _handle_progress(self, message: str) -> None:
        self._append_log(message)

    async def _render_results(self, ranked_products: list[RankedProduct]) -> None:
        table = self.query_one("#recommendations-table", DataTable)
        table.clear(columns=False)
        self.result_urls = {}

        top_results = sorted(ranked_products, key=lambda item: item.score, reverse=True)[:10]
        rank_values = [str(index) for index in range(len(top_results))]
        price_values = [f"${item.product.price:.2f}" for item in top_results]
        site_values = [item.product.source_site for item in top_results]

        self.compact_column_widths = {
            "rank": max([len("RANK"), *[len(value) for value in rank_values]]),
            "price": max([len("PRICE"), *[len(value) for value in price_values]]),
            "site": max([len("SITE"), *[len(value) for value in site_values]]),
        }

        for display_rank, item in enumerate(top_results):
            row_key = f"result-{display_rank}"
            table.add_row(
                str(display_rank),
                item.product.title,
                f"${item.product.price:.2f}",
                item.product.source_site,
                key=row_key,
            )
            self.result_urls[row_key] = item.product.product_url

        if not top_results:
            self._append_log(
                "[action] No recommendations are available because retrieval returned no products."
            )

        self._resize_recommendations_columns()
        self.call_after_refresh(self._resize_recommendations_columns)

    def _reset_search_views(self) -> None:
        self.query_one("#workspace-tabs", TabbedContent).active = "logs-tab"
        self.query_one("#logs-view", RichLog).clear()
        self.query_one("#recommendations-table", DataTable).clear(columns=False)
        self.result_urls = {}

    def _enter_main_layout(self, query: str) -> None:
        self.query_one("#launch-view").display = False
        self.query_one("#main-view").display = True
        self.query_one("#active-prompt", Static).update(query)
        self.query_one("#active-query-input", Input).value = query

    def _set_loading(self, is_loading: bool, status_message: str) -> None:
        self.query_one("#loading").display = is_loading
        self.query_one("#launch-query-input", Input).disabled = is_loading
        self.query_one("#active-query-input", Input).disabled = is_loading
        self.query_one("#status-copy", Static).update(status_message)

    def _append_log(self, message: str) -> None:
        self.query_one("#logs-view", RichLog).write(message)

    def on_resize(self, _: events.Resize) -> None:
        if not self.recommendation_columns:
            return
        self.call_after_refresh(self._resize_recommendations_columns)

    def _resize_recommendations_columns(self) -> None:
        table = self.query_one("#recommendations-table", DataTable)
        if not self.recommendation_columns:
            return

        viewport_width = table.scrollable_content_region.width or table.content_region.width
        if viewport_width <= 0:
            viewport_width = table.size.width
        if viewport_width <= 0:
            return

        cell_padding_width = table.cell_padding * 2
        compact_render_width = 0
        for column_name in ("rank", "price", "site"):
            column_key = self.recommendation_columns[column_name]
            column = table.columns[column_key]
            column.width = self.compact_column_widths[column_name]
            column.auto_width = False
            compact_render_width += column.width + cell_padding_width

        remaining_for_name = max(1, viewport_width - compact_render_width - cell_padding_width)
        name_column = table.columns[self.recommendation_columns["name"]]
        name_column.width = remaining_for_name
        name_column.auto_width = False

        table._require_update_dimensions = True
        table.check_idle()


def run_app(
    agent: ShoppingAgent,
    design: RankingDesign = RankingDesign.DIRECT_JSON,
) -> None:
    app = ShoppingAgentApp(design=design, agent=agent)
    app.run()
