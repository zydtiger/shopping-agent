from __future__ import annotations

import webbrowser
from datetime import datetime

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

from ..agent import AgentHarnessError, ShoppingAgent, build_default_agent
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
        design: RankingDesign = RankingDesign.DIRECT_JSON,
        agent: ShoppingAgent | None = None,
    ) -> None:
        super().__init__()
        self.agent = agent or build_default_agent()
        self.selected_design = design
        self.pending_query = ""
        self.selected_answers: dict[str, str] = {}
        self.result_urls: dict[str, str] = {}
        self.search_in_flight = False

    def compose(self) -> ComposeResult:
        with Container(id="launch-view"):
            yield Input(
                placeholder="Describe what you want to shop for and press Enter",
                id="launch-query-input",
            )

        with Vertical(id="main-view"):
            with Container(id="top-panel", classes="panel"):
                yield Static("Active Prompt", classes="panel-title")
                yield Static("No search submitted yet.", id="active-prompt")
                yield Static(
                    f"Pipeline: {self.selected_design.label}",
                    id="design-status",
                )
                with Horizontal(id="search-row"):
                    yield Input(
                        placeholder="Enter a new shopping search and press Enter",
                        id="active-query-input",
                    )
                    yield LoadingIndicator(id="loading")
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
        table.add_columns("RANK", "NAME", "PRICE", "SITE")

        self.query_one("#workspace-tabs", TabbedContent).active = "logs-tab"
        self._append_log("[plan] Waiting for the first user prompt.")

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id not in {"launch-query-input", "active-query-input"}:
            return
        query = event.value.strip()
        if not query or self.search_in_flight:
            return
        await self._run_search(query)

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

        await self._render_results(response.ranked_products)
        for note in response.debug_notes:
            self._append_log(f"[action] {note}")

        self._set_loading(False, response.status_message)
        self.query_one("#workspace-tabs", TabbedContent).active = "recommendations-tab"
        self.search_in_flight = False

    async def _ask_clarification(
        self, question: ClarificationQuestion
    ) -> dict[str, str | None]:
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

        for display_rank, item in enumerate(ranked_products):
            row_key = f"result-{display_rank}"
            table.add_row(
                str(display_rank),
                item.product.title,
                f"${item.product.price:.2f}",
                item.product.source_site,
                key=row_key,
            )
            self.result_urls[row_key] = item.product.product_url

        if not ranked_products:
            self._append_log(
                "[action] No recommendations are available because retrieval returned no products."
            )

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
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.query_one("#logs-view", RichLog).write(f"{timestamp}  {message}")


def run_app(
    design: RankingDesign = RankingDesign.DIRECT_JSON,
    agent: ShoppingAgent | None = None,
) -> None:
    app = ShoppingAgentApp(design=design, agent=agent)
    app.run()
