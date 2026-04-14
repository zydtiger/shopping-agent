from __future__ import annotations

import webbrowser

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Button, Footer, Header, Input, LoadingIndicator, Static

from shopping_agent.domain import ClarificationQuestion, RankedProduct, RankingDesign
from shopping_agent.service import ShoppingAgentService, build_default_service


class ClarificationCard(Static):
    def __init__(self, question: ClarificationQuestion) -> None:
        super().__init__(classes="question-card")
        self.question = question

    def compose(self) -> ComposeResult:
        yield Static(self.question.prompt, classes="question-prompt")
        yield Static(self.question.reason, classes="question-reason")
        with Horizontal(classes="option-row"):
            for option in self.question.options:
                yield Button(
                    option.label,
                    id=f"answer-{self.question.id}-{option.id}",
                    classes="answer-option",
                )


class ResultCard(Static):
    def __init__(self, item: RankedProduct) -> None:
        super().__init__(classes="result-card")
        self.item = item

    def compose(self) -> ComposeResult:
        product = self.item.product
        summary = (
            f"[#{self.item.rank}] {product.title}\n"
            f"${product.price:.2f} {product.currency}  |  {product.source_site}\n"
            f"{product.short_description}\n"
            f"Why ranked: {self.item.rationale}"
        )
        yield Static(summary, classes="result-copy")
        yield Button("Open Listing", id=f"open-{self.item.rank}", variant="primary")


class ShoppingAgentApp(App[None]):
    CSS_PATH = "app.tcss"
    TITLE = "Shopping Agent"
    SUB_TITLE = "Interactive retrieval and ranking scaffold"

    def __init__(
        self,
        design: RankingDesign = RankingDesign.DIRECT_JSON,
        service: ShoppingAgentService | None = None,
    ) -> None:
        super().__init__()
        self.service = service or build_default_service()
        self.selected_design = design
        self.pending_query = ""
        self.pending_questions: list[ClarificationQuestion] = []
        self.selected_answers: dict[str, str] = {}
        self.result_urls: dict[int, str] = {}

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Container(id="shell"):
            yield Static("Atlas Shopping Agent", id="hero")
            yield Static(
                "Clarify vague shopping requests, compare ranking designs, and inspect mock retrieval output.",
                id="subhero",
            )
            yield Static(
                f"Active pipeline: {self.selected_design.label}",
                id="design-status",
            )
            yield Static("Shopping Request", classes="section-label")
            yield Input(
                placeholder="Example: need a durable laptop backpack for weekend travel",
                id="query-input",
            )
            with Horizontal(id="search-row"):
                yield Button("Run Search", id="submit", variant="success")
                yield LoadingIndicator(id="loading")
            yield Static("Enter a request to start.", id="status-copy")
            yield Static("Clarification", classes="section-label")
            yield Vertical(id="clarification-panel")
            yield Button("Continue With Answers", id="continue", variant="primary")
            yield Static("Ranked Results", classes="section-label")
            yield Vertical(id="results-panel")
            yield Static("Recent Logs", classes="section-label")
            yield Vertical(id="log-panel")
        yield Footer()

    async def on_mount(self) -> None:
        self.query_one("#loading").display = False
        self.query_one("#continue", Button).display = False
        await self._render_logs()

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""
        if button_id == "submit":
            await self._start_search()
            return
        if button_id == "continue":
            await self._complete_search()
            return
        if button_id.startswith("answer-"):
            _, question_id, option_id = button_id.split("-", maxsplit=2)
            self.selected_answers[question_id] = option_id
            self._highlight_selected_answers()
            return
        if button_id.startswith("open-"):
            rank = int(button_id.removeprefix("open-"))
            url = self.result_urls.get(rank)
            if url:
                webbrowser.open(url)
                self.query_one("#status-copy", Static).update(
                    f"Opened result #{rank} in the default browser."
                )

    async def _start_search(self) -> None:
        query = self.query_one("#query-input", Input).value.strip()
        if not query:
            self.query_one("#status-copy", Static).update(
                "Enter a shopping request before running the agent."
            )
            return

        self.pending_query = query
        self.selected_answers = {}
        self._set_loading(True)
        response = await self.service.start_search(query, self.selected_design)
        self._set_loading(False)
        await self._render_logs()

        clarification_panel = self.query_one("#clarification-panel", Vertical)
        results_panel = self.query_one("#results-panel", Vertical)
        await clarification_panel.remove_children()
        await results_panel.remove_children()

        if response.requires_clarification:
            self.pending_questions = response.questions
            for question in response.questions:
                await clarification_panel.mount(ClarificationCard(question))
            self.query_one("#continue", Button).display = True
            self.query_one("#status-copy", Static).update(response.status_message)
            return

        self.pending_questions = []
        self.query_one("#continue", Button).display = False
        await self._render_results(response.ranked_products, response.status_message)

    async def _complete_search(self) -> None:
        missing = [
            question.prompt
            for question in self.pending_questions
            if question.id not in self.selected_answers
        ]
        if missing:
            self.query_one("#status-copy", Static).update(
                "Pick one option for each follow-up question before continuing."
            )
            return

        self._set_loading(True)
        response = await self.service.finalize_search(
            self.pending_query,
            self.selected_answers,
            self.selected_design,
        )
        self._set_loading(False)
        self.query_one("#continue", Button).display = False
        await self.query_one("#clarification-panel", Vertical).remove_children()
        await self._render_logs()
        await self._render_results(response.ranked_products, response.status_message)

    async def _render_results(
        self, ranked_products: list[RankedProduct], status_message: str
    ) -> None:
        results_panel = self.query_one("#results-panel", Vertical)
        await results_panel.remove_children()
        self.result_urls = {
            item.rank: item.product.product_url for item in ranked_products
        }

        for item in ranked_products:
            await results_panel.mount(ResultCard(item))
        self.query_one("#status-copy", Static).update(status_message)

    def _highlight_selected_answers(self) -> None:
        for button in self.query(".answer-option", Button):
            button.remove_class("selected")
            button_id = button.id or ""
            if not button_id.startswith("answer-"):
                continue
            _, question_id, option_id = button_id.split("-", maxsplit=2)
            if self.selected_answers.get(question_id) == option_id:
                button.add_class("selected")

    def _set_loading(self, is_loading: bool) -> None:
        self.query_one("#loading").display = is_loading
        self.query_one("#submit", Button).disabled = is_loading
        self.query_one("#continue", Button).disabled = is_loading
        if is_loading:
            self.query_one("#status-copy", Static).update(
                "Retrieving mock products and preparing ranked output..."
            )

    async def _render_logs(self) -> None:
        log_panel = self.query_one("#log-panel", Vertical)
        await log_panel.remove_children()
        recent = self.service.logger.read_recent()
        if not recent:
            await log_panel.mount(Static("No runs logged yet.", classes="log-line"))
            return
        for entry in reversed(recent):
            event_type = entry["event_type"]
            timestamp = entry["timestamp"].split("T", maxsplit=1)[1][:8]
            await log_panel.mount(
                Static(f"{timestamp}  {event_type}", classes="log-line")
            )


def run_app(design: RankingDesign = RankingDesign.DIRECT_JSON) -> None:
    app = ShoppingAgentApp(design=design)
    app.run()
