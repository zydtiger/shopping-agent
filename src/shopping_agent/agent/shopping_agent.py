from __future__ import annotations

from typing import Any

from ..config import AppConfig
from ..retrieval import AmazonAdapter, EbayAdapter, NeweggAdapter
from ..types import RankingDesign, SearchResponse
from .api import ProgressCallback, build_client, default_response_runner, emit_progress
from .errors import AgentHarnessError
from .ranking_agent import RankingAgent
from .retrieval_agent import ClarificationCallback, RetrievalAgent
from .sql import ProductSQLStore


class ShoppingAgent:
    def __init__(self, config: AppConfig, result_limit: int = 50) -> None:
        self.config = config
        self.result_limit = max(1, result_limit)
        self.amazon_adapter = AmazonAdapter()
        self.ebay_adapter = EbayAdapter()
        self.newegg_adapter = NeweggAdapter()
        self.client = build_client(config)
        self.response_runner = self._default_response_runner

    async def run_search(
        self,
        query: str,
        design: RankingDesign,
        progress: ProgressCallback | None = None,
        ask_user: ClarificationCallback | None = None,
    ) -> SearchResponse:
        if ask_user is None:
            raise AgentHarnessError(
                "The harness needs a clarification callback to handle ask_clarification."
            )

        product_store = ProductSQLStore()
        retrieval_agent = RetrievalAgent(
            model_id=self.config.agent.openai_model_id,
            result_limit=self.result_limit,
            response_runner=self.response_runner,
            amazon_adapter=self.amazon_adapter,
            ebay_adapter=self.ebay_adapter,
            newegg_adapter=self.newegg_adapter,
        )
        retrieval = await retrieval_agent.run(
            query=query,
            progress=progress,
            ask_user=ask_user,
            product_store=product_store,
        )

        debug_notes = [f"Selected ranking design: {design.label}.", *retrieval.debug_notes]

        if not product_store.all_products():
            status_message = (
                "Retrieval finished but no products were collected, so ranking was skipped."
            )
            await emit_progress(progress, f"[action] {status_message}")
            return SearchResponse(
                query=query,
                design=design,
                profile=retrieval.profile,
                stage="results",
                status_message=status_message,
                retrieval_batches=retrieval.retrieval_batches,
                ranked_products=[],
                debug_notes=debug_notes,
                total_tokens=retrieval.total_tokens,
            )

        await emit_progress(
            progress,
            "[plan] RetrievalAgent finished. Handing the shared product store to "
            f"RankingAgent using {design.label}.",
        )
        ranking_agent = RankingAgent(
            model_id=self.config.agent.openai_model_id,
            response_runner=self.response_runner,
        )
        ranking = await ranking_agent.run(
            design=design,
            profile=retrieval.profile,
            product_store=product_store,
            progress=progress,
        )
        debug_notes.extend(ranking.debug_notes)

        await emit_progress(
            progress,
            "[action] Final ranking prepared with "
            f"{len(ranking.ranked_products)} recommendation(s).",
        )
        return SearchResponse(
            query=query,
            design=design,
            profile=retrieval.profile,
            stage="results",
            status_message=ranking.status_message or retrieval.status_message,
            retrieval_batches=retrieval.retrieval_batches,
            ranked_products=ranking.ranked_products,
            debug_notes=debug_notes,
            total_tokens=retrieval.total_tokens + ranking.total_tokens,
        )

    async def _default_response_runner(self, **kwargs: Any) -> Any:
        return await default_response_runner(self.client, **kwargs)
