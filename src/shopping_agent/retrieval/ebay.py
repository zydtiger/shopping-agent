from __future__ import annotations

from playwright.async_api import Page

from ..types import Product
from .base import ProductSourceAdapter


class EbayAdapter(ProductSourceAdapter):
    source_name = "eBay"
    base_url = ""
    results_selector = ""

    @classmethod
    def build_search_url(cls, query: str, page_number: int = 1) -> str:
        return ""

    @classmethod
    async def search(cls, query: str, limit: int = 50) -> list[Product]:
        """Return normalized eBay product results."""
        return []  # TODO: implement here

    @classmethod
    async def extract_products(cls, page: Page) -> list[Product]:
        return []
