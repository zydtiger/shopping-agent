from __future__ import annotations

from ..types import Product
from .base import ProductSourceAdapter


class EbayAdapter(ProductSourceAdapter):
    source_name = "eBay"

    async def search(self, query: str, limit: int = 50) -> list[Product]:
        """Return normalized eBay product results."""
        return []  # TODO: implement here
