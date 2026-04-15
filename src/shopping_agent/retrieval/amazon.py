from __future__ import annotations

from ..types import Product
from .base import ProductSourceAdapter


class AmazonAdapter(ProductSourceAdapter):
    source_name = "Amazon"

    async def search(self, query: str, limit: int = 50) -> list[Product]:
        """Return normalized Amazon product results."""
        return []  # TODO: implement here
