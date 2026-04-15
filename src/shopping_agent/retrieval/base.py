from __future__ import annotations

from typing import Protocol

from ..types import Product


class ProductSourceAdapter(Protocol):
    source_name: str

    async def search(self, query: str, limit: int = 50) -> list[Product]:
        """Return normalized product records for a source."""
        ...


class AmazonAdapter(ProductSourceAdapter):
    source_name = "Amazon"

    async def search(self, query: str, limit: int = 50) -> list[Product]:
        """Return normalized Amazon product results."""
        return []  # TODO: implement here


class EbayAdapter(ProductSourceAdapter):
    source_name = "eBay"

    async def search(self, query: str, limit: int = 50) -> list[Product]:
        """Return normalized eBay product results."""
        return []  # TODO: implement here
