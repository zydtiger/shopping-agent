from __future__ import annotations

from typing import Protocol

from ..types import Product


class ProductSourceAdapter(Protocol):
    source_name: str

    async def search(self, query: str, limit: int = 50) -> list[Product]:
        """Return normalized product records for a source."""
        ...
