from __future__ import annotations

from typing import Protocol

from ..domain import Product, UserPreferenceProfile


class ProductSourceAdapter(Protocol):
    source_name: str

    async def search(self, query: str, profile: UserPreferenceProfile) -> list[Product]:
        """Return normalized product records for a source."""
        ...
