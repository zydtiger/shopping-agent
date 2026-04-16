from __future__ import annotations

import sqlite3
from collections import Counter
from typing import Any

from ..types import Product


class ProductSQLStore:
    def __init__(self) -> None:
        self._products: dict[str, Product] = {}
        self._connection = sqlite3.connect(":memory:")
        self._connection.row_factory = sqlite3.Row
        self._initialize_schema()

    def _initialize_schema(self) -> None:
        self._connection.execute(
            """
            CREATE TABLE products (
                title TEXT NOT NULL,
                price REAL NOT NULL,
                source_site TEXT NOT NULL,
                product_url TEXT PRIMARY KEY,
                rating REAL
            )
            """
        )
        self._connection.commit()

    def add_products(self, products: list[Product]) -> int:
        inserted = 0
        for product in products:
            if product.product_url in self._products:
                continue
            self._products[product.product_url] = product
            self._connection.execute(
                """
                INSERT INTO products (title, price, source_site, product_url, rating)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    product.title,
                    product.price,
                    product.source_site,
                    product.product_url,
                    product.rating,
                ),
            )
            inserted += 1
        self._connection.commit()
        return inserted

    def all_products(self) -> list[Product]:
        return list(self._products.values())

    def get_product(self, product_url: str) -> Product | None:
        return self._products.get(product_url)

    def source_counts(self) -> dict[str, int]:
        return dict(Counter(product.source_site for product in self._products.values()))

    def summary(self) -> dict[str, Any]:
        prices = [product.price for product in self._products.values()]
        return {
            "total_products": len(self._products),
            "source_counts": self.source_counts(),
            "min_price": min(prices) if prices else None,
            "max_price": max(prices) if prices else None,
        }

    def query(self, sql: str, max_rows: int = 50) -> dict[str, Any]:
        normalized = sql.strip().lower()
        if not normalized.startswith(("select", "with")):
            raise ValueError("Only read-only SELECT queries are allowed for query_product_store.")

        cursor = self._connection.execute(sql)
        rows = [dict(row) for row in cursor.fetchmany(max_rows + 1)]
        truncated = len(rows) > max_rows
        if truncated:
            rows = rows[:max_rows]
        return {
            "sql": sql,
            "row_count": len(rows),
            "truncated": truncated,
            "rows": rows,
        }
