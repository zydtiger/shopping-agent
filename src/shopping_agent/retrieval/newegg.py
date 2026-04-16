from __future__ import annotations

import re
from urllib.parse import quote_plus, urlsplit, urlunsplit

from playwright.async_api import Page

from ..types import Product
from .base import ProductSourceAdapter

NEWEGG_DOMAIN = "www.newegg.com"


def parse_price(value: str | None) -> float | None:
    if not value:
        return None
    match = re.search(r"(\d[\d,]*)(?:\.(\d{1,2}))?", value)
    if not match:
        return None
    whole = match.group(1).replace(",", "")
    cents = (match.group(2) or "00").ljust(2, "0")[:2]
    return float(f"{whole}.{cents}")


class NeweggAdapter(ProductSourceAdapter):
    source_name = "Newegg"
    base_url = f"https://{NEWEGG_DOMAIN}"
    results_selector = "div.item-cell"

    @classmethod
    def build_search_url(cls, query: str, page_number: int = 1) -> str:
        return f"{cls.base_url}/p/pl?d={quote_plus(query)}&page={page_number}"

    @classmethod
    def normalize_product_url(cls, url: str | None) -> str | None:
        normalized = cls.normalize_url(url)
        if not normalized:
            return None

        parsed = urlsplit(normalized)
        path_match = re.search(r"(/.+/p/N\d+E\d+)(?:[/?]|$)", parsed.path, flags=re.IGNORECASE)
        if path_match:
            return urlunsplit((parsed.scheme, parsed.netloc, path_match.group(1), "", ""))
        return urlunsplit((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", ""))

    @classmethod
    async def extract_products(cls, page: Page) -> list[Product]:
        cards = page.locator(cls.results_selector)
        count = await cards.count()
        products: list[Product] = []

        for index in range(count):
            card = cards.nth(index)

            title = await cls.text_from_first(card, "a.item-title")
            href = await cls.attr_from_first_match(
                card,
                [
                    "a.item-title",
                    "a.item-img",
                ],
                "href",
            )
            price_text = await cls.text_from_first_match(
                card,
                [
                    ".price-current",
                    ".price-current strong",
                ],
            )
            rating_text = await cls.attr_from_first_match(
                card,
                [
                    "a.item-rating",
                    "i.rating",
                ],
                "title",
            )
            if not rating_text:
                rating_text = await cls.attr_from_first_match(
                    card,
                    [
                        "a.item-rating",
                        "i.rating",
                    ],
                    "aria-label",
                )

            product_url = cls.normalize_product_url(href)
            price = parse_price(price_text)
            if price is None or not title or not product_url:
                continue

            products.append(
                Product(
                    title=title,
                    price=price,
                    source_site=cls.source_name,
                    product_url=product_url,
                    rating=cls.parse_rating(rating_text),
                )
            )

        return products
