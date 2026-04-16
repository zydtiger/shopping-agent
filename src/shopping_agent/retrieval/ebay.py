from __future__ import annotations

import re
from urllib.parse import quote_plus, urlsplit, urlunsplit

from playwright.async_api import Page

from ..types import Product
from .base import ProductSourceAdapter

EBAY_DOMAIN = "www.ebay.com"


def parse_price(value: str | None) -> float | None:
    if not value:
        return None
    match = re.search(r"(\d[\d,]*)(?:\.(\d{1,2}))?", value)
    if not match:
        return None
    whole = match.group(1).replace(",", "")
    cents = (match.group(2) or "00").ljust(2, "0")[:2]
    return float(f"{whole}.{cents}")


class EbayAdapter(ProductSourceAdapter):
    source_name = "eBay"
    base_url = f"https://{EBAY_DOMAIN}"
    results_selector = "li.s-card[data-listingid]"

    @classmethod
    def build_search_url(cls, query: str, page_number: int = 1) -> str:
        return f"{cls.base_url}/sch/i.html?_nkw={quote_plus(query)}&_pgn={page_number}"

    @classmethod
    def normalize_product_url(cls, url: str | None) -> str | None:
        normalized = cls.normalize_url(url)
        if not normalized:
            return None
        parsed = urlsplit(normalized)
        if "/itm/" in parsed.path:
            return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))
        return normalized

    @classmethod
    async def extract_products(cls, page: Page) -> list[Product]:
        cards = page.locator(cls.results_selector)
        count = await cards.count()
        products: list[Product] = []

        for index in range(count):
            card = cards.nth(index)

            title = await cls.text_from_first_match(
                card,
                [
                    ".s-card__title .su-styled-text",
                    ".s-card__title",
                    ".s-item__title",
                ],
            )
            href = await cls.attr_from_first_match(
                card,
                [
                    ".su-card-container__header > a.s-card__link",
                    "a.s-card__link.image-treatment",
                    ".s-item__link",
                ],
                "href",
            )
            price_text = await cls.text_from_first_match(
                card,
                [
                    ".s-card__price",
                    ".s-item__price",
                ],
            )
            rating_text = await cls.text_from_first_match(
                card,
                [
                    ".x-star-rating span.clipped",
                    "[aria-label*='out of 5 stars']",
                ],
            )

            if not title or title.lower() == "shop on ebay":
                continue

            product_url = cls.normalize_product_url(href)
            price = parse_price(price_text)
            if price is None or not product_url:
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
