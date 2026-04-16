from __future__ import annotations

import re
from urllib.parse import quote_plus, urlsplit, urlunsplit

from playwright.async_api import Page

from ..types import Product
from .base import ProductSourceAdapter

AMAZON_DOMAIN = "www.amazon.com"


def parse_price(whole: str | None, fraction: str | None) -> float | None:
    if not whole:
        return None
    numeric = whole.replace(",", "").replace(".", "").strip()
    if not numeric.isdigit():
        return None
    cents = (fraction or "00").strip()
    if not cents.isdigit():
        cents = "00"
    return float(f"{numeric}.{cents}")


class AmazonAdapter(ProductSourceAdapter):
    source_name = "Amazon"
    base_url = f"https://{AMAZON_DOMAIN}"
    results_selector = "[data-component-type='s-search-result']"

    @classmethod
    def build_search_url(cls, query: str, page_number: int = 1) -> str:
        return f"https://{AMAZON_DOMAIN}/s?k={quote_plus(query)}&page={page_number}"

    @classmethod
    def normalize_product_url(cls, url: str | None) -> str | None:
        normalized = cls.normalize_url(url)
        if not normalized:
            return None

        parsed = urlsplit(normalized)
        path_match = re.search(r"(/dp/[A-Z0-9]{10})(?:[/?]|$)", parsed.path, flags=re.IGNORECASE)
        if path_match:
            return urlunsplit((parsed.scheme, parsed.netloc, path_match.group(1), "", ""))

        product_match = re.search(
            r"(/gp/product/[A-Z0-9]{10})(?:[/?]|$)",
            parsed.path,
            flags=re.IGNORECASE,
        )
        if product_match:
            return urlunsplit((parsed.scheme, parsed.netloc, product_match.group(1), "", ""))

        return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))

    @classmethod
    async def extract_products(cls, page: Page) -> list[Product]:
        cards = page.locator(cls.results_selector)
        count = await cards.count()
        products: list[Product] = []

        for index in range(count):
            card = cards.nth(index)

            asin = await card.get_attribute("data-asin")
            if not asin:
                continue

            title = await cls.text_from_first(card, "h2 span")
            href = await cls.attr_from_first(card, "a.a-link-normal", "href")
            product_url = cls.normalize_product_url(href)
            whole = cls.clean_text(
                await card.locator(".a-price .a-price-whole").first.text_content()
            )
            fraction = cls.clean_text(
                await card.locator(".a-price .a-price-fraction").first.text_content()
            )
            price = parse_price(whole, fraction)
            rating_locator = card.locator("[aria-label*='out of 5 stars']")
            rating_text = None
            if await rating_locator.count():
                rating_text = cls.clean_text(await rating_locator.first.get_attribute("aria-label"))

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
