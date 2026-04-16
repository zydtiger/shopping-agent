from __future__ import annotations

import re
from urllib.parse import quote_plus, urlsplit, urlunsplit

from playwright.async_api import Locator, Page

from ..types import Product
from .base import ProductSourceAdapter
from .launch import launch_browser

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


def parse_rating(value: str | None) -> float | None:
    if not value:
        return None
    match = re.search(r"(\d+(?:\.\d+)?)", value)
    return float(match.group(1)) if match else None


class EbayAdapter(ProductSourceAdapter):
    source_name = "eBay"
    base_url = f"https://{EBAY_DOMAIN}"
    results_selector = "li.s-card[data-listingid]"

    @classmethod
    def build_search_url(cls, query: str, page_number: int = 1) -> str:
        return f"{cls.base_url}/sch/i.html?_nkw={quote_plus(query)}&_pgn={page_number}"

    @classmethod
    async def search(cls, query: str, limit: int = 50) -> list[Product]:
        """Return normalized eBay product results."""
        normalized_limit = max(1, limit)
        unique_products: dict[str, Product] = {}
        page_number = 1

        async with launch_browser() as page:
            while len(unique_products) < normalized_limit:
                target_url = cls.build_search_url(query, page_number=page_number)
                await page.goto(target_url, wait_until="domcontentloaded")

                await cls.wait_for_results_or_block(page, pause_on_block=False)
                await page.mouse.wheel(0, 1800)
                await page.wait_for_timeout(1000)

                page_products = await cls.extract_products(page)
                if not page_products:
                    break

                new_products = 0
                for product in page_products:
                    key = product.product_url
                    if key in unique_products:
                        continue
                    unique_products[key] = product
                    new_products += 1
                    if len(unique_products) >= normalized_limit:
                        break

                if new_products == 0:
                    break

                page_number += 1

        return list(unique_products.values())[:normalized_limit]

    @classmethod
    async def extract_products(cls, page: Page) -> list[Product]:
        cards = page.locator(cls.results_selector)
        count = await cards.count()
        products: list[Product] = []

        for index in range(count):
            card = cards.nth(index)

            title = await cls._text_from_first_match(
                card,
                [
                    ".s-card__title .su-styled-text",
                    ".s-card__title",
                    ".s-item__title",
                ],
            )
            href = await cls._attr_from_first_match(
                card,
                [
                    ".su-card-container__header > a.s-card__link",
                    "a.s-card__link.image-treatment",
                    ".s-item__link",
                ],
                "href",
            )
            price_text = await cls._text_from_first_match(
                card,
                [
                    ".s-card__price",
                    ".s-item__price",
                ],
            )
            rating_text = await cls._text_from_first_match(
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
                    rating=parse_rating(rating_text),
                )
            )

        return products

    @classmethod
    async def _text_from_first(cls, card: Locator, selector: str) -> str | None:
        locator = card.locator(selector)
        if await locator.count() == 0:
            return None
        return cls.clean_text(await locator.first.text_content())

    @classmethod
    async def _attr_from_first(cls, card: Locator, selector: str, attribute: str) -> str | None:
        locator = card.locator(selector)
        if await locator.count() == 0:
            return None
        return cls.clean_text(await locator.first.get_attribute(attribute))

    @classmethod
    async def _text_from_first_match(cls, card: Locator, selectors: list[str]) -> str | None:
        for selector in selectors:
            value = await cls._text_from_first(card, selector)
            if value:
                return value
        return None

    @classmethod
    async def _attr_from_first_match(
        cls,
        card: Locator,
        selectors: list[str],
        attribute: str,
    ) -> str | None:
        for selector in selectors:
            value = await cls._attr_from_first(card, selector, attribute)
            if value:
                return value
        return None

    @classmethod
    def normalize_product_url(cls, url: str | None) -> str | None:
        normalized = cls.normalize_url(url)
        if not normalized:
            return None
        parsed = urlsplit(normalized)
        if "/itm/" in parsed.path:
            return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))
        return normalized
