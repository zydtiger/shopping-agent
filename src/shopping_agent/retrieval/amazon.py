from __future__ import annotations

import re
from urllib.parse import quote_plus

from playwright.async_api import Page

from ..types import Product
from .base import ProductSourceAdapter
from .launch import launch_browser

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


def parse_rating(value: str | None) -> float | None:
    if not value:
        return None
    match = re.search(r"(\d+(?:\.\d+)?)", value)
    return float(match.group(1)) if match else None


class AmazonAdapter(ProductSourceAdapter):
    source_name = "Amazon"
    base_url = f"https://{AMAZON_DOMAIN}"
    results_selector = "[data-component-type='s-search-result']"

    @classmethod
    def build_search_url(cls, query: str, page_number: int = 1) -> str:
        return f"https://{AMAZON_DOMAIN}/s?k={quote_plus(query)}&page={page_number}"

    @classmethod
    async def search(cls, query: str, limit: int = 50) -> list[Product]:
        """Return normalized Amazon product results."""
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
        cards = page.locator("[data-component-type='s-search-result']")
        count = await cards.count()
        products: list[Product] = []

        for index in range(count):
            card = cards.nth(index)

            asin = await card.get_attribute("data-asin")
            if not asin:
                continue

            title = cls.clean_text(await card.locator("h2 span").first.text_content())
            href = await card.locator("a.a-link-normal").first.get_attribute("href")
            product_url = cls.normalize_url(href)
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
                    rating=parse_rating(rating_text),
                )
            )
        return products
