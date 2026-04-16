from __future__ import annotations

import asyncio
import re
import sys
from abc import ABC, abstractmethod

from playwright.async_api import Locator, Page
from playwright.async_api import TimeoutError as PlaywrightTimeoutError

from ..types import Product
from .launch import launch_browser


class ProductSourceAdapter(ABC):
    source_name: str
    base_url = ""
    results_selector = ""

    @classmethod
    @abstractmethod
    def build_search_url(cls, query: str, page_number: int = 1) -> str:
        """Build the search URL for a given query and page."""

    @classmethod
    @abstractmethod
    def normalize_product_url(cls, url: str | None) -> str | None:
        """Normalize a source-specific product URL for stable deduplication."""

    @classmethod
    @abstractmethod
    async def extract_products(cls, page: Page) -> list[Product]:
        """Extract products from a source results page."""

    @classmethod
    def clean_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = re.sub(r"\s+", " ", value).strip()
        return cleaned or None

    @classmethod
    def normalize_url(cls, url: str | None) -> str | None:
        if not url:
            return None
        if url.startswith("http://") or url.startswith("https://"):
            return url
        if not cls.base_url:
            raise NotImplementedError(f"{cls.__name__} must define base_url.")
        return f"{cls.base_url}{url}"

    @classmethod
    async def wait_for_results_or_block(cls, page: Page, pause_on_block: bool) -> None:
        if not cls.results_selector:
            raise NotImplementedError(f"{cls.__name__} must define results_selector.")

        try:
            await page.wait_for_selector(
                cls.results_selector,
                timeout=15_000,
            )
            return
        except PlaywrightTimeoutError:
            pass

        current_url = page.url.lower()
        page_text = (await page.text_content("body") or "").lower()
        blocked = any(
            token in current_url or token in page_text
            for token in ("captcha", "robot", "sorry", "validatecaptcha")
        )
        consent = "consent" in current_url or "cookies" in current_url

        if (blocked or consent) and pause_on_block:
            print(
                f"{cls.source_name} blocked the automated search or showed an interstitial. "
                "Complete it in the browser window, then press Enter here.",
                file=sys.stderr,
            )
            await asyncio.to_thread(input)
            await page.wait_for_selector(
                cls.results_selector,
                timeout=60_000,
            )
            return

        if blocked:
            raise RuntimeError(
                f"{cls.source_name} blocked the request with a captcha or bot check. "
                "Re-run with --pause-on-block for manual intervention."
            )

        if consent:
            raise RuntimeError(
                f"{cls.source_name} showed a consent/interstitial page before results loaded. "
                "Re-run with --pause-on-block to handle it manually."
            )

        raise RuntimeError(f"{cls.source_name} search results did not load.")

    @classmethod
    async def search(cls, query: str, limit: int = 50) -> list[Product]:
        """Return normalized product records for a source."""
        normalized_limit = max(1, limit)
        unique_products: dict[str, Product] = {}
        page_number = 1

        async with launch_browser() as page:
            while len(unique_products) < normalized_limit:
                target_url = cls.build_search_url(query, page_number=page_number)
                await cls.load_results_page(page, target_url)

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
    async def load_results_page(cls, page: Page, target_url: str) -> None:
        await page.goto(target_url, wait_until="domcontentloaded")
        await cls.wait_for_results_or_block(page, pause_on_block=False)
        await page.mouse.wheel(0, 1800)
        await page.wait_for_timeout(1000)

    @classmethod
    def parse_rating(cls, value: str | None) -> float | None:
        if not value:
            return None
        match = re.search(r"(\d+(?:\.\d+)?)", value)
        return float(match.group(1)) if match else None

    @classmethod
    async def text_from_first(cls, container: Locator, selector: str) -> str | None:
        locator = container.locator(selector)
        if await locator.count() == 0:
            return None
        return cls.clean_text(await locator.first.text_content())

    @classmethod
    async def attr_from_first(
        cls,
        container: Locator,
        selector: str,
        attribute: str,
    ) -> str | None:
        locator = container.locator(selector)
        if await locator.count() == 0:
            return None
        return cls.clean_text(await locator.first.get_attribute(attribute))

    @classmethod
    async def text_from_first_match(cls, container: Locator, selectors: list[str]) -> str | None:
        for selector in selectors:
            value = await cls.text_from_first(container, selector)
            if value:
                return value
        return None

    @classmethod
    async def attr_from_first_match(
        cls,
        container: Locator,
        selectors: list[str],
        attribute: str,
    ) -> str | None:
        for selector in selectors:
            value = await cls.attr_from_first(container, selector, attribute)
            if value:
                return value
        return None
