from __future__ import annotations

import asyncio
import re
import sys
from abc import ABC, abstractmethod

from playwright.async_api import Page
from playwright.async_api import TimeoutError as PlaywrightTimeoutError

from ..types import Product


class ProductSourceAdapter(ABC):
    source_name: str
    base_url = ""
    results_selector = ""

    @classmethod
    @abstractmethod
    def build_search_url(cls, query: str, page_number: int = 1) -> str:
        """Build the search URL for a given query and page."""

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
    @abstractmethod
    async def search(cls, query: str, limit: int = 50) -> list[Product]:
        """Return normalized product records for a source."""

    @classmethod
    @abstractmethod
    async def extract_products(cls, page: Page) -> list[Product]:
        """Extract products from a source results page."""
