from __future__ import annotations

import pathlib
import sys
import unittest
from contextlib import asynccontextmanager
from unittest.mock import patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from shopping_agent.retrieval.amazon import AmazonAdapter
from shopping_agent.types import Product


class FakeTextLocator:
    def __init__(self, *, text: str | None = None, attrs: dict[str, str] | None = None) -> None:
        self._text = text
        self._attrs = attrs or {}

    @property
    def first(self) -> FakeTextLocator:
        return self

    async def count(self) -> int:
        return 1 if self._text is not None or self._attrs else 0

    async def text_content(self) -> str | None:
        return self._text

    async def get_attribute(self, name: str) -> str | None:
        return self._attrs.get(name)


class FakeListLocator:
    def __init__(self, items: list[object]) -> None:
        self._items = items

    @property
    def first(self) -> object:
        return self._items[0]

    async def count(self) -> int:
        return len(self._items)

    def nth(self, index: int) -> object:
        return self._items[index]


class FakeCard:
    def __init__(
        self,
        *,
        asin: str | None,
        title: str | None,
        href: str | None,
        whole: str | None,
        fraction: str | None,
        rating_text: str | None,
    ) -> None:
        self._asin = asin
        self._locators = {
            "h2 span": FakeTextLocator(text=title),
            "a.a-link-normal": FakeTextLocator(attrs={"href": href} if href else {}),
            ".a-price .a-price-whole": FakeTextLocator(text=whole),
            ".a-price .a-price-fraction": FakeTextLocator(text=fraction),
            "[aria-label*='out of 5 stars']": FakeListLocator(
                [] if rating_text is None else [FakeTextLocator(attrs={"aria-label": rating_text})]
            ),
        }

    async def get_attribute(self, name: str) -> str | None:
        if name == "data-asin":
            return self._asin
        return None

    def locator(self, selector: str) -> object:
        return self._locators[selector]


class FakePage:
    def __init__(self, cards: list[FakeCard] | None = None) -> None:
        self._cards = cards or []
        self.visited_urls: list[str] = []
        self.mouse = _FakeMouse()

    def locator(self, selector: str) -> FakeListLocator:
        if selector != "[data-component-type='s-search-result']":
            raise AssertionError(f"Unexpected selector: {selector}")
        return FakeListLocator(self._cards)

    async def goto(self, url: str, wait_until: str) -> None:
        self.visited_urls.append(url)

    async def wait_for_timeout(self, timeout_ms: int) -> None:
        return None


class _FakeMouse:
    async def wheel(self, x: int, y: int) -> None:
        return None


class AmazonAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_extract_products_parses_card_and_allows_missing_rating(self) -> None:
        page = FakePage(
            cards=[
                FakeCard(
                    asin="A1",
                    title="  Travel  Backpack ",
                    href=(
                        "/AmazonBasics-Wireless-Computer-Mouse-Receiver/dp/B005EJH6Z4/"
                        "ref=sr_1_50?dib=abc123&keywords=mouse&qid=1776310076&sr=8-50"
                    ),
                    whole="129",
                    fraction="99",
                    rating_text="4.7 out of 5 stars",
                ),
                FakeCard(
                    asin="A2",
                    title="Desk Lamp",
                    href="/dp/A2",
                    whole="49",
                    fraction="50",
                    rating_text=None,
                ),
            ]
        )

        products = await AmazonAdapter.extract_products(page)

        self.assertEqual(len(products), 2)
        self.assertEqual(products[0].title, "Travel Backpack")
        self.assertEqual(products[0].price, 129.99)
        self.assertEqual(products[0].product_url, "https://www.amazon.com/dp/B005EJH6Z4")
        self.assertEqual(products[0].rating, 4.7)
        self.assertIsNone(products[1].rating)

    async def test_search_keeps_paging_until_limit(self) -> None:
        page = FakePage()
        first = Product(
            title="One",
            price=10.0,
            source_site="Amazon",
            product_url="https://www.amazon.com/dp/1",
            rating=4.1,
        )
        duplicate = Product(
            title="One duplicate",
            price=10.0,
            source_site="Amazon",
            product_url="https://www.amazon.com/dp/1",
            rating=4.1,
        )
        second = Product(
            title="Two",
            price=20.0,
            source_site="Amazon",
            product_url="https://www.amazon.com/dp/2",
            rating=4.5,
        )

        pages = [[first], [duplicate, second]]

        async def fake_wait(cls, page_obj, pause_on_block: bool) -> None:
            return None

        async def fake_extract(cls, page_obj) -> list[Product]:
            return pages.pop(0)

        @asynccontextmanager
        async def fake_launch_browser():
            yield page

        with (
            patch("shopping_agent.retrieval.base.launch_browser", fake_launch_browser),
            patch.object(
                AmazonAdapter,
                "wait_for_results_or_block",
                classmethod(fake_wait),
            ),
            patch.object(
                AmazonAdapter,
                "extract_products",
                classmethod(fake_extract),
            ),
        ):
            products = await AmazonAdapter.search("backpack", limit=2)

        self.assertEqual([product.title for product in products], ["One", "Two"])
        self.assertEqual(len(page.visited_urls), 2)
        self.assertTrue(page.visited_urls[0].endswith("page=1"))
        self.assertTrue(page.visited_urls[1].endswith("page=2"))


if __name__ == "__main__":
    unittest.main()
