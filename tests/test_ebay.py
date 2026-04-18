from __future__ import annotations

import unittest
from contextlib import asynccontextmanager
from unittest.mock import patch

from shopping_agent.retrieval.ebay import EbayAdapter
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
        title: str | None,
        href: str | None,
        price: str | None,
        rating_text: str | None,
    ) -> None:
        self._locators = {
            ".s-card__title .su-styled-text": FakeTextLocator(text=title),
            ".s-card__title": FakeTextLocator(text=title),
            ".su-card-container__header > a.s-card__link": FakeTextLocator(
                attrs={"href": href} if href else {}
            ),
            "a.s-card__link.image-treatment": FakeTextLocator(attrs={"href": href} if href else {}),
            ".s-item__title": FakeTextLocator(text=None),
            ".s-item__link": FakeTextLocator(attrs={}),
            ".s-card__price": FakeTextLocator(text=price),
            ".s-item__price": FakeTextLocator(text=None),
            ".x-star-rating span.clipped": FakeTextLocator(text=rating_text),
            "[aria-label*='out of 5 stars']": FakeTextLocator(text=rating_text),
        }

    def locator(self, selector: str) -> object:
        return self._locators[selector]


class _FakeMouse:
    async def wheel(self, x: int, y: int) -> None:
        return None


class FakePage:
    def __init__(self, cards: list[FakeCard] | None = None) -> None:
        self._cards = cards or []
        self.visited_urls: list[str] = []
        self.mouse = _FakeMouse()

    def locator(self, selector: str) -> FakeListLocator:
        if selector != "li.s-card[data-listingid]":
            raise AssertionError(f"Unexpected selector: {selector}")
        return FakeListLocator(self._cards)

    async def goto(self, url: str, wait_until: str) -> None:
        self.visited_urls.append(url)

    async def wait_for_timeout(self, timeout_ms: int) -> None:
        return None


class EbayAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_extract_products_parses_card_and_skips_non_product_rows(self) -> None:
        page = FakePage(
            cards=[
                FakeCard(
                    title="Logitech Wireless Mouse",
                    href="/itm/123?_skw=wireless+mouse&itmmeta=abc123",
                    price="$19.99",
                    rating_text="4.8 out of 5 stars",
                ),
                FakeCard(
                    title="Shop on eBay",
                    href="/shop",
                    price="$0.00",
                    rating_text=None,
                ),
                FakeCard(
                    title="Gaming Mouse",
                    href="https://www.ebay.com/itm/456",
                    price="US $24.50",
                    rating_text=None,
                ),
            ]
        )

        products = await EbayAdapter.extract_products(page)

        self.assertEqual(len(products), 2)
        self.assertEqual(products[0].title, "Logitech Wireless Mouse")
        self.assertEqual(products[0].price, 19.99)
        self.assertEqual(products[0].product_url, "https://www.ebay.com/itm/123")
        self.assertEqual(products[0].rating, 4.8)
        self.assertIsNone(products[1].rating)

    async def test_search_keeps_paging_until_limit(self) -> None:
        page = FakePage()
        first = Product(
            title="One",
            price=10.0,
            source_site="eBay",
            product_url="https://www.ebay.com/itm/1",
            rating=4.1,
        )
        duplicate = Product(
            title="One duplicate",
            price=10.0,
            source_site="eBay",
            product_url="https://www.ebay.com/itm/1",
            rating=4.1,
        )
        second = Product(
            title="Two",
            price=20.0,
            source_site="eBay",
            product_url="https://www.ebay.com/itm/2",
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
                EbayAdapter,
                "wait_for_results_or_block",
                classmethod(fake_wait),
            ),
            patch.object(
                EbayAdapter,
                "extract_products",
                classmethod(fake_extract),
            ),
        ):
            products = await EbayAdapter.search("wireless mouse", limit=2)

        self.assertEqual([product.title for product in products], ["One", "Two"])
        self.assertEqual(len(page.visited_urls), 2)
        self.assertTrue(page.visited_urls[0].endswith("_pgn=1"))
        self.assertTrue(page.visited_urls[1].endswith("_pgn=2"))


if __name__ == "__main__":
    unittest.main()
