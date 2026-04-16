"""Retrieval adapters and provider scaffolds."""

from .amazon import AmazonAdapter
from .base import ProductSourceAdapter
from .ebay import EbayAdapter
from .launch import launch_browser, set_browser_headless
from .newegg import NeweggAdapter

__all__ = [
    "AmazonAdapter",
    "EbayAdapter",
    "NeweggAdapter",
    "ProductSourceAdapter",
    "launch_browser",
    "set_browser_headless",
]
