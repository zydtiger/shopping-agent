"""Retrieval adapters and provider scaffolds."""

from .amazon import AmazonAdapter
from .base import ProductSourceAdapter
from .ebay import EbayAdapter

__all__ = [
    "AmazonAdapter",
    "EbayAdapter",
    "ProductSourceAdapter",
]
