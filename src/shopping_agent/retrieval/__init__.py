"""Retrieval adapters and provider scaffolds."""

from .amazon import AmazonAdapter
from .base import ProductSourceAdapter
from .ebay import EbayAdapter
from .launch import launch_browser

__all__ = ["AmazonAdapter", "EbayAdapter", "ProductSourceAdapter", "launch_browser"]
