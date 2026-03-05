"""
StrataCache: standalone tiered artifact cache.

This package intentionally does not depend on `lmcache`.
"""

from .core.version import __version__
from .engine import AccessMode, StorageEngine

__all__ = ["__version__", "StorageEngine", "AccessMode"]
