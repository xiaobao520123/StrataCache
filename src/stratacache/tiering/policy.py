from __future__ import annotations

from enum import Enum


class LinkPolicy(str, Enum):
    WRITE_THROUGH = "write_through"
    WRITE_BACK = "write_back"


class StoreReason(str, Enum):
    CLIENT_WRITE = "client_write"
    PROMOTION = "promotion"
    WRITEBACK_FLUSH = "writeback_flush"

