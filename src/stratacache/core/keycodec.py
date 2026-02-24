from __future__ import annotations

import hashlib


class KeyCodec:
    """
    Stable encoding of logical ArtifactId strings into backend-specific keys.

    The CXL store has a hard limit (20 bytes) for object names; we therefore
    expose helpers that generate short deterministic names.
    """

    @staticmethod
    def stable_bytes(s: str) -> bytes:
        # Use utf-8 as canonical encoding for stable hashing.
        return s.encode("utf-8", errors="strict")

    @staticmethod
    def short_hash_name(s: str, prefix: str = "H", hex_chars: int = 16, max_len: int = 20) -> str:
        """
        Generate a deterministic short name within max_len bytes (ASCII).

        Format: prefix + first `hex_chars` of sha256 hex digest.
        Default yields 1 + 16 = 17 chars, within CXL's 20-byte limit.
        """
        if len(prefix) >= max_len:
            raise ValueError("prefix too long for max_len")
        h = hashlib.sha256(KeyCodec.stable_bytes(s)).hexdigest()[:hex_chars]
        name = f"{prefix}{h}"
        if len(name) > max_len:
            name = name[:max_len]
        return name

