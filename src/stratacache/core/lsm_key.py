import hashlib
import struct
from typing import Sequence, Callable, Any
from typing_extensions import NewType
from collections.abc import Iterable

import logging

logger = logging.getLogger(__name__)

HASH_SIZE = 8
TOKEN_SIZE = 4
PAD_TOKEN_ID = 0xFFFFFFFF

LSMBlockKey = NewType("LSMBlockKey", list[bytes])

DEFAULT_KEY_TAIL_LENGTH = 16
DEFAULT_SUB_BLOCK_SIZE = 16

def _encode_sub_key(prefix_hash: int, tail_tokens: list[int]) -> bytes:
    """Encode a single sub-block key as [prefix_hash (8B)] + [tail tokens (4B each)]."""
    prefix = struct.pack(">Q", prefix_hash)
    tail_data = struct.pack(f"{len(tail_tokens)}I", *tail_tokens)
    return prefix + tail_data


def build_full_token_block_key(
    token_ids: list[int],
    boundaries: list[int],
    key_tail_length: int = DEFAULT_KEY_TAIL_LENGTH,
    sub_block_size: int = DEFAULT_SUB_BLOCK_SIZE,
) -> dict[int, LSMBlockKey]:
    """
    Build a mapping of block boundaries to LSMBlockKeys for a sequence of token ids.

    Each block (between consecutive boundaries) is subdivided into sub-blocks of
    ``sub_block_size`` tokens. An LSMBlockKey is a list of sub-key bytes — one
    per sub-block — so a 256-token chunk with sub_block_size=16 produces a key
    containing 16 sub-keys.

    The rolling hash is continuous across the entire sequence so that sub-keys
    chain naturally.

    Args:
        token_ids: A list of token ids representing the entire sequence.
        boundaries: A sorted list of exclusive-end indices that define blocks.
        key_tail_length: Number of trailing tokens to keep literally in each sub-key.
        sub_block_size: Number of tokens per sub-block (default 16).

    Return:
        A dictionary mapping each block boundary to its LSMBlockKey (a list of
        sub-key bytes, one per sub-block).
    """
    out: dict[int, LSMBlockKey] = {}

    if not token_ids or not boundaries:
        return out

    sorted_boundaries = sorted(boundaries)

    # Rolling hash: update token-by-token, emit sub-block keys.
    rolling = hashlib.blake2b(digest_size=HASH_SIZE)
    recent: list[int] = []  # sliding window for tail

    boundary_idx = 0
    current_sub_keys: list[bytes] = []
    sub_count = 0  # tokens consumed in current sub-block

    for i, tok in enumerate(token_ids):
        if boundary_idx >= len(sorted_boundaries):
            break

        rolling.update(struct.pack("I", tok))
        recent.append(tok)
        if len(recent) > key_tail_length:
            recent.pop(0)

        sub_count += 1
        pos = i + 1  # exclusive end index

        # Emit a sub-key when sub-block is full or we hit the boundary
        if sub_count == sub_block_size or pos == sorted_boundaries[boundary_idx]:
            prefix_hash = struct.unpack(">Q", rolling.digest())[0]
            current_sub_keys.append(_encode_sub_key(prefix_hash, list(recent)))
            sub_count = 0

        # Reached a block boundary – wrap sub-keys into one LSMBlockKey
        if pos == sorted_boundaries[boundary_idx]:
            out[pos] = LSMBlockKey(current_sub_keys)
            current_sub_keys = []
            sub_count = 0
            boundary_idx += 1
    return out
