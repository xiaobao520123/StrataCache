import hashlib
from typing import Sequence, Callable, Any
from typing_extensions import NewType
from collections.abc import Iterable

import logging

logger = logging.getLogger(__name__)

HASH_SIZE = 8
TOKEN_SIZE = 4
PAD_TOKEN_ID = 0xFFFFFFFF

LSMBlockKey = NewType("LSMBlockKey", bytes)


def build_full_token_block_key(
    token_ids: list[int],
    boundaries: list[int],
    block_size: int = 256,
) -> dict[int, LSMBlockKey]:
    """
    Build a mapping of block boundaries to LSMBlockKeys for a sequence of token ids.
    Each block key is constructed by hashing the previous block's key and the current block's token
    ids, ensuring a fixed key size regardless of block content.


    Args:
        token_ids: A list of token ids representing the entire sequence.
        boundaries: A list of indices in token_ids that represent block boundaries.
        hash_fn: A function to compute the hash for the parent block. Default is blake2b with 8-byte digest.

    Return:
        A dictionary mapping block end indices to their corresponding LSMBlockKeys.
    """
    out: dict[int, LSMBlockKey] = {}

    if not token_ids or not boundaries:
        return out

    want = set(boundaries)
    

    logger.info(f"Built {len(out)} block keys for token ids with boundaries at {sorted(want)}")
    return out
