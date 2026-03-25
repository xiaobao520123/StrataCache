import hashlib
from typing import Sequence, Callable, Any
from typing_extensions import NewType
from collections.abc import Iterable

HASH_SIZE = 8
TOKEN_SIZE = 4
PAD_TOKEN_ID = 0xFFFFFFFF

LSMBlockKey = NewType("LSMBlockKey", bytes)


def default_hash_fn(data: bytes) -> bytes:
    return hashlib.blake2b(data, digest_size=HASH_SIZE).digest()


def build_token_block_key(
    curr_block_token_ids: Sequence[int],
    parent_block_hash: bytes,
    block_size: int = 16,
) -> LSMBlockKey:
    """
    Build a single block key by combining the parent block hash and the current block's token IDs.
    """
    assert (
        len(parent_block_hash) == HASH_SIZE
    ), f"Parent hash must be exactly {HASH_SIZE} bytes"

    # Padding logic: if the length is insufficient, pad with 0.
    # Note: When storing in Value later, you must record len(curr_block_token_ids)
    pad_len = block_size - len(curr_block_token_ids)
    if pad_len > 0:
        padded_ids = list(curr_block_token_ids) + [PAD_TOKEN_ID] * pad_len
    else:
        padded_ids = curr_block_token_ids[:block_size]

    # Convert token ids to bytes
    token_bytes = b"".join(
        tid.to_bytes(TOKEN_SIZE, byteorder="big") for tid in padded_ids
    )

    final_key = parent_block_hash + token_bytes
    assert len(final_key) == HASH_SIZE + (
        block_size * TOKEN_SIZE
    ), f"Key size mismatch! Expected {HASH_SIZE + (block_size * TOKEN_SIZE)}, got {len(final_key)}"

    return LSMBlockKey(final_key)


def build_full_token_block_key(
    token_ids: list[int],
    boundaries: list[int],
    hash_fn: Callable[[Any], bytes] = default_hash_fn,
    block_size: int = 16,
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

    sorted_boundaries = sorted(list(set(boundaries)))

    parent_hash = b"\x00" * HASH_SIZE

    start_idx = 1
    for end_idx in sorted_boundaries:

        if start_idx >= len(token_ids):
            break

        curr_block_tokens = token_ids[start_idx:end_idx]

        block_key = build_token_block_key(
            curr_block_tokens, parent_hash, block_size=block_size
        )
        out[end_idx] = block_key

        # Update parent hash for the next block
        parent_hash = hash_fn(block_key)

        start_idx = end_idx

    return out
