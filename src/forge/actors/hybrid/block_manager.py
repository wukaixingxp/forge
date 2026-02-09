# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Block manager for paged KV cache with prefix caching support.

This is adapted from nano-vLLM's block manager for our hybrid training/inference use case.
The block manager handles allocation and deallocation of KV cache blocks with automatic
prefix caching based on token sequence hashing.
"""

from collections import deque
import xxhash
import numpy as np

from forge.actors.hybrid.sequence import Sequence


class Block:
    """Represents a single KV cache block.

    Attributes:
        block_id: Unique block identifier
        ref_count: Number of sequences referencing this block
        hash: Hash of token IDs in this block (for prefix caching)
        token_ids: Token IDs stored in this block
    """

    def __init__(self, block_id: int):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids: list[int] = []

    def update(self, hash: int, token_ids: list[int]):
        """Update block with new hash and tokens."""
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        """Reset block to initial state (ref_count=1, no hash)."""
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []

    def __repr__(self) -> str:
        return (
            f"Block(id={self.block_id}, "
            f"ref_count={self.ref_count}, "
            f"hash={self.hash}, "
            f"tokens={len(self.token_ids)})"
        )


class BlockManager:
    """Manages allocation of KV cache blocks with prefix caching.

    This manager handles:
    1. Block allocation and deallocation
    2. Automatic prefix caching via token hash matching
    3. Reference counting for shared blocks

    Args:
        num_blocks: Total number of blocks available
        block_size: Number of tokens per block
    """

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = {}
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1) -> int:
        """Compute hash of token sequence with optional prefix.

        Args:
            token_ids: Token IDs to hash
            prefix: Hash of previous block (for chaining)

        Returns:
            Hash value as integer
        """
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block:
        """Allocate a specific block."""
        block = self.blocks[block_id]
        assert block.ref_count == 0, f"Block {block_id} already in use"
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int):
        """Deallocate a specific block."""
        assert self.blocks[block_id].ref_count == 0, f"Block {block_id} still has references"
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> bool:
        """Check if enough free blocks exist for sequence.

        Args:
            seq: Sequence to check

        Returns:
            True if allocation is possible
        """
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence):
        """Allocate blocks for a sequence with prefix caching.

        This method:
        1. Checks each block's tokens against the hash table
        2. Reuses cached blocks if hash matches
        3. Allocates new blocks for cache misses
        4. Updates hash table for new blocks

        CRITICAL: The last block (if partial) should NEVER be shared between sequences
        that will generate different completions, as they will write different tokens
        to the same cache location causing KV cache corruption.

        Args:
            seq: Sequence to allocate blocks for
        """
        assert not seq.block_table, "Sequence already has blocks allocated"
        h = -1
        cache_miss = False

        import logging
        import os
        logger = logging.getLogger(__name__)
        DEBUG = os.environ.get('FORGE_DEBUG', '0') == '1'

        if DEBUG:
            logger.info(f"[BLOCK_MGR] Allocating blocks for seq {seq.seq_id}, num_blocks={seq.num_blocks}, tokens={seq.token_ids}")

        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            is_last_block = (i == seq.num_blocks - 1)
            is_partial_block = len(token_ids) < self.block_size

            # Only hash full blocks (prefix caching requires complete blocks)
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1

            # Check for cache hit
            block_id = self.hash_to_block_id.get(h, -1)

            if DEBUG:
                logger.info(f"[BLOCK_MGR] Seq {seq.seq_id}, block {i}: h={h}, cached_block_id={block_id}, "
                           f"is_last={is_last_block}, is_partial={is_partial_block}, "
                           f"tokens={token_ids}, cache_miss={cache_miss}")

            # Verify tokens match (hash collision protection)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True

            # CRITICAL FIX: Never share the last partial block
            # Reason: During decode, different sequences will write different tokens to it,
            # causing KV cache corruption if they share the same physical block
            force_new_block = is_last_block and is_partial_block

            if cache_miss or force_new_block:
                # Cache miss or forced new block: allocate new block
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
                if DEBUG:
                    logger.info(f"[BLOCK_MGR] Seq {seq.seq_id}, block {i}: ALLOCATED NEW block_id={block_id}")
            else:
                # Cache hit: reuse existing block
                seq.num_cached_tokens += self.block_size
                if block_id in self.used_block_ids:
                    # Block in use: increment ref count
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    # Block not in use: allocate it
                    block = self._allocate_block(block_id)

            # Update hash table for full blocks
            if h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id

            seq.block_table.append(block_id)

    def deallocate(self, seq: Sequence):
        """Deallocate blocks for a sequence.

        Decrements ref counts and frees blocks with zero refs.

        Args:
            seq: Sequence to deallocate blocks for
        """
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)

        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        """Check if we can append a token to sequence.

        Args:
            seq: Sequence to check

        Returns:
            True if append is possible (may need new block)
        """
        # Need new block if current block is full
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        """Prepare for token append (allocate new block if needed).

        This is called during decode when a new token is about to be generated.

        Args:
            seq: Sequence that will receive a new token
        """
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]

        if len(seq) % self.block_size == 1:
            # Current block just became full, need new block
            assert last_block.hash != -1, "Last block should be hashed"
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)

        elif len(seq) % self.block_size == 0:
            # Previous block is now complete, hash it
            assert last_block.hash == -1, "Last block should not be hashed yet"
            token_ids = seq.block(seq.num_blocks - 1)
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            h = self.compute_hash(token_ids, prefix)
            last_block.update(h, token_ids)
            self.hash_to_block_id[h] = last_block.block_id

        else:
            # Still filling current block
            assert last_block.hash == -1, "Last block should not be hashed yet"

    def get_stats(self) -> dict:
        """Get block manager statistics.

        Returns:
            Dict with statistics
        """
        return {
            'total_blocks': len(self.blocks),
            'free_blocks': len(self.free_block_ids),
            'used_blocks': len(self.used_block_ids),
            'cached_blocks': len(self.hash_to_block_id),
            'block_size': self.block_size,
        }

    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"BlockManager("
            f"total={stats['total_blocks']}, "
            f"free={stats['free_blocks']}, "
            f"used={stats['used_blocks']}, "
            f"cached={stats['cached_blocks']})"
        )
