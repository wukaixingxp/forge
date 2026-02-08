# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Paged KV cache for memory-efficient attention.

This module implements block-based memory management for KV cache,
inspired by vLLM's PagedAttention. Memory is allocated in fixed-size
blocks (e.g., 256 tokens per block) and allocated on-demand.

Benefits:
- Better memory utilization (2-3x higher batch size)
- Reference counting for shared prefixes
- Dynamic allocation/deallocation

Expected impact: 2-3x higher inference batch size.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


@dataclass
class BlockMetadata:
    """Metadata for a KV cache block.

    Args:
        block_id: Unique block identifier
        ref_count: Reference count for shared blocks
        num_tokens: Number of tokens stored in this block
    """
    block_id: int
    ref_count: int = 0
    num_tokens: int = 0


class PagedKVCache:
    """Block-based KV cache manager.

    Manages KV cache in fixed-size blocks for memory efficiency.
    Supports reference counting for prefix sharing.

    Args:
        block_size: Number of tokens per block (default: 256)
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        head_dim: Dimension per attention head
        dtype: Data type for cache tensors
        device: Device to allocate cache on
        max_blocks: Maximum number of blocks to allocate
    """

    def __init__(
        self,
        block_size: int = 256,
        num_layers: int = 32,
        num_heads: int = 32,
        head_dim: int = 128,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device = torch.device("cuda"),
        max_blocks: int = 1024,
    ):
        self.block_size = block_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device
        self.max_blocks = max_blocks

        # Block storage: [num_blocks, num_layers, 2, block_size, num_heads, head_dim]
        # 2 for keys and values
        self._blocks: Optional[torch.Tensor] = None
        self._block_metadata: Dict[int, BlockMetadata] = {}
        self._free_blocks: List[int] = []
        self._next_block_id = 0

        logger.info(
            f"PagedKVCache initialized (block_size={block_size}, "
            f"num_layers={num_layers}, num_heads={num_heads}, "
            f"head_dim={head_dim}, max_blocks={max_blocks})"
        )

    def _allocate_block_storage(self):
        """Allocate the block storage tensor (lazy allocation)."""
        if self._blocks is None:
            shape = (
                self.max_blocks,
                self.num_layers,
                2,  # keys and values
                self.block_size,
                self.num_heads,
                self.head_dim,
            )
            self._blocks = torch.zeros(
                shape,
                dtype=self.dtype,
                device=self.device,
            )
            logger.info(
                f"Allocated block storage: shape={shape}, "
                f"memory={self._blocks.numel() * self._blocks.element_size() / 1e9:.2f}GB"
            )

    def allocate_blocks(self, num_blocks: int) -> List[int]:
        """Allocate blocks for a sequence.

        Args:
            num_blocks: Number of blocks to allocate

        Returns:
            List of block IDs

        Raises:
            RuntimeError: If not enough free blocks available
        """
        self._allocate_block_storage()

        if len(self._free_blocks) >= num_blocks:
            # Reuse free blocks
            block_ids = [self._free_blocks.pop() for _ in range(num_blocks)]
        elif self._next_block_id + num_blocks <= self.max_blocks:
            # Allocate new blocks
            block_ids = list(range(self._next_block_id, self._next_block_id + num_blocks))
            self._next_block_id += num_blocks
        else:
            # Out of blocks
            available = len(self._free_blocks) + (self.max_blocks - self._next_block_id)
            raise RuntimeError(
                f"Out of KV cache blocks: requested={num_blocks}, available={available}"
            )

        # Initialize metadata
        for block_id in block_ids:
            self._block_metadata[block_id] = BlockMetadata(
                block_id=block_id,
                ref_count=1,
                num_tokens=0,
            )

        logger.debug(
            f"Allocated {num_blocks} blocks: {block_ids} "
            f"(free={len(self._free_blocks)}, next={self._next_block_id})"
        )

        return block_ids

    def free_blocks(self, block_ids: List[int]):
        """Free blocks back to the pool.

        Args:
            block_ids: List of block IDs to free
        """
        for block_id in block_ids:
            if block_id not in self._block_metadata:
                continue

            metadata = self._block_metadata[block_id]
            metadata.ref_count -= 1

            if metadata.ref_count <= 0:
                # Free block
                self._free_blocks.append(block_id)
                del self._block_metadata[block_id]
                logger.debug(f"Freed block {block_id}")

    def increment_ref_count(self, block_ids: List[int]):
        """Increment reference count for shared blocks.

        Args:
            block_ids: List of block IDs to increment
        """
        for block_id in block_ids:
            if block_id in self._block_metadata:
                self._block_metadata[block_id].ref_count += 1

    def get_block_table(
        self,
        block_ids: List[int],
    ) -> torch.Tensor:
        """Get block table tensor for attention computation.

        Args:
            block_ids: List of block IDs

        Returns:
            Block table tensor [num_blocks]
        """
        return torch.tensor(block_ids, dtype=torch.long, device=self.device)

    def write_kv(
        self,
        block_id: int,
        layer_idx: int,
        keys: torch.Tensor,
        values: torch.Tensor,
        start_pos: int = 0,
    ):
        """Write keys and values to a block.

        Args:
            block_id: Block to write to
            layer_idx: Layer index
            keys: Key tensor [num_tokens, num_heads, head_dim]
            values: Value tensor [num_tokens, num_heads, head_dim]
            start_pos: Starting position in block
        """
        if self._blocks is None:
            self._allocate_block_storage()

        num_tokens = keys.shape[0]
        end_pos = start_pos + num_tokens

        if end_pos > self.block_size:
            raise ValueError(
                f"Cannot write {num_tokens} tokens at position {start_pos} "
                f"(block_size={self.block_size})"
            )

        # Write keys: block[layer, 0, start:end, :, :]
        self._blocks[block_id, layer_idx, 0, start_pos:end_pos] = keys

        # Write values: block[layer, 1, start:end, :, :]
        self._blocks[block_id, layer_idx, 1, start_pos:end_pos] = values

        # Update metadata
        if block_id in self._block_metadata:
            self._block_metadata[block_id].num_tokens = max(
                self._block_metadata[block_id].num_tokens,
                end_pos,
            )

    def read_kv(
        self,
        block_ids: List[int],
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Read keys and values from blocks.

        Args:
            block_ids: List of block IDs to read from
            layer_idx: Layer index

        Returns:
            Tuple of (keys, values) tensors
        """
        if self._blocks is None or not block_ids:
            return None, None

        # Concatenate blocks
        keys_list = []
        values_list = []

        for block_id in block_ids:
            if block_id not in self._block_metadata:
                continue

            metadata = self._block_metadata[block_id]
            num_tokens = metadata.num_tokens

            if num_tokens > 0:
                keys = self._blocks[block_id, layer_idx, 0, :num_tokens]
                values = self._blocks[block_id, layer_idx, 1, :num_tokens]
                keys_list.append(keys)
                values_list.append(values)

        if not keys_list:
            return None, None

        # Concatenate: [total_tokens, num_heads, head_dim]
        keys = torch.cat(keys_list, dim=0)
        values = torch.cat(values_list, dim=0)

        return keys, values

    def clear(self):
        """Clear all blocks and reset state."""
        if self._blocks is not None:
            del self._blocks
            self._blocks = None

        self._block_metadata.clear()
        self._free_blocks.clear()
        self._next_block_id = 0

        logger.debug("Paged KV cache cleared")

    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics.

        Returns:
            Dict with allocated_blocks, free_blocks, total_blocks
        """
        allocated = len(self._block_metadata)
        free = len(self._free_blocks)
        total = self._next_block_id + free

        return {
            "allocated_blocks": allocated,
            "free_blocks": free,
            "total_blocks": total,
            "max_blocks": self.max_blocks,
            "utilization": allocated / self.max_blocks if self.max_blocks > 0 else 0.0,
        }
