# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Prefix caching for shared prompt prefixes.

This module implements hash-based prefix matching to reuse KV cache
for shared prompt prefixes, significantly speeding up RL training where
many prompts share common system messages or few-shot examples.

Example:
    Prompt 1: [system_msg] + [user_prompt_1]
    Prompt 2: [system_msg] + [user_prompt_2]
             ^^^^^^^^^^^^^ Shared KV cache (30-50% of tokens)

Expected impact: 2-5x speedup for prompts with common prefixes.
"""

import hashlib
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Entry in the prefix cache.

    Args:
        token_ids: The token IDs for this prefix
        kv_cache: Cached key-value tensors for this prefix
        ref_count: Reference count for cache eviction
        last_used: Timestamp of last access
    """
    token_ids: List[int]
    kv_cache: Tuple[torch.Tensor, torch.Tensor]  # (keys, values)
    ref_count: int = 0
    last_used: float = 0.0


class PrefixCache:
    """Hash-based prefix cache for KV tensors.

    Stores and retrieves KV cache for common prompt prefixes,
    avoiding redundant computation for shared portions of prompts.

    Args:
        max_entries: Maximum number of cached prefixes
        min_prefix_len: Minimum prefix length to cache (in tokens)
    """

    def __init__(
        self,
        max_entries: int = 1000,
        min_prefix_len: int = 10,
    ):
        self.max_entries = max_entries
        self.min_prefix_len = min_prefix_len
        self._cache: Dict[str, CacheEntry] = {}
        self._access_count = 0
        self._hit_count = 0

        logger.info(
            f"PrefixCache initialized (max_entries={max_entries}, "
            f"min_prefix_len={min_prefix_len})"
        )

    def _compute_hash(self, token_ids: List[int]) -> str:
        """Compute hash of token sequence.

        Args:
            token_ids: List of token IDs

        Returns:
            Hash string
        """
        # Use SHA256 for collision resistance
        # Convert token IDs to bytes using numpy (token IDs can be > 255)
        import numpy as np
        token_array = np.array(token_ids, dtype=np.int32)
        token_bytes = token_array.tobytes()
        return hashlib.sha256(token_bytes).hexdigest()

    def find_longest_prefix(
        self,
        token_ids: List[int],
    ) -> Optional[Tuple[List[int], Tuple[torch.Tensor, torch.Tensor]]]:
        """Find longest cached prefix matching the input.

        Args:
            token_ids: Input token sequence

        Returns:
            Tuple of (matched_token_ids, kv_cache) if found, None otherwise
        """
        self._access_count += 1

        if len(token_ids) < self.min_prefix_len:
            return None

        # Try progressively shorter prefixes
        for prefix_len in range(len(token_ids), self.min_prefix_len - 1, -1):
            prefix = token_ids[:prefix_len]
            prefix_hash = self._compute_hash(prefix)

            if prefix_hash in self._cache:
                entry = self._cache[prefix_hash]
                entry.ref_count += 1
                entry.last_used = self._access_count
                self._hit_count += 1

                logger.debug(
                    f"Cache hit: prefix_len={prefix_len}/{len(token_ids)} "
                    f"(hit_rate={self.hit_rate:.2%})"
                )

                return (entry.token_ids, entry.kv_cache)

        return None

    def insert(
        self,
        token_ids: List[int],
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
    ):
        """Insert a prefix and its KV cache.

        Args:
            token_ids: Token sequence
            kv_cache: Tuple of (keys, values) tensors
        """
        if len(token_ids) < self.min_prefix_len:
            return

        prefix_hash = self._compute_hash(token_ids)

        # Check if we need to evict entries
        if len(self._cache) >= self.max_entries and prefix_hash not in self._cache:
            self._evict_lru()

        # Clone tensors to avoid sharing memory
        kv_cache_cloned = (
            kv_cache[0].clone() if kv_cache[0] is not None else None,
            kv_cache[1].clone() if kv_cache[1] is not None else None,
        )

        self._cache[prefix_hash] = CacheEntry(
            token_ids=token_ids.copy() if isinstance(token_ids, list) else token_ids,
            kv_cache=kv_cache_cloned,
            ref_count=0,
            last_used=self._access_count,
        )

        logger.debug(f"Cached prefix: len={len(token_ids)}, cache_size={len(self._cache)}")

    def _evict_lru(self):
        """Evict least recently used entry."""
        if not self._cache:
            return

        # Find entry with lowest last_used timestamp
        lru_hash = min(self._cache.keys(), key=lambda k: self._cache[k].last_used)
        evicted = self._cache.pop(lru_hash)

        # Free GPU memory
        if evicted.kv_cache[0] is not None:
            del evicted.kv_cache[0]
        if evicted.kv_cache[1] is not None:
            del evicted.kv_cache[1]

        logger.debug(f"Evicted LRU entry: len={len(evicted.token_ids)}")

    def clear(self):
        """Clear all cached entries."""
        # Free GPU memory
        for entry in self._cache.values():
            if entry.kv_cache[0] is not None:
                del entry.kv_cache[0]
            if entry.kv_cache[1] is not None:
                del entry.kv_cache[1]

        self._cache.clear()
        self._access_count = 0
        self._hit_count = 0
        logger.debug("Prefix cache cleared")

    @property
    def hit_rate(self) -> float:
        """Compute cache hit rate."""
        if self._access_count == 0:
            return 0.0
        return self._hit_count / self._access_count

    @property
    def size(self) -> int:
        """Number of cached entries."""
        return len(self._cache)

    def get_stats(self) -> Dict[str, float]:
        """Get cache statistics.

        Returns:
            Dict with hit_rate, size, access_count, hit_count
        """
        return {
            "hit_rate": self.hit_rate,
            "size": self.size,
            "access_count": self._access_count,
            "hit_count": self._hit_count,
        }
