# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Prefix cache for reusing KV cache across prompts with shared prefixes.

This module implements a simple LRU cache that stores KV cache slices for
common prompt prefixes, enabling significant speedup for workloads with
repeated system prompts (e.g., RLHF/GRPO).

Example:
    All prompts start with "You are a helpful assistant. "
    → Cache the KV for this prefix once, reuse for all subsequent prompts
    → Saves 2-5x on prefill time for the shared prefix
"""

import logging
from collections import OrderedDict
from typing import Optional, Tuple, List
import torch

logger = logging.getLogger(__name__)


class PrefixCache:
    """LRU cache for prompt prefix KV states.

    This cache stores KV cache slices for common prompt prefixes to avoid
    recomputing attention for repeated prefixes. It's particularly effective
    for RLHF/GRPO workloads where all prompts share a common system prompt.

    The cache uses token IDs as keys (not the actual text) and stores references
    to the KV cache blocks rather than copying the KV tensors.

    Args:
        max_entries: Maximum number of prefix entries to cache (LRU eviction)
        min_prefix_len: Minimum prefix length to cache (tokens)
        enable_stats: Whether to track cache statistics
    """

    def __init__(
        self,
        max_entries: int = 1000,
        min_prefix_len: int = 10,
        enable_stats: bool = True,
    ):
        self.max_entries = max_entries
        self.min_prefix_len = min_prefix_len
        self.enable_stats = enable_stats

        # LRU cache: token_prefix_tuple -> (block_table, num_cached_tokens)
        # Using OrderedDict for LRU behavior
        self.cache: OrderedDict[Tuple[int, ...], Tuple[List[int], int]] = OrderedDict()

        # Statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'inserts': 0,
            'evictions': 0,
            'total_tokens_saved': 0,
        }

        logger.info(
            f"PrefixCache initialized: max_entries={max_entries}, "
            f"min_prefix_len={min_prefix_len}"
        )

    def find_longest_prefix(
        self,
        token_ids: List[int],
    ) -> Optional[Tuple[List[int], List[int], int]]:
        """Find the longest cached prefix for given token sequence.

        Args:
            token_ids: Token IDs to search for

        Returns:
            If cache hit: (matched_tokens, block_table, num_cached_tokens)
            If cache miss: None
        """
        if len(token_ids) < self.min_prefix_len:
            return None

        # Try progressively shorter prefixes (longest first)
        for prefix_len in range(len(token_ids), self.min_prefix_len - 1, -1):
            prefix_tuple = tuple(token_ids[:prefix_len])

            if prefix_tuple in self.cache:
                # Cache hit!
                block_table, num_cached = self.cache[prefix_tuple]

                # Move to end (most recently used)
                self.cache.move_to_end(prefix_tuple)

                if self.enable_stats:
                    self.stats['hits'] += 1
                    self.stats['total_tokens_saved'] += num_cached

                logger.debug(
                    f"Prefix cache HIT: {num_cached}/{len(token_ids)} tokens "
                    f"(hit rate: {self.get_hit_rate():.1%})"
                )

                return (token_ids[:prefix_len], block_table, num_cached)

        # No cache hit
        if self.enable_stats:
            self.stats['misses'] += 1

        logger.debug(
            f"Prefix cache MISS for {len(token_ids)} tokens "
            f"(hit rate: {self.get_hit_rate():.1%})"
        )

        return None

    def insert(
        self,
        token_ids: List[int],
        block_table: List[int],
        num_cached_tokens: int,
    ):
        """Insert a prefix into the cache.

        Args:
            token_ids: Token IDs of the prefix
            block_table: Block table for this prefix's KV cache
            num_cached_tokens: Number of tokens with cached KV
        """
        if len(token_ids) < self.min_prefix_len:
            logger.debug(
                f"Prefix too short to cache: {len(token_ids)} < {self.min_prefix_len}"
            )
            return

        prefix_tuple = tuple(token_ids)

        # Check if we need to evict (before insertion to ensure we stay at max)
        if len(self.cache) >= self.max_entries and prefix_tuple not in self.cache:
            # Evict least recently used (first item)
            evicted_key, evicted_value = self.cache.popitem(last=False)
            if self.enable_stats:
                self.stats['evictions'] += 1
            logger.debug(
                f"Evicted LRU prefix: {len(evicted_key)} tokens "
                f"(cache size: {len(self.cache)})"
            )

        # Insert or update
        self.cache[prefix_tuple] = (block_table.copy(), num_cached_tokens)

        # Move to end (most recently used)
        self.cache.move_to_end(prefix_tuple)

        if self.enable_stats:
            self.stats['inserts'] += 1

        logger.debug(
            f"Cached prefix: {len(token_ids)} tokens, "
            f"{len(block_table)} blocks "
            f"(cache size: {len(self.cache)}/{self.max_entries})"
        )

    def clear(self):
        """Clear all cached entries."""
        num_entries = len(self.cache)
        self.cache.clear()

        logger.info(
            f"Cleared prefix cache: {num_entries} entries removed, "
            f"{self.stats['total_tokens_saved']} tokens saved (lifetime)"
        )

        # Reset stats except lifetime counters
        if self.enable_stats:
            self.stats = {
                'hits': 0,
                'misses': 0,
                'inserts': 0,
                'evictions': 0,
                'total_tokens_saved': self.stats['total_tokens_saved'],
            }

    def get_hit_rate(self) -> float:
        """Calculate cache hit rate.

        Returns:
            Hit rate as a fraction (0.0 to 1.0)
        """
        total_queries = self.stats['hits'] + self.stats['misses']
        if total_queries == 0:
            return 0.0
        return self.stats['hits'] / total_queries

    def get_stats(self) -> dict:
        """Get cache statistics.

        Returns:
            Dict with statistics
        """
        total_queries = self.stats['hits'] + self.stats['misses']
        hit_rate = self.get_hit_rate()

        return {
            'num_entries': len(self.cache),
            'max_entries': self.max_entries,
            'min_prefix_len': self.min_prefix_len,
            'cache_hits': self.stats['hits'],
            'cache_misses': self.stats['misses'],
            'total_queries': total_queries,
            'hit_rate': hit_rate,
            'inserts': self.stats['inserts'],
            'evictions': self.stats['evictions'],
            'total_tokens_saved': self.stats['total_tokens_saved'],
            'avg_tokens_saved_per_hit': (
                self.stats['total_tokens_saved'] / self.stats['hits']
                if self.stats['hits'] > 0
                else 0.0
            ),
        }

    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"PrefixCache("
            f"entries={stats['num_entries']}/{stats['max_entries']}, "
            f"hit_rate={stats['hit_rate']:.1%}, "
            f"tokens_saved={stats['total_tokens_saved']})"
        )
