# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""KV cache manager for nano-vLLM style attention.

This module manages the allocation and assignment of KV cache buffers to
attention layers in a model.
"""

import torch
import torch.nn as nn
from typing import Optional


class NanoStyleKVCache:
    """Manages KV cache allocation and assignment to attention layers.

    This class:
    1. Allocates a single large KV cache tensor
    2. Assigns views of this tensor to each attention layer
    3. Provides cache management (clear, resize, etc.)

    Args:
        model: The model containing attention layers
        num_blocks: Number of KV cache blocks
        block_size: Size of each block (number of tokens)
        num_layers: Number of attention layers (auto-detected if None)
        num_kv_heads: Number of key/value heads (auto-detected if None)
        head_dim: Dimension of each head (auto-detected if None)
        dtype: Data type for cache (default: bfloat16)
        device: Device for cache (default: cuda)
    """

    def __init__(
        self,
        model: nn.Module,
        num_blocks: int,
        block_size: int = 16,
        num_layers: Optional[int] = None,
        num_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        dtype: torch.dtype = torch.bfloat16,
        device: str = 'cuda',
    ):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.dtype = dtype
        self.device = device

        # Find attention layers and extract parameters
        self.attention_layers = self._find_attention_layers(model)

        if not self.attention_layers:
            raise ValueError("No attention layers found in model")

        # Auto-detect parameters from first layer if not provided
        first_layer = self.attention_layers[0]
        self.num_layers = num_layers or len(self.attention_layers)
        self.num_kv_heads = num_kv_heads or getattr(
            first_layer, 'num_kv_heads',
            getattr(first_layer, 'n_kv_heads', None)
        )
        self.head_dim = head_dim or getattr(first_layer, 'head_dim', None)

        if self.num_kv_heads is None or self.head_dim is None:
            raise ValueError("Could not auto-detect num_kv_heads or head_dim")

        # Allocate KV cache
        self.kv_cache = self._allocate_cache()

        # Assign cache to layers
        self._assign_cache_to_layers()

        print(f"Allocated KV cache: {self.num_layers} layers, "
              f"{self.num_blocks} blocks, {self.block_size} tokens/block, "
              f"{self.num_kv_heads} KV heads, {self.head_dim} head dim")
        print(f"Total cache size: {self.kv_cache.numel() * self.kv_cache.element_size() / 1e9:.2f} GB")

    def _find_attention_layers(self, model: nn.Module) -> list[nn.Module]:
        """Find all attention layers in the model."""
        attention_layers = []

        def _find_recursive(module: nn.Module):
            # Check if this module has k_cache and v_cache attributes
            # (indicating it's a NanoStyleAttention layer)
            if hasattr(module, 'k_cache') and hasattr(module, 'v_cache'):
                attention_layers.append(module)
            else:
                for child in module.children():
                    _find_recursive(child)

        _find_recursive(model)
        return attention_layers

    def _allocate_cache(self) -> torch.Tensor:
        """Allocate the KV cache tensor.

        Returns:
            Tensor of shape [2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]
        """
        cache = torch.empty(
            2,  # K and V
            self.num_layers,
            self.num_blocks,
            self.block_size,
            self.num_kv_heads,
            self.head_dim,
            dtype=self.dtype,
            device=self.device,
        )

        return cache

    def _assign_cache_to_layers(self):
        """Assign cache views to each attention layer."""
        for layer_id, layer in enumerate(self.attention_layers):
            # Assign views into the cache tensor
            # k_cache shape: [num_blocks, block_size, num_kv_heads, head_dim]
            layer.k_cache = self.kv_cache[0, layer_id]
            layer.v_cache = self.kv_cache[1, layer_id]

    def clear(self):
        """Clear (zero out) the KV cache."""
        self.kv_cache.zero_()

    def get_memory_usage(self) -> dict:
        """Get memory usage statistics.

        Returns:
            Dict with memory statistics
        """
        total_bytes = self.kv_cache.numel() * self.kv_cache.element_size()

        return {
            'total_bytes': total_bytes,
            'total_gb': total_bytes / 1e9,
            'num_layers': self.num_layers,
            'num_blocks': self.num_blocks,
            'block_size': self.block_size,
            'num_kv_heads': self.num_kv_heads,
            'head_dim': self.head_dim,
            'dtype': str(self.dtype),
            'device': str(self.device),
        }


def estimate_kv_cache_blocks(
    gpu_memory_utilization: float = 0.5,
    num_layers: int = 28,
    block_size: int = 16,
    num_kv_heads: int = 8,
    head_dim: int = 128,
    dtype: torch.dtype = torch.bfloat16,
) -> int:
    """Estimate number of KV cache blocks that fit in GPU memory.

    Args:
        gpu_memory_utilization: Fraction of GPU memory to use for KV cache
        num_layers: Number of attention layers
        block_size: Size of each block (number of tokens)
        num_kv_heads: Number of key/value heads
        head_dim: Dimension of each head
        dtype: Data type for cache

    Returns:
        Estimated number of blocks
    """
    # Get available GPU memory
    if not torch.cuda.is_available():
        return 1000  # Default for CPU or testing

    free, total = torch.cuda.mem_get_info()
    available = total * gpu_memory_utilization

    # Calculate memory per block
    # 2 (K and V) * num_layers * block_size * num_kv_heads * head_dim * dtype_size
    dtype_size = torch.tensor([], dtype=dtype).element_size()
    bytes_per_block = 2 * num_layers * block_size * num_kv_heads * head_dim * dtype_size

    num_blocks = int(available / bytes_per_block)

    return max(num_blocks, 1)  # At least 1 block
