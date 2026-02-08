# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Nano-vLLM style attention layer supporting both training and cached inference.

This module provides an attention layer that works in two modes:
1. Training mode: Standard flash attention (no KV cache)
2. Inference mode: Cached attention using nano-vLLM style KV cache

The key insight is using a context variable to switch modes without changing
the model architecture.
"""

import torch
import torch.nn as nn
from typing import Optional
import triton
import triton.language as tl

# Import flash attention functions
try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func, flash_attn_with_kvcache
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    print("Warning: flash_attn not available, will use slower attention")


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    """Triton kernel to store key/value into paged KV cache.

    Args:
        key_ptr: Pointer to key tensor [N, num_heads, head_dim]
        key_stride: Stride of key tensor
        value_ptr: Pointer to value tensor [N, num_heads, head_dim]
        value_stride: Stride of value tensor
        k_cache_ptr: Pointer to key cache [num_blocks, block_size, num_heads * head_dim]
        v_cache_ptr: Pointer to value cache [num_blocks, block_size, num_heads * head_dim]
        slot_mapping_ptr: Pointer to slot mapping [N]
        D: num_heads * head_dim
    """
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1:
        return

    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)

    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor
):
    """Store key/value into paged KV cache.

    Args:
        key: [N, num_heads, head_dim]
        value: [N, num_heads, head_dim]
        k_cache: [num_blocks, block_size, num_heads * head_dim]
        v_cache: [num_blocks, block_size, num_heads * head_dim]
        slot_mapping: [N] - maps token index to cache slot
    """
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim

    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N

    store_kvcache_kernel[(N,)](
        key, key.stride(0),
        value, value.stride(0),
        k_cache, v_cache, slot_mapping, D
    )


class NanoStyleAttention(nn.Module):
    """Attention layer supporting both training and cached inference.

    This layer can operate in two modes:
    - Training: Uses standard flash attention, no KV cache
    - Inference: Uses cached attention with nano-vLLM style paged KV cache

    Mode is determined by inference context (see inference_context.py).

    This class is designed to replace TorchTitan's Attention class and maintains
    the same interface and weight references.

    Args:
        num_heads: Number of attention heads
        head_dim: Dimension of each attention head
        num_kv_heads: Number of key/value heads (for GQA)
        scale: Attention scale factor (typically 1/sqrt(head_dim))
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        num_kv_heads: int,
        scale: float,
    ):
        super().__init__()
        self.n_heads = num_heads
        self.num_heads = num_heads  # Keep both for compatibility
        self.head_dim = head_dim
        self.n_kv_heads = num_kv_heads
        self.num_kv_heads = num_kv_heads  # Keep both for compatibility
        self.scaling = scale
        self.scale = scale  # Keep both for compatibility

        # Weight references (will be copied from original Attention during replacement)
        self.wq = None
        self.wk = None
        self.wv = None
        self.wo = None
        self.q_norm = None
        self.k_norm = None
        self.inner_attention = None

        # Reference to original TorchTitan Attention module (set during replacement)
        self.original_attention = None

        # KV cache buffers (assigned by KV cache manager)
        # Shape: [num_blocks, block_size, num_kv_heads, head_dim]
        self.k_cache = torch.tensor([])
        self.v_cache = torch.tensor([])

    def forward(
        self,
        x: torch.Tensor,
        rope_cache: torch.Tensor,
        attention_masks=None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass matching TorchTitan's Attention interface.

        Args:
            x: Input tensor [batch, seq_len, hidden_dim]
            rope_cache: RoPE cache
            attention_masks: Attention masks (not used in cached mode)
            positions: Position indices for RoPE (optional)

        Returns:
            output: Same shape as input
        """
        # Import here to avoid circular dependency
        from forge.actors.hybrid.inference_context import get_inference_context

        inference_context = get_inference_context()

        if inference_context is None:
            # Training mode: use original TorchTitan behavior
            return self._forward_training(x, rope_cache, attention_masks, positions)
        else:
            # Inference mode: use cached attention
            return self._forward_inference(x, rope_cache, positions, inference_context)

    def _forward_training(
        self,
        x: torch.Tensor,
        rope_cache: torch.Tensor,
        attention_masks,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Training mode: use inner_attention like original TorchTitan."""
        # This mirrors TorchTitan's Attention.forward() exactly
        bs, seqlen, _ = x.shape

        # Project to q, k, v
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # Reshape
        xq = xq.view(bs, seqlen, -1, self.head_dim)
        xk = xk.view(bs, seqlen, -1, self.head_dim)
        xv = xv.view(bs, seqlen, -1, self.head_dim)

        # Apply q/k norm if present
        if self.q_norm is not None:
            xq = self.q_norm(xq)
        if self.k_norm is not None:
            xk = self.k_norm(xk)

        # Apply RoPE
        from torchtitan.models.qwen3.model.model import apply_rotary_emb
        if positions is not None:
            xq, xk = apply_rotary_emb(xq, xk, rope_cache, positions)
        else:
            xq, xk = apply_rotary_emb(xq, xk, rope_cache)

        # Transpose to [bs, n_heads, seqlen, head_dim]
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # Handle GQA: expand KV heads to match Q heads for PyTorch SDPA
        if self.n_kv_heads < self.n_heads:
            n_rep = self.n_heads // self.n_kv_heads
            # Repeat KV heads: [bs, n_kv_heads, seqlen, head_dim] -> [bs, n_heads, seqlen, head_dim]
            xk = xk.repeat_interleave(n_rep, dim=1)
            xv = xv.repeat_interleave(n_rep, dim=1)

        # Use inner_attention (same as original)
        # Note: inner_attention is ScaledDotProductAttentionWrapper which doesn't take enable_gqa
        output = self.inner_attention(
            xq, xk, xv,
            scale=self.scaling,
        ).transpose(1, 2).contiguous()

        # Project output
        output = output.view(bs, seqlen, -1)
        return self.wo(output)

    def _forward_inference(
        self,
        x: torch.Tensor,
        rope_cache: torch.Tensor,
        positions: torch.Tensor | None,
        context: 'InferenceContext',
    ) -> torch.Tensor:
        """Inference mode: cached attention with nano-vLLM style KV cache."""
        if not FLASH_ATTN_AVAILABLE:
            raise RuntimeError("Flash attention required for cached inference")

        bs, seqlen, _ = x.shape

        # DIAGNOSTIC: Check input
        import logging
        logger = logging.getLogger(__name__)
        if torch.isnan(x).any():
            logger.error(f"[ATTN] Input x contains NaN! Count: {torch.isnan(x).sum()}")

        # Project to q, k, v
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # DIAGNOSTIC: Check projections
        if torch.isnan(xq).any():
            logger.error(f"[ATTN] xq contains NaN after projection! Count: {torch.isnan(xq).sum()}")
        if torch.isnan(xk).any():
            logger.error(f"[ATTN] xk contains NaN after projection! Count: {torch.isnan(xk).sum()}")
        if torch.isnan(xv).any():
            logger.error(f"[ATTN] xv contains NaN after projection! Count: {torch.isnan(xv).sum()}")

        # Reshape
        xq = xq.view(bs, seqlen, -1, self.head_dim)
        xk = xk.view(bs, seqlen, -1, self.head_dim)
        xv = xv.view(bs, seqlen, -1, self.head_dim)

        # Apply q/k norm if present
        if self.q_norm is not None:
            xq = self.q_norm(xq)
        if self.k_norm is not None:
            xk = self.k_norm(xk)

        # Apply RoPE
        from torchtitan.models.qwen3.model.model import apply_rotary_emb
        if positions is not None:
            xq, xk = apply_rotary_emb(xq, xk, rope_cache, positions)
        else:
            xq, xk = apply_rotary_emb(xq, xk, rope_cache)

        # DIAGNOSTIC: Check after RoPE
        if torch.isnan(xq).any():
            logger.error(f"[ATTN] xq contains NaN after RoPE! Count: {torch.isnan(xq).sum()}")
        if torch.isnan(xk).any():
            logger.error(f"[ATTN] xk contains NaN after RoPE! Count: {torch.isnan(xk).sum()}")

        # Flatten batch dimension: [bs, seqlen, n_heads, head_dim] -> [bs*seqlen, n_heads, head_dim]
        # CRITICAL: Make contiguous BEFORE reshape to ensure correct memory layout
        num_tokens = bs * seqlen
        xq = xq.contiguous()
        xk = xk.contiguous()
        xv = xv.contiguous()

        q = xq.view(num_tokens, self.n_heads, self.head_dim)
        k = xk.view(num_tokens, self.n_kv_heads, self.head_dim)
        v = xv.view(num_tokens, self.n_kv_heads, self.head_dim)

        # Store new key/value in cache
        if self.k_cache.numel() and self.v_cache.numel():
            store_kvcache(k, v, self.k_cache, self.v_cache, context.slot_mapping)

        if context.is_prefill:
            # Prefill: use varlen flash attention
            # CRITICAL: GQA support - expand KV heads to match Q heads
            if self.n_kv_heads < self.n_heads:
                n_rep = self.n_heads // self.n_kv_heads
                k = k.repeat_interleave(n_rep, dim=1).contiguous()
                v = v.repeat_interleave(n_rep, dim=1).contiguous()

            output = flash_attn_varlen_func(
                q, k, v,
                cu_seqlens_q=context.cu_seqlens_q,
                cu_seqlens_k=context.cu_seqlens_k,
                max_seqlen_q=context.max_seqlen_q,
                max_seqlen_k=context.max_seqlen_k,
                softmax_scale=self.scale,
                causal=True,
                block_table=None,  # Disabled for now
            )
        else:
            # Decode: use cached KV attention
            output = flash_attn_with_kvcache(
                q.unsqueeze(1),  # Add seq_len=1 dimension
                self.k_cache,
                self.v_cache,
                cache_seqlens=context.context_lens,
                block_table=context.block_tables,
                softmax_scale=self.scale,
                causal=True,
            )

        # DIAGNOSTIC: Check flash attention output
        if torch.isnan(output).any():
            logger.error(f"[ATTN] output contains NaN after flash_attn! Count: {torch.isnan(output).sum()}")

        # Reshape back: [bs, seqlen, n_heads, head_dim]
        output = output.view(bs, seqlen, self.n_heads, self.head_dim)

        # Project output
        output = output.view(bs, seqlen, -1)
        output = self.wo(output)

        # DIAGNOSTIC: Check final output
        if torch.isnan(output).any():
            logger.error(f"[ATTN] Final output contains NaN after wo projection! Count: {torch.isnan(output).sum()}")

        return output


def replace_attention_with_nano_style(model: nn.Module, attention_class_name: str = "Attention"):
    """Replace attention layers in model with NanoStyleAttention.

    This function finds all attention layers in the model and replaces them
    with NanoStyleAttention layers that support KV caching. It copies all
    weight references and attributes from the original attention.

    Args:
        model: The model to modify
        attention_class_name: Name of the attention class to replace

    Returns:
        Number of layers replaced
    """
    num_replaced = 0

    def _replace_recursive(module: nn.Module, parent: nn.Module, name: str):
        nonlocal num_replaced

        # Check if this is an attention layer
        if module.__class__.__name__ == attention_class_name:
            # Extract parameters
            num_heads = getattr(module, 'num_heads', getattr(module, 'n_heads', None))
            head_dim = getattr(module, 'head_dim', None)
            num_kv_heads = getattr(module, 'num_kv_heads', getattr(module, 'n_kv_heads', num_heads))
            scale = getattr(module, 'scale', getattr(module, 'scaling', None))

            if num_heads is None or head_dim is None:
                print(f"Warning: Could not extract attention parameters from {name}")
                return

            if scale is None:
                scale = 1.0 / (head_dim ** 0.5)

            # Create replacement
            new_attn = NanoStyleAttention(
                num_heads=num_heads,
                head_dim=head_dim,
                num_kv_heads=num_kv_heads,
                scale=scale,
            )

            # CRITICAL: Save reference to original attention module
            # This allows us to delegate to the original TorchTitan implementation
            new_attn.original_attention = module

            # Copy weight references from original attention
            new_attn.wq = module.wq
            new_attn.wk = module.wk
            new_attn.wv = module.wv
            new_attn.wo = module.wo

            # Copy optional attributes
            if hasattr(module, 'q_norm'):
                new_attn.q_norm = module.q_norm
            if hasattr(module, 'k_norm'):
                new_attn.k_norm = module.k_norm
            if hasattr(module, 'inner_attention'):
                new_attn.inner_attention = module.inner_attention

            # Copy other attributes that might be needed
            if hasattr(module, 'n_rep'):
                new_attn.n_rep = module.n_rep
            if hasattr(module, 'attn_type'):
                new_attn.attn_type = module.attn_type
            if hasattr(module, 'enable_gqa'):
                new_attn.enable_gqa = module.enable_gqa

            # Replace
            setattr(parent, name.split('.')[-1], new_attn)
            num_replaced += 1
            print(f"Replaced {name} with NanoStyleAttention (weights copied)")
        else:
            # Recurse into children
            for child_name, child_module in module.named_children():
                _replace_recursive(child_module, module, f"{name}.{child_name}")

    _replace_recursive(model, None, "model")

    return num_replaced
