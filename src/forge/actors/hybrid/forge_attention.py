# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Custom attention layer for Forge hybrid engine.

This module provides ForgeAttention which combines:
- TorchTitan's weight structure (wq, wk, wv, wo with Q-K norm)
- nano-vllm's inference patterns (explicit positions, KV cache integration)

The key insight is accepting explicit positions parameter and using position-indexed
RoPE, which allows proper varlen format support for inference.
"""

import torch
import torch.nn as nn
from typing import Optional
import logging
import os
import triton
import triton.language as tl

# Import flash attention functions
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

logger = logging.getLogger(__name__)

# Debug mode controlled by environment variable
DEBUG_MODE = os.environ.get('FORGE_DEBUG', '0') == '1'


def apply_rotary_emb_with_positions(
    q: torch.Tensor,
    k: torch.Tensor,
    positions: torch.Tensor,
    rope_cache: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE using explicit position indices.

    This is a position-indexed version compatible with varlen format.

    Args:
        q: Query tensor [batch, seq_len, num_heads, head_dim]
        k: Key tensor [batch, seq_len, num_kv_heads, head_dim]
        positions: Position indices [batch, seq_len]
        rope_cache: Precomputed rope cache [max_seq_len, head_dim * 2]

    Returns:
        Rotated query and key tensors
    """
    batch_size, seq_len, num_heads, head_dim = q.shape

    # Index rope cache by positions: [batch, seq_len, head_dim * 2]
    # positions: [batch, seq_len]
    rope_selected = rope_cache[positions]  # [batch, seq_len, head_dim * 2]

    # Split into cos and sin
    cos = rope_selected[..., :head_dim]  # [batch, seq_len, head_dim]
    sin = rope_selected[..., head_dim:]  # [batch, seq_len, head_dim]

    # Expand dims for broadcasting with num_heads
    cos = cos.unsqueeze(2)  # [batch, seq_len, 1, head_dim]
    sin = sin.unsqueeze(2)  # [batch, seq_len, 1, head_dim]

    # Apply rotation: q_out = q * cos + rotate_half(q) * sin
    def rotate_half(x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed.type_as(q), k_embed.type_as(k)


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
    """Triton kernel to store key/value into paged KV cache."""
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


class ForgeAttention(nn.Module):
    """Attention layer with explicit position support for inference.

    This layer combines TorchTitan's weight structure with nano-vllm's inference
    patterns. Key features:
    - Accepts explicit positions parameter (nano-vllm style)
    - Uses position-indexed RoPE
    - Supports KV caching with varlen format
    - Separate prefill/decode paths

    Args:
        num_heads: Number of attention heads
        head_dim: Dimension of each attention head
        num_kv_heads: Number of key/value heads (for GQA)
        scale: Attention scale factor (typically 1/sqrt(head_dim))
        qk_norm: Whether to apply RMSNorm to Q and K (Qwen3-specific)
        rms_norm_eps: Epsilon for RMSNorm
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        num_kv_heads: int,
        scale: float,
        qk_norm: bool = True,
        rms_norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.scale = scale
        self.qk_norm = qk_norm

        # Weight references (will be copied from TorchTitan attention)
        self.wq: Optional[nn.Module] = None
        self.wk: Optional[nn.Module] = None
        self.wv: Optional[nn.Module] = None
        self.wo: Optional[nn.Module] = None

        # Q-K normalization (Qwen3-specific, will be copied if present)
        self.q_norm: Optional[nn.Module] = None
        self.k_norm: Optional[nn.Module] = None

        # KV cache buffers (assigned by KV cache manager)
        self.k_cache = torch.tensor([])
        self.v_cache = torch.tensor([])

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass with explicit positions.

        This is the nano-vllm style signature with explicit positions parameter.

        Args:
            positions: Position indices [total_tokens] or [batch, seq_len]
            hidden_states: Input tensor [total_tokens, hidden_dim] or [batch, seq_len, hidden_dim]

        Returns:
            output: Same shape as hidden_states
        """
        # Import here to avoid circular dependency
        from forge.actors.hybrid.inference_context import get_inference_context

        context = get_inference_context()

        if context is None:
            # Training mode: standard attention without KV cache
            return self._forward_training(positions, hidden_states)
        else:
            # Inference mode: cached attention with varlen format
            return self._forward_inference(positions, hidden_states, context)

    def _forward_training(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """Training mode: standard attention without KV cache."""
        # Determine input format
        if hidden_states.dim() == 2:
            # Varlen format: [total_tokens, hidden_dim]
            total_tokens, hidden_dim = hidden_states.shape
            batch_size = 1
            seq_len = total_tokens
            # Add batch dimension
            hidden_states = hidden_states.unsqueeze(0)
            positions = positions.unsqueeze(0)
        else:
            # Batched format: [batch, seq_len, hidden_dim]
            batch_size, seq_len, hidden_dim = hidden_states.shape

        # Project to Q, K, V
        q = self.wq(hidden_states)  # [batch, seq_len, num_heads * head_dim]
        k = self.wk(hidden_states)  # [batch, seq_len, num_kv_heads * head_dim]
        v = self.wv(hidden_states)  # [batch, seq_len, num_kv_heads * head_dim]

        # Reshape: [batch, seq_len, num_heads, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # Apply Q-K normalization if enabled
        if self.q_norm is not None:
            q = self.q_norm(q)
        if self.k_norm is not None:
            k = self.k_norm(k)

        # Apply RoPE with explicit positions
        from torchtitan.models.qwen3.model.model import precompute_rope_cache

        # Get rope cache
        rope_dim = self.head_dim
        rope_base = 1000000  # Qwen3 default
        max_seq_len = positions.max().item() + 1 if positions.numel() > 0 else 4096
        rope_cache = precompute_rope_cache(rope_dim, max_seq_len, rope_base)
        rope_cache = rope_cache.to(hidden_states.device)

        q, k = apply_rotary_emb_with_positions(q, k, positions, rope_cache)

        # Transpose to [batch, num_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # GQA: Expand KV heads to match Q heads
        if self.num_kv_heads < self.num_heads:
            n_rep = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)

        # Scaled dot product attention
        output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
            scale=self.scale,
        )

        # Transpose back: [batch, seq_len, num_heads, head_dim]
        output = output.transpose(1, 2).contiguous()

        # Reshape and project
        output = output.view(batch_size, seq_len, self.num_heads * self.head_dim)
        output = self.wo(output)

        # Remove batch dimension if input was varlen
        if batch_size == 1 and total_tokens > 1:
            output = output.squeeze(0)

        return output

    def _forward_inference(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        context: 'InferenceContext',
    ) -> torch.Tensor:
        """Inference mode: cached attention with varlen format."""

        # Determine input format
        if hidden_states.dim() == 2:
            # Varlen format: [total_tokens, hidden_dim]
            total_tokens, hidden_dim = hidden_states.shape
            batch_size = 1
            seq_len = total_tokens
        else:
            # Batched format: [batch, seq_len, hidden_dim]
            batch_size, seq_len, hidden_dim = hidden_states.shape
            total_tokens = batch_size * seq_len
            # Flatten to varlen format
            hidden_states = hidden_states.view(total_tokens, hidden_dim)
            positions = positions.view(total_tokens)

        # Project to Q, K, V
        q = self.wq(hidden_states)  # [total_tokens, num_heads * head_dim]
        k = self.wk(hidden_states)  # [total_tokens, num_kv_heads * head_dim]
        v = self.wv(hidden_states)  # [total_tokens, num_kv_heads * head_dim]

        # DIAGNOSTIC: Check projections

        # Reshape: [total_tokens, num_heads, head_dim]
        q = q.view(total_tokens, self.num_heads, self.head_dim)
        k = k.view(total_tokens, self.num_kv_heads, self.head_dim)
        v = v.view(total_tokens, self.num_kv_heads, self.head_dim)

        # Apply Q-K normalization if enabled
        if self.q_norm is not None:
            q = self.q_norm(q)
        if self.k_norm is not None:
            k = self.k_norm(k)

        # Apply RoPE with explicit positions
        from torchtitan.models.qwen3.model.model import precompute_rope_cache

        # Create rope cache
        rope_dim = self.head_dim
        rope_base = 1000000  # Qwen3 default
        max_seq_len = positions.max().item() + 1 if positions.numel() > 0 else 4096
        rope_cache = precompute_rope_cache(rope_dim, max_seq_len, rope_base)
        rope_cache = rope_cache.to(hidden_states.device)


        # Add batch dimension for apply_rotary_emb_with_positions
        q_batched = q.unsqueeze(0)  # [1, total_tokens, num_heads, head_dim]
        k_batched = k.unsqueeze(0)  # [1, total_tokens, num_kv_heads, head_dim]
        positions_batched = positions.unsqueeze(0)  # [1, total_tokens]

        q_batched, k_batched = apply_rotary_emb_with_positions(
            q_batched, k_batched, positions_batched, rope_cache
        )

        # Remove batch dimension
        q = q_batched.squeeze(0)
        k = k_batched.squeeze(0)

        if context.is_prefill:
            # Prefill: use varlen flash attention
            # Note: flash_attn_varlen_func does NOT support GQA natively,
            # so we must expand K/V heads to match Q heads
            if self.num_kv_heads < self.num_heads:
                n_rep = self.num_heads // self.num_kv_heads
                k_expanded = k.repeat_interleave(n_rep, dim=1)
                v_expanded = v.repeat_interleave(n_rep, dim=1)
            else:
                k_expanded = k
                v_expanded = v

            if DEBUG_MODE:
                logger.info(f"[FORGE_ATTN] Prefill: total_tokens={total_tokens}, "
                           f"q shape={q.shape}, k shape={k_expanded.shape}")
                logger.info(f"[FORGE_ATTN] Prefill positions: {positions[:10].tolist() if positions.numel() < 10 else positions[:10].tolist()}")

            # Store UNEXPANDED K/V in cache (flash_attn_with_kvcache supports GQA)
            if self.k_cache.numel() and self.v_cache.numel():
                store_kvcache(k, v, self.k_cache, self.v_cache, context.slot_mapping)

            output = flash_attn_varlen_func(
                q, k_expanded, v_expanded,
                cu_seqlens_q=context.cu_seqlens_q,
                cu_seqlens_k=context.cu_seqlens_k,
                max_seqlen_q=context.max_seqlen_q,
                max_seqlen_k=context.max_seqlen_k,
                softmax_scale=self.scale,
                causal=True,
            )
        else:
            # Decode: use cached KV attention
            # flash_attn_with_kvcache supports GQA natively, so use unexpanded K/V cache

            if DEBUG_MODE:
                logger.info(f"[FORGE_ATTN] Decode: total_tokens={total_tokens}, num_seqs={len(context.sequences)}")
                logger.info(f"[FORGE_ATTN] Decode positions: {positions.tolist()}")
                logger.info(f"[FORGE_ATTN] cache_seqlens: {context.context_lens.tolist()}")
                logger.info(f"[FORGE_ATTN] block_tables shape: {context.block_tables.shape}, first seq blocks: {context.block_tables[0][:5].tolist()}")
                logger.info(f"[FORGE_ATTN] q stats: min={q.min().item():.4f}, max={q.max().item():.4f}, has_nan={torch.isnan(q).any()}")
                logger.info(f"[FORGE_ATTN] k stats: min={k.min().item():.4f}, max={k.max().item():.4f}, has_nan={torch.isnan(k).any()}")
                logger.info(f"[FORGE_ATTN] v stats: min={v.min().item():.4f}, max={v.max().item():.4f}, has_nan={torch.isnan(v).any()}")

            # CRITICAL: We need to store the NEW K/V into cache for this decode step!
            # flash_attn_with_kvcache can update the cache in-place when k and v are provided
            output = flash_attn_with_kvcache(
                q.unsqueeze(1),  # Add seq_len=1 dimension: [total_tokens, 1, num_heads, head_dim]
                self.k_cache,
                self.v_cache,
                k=k.unsqueeze(1),  # New K to append: [total_tokens, 1, num_kv_heads, head_dim]
                v=v.unsqueeze(1),  # New V to append: [total_tokens, 1, num_kv_heads, head_dim]
                cache_seqlens=context.context_lens,
                block_table=context.block_tables,
                softmax_scale=self.scale,
                causal=True,
            ).squeeze(1)  # Remove seq_len dimension

            if DEBUG_MODE:
                logger.info(f"[FORGE_ATTN] output stats: min={output.min().item():.4f}, max={output.max().item():.4f}, has_nan={torch.isnan(output).any()}")

        # Project output: [total_tokens, num_heads * head_dim]
        output = output.view(total_tokens, self.num_heads * self.head_dim)

        output = self.wo(output)

        # Reshape back to original format if needed
        if batch_size > 1 or (batch_size == 1 and hidden_states.dim() == 3):
            output = output.view(batch_size, seq_len, -1)

        return output


def replace_attention_with_forge(model: nn.Module, attention_class_name: str = "Attention") -> int:
    """Replace attention layers in TorchTitan model with ForgeAttention.

    This function finds all attention layers and replaces them with ForgeAttention
    layers that support explicit positions and KV caching. It copies all weight
    references from the original attention.

    Args:
        model: The TorchTitan model to modify
        attention_class_name: Name of the attention class to replace (default: "Attention")

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
                logger.warning(f"Could not extract attention parameters from {name}")
                return

            if scale is None:
                scale = 1.0 / (head_dim ** 0.5)

            # Determine if Q-K norm is used
            qk_norm = hasattr(module, 'q_norm') and hasattr(module, 'k_norm')

            # Create ForgeAttention replacement
            new_attn = ForgeAttention(
                num_heads=num_heads,
                head_dim=head_dim,
                num_kv_heads=num_kv_heads,
                scale=scale,
                qk_norm=qk_norm,
            )

            # Copy weight references (not copies - share the same tensors)
            new_attn.wq = module.wq
            new_attn.wk = module.wk
            new_attn.wv = module.wv
            new_attn.wo = module.wo

            # Copy Q-K norm if present
            if hasattr(module, 'q_norm'):
                new_attn.q_norm = module.q_norm
            if hasattr(module, 'k_norm'):
                new_attn.k_norm = module.k_norm

            # Replace in parent module
            setattr(parent, name.split('.')[-1], new_attn)
            num_replaced += 1
            logger.info(f"Replaced {name} with ForgeAttention (weights shared)")

            # Verify weight sharing
            assert id(new_attn.wq.weight) == id(module.wq.weight), "Weight sharing verification failed!"

        else:
            # Recurse into children
            for child_name, child_module in module.named_children():
                _replace_recursive(child_module, module, f"{name}.{child_name}")

    # Start recursion from model root
    for name, child in model.named_children():
        _replace_recursive(child, model, name)

    logger.info(f"Replaced {num_replaced} attention layers with ForgeAttention")
    return num_replaced
