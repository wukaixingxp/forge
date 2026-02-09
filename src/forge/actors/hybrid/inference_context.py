# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Inference context for passing metadata to attention layers.

This module uses contextvars to provide thread-safe context switching between
training mode (no context) and inference mode (with KV cache metadata).
"""

from contextvars import ContextVar
from contextlib import contextmanager
from typing import Optional
import torch

from forge.actors.hybrid.sequence import Sequence
from forge.actors.hybrid.block_manager import BlockManager


# Global context variable (thread-safe)
_inference_context: ContextVar[Optional['InferenceContext']] = ContextVar(
    'inference_context',
    default=None
)


class InferenceContext:
    """Inference metadata passed to attention layers during cached inference.

    This context provides all information needed by NanoStyleAttention layers
    to use the KV cache during inference:

    - slot_mapping: Maps each token to its cache slot
    - block_tables: Block tables for each sequence
    - context_lens: Context length for each sequence
    - is_prefill: Whether this is prefill (True) or decode (False)

    Attributes:
        sequences: List of sequences being processed
        block_manager: Block manager for cache allocation
        is_prefill: True for prefill, False for decode
        slot_mapping: Tensor mapping tokens to cache slots
        block_tables: Tensor of block tables (padded)
        context_lens: Tensor of context lengths
        cu_seqlens_q: Cumulative sequence lengths for query (prefill only)
        cu_seqlens_k: Cumulative sequence lengths for key (prefill only)
        max_seqlen_q: Max sequence length for query (prefill only)
        max_seqlen_k: Max sequence length for key (prefill only)
    """

    def __init__(
        self,
        sequences: list[Sequence],
        block_manager: BlockManager,
        is_prefill: bool = True,
    ):
        self.sequences = sequences
        self.block_manager = block_manager
        self.is_prefill = is_prefill

        # Prepare metadata
        if is_prefill:
            self._prepare_prefill_metadata()
        else:
            self._prepare_decode_metadata()

    def _prepare_prefill_metadata(self):
        """Prepare metadata for prefill (first forward pass)."""
        # For prefill, we process all prompt tokens
        # Slot mapping: sequential assignment of tokens to cache slots
        slot_mapping = []
        block_tables = []
        context_lens = []

        cu_seqlens = [0]
        total_tokens = 0

        for seq in self.sequences:
            # Build slot mapping for this sequence
            seq_slots = []
            for block_idx, block_id in enumerate(seq.block_table):
                block_start = block_idx * self.block_manager.block_size
                block_tokens = min(
                    self.block_manager.block_size,
                    seq.num_tokens - block_start
                )
                for token_idx in range(block_tokens):
                    slot = block_id * self.block_manager.block_size + token_idx
                    seq_slots.append(slot)

            slot_mapping.extend(seq_slots)
            block_tables.append(seq.block_table)
            context_lens.append(seq.num_tokens)

            total_tokens += seq.num_tokens
            cu_seqlens.append(total_tokens)

        # Convert to tensors
        self.slot_mapping = torch.tensor(slot_mapping, dtype=torch.long, device='cuda')

        # Pad block tables to same length
        max_blocks = max(len(bt) for bt in block_tables)
        padded_block_tables = []
        for bt in block_tables:
            padded = bt + [-1] * (max_blocks - len(bt))
            padded_block_tables.append(padded)
        self.block_tables = torch.tensor(padded_block_tables, dtype=torch.int32, device='cuda')

        self.context_lens = torch.tensor(context_lens, dtype=torch.int32, device='cuda')

        # For varlen flash attention
        self.cu_seqlens_q = torch.tensor(cu_seqlens, dtype=torch.int32, device='cuda')
        self.cu_seqlens_k = torch.tensor(cu_seqlens, dtype=torch.int32, device='cuda')
        self.max_seqlen_q = max(context_lens)
        self.max_seqlen_k = max(context_lens)

    def _prepare_decode_metadata(self):
        """Prepare metadata for decode (subsequent tokens)."""
        # For decode, we only process 1 new token per sequence
        slot_mapping = []
        block_tables = []
        context_lens = []

        import logging
        import os
        logger = logging.getLogger(__name__)
        DEBUG = os.environ.get('FORGE_DEBUG', '0') == '1'

        for seq in self.sequences:
            # Find slot for new token
            last_block_id = seq.block_table[-1]
            token_offset = (seq.num_tokens - 1) % self.block_manager.block_size
            slot = last_block_id * self.block_manager.block_size + token_offset
            slot_mapping.append(slot)

            if DEBUG:
                logger.info(f"[DECODE_META] Seq {seq.seq_id}: block_table={seq.block_table}, num_tokens={seq.num_tokens}, slot={slot}")

            block_tables.append(seq.block_table)
            context_lens.append(seq.num_tokens - 1)  # Context = all tokens except new one

        # Convert to tensors
        self.slot_mapping = torch.tensor(slot_mapping, dtype=torch.long, device='cuda')

        # Pad block tables
        max_blocks = max(len(bt) for bt in block_tables)
        padded_block_tables = []
        for bt in block_tables:
            padded = bt + [-1] * (max_blocks - len(bt))
            padded_block_tables.append(padded)

        if DEBUG:
            logger.info(f"[DECODE_META] block_tables before tensor: {block_tables}")
            logger.info(f"[DECODE_META] padded_block_tables: {padded_block_tables}")

        self.block_tables = torch.tensor(padded_block_tables, dtype=torch.int32, device='cuda')

        if DEBUG:
            logger.info(f"[DECODE_META] block_tables tensor: {self.block_tables.tolist()}")

        self.context_lens = torch.tensor(context_lens, dtype=torch.int32, device='cuda')

        # Decode doesn't need cu_seqlens (single token per sequence)
        self.cu_seqlens_q = None
        self.cu_seqlens_k = None
        self.max_seqlen_q = 1
        self.max_seqlen_k = max(context_lens)

    def __repr__(self) -> str:
        return (
            f"InferenceContext("
            f"num_seqs={len(self.sequences)}, "
            f"is_prefill={self.is_prefill}, "
            f"total_tokens={len(self.slot_mapping)})"
        )


def get_inference_context() -> Optional[InferenceContext]:
    """Get the current inference context.

    Returns:
        Current InferenceContext or None if in training mode
    """
    return _inference_context.get()


@contextmanager
def inference_context(
    sequences: list[Sequence],
    block_manager: BlockManager,
    is_prefill: bool = True,
):
    """Context manager for inference mode.

    Usage:
        with inference_context(sequences, block_manager, is_prefill=True):
            output = model(input_ids)  # Attention layers use KV cache

    Args:
        sequences: List of sequences being processed
        block_manager: Block manager for cache allocation
        is_prefill: True for prefill, False for decode

    Yields:
        InferenceContext instance
    """
    ctx = InferenceContext(sequences, block_manager, is_prefill)
    token = _inference_context.set(ctx)
    try:
        yield ctx
    finally:
        _inference_context.reset(token)
