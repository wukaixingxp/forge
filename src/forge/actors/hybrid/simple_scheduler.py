# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Simple scheduler for single-batch generation without continuous batching.

This scheduler is a simplified version compared to vLLM's full scheduler:
- No continuous batching (fixed batch per generation call)
- No preemption or swapping
- Single-pass prefill, then decode loop
- But: supports prefix caching via BlockManager
"""

from typing import Optional
import torch

from forge.actors.hybrid.sequence import Sequence, SequenceStatus
from forge.actors.hybrid.block_manager import BlockManager
from forge.actors.hybrid.inference_context import InferenceContext


class SimpleScheduler:
    """Simplified scheduler for batch generation.

    This scheduler handles:
    1. Block allocation for sequences
    2. Prefill phase (process all prompt tokens)
    3. Decode phase (generate tokens one by one)
    4. Block deallocation when done

    Unlike vLLM's scheduler, this does NOT support:
    - Continuous batching (adding/removing sequences mid-generation)
    - Preemption (pausing sequences when memory is low)
    - Swapping (moving sequences to CPU)

    Args:
        block_manager: Block manager for KV cache allocation
        max_model_len: Maximum sequence length (default: 2048)
    """

    def __init__(
        self,
        block_manager: BlockManager,
        max_model_len: int = 2048,
    ):
        self.block_manager = block_manager
        self.max_model_len = max_model_len

        # Current batch of sequences
        self.sequences: list[Sequence] = []
        self.finished_sequences: list[Sequence] = []

    def add_sequence(self, seq: Sequence) -> bool:
        """Add a sequence to the scheduler.

        Args:
            seq: Sequence to add

        Returns:
            True if sequence was added successfully
        """
        # Check if we can allocate blocks
        if not self.block_manager.can_allocate(seq):
            return False

        # Allocate blocks
        self.block_manager.allocate(seq)
        seq.status = SequenceStatus.WAITING
        self.sequences.append(seq)
        return True

    def add_sequences(self, sequences: list[Sequence]) -> int:
        """Add multiple sequences.

        Args:
            sequences: List of sequences to add

        Returns:
            Number of sequences successfully added
        """
        added = 0
        for seq in sequences:
            if self.add_sequence(seq):
                added += 1
            else:
                break  # Stop if we run out of blocks
        return added

    def schedule_prefill(self) -> Optional[InferenceContext]:
        """Schedule prefill phase for waiting sequences.

        Returns:
            InferenceContext for prefill or None if no sequences waiting
        """
        waiting = [s for s in self.sequences if s.status == SequenceStatus.WAITING]
        if not waiting:
            return None

        # Mark as running
        for seq in waiting:
            seq.status = SequenceStatus.RUNNING

        # Create context for prefill
        context = InferenceContext(
            sequences=waiting,
            block_manager=self.block_manager,
            is_prefill=True,
        )

        return context

    def schedule_decode(self) -> Optional[InferenceContext]:
        """Schedule decode phase for running sequences.

        Returns:
            InferenceContext for decode or None if no sequences running
        """
        running = [s for s in self.sequences if s.status == SequenceStatus.RUNNING]
        if not running:
            return None

        # Prepare for token append (allocate new blocks if needed)
        for seq in running:
            if not self.block_manager.can_append(seq):
                # Out of memory - would need preemption in full vLLM
                raise RuntimeError(f"Out of blocks for sequence {seq.seq_id}")
            self.block_manager.may_append(seq)

        # Create context for decode
        context = InferenceContext(
            sequences=running,
            block_manager=self.block_manager,
            is_prefill=False,
        )

        return context

    def update_sequences(
        self,
        new_token_ids: torch.Tensor,
        eos_token_id: Optional[int] = None,
    ):
        """Update sequences with newly generated tokens.

        Args:
            new_token_ids: Tensor of shape [num_seqs] with new token IDs
            eos_token_id: EOS token ID to check for completion
        """
        running = [s for s in self.sequences if s.status == SequenceStatus.RUNNING]

        for i, seq in enumerate(running):
            token_id = new_token_ids[i].item()

            # Append token
            seq.append_token(token_id)

            # Check if finished
            is_eos = (eos_token_id is not None and token_id == eos_token_id)
            max_len_reached = seq.num_completion_tokens >= seq.max_tokens
            too_long = seq.num_tokens >= self.max_model_len

            if (is_eos and not seq.ignore_eos) or max_len_reached or too_long:
                seq.status = SequenceStatus.FINISHED
                self.finished_sequences.append(seq)

    def has_unfinished(self) -> bool:
        """Check if there are unfinished sequences.

        Returns:
            True if any sequences are waiting or running
        """
        return any(
            s.status in (SequenceStatus.WAITING, SequenceStatus.RUNNING)
            for s in self.sequences
        )

    def cleanup(self):
        """Deallocate blocks for finished sequences."""
        for seq in self.finished_sequences:
            self.block_manager.deallocate(seq)

        # Remove finished sequences
        self.sequences = [
            s for s in self.sequences
            if s.status != SequenceStatus.FINISHED
        ]
        self.finished_sequences.clear()

    def get_finished_sequences(self) -> list[Sequence]:
        """Get and clear finished sequences.

        Returns:
            List of finished sequences
        """
        finished = self.finished_sequences.copy()
        self.finished_sequences.clear()
        return finished

    def clear(self):
        """Clear all sequences and deallocate blocks."""
        for seq in self.sequences:
            self.block_manager.deallocate(seq)
        self.sequences.clear()
        self.finished_sequences.clear()

    def get_stats(self) -> dict:
        """Get scheduler statistics.

        Returns:
            Dict with statistics
        """
        waiting = sum(1 for s in self.sequences if s.status == SequenceStatus.WAITING)
        running = sum(1 for s in self.sequences if s.status == SequenceStatus.RUNNING)
        finished = len(self.finished_sequences)

        return {
            'waiting': waiting,
            'running': running,
            'finished': finished,
            'total': len(self.sequences) + len(self.finished_sequences),
            'block_stats': self.block_manager.get_stats(),
        }

    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"SimpleScheduler("
            f"waiting={stats['waiting']}, "
            f"running={stats['running']}, "
            f"finished={stats['finished']})"
        )
