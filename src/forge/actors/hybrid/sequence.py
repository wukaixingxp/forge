# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Sequence class for managing token sequences and KV cache blocks.

This is a simplified version adapted from nano-vLLM for our hybrid training/inference use case.
"""

from copy import copy
from enum import Enum, auto
from itertools import count
from typing import Optional


class SequenceStatus(Enum):
    """Status of a sequence during generation."""
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    """Represents a token sequence with block-based KV cache.

    Attributes:
        seq_id: Unique sequence identifier
        token_ids: List of token IDs (prompt + generated)
        block_table: List of block IDs allocated for this sequence
        num_cached_tokens: Number of tokens with cached KV
        status: Current status (WAITING, RUNNING, FINISHED)
    """

    block_size = 16  # Tokens per block (configurable)
    counter = count()

    def __init__(
        self,
        token_ids: list[int],
        max_tokens: int = 512,
        temperature: float = 1.0,
        ignore_eos: bool = False,
    ):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.block_table: list[int] = []

        # Sampling parameters
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.ignore_eos = ignore_eos

        # Logprobs for generated tokens
        self.logprobs: list[float] = []

    def __len__(self) -> int:
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self) -> bool:
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self) -> int:
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self) -> list[int]:
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self) -> list[int]:
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_cached_blocks(self) -> int:
        return self.num_cached_tokens // self.block_size

    @property
    def num_blocks(self) -> int:
        """Number of blocks needed for current tokens."""
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self) -> int:
        """Number of tokens in the last block."""
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i: int) -> list[int]:
        """Get token IDs for block i."""
        assert 0 <= i < self.num_blocks
        start = i * self.block_size
        end = min((i + 1) * self.block_size, self.num_tokens)
        return self.token_ids[start:end]

    def append_token(self, token_id: int):
        """Add a generated token to the sequence."""
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def __repr__(self) -> str:
        return (
            f"Sequence(seq_id={self.seq_id}, "
            f"num_tokens={self.num_tokens}, "
            f"num_blocks={self.num_blocks}, "
            f"status={self.status.name})"
        )
