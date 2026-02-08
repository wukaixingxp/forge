# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CUDA graphs for efficient autoregressive decoding.

This module captures the decode phase computation as a CUDA graph
to eliminate kernel launch overhead. This significantly speeds up
token-by-token generation.

Benefits:
- Eliminates kernel launch overhead
- Fixed memory layout for replay
- Works with FSDP-sharded models

Expected impact: 1.3-1.8x speedup for autoregressive decoding.
"""

import logging
from typing import Callable, Dict, Optional

import torch

logger = logging.getLogger(__name__)


class CUDAGraphRunner:
    """Captures and replays CUDA graphs for decoding.

    Captures the forward pass for a single decode step (fixed batch size
    and sequence length) and replays it efficiently.

    Args:
        model: The model to capture
        batch_size: Fixed batch size for graph
        max_seq_len: Maximum sequence length
        device: CUDA device
    """

    def __init__(
        self,
        model: torch.nn.Module,
        batch_size: int = 1,
        max_seq_len: int = 4096,
        device: torch.device = torch.device("cuda"),
    ):
        self.model = model
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.device = device

        # Graph storage: {(batch_size, seq_len): graph}
        self._graphs: Dict[tuple, torch.cuda.CUDAGraph] = {}
        self._static_inputs: Dict[tuple, torch.Tensor] = {}
        self._static_outputs: Dict[tuple, torch.Tensor] = {}

        logger.info(
            f"CUDAGraphRunner initialized (batch_size={batch_size}, "
            f"max_seq_len={max_seq_len})"
        )

    def capture(
        self,
        batch_size: int,
        seq_len: int,
        forward_fn: Callable[[torch.Tensor], torch.Tensor],
    ):
        """Capture a CUDA graph for the given shape.

        Args:
            batch_size: Batch size for this graph
            seq_len: Sequence length for this graph
            forward_fn: Function that takes input_ids and returns logits
        """
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, skipping graph capture")
            return

        shape_key = (batch_size, seq_len)

        if shape_key in self._graphs:
            logger.debug(f"Graph already captured for shape {shape_key}")
            return

        logger.info(f"Capturing CUDA graph for shape {shape_key}...")

        # Create static input/output tensors
        static_input = torch.zeros(
            (batch_size, seq_len),
            dtype=torch.long,
            device=self.device,
        )

        # Warmup: run forward pass a few times
        with torch.cuda.stream(torch.cuda.Stream()):
            for _ in range(3):
                _ = forward_fn(static_input)

        # Capture graph
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_output = forward_fn(static_input)

        self._graphs[shape_key] = graph
        self._static_inputs[shape_key] = static_input
        self._static_outputs[shape_key] = static_output

        logger.info(f"CUDA graph captured for shape {shape_key}")

    def replay(
        self,
        input_ids: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Replay captured graph with new input.

        Args:
            input_ids: Input tensor [batch_size, seq_len]

        Returns:
            Output tensor if graph exists, None otherwise
        """
        if not torch.cuda.is_available():
            return None

        batch_size, seq_len = input_ids.shape
        shape_key = (batch_size, seq_len)

        if shape_key not in self._graphs:
            return None

        # Copy input to static buffer
        self._static_inputs[shape_key].copy_(input_ids)

        # Replay graph
        self._graphs[shape_key].replay()

        # Return output (clone to avoid graph memory issues)
        return self._static_outputs[shape_key].clone()

    def can_replay(self, batch_size: int, seq_len: int) -> bool:
        """Check if graph exists for given shape.

        Args:
            batch_size: Batch size
            seq_len: Sequence length

        Returns:
            True if graph captured for this shape
        """
        return (batch_size, seq_len) in self._graphs

    def clear(self):
        """Clear all captured graphs."""
        self._graphs.clear()
        self._static_inputs.clear()
        self._static_outputs.clear()
        logger.debug("CUDA graphs cleared")

    def get_stats(self) -> Dict[str, int]:
        """Get graph statistics.

        Returns:
            Dict with num_graphs and captured_shapes
        """
        return {
            "num_graphs": len(self._graphs),
            "captured_shapes": list(self._graphs.keys()),
        }


class CUDAGraphDecoder:
    """Helper for graph-accelerated autoregressive decoding.

    Wraps the model's decode step with CUDA graph support.

    Args:
        model: The model to wrap
        enable_graphs: Whether to enable CUDA graphs
        device: CUDA device
    """

    def __init__(
        self,
        model: torch.nn.Module,
        enable_graphs: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        self.model = model
        self.enable_graphs = enable_graphs and torch.cuda.is_available()
        self.device = device

        self.graph_runner = (
            CUDAGraphRunner(model, device=device)
            if self.enable_graphs
            else None
        )

        self._capture_shapes = [
            (1, 1),  # Single token decode
            (1, 2),
            (1, 4),
            (1, 8),
        ]

        logger.info(
            f"CUDAGraphDecoder initialized (enable_graphs={self.enable_graphs})"
        )

    def warmup(self):
        """Warmup: capture common decode shapes."""
        if not self.enable_graphs or self.graph_runner is None:
            return

        logger.info("Warming up CUDA graphs...")

        def forward_fn(input_ids: torch.Tensor) -> torch.Tensor:
            with torch.inference_mode():
                return self.model(input_ids)

        for shape in self._capture_shapes:
            try:
                self.graph_runner.capture(
                    batch_size=shape[0],
                    seq_len=shape[1],
                    forward_fn=forward_fn,
                )
            except Exception as e:
                logger.warning(f"Failed to capture graph for {shape}: {e}")

        logger.info("CUDA graph warmup complete")

    def decode_step(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Run a single decode step (with graph if available).

        Args:
            input_ids: Input tensor [batch_size, seq_len]

        Returns:
            Logits tensor [batch_size, seq_len, vocab_size]
        """
        # Try graph replay first
        if self.enable_graphs and self.graph_runner is not None:
            output = self.graph_runner.replay(input_ids)
            if output is not None:
                return output

        # Fallback to regular forward pass
        with torch.inference_mode():
            return self.model(input_ids)

    def clear(self):
        """Clear cached graphs."""
        if self.graph_runner is not None:
            self.graph_runner.clear()
