# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""vLLM engine using TorchTitan's model wrapper.

This module provides integration between TorchTitan models and vLLM inference engine.
Uses TorchTitan's experimental TorchTitanVLLMModelWrapper which:
- Wraps TorchTitan models for vLLM inference
- Replaces attention with vLLM's paged attention (KV cache)
- Maintains single model copy (no weight duplication)
"""

import logging
import sys
import os
import torch
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

from forge.data_models.completion import Completion
from forge.data_models.prompt import to_prompt
from vllm.sampling_params import SamplingParams

logger = logging.getLogger(__name__)

# Add torchtitan source directory to path if experimental code not available
def _ensure_torchtitan_experimental_available():
    """Ensure TorchTitan experimental code is accessible."""
    try:
        import torchtitan.experiments.rl.unified
        return  # Already available
    except ImportError:
        pass

    # Try to find torchtitan source directory
    possible_paths = [
        Path(__file__).parent.parent.parent.parent.parent.parent / "torchtitan",  # ../torchtitan relative to forge
        Path.home() / "work" / "kaiwu" / "torchtitan",
        Path("/home/dev/work/kaiwu/torchtitan"),
    ]

    for torchtitan_path in possible_paths:
        if torchtitan_path.exists():
            torchtitan_str = str(torchtitan_path)
            if torchtitan_str not in sys.path:
                logger.info(f"Adding TorchTitan source to path: {torchtitan_str}")
                sys.path.insert(0, torchtitan_str)
            try:
                import torchtitan.experiments.rl.unified
                logger.info("Successfully loaded TorchTitan experimental code")
                return
            except ImportError as e:
                logger.warning(f"Found torchtitan at {torchtitan_path} but failed to import: {e}")
                continue

    raise ImportError(
        "TorchTitan experimental code not found. "
        "Please ensure ../torchtitan source directory exists with experimental RL code."
    )


@dataclass
class TorchTitanVLLMConfig:
    """Configuration for TorchTitan + vLLM integration.

    Args:
        tensor_parallel_size: Number of GPUs for tensor parallelism
        enable_cuda_graphs: Enable CUDA graphs for decoding
        max_num_seqs: Maximum number of sequences to process in parallel
        gpu_memory_utilization: Fraction of GPU memory to use for KV cache
    """
    tensor_parallel_size: int = 1
    enable_cuda_graphs: bool = True
    max_num_seqs: int = 16
    gpu_memory_utilization: float = 0.9


class TorchTitanVLLMEngine:
    """vLLM engine using TorchTitan's model wrapper.

    This engine provides high-performance inference by combining:
    - TorchTitan's model (single copy, used for training too)
    - vLLM's paged attention (KV cache, 50-100x speedup)
    - vLLM's CUDA graphs and continuous batching

    The key advantage is maintaining a single model copy while getting
    vLLM's inference optimizations.

    Args:
        model_name: Base model name (e.g., "Qwen3", will use Qwen3TorchTitanForCausalLM)
        config: TorchTitanVLLMConfig with inference settings
    """

    def __init__(self, model_name: str, config: TorchTitanVLLMConfig):
        self.config = config
        self.model_name = model_name

        # Ensure TorchTitan experimental code is accessible
        _ensure_torchtitan_experimental_available()

        try:
            # Import TorchTitan's vLLM wrapper (auto-registers models)
            from torchtitan.experiments.rl.unified import TorchTitanVLLMModelWrapper
            logger.info(f"Successfully imported TorchTitan vLLM wrapper")
        except ImportError as e:
            raise ImportError(
                "TorchTitan's vLLM wrapper not found. "
                "Make sure torchtitan is installed with experimental RL support."
            ) from e

        # Import vLLM after TorchTitan to ensure models are registered
        try:
            from vllm import LLM
        except ImportError as e:
            raise ImportError(
                "vLLM not installed. Install with: pip install vllm"
            ) from e

        # Construct vLLM model name (TorchTitan auto-registers these)
        vllm_model_name = f"{model_name}TorchTitanForCausalLM"
        logger.info(
            f"Initializing vLLM with TorchTitan model: {vllm_model_name}, "
            f"TP={config.tensor_parallel_size}, "
            f"CUDA_graphs={config.enable_cuda_graphs}, "
            f"max_num_seqs={config.max_num_seqs}"
        )

        # Create vLLM LLM with TorchTitan model
        try:
            self.llm = LLM(
                model=vllm_model_name,
                tensor_parallel_size=config.tensor_parallel_size,
                enforce_eager=not config.enable_cuda_graphs,
                max_num_seqs=config.max_num_seqs,
                gpu_memory_utilization=config.gpu_memory_utilization,
                trust_remote_code=True,
            )
            logger.info("vLLM initialized successfully with TorchTitan model")
        except Exception as e:
            logger.error(f"Failed to initialize vLLM: {e}")
            raise

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        sampling_params: SamplingParams,
    ) -> list[Completion]:
        """Generate completions using vLLM with TorchTitan model.

        Args:
            prompt: Input text prompt
            sampling_params: Sampling parameters (temperature, top_p, max_tokens, n)

        Returns:
            List of n Completion objects
        """
        logger.info(
            f"[TORCHTITAN_VLLM] generate() called, n={sampling_params.n}, "
            f"max_tokens={sampling_params.max_tokens}"
        )

        # Generate using vLLM (includes paged KV cache automatically)
        outputs = self.llm.generate([prompt], sampling_params, use_tqdm=False)

        logger.info(f"[TORCHTITAN_VLLM] Generated {len(outputs)} output sequences")

        # Convert vLLM outputs to Completion format
        completions = []
        for output in outputs:
            for completion_output in output.outputs:
                # Extract logprobs if requested
                logprobs = None
                if sampling_params.logprobs and completion_output.logprobs:
                    # Convert vLLM logprobs format to tensor
                    logprob_values = [
                        list(token_logprobs.values())[0].logprob
                        for token_logprobs in completion_output.logprobs
                    ]
                    logprobs = torch.tensor(logprob_values, device="cpu")

                completion = Completion(
                    prompt=to_prompt(prompt),
                    text=completion_output.text,
                    prompt_ids=torch.tensor(output.prompt_token_ids, device="cpu"),
                    token_ids=torch.tensor(completion_output.token_ids, device="cpu"),
                    logprobs=logprobs,
                    stop_reason=completion_output.finish_reason,
                    generator_version="torchtitan-vllm",
                    metadata={
                        "torchtitan_vllm": True,
                        "paged_attention": True,
                        "tensor_parallel_size": self.config.tensor_parallel_size,
                    },
                )
                completions.append(completion)

        return completions

    def clear_cache(self):
        """Clear KV cache.

        vLLM manages cache automatically, so this is a no-op.
        """
        # vLLM handles cache lifecycle automatically
        pass

    def get_stats(self) -> dict:
        """Get statistics from vLLM engine.

        Returns:
            Dict with engine statistics
        """
        return {
            "engine": "torchtitan-vllm",
            "model_name": self.model_name,
            "tensor_parallel_size": self.config.tensor_parallel_size,
            "cuda_graphs_enabled": self.config.enable_cuda_graphs,
            "max_num_seqs": self.config.max_num_seqs,
            "paged_attention": True,
        }
