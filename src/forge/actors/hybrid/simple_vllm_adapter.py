# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Simplified vLLM adapter for TorchTitan models.

This provides a lightweight alternative to TorchTitan's experimental vLLM wrapper
when the experimental code is not available or has version compatibility issues.

Key difference from experimental wrapper:
- Does NOT use TorchTitan model directly for inference
- Uses separate vLLM model instance with HF checkpoint
- Still provides paged KV cache and 50-100x speedup
- Maintains compatibility with any vLLM-supported model
"""

import logging
import torch
from dataclasses import dataclass
from typing import Optional

from forge.data_models.completion import Completion
from forge.data_models.prompt import to_prompt
from vllm.sampling_params import SamplingParams

logger = logging.getLogger(__name__)


@dataclass
class SimpleVLLMConfig:
    """Configuration for simple vLLM integration.

    Args:
        model_name: HuggingFace model name (e.g., "Qwen/Qwen3-1.7B")
        tensor_parallel_size: Number of GPUs for tensor parallelism
        enable_cuda_graphs: Enable CUDA graphs for decoding
        max_num_seqs: Maximum number of sequences to process in parallel
        gpu_memory_utilization: Fraction of GPU memory to use for KV cache
    """
    model_name: str
    tensor_parallel_size: int = 1
    enable_cuda_graphs: bool = True
    max_num_seqs: int = 16
    gpu_memory_utilization: float = 0.9


class SimpleVLLMEngine:
    """Simplified vLLM engine using standard HF model loading.

    This is a fallback when TorchTitan's experimental vLLM wrapper is not available.
    It provides the same KV cache acceleration benefits but loads a separate
    vLLM model instance rather than wrapping the TorchTitan training model.

    Note: This uses more GPU memory than TorchTitan's integrated approach because
    it maintains two model copies (training model + inference model).

    Args:
        config: SimpleVLLMConfig with model name and inference settings
    """

    def __init__(self, config: SimpleVLLMConfig):
        self.config = config

        logger.info(
            f"Initializing simple vLLM engine with {config.model_name}, "
            f"TP={config.tensor_parallel_size}, "
            f"CUDA_graphs={config.enable_cuda_graphs}, "
            f"max_num_seqs={config.max_num_seqs}"
        )

        # Import vLLM
        try:
            from vllm import LLM
        except ImportError as e:
            raise ImportError(
                "vLLM not installed. Install with: pip install vllm"
            ) from e

        # Create standard vLLM LLM with HF model
        try:
            self.llm = LLM(
                model=config.model_name,
                tensor_parallel_size=config.tensor_parallel_size,
                enforce_eager=not config.enable_cuda_graphs,
                max_num_seqs=config.max_num_seqs,
                gpu_memory_utilization=config.gpu_memory_utilization,
                trust_remote_code=True,
            )
            logger.info(f"vLLM initialized successfully with {config.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize vLLM: {e}")
            raise

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        sampling_params: SamplingParams,
    ) -> list[Completion]:
        """Generate completions using vLLM.

        Args:
            prompt: Input text prompt
            sampling_params: Sampling parameters (temperature, top_p, max_tokens, n)

        Returns:
            List of n Completion objects
        """
        logger.info(
            f"[SIMPLE_VLLM] generate() called, n={sampling_params.n}, "
            f"max_tokens={sampling_params.max_tokens}"
        )

        # Generate using vLLM (includes paged KV cache automatically)
        outputs = self.llm.generate([prompt], sampling_params, use_tqdm=False)

        logger.info(f"[SIMPLE_VLLM] Generated {len(outputs[0].outputs)} completions")

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
                    generator_version="simple-vllm",
                    metadata={
                        "simple_vllm": True,
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
            "engine": "simple-vllm",
            "model_name": self.config.model_name,
            "tensor_parallel_size": self.config.tensor_parallel_size,
            "cuda_graphs_enabled": self.config.enable_cuda_graphs,
            "max_num_seqs": self.config.max_num_seqs,
            "paged_attention": True,
        }
