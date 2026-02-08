# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import torch
from dataclasses import dataclass
from typing import Optional

from forge.data_models.completion import Completion
from forge.data_models.prompt import to_prompt
from vllm.sampling_params import SamplingParams

logger = logging.getLogger(__name__)


@dataclass
class NanoVLLMConfig:
    """Configuration for nano-vLLM inference engine.

    Args:
        tensor_parallel_size: Number of GPUs for tensor parallelism
        enable_cuda_graphs: Enable CUDA graphs for decoding
        max_num_seqs: Maximum number of sequences to process in parallel
        block_size: KV cache block size (default: 16)
        max_model_len: Maximum model length (default: None, auto-detected)
        gpu_memory_utilization: Fraction of GPU memory to use (default: 0.9)
    """
    tensor_parallel_size: int = 1
    enable_cuda_graphs: bool = True
    max_num_seqs: int = 16
    block_size: int = 16
    max_model_len: Optional[int] = None
    gpu_memory_utilization: float = 0.9


class NanoVLLMEngine:
    """Wrapper for nano-vLLM with async-compatible interface.

    This engine provides high-performance inference using nano-vLLM's:
    - Paged KV cache (10-50x speedup over naive generation)
    - Continuous batching (2-5x throughput improvement)
    - CUDA graphs (1.5-2x speedup for decode)
    - Flash attention (memory-efficient attention)

    The engine loads a separate model instance for inference, keeping it
    separate from the training model. This allows using tensor parallelism
    for inference while training uses FSDP.

    Args:
        model_path: HuggingFace model path
        config: NanoVLLMConfig with inference settings
    """

    def __init__(self, model_path: str, config: NanoVLLMConfig):
        try:
            from nanovllm import LLM, SamplingParams as NanoSamplingParams
        except ImportError as e:
            raise ImportError(
                "nano-vLLM not installed. Install with: "
                "pip install git+https://github.com/GeeeekExplorer/nano-vllm.git"
            ) from e

        self.config = config
        self.NanoSamplingParams = NanoSamplingParams

        logger.info(
            f"Initializing nano-vLLM with model={model_path}, "
            f"TP={config.tensor_parallel_size}, "
            f"CUDA_graphs={config.enable_cuda_graphs}, "
            f"max_num_seqs={config.max_num_seqs}"
        )

        # Initialize nano-vLLM LLM
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=config.tensor_parallel_size,
            enforce_eager=not config.enable_cuda_graphs,
            max_num_seqs=config.max_num_seqs,
            block_size=config.block_size,
            max_model_len=config.max_model_len,
            gpu_memory_utilization=config.gpu_memory_utilization,
            trust_remote_code=True,  # Required for Qwen3
        )

        logger.info("nano-vLLM initialized successfully")

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        sampling_params: SamplingParams,
    ) -> list[Completion]:
        """Generate completions for a single prompt.

        Args:
            prompt: Input text prompt
            sampling_params: Sampling parameters (temperature, top_p, max_tokens, n)

        Returns:
            List of n Completion objects
        """
        logger.info(
            f"[NANO_VLLM] generate() called, n={sampling_params.n}, "
            f"max_tokens={sampling_params.max_tokens}"
        )

        # Convert vLLM SamplingParams to nano-vLLM format
        nano_params = self.NanoSamplingParams(
            n=sampling_params.n or 1,
            max_tokens=sampling_params.max_tokens or 512,
            temperature=sampling_params.temperature or 1.0,
            top_p=sampling_params.top_p or 1.0,
            top_k=sampling_params.top_k or -1,
            logprobs=sampling_params.logprobs,
            stop=sampling_params.stop,
            stop_token_ids=sampling_params.stop_token_ids,
        )

        # Generate (synchronous in nano-vLLM)
        outputs = self.llm.generate([prompt], nano_params, use_tqdm=False)

        logger.info(f"[NANO_VLLM] Generated {len(outputs)} completions")

        # Convert nano-vLLM outputs to Completion format
        completions = []
        for output in outputs:
            # Extract data from nano-vLLM output format
            text = output.outputs[0].text
            token_ids = output.outputs[0].token_ids
            logprobs_data = output.outputs[0].logprobs if sampling_params.logprobs else None

            # Convert logprobs format if present
            logprobs = None
            if logprobs_data is not None:
                # nano-vLLM returns list of dicts: [{token_id: Logprob}, ...]
                # We extract the logprob values for the sampled tokens
                logprobs = torch.tensor(
                    [list(lp.values())[0].logprob for lp in logprobs_data],
                    device="cpu",
                )

            completion = Completion(
                prompt=to_prompt(prompt),
                text=text,
                prompt_ids=torch.tensor(output.prompt_token_ids, device="cpu"),
                token_ids=torch.tensor(token_ids, device="cpu"),
                logprobs=logprobs,
                stop_reason=output.outputs[0].finish_reason,
                generator_version="nano-vllm",
                metadata={
                    "nano_vllm": True,
                    "tensor_parallel_size": self.config.tensor_parallel_size,
                    "cuda_graphs_enabled": self.config.enable_cuda_graphs,
                },
            )
            completions.append(completion)

        return completions

    def clear_cache(self):
        """Clear KV cache.

        nano-vLLM manages its own cache automatically, so this is a no-op.
        Called when switching from inference mode back to training mode.
        """
        # nano-vLLM's block manager handles cache lifecycle automatically
        pass

    def get_stats(self) -> dict:
        """Get statistics from nano-vLLM engine.

        Returns:
            Dict with cache statistics and performance metrics
        """
        # nano-vLLM doesn't expose detailed stats yet
        # Could be extended in the future
        return {
            "engine": "nano-vllm",
            "tensor_parallel_size": self.config.tensor_parallel_size,
            "cuda_graphs_enabled": self.config.enable_cuda_graphs,
            "max_num_seqs": self.config.max_num_seqs,
        }
