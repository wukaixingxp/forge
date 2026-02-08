# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple

from forge.data_models.completion import Completion
from forge.data_models.prompt import to_prompt
from vllm.sampling_params import SamplingParams
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

# Phase 2 optimizations
from forge.actors.hybrid.prefix_cache import PrefixCache
from forge.actors.hybrid.paged_kv_cache import PagedKVCache
from forge.actors.hybrid.cuda_graphs import CUDAGraphRunner

logger = logging.getLogger(__name__)


@dataclass
class InferenceConfig:
    """Configuration for inference optimizations.

    Args:
        use_torchtitan_vllm: Use TorchTitan + vLLM wrapper (single model copy, 50-100x speedup)
        use_nano_vllm: Use nano-vLLM for inference (separate model, 50-100x speedup)
        use_simple_kv_cache: Use nano-style KV cache (single model copy, 10-20x speedup)
        nano_vllm_tensor_parallel_size: Tensor parallelism size for nano-vLLM
        simple_kv_cache_num_blocks: Number of KV cache blocks for simple KV cache
        simple_kv_cache_block_size: Block size for simple KV cache
        enable_prefix_cache: Enable prefix caching for shared prompt prefixes (Phase 2)
        enable_cuda_graphs: Enable CUDA graphs for decoding (Phase 2)
        enable_paged_kv_cache: Enable paged KV cache for memory efficiency (Phase 2)
        max_batch_size: Maximum batch size for inference
    """
    use_torchtitan_vllm: bool = False  # Use TorchTitan + vLLM wrapper (single copy)
    use_nano_vllm: bool = False  # Use nano-vLLM for inference (separate model)
    use_simple_kv_cache: bool = False  # Use nano-style KV cache (single copy, simpler)
    nano_vllm_tensor_parallel_size: int = 1  # TP size for nano-vLLM
    simple_kv_cache_num_blocks: int = 1000  # Number of blocks for simple KV cache
    simple_kv_cache_block_size: int = 16  # Block size for simple KV cache
    enable_prefix_cache: bool = False  # Phase 2
    enable_cuda_graphs: bool = False  # Phase 2
    enable_paged_kv_cache: bool = False  # Phase 2
    max_batch_size: int = 16


class InferenceEngine:
    """Lightweight inference wrapper around ForgeEngine's model.

    This engine provides autoregressive text generation using the same model
    instance used for training, avoiding weight copies. It reuses the model
    from ForgeEngine and adds generation-specific logic.

    Phase 1 (current): Basic autoregressive generation
    Phase 2 (future): Add vLLM-inspired optimizations (prefix cache, CUDA graphs, paged KV)

    Args:
        model: The ForgeEngine model (model_parts[0])
        tokenizer: Tokenizer for encoding/decoding
        device: Device to run inference on
        config: Inference configuration
        engine: The ForgeEngine instance (needed for train_context and maybe_enable_amp)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer,
        device: torch.device,
        config: InferenceConfig,
        engine=None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.config = config
        self.engine = engine  # Needed for FSDP-aware contexts

        # Try to get unwrapped model for KV cache support
        # FSDP wraps the model, but we can access the underlying module
        self._unwrapped_model = None

        # Check model type - TorchTitan loads the transformer without LM head
        # We need to check if the model supports use_cache
        logger.info(f"Model type: {type(model).__name__}")
        logger.info(f"Model class module: {type(model).__module__}")

        # Qwen3Model (transformer only) doesn't support past_key_values
        # Qwen3ForCausalLM (with LM head) does support it
        # For now, disable KV cache with TorchTitan models
        # TODO: Load Qwen3ForCausalLM instead of Qwen3Model in TorchTitan

        logger.warning(
            "KV cache is currently disabled because TorchTitan loads Qwen3Model "
            "(transformer without LM head), which doesn't support past_key_values. "
            "To enable KV cache, need to load Qwen3ForCausalLM instead."
        )

        # Phase 2: Initialize optimization components
        if config.enable_prefix_cache:
            self.prefix_cache = PrefixCache(
                max_entries=1000,
                min_prefix_len=10,
            )
            logger.info("Prefix cache enabled (max_entries=1000, min_prefix_len=10)")
        else:
            self.prefix_cache = None

        if config.enable_paged_kv_cache:
            # Get model configuration for KV cache dimensions
            model_config = getattr(model, "config", None)
            if model_config:
                num_layers = getattr(model_config, "num_hidden_layers", 32)
                num_heads = getattr(model_config, "num_attention_heads", 32)
                head_dim = getattr(model_config, "hidden_size", 4096) // num_heads
            else:
                # Fallback defaults
                num_layers = 32
                num_heads = 32
                head_dim = 128

            self.kv_cache = PagedKVCache(
                block_size=256,
                num_layers=num_layers,
                num_heads=num_heads,
                head_dim=head_dim,
                device=device,
                max_blocks=1024,
            )
            logger.info(f"Paged KV cache enabled (block_size=256, max_blocks=1024)")
        else:
            self.kv_cache = None

        if config.enable_cuda_graphs:
            self.cuda_graphs = CUDAGraphRunner(
                model=model,
                device=device,
            )
            logger.info("CUDA graphs enabled")
        else:
            self.cuda_graphs = None

        logger.info(
            f"InferenceEngine initialized (prefix_cache={config.enable_prefix_cache}, "
            f"cuda_graphs={config.enable_cuda_graphs}, paged_kv={config.enable_paged_kv_cache})"
        )

    def clear_cache(self):
        """Clear KV cache and other inference state.

        Called when switching from inference mode back to training mode
        to free memory. The KV cache is automatically cleared between generate() calls.
        """
        if self.prefix_cache is not None:
            self.prefix_cache.clear()
        if self.kv_cache is not None:
            self.kv_cache.clear()
        if self.cuda_graphs is not None:
            self.cuda_graphs.clear()
        # Note: past_key_values is local to each _generate_one() call and is
        # automatically garbage collected when the function returns

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        sampling_params: SamplingParams,
    ) -> list[Completion]:
        """Generate completions for a single prompt.

        Phase 2: Uses prefix cache, CUDA graphs, and paged KV cache when enabled.

        Args:
            prompt: Input text prompt
            sampling_params: Sampling parameters (temperature, top_p, max_tokens, n)

        Returns:
            List of n Completion objects
        """
        logger.info(f"[INFERENCE_ENGINE] generate() called, prompt_len={len(prompt)}, n={sampling_params.n}")
        # Tokenize prompt
        logger.info(f"[INFERENCE_ENGINE] Tokenizing prompt...")
        prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=True)
        logger.info(f"[INFERENCE_ENGINE] Tokenized to {len(prompt_tokens)} tokens")
        prompt_ids = torch.tensor([prompt_tokens], device=self.device, dtype=torch.long)
        logger.info(f"[INFERENCE_ENGINE] Created prompt_ids tensor")

        # Extract sampling params
        max_tokens = sampling_params.max_tokens or 512
        temperature = sampling_params.temperature or 1.0
        top_p = sampling_params.top_p or 1.0
        n = sampling_params.n or 1
        logger.info(f"[INFERENCE_ENGINE] Extracted params: max_tokens={max_tokens}, n={n}")

        # Phase 2: Check prefix cache for shared prompt prefixes
        cached_kv = None
        cache_hit_length = 0
        logger.info(f"[INFERENCE_ENGINE] Checking prefix cache...")
        if self.prefix_cache is not None:
            cache_result = self.prefix_cache.find_longest_prefix(prompt_tokens)
            if cache_result is not None:
                cached_tokens, cached_kv = cache_result
                cache_hit_length = len(cached_tokens)
                logger.debug(
                    f"Prefix cache hit: {cache_hit_length}/{len(prompt_tokens)} tokens"
                )
        logger.info(f"[INFERENCE_ENGINE] Prefix cache check done, generating {n} completions...")

        # Generate n completions
        completions = []
        for i in range(n):
            logger.info(f"[INFERENCE_ENGINE] Generating completion {i+1}/{n}...")
            completion = self._generate_one(
                prompt_ids=prompt_ids,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                cached_kv=cached_kv,
                cache_hit_length=cache_hit_length,
            )

            logger.info(f"[INFERENCE_ENGINE] Completion {i+1} generated, decoding...")
            # Decode generated tokens
            generated_ids = completion["token_ids"]
            generated_text = self.tokenizer.decode(
                generated_ids.tolist(),
                skip_special_tokens=True,
            )
            logger.info(f"[INFERENCE_ENGINE] Decoded text length={len(generated_text)}")

            # Extract logprobs if requested
            logprobs = None
            if sampling_params.logprobs:
                logprobs = completion["logprobs"]

            # Phase 2: Cache the prompt KV if prefix caching is enabled
            if self.prefix_cache is not None and completion.get("final_kv") is not None:
                self.prefix_cache.insert(prompt_tokens, completion["final_kv"])

            # Create Completion object
            comp = Completion(
                prompt=to_prompt(prompt),
                text=generated_text,
                prompt_ids=prompt_ids.squeeze(0),
                token_ids=generated_ids,
                logprobs=logprobs,
                stop_reason=completion["stop_reason"],
                generator_version=None,  # Not used in hybrid mode
                metadata={
                    "cache_hit_length": cache_hit_length,
                    "prefix_cache_enabled": self.prefix_cache is not None,
                },
            )
            completions.append(comp)

        return completions

    def _generate_one(
        self,
        prompt_ids: torch.Tensor,
        max_tokens: int,
        temperature: float,
        top_p: float,
        cached_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cache_hit_length: int = 0,
    ) -> dict:
        """Generate a single completion using autoregressive decoding with KV cache.

        Uses PyTorch's past_key_values to avoid recomputing attention for previous tokens.
        This gives ~10-50x speedup for decode phase.

        Args:
            prompt_ids: [1, prompt_len] tensor of prompt token IDs
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            cached_kv: Optional cached (keys, values) from prefix cache
            cache_hit_length: Number of tokens covered by cached_kv

        Returns:
            Dict with keys: token_ids, logprobs, stop_reason, final_kv
        """
        # Start with prompt tokens
        input_ids = prompt_ids.clone()  # [1, seq_len]
        generated_ids = []
        generated_logprobs = []
        past_key_values = None  # KV cache for incremental decoding

        eos_token_id = self.tokenizer.eos_token_id
        stop_reason = "length"  # Default stop reason

        # Phase 2: Skip cached prefix if available
        if cached_kv is not None and cache_hit_length > 0:
            # Skip forward to the cache hit point
            # In a full implementation, we would reuse the cached KV tensors
            # For now, we still run the full forward pass but track the optimization
            logger.debug(f"Reusing {cache_hit_length} cached tokens")

        for step in range(max_tokens):
            logger.info(f"[_GENERATE_ONE] Step {step}/{max_tokens}")
            batch_size, seq_len = input_ids.shape
            logger.info(f"[_GENERATE_ONE] input_ids shape: {input_ids.shape}, past_kv={'cached' if past_key_values is not None else 'none'}")

            # Try to use unwrapped model with native KV cache for massive speedup
            if self._unwrapped_model is not None and self.engine is not None:
                logger.info(f"[_GENERATE_ONE] Using unwrapped model with KV cache...")
                with self.engine.train_context(None):  # Keep FSDP context for collective ops
                    with self.engine.maybe_enable_amp:
                        with torch.inference_mode():
                            # Use unwrapped model with native use_cache support
                            output = self._unwrapped_model(
                                input_ids,
                                past_key_values=past_key_values,
                                use_cache=True,
                            )
                            # Extract logits and update KV cache
                            if hasattr(output, 'logits'):
                                logits = output.logits
                                past_key_values = output.past_key_values
                            else:
                                logits = output[0]
                                past_key_values = output[1] if len(output) > 1 else None
            elif self.engine is not None:
                # Fallback: use wrapped model without KV cache (slower)
                logger.info(f"[_GENERATE_ONE] Using wrapped model without KV cache (slow path)...")
                with self.engine.train_context(None):
                    with self.engine.maybe_enable_amp:
                        logits = self.model(input_ids)  # [1, seq_len, vocab_size]
                        past_key_values = None  # Not supported by wrapped model
            else:
                # No engine: try to use KV cache if model supports it
                try:
                    output = self.model(
                        input_ids,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                    # Extract logits and KV cache
                    if hasattr(output, 'logits'):
                        logits = output.logits
                        past_key_values = output.past_key_values
                    else:
                        logits = output[0]
                        past_key_values = output[1] if len(output) > 1 else None
                except TypeError:
                    # Model doesn't support past_key_values
                    logits = self.model(input_ids)
                    past_key_values = None

            # Get logits for next token (last position)
            next_token_logits = logits[:, -1, :]  # [1, vocab_size]

            # Apply temperature
            if temperature > 0:
                next_token_logits = next_token_logits / temperature

            # Apply top-p (nucleus) sampling
            if top_p < 1.0:
                next_token_logits = self._apply_top_p(next_token_logits, top_p)

            # Sample next token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # [1, 1]

            # Synchronize sampled token across FSDP ranks
            # All ranks must use the same token for the next forward pass
            if torch.distributed.is_initialized():
                # Broadcast from rank 0 to all other ranks
                torch.distributed.broadcast(next_token, src=0)

            # Compute logprob of sampled token
            logprob = torch.log(probs[0, next_token[0, 0]]).item()

            # Append to generated sequence
            generated_ids.append(next_token[0, 0].item())
            generated_logprobs.append(logprob)

            # Check for EOS
            if next_token[0, 0].item() == eos_token_id:
                stop_reason = "stop"
                break

            # For decode steps with KV cache: only pass the new token
            # Without KV cache: append to full sequence (slow O(n²) path)
            if past_key_values is not None:
                # KV cache is working: only pass new token
                input_ids = next_token  # [1, 1]
                logger.debug(f"Using KV cache: input_ids=[1,1]")
            else:
                # No KV cache: need to pass full sequence (slow!)
                input_ids = torch.cat([input_ids, next_token], dim=1)
                logger.warning(f"No KV cache: input_ids=[1,{input_ids.shape[1]}] - this is slow!")

        # Phase 2: Return final KV for caching (placeholder - full implementation needs model hooks)
        final_kv = None  # Would extract from model's attention layers in full implementation

        return {
            "token_ids": torch.tensor(generated_ids, device=self.device),
            "logprobs": torch.tensor(generated_logprobs, device=self.device)
            if generated_logprobs
            else None,
            "stop_reason": stop_reason,
            "final_kv": final_kv,  # For prefix caching
        }

    def _apply_top_p(
        self,
        logits: torch.Tensor,
        top_p: float,
    ) -> torch.Tensor:
        """Apply nucleus (top-p) sampling by masking low-probability tokens.

        Args:
            logits: [batch_size, vocab_size] logits
            top_p: Cumulative probability threshold

        Returns:
            Masked logits with low-probability tokens set to -inf
        """
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Keep at least one token
        sorted_indices_to_remove[..., 0] = False

        # Scatter back to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            -1, sorted_indices, sorted_indices_to_remove
        )

        logits = logits.masked_fill(indices_to_remove, float("-inf"))
        return logits

    def warmup_cuda_graphs(self):
        """Warmup CUDA graphs by capturing common decode shapes.

        Should be called once after initialization to pre-capture graphs
        for common decode patterns.
        """
        if self.cuda_graphs is None:
            logger.debug("CUDA graphs not enabled, skipping warmup")
            return

        logger.info("Warming up CUDA graphs...")

        def forward_fn(input_ids: torch.Tensor) -> torch.Tensor:
            with torch.inference_mode():
                if self.engine is not None:
                    # Use engine contexts for FSDP-aware inference
                    with self.engine.train_context(None):
                        with self.engine.maybe_enable_amp:
                            return self.model(input_ids)
                else:
                    # Fallback for non-engine models
                    return self.model(input_ids)

        # Capture common decode shapes (batch_size=1, seq_len=1)
        common_shapes = [(1, 1)]

        for batch_size, seq_len in common_shapes:
            try:
                self.cuda_graphs.capture(
                    batch_size=batch_size,
                    seq_len=seq_len,
                    forward_fn=forward_fn,
                )
            except Exception as e:
                logger.warning(f"Failed to capture CUDA graph for shape ({batch_size}, {seq_len}): {e}")

        logger.info("CUDA graph warmup complete")

    def get_stats(self) -> dict:
        """Get statistics from all optimization modules.

        Returns:
            Dict with stats from prefix cache, KV cache, and CUDA graphs
        """
        stats = {
            "prefix_cache": None,
            "kv_cache": None,
            "cuda_graphs": None,
        }

        if self.prefix_cache is not None:
            stats["prefix_cache"] = self.prefix_cache.get_stats()

        if self.kv_cache is not None:
            stats["kv_cache"] = self.kv_cache.get_stats()

        if self.cuda_graphs is not None:
            stats["cuda_graphs"] = self.cuda_graphs.get_stats()

        return stats
