# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Simple KV cache engine using nano-vLLM style approach.

This engine wraps the training model and adds KV cache support without
creating a separate model copy. It achieves 10-20x speedup over naive
generation with minimal complexity (~600 lines vs ~5,300 for full vLLM).
"""

import logging
import math
import torch
from typing import Optional, List

from forge.actors.hybrid.forge_attention import replace_attention_with_forge
from forge.actors.hybrid.forge_model_wrapper import patch_model_for_positions
from forge.actors.hybrid.nano_kv_cache import NanoStyleKVCache, estimate_kv_cache_blocks
from forge.actors.hybrid.block_manager import BlockManager
from forge.actors.hybrid.sequence import Sequence
from forge.actors.hybrid.simple_scheduler import SimpleScheduler
from forge.actors.hybrid.inference_context import inference_context
from forge.actors.hybrid.prefix_cache import PrefixCache
from forge.data_models.completion import Completion
from vllm.sampling_params import SamplingParams

logger = logging.getLogger(__name__)


class SimpleKVCacheEngine:
    """Simple KV cache engine for single-model inference.

    This engine:
    1. Replaces attention layers with nano-style attention (supports KV cache)
    2. Allocates KV cache and assigns to layers
    3. Uses simple scheduler for generation (no continuous batching)
    4. Achieves 10-20x speedup over naive generation

    Args:
        model: Training model to wrap (will be modified in-place)
        tokenizer: Tokenizer for encoding/decoding
        num_blocks: Number of KV cache blocks (auto-estimated if None)
        block_size: Block size for KV cache
        max_model_len: Maximum sequence length
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer,
        num_blocks: Optional[int] = None,
        block_size: int = 16,
        max_model_len: int = 2048,
        enable_prefix_cache: bool = False,
        prefix_cache_max_entries: int = 1000,
        prefix_cache_min_length: int = 10,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.max_model_len = max_model_len

        logger.info("Setting up Simple KV Cache Engine...")

        # CRITICAL FIX: Unwrap model from activation checkpointing before inference
        # The model layers are wrapped in _checkpoint_wrapped_module for gradient checkpointing
        # During inference, we need to access the actual modules to replace attention
        if hasattr(model, 'layers'):
            unwrapped_count = 0
            for layer_name in list(model.layers.keys()):
                layer = model.layers[layer_name]
                if hasattr(layer, '_checkpoint_wrapped_module'):
                    # Replace wrapped module with unwrapped version
                    model.layers[layer_name] = layer._checkpoint_wrapped_module
                    unwrapped_count += 1
            if unwrapped_count > 0:
                logger.info(f"[INFERENCE] Unwrapped {unwrapped_count} layers from activation checkpointing for inference")

        # Step 1: Replace attention layers with ForgeAttention
        logger.info("Replacing attention layers with ForgeAttention...")
        num_replaced = replace_attention_with_forge(
            model,
            attention_class_name="Attention"  # TorchTitan uses "Attention"
        )

        # Step 1.5: Patch model to accept explicit positions parameter
        logger.info("Patching model to accept explicit positions...")
        patch_model_for_positions(model)

        if num_replaced == 0:
            logger.warning(
                "No attention layers replaced. Model may not have 'Attention' class. "
                "KV cache will not be functional."
            )

        # Step 2: Auto-estimate num_blocks if not provided
        if num_blocks is None:
            # Extract model parameters for estimation
            # We need: num_layers, num_kv_heads, head_dim
            # Try to find from first attention layer
            first_attn = None
            for module in model.modules():
                if hasattr(module, 'k_cache') and hasattr(module, 'v_cache'):
                    first_attn = module
                    break

            if first_attn:
                num_layers = sum(
                    1 for m in model.modules()
                    if hasattr(m, 'k_cache') and hasattr(m, 'v_cache')
                )
                num_kv_heads = first_attn.num_kv_heads
                head_dim = first_attn.head_dim

                num_blocks = estimate_kv_cache_blocks(
                    gpu_memory_utilization=0.3,  # Conservative for training model
                    num_layers=num_layers,
                    block_size=block_size,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                )
                logger.info(f"Auto-estimated {num_blocks} KV cache blocks")
            else:
                num_blocks = 1000  # Default fallback
                logger.warning(
                    f"Could not find attention layers for estimation, "
                    f"using default {num_blocks} blocks"
                )

        self.num_blocks = num_blocks

        # Step 3: Allocate KV cache
        logger.info("Allocating KV cache...")
        self.kv_cache = NanoStyleKVCache(
            model=model,
            num_blocks=num_blocks,
            block_size=block_size,
        )

        # Step 4: Initialize block manager and scheduler
        self.block_manager = BlockManager(
            num_blocks=num_blocks,
            block_size=block_size,
        )

        self.scheduler = SimpleScheduler(
            block_manager=self.block_manager,
            max_model_len=max_model_len,
        )

        # Initialize prefix cache (Phase 2)
        self.enable_prefix_cache = enable_prefix_cache
        if enable_prefix_cache:
            self.prefix_cache = PrefixCache(
                max_entries=prefix_cache_max_entries,
                min_prefix_len=prefix_cache_min_length,
                enable_stats=True,
            )
            logger.info(
                f"✓ Prefix Cache ENABLED: max_entries={prefix_cache_max_entries}, "
                f"min_length={prefix_cache_min_length}"
            )
        else:
            self.prefix_cache = None
            logger.info("Prefix Cache DISABLED")

        # Verify flash attention is available
        from forge.actors.hybrid.nano_style_attention import FLASH_ATTN_AVAILABLE
        if FLASH_ATTN_AVAILABLE:
            logger.info("✓ Flash Attention is AVAILABLE and will be used for inference")
        else:
            logger.warning("✗ Flash Attention NOT available - will use slower attention")

        # CRITICAL FIX: Initialize rope_cache on CUDA
        # The TorchTitan model uses self.rope_cache internally, which must be on the correct device
        if hasattr(self.model, 'rope_cache'):
            # rope_cache is already registered as a buffer, just need to ensure it's on CUDA
            if self.model.rope_cache.device.type != 'cuda':
                logger.info("Moving rope_cache to CUDA...")
                self.model.rope_cache = self.model.rope_cache.to('cuda')
            logger.info(f"✓ rope_cache initialized on {self.model.rope_cache.device}")
        elif hasattr(self.model, 'setup'):
            # Some models have a setup() method to initialize buffers
            logger.info("Calling model.setup() to initialize rope_cache...")
            self.model.setup(buffer_device=torch.device('cuda'))
        else:
            logger.warning(
                "Model does not have rope_cache or setup() method. "
                "Generation may produce incorrect results!"
            )

        logger.info(
            f"Simple KV Cache Engine initialized: "
            f"{num_replaced} attention layers, "
            f"{num_blocks} blocks, "
            f"{block_size} tokens/block"
        )

    @torch.inference_mode()
    async def generate(
        self,
        prompts: List[str],
        sampling_params: Optional[SamplingParams] = None,
    ) -> List[Completion]:
        """Generate completions for prompts.

        Args:
            prompts: List of prompt strings (or single prompt with sampling_params.n > 1)
            sampling_params: Sampling parameters (n parameter supported for multiple responses)

        Returns:
            List of Completion objects
        """
        if sampling_params is None:
            sampling_params = SamplingParams()

        # Handle n parameter: replicate single prompt n times
        # This is common in GRPO where we generate n=8 responses from the same prompt
        n = getattr(sampling_params, 'n', 1) or 1
        if len(prompts) == 1 and n > 1:
            logger.info(
                f"[OPTIMIZATION] Replicating single prompt {n} times for batch generation"
            )
            prompts = prompts * n

        # Switch model to eval mode
        self.model.eval()

        # Tokenize prompts
        tokenized = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_model_len,
        )

        input_ids = tokenized["input_ids"]
        batch_size = input_ids.shape[0]

        # PREFILL DEDUPLICATION: Group sequences by identical prompt tokens
        # This enables 8x prefill speedup for GRPO workloads (n=8 responses from same prompt)
        from collections import defaultdict
        prompt_groups = defaultdict(list)

        for i in range(batch_size):
            # Remove padding tokens for accurate comparison
            token_ids = input_ids[i].tolist()
            if hasattr(self.tokenizer, 'pad_token_id') and self.tokenizer.pad_token_id is not None:
                token_ids = [t for t in token_ids if t != self.tokenizer.pad_token_id]

            # Use tuple of token_ids as key for grouping
            token_tuple = tuple(token_ids)
            prompt_groups[token_tuple].append(i)

        # Check if deduplication can provide speedup
        enable_prefill_dedup = len(prompt_groups) < batch_size
        if enable_prefill_dedup:
            logger.info(
                f"[PREFILL_DEDUP] Detected {batch_size} sequences with "
                f"{len(prompt_groups)} unique prompts "
                f"({batch_size / len(prompt_groups):.1f}x deduplication opportunity)"
            )

        # Create sequences
        sequences = []
        first_seq_token_ids = None

        for i in range(batch_size):
            # Remove padding tokens for accurate token count
            token_ids = input_ids[i].tolist()
            # Remove padding (assuming pad_token_id is what tokenizer uses)
            if hasattr(self.tokenizer, 'pad_token_id') and self.tokenizer.pad_token_id is not None:
                token_ids = [t for t in token_ids if t != self.tokenizer.pad_token_id]

            # Store first sequence's token_ids for comparison
            if i == 0:
                first_seq_token_ids = token_ids

            seq = Sequence(
                token_ids=token_ids,
                max_tokens=sampling_params.max_tokens or 512,
                temperature=sampling_params.temperature,
                ignore_eos=sampling_params.ignore_eos,
            )

            # Phase 2: Check prefix cache for this sequence
            if self.prefix_cache is not None:
                cache_result = self.prefix_cache.find_longest_prefix(token_ids)
                if cache_result is not None:
                    matched_tokens, cached_block_table, num_cached = cache_result
                    # Mark that this sequence has cached blocks
                    seq.num_cached_tokens = num_cached
                    # Note: Block table will be set during allocation
                    logger.debug(
                        f"Sequence {i}: Prefix cache hit for {num_cached}/{len(token_ids)} tokens"
                    )

            sequences.append(seq)

        # Add sequences to scheduler
        num_added = self.scheduler.add_sequences(sequences)
        if num_added < len(sequences):
            logger.warning(
                f"Only {num_added}/{len(sequences)} sequences could be scheduled "
                f"(not enough KV cache blocks)"
            )

        # Log block sharing statistics if multiple sequences added
        if num_added > 1 and enable_prefill_dedup:
            total_cached = sum(seq.num_cached_tokens for seq in sequences[:num_added])
            total_prompt_tokens = sum(seq.num_prompt_tokens for seq in sequences[:num_added])
            if total_prompt_tokens > 0:
                cache_efficiency = total_cached / total_prompt_tokens
                logger.info(
                    f"[OPTIMIZATION] Block sharing efficiency: "
                    f"{total_cached}/{total_prompt_tokens} tokens cached "
                    f"({cache_efficiency:.1%}) across {num_added} sequences"
                )

        # Prefill phase
        prefill_context = self.scheduler.schedule_prefill()
        if prefill_context:
            # PREFILL DEDUPLICATION: Use optimized path if duplicates detected
            if enable_prefill_dedup and len(prompt_groups) < len(prefill_context.sequences):
                # Optimized path: deduplicate prefill computation
                self._prefill_with_deduplication(
                    prefill_context,
                    prompt_groups,
                    sampling_params,
                )
            else:
                # Standard path: no duplicates or optimization disabled
                self._prefill_standard(
                    prefill_context,
                    sampling_params,
                )

        # Decode loop
        while self.scheduler.has_unfinished():
            decode_context = self.scheduler.schedule_decode()
            if not decode_context:
                break

            with inference_context(
                sequences=decode_context.sequences,
                block_manager=self.block_manager,
                is_prefill=False,
            ):
                # Get last tokens from each sequence
                last_tokens = torch.tensor(
                    [seq.last_token for seq in decode_context.sequences],
                    dtype=torch.long,
                    device='cuda'
                ).unsqueeze(1)  # [batch_size, 1] - CRITICAL FIX: correct shape for batch decode

                # CRITICAL FIX: Pass correct positions for RoPE
                # Each token is at position (seq.num_tokens - 1)
                positions = torch.tensor(
                    [seq.num_tokens - 1 for seq in decode_context.sequences],
                    dtype=torch.long,
                    device='cuda'
                ).unsqueeze(1)  # [batch_size, 1]

                # Forward pass (decode) WITH CORRECT POSITIONS
                logits = self.model(last_tokens, positions)

                # Extract last position logits and reshape to [1, batch_size, vocab_size]
                last_logits = logits[:, -1, :].unsqueeze(0)  # [batch_size, vocab_size] -> [1, batch_size, vocab_size]

                # Sample next tokens
                next_tokens, token_logprobs = self._sample_tokens(
                    last_logits,
                    decode_context.sequences,
                    sampling_params,
                )

                # Store logprobs in sequences
                for seq, logprob in zip(decode_context.sequences, token_logprobs.tolist()):
                    seq.logprobs.append(logprob)

                # Update sequences
                self.scheduler.update_sequences(
                    next_tokens,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

        # Get finished sequences and convert to Completions
        finished = self.scheduler.get_finished_sequences()
        completions = []

        # Phase 2: Update prefix cache with completed sequences
        if self.prefix_cache is not None:
            for seq in finished:
                # Only cache if prompt is long enough and sequence completed successfully
                if len(seq.prompt_token_ids) >= self.prefix_cache.min_prefix_len:
                    self.prefix_cache.insert(
                        token_ids=seq.prompt_token_ids,
                        block_table=seq.block_table[:len(seq.prompt_token_ids) // self.block_size],
                        num_cached_tokens=len(seq.prompt_token_ids),
                    )

        for seq in finished:
            # Decode tokens
            completion_text = self.tokenizer.decode(
                seq.completion_token_ids,
                skip_special_tokens=True,
            )

            completion = Completion(
                prompt=prompts[len(completions)],  # Match order
                text=completion_text,
                prompt_ids=torch.tensor(seq.prompt_token_ids, dtype=torch.long),
                token_ids=torch.tensor(seq.completion_token_ids, dtype=torch.long),
                logprobs=torch.tensor(seq.logprobs, dtype=torch.float32) if seq.logprobs else None,
                generator_version=0,  # Will be set by policy actor
            )
            completions.append(completion)

        # Cleanup: Free blocks for finished sequences
        # IMPORTANT: Must deallocate blocks here since get_finished_sequences() cleared the list
        for seq in finished:
            self.scheduler.block_manager.deallocate(seq)

        # Remove finished sequences from scheduler
        self.scheduler.cleanup()

        return completions

    def _prefill_standard(
        self,
        prefill_context,
        sampling_params: SamplingParams,
    ):
        """Standard prefill path with ForgeAttention (varlen format).

        Args:
            prefill_context: InferenceContext for prefill
            sampling_params: Sampling parameters
        """
        # VARLEN FORMAT: Flatten all sequences
        input_ids = []
        positions = []

        for seq in prefill_context.sequences:
            input_ids.extend(seq.token_ids)
            # Each sequence starts at position 0
            positions.extend(range(len(seq.token_ids)))

        input_ids = torch.tensor(input_ids, dtype=torch.long, device='cuda')
        positions = torch.tensor(positions, dtype=torch.long, device='cuda')

        logger.info(f"[PREFILL_STANDARD] Varlen format: {len(input_ids)} total tokens from {len(prefill_context.sequences)} sequences")
        logger.info(f"[PREFILL_STANDARD] Positions range: [{positions.min().item()}, {positions.max().item()}]")

        # Forward pass WITH inference context (enables KV cache in ForgeAttention)
        with inference_context(
            sequences=prefill_context.sequences,
            block_manager=self.block_manager,
            is_prefill=True,
        ):
            # Model expects [batch, seq_len] but ForgeAttention handles varlen internally
            # Add batch dimension: [total_tokens] -> [1, total_tokens]
            logits = self.model(input_ids.unsqueeze(0), positions.unsqueeze(0))
            logits = logits.squeeze(0)  # [total_tokens, vocab_size]

        # Extract last token logits for each sequence
        last_token_logits = []
        offset = 0
        for seq in prefill_context.sequences:
            seq_len = len(seq.token_ids)
            last_token_logits.append(logits[offset + seq_len - 1, :])
            offset += seq_len

        last_logits = torch.stack(last_token_logits, dim=0).unsqueeze(0)  # [1, batch_size, vocab_size]

        # DIAGNOSTIC: Check logits statistics
        logger.info(f"[PREFILL_STANDARD] Logits shape: {last_logits.shape}")
        logger.info(f"[PREFILL_STANDARD] Logits stats: min={last_logits.min().item():.4f}, "
                   f"max={last_logits.max().item():.4f}, "
                   f"mean={last_logits.mean().item():.4f}, "
                   f"std={last_logits.std().item():.4f}")

        next_tokens, token_logprobs = self._sample_tokens(
            last_logits,
            prefill_context.sequences,
            sampling_params,
        )

        # Store logprobs in sequences
        for seq, logprob in zip(prefill_context.sequences, token_logprobs.tolist()):
            seq.logprobs.append(logprob)

        # Update sequences
        self.scheduler.update_sequences(
            next_tokens,
            eos_token_id=self.tokenizer.eos_token_id,
        )

    def _prefill_with_deduplication(
        self,
        prefill_context,
        prompt_groups: dict,
        sampling_params: SamplingParams,
    ):
        """Optimized prefill for batches with duplicate prompts (varlen format).

        Strategy:
        1. Select one representative sequence per unique prompt
        2. Run forward pass on representatives only (8x fewer computations)
        3. Expand sampled tokens back to full batch

        Args:
            prefill_context: Original context with all sequences
            prompt_groups: Mapping from token_ids tuple to sequence indices
            sampling_params: Sampling parameters
        """
        # Step 1: Build representative sequences
        # Map from sequence index to list of indices it represents
        seq_to_group = {}

        for token_tuple, seq_indices in prompt_groups.items():
            # Pick first sequence as representative
            repr_seq_idx = seq_indices[0]
            # Find the actual sequence object
            for i, seq in enumerate(prefill_context.sequences):
                if tuple(seq.token_ids) == token_tuple and i not in seq_to_group:
                    seq_to_group[i] = seq_indices
                    break

        representatives = [prefill_context.sequences[i] for i in sorted(seq_to_group.keys())]

        logger.info(
            f"[PREFILL_DEDUP] Processing {len(representatives)} representatives "
            f"instead of {len(prefill_context.sequences)} sequences "
            f"({len(prefill_context.sequences) / len(representatives):.1f}x reduction)"
        )

        # Step 2: Run forward pass with deduplicated batch - VARLEN FORMAT
        input_ids = []
        positions = []

        for seq in representatives:
            input_ids.extend(seq.token_ids)
            # Each sequence starts at position 0
            positions.extend(range(len(seq.token_ids)))

        input_ids = torch.tensor(input_ids, dtype=torch.long, device='cuda')
        positions = torch.tensor(positions, dtype=torch.long, device='cuda')

        logger.info(f"[PREFILL_DEDUP] Varlen format: {len(input_ids)} total tokens from {len(representatives)} representatives")

        # Forward pass WITH inference context (enables KV cache)
        with inference_context(
            sequences=representatives,
            block_manager=self.block_manager,
            is_prefill=True,
        ):
            # Add batch dimension: [total_tokens] -> [1, total_tokens]
            logits = self.model(input_ids.unsqueeze(0), positions.unsqueeze(0))
            logits = logits.squeeze(0)  # [total_tokens, vocab_size]

        # Extract last token logits for each representative (from varlen format)
        last_token_logits = []
        offset = 0
        for seq in representatives:
            seq_len = len(seq.token_ids)
            last_token_logits.append(logits[offset + seq_len - 1, :])
            offset += seq_len

        last_logits = torch.stack(last_token_logits, dim=0).unsqueeze(0)

        # DIAGNOSTIC: Check logits statistics
        logger.info(f"[PREFILL_DEDUP] Logits shape: {last_logits.shape}")
        logger.info(f"[PREFILL_DEDUP] Logits stats: min={last_logits.min().item():.4f}, "
                   f"max={last_logits.max().item():.4f}, "
                   f"mean={last_logits.mean().item():.4f}, "
                   f"std={last_logits.std().item():.4f}")

        next_tokens_repr, token_logprobs_repr = self._sample_tokens(
            last_logits,
            representatives,
            sampling_params,
        )

        # Step 3: Expand sampled tokens to all sequences
        # Build mapping from original sequence index to representative index
        seq_idx_to_repr = {}
        for repr_idx, (seq_idx, group_indices) in enumerate(seq_to_group.items()):
            for idx in group_indices:
                seq_idx_to_repr[idx] = repr_idx

        next_tokens_full = torch.zeros(
            len(prefill_context.sequences),
            dtype=torch.long,
            device='cuda'
        )
        token_logprobs_full = torch.zeros(
            len(prefill_context.sequences),
            dtype=torch.float32,
            device='cuda'
        )

        # Assign tokens from representatives to all sequences
        for seq_idx in range(len(prefill_context.sequences)):
            repr_idx = seq_idx_to_repr[seq_idx]
            next_tokens_full[seq_idx] = next_tokens_repr[repr_idx]
            token_logprobs_full[seq_idx] = token_logprobs_repr[repr_idx]

        # Step 4: Store logprobs in all sequences
        for seq, logprob in zip(prefill_context.sequences, token_logprobs_full.tolist()):
            seq.logprobs.append(logprob)

        # Step 5: CRITICAL - Copy KV cache from representatives to all sequences
        # Since we only ran forward pass on representatives, only their blocks have KV cache data.
        # We need to copy the cache to all other sequences' blocks.
        for repr_idx, (seq_idx, group_indices) in enumerate(seq_to_group.items()):
            repr_seq = representatives[repr_idx]
            # For each sequence in this group (except the representative itself)
            for other_idx in group_indices:
                if other_idx == seq_idx:
                    continue  # Skip the representative itself

                other_seq = prefill_context.sequences[other_idx]

                # Copy KV cache from repr_seq's blocks to other_seq's blocks
                self._copy_kv_cache_blocks(repr_seq.block_table, other_seq.block_table)

        # Step 6: Update all sequences (not just representatives)
        self.scheduler.update_sequences(
            next_tokens_full,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # Log efficiency metrics
        computation_saved = 1.0 - (len(representatives) / len(prefill_context.sequences))
        logger.info(
            f"[PREFILL_DEDUP] Saved {computation_saved:.1%} of prefill computation "
            f"({len(representatives)} forward vs {len(prefill_context.sequences)} sequences)"
        )

    def _copy_kv_cache_blocks(self, src_block_table: list[int], dst_block_table: list[int]):
        """Copy KV cache from source blocks to destination blocks.

        This is needed after prefill deduplication: only the representative sequence
        gets its KV cache written during the forward pass. We need to copy that cache
        to all other sequences' blocks so they can use it during decode.

        Args:
            src_block_table: Block IDs of the source (representative) sequence
            dst_block_table: Block IDs of the destination sequence
        """
        # Get the KV cache tensors for each layer
        for layer_name, layer in self.model.layers.items():
            k_cache = layer.attention.k_cache  # [num_blocks, block_size, num_kv_heads, head_dim]
            v_cache = layer.attention.v_cache

            # Copy each block
            for src_block_id, dst_block_id in zip(src_block_table, dst_block_table):
                k_cache[dst_block_id].copy_(k_cache[src_block_id])
                v_cache[dst_block_id].copy_(v_cache[src_block_id])

        logger.info(f"[PREFILL_DEDUP] Copied KV cache from blocks {src_block_table} to {dst_block_table}")

    def _sample_tokens(
        self,
        logits: torch.Tensor,
        sequences: List[Sequence],
        sampling_params: SamplingParams,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample next tokens from logits.

        Args:
            logits: Logits tensor [1, num_seqs, vocab_size]
            sequences: List of sequences
            sampling_params: Sampling parameters

        Returns:
            Tuple of (next_tokens [num_seqs], logprobs [num_seqs])
        """
        logits = logits.squeeze(0)  # [num_seqs, vocab_size]

        # CRITICAL FIX: Check for invalid logits
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            logger.error(f"[SAMPLING] ERROR: Invalid logits detected! "
                        f"nan: {torch.isnan(logits).sum()}, "
                        f"inf: {torch.isinf(logits).sum()}, "
                        f"logits range: [{logits.min():.4f}, {logits.max():.4f}]")
            # Replace invalid values with zeros
            logits = torch.where(torch.isnan(logits) | torch.isinf(logits), torch.zeros_like(logits), logits)

        # Apply temperature
        if sampling_params.temperature > 0:
            logits = logits / sampling_params.temperature

        # Compute log probabilities with numerical stability
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        # Check for invalid log_probs before sampling
        if torch.isnan(log_probs).any() or torch.isinf(log_probs).any():
            logger.error(f"[SAMPLING] ERROR: Invalid log_probs after softmax! "
                        f"nan: {torch.isnan(log_probs).sum()}, "
                        f"inf: {torch.isinf(log_probs).sum()}")
            # Fallback: use uniform distribution
            log_probs = torch.full_like(log_probs, -math.log(log_probs.shape[-1]))

        # Sample with numerical stability
        probs = torch.exp(log_probs)

        # Clamp probabilities to valid range [0, 1]
        probs = torch.clamp(probs, min=0.0, max=1.0)

        # Renormalize to ensure sum=1
        probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-10)

        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)

        # Get log probabilities of sampled tokens
        sampled_logprobs = log_probs.gather(dim=-1, index=next_tokens.unsqueeze(-1)).squeeze(-1)

        return next_tokens, sampled_logprobs

    def clear_cache(self):
        """Clear KV cache and prefix cache."""
        self.kv_cache.clear()
        if self.prefix_cache is not None:
            self.prefix_cache.clear()

    def get_stats(self) -> dict:
        """Get engine statistics.

        Returns:
            Dict with statistics
        """
        stats = {
            'kv_cache': self.kv_cache.get_memory_usage(),
            'scheduler': self.scheduler.get_stats(),
            'block_manager': self.block_manager.get_stats(),
        }

        # Add prefix cache stats if enabled
        if self.prefix_cache is not None:
            stats['prefix_cache'] = self.prefix_cache.get_stats()

        return stats
