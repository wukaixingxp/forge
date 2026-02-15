# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Forge-specific MonarchExecutor with TorchStore weight sync.

This module extends the upstream-compatible MonarchExecutor with TorchStore
integration for weight synchronization in RL training loops. It provides:

- ForgeWorkerWrapper: Extends WorkerWrapper with TorchStore weight loading
- ForgeMonarchExecutor: Extends MonarchExecutor with TorchStore Controller handling

Use this executor when you need weight updates from TorchStore (e.g., GRPO training).
For inference-only workloads, use the base MonarchExecutor directly.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
from typing import Optional

import cloudpickle
from forge.actors._torchstore_utils import extract_param_name, get_param_prefix
from forge.actors.vllm.v1.monarch_executor import MonarchExecutor, WorkerWrapper
from forge.observability.perf_tracker import trace
from forge.util._shared_tensor import SharedTensorHandle
from monarch.actor import endpoint
from torchstore.client import LocalClient

logger = logging.getLogger(__name__)


class ForgeWorkerWrapper(WorkerWrapper):
    """Worker wrapper with TorchStore weight sync capabilities."""

    def __init__(self, vllm_config):
        super().__init__(vllm_config)
        self._torchstore_controller = None
        self._torchstore_client: Optional[LocalClient] = None
        self._cached_param_map = None

    @endpoint
    def set_torchstore_controller(self, controller) -> None:
        """Store TorchStore Controller reference for weight updates.

        Workers run in a subprocess with a different _controller_controller,
        so they can't find the Controller via get_or_spawn_controller.
        The Controller reference is passed explicitly from ForgeMonarchExecutor.
        """
        self._torchstore_controller = controller
        self._torchstore_client = None  # Reset cached client

    @endpoint
    @trace(
        prefix="generator_perf/update_weights/apply_prefetched_weights",
        track_memory=False,
        timer="gpu",
    )
    def apply_prefetched_weights(
        self, shared_memory_handles: dict[str, SharedTensorHandle]
    ) -> int:
        """Load weights from shared memory handles into the model.

        All workers call this method with the same handles (obtained by rank 0
        via prefetch_weights). Each worker loads the weights from shared memory
        into its local model.

        Args:
            shared_memory_handles: Dict mapping param names to SharedTensorHandle
                objects, obtained from prefetch_weights() on rank 0.

        Returns:
            Number of parameters loaded
        """
        if not shared_memory_handles:
            logger.warning(
                "[ForgeWorkerWrapper] Empty handles, apply_prefetched_weights is a no-op"
            )
            return 0

        loaded_count = self._load_from_shared_memory(shared_memory_handles)
        logger.info(
            f"[ForgeWorkerWrapper] Applied {loaded_count} weights from shared memory"
        )
        return loaded_count

    @endpoint
    @trace(
        prefix="generator_perf/update_weights/generator_worker_update",
        track_memory=False,
        timer="gpu",
    )
    def update_weights(
        self,
        version: Optional[int] = None,
        shared_memory_state_dict: Optional[dict[str, SharedTensorHandle]] = None,
    ) -> int:
        """Load weights from torchstore or shared memory.

        Args:
            version: Policy version to load from torchstore (if shared_memory_state_dict is None)
            shared_memory_state_dict: Pre-fetched weights in shared memory (if provided, version is ignored)

        Returns:
            Number of parameters loaded
        """
        if shared_memory_state_dict is not None:
            # Load from shared memory (prefetched weights)
            return self._load_from_shared_memory(shared_memory_state_dict)
        elif version is not None:
            # Load directly from torchstore
            return asyncio.run(self._load_from_torchstore(version))
        else:
            raise ValueError(
                "Either version or shared_memory_state_dict must be provided"
            )

    def _load_from_shared_memory(
        self, state_dict: dict[str, SharedTensorHandle]
    ) -> int:
        """Load weights from shared memory handles."""
        model = self.worker.model_runner.model
        loaded_count = 0
        batch = []
        batch_size = 32

        for name, param_handle in state_dict.items():
            with param_handle.to_shared_tensor() as shared_tensor:
                batch.append((name, shared_tensor.tensor.cuda()))
            loaded_count += 1

            if len(batch) >= batch_size:
                model.load_weights(batch)
                batch = []

        # Load any remaining params
        if batch:
            model.load_weights(batch)

        return loaded_count

    @endpoint
    @trace(
        prefix="generator_perf/update_weights/apply_gpu_weights",
        track_memory=False,
        timer="gpu",
    )
    def apply_gpu_weights(
        self,
        gpu_state_dict: dict[str, any],
    ) -> int:
        """Load weights that are already on GPU (GPU-direct path).

        Unlike apply_prefetched_weights(), tensors may already be on GPU or
        may be sliced for this worker's TP rank. This method handles:
        - Full tensors (replicated params)
        - TP-sliced tensors (dict mapping tp_rank -> tensor)

        Args:
            gpu_state_dict: Dict mapping param names to either:
                - torch.Tensor: Full tensor (replicated) or already-sliced tensor
                - dict[int, torch.Tensor]: TP-sliced tensors keyed by rank

        Returns:
            Number of parameters loaded
        """
        import torch
        from monarch.actor import context

        model = self.worker.model_runner.model

        # Get this worker's TP rank from context
        # In vLLM, workers are indexed by global rank
        rank = context().actor_instance.rank.rank
        tp_size = self.vllm_config.parallel_config.tensor_parallel_size
        tp_rank = rank % tp_size

        loaded_count = 0

        for name, param_data in gpu_state_dict.items():
            if isinstance(param_data, dict):
                # TP-sliced: get this rank's slice
                if tp_rank in param_data:
                    tensor = param_data[tp_rank]
                else:
                    logger.warning(f"No slice for TP rank {tp_rank} in {name}")
                    continue
            elif isinstance(param_data, torch.Tensor):
                # Full tensor (replicated) - use as-is
                tensor = param_data
            else:
                logger.warning(f"Unknown param type for {name}: {type(param_data)}")
                continue

            # Ensure tensor is on GPU
            if not tensor.is_cuda:
                tensor = tensor.cuda()

            # Load into model
            model.load_weights([(name, tensor)])
            loaded_count += 1

        logger.info(
            f"[ForgeWorkerWrapper] Applied {loaded_count} GPU weights (TP rank {tp_rank})"
        )
        return loaded_count

    @endpoint
    @trace(
        prefix="generator_perf/update_weights/receive_weights_ipc",
        track_memory=False,
        timer="gpu",
    )
    def receive_weights_ipc(
        self,
        policy_version: int,
        ipc_handles: dict[str, "CudaIPCHandle"],
    ) -> int:
        """Receive weights via CUDA IPC handles from trainer.

        This is the Phase 2 receiver that reconstructs tensors directly from
        the trainer's GPU memory using CUDA IPC, bypassing serialization entirely.

        Args:
            policy_version: Version number for these weights
            ipc_handles: Dict mapping param names to CudaIPCHandle objects

        Returns:
            Number of parameters loaded
        """
        import torch

        logger.info(f"[IPC] Receiving {len(ipc_handles)} weights for v{policy_version}")

        model = self.worker.model_runner.model
        loaded_count = 0
        batch = []
        batch_size = 32

        for name, handle in ipc_handles.items():
            try:
                # Reconstruct tensor from IPC handle (GPU-direct, no copy)
                tensor = handle.reconstruct_tensor()

                # Clone to ensure we own the data (IPC memory might be freed by trainer)
                tensor = tensor.clone()

                batch.append((name, tensor))
                loaded_count += 1

                if len(batch) >= batch_size:
                    model.load_weights(batch)
                    batch = []

            except Exception as e:
                logger.warning(f"[IPC] Failed to reconstruct {name}: {e}")

        # Load remaining batch
        if batch:
            model.load_weights(batch)

        logger.info(f"[IPC] Loaded {loaded_count} weights via CUDA IPC")
        return loaded_count

    @endpoint
    @trace(
        prefix="generator_perf/update_weights/receive_weights_ipc_sliced",
        track_memory=False,
        timer="gpu",
    )
    def receive_weights_ipc_sliced(
        self,
        policy_version: int,
        ipc_handles_per_rank: dict[int, dict[str, "CudaIPCHandle"]],
    ) -> int:
        """Receive pre-sliced weights via CUDA IPC handles from trainer.

        This endpoint receives weights that have already been sliced for each
        TP rank by the trainer. Each worker selects the handles for its TP rank.

        Args:
            policy_version: Version number for these weights
            ipc_handles_per_rank: Dict mapping tp_rank to {param_name: handle}

        Returns:
            Number of parameters loaded
        """
        import torch
        from monarch.actor import context

        # Get this worker's TP rank
        rank = context().actor_instance.rank.rank
        tp_size = self.vllm_config.parallel_config.tensor_parallel_size
        tp_rank = rank % tp_size

        logger.info(f"[IPC-Sliced] Worker rank {rank} (TP rank {tp_rank}) receiving weights for v{policy_version}")

        # Get handles for this TP rank
        if tp_rank not in ipc_handles_per_rank:
            logger.error(f"[IPC-Sliced] No handles for TP rank {tp_rank}")
            return 0

        ipc_handles = ipc_handles_per_rank[tp_rank]
        logger.info(f"[IPC-Sliced] Receiving {len(ipc_handles)} sliced weights for TP rank {tp_rank}")

        model = self.worker.model_runner.model
        loaded_count = 0

        # Build mapping from HF names to model parameters
        # vLLM uses different naming, so we need to map
        if self._cached_param_map is None:
            self._cached_param_map = self._build_param_map(model)
        param_map = self._cached_param_map
        logger.info(f"[IPC-Sliced] Built param map with {len(param_map)} entries")

        for name, handle in ipc_handles.items():
            try:
                # Reconstruct tensor from IPC handle (GPU-direct, no copy)
                tensor = handle.reconstruct_tensor()

                # Clone to own the data and move to correct GPU
                tensor = tensor.clone()

                # Find the parameter in the model
                if name in param_map:
                    mapping = param_map[name]

                    # Handle merged weights (qkv_proj, gate_up_proj)
                    if isinstance(mapping, tuple):
                        merge_type, param = mapping
                        tensor = self._copy_to_merged_param(
                            merge_type, tensor, param, tp_rank, tp_size
                        )
                        if tensor is not None:
                            loaded_count += 1
                    else:
                        param = mapping
                        # Handle shape mismatch for TP-sharded params
                        if tensor.shape != param.shape:
                            tensor = self._slice_for_tp(name, tensor, param.shape, tp_rank, tp_size)

                        # Direct update bypassing vLLM's weight loader
                        param.data.copy_(tensor)
                        loaded_count += 1
                else:
                    logger.warning(f"[IPC-Sliced] Parameter not found in model: {name}")

            except Exception as e:
                import traceback
                logger.warning(
                    f"[IPC-Sliced] Failed to load {name}: {e}\n"
                    f"  Handle info: device={handle.storage_device}, size={handle.tensor_size}, dtype={handle.dtype}\n"
                    f"  Traceback: {traceback.format_exc()}"
                )

        logger.info(f"[IPC-Sliced] Loaded {loaded_count} weights for TP rank {tp_rank}")
        return loaded_count

    @endpoint
    @trace(
        prefix="generator_perf/update_weights/receive_shards_ipc",
        track_memory=False,
        timer="gpu",
    )
    def receive_shards_ipc(
        self,
        policy_version: int,
        shard_handles: dict[int, dict[str, "CudaIPCHandle"]],  # {fsdp_rank: {param_name: handle}}
        shard_metadata: dict[int, dict[str, dict]],  # {fsdp_rank: {param_name: {global_shape, local_shape, offsets}}}
        fsdp_size: int,
        tp_size: int,
    ) -> int:
        """Receive FSDP shards from all trainer ranks, combine, and load via model.load_weights.

        This endpoint is used when both FSDP and TP are > 1. Instead of having
        the trainer do an all_gather (expensive), each trainer rank sends its
        shard with metadata. Workers combine shards and batch-load via model.load_weights.

        Args:
            policy_version: Version number for these weights
            shard_handles: Dict mapping fsdp_rank to {param_name: IPC handle}
            shard_metadata: Dict mapping fsdp_rank to {param_name: metadata dict}
            fsdp_size: Number of FSDP ranks (shards to combine)
            tp_size: Tensor parallel size for slicing

        Returns:
            Number of parameters loaded
        """
        import torch
        from monarch.actor import context

        # Get this worker's TP rank
        rank = context().actor_instance.rank.rank
        tp_rank = rank % tp_size

        logger.info(
            f"[IPC-Sharded] Worker rank {rank} (TP rank {tp_rank}) receiving shards from "
            f"{fsdp_size} FSDP ranks for v{policy_version}"
        )

        model = self.worker.model_runner.model
        loaded_count = 0

        # Phase 4: Build param_map for fast-path routing
        if self._cached_param_map is None:
            self._cached_param_map = self._build_param_map(model)
        param_map = self._cached_param_map

        # Get all parameter names from first shard (all ranks should have same params)
        first_rank = min(shard_handles.keys())
        first_rank_handles = shard_handles[first_rank]
        param_names = list(first_rank_handles.keys())
        logger.info(f"[IPC-Sharded] Processing {len(param_names)} parameters")

        # Compute target device once (same for all params)
        target_device = f"cuda:{rank % torch.cuda.device_count()}"

        # Phase 3: Create CUDA stream for async cross-GPU copies
        copy_stream = torch.cuda.Stream(device=target_device)

        # Batch for model.load_weights (renamed to merged_batch for clarity)
        merged_batch = []
        batch_size = 32

        for param_name in param_names:
            try:
                mapping = param_map.get(param_name)

                # Phase 4: Fast path for non-merged params
                if mapping is not None and not isinstance(mapping, tuple):
                    param = mapping
                    if self._try_scatter_direct(
                        param_name, param, shard_handles, shard_metadata,
                        tp_rank, tp_size, fsdp_size, first_rank,
                        target_device, copy_stream
                    ):
                        loaded_count += 1
                        continue

                # Standard path: combine all shards for model.load_weights
                # Collect and reconstruct shards from all FSDP ranks
                shards = []
                offsets_list = []

                for fsdp_rank in range(fsdp_size):
                    if fsdp_rank not in shard_handles:
                        logger.warning(f"[IPC-Sharded] Missing shard from FSDP rank {fsdp_rank} for {param_name}")
                        continue

                    handle = shard_handles[fsdp_rank].get(param_name)
                    meta = shard_metadata[fsdp_rank].get(param_name)

                    if handle is None or meta is None:
                        logger.warning(f"[IPC-Sharded] Missing handle/metadata for {param_name} from rank {fsdp_rank}")
                        continue

                    # Phase 3: Wrap shard reconstruction with CUDA stream
                    with torch.cuda.stream(copy_stream):
                        shard_tensor = handle.reconstruct_tensor()
                        if str(shard_tensor.device) != target_device:
                            shard_tensor = shard_tensor.to(target_device, non_blocking=True)
                        else:
                            shard_tensor = shard_tensor.clone()

                    shards.append(shard_tensor)
                    offsets_list.append(meta["offsets"])

                if len(shards) != fsdp_size:
                    logger.warning(f"[IPC-Sharded] Incomplete shards for {param_name}: got {len(shards)}/{fsdp_size}")
                    continue

                # Phase 3: Synchronize before combining shards
                copy_stream.synchronize()

                # Get global shape from first shard's metadata
                global_shape = shard_metadata[first_rank][param_name]["global_shape"]

                # Combine shards into full tensor based on offsets
                # FSDP typically shards along dim 0 (rows)
                full_tensor = self._combine_shards(shards, offsets_list, global_shape)

                # Phase 2: Batch full tensors for model.load_weights
                merged_batch.append((param_name, full_tensor))
                loaded_count += 1

                if len(merged_batch) >= batch_size:
                    model.load_weights(merged_batch)
                    merged_batch = []

            except Exception as e:
                import traceback
                logger.warning(f"[IPC-Sharded] Failed to load {param_name}: {e}\n{traceback.format_exc()}")

        # Load remaining batch
        if merged_batch:
            model.load_weights(merged_batch)

        # Phase 3: Final synchronization before returning
        copy_stream.synchronize()

        logger.info(f"[IPC-Sharded] Loaded {loaded_count} weights for TP rank {tp_rank}")
        return loaded_count

    def _combine_shards(
        self,
        shards: list,
        offsets_list: list[tuple],
        global_shape: tuple,
    ) -> "torch.Tensor":
        """Combine FSDP shards into a full tensor based on offsets.

        FSDP typically shards along dimension 0 (rows). Shards are ordered
        by their offset in that dimension.

        Args:
            shards: List of shard tensors
            offsets_list: List of offset tuples for each shard
            global_shape: Shape of the full combined tensor

        Returns:
            Combined full tensor
        """
        import torch

        if len(shards) == 1:
            return shards[0]

        # Sort shards by their offset in dimension 0
        indexed_shards = list(zip(offsets_list, shards))
        indexed_shards.sort(key=lambda x: x[0][0])  # Sort by first offset

        # Concatenate along dimension 0 (FSDP's shard dimension)
        sorted_shards = [s for _, s in indexed_shards]
        full_tensor = torch.cat(sorted_shards, dim=0)

        return full_tensor

    def _try_scatter_direct(
        self,
        param_name: str,
        param: "torch.nn.Parameter",
        shard_handles: dict,
        shard_metadata: dict,
        tp_rank: int,
        tp_size: int,
        fsdp_size: int,
        first_rank: int,
        target_device: str,
        copy_stream: "torch.cuda.Stream",
    ) -> bool:
        """Try to copy FSDP shards directly to param without combining all shards.

        Returns True if direct scatter succeeded, False to fall back to standard path.
        Handles three cases:
        1. Replicated params (shapes match) — copy one shard
        2. Column-parallel (TP dim 0, same as FSDP) — copy only overlapping shards
        3. Row-parallel (TP dim 1, different from FSDP dim 0) — slice columns first, then cat
        """
        import torch

        meta_first = shard_metadata[first_rank].get(param_name)
        if meta_first is None:
            return False

        global_shape = tuple(meta_first["global_shape"])
        param_shape = tuple(param.shape)

        # Case 1: Replicated (param shape == global shape, no TP slicing needed)
        if param_shape == global_shape:
            handle = shard_handles[first_rank].get(param_name)
            if handle is None:
                return False
            with torch.cuda.stream(copy_stream):
                shard = handle.reconstruct_tensor()
                if str(shard.device) != target_device:
                    shard = shard.to(target_device, non_blocking=True)
                else:
                    shard = shard.clone()
            copy_stream.synchronize()

            if fsdp_size == 1:
                # Single shard IS the full tensor
                param.data.copy_(shard)
            else:
                # Multiple shards but replicated — any single shard may be a subset.
                # Need to combine all if shard != global shape. Fall back.
                if tuple(shard.shape) == global_shape:
                    param.data.copy_(shard)
                else:
                    return False
            return True

        # Case 2: Column-parallel (param dim 0 < global dim 0, dim 1 matches)
        # TP slices rows (dim 0). FSDP also shards rows (dim 0).
        # Each TP rank only needs the FSDP shard(s) that overlap its row range.
        if (len(global_shape) >= 2 and len(param_shape) >= 2
                and param_shape[0] < global_shape[0]
                and param_shape[-1] == global_shape[-1]):
            tp_chunk = param_shape[0]
            tp_start = tp_rank * tp_chunk
            tp_end = tp_start + tp_chunk

            # Collect overlapping shard regions
            parts = []  # (dst_offset, src_slice_start, src_slice_end, fsdp_rank)
            for fsdp_rank in range(fsdp_size):
                meta = shard_metadata.get(fsdp_rank, {}).get(param_name)
                if meta is None:
                    return False
                shard_start = meta["offsets"][0]
                shard_rows = meta["local_shape"][0]
                shard_end = shard_start + shard_rows

                # Overlap between [tp_start, tp_end) and [shard_start, shard_end)
                ov_start = max(tp_start, shard_start)
                ov_end = min(tp_end, shard_end)
                if ov_start < ov_end:
                    parts.append((
                        ov_start - tp_start,      # dst offset in param
                        ov_start - shard_start,    # src start in shard
                        ov_end - shard_start,      # src end in shard
                        fsdp_rank,
                    ))

            if not parts:
                return False

            # Reconstruct only needed shards and scatter directly to param
            for dst_offset, src_start, src_end, fsdp_rank in parts:
                handle = shard_handles[fsdp_rank].get(param_name)
                if handle is None:
                    return False
                with torch.cuda.stream(copy_stream):
                    shard = handle.reconstruct_tensor()
                    if str(shard.device) != target_device:
                        shard = shard.to(target_device, non_blocking=True)
                    else:
                        shard = shard.clone()
                copy_stream.synchronize()

                src_slice = shard[src_start:src_end]
                dst_end = dst_offset + (src_end - src_start)
                param.data[dst_offset:dst_end].copy_(src_slice)

            return True

        # Case 3: Row-parallel (param dim 1 < global dim 1, dim 0 matches)
        # TP slices columns (dim 1). FSDP shards rows (dim 0).
        # Need all shards (all rows), but can slice columns from each FIRST
        # to avoid allocating the full combined tensor.
        if (len(global_shape) == 2 and len(param_shape) == 2
                and param_shape[0] == global_shape[0]
                and param_shape[1] < global_shape[1]):
            tp_chunk = param_shape[1]
            tp_start = tp_rank * tp_chunk
            tp_end = tp_start + tp_chunk

            sliced_shards = []
            for fsdp_rank in range(fsdp_size):
                handle = shard_handles[fsdp_rank].get(param_name)
                meta = shard_metadata[fsdp_rank].get(param_name)
                if handle is None or meta is None:
                    return False

                with torch.cuda.stream(copy_stream):
                    shard = handle.reconstruct_tensor()
                    if str(shard.device) != target_device:
                        shard = shard.to(target_device, non_blocking=True)
                    else:
                        shard = shard.clone()
                copy_stream.synchronize()

                # Slice columns first (narrow is a view, no copy)
                sliced = shard[:, tp_start:tp_end].contiguous()
                sliced_shards.append((meta["offsets"][0], sliced))

            # Sort by row offset and cat
            sliced_shards.sort(key=lambda x: x[0])
            combined = torch.cat([s for _, s in sliced_shards], dim=0)
            param.data.copy_(combined)
            return True

        # Not a recognized pattern — use standard path
        return False

    async def _get_torchstore_client(self) -> LocalClient:
        """Get or create a LocalClient using the passed Controller reference.

        Workers can't use ts.client() directly because they're in a subprocess
        with a different _controller_controller. Instead, we create a LocalClient
        using the Controller reference passed from ForgeMonarchExecutor.
        """
        if self._torchstore_client is not None:
            return self._torchstore_client

        if self._torchstore_controller is None:
            raise RuntimeError(
                "TorchStore Controller not set. "
                "ForgeMonarchExecutor must call set_torchstore_controller before weight updates."
            )

        strategy = await self._torchstore_controller.get_controller_strategy.call_one()
        self._torchstore_client = LocalClient(
            controller=self._torchstore_controller,
            strategy=strategy,
        )
        return self._torchstore_client

    async def _load_from_torchstore(self, version: int) -> int:
        """Async helper to load from torchstore using the passed Controller."""
        client = await self._get_torchstore_client()
        prefix = get_param_prefix(version)
        matching_keys = await client.keys(prefix)
        model = self.worker.model_runner.model
        loaded_count = 0
        for key in matching_keys:
            name = extract_param_name(key)
            param = await client.get(key)
            model.load_weights([(name, param.cuda())])
            del param
            loaded_count += 1
        return loaded_count

    @endpoint
    def save_model_params(self):
        """Save model parameters before weight update, used for testing purposes only."""
        logger.info("[WorkerWrapper] save model parameters for testing.")
        if not hasattr(self, "_test_prev_params"):
            self._test_prev_params = {}
        for name, param in self.worker.model_runner.model.named_parameters():
            self._test_prev_params[name] = param.detach().cpu()
        logger.info(
            "[WorkerWrapper] finished saving model parameters, len = %d",
            len(self._test_prev_params),
        )

    @endpoint
    def validate_model_params(self, validate_fn):
        """Validate updated model params using validate_fn."""
        logger.info("[WorkerWrapper] start validating model parameters.")
        if not hasattr(self, "_test_prev_params"):
            self._test_prev_params = {}
        return validate_fn(
            self._test_prev_params, self.worker.model_runner.model, logger
        )

    @endpoint
    def get_sample_weights(self) -> dict[str, dict]:
        """Get sample weights for validation.

        Returns statistics for a subset of parameters to verify weight updates
        without transferring full tensors.
        """
        import torch

        model = self.worker.model_runner.model
        sample_params = {}

        # Dynamically find layer indices from the model
        import re
        layer_indices = set()
        for name in dict(model.named_parameters()).keys():
            match = re.search(r'model\.layers\.(\d+)\.', name)
            if match:
                layer_indices.add(int(match.group(1)))

        if layer_indices:
            max_layer = max(layer_indices)
            mid_layer = max_layer // 2
            sample_names = [
                "model.layers.0.self_attn.qkv_proj.weight",
                "model.layers.0.mlp.gate_up_proj.weight",
                f"model.layers.{mid_layer}.self_attn.qkv_proj.weight",
                f"model.layers.{max_layer}.self_attn.qkv_proj.weight",
                "model.embed_tokens.weight",
            ]
        else:
            sample_names = ["model.embed_tokens.weight"]

        for name, param in model.named_parameters():
            if name in sample_names or len(sample_params) < 5:
                with torch.no_grad():
                    sample_params[name] = {
                        "mean": param.data.float().mean().item(),
                        "std": param.data.float().std().item(),
                        "shape": list(param.shape),
                        "sum": param.data.float().sum().item(),
                    }

        logger.info(f"[WorkerWrapper] Sampled {len(sample_params)} parameters for validation")
        return sample_params

    def _build_param_map(self, model) -> dict:
        """Build mapping from HF-style names to model parameters.

        vLLM uses different internal naming, so we map HF names to vLLM params.
        Key differences:
        - vLLM merges q/k/v_proj -> qkv_proj
        - vLLM merges gate/up_proj -> gate_up_proj
        - vLLM uses 'self_attn' while HF may use 'attention'
        """
        import re
        param_map = {}
        vllm_params = {}

        for name, param in model.named_parameters():
            # Direct mapping
            param_map[name] = param
            vllm_params[name] = param

            # Also add without 'model.' prefix
            if name.startswith("model."):
                param_map[name[6:]] = param

        # Build reverse mappings for merged weights
        for vllm_name, param in vllm_params.items():
            # Map separate q/k/v_proj to merged qkv_proj
            # vllm_name example: model.layers.0.self_attn.qkv_proj.weight
            if "qkv_proj" in vllm_name:
                # Replace qkv_proj.weight with individual proj names
                for proj in ['q_proj', 'k_proj', 'v_proj']:
                    hf_name = vllm_name.replace("qkv_proj", proj)
                    merge_key = f"qkv_proj_{proj[0]}"  # qkv_proj_q, qkv_proj_k, qkv_proj_v
                    param_map[hf_name] = (merge_key, param)
                    # Also with attention variations
                    if "self_attn" in hf_name:
                        param_map[hf_name.replace("self_attn", "attention")] = (merge_key, param)

            # Map separate gate/up_proj to merged gate_up_proj
            # vllm_name example: model.layers.0.mlp.gate_up_proj.weight
            if "gate_up_proj" in vllm_name:
                for proj, key in [('gate_proj', 'gate_up_proj_gate'), ('up_proj', 'gate_up_proj_up')]:
                    hf_name = vllm_name.replace("gate_up_proj", proj)
                    param_map[hf_name] = (key, param)

            # Map attention vs self_attn variations for non-merged params
            if "self_attn" in vllm_name and "qkv_proj" not in vllm_name:
                param_map[vllm_name.replace("self_attn", "attention")] = param

        # Handle lm_head which may be tied to embed_tokens
        if "lm_head.weight" not in param_map:
            for name, param in vllm_params.items():
                if "embed_tokens" in name:
                    param_map["lm_head.weight"] = param
                    break

        return param_map

    def _copy_to_merged_param(
        self,
        merge_type: str,
        tensor: "torch.Tensor",
        param: "torch.nn.Parameter",
        tp_rank: int,
        tp_size: int,
    ) -> bool:
        """Copy a tensor to a portion of a merged parameter.

        vLLM merges Q/K/V projections into a single qkv_proj parameter,
        and gate/up projections into gate_up_proj. This method copies
        individual weights into the correct slice of the merged param.

        Args:
            merge_type: One of 'qkv_proj_q', 'qkv_proj_k', 'qkv_proj_v',
                       'gate_up_proj_gate', 'gate_up_proj_up'
            tensor: Source tensor to copy
            param: Target merged parameter
            tp_rank: Tensor parallel rank of this worker
            tp_size: Total tensor parallel size

        Returns:
            True if copy was successful
        """
        import torch

        # Get dimensions
        param_shape = param.shape
        src_shape = tensor.shape

        try:
            if merge_type.startswith("qkv_proj_"):
                # QKV is merged along output dim (dim 0)
                # For GQA models, Q has more heads than K/V, so sizes differ!
                # Order: Q, K, V
                qkv_part = merge_type.split("_")[-1]  # 'q', 'k', or 'v'

                # First, slice source tensor for TP if needed
                src_part_size = src_shape[0] // tp_size
                if tp_size > 1:
                    tensor = tensor[tp_rank * src_part_size : (tp_rank + 1) * src_part_size]

                # The sliced tensor size tells us the actual part size for this component
                sliced_size = tensor.shape[0]

                # For GQA, we need to find the correct offset in qkv_proj
                # vLLM stores: [Q, K, V] where Q may be larger than K, V
                # We need model config to know exact sizes, but we can infer from src tensor:
                # - Q size per TP rank = src_q_shape[0] / tp_size
                # - K/V size per TP rank = src_kv_shape[0] / tp_size
                #
                # As a workaround, compute offset based on which part we're copying
                total_qkv_size = param_shape[0]

                if qkv_part == 'q':
                    start_idx = 0
                elif qkv_part == 'k':
                    # K starts after Q - but we need to know Q's size
                    # Infer Q size: for standard attention Q=K=V, for GQA Q > K=V
                    # We'll use the fact that Q + K + V = total_qkv_size
                    # And the tensor we have is K's size per TP rank
                    # This is complex without model config, so use a heuristic:
                    # Check if total/3 == sliced_size (standard) or not (GQA)
                    q_size_per_tp = total_qkv_size - 2 * sliced_size  # Q = total - K - V
                    start_idx = q_size_per_tp
                else:  # v
                    q_size_per_tp = total_qkv_size - 2 * sliced_size
                    start_idx = q_size_per_tp + sliced_size  # After Q and K

                end_idx = start_idx + sliced_size
                param.data[start_idx:end_idx].copy_(tensor)
                return True

            elif merge_type.startswith("gate_up_proj_"):
                # Gate/Up is merged along output dim (dim 0)
                # Order: gate, up
                part = merge_type.split("_")[-1]  # 'gate' or 'up'
                part_idx = 0 if part == 'gate' else 1

                # Total param shape: [2 * intermediate_size / tp_size, hidden_size]
                total_size = param_shape[0]
                part_size = total_size // 2

                # Slice the source tensor for TP if needed
                if src_shape[0] > part_size:
                    src_part_size = src_shape[0] // tp_size
                    tensor = tensor[tp_rank * src_part_size : (tp_rank + 1) * src_part_size]

                start_idx = part_idx * part_size
                end_idx = start_idx + part_size
                param.data[start_idx:end_idx].copy_(tensor)
                return True

        except Exception as e:
            logger.warning(
                f"[IPC-Sliced] Failed to copy merged param {merge_type}: {e}"
            )

        return False

    def _slice_for_tp(
        self,
        name: str,
        tensor: "torch.Tensor",
        target_shape: tuple,
        tp_rank: int,
        tp_size: int,
    ) -> "torch.Tensor":
        """Slice a tensor to match target shape for TP.

        When the source tensor is full size but the model parameter is
        already sharded for TP, we need to extract the correct slice.
        """
        import torch

        # Determine sharding pattern from parameter name
        if any(p in name for p in ["q_proj", "k_proj", "v_proj", "qkv_proj"]):
            # Column parallel - shard output dim (usually dim 0 for weight)
            if tensor.shape[0] > target_shape[0]:
                shard_size = target_shape[0]
                return tensor[tp_rank * shard_size : (tp_rank + 1) * shard_size, :]
        elif any(p in name for p in ["o_proj", "down_proj"]):
            # Row parallel - shard input dim (usually dim 1 for weight)
            if len(tensor.shape) > 1 and tensor.shape[1] > target_shape[1]:
                shard_size = target_shape[1]
                return tensor[:, tp_rank * shard_size : (tp_rank + 1) * shard_size]
        elif any(p in name for p in ["gate_proj", "up_proj", "gate_up_proj"]):
            # Column parallel
            if tensor.shape[0] > target_shape[0]:
                shard_size = target_shape[0]
                return tensor[tp_rank * shard_size : (tp_rank + 1) * shard_size, :]
        elif any(p in name for p in ["embed_tokens", "lm_head"]):
            # Vocab parallel - shard vocab dim
            if tensor.shape[0] > target_shape[0]:
                shard_size = target_shape[0]
                return tensor[tp_rank * shard_size : (tp_rank + 1) * shard_size, :]

        # If shapes match or no slicing needed, return as-is
        if tensor.shape == target_shape:
            return tensor

        logger.warning(
            f"[IPC-Sliced] Shape mismatch for {name}: "
            f"tensor={tensor.shape}, target={target_shape}, tp_rank={tp_rank}"
        )
        return tensor


class ForgeMonarchExecutor(MonarchExecutor):
    """MonarchExecutor with TorchStore integration for weight sync.

    Extends the base MonarchExecutor to:
    - Deserialize TorchStore Controller from environment
    - Pass Controller to workers for direct weight loading
    - Use ForgeWorkerWrapper instead of base WorkerWrapper
    """

    worker_class = ForgeWorkerWrapper

    def _init_executor(self) -> None:
        """Initialize executor and deserialize TorchStore Controller."""
        super()._init_executor()

        controller_str = os.environ.get("VLLM_TORCHSTORE_CONTROLLER")
        if controller_str:
            logger.info(
                "[ForgeMonarchExecutor] Deserializing TorchStore Controller from environment..."
            )
            self.torchstore_controller = cloudpickle.loads(
                base64.b64decode(controller_str)
            )
            logger.info(
                f"[ForgeMonarchExecutor] TorchStore Controller deserialized: {self.torchstore_controller}"
            )
            self.workers.set_torchstore_controller.call(
                self.torchstore_controller
            ).get()

        else:
            self.torchstore_controller = None
            logger.warning(
                "[ForgeMonarchExecutor] No TorchStore Controller found in environment. "
                "Weight updates via torchstore will not work."
            )

    def shutdown(self):
        """Shutdown workers and stop ProcMeshes."""
        super().shutdown()
