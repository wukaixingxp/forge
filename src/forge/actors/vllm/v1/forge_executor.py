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
        param_map = self._build_param_map(model)
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

        # Sample a few parameters from different layers
        sample_names = [
            "model.layers.0.self_attn.qkv_proj.weight",
            "model.layers.0.mlp.gate_up_proj.weight",
            "model.layers.17.self_attn.qkv_proj.weight",  # Middle layer
            "model.layers.35.self_attn.qkv_proj.weight",  # Last layer
            "model.embed_tokens.weight",
        ]

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
                # For GQA models, Q has more heads than K/V, so sizes differ
                # Order: Q, K, V
                qkv_part = merge_type.split("_")[-1]  # 'q', 'k', or 'v'

                # Get head configuration from vLLM model config for GQA support
                model_config = self.vllm_config.model_config
                parallel_config = self.vllm_config.parallel_config
                # These return total heads, not per-rank
                total_attention_heads = model_config.get_num_attention_heads(parallel_config)
                total_kv_heads = model_config.get_num_kv_heads(parallel_config)
                head_dim = model_config.get_head_size()

                # Calculate sizes per TP rank (heads are sharded across TP ranks)
                q_size_per_rank = (total_attention_heads // tp_size) * head_dim
                kv_size_per_rank = (total_kv_heads // tp_size) * head_dim

                # Slice the source tensor for TP
                src_part_size = src_shape[0] // tp_size
                tensor = tensor[tp_rank * src_part_size : (tp_rank + 1) * src_part_size]

                # Calculate offset based on Q/K/V part
                if qkv_part == 'q':
                    start_idx = 0
                    part_size = q_size_per_rank
                elif qkv_part == 'k':
                    start_idx = q_size_per_rank
                    part_size = kv_size_per_rank
                else:  # 'v'
                    start_idx = q_size_per_rank + kv_size_per_rank
                    part_size = kv_size_per_rank

                end_idx = start_idx + part_size
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
