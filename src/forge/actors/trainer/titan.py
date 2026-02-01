# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import Callable

import torch
import torchstore as ts
from forge.actors._torchstore_utils import get_param_key
from forge.api.trainer import ParallelismConfig, TrainerConfig, TrainerStatus
from forge.controller import ForgeActor
from forge.data.utils import batch_to_device
from forge.observability.metrics import record_metric, Reduce
from forge.observability.perf_tracker import Tracer
from forge.rl.loss import create_shifted_targets
from forge.types import TrainBatch
from monarch.actor import endpoint
from torch import Tensor
from torch.distributed.checkpoint._nested_dict import flatten_state_dict
from torchtitan.config.job_config import (
    ActivationCheckpoint,
    Checkpoint,
    Comm,
    Compile,
    Job,
    LRScheduler,
    MemoryEstimation,
    Model,
    Optimizer,
    Parallelism,
    Quantize,
    Training,
)
from torchtitan.experiments.forge.engine import ForgeEngine
from torchtitan.experiments.forge.job_config import ForgeJobConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@dataclass
class TitanTrainer(ForgeActor):
    """A generic trainer actor implementation built on top of TorchTitan.

    Built on top of TorchTitan's training engine, this actor provides a complete training
    loop for reinforcement learning. It performs forward and backward passes with gradient
    computation, optimization steps, and checkpoint management. Unlike the ReferenceModel
    actor which only runs forward passes, RLTrainer actively updates the policy model
    parameters through gradient descent.

    The trainer supports the same distributed training strategies that TorchTitan does,
    including but not limited to, tensor parallelism, data parallelism, and FSDP
    (Fully Sharded Data Parallel). It is typically used in conjunction with ReferenceModel
    for policy optimization algorithms like GRPO (Group Relative Policy Optimization),
    where it optimizes the policy against a loss that includes KL divergence penalties
    from the reference model.

    The trainer handles:
    - Forward and backward propagation with automatic mixed precision (AMP)
    - Optimizer steps with learning rate scheduling
    """

    job: Job = field(default_factory=Job)
    model: Model = field(default_factory=Model)
    optimizer: Optimizer = field(default_factory=Optimizer)
    lr_scheduler: LRScheduler = field(default_factory=LRScheduler)
    training: Training = field(default_factory=Training)
    parallelism: Parallelism = field(default_factory=Parallelism)
    checkpoint: Checkpoint = field(default_factory=Checkpoint)
    activation_checkpoint: ActivationCheckpoint = field(
        default_factory=ActivationCheckpoint
    )
    compile: Compile = field(default_factory=Compile)
    quantize: Quantize = field(default_factory=Quantize)
    comm: Comm = field(default_factory=Comm)
    memory_estimation: MemoryEstimation = field(default_factory=MemoryEstimation)
    # Non JobConfig-related fields
    loss: Callable = lambda logits, **targets: logits
    state_dict_key: str = "model_state_dict"

    def __post_init__(self):
        super().__init__()

        for f in fields(self):
            attr = getattr(self, f.name)
            if isinstance(attr, Mapping):
                setattr(self, f.name, f.type(**attr))
            elif not isinstance(attr, f.type):
                raise TypeError(
                    f"{f.name} should be a {f.type} type or a dict like object"
                )

        self.step = 1  # fragile contract.
        self.num_training_steps = self.training.steps
        self.gradient_accumulation_steps = 1
        self._accumulated_microbatches = 0
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        logger.info("Compiling loss")
        self.loss = torch.compile(self.loss)

    @endpoint
    async def setup(self):
        # TODO: update ForgeEngine to not use ForgeJobConfig
        engine_config = {f.name: getattr(self, f.name) for f in fields(self)}
        for key in {
            "loss",
            "state_dict_key",
        }:
            engine_config.pop(key)  # Not part of job config
        self.engine = ForgeEngine(ForgeJobConfig(**engine_config))
        self.engine.checkpointer.load(step=self.step)
        self.engine.optimizers.zero_grad()

    def forward_backward(self, batch: TrainBatch) -> Tensor:
        model_parts = self.engine.model_parts
        parallel_dims = self.engine.parallel_dims
        optional_context_parallel_ctx = None

        # Create shifted target_ids for next-token prediction
        # target_ids[i] = input_ids[i+1], with loss_mask applied
        batch.loss_inputs["target_ids"] = create_shifted_targets(
            batch.model_inputs["tokens"], batch.loss_inputs.get("loss_mask")
        )

        if parallel_dims.pp_enabled:
            raise NotImplementedError("PP not implemented yet")
        else:
            with self.engine.train_context(optional_context_parallel_ctx):
                assert len(model_parts) == 1
                with self.engine.maybe_enable_amp:
                    logits = model_parts[0](**batch.model_inputs)
                    loss_output = self.loss(logits, **batch.loss_inputs)
                    loss = loss_output.loss

                # Record metrics from loss output
                for metric in loss_output.metrics:
                    value = (
                        metric.value.item()
                        if isinstance(metric.value, torch.Tensor)
                        else metric.value
                    )
                    record_metric(metric.key, value, metric.reduction, metric.timestamp)

                # Free to before bwd to avoid peaking memory
                del logits, loss_output.metrics
                loss.backward()
        self._accumulated_microbatches += 1
        return loss

    @endpoint
    async def train_step(self, batches: list[TrainBatch]) -> float:
        t = Tracer("rl_trainer_perf/step", timer="gpu", track_memory=True)
        t.start()

        self.engine.gc_handler.run(self.step)
        batch = batches[self.engine.dp_rank]
        batch_to_device(batch.model_inputs, self.engine.device)
        batch_to_device(batch.loss_inputs, self.engine.device)

        loss = self.forward_backward(batch)
        torch.distributed.all_reduce(loss)

        t.step("forward_backward")

        current_lr = self.engine.lr_schedulers.schedulers[0].get_last_lr()[0]
        record_metric("rl_trainer/learning_rate", current_lr, Reduce.MIN)

        self.engine.optimizers.step()
        self.engine.optimizers.zero_grad()
        self.engine.lr_schedulers.step()
        self._accumulated_microbatches = 0
        self.step += 1
        t.step("optimizer_step")

        # TODO: delete item() to avoid cpu-gpu sync
        loss = loss.detach().item()
        record_metric("rl_trainer/loss", loss, Reduce.MEAN)

        self.engine.checkpointer.save(
            curr_step=self.step,
            last_step=self.step == self.num_training_steps,
        )
        t.step("save_checkpoint")
        t.stop()
        return loss

    @endpoint
    async def get_config(self) -> TrainerConfig:
        """Get static trainer and model configuration.

        Returns configuration information that doesn't change during training.
        For runtime state like current step, use get_status() instead.

        Returns:
            TrainerConfig containing model name, model_config, and parallelism settings

        """
        parallel_dims = self.engine.parallel_dims
        parallelism = ParallelismConfig(
            dp_degree=parallel_dims.dp_shard * parallel_dims.dp_replicate,
            tp_degree=parallel_dims.tp,
            pp_degree=parallel_dims.pp,
            cp_degree=parallel_dims.cp,
            ep_degree=parallel_dims.ep,
            world_size=parallel_dims.world_size,
            dp_rank=self.engine.dp_rank,
            tp_rank=parallel_dims.tp_coord,
            device=str(self.engine.device),
        )
        return TrainerConfig(
            model_name=self.model.name,
            model_config=self.model.model_dump(),
            parallelism=parallelism,
        )

    @endpoint
    async def get_status(self) -> TrainerStatus:
        """Get current runtime status of the trainer.

        Returns dynamic information about the trainer's current state that changes
        during training.

        Returns:
            TrainerStatus containing current step and accumulated batch count

        """
        return TrainerStatus(
            step=self.step,
            accumulated_microbatches=self._accumulated_microbatches,
        )

    @endpoint
    async def clear_gradients(self) -> None:
        """Clear accumulated gradients without applying them.

        Use this when you need to discard accumulated gradients without performing
        an optimizer step. Common scenarios:
        - Exception during gradient accumulation
        - Skipping a training step due to some condition
        - Recovering from OOM or other errors

        This is equivalent to calling optimizer.zero_grad() and resetting internal
        accumulation counters.
        """
        self.engine.optimizers.zero_grad()
        self._accumulated_microbatches = 0

    @endpoint
    async def save(
        self,
        name: str | None = None,
        path: str | None = None,
        weights_only: bool = False,
    ) -> str:
        """Save trainer state or weights to persistent storage.

        By default, saves complete training state (model weights, optimizer state,
        learning rate scheduler state, and step counter).

        Args:
            name: Not supported. TitanTrainer uses step-based checkpoint naming.
            path: Not supported. TitanTrainer uses checkpoint.folder from config.
            weights_only: Not supported. TitanTrainer always saves full training state.

        Returns:
            Full path where checkpoint was saved
        """
        if name is not None:
            raise NotImplementedError(
                "TitanTrainer uses step-based checkpoint naming; custom names are not supported"
            )
        if path is not None:
            raise NotImplementedError(
                "TitanTrainer uses the checkpoint.folder from config; custom paths are not supported"
            )
        if weights_only:
            raise NotImplementedError(
                "weights_only is not supported; TitanTrainer always saves full training state"
            )

        self.engine.checkpointer.save(
            curr_step=self.step,
            last_step=False,
        )
        return f"{self.checkpoint.folder}/step-{self.step}"

    @endpoint
    async def load(self, path: str | None = None) -> str:
        """Load a previously saved checkpoint.

        Restores training state from a checkpoint.

        Args:
            path: Not supported. TitanTrainer uses checkpoint.folder from config.

        Returns:
            Path that was loaded
        """
        if path is not None:
            raise NotImplementedError(
                "TitanTrainer uses the checkpoint.folder from config; custom paths are not supported"
            )

        self.engine.checkpointer.load(step=self.step)
        return f"{self.checkpoint.folder}/step-{self.step}"

    @endpoint
    async def push_weights(self, policy_version: int) -> None:
        """Push weights to torchstore in HF format."""
        from torch.distributed.tensor import DTensor

        logger.info(f"Pushing weights for policy version {policy_version}")

        start_time = time.perf_counter()

        # Get model state dict directly from model_parts (works with or without checkpointing)
        # NOTE: For FSDP2 models, state_dict() returns DTensors (sharded), NOT full tensors!
        # We must call .full_tensor() to gather across FSDP ranks before storing.
        if len(self.engine.model_parts) != 1:
            raise RuntimeError("push_weights only supports single model part (no PP)")

        sd = self.engine.model_parts[0].state_dict()
        flattened_state_dict, _ = flatten_state_dict(sd)

        # Convert to HF format if adapter is available
        # Note: when checkpoint.enable=False, checkpointer doesn't have sd_adapter attribute
        sd_adapter = getattr(self.engine.checkpointer, 'sd_adapter', None)
        if sd_adapter is not None:
            hf_state_dict = sd_adapter.to_hf(flattened_state_dict)
        else:
            # Convert native names to HF names using our helper
            hf_state_dict = {
                self._native_to_hf_name(name): param
                for name, param in flattened_state_dict.items()
            }

        # Convert DTensors to full tensors for TorchStore compatibility
        # FSDP2 returns DTensors which TorchStore stores as sharded dicts.
        # TorchStore's get_meta() can't handle this format, causing silent failures.
        # Solution: Gather full tensors on all ranks, but only rank 0 pushes to TorchStore.
        gathered_state_dict = {}
        for name, param in hf_state_dict.items():
            if isinstance(param, DTensor):
                # full_tensor() gathers shards across FSDP ranks
                gathered_state_dict[name] = param.full_tensor()
            else:
                gathered_state_dict[name] = param

        # Only rank 0 pushes to TorchStore to avoid duplicate writes
        if self.engine.dp_rank != 0:
            logger.info(f"Rank {self.engine.dp_rank} skipping push (only rank 0 pushes)")
            return

        # Use batched put for better performance
        total_params = len(gathered_state_dict)

        # Build key -> tensor mapping
        keyed_params = {
            get_param_key(policy_version, name): param
            for name, param in gathered_state_dict.items()
        }

        # Use put_batch for all params at once
        try:
            await ts.put_batch(keyed_params)
        except AttributeError:
            # Fallback for older torchstore without put_batch
            logger.warning("ts.put_batch not available, falling back to individual puts")
            import asyncio
            batch_size = 100
            items = list(keyed_params.items())
            for batch_start in range(0, total_params, batch_size):
                batch_end = min(batch_start + batch_size, total_params)
                batch = items[batch_start:batch_end]

                async def put_param(key: str, param: torch.Tensor) -> None:
                    await ts.put(key, param)

                await asyncio.gather(*[put_param(key, param) for key, param in batch])

        logger.info(f"Push progress: {total_params}/{total_params} params")

        end_time = time.perf_counter()
        logger.info("Completed weights push in %.2f seconds", end_time - start_time)

    @endpoint
    async def push_weights_ipc(
        self,
        policy_version: int,
        generator_workers,  # ActorMesh of generator workers
        tp_size: int = 1,  # Tensor parallel size of generator
    ) -> dict:
        """Push weights directly to generator workers using CUDA IPC.

        This is Phase 2 optimization that bypasses TorchStore entirely for
        same-node deployments. Instead of serializing tensors through RPC,
        we send lightweight CUDA IPC handles (66 bytes each) that allow
        the generator to directly access trainer's GPU memory.

        Key optimizations:
        1. Skip Python serialization - use CUDA IPC handles (66 bytes vs full tensor)
        2. Skip TorchStore - direct trainer -> generator communication
        3. For non-FSDP: skip state_dict() entirely
        4. For TP>1: send handles to all workers (they slice locally)
        5. **NO all_gather required** - IPC handles work with DTensor local shards

        IMPORTANT: Unlike push_weights() which calls full_tensor() to gather
        FSDP shards (expensive!), IPC handles work directly with DTensor's
        local storage via dtensor._typed_storage() proxying to ._local_tensor.
        This bypasses the O(model_size) all_gather collective entirely.

        Args:
            policy_version: Version number for these weights
            generator_workers: ActorMesh of ForgeWorkerWrapper actors
            tp_size: Tensor parallel size of generator (default 1)

        Returns:
            Dict with push metadata (param_count, duration)
        """
        from torchstore.transport.cuda_ipc import create_ipc_handle

        logger.info(f"[IPC] Pushing weights directly to generators for v{policy_version} (TP={tp_size})")
        start_time = time.perf_counter()

        if len(self.engine.model_parts) != 1:
            raise RuntimeError("push_weights_ipc only supports single model part (no PP)")

        model = self.engine.model_parts[0]

        # Check if this is an FSDP model by looking at parallelism config
        is_fsdp = self.parallelism.data_parallel_shard_degree > 1

        # Get full state dict (handles FSDP gather if needed)
        handle_creation_start = time.perf_counter()

        if is_fsdp:
            # NOTE: For FSDP2, state_dict() returns DTensors (sharded, NOT gathered).
            # IPC handles work directly with the local shard via dtensor._local_tensor,
            # bypassing the expensive all_gather that baseline push_weights() requires.
            logger.info("[IPC] FSDP detected, using state_dict() (shards via DTensor)")
            sd = model.state_dict()
            flattened_state_dict, _ = flatten_state_dict(sd)

            sd_adapter = getattr(self.engine.checkpointer, 'sd_adapter', None)
            if sd_adapter is not None:
                hf_state_dict = sd_adapter.to_hf(flattened_state_dict)
            else:
                hf_state_dict = {
                    self._native_to_hf_name(name): param
                    for name, param in flattened_state_dict.items()
                }
        else:
            # For non-FSDP, access parameters directly
            hf_state_dict = {}
            for name, param in model.named_parameters():
                tensor = param.data
                if not tensor.is_cuda:
                    continue
                hf_name = self._native_to_hf_name(name)
                hf_state_dict[hf_name] = tensor

        # Build IPC handles - slice for TP if needed
        if tp_size > 1:
            # For TP>1, create per-rank sliced handles
            # Structure: {param_name: {tp_rank: handle}}
            ipc_handles_per_rank = {tp_rank: {} for tp_rank in range(tp_size)}
            sliced_tensors = {}  # Keep tensors alive

            for hf_name, tensor in hf_state_dict.items():
                if not tensor.is_cuda:
                    continue
                if not tensor.is_contiguous():
                    tensor = tensor.contiguous()

                # vLLM's weight loaders (QKVParallelLinear, RowParallelLinear, etc.)
                # expect full tensors and handle TP slicing internally. We should NOT
                # pre-slice tensors ourselves - send full tensors to all ranks.
                try:
                    handle = create_ipc_handle(tensor)
                    for tp_rank in range(tp_size):
                        ipc_handles_per_rank[tp_rank][hf_name] = handle
                except Exception as e:
                    logger.warning(f"[IPC] Failed to create handle for {hf_name}: {e}")

            # Keep tensors alive until transfer completes
            self._ipc_state_dict = hf_state_dict
            self._ipc_sliced_tensors = sliced_tensors

            handle_creation_time = time.perf_counter() - handle_creation_start
            logger.info(f"[IPC] Created {len(hf_state_dict)} params × {tp_size} TP ranks in {handle_creation_time:.3f}s")

            # Send handles to each worker with its TP rank
            send_start = time.perf_counter()
            await generator_workers.receive_weights_ipc_sliced.call(
                policy_version=policy_version,
                ipc_handles_per_rank=ipc_handles_per_rank,
            )
            send_time = time.perf_counter() - send_start
        else:
            # For TP=1, send full tensors (original behavior)
            ipc_handles = {}
            for hf_name, tensor in hf_state_dict.items():
                if not tensor.is_cuda:
                    continue
                if not tensor.is_contiguous():
                    tensor = tensor.contiguous()
                try:
                    handle = create_ipc_handle(tensor)
                    ipc_handles[hf_name] = handle
                except Exception as e:
                    logger.warning(f"[IPC] Failed to create handle for {hf_name}: {e}")

            self._ipc_state_dict = hf_state_dict

            handle_creation_time = time.perf_counter() - handle_creation_start
            logger.info(f"[IPC] Created {len(ipc_handles)} IPC handles in {handle_creation_time:.3f}s")

            send_start = time.perf_counter()
            await generator_workers.receive_weights_ipc.call(
                policy_version=policy_version,
                ipc_handles=ipc_handles,
            )
            send_time = time.perf_counter() - send_start

        # Clean up state dict references after transfer
        if hasattr(self, '_ipc_state_dict'):
            del self._ipc_state_dict
        if hasattr(self, '_ipc_sliced_tensors'):
            del self._ipc_sliced_tensors

        total_time = time.perf_counter() - start_time
        logger.info(
            f"[IPC] Push complete in {total_time:.2f}s "
            f"(handles: {handle_creation_time:.2f}s, send: {send_time:.2f}s)"
        )

        return {
            "param_count": len(hf_state_dict),
            "handle_creation_time": handle_creation_time,
            "send_time": send_time,
            "total_time": total_time,
            "is_fsdp": is_fsdp,
            "tp_size": tp_size,
        }

    def _get_tp_sharding_type(self, param_name: str) -> str:
        """Determine TP sharding type for a parameter based on its name."""
        column_parallel_patterns = [
            "q_proj", "k_proj", "v_proj", "qkv_proj",
            "gate_proj", "up_proj", "gate_up_proj",
        ]
        row_parallel_patterns = ["o_proj", "down_proj"]
        vocab_parallel_patterns = ["embed_tokens", "lm_head"]

        param_lower = param_name.lower()

        for pattern in column_parallel_patterns:
            if pattern in param_lower:
                return "column_parallel"
        for pattern in row_parallel_patterns:
            if pattern in param_lower:
                return "row_parallel"
        for pattern in vocab_parallel_patterns:
            if pattern in param_lower:
                return "vocab_parallel"

        return "replicated"

    def _slice_tensor_for_tp(
        self,
        tensor: torch.Tensor,
        shard_type: str,
        tp_rank: int,
        tp_size: int,
    ) -> torch.Tensor:
        """Slice a tensor for a specific TP rank."""
        if shard_type == "column_parallel":
            # Shard columns (dim=1)
            col_size = tensor.shape[1] // tp_size
            return tensor[:, tp_rank * col_size : (tp_rank + 1) * col_size]
        elif shard_type == "row_parallel":
            # Shard rows (dim=0)
            row_size = tensor.shape[0] // tp_size
            return tensor[tp_rank * row_size : (tp_rank + 1) * row_size, :]
        elif shard_type == "vocab_parallel":
            # Shard vocab dimension (dim=0)
            vocab_size = tensor.shape[0] // tp_size
            return tensor[tp_rank * vocab_size : (tp_rank + 1) * vocab_size, :]
        return tensor

    @endpoint
    async def push_weights_sharded(self, policy_version: int) -> dict:
        """Push FSDP shards directly to TorchStore without gathering.

        NOTE: This method is NOT currently used. It's reserved for future
        MULTI-NODE weight sync where CUDA IPC is not available.

        ┌─────────────────────────────────────────────────────────────────┐
        │  WHY THIS EXISTS (Multi-Node Weight Sync Without All-Gather)   │
        └─────────────────────────────────────────────────────────────────┘

        CUDA IPC (push_weights_ipc) only works on SINGLE NODE because IPC
        handles reference local GPU memory addresses. For multi-node:

        PROBLEM: How to sync weights without expensive all_gather?

        SOLUTION: Sharded push + slice-aware fetch

        ┌─────────────────────────────────────────────────────────────────┐
        │  TRAINER NODE 0          TRAINER NODE 1                        │
        │  ┌─────────────┐         ┌─────────────┐                       │
        │  │ GPU 0       │         │ GPU 2       │                       │
        │  │ shard_0     │         │ shard_1     │                       │
        │  └─────┬───────┘         └─────┬───────┘                       │
        │        │                       │                               │
        │        │ put_slice()           │ put_slice()                   │
        │        │ (with TensorSlice     │ (with TensorSlice             │
        │        │  metadata)            │  metadata)                    │
        │        ▼                       ▼                               │
        │  ┌─────────────────────────────────────────┐                   │
        │  │           TORCHSTORE (distributed)      │                   │
        │  │  key: "v1/layer.0.weight"               │                   │
        │  │  ├── slice[0]: shard_0, offsets=(0,0)   │                   │
        │  │  └── slice[1]: shard_1, offsets=(N/2,0) │                   │
        │  └─────────────────────────────────────────┘                   │
        │                       │                                        │
        │                       │ get_slice() or get_assembled()         │
        │                       ▼                                        │
        │  ┌─────────────────────────────────────────┐                   │
        │  │           GENERATOR NODE                │                   │
        │  │  Option A: Fetch all slices, assemble   │                   │
        │  │  Option B: Fetch only needed slice      │                   │
        │  │            (if generator also sharded)  │                   │
        │  └─────────────────────────────────────────┘                   │
        └─────────────────────────────────────────────────────────────────┘

        KEY INSIGHT: Each FSDP rank pushes ONLY its local shard with metadata
        about where it fits in the global tensor. No rank needs to see the
        full tensor, avoiding the O(model_size) all_gather collective.

        REQUIREMENTS TO COMPLETE THIS FEATURE:
        1. TorchStore needs get_assembled() or similar to reconstruct from slices
        2. Generator needs fetch_weights_sharded() to handle slice assembly
        3. For multi-node RDMA: TorchStore transport needs GPU-direct RDMA support

        ALTERNATIVE APPROACHES FOR MULTI-NODE:
        1. NCCL send/recv: Point-to-point GPU transfers (requires careful scheduling)
        2. GPU-Direct RDMA: InfiniBand/RoCE for direct GPU-to-GPU across nodes
        3. Checkpoint-based: Write shards to shared filesystem, generator reads

        Args:
            policy_version: Version number for these weights.

        Returns:
            Dict with metadata about pushed shards (param_count, shapes).
        """
        from torch.distributed.tensor import DTensor
        from torch.distributed.tensor._utils import _compute_local_shape_and_global_offset
        from torchstore.transport.types import TensorSlice

        logger.info(f"Pushing sharded weights for policy version {policy_version}")
        start_time = time.perf_counter()

        # Get FSDP mesh info from parallel_dims
        dp_rank = self.engine.dp_rank
        dp_world_size = self.engine.parallel_dims.dp_shard

        import asyncio

        param_metadata = {}
        put_tasks = []  # Collect all put operations for batch execution

        # Iterate through model parts (usually just one for non-PP)
        for model_part in self.engine.model_parts:
            for fqn, param in model_part.named_parameters():
                # Check if parameter is a DTensor (sharded by FSDP)
                if isinstance(param, DTensor):
                    # Get local shard WITHOUT triggering all_gather
                    local_shard = param._local_tensor

                    # Compute global offsets from DTensor placement info
                    coordinates = param.device_mesh.get_coordinate()
                    _, offsets = _compute_local_shape_and_global_offset(
                        param.shape,  # Global shape
                        mesh_shape=param.device_mesh.shape,
                        my_coordinate=coordinates,
                        placements=param.placements,
                    )

                    # Create TensorSlice metadata
                    tensor_slice = TensorSlice(
                        offsets=tuple(offsets),
                        coordinates=tuple(coordinates) if coordinates else (dp_rank,),
                        global_shape=tuple(param.shape),
                        local_shape=tuple(local_shard.shape),
                        mesh_shape=tuple(param.device_mesh.shape) if param.device_mesh else (dp_world_size,),
                    )

                    # Convert to HF-style naming if adapter available
                    # For sharded push, we store native names and let consumer convert
                    hf_name = self._native_to_hf_name(fqn)

                    key = get_param_key(policy_version, hf_name)
                    put_tasks.append((key, local_shard, tensor_slice, True))  # True = slice

                    param_metadata[hf_name] = {
                        "global_shape": tuple(param.shape),
                        "local_shape": tuple(local_shard.shape),
                        "offsets": tuple(offsets),
                        "coordinates": tuple(coordinates) if coordinates else (dp_rank,),
                    }
                else:
                    # Non-DTensor parameter (shouldn't happen with FSDP, but handle it)
                    hf_name = self._native_to_hf_name(fqn)
                    key = get_param_key(policy_version, hf_name)
                    put_tasks.append((key, param, None, False))  # False = regular put

                    param_metadata[hf_name] = {
                        "global_shape": tuple(param.shape),
                        "local_shape": tuple(param.shape),
                        "offsets": (0,) * len(param.shape),
                        "coordinates": (0,),
                    }

        # Execute puts in parallel batches
        batch_size = 100
        pushed_count = len(put_tasks)

        for batch_start in range(0, pushed_count, batch_size):
            batch_end = min(batch_start + batch_size, pushed_count)
            batch = put_tasks[batch_start:batch_end]

            async def do_put(key, tensor, tensor_slice, is_slice):
                if is_slice:
                    await ts.put_slice(key, tensor, tensor_slice)
                else:
                    await ts.put(key, tensor)

            await asyncio.gather(*[do_put(k, t, s, is_s) for k, t, s, is_s in batch])

            if batch_start % 500 == 0:
                logger.info(f"[Rank {dp_rank}] Push progress: {batch_end}/{pushed_count} params")

        # Barrier to ensure all ranks have finished pushing
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        end_time = time.perf_counter()
        logger.info(
            f"[Rank {dp_rank}] Pushed {pushed_count} shards for "
            f"policy version {policy_version} in {end_time - start_time:.2f}s"
        )

        return {
            "param_count": pushed_count,
            "dp_rank": dp_rank,
            "dp_world_size": dp_world_size,
            "metadata": param_metadata,
        }

    def _native_to_hf_name(self, native_fqn: str) -> str:
        """Convert native TorchTitan parameter name to HuggingFace format.

        This is a simplified version that handles common patterns.
        For full conversion, use the sd_adapter.to_hf() method on gathered tensors.
        """
        # Common mappings for Llama-style models
        mappings = {
            "tok_embeddings.weight": "model.embed_tokens.weight",
            "norm.weight": "model.norm.weight",
            "output.weight": "lm_head.weight",
        }

        # Direct match
        if native_fqn in mappings:
            return mappings[native_fqn]

        # Layer-based mappings
        import re
        layer_mappings = {
            r"layers\.(\d+)\.attention\.wq\.weight": r"model.layers.\1.self_attn.q_proj.weight",
            r"layers\.(\d+)\.attention\.wk\.weight": r"model.layers.\1.self_attn.k_proj.weight",
            r"layers\.(\d+)\.attention\.wv\.weight": r"model.layers.\1.self_attn.v_proj.weight",
            r"layers\.(\d+)\.attention\.wo\.weight": r"model.layers.\1.self_attn.o_proj.weight",
            r"layers\.(\d+)\.attention_norm\.weight": r"model.layers.\1.input_layernorm.weight",
            r"layers\.(\d+)\.ffn_norm\.weight": r"model.layers.\1.post_attention_layernorm.weight",
            # MoE patterns for Llama4
            r"layers\.(\d+)\.moe\.router\.gate\.weight": r"model.layers.\1.feed_forward.router.weight",
            r"layers\.(\d+)\.moe\.shared_experts\.w1\.weight": r"model.layers.\1.feed_forward.shared_expert.gate_proj.weight",
            r"layers\.(\d+)\.moe\.shared_experts\.w2\.weight": r"model.layers.\1.feed_forward.shared_expert.down_proj.weight",
            r"layers\.(\d+)\.moe\.shared_experts\.w3\.weight": r"model.layers.\1.feed_forward.shared_expert.up_proj.weight",
            r"layers\.(\d+)\.moe\.experts\.w1": r"model.layers.\1.feed_forward.experts.gate_proj",
            r"layers\.(\d+)\.moe\.experts\.w2": r"model.layers.\1.feed_forward.experts.down_proj",
            r"layers\.(\d+)\.moe\.experts\.w3": r"model.layers.\1.feed_forward.experts.up_proj",
            # Standard MLP patterns (for non-MoE layers)
            r"layers\.(\d+)\.feed_forward\.w1\.weight": r"model.layers.\1.mlp.gate_proj.weight",
            r"layers\.(\d+)\.feed_forward\.w2\.weight": r"model.layers.\1.mlp.down_proj.weight",
            r"layers\.(\d+)\.feed_forward\.w3\.weight": r"model.layers.\1.mlp.up_proj.weight",
        }

        for pattern, replacement in layer_mappings.items():
            if re.match(pattern, native_fqn):
                return re.sub(pattern, replacement, native_fqn)

        # If no mapping found, return as-is (prefixed with model.)
        if not native_fqn.startswith("model."):
            return f"model.{native_fqn}"
        return native_fqn

    @endpoint
    async def get_param_shapes(self) -> dict:
        """Get global shapes of all parameters for generator to compute TP slices.

        Returns:
            Dict mapping HF param names to their global shapes.
        """
        param_shapes = {}
        for model_part in self.engine.model_parts:
            for fqn, param in model_part.named_parameters():
                hf_name = self._native_to_hf_name(fqn)
                # For DTensors, param.shape gives global shape
                param_shapes[hf_name] = tuple(param.shape)
        return param_shapes

    @endpoint
    async def add_noise_to_weights(self, noise_scale: float = 0.01) -> dict:
        """Add noise to model weights for testing weight sync.

        This is a test utility to verify that weight updates are working correctly.
        After calling this, the trainer's weights will be different from the
        generator's weights until sync is performed.

        Args:
            noise_scale: Standard deviation of Gaussian noise to add

        Returns:
            Dict with statistics about the noise added
        """
        import torch

        logger.info(f"[TitanTrainer] Adding noise (scale={noise_scale}) to weights for testing")

        num_params = 0
        total_noise = 0.0

        for model_part in self.engine.model_parts:
            for name, param in model_part.named_parameters():
                with torch.no_grad():
                    noise = torch.randn_like(param.data) * noise_scale
                    param.data.add_(noise)
                    num_params += 1
                    total_noise += noise.abs().mean().item()

        logger.info(f"[TitanTrainer] Added noise to {num_params} parameters")
        return {
            "num_params": num_params,
            "avg_noise": total_noise / num_params if num_params > 0 else 0,
            "noise_scale": noise_scale,
        }

    @endpoint
    async def cleanup(self) -> None:
        if self.engine.checkpointer:
            self.engine.checkpointer.close()
