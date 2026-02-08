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
from typing import Callable, Literal, get_origin, get_args, get_origin, get_args, Union

import torch
from forge.actors.hybrid.inference_engine import InferenceConfig, InferenceEngine
from forge.api.trainer import ParallelismConfig, TrainerConfig, TrainerStatus
from forge.controller import ForgeActor
from forge.data.utils import batch_to_device
from forge.data_models.completion import Completion
from forge.observability.metrics import record_metric, Reduce
from forge.observability.perf_tracker import Tracer
from forge.rl.loss import create_shifted_targets
from forge.types import TrainBatch
from monarch.actor import endpoint
from torch import Tensor
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
from transformers import AutoTokenizer
from vllm.sampling_params import SamplingParams

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@dataclass
class HybridPolicyActor(ForgeActor):
    """Hybrid actor that combines training and inference in a single model instance.

    This actor eliminates weight synchronization overhead by maintaining a single
    model in GPU memory and switching between training and inference modes in-place.
    The model stays in GPU memory throughout the RL loop—no weight copies, no
    TorchStore round trips, no generation pauses.

    Key Innovation: Mode-switched execution
    - Training Mode: ForgeEngine with FSDP (gradients enabled, optimizer active)
    - Inference Mode: Custom lightweight inference (gradients disabled, basic generation)

    Mode switch overhead: ~10-50ms (just metadata changes)
    vs. Current sync overhead: 1-3 seconds (weight serialization, network, deserialization)
    Expected speedup: 20-100x reduction in sync overhead

    Architecture:
        Single Model in GPU Memory
        ├── Training Mode: ForgeEngine with FSDP (gradients enabled, optimizer active)
        └── Inference Mode: InferenceEngine wrapper (gradients disabled, autoregressive generation)

    Args:
        Training configuration (from TitanTrainer):
            job: Job configuration
            model: Model configuration
            optimizer: Optimizer configuration
            lr_scheduler: Learning rate scheduler configuration
            training: Training configuration
            parallelism: Parallelism configuration (FSDP, TP, etc.)
            checkpoint: Checkpoint configuration
            activation_checkpoint: Activation checkpointing configuration
            compile: Compilation configuration
            quantize: Quantization configuration
            comm: Communication configuration
            memory_estimation: Memory estimation configuration

        Inference configuration (new):
            inference: InferenceConfig for optimization features
            sampling_params: Default SamplingParams for generation

        Internal state:
            loss: Loss function for training
            mode: Current execution mode ('train' or 'infer')
            engine: ForgeEngine for training
            inference_engine: InferenceEngine for generation
    """

    # Training config (from TitanTrainer)
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

    # Inference config (new)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    sampling_params: SamplingParams | Mapping = field(default_factory=dict)

    # Non JobConfig-related fields
    loss: Callable = lambda logits, **targets: logits
    state_dict_key: str = "model_state_dict"

    def __post_init__(self):
        super().__init__()

        # Initialize config fields from dicts
        for f in fields(self):
            attr = getattr(self, f.name)
            if isinstance(attr, Mapping) and f.type != Mapping:
                # Handle Union types (e.g., SamplingParams | Mapping)
                field_type = f.type
                origin = get_origin(field_type)

                # If it's a Union type, extract the non-Mapping type
                if origin is Union or hasattr(field_type, '__or__'):  # Handle both typing.Union and |
                    args = get_args(field_type) if origin is Union else []
                    if not args:  # Python 3.10+ uses types.UnionType for |
                        import types
                        if isinstance(field_type, types.UnionType):
                            args = get_args(field_type)

                    # Find the first non-Mapping type in the Union
                    for arg in args:
                        if arg != Mapping and not (get_origin(arg) == type and issubclass(get_origin(arg) or arg, Mapping)):
                            field_type = arg
                            break

                setattr(self, f.name, field_type(**attr))
            elif not isinstance(attr, f.type) and f.type != Mapping:
                # For Union types, check if attr is an instance of any of the union members
                origin = get_origin(f.type)
                if origin is Union or hasattr(f.type, '__or__'):
                    args = get_args(f.type) if origin is Union else []
                    if not args:  # Python 3.10+ uses types.UnionType
                        import types
                        if isinstance(f.type, types.UnionType):
                            args = get_args(f.type)

                    if not any(isinstance(attr, arg) for arg in args if arg != type(None)):
                        raise TypeError(
                            f"{f.name} should be one of {args} or a dict like object"
                        )
                else:
                    raise TypeError(
                        f"{f.name} should be a {f.type} type or a dict like object"
                    )

        # Initialize sampling params
        if isinstance(self.sampling_params, Mapping):
            self.sampling_params = SamplingParams.from_optional(**self.sampling_params)

        self.step = 1
        self.num_training_steps = self.training.steps
        self.gradient_accumulation_steps = 1
        self._accumulated_microbatches = 0
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        # Initialize mode
        self.mode: Literal["train", "infer"] = "train"
        self.engine: ForgeEngine = None
        self.inference_engine: InferenceEngine = None
        self.tokenizer = None

        logger.info("Compiling loss function")
        self.loss = torch.compile(self.loss)

    @endpoint
    async def setup(self):
        """Initialize ForgeEngine and InferenceEngine.

        Sets up:
        1. ForgeEngine for training (with FSDP support)
        2. InferenceEngine wrapper around the same model
        3. Tokenizer for inference
        """
        # Setup ForgeEngine (same as TitanTrainer)
        engine_config = {f.name: getattr(self, f.name) for f in fields(self)}
        for key in {
            "loss",
            "state_dict_key",
            "inference",
            "sampling_params",
        }:
            if key in engine_config:
                engine_config.pop(key)  # Not part of job config

        self.engine = ForgeEngine(ForgeJobConfig(**engine_config))
        self.engine.checkpointer.load(step=self.step)
        self.engine.optimizers.zero_grad()

        # Setup tokenizer for inference
        # Get tokenizer from model HF path
        hf_model_path = self.model.hf_assets_path
        if hf_model_path.startswith("hf://"):
            hf_model_path = hf_model_path[5:]  # Remove hf:// prefix

        self.tokenizer = AutoTokenizer.from_pretrained(hf_model_path)
        logger.info(f"Loaded tokenizer from {hf_model_path}")

        # Setup InferenceEngine (wraps the same model used by ForgeEngine)
        # Note: We use model_parts[0] because we don't support pipeline parallelism yet
        assert len(self.engine.model_parts) == 1, "Pipeline parallelism not supported"

        # Choose inference engine based on config
        if self.inference.use_torchtitan_vllm:
            # Try to use TorchTitan model with vLLM paged attention (single model copy)
            # Fall back to simple vLLM if experimental wrapper not available
            try:
                from forge.actors.hybrid.torchtitan_vllm_engine import (
                    TorchTitanVLLMEngine,
                    TorchTitanVLLMConfig,
                )

                vllm_config = TorchTitanVLLMConfig(
                    tensor_parallel_size=self.parallelism.tensor_parallel_degree,
                    enable_cuda_graphs=self.inference.enable_cuda_graphs,
                    max_num_seqs=self.inference.max_batch_size,
                )

                # Extract model name from config (e.g., "qwen3" -> "Qwen3")
                model_name = self.model.name.capitalize()

                self.inference_engine = TorchTitanVLLMEngine(
                    model_name=model_name,
                    config=vllm_config,
                )
                logger.info(
                    f"Using TorchTitan + vLLM wrapper for inference "
                    f"(single model copy, 50-100x speedup expected)"
                )
            except (ImportError, Exception) as e:
                logger.warning(
                    f"TorchTitan vLLM wrapper not available ({e}), "
                    f"falling back to simple vLLM engine (uses separate model copy)"
                )
                from forge.actors.hybrid.simple_vllm_adapter import (
                    SimpleVLLMEngine,
                    SimpleVLLMConfig,
                )

                # Get HF model path for vLLM
                hf_model_path = self.model.hf_assets_path
                if hf_model_path.startswith("hf://"):
                    hf_model_path = hf_model_path[5:]

                simple_config = SimpleVLLMConfig(
                    model_name=hf_model_path,
                    tensor_parallel_size=self.parallelism.tensor_parallel_degree,
                    enable_cuda_graphs=self.inference.enable_cuda_graphs,
                    max_num_seqs=self.inference.max_batch_size,
                )

                self.inference_engine = SimpleVLLMEngine(config=simple_config)
                logger.info(
                    f"Using simple vLLM engine for inference "
                    f"(separate model copy, 50-100x speedup expected)"
                )
        elif self.inference.use_nano_vllm:
            # Use nano-vLLM for high-performance inference (separate model)
            from forge.actors.hybrid.nano_vllm_engine import NanoVLLMEngine, NanoVLLMConfig

            nano_config = NanoVLLMConfig(
                tensor_parallel_size=self.inference.nano_vllm_tensor_parallel_size,
                enable_cuda_graphs=self.inference.enable_cuda_graphs,
                max_num_seqs=self.inference.max_batch_size,
            )

            self.inference_engine = NanoVLLMEngine(
                model_path=hf_model_path,
                config=nano_config,
            )
            logger.info("Using nano-vLLM for inference (50-100x speedup expected, dual model)")
        elif self.inference.use_simple_kv_cache:
            # Use simple KV cache (nano-vLLM style, single model copy)
            from forge.actors.hybrid.simple_kv_cache_engine import SimpleKVCacheEngine

            self.inference_engine = SimpleKVCacheEngine(
                model=self.engine.model_parts[0],
                tokenizer=self.tokenizer,
                num_blocks=self.inference.simple_kv_cache_num_blocks,
                block_size=self.inference.simple_kv_cache_block_size,
                max_model_len=2048,
            )
            logger.info(
                f"Using simple KV cache (nano-style, single model copy, 10-20x speedup expected)"
            )
        else:
            # Use custom InferenceEngine (fallback)
            self.inference_engine = InferenceEngine(
                model=self.engine.model_parts[0],
                tokenizer=self.tokenizer,
                device=self.engine.device,
                config=self.inference,
                engine=self.engine,  # Pass engine for FSDP-aware contexts
            )

            # Phase 2: Warmup CUDA graphs if enabled
            if self.inference.enable_cuda_graphs:
                logger.info("Warming up CUDA graphs...")
                self.inference_engine.warmup_cuda_graphs()

            logger.info("Using custom InferenceEngine (no KV cache acceleration)")

        logger.info(
            f"HybridPolicyActor initialized in '{self.mode}' mode "
            f"(FSDP={self.parallelism.data_parallel_shard_degree})"
        )

    async def switch_mode(self, mode: Literal["train", "infer"]):
        """Switch between training and inference modes.

        Mode switching is fast (~10-50ms) because it only changes execution flags,
        not weights. Parameters stay in GPU memory.

        Args:
            mode: Target mode ('train' or 'infer')
        """
        if self.mode == mode:
            return  # Already in target mode

        switch_start = time.perf_counter()

        if mode == "infer":
            # Switch to inference mode
            torch.set_grad_enabled(False)
            if not self.inference.use_nano_vllm:
                # Only need to switch model mode if using custom InferenceEngine
                # nano-vLLM uses a separate model instance
                self.engine.model_parts[0].eval()
            logger.debug("Switched to inference mode")
        else:  # mode == "train"
            # Switch to training mode
            torch.set_grad_enabled(True)
            if not self.inference.use_nano_vllm:
                self.engine.model_parts[0].train()
            self.inference_engine.clear_cache()  # Free KV cache memory
            logger.debug("Switched to training mode")

        self.mode = mode
        switch_duration = time.perf_counter() - switch_start

        record_metric(
            f"hybrid_policy/mode_switch/{mode}_duration_ms",
            switch_duration * 1000,
            Reduce.MEAN,
        )

    @endpoint
    async def generate(
        self,
        prompt: str,
        *,
        priority: int = 0,
        sampling_params: SamplingParams | None = None,
    ) -> list[Completion]:
        """Generate completions for a given prompt.

        Automatically switches to inference mode if needed.

        Args:
            prompt: Input text prompt
            priority: Not used (for API compatibility with Generator)
            sampling_params: Sampling parameters (or use defaults)

        Returns:
            List of Completion objects
        """
        logger.info(f"[HYBRID] generate() called, prompt length={len(prompt)}")
        t = Tracer("hybrid_policy_perf/generate", timer="gpu")
        t.start()

        # Switch to inference mode
        logger.info(f"[HYBRID] Switching to infer mode...")
        await self.switch_mode("infer")
        logger.info(f"[HYBRID] Switched to infer mode")
        t.step("switch_to_infer")

        # Generate using InferenceEngine
        # All ranks must call generate() to participate in FSDP forward passes
        # Token sampling is synchronized via broadcast in inference engine
        params = sampling_params or self.sampling_params
        logger.info(f"[HYBRID] Calling inference_engine.generate()...")
        completions = await self.inference_engine.generate(prompt, params)
        logger.info(f"[HYBRID] Got {len(completions)} completions")

        record_metric("hybrid_policy/generate/count_requests", 1, Reduce.SUM)
        record_metric(
            "hybrid_policy/generate/count_sequences_completed",
            len(completions),
            Reduce.SUM,
        )

        t.stop()
        logger.info(f"[HYBRID] generate() returning")
        return completions

    def forward_backward(self, batch: TrainBatch) -> Tensor:
        """Run forward and backward pass (same as TitanTrainer)."""
        model_parts = self.engine.model_parts
        parallel_dims = self.engine.parallel_dims
        optional_context_parallel_ctx = None

        # Create shifted target_ids for next-token prediction
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

                # Free before bwd to avoid peaking memory
                del logits, loss_output.metrics
                loss.backward()

        self._accumulated_microbatches += 1
        return loss

    @endpoint
    async def train_step(self, batches: list[TrainBatch]) -> float:
        """Run a training step.

        Automatically switches to training mode if needed.

        Args:
            batches: List of batches (one per DP rank)

        Returns:
            Loss value
        """
        t = Tracer("hybrid_policy_perf/train_step", timer="gpu", track_memory=True)
        t.start()

        # Switch to training mode
        await self.switch_mode("train")
        t.step("switch_to_train")

        self.engine.gc_handler.run(self.step)
        batch = batches[self.engine.dp_rank]
        batch_to_device(batch.model_inputs, self.engine.device)
        batch_to_device(batch.loss_inputs, self.engine.device)

        loss = self.forward_backward(batch)
        torch.distributed.all_reduce(loss)

        t.step("forward_backward")

        current_lr = self.engine.lr_schedulers.schedulers[0].get_last_lr()[0]
        record_metric("hybrid_policy/learning_rate", current_lr, Reduce.MIN)

        self.engine.optimizers.step()
        self.engine.optimizers.zero_grad()
        self.engine.lr_schedulers.step()
        self._accumulated_microbatches = 0
        self.step += 1
        t.step("optimizer_step")

        loss = loss.detach().item()
        record_metric("hybrid_policy/loss", loss, Reduce.MEAN)

        self.engine.checkpointer.save(
            curr_step=self.step,
            last_step=self.step == self.num_training_steps,
        )
        t.step("save_checkpoint")
        t.stop()
        return loss

    @endpoint
    async def get_config(self) -> TrainerConfig:
        """Get static trainer and model configuration."""
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
        """Get current runtime status of the actor."""
        return TrainerStatus(
            step=self.step,
            accumulated_microbatches=self._accumulated_microbatches,
        )

    @endpoint
    async def clear_gradients(self) -> None:
        """Clear accumulated gradients without applying them."""
        self.engine.optimizers.zero_grad()
        self._accumulated_microbatches = 0

    @endpoint
    async def save(
        self,
        name: str | None = None,
        path: str | None = None,
        weights_only: bool = False,
    ) -> str:
        """Save trainer state to persistent storage."""
        if name is not None:
            raise NotImplementedError(
                "HybridPolicyActor uses step-based checkpoint naming"
            )
        if path is not None:
            raise NotImplementedError(
                "HybridPolicyActor uses checkpoint.folder from config"
            )
        if weights_only:
            raise NotImplementedError(
                "weights_only not supported; always saves full training state"
            )

        self.engine.checkpointer.save(
            curr_step=self.step,
            last_step=False,
        )
        return f"{self.checkpoint.folder}/step-{self.step}"

    @endpoint
    async def load(self, path: str | None = None) -> str:
        """Load a previously saved checkpoint."""
        if path is not None:
            raise NotImplementedError(
                "HybridPolicyActor uses checkpoint.folder from config"
            )

        self.engine.checkpointer.load(step=self.step)
        return f"{self.checkpoint.folder}/step-{self.step}"

    @endpoint
    async def push_weights(self, policy_version: int) -> None:
        """No-op: weights are already in the actor.

        In hybrid mode, there's no need to push weights because training
        and inference share the same model instance in GPU memory.
        """
        logger.debug(
            f"push_weights({policy_version}) is a no-op in hybrid mode "
            "(weights already shared between train/infer)"
        )

    @endpoint
    async def update_weights(self, policy_version: int) -> None:
        """No-op: weights are already updated.

        In hybrid mode, there's no need to update weights because training
        and inference share the same model instance in GPU memory.
        """
        logger.debug(
            f"update_weights({policy_version}) is a no-op in hybrid mode "
            "(weights already shared between train/infer)"
        )

    @endpoint
    async def get_inference_stats(self) -> dict:
        """Get statistics from inference engine optimizations.

        Returns:
            Dict with statistics from prefix cache, KV cache, and CUDA graphs
        """
        if self.inference_engine is None:
            return {
                "prefix_cache": None,
                "kv_cache": None,
                "cuda_graphs": None,
            }

        return self.inference_engine.get_stats()

    @endpoint
    async def cleanup(self) -> None:
        """Cleanup resources."""
        if self.engine.checkpointer:
            self.engine.checkpointer.close()
