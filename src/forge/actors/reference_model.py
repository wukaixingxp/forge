# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import logging
import math
import os

from collections.abc import Mapping
from dataclasses import dataclass, field, fields

import torch

from forge.controller import ForgeActor
from forge.observability.metrics import record_metric, Reduce
from forge.observability.perf_tracker import Tracer
from forge.util.ops import compute_logprobs
from monarch.actor import current_rank, current_size, endpoint
from torch.distributed.tensor import DTensor

from torchtitan.config.job_config import (
    Checkpoint,
    Comm,
    Compile,
    Model,
    Parallelism,
    Training,
)
from torchtitan.experiments.forge.engine import ForgeEngine
from torchtitan.experiments.forge.job_config import ForgeJobConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class ReferenceModel(ForgeActor):
    """
    A reference model actor for reinforcement learning (RL) training.

    Based on TorchTitan's engine architecture, this actor provides a
    frozen model that only runs forward passes without gradient
    computation. It is typically used to maintain algorithmic
    consistency in policy optimization methods such as GRPO
    (Group Relative Policy Optimization) or PPO (Proximal Policy
    Optimization), where it serves as a fixed reference point to
    compute KL divergence penalties against the training policy.

    The reference model is loaded from a checkpoint and runs in
    evaluation mode with inference_mode enabled to optimize memory and
    compute efficiency.

    Attributes:

        model (Model): Model configuration (architecture, vocab size,
        etc.)
        parallelism (Parallelism): Parallelism strategy configuration
        (TP, PP, CP, DP)
        checkpoint (Checkpoint): Checkpoint loading configuration
        compile (Compile): Torch compilation settings
        comm (Comm): Communication backend configuration
        training (Training): Training-related settings (dtype, garbage
        collection, etc.)
    """

    # Refer to titan JobConfig for enabling more ForgeEngine configuration
    model: Model = field(default_factory=Model)
    parallelism: Parallelism = field(default_factory=Parallelism)
    checkpoint: Checkpoint = field(default_factory=Checkpoint)
    compile: Compile = field(default_factory=Compile)
    comm: Comm = field(default_factory=Comm)
    training: Training = field(
        default_factory=Training
    )  # Needed in order to set attrs like dtype, garbage collection freq, etc.

    # Populated in setup
    # TODO: Commented out since engine_config parsing extracts from class members
    # engine: ForgeEngine | None = None

    def __post_init__(self):
        """Initializes config types and env variables."""
        super().__init__()

        # Instantiate dict fields
        for f in fields(self):
            attr = getattr(self, f.name)
            if isinstance(attr, Mapping):
                setattr(self, f.name, f.type(**attr))
            elif not isinstance(attr, f.type):
                raise TypeError(
                    f"{f.name} should be a {f.type} type or a dict like object"
                )

        self.step = 0
        self.rank = current_rank().rank
        self.size = math.prod(current_size().values())

        env = {
            "RANK": str(self.rank),
            "LOCAL_RANK": str(self.rank),
            "LOCAL_WORLD_SIZE": str(self.size),
            "GROUP_RANK": str(self.size),
            "GROUP_WORLD_SIZE": str(self.size),
            "ROLE_RANK": str(self.rank),
            "ROLE_WORLD_SIZE": str(self.size),
            "ROLE_NAME": "rank",
            "WORLD_SIZE": str(self.size),
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
        os.environ.update(env)

    @endpoint
    async def setup(self):
        engine_config = {f.name: getattr(self, f.name) for f in fields(self)}
        engine_config = ForgeJobConfig(**engine_config)
        engine_config.checkpoint.folder = (
            ""  # hardcode to empty to force load from initial_load_path
        )
        self.engine = ForgeEngine(engine_config)
        self.engine.checkpointer.load()
        self.model = self.engine.model_parts[0]  # No pipeline parallelism yet

        # Reinitialize freqs_cis after checkpoint loading to ensure correct dimensions
        # This is necessary because HF checkpoints may have incompatible freqs_cis dimensions
        if hasattr(self.model, "freqs_cis"):
            old_freqs_cis_len = self.model.freqs_cis.shape[0]
            logger.info(
                f"Original freqs_cis length from checkpoint: {old_freqs_cis_len}"
            )

            # Update model_args.max_seq_len to match the training sequence length
            # This ensures precomputed freqs_cis has sufficient length
            if hasattr(self.model, "model_args") and hasattr(
                self.model.model_args, "max_seq_len"
            ):
                # Use the sequence length from training config, or default to current max_seq_len
                target_seq_len = getattr(
                    self.training, "seq_len", self.model.model_args.max_seq_len
                )
                old_max_seq_len = self.model.model_args.max_seq_len
                logger.info(
                    f"Updating max_seq_len from {old_max_seq_len} to {target_seq_len}"
                )
                self.model.model_args.max_seq_len = target_seq_len
                self.model.freqs_cis = self.model._precompute_freqs_cis()
                self.model.freqs_cis = self.model.freqs_cis.to(
                    next(self.model.parameters()).device
                )
                new_freqs_cis_len = self.model.freqs_cis.shape[0]
                logger.info(
                    f"New freqs_cis length after recompute: {new_freqs_cis_len}"
                )
            else:
                logger.warning(
                    "Model does not have model_args or max_seq_len attribute, cannot update freqs_cis"
                )

        self.model.eval()

    @endpoint
    async def forward(
        self, input_ids: torch.Tensor, max_req_tokens: int, return_logprobs: bool
    ) -> torch.Tensor:
        """
        Args:
            input_ids (torch.Tensor): input token ids with shape [group_size, req + res length].
            max_req_tokens (int): maximum request length.
            return_logprobs (bool): whether to return og probabilities instead of raw logits.

            return_logprobs flag significantly impacts the amount of data transferred to the caller:
            - When False: Returns logits with shape [group_size, req + res_length, vocab_size].
              This includes the full vocabulary distribution for each token position.

            - When True: Returns log probabilities with shape [group_size, req_length].
              This only includes probabilities for the request tokens, significantly reducing memory
              usage and transfer overhead.
        """
        # Record reference model metrics
        record_metric("reference_perf/forward/count_forward_passes", 1, Reduce.SUM)
        record_metric(
            "reference_perf/forward/avg_sequence_length",
            input_ids.shape[1],
            Reduce.MEAN,
        )

        t = Tracer("reference_perf/forward", timer="gpu", track_memory=True)
        t.start()
        self.engine.gc_handler.run(self.step)
        t.step("garbage_collection")

        model_parts = self.engine.model_parts
        parallel_dims = self.engine.parallel_dims
        input_ids = input_ids.to("cuda")
        t.step("to_device")
        # optional_context_parallel_ctx = (
        #     dist_utils.create_context_parallel_ctx(
        #         cp_mesh=parallel_dims.world_mesh["cp"],
        #         cp_buffers=[inputs, labels] + [m.freqs_cis for m in model_parts],
        #         cp_seq_dims=[1, 1] + [0 for _ in model_parts],
        #         cp_no_restore_buffers={inputs, labels},
        #         cp_rotate_method=self.job_config.parallelism.context_parallel_rotate_method,
        #     )
        #     if parallel_dims.cp_enabled
        #     else None
        # )
        optional_context_parallel_ctx = None
        if self.engine.parallel_dims.pp_enabled:
            raise NotImplementedError("PP not implemented yet")
        else:
            # (jackkhuu) Not sure if either context are needed for inference here
            with self.engine.train_context(optional_context_parallel_ctx):
                with self.engine.maybe_enable_amp:
                    with torch.inference_mode():
                        logits = self.model(input_ids)
        self.step += 1
        if isinstance(logits, DTensor):
            logits = logits.full_tensor()
        t.step("forward")

        if not return_logprobs:
            t.stop()
            return logits
        else:
            logprobs = compute_logprobs(logits, input_ids[:, max_req_tokens:])
            t.step("compute_logprobs")
            t.stop()
            return logprobs
