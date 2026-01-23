# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Generic OpenEnv GRPO Training Script using GenericEnvClient.

This version uses GenericEnvClient and GenericAction to work with ANY
OpenEnv environment without requiring environment-specific packages.

Usage: python -m apps.openenv.main_generic --config apps/openenv/llama3_8b_julia_generic.yaml
"""

from __future__ import annotations

import asyncio
import importlib
import logging
import os
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, TYPE_CHECKING

# Type-only imports to avoid runtime import of openenv (which pulls in fastmcp/docket
# and conflicts with monarch's OpenTelemetry meter provider)
if TYPE_CHECKING:
    from openenv import GenericAction
    from openenv.core.client_types import StepResult

# CRITICAL: Add openenv directory to sys.path at module level
_appdir = Path(__file__).parent
if str(_appdir) not in sys.path:
    sys.path.insert(0, str(_appdir))

import torch
import torch.nn.functional as F
import torchstore as ts
import yaml
from datasets import load_dataset
from forge.actors.generator import Generator
from forge.actors.generic_openenv_client import GenericOpenEnvClientActor
from forge.actors.reference_model import ReferenceModel
from forge.actors.replay_buffer import ReplayBuffer
from forge.actors.trainer import RLTrainer
from forge.controller.actor import ForgeActor
from forge.controller.provisioner import init_provisioner, shutdown
from forge.data_models.completion import Completion
from forge.observability.metric_actors import get_or_create_metric_logger
from forge.observability.metrics import record_metric, Reduce
from forge.observability.perf_tracker import Tracer
from forge.types import LauncherConfig, ProvisionerConfig
from forge.util.checkpoint import drop_weights
from forge.util.config import parse
from forge.util.ops import compute_logprobs
from monarch.actor import endpoint
from omegaconf import DictConfig, ListConfig, OmegaConf
from vllm.transformers_utils.tokenizer import get_tokenizer


# Set up module logger
logger = logging.getLogger(__name__)
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="[%(levelname)s %(name)s] %(message)s",
)


@dataclass
class Episode:
    episode_id: str
    pad_id: int
    request_len: int
    response_len: int
    target: Any | None = None
    completion: Completion | None = None
    ref_logprobs: torch.Tensor | None = None
    reward: float | None = None
    advantage: float | None = None

    @property
    def policy_version(self) -> int | None:
        return self.completion.generator_version if self.completion else None

    @property
    def request_tensor(self) -> torch.Tensor:
        tensor: torch.Tensor = self.completion.prompt_ids.to(torch.long)
        if tensor.shape[0] < self.request_len:
            diff = self.request_len - tensor.shape[0]
            tensor = F.pad(tensor, (diff, 0), value=self.pad_id)
        return tensor

    @property
    def response_tensor(self) -> torch.Tensor:
        tensor: torch.Tensor = self.completion.token_ids.to(torch.long)
        if tensor.shape[0] < self.response_len:
            diff = self.response_len - tensor.shape[0]
            tensor = F.pad(tensor, (0, diff), value=self.pad_id)
        return tensor


Group = list[Episode]
Policy = Generator


def load_function_from_string(func_ref: str) -> Callable:
    """Load a function from a string reference like 'module.function_name'."""
    openenv_dir = Path(__file__).parent
    if str(openenv_dir) not in sys.path:
        sys.path.insert(0, str(openenv_dir))

    module_name, func_name = func_ref.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, func_name)


def function_constructor(loader, node):
    """YAML constructor for !function tag."""
    value = loader.construct_scalar(node)
    return ("!function", value)


yaml.add_constructor("!function", function_constructor, Loader=yaml.SafeLoader)


def collate(
    batches: list[Group],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Collates a list of batches into a single batch of inputs and targets."""
    inputs = []
    targets = []
    for batch_idx, batch in enumerate(batches):
        logger.debug(f"collate Processing batch {batch_idx}, len={len(batch)}")

        request = [e.request_tensor for e in batch]
        request = torch.stack(request)

        response = [e.response_tensor for e in batch]
        response = torch.stack(response)

        ref_logprobs = [e.ref_logprobs for e in batch]
        ref_logprobs = torch.stack(ref_logprobs)

        if ref_logprobs.dim() > 2:
            ref_logprobs = ref_logprobs.squeeze(0)

        advantages = [e.advantage for e in batch]
        advantages = torch.tensor(advantages).unsqueeze(-1)

        pad_id = batch[0].pad_id
        mask = torch.ne(response, pad_id)

        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=torch.bool)

        if mask.dim() == 0:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.dim() == 1:
            mask = mask.unsqueeze(0)

        input = {"tokens": torch.cat([request, response], dim=1)}
        target = {
            "response": response,
            "ref_logprobs": ref_logprobs,
            "advantages": advantages,
            "padding_mask": mask,
        }
        inputs.append(input)
        targets.append(target)

    return inputs, targets


def make_dapo_loss(
    beta: float = 0.01,
    clip_eps_low: float = 0.2,
    clip_eps_high: float = 0.28,
    max_kl_threshold: float = 0.5,
):
    """Factory function to create DAPO loss with configurable parameters.

    Args:
        beta: KL penalty coefficient (default 0.02, increased from 0.005 for stability)
        clip_eps_low: Lower clipping bound for policy ratio
        clip_eps_high: Upper clipping bound for policy ratio
        max_kl_threshold: If KL exceeds this, log a warning (early stopping signal)
    """

    def dapo_loss(
        logits: torch.Tensor,
        response: torch.Tensor,
        ref_logprobs: torch.Tensor,
        advantages: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """DAPO (Direct Alignment Policy Optimization) loss function."""
        action_log_probs = compute_logprobs(logits, response)

        # Compute KL divergence for monitoring and regularization
        # KL(policy || ref) approximation: mean(log_policy - log_ref)
        log_ratio = action_log_probs - ref_logprobs
        log_ratio_masked = log_ratio * padding_mask
        kl_div = (
            log_ratio_masked.sum(dim=1) / padding_mask.sum(dim=1).clamp(min=1.0)
        ).mean()

        # Log KL divergence for monitoring
        record_metric("rl_trainer/kl_divergence", kl_div.item(), Reduce.MEAN)

        # Warn if KL is too high (potential training collapse)
        kl_div_val = kl_div.abs().item()
        if kl_div_val > max_kl_threshold:
            logger.warning(
                f"KL divergence ({kl_div_val}) exceeds threshold ({max_kl_threshold}). "
                "Consider stopping training or increasing beta."
            )
            record_metric("rl_trainer/kl_threshold_exceeded", 1, Reduce.SUM)

        # KL penalty term (k3 approximation for stability)
        # Note: using ref_logprobs - action_log_probs for the penalty direction
        if beta != 0.0:
            kl_for_penalty = (ref_logprobs - action_log_probs) * padding_mask
            k3 = kl_for_penalty.exp() - 1 - kl_for_penalty

        old_action_log_probs = action_log_probs.detach()

        coef_1 = torch.exp(action_log_probs - old_action_log_probs)
        coef_2 = torch.clamp(coef_1, 1 - clip_eps_low, 1 + clip_eps_high)

        per_token_loss1 = coef_1 * advantages
        per_token_loss2 = coef_2 * advantages

        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        per_token_loss = per_token_loss * padding_mask

        if beta != 0.0:
            per_token_loss = per_token_loss + beta * k3
            # Log the KL penalty contribution
            kl_penalty = (
                (beta * k3).sum(dim=1) / padding_mask.sum(dim=1).clamp(min=1.0)
            ).mean()
            record_metric("rl_trainer/kl_penalty", kl_penalty.item(), Reduce.MEAN)

        loss = (
            per_token_loss.sum(dim=1) / padding_mask.sum(dim=1).clamp(min=1.0)
        ).mean()

        # Safety check: warn if loss is negative (sign of training collapse)
        loss_val = loss.item()
        if loss_val < 0:
            logger.warning(
                f"Negative loss detected ({loss_val}). Training may be unstable."
            )
            record_metric("rl_trainer/negative_loss_count", 1, Reduce.SUM)

        return loss

    return dapo_loss


@dataclass
class GenericRewardActor(ForgeActor):
    """Generic reward actor that uses GenericEnvClient and GenericAction."""

    env_actor: GenericOpenEnvClientActor
    build_action_fn: Callable[[str, Dict[str, Any]], GenericAction]
    evaluate_response_fn: Callable[[StepResult, str, Dict[str, Any]], float]
    evaluation_timeout_s: float = 60.0

    @endpoint
    async def setup(self):
        """Ensure the openenv directory is in sys.path for imports."""
        logger.debug("GenericRewardActor.setup Starting setup...")
        openenv_dir = Path(__file__).parent
        if str(openenv_dir) not in sys.path:
            sys.path.insert(0, str(openenv_dir))
        logger.debug(
            f"GenericRewardActor.setup Timeout set to {self.evaluation_timeout_s}s"
        )
        logger.debug("GenericRewardActor.setup Setup complete!")

    @endpoint
    async def evaluate_response(self, prompt: str, response: str, target: Any) -> float:
        """
        Evaluate response using task-specific functions with timeout protection.

        Args:
            prompt: The problem description
            response: The model's generated response
            target: The target/test data from dataset

        Returns:
            Reward score (0.0 if timeout or error)
        """
        try:
            # Build action using task-specific function (returns GenericAction)
            sample = {"target": target}
            action = self.build_action_fn(response, sample)

            # Execute in environment with timeout protection
            result = await asyncio.wait_for(
                self.env_actor.execute.call_one(
                    dict(action)
                ),  # Convert to dict for serialization
                timeout=self.evaluation_timeout_s,
            )

            # Evaluate result using task-specific function
            reward = self.evaluate_response_fn(result, response, sample)

            record_metric("reward/evaluate_response/sum_reward", reward, Reduce.SUM)
            record_metric("reward/evaluate_response/avg_reward", reward, Reduce.MEAN)
            record_metric("reward/evaluate_response/count_calls", 1, Reduce.SUM)

            return reward

        except asyncio.TimeoutError:
            logger.warning(
                f"Evaluation timeout after {self.evaluation_timeout_s}s - likely infinite loop in generated code"
            )
            record_metric("reward/evaluate_response/timeout_count", 1, Reduce.SUM)
            return 0.0

        except Exception as e:
            logger.error(f"Evaluation error: {e}")
            record_metric("reward/evaluate_response/error_count", 1, Reduce.SUM)
            return 0.0


@dataclass
class ComputeAdvantages(ForgeActor):
    @endpoint
    async def setup(self):
        logger.debug("ComputeAdvantages.setup Setup complete!")

    @endpoint
    async def compute(self, group: Group) -> list[float]:
        rewards = torch.tensor([[e.reward for e in group]], dtype=torch.float32)
        mean = rewards.mean(1, keepdim=True)
        std = rewards.std(1, keepdim=True)
        advantages = (rewards - mean) / (std + 1e-4)
        return advantages.squeeze(0).tolist()


@dataclass
class GenericDatasetActor(ForgeActor):
    """Generic dataset actor that uses task-specific transformation function."""

    path: str
    revision: str = "main"
    data_split: str = "train"
    streaming: bool = False
    model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    transform_sample_fn: Callable | None = None

    @endpoint
    async def setup(self):
        openenv_dir = Path(__file__).parent
        if str(openenv_dir) not in sys.path:
            sys.path.insert(0, str(openenv_dir))

        logger.debug("GenericDatasetActor.setup Starting setup...")
        self._tokenizer = get_tokenizer(self.model)
        logger.debug("GenericDatasetActor.setup Tokenizer loaded successfully")

        logger.debug(f"GenericDatasetActor.setup Loading dataset from: {self.path}")
        if os.path.isfile(self.path):
            if self.path.endswith(".parquet"):
                ds = load_dataset(
                    "parquet",
                    data_files={"train": self.path},
                    split=self.data_split,
                )
            elif self.path.endswith(".json"):
                ds = load_dataset(
                    "json",
                    data_files={"train": self.path},
                    split=self.data_split,
                )
            else:
                raise ValueError(f"Unsupported file format: {self.path}")
        else:
            ds = load_dataset(
                self.path,
                split=self.data_split,
                streaming=self.streaming,
                revision=self.revision,
            )

        if len(ds) == 0:
            raise ValueError(f"Dataset is empty after loading from {self.path}.")

        if self.transform_sample_fn:

            def transform_wrapper(sample):
                return self.transform_sample_fn(sample, self._tokenizer)

            original_size = len(ds)
            ds = ds.filter(lambda x: transform_wrapper(x) is not None)
            filtered_size = len(ds)

            if filtered_size == 0:
                raise ValueError(
                    f"Dataset transform filtered out ALL {original_size} samples!"
                )

            ds = ds.map(transform_wrapper)

        ds = ds.shuffle()

        if len(ds) == 0:
            raise ValueError("Dataset is empty after all transformations!")

        self._iterator = iter(ds)
        logger.debug("GenericDatasetActor.setup Setup complete!")

    @endpoint
    async def sample(self) -> dict[str, str] | None:
        try:
            sample = next(self._iterator)
            record_metric("dataset/sample/count_samples_generated", 1, Reduce.SUM)
            if "request" in sample:
                record_metric(
                    "dataset/sample/avg_sample_len",
                    len(sample["request"]),
                    Reduce.MEAN,
                )
            return sample
        except StopIteration:
            return None

    @endpoint
    async def pad_token(self):
        if self._tokenizer.pad_token_id is not None:
            return self._tokenizer.pad_token_id
        else:
            return self._tokenizer.eos_token_id


async def main(cfg: DictConfig):
    """Main GRPO training loop using GenericEnvClient."""
    group_size = cfg.group_size
    max_req_tokens = cfg.max_req_tokens
    max_res_tokens = cfg.max_res_tokens

    # Load task-specific functions
    logger.debug("main Loading task-specific functions...")
    task_config = cfg.task

    build_action_fn = None
    evaluate_response_fn = None
    transform_sample_fn = None

    if (
        isinstance(task_config.build_action, (tuple, list, ListConfig))
        and len(task_config.build_action) == 2
        and task_config.build_action[0] == "!function"
    ):
        build_action_fn = load_function_from_string(task_config.build_action[1])

    if (
        isinstance(task_config.evaluate_response, (tuple, list, ListConfig))
        and len(task_config.evaluate_response) == 2
        and task_config.evaluate_response[0] == "!function"
    ):
        evaluate_response_fn = load_function_from_string(
            task_config.evaluate_response[1]
        )

    if hasattr(task_config, "transform_sample"):
        if (
            isinstance(task_config.transform_sample, (tuple, list, ListConfig))
            and len(task_config.transform_sample) == 2
            and task_config.transform_sample[0] == "!function"
        ):
            transform_sample_fn = load_function_from_string(
                task_config.transform_sample[1]
            )

    logger.debug("main All task-specific functions loaded successfully")

    # Global setups
    provisioner = None
    if cfg.get("provisioner", None) is not None:
        provisioner = await init_provisioner(
            ProvisionerConfig(launcher_config=LauncherConfig(**cfg.provisioner))
        )
    else:
        provisioner = await init_provisioner()

    metric_logging_cfg = cfg.get("metric_logging", {})
    mlogger = await get_or_create_metric_logger(process_name="Controller")
    await mlogger.init_backends.call_one(metric_logging_cfg)

    # Setup GenericOpenEnvClientActor - works with ANY OpenEnv Docker image
    openenv_config = cfg.get("openenv_config", {})
    docker_image = openenv_config.get("docker_image")
    env_vars = openenv_config.get("env_vars", {})
    container_timeout_s = openenv_config.get("container_timeout_s", 180.0)
    request_timeout_s = openenv_config.get("request_timeout_s", 120.0)
    container_memory_gb = openenv_config.get("container_memory_gb", 4)

    # Set environment variables from config
    if "PORT" not in env_vars:
        env_vars["PORT"] = str(openenv_config.get("port", 8000))
    if "NUM_WORKER" not in env_vars:
        env_vars["NUM_WORKER"] = str(openenv_config.get("num_worker", 4))

    # Julia-specific env vars
    env_name = task_config.get("env_name", "generic")
    if env_name == "julia":
        if "JULIA_MAX_WORKERS" not in env_vars:
            env_vars["JULIA_MAX_WORKERS"] = str(
                openenv_config.get("julia_max_workers", 16)
            )
        if "JULIA_EXECUTION_TIMEOUT" not in env_vars:
            env_vars["JULIA_EXECUTION_TIMEOUT"] = str(int(request_timeout_s))

    logger.debug(
        f"main Initializing GenericOpenEnvClientActor with image={docker_image}..."
    )

    # Deploy GenericOpenEnvClientActor as Monarch actor
    env_actor = await GenericOpenEnvClientActor.options(
        **cfg.actors.get(f"{env_name}_env", cfg.actors.get("env", {}))
    ).as_actor(
        docker_image=docker_image,
        env_vars=env_vars,
        container_timeout_s=container_timeout_s,
        request_timeout_s=request_timeout_s,
        container_memory_gb=container_memory_gb,
    )
    logger.debug("main GenericOpenEnvClientActor initialized successfully")

    # Create all other actors
    (
        dataloader,
        policy,
        trainer,
        replay_buffer,
        compute_advantages,
        ref_model,
        reward_actor,
    ) = await asyncio.gather(
        GenericDatasetActor.options(**cfg.actors.dataset).as_actor(
            path=cfg.dataset.path,
            revision=cfg.dataset.get("revision", "main"),
            data_split=cfg.dataset.get("data_split", "train"),
            streaming=cfg.dataset.get("streaming", False),
            model=cfg.model,
            transform_sample_fn=transform_sample_fn,
        ),
        Policy.options(**cfg.services.policy).as_service(**cfg.policy),
        RLTrainer.options(**cfg.actors.trainer).as_actor(
            **cfg.trainer,
            loss=make_dapo_loss(
                beta=cfg.get("grpo", {}).get("beta", 0.02),
                clip_eps_low=cfg.get("grpo", {}).get("clip_eps_low", 0.2),
                clip_eps_high=cfg.get("grpo", {}).get("clip_eps_high", 0.28),
                max_kl_threshold=cfg.get("grpo", {}).get("max_kl_threshold", 0.5),
            ),
        ),
        ReplayBuffer.options(**cfg.actors.replay_buffer).as_actor(
            **cfg.replay_buffer, collate=collate
        ),
        ComputeAdvantages.options(**cfg.actors.compute_advantages).as_actor(),
        ReferenceModel.options(**cfg.services.ref_model).as_service(**cfg.ref_model),
        GenericRewardActor.options(**cfg.services.reward_actor).as_service(
            env_actor=env_actor,
            build_action_fn=build_action_fn,
            evaluate_response_fn=evaluate_response_fn,
            evaluation_timeout_s=cfg.get("evaluation_timeout_s", 60.0),
        ),
    )
    logger.debug("main asyncio.gather completed successfully!")

    max_steps = cfg.trainer.training.steps or -1

    print("All services initialized successfully!")
    shutdown_event = asyncio.Event()

    # Initialize torchstore
    trainer_num_procs = cfg.actors.trainer["procs"]
    trainer_host_mesh_name = cfg.actors.trainer["mesh_name"]
    trainer_hosts = await provisioner.get_host_mesh(trainer_host_mesh_name)
    await ts.initialize(
        mesh=trainer_hosts.spawn_procs(per_host={"procs": trainer_num_procs}),
        strategy=ts.LocalRankStrategy(),
    )
    print("Torchstore successfully initialized with local rank strategy")

    # Core RL loops
    async def continuous_rollouts():
        rollout_count = 0
        consecutive_errors = 0
        max_consecutive_errors = int(os.environ.get("FORGE_MAX_ROLLOUT_ERRORS", "50"))
        rollout_timeout_s = float(os.environ.get("FORGE_ROLLOUT_TIMEOUT_S", "300"))

        pad_id = await dataloader.pad_token.call_one()
        while not shutdown_event.is_set():
            try:
                t = Tracer("main_perf/continuous_rollouts")
                t.start()

                # Timeout on data loading
                try:
                    sample = await asyncio.wait_for(
                        dataloader.sample.call_one(),
                        timeout=30.0,
                    )
                except asyncio.TimeoutError:
                    logger.warning("[ROLLOUT] Timeout waiting for dataloader sample")
                    record_metric("main/continuous_rollouts/dataloader_timeout", 1, Reduce.SUM)
                    continue

                if sample is None:
                    print("Dataloader is empty, exiting continuous rollout")
                    return

                t.step("data_loading")

                prompt, target = sample["request"], sample["target"]

                # Timeout on policy generation
                try:
                    responses: list[Completion] = await asyncio.wait_for(
                        policy.generate.route(prompt),
                        timeout=rollout_timeout_s,
                    )
                except asyncio.TimeoutError:
                    logger.error(
                        f"[ROLLOUT] Timeout after {rollout_timeout_s}s waiting for policy.generate(). "
                        f"Generator may be stuck during weight update."
                    )
                    record_metric("main/continuous_rollouts/generation_timeout", 1, Reduce.SUM)
                    consecutive_errors += 1
                    if consecutive_errors >= max_consecutive_errors:
                        raise RuntimeError(
                            f"[ROLLOUT FAILURE] {consecutive_errors} consecutive rollout errors. "
                            f"Generator appears to be unresponsive."
                        )
                    continue

                t.step("policy_generation")

                episodes = []
                input_ids = torch.ones(
                    (group_size, max_req_tokens + max_res_tokens),
                    dtype=torch.long,
                )

                # Track evaluation errors for circuit breaker
                eval_errors_this_batch = 0

                for i, response in enumerate(responses):
                    episode = Episode(
                        episode_id=str(uuid.uuid4()),
                        pad_id=pad_id,
                        request_len=max_req_tokens,
                        response_len=max_res_tokens,
                        target=target,
                        completion=response,
                    )

                    try:
                        episode.reward = await reward_actor.evaluate_response.route(
                            prompt=prompt, response=response.text, target=target
                        )
                    except Exception as e:
                        logger.warning(f"[ROLLOUT] Reward evaluation failed: {e}")
                        episode.reward = 0.0
                        eval_errors_this_batch += 1
                        record_metric("main/continuous_rollouts/eval_error", 1, Reduce.SUM)

                    episodes.append(episode)
                    input_ids[i, :max_req_tokens] = episode.request_tensor
                    input_ids[i, max_req_tokens:] = episode.response_tensor

                t.step("reward_evaluation")

                # Circuit breaker: if ALL evaluations failed, something is wrong
                if eval_errors_this_batch == len(responses):
                    consecutive_errors += 1
                    logger.warning(
                        f"[CIRCUIT BREAKER] All {len(responses)} evaluations failed. "
                        f"Consecutive error batches: {consecutive_errors}/{max_consecutive_errors}"
                    )
                    if consecutive_errors >= max_consecutive_errors:
                        raise RuntimeError(
                            f"[ROLLOUT FAILURE] {consecutive_errors} consecutive batches with all evaluations failing. "
                            f"Environment actor appears to be unresponsive. Check container health."
                        )
                else:
                    # Reset error counter on partial success
                    consecutive_errors = 0

                # Timeout on reference model forward
                try:
                    ref_logprobs = await asyncio.wait_for(
                        ref_model.forward.route(input_ids, max_req_tokens, return_logprobs=True),
                        timeout=60.0,
                    )
                except asyncio.TimeoutError:
                    logger.error("[ROLLOUT] Timeout waiting for ref_model.forward()")
                    record_metric("main/continuous_rollouts/ref_model_timeout", 1, Reduce.SUM)
                    continue

                t.step("reference_model_calculate_logprobs")

                if not isinstance(ref_logprobs, torch.Tensor):
                    raise TypeError(
                        f"ref_model.forward.route() returned {type(ref_logprobs)} instead of torch.Tensor"
                    )

                for i, episode in enumerate(episodes):
                    episode.ref_logprobs = ref_logprobs[i]
                del ref_logprobs, input_ids

                advantages = await compute_advantages.compute.call_one(episodes)
                for episode, advantage in zip(episodes, advantages):
                    episode.advantage = advantage
                    await replay_buffer.add.call_one(episode)

                rollout_count += 1
                record_metric(
                    "main/continuous_rollouts/count_rollout_iterations", 1, Reduce.SUM
                )
                t.stop()

            except Exception as e:
                # Catch any unexpected errors in rollout loop to prevent thread crash
                logger.error(f"[ROLLOUT] Unexpected error in rollout loop: {e}")
                record_metric("main/continuous_rollouts/unexpected_error", 1, Reduce.SUM)
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    raise RuntimeError(
                        f"[ROLLOUT FAILURE] {consecutive_errors} consecutive errors in rollout loop. "
                        f"Last error: {e}"
                    )
                # Brief pause before retry
                await asyncio.sleep(1.0)

    async def continuous_training():
        training_step = 0
        restart_tracer = True
        consecutive_empty_samples = 0
        # Configurable via environment variable for advanced tuning
        max_empty_samples_before_error = int(
            os.environ.get("FORGE_MAX_EMPTY_BUFFER_WAIT_S", "60")
        ) * 10  # Convert seconds to 0.1s intervals

        while max_steps == -1 or training_step < max_steps:
            if restart_tracer:
                t = Tracer("main_perf/continuous_training")
                t.start()
                restart_tracer = False

            batch = await replay_buffer.sample.call_one(
                curr_policy_version=training_step
            )
            if batch is None:
                consecutive_empty_samples += 1

                # Log warning at increasing intervals
                if consecutive_empty_samples == 10:  # 1 second
                    logger.warning(
                        f"[BUFFER STARVATION] Buffer empty for 1s at step {training_step}. "
                        f"Rollouts may be blocked during weight update."
                    )
                elif consecutive_empty_samples == 100:  # 10 seconds
                    logger.warning(
                        f"[BUFFER STARVATION] Buffer empty for 10s at step {training_step}. "
                        f"Consider increasing max_policy_age or rollout_threads."
                    )
                elif consecutive_empty_samples == 300:  # 30 seconds
                    logger.error(
                        f"[BUFFER STARVATION] Buffer empty for 30s at step {training_step}. "
                        f"This indicates a likely deadlock. Check generator weight updates."
                    )

                # Fail after max wait to prevent infinite hangs
                if consecutive_empty_samples >= max_empty_samples_before_error:
                    raise RuntimeError(
                        f"[BUFFER STARVATION DEADLOCK] Replay buffer has been empty for "
                        f"{consecutive_empty_samples * 0.1:.1f} seconds at training step {training_step}. "
                        f"This typically indicates that:\n"
                        f"  1. All policy replicas are blocked during weight updates\n"
                        f"  2. max_policy_age ({cfg.get('off_by_n', 1)}) is too aggressive\n"
                        f"  3. rollout_threads ({num_rollout_threads}) is insufficient\n"
                        f"Solutions:\n"
                        f"  - Increase 'off_by_n' in config (recommended: 2-3)\n"
                        f"  - Increase 'rollout_threads' in config\n"
                        f"  - Increase policy service 'num_replicas'\n"
                        f"  - Set FORGE_MAX_EMPTY_BUFFER_WAIT_S env var to increase timeout"
                    )

                logger.debug("Running out of batch, now waiting")
                await asyncio.sleep(0.1)
            else:
                # Reset starvation counter on successful sample
                consecutive_empty_samples = 0
                t.step("waiting_for_buffer")

                inputs, targets = batch
                await trainer.train_step.call(inputs, targets)
                training_step += 1
                t.step("train_step")

                await trainer.push_weights.call(training_step)
                t.step("push_weights")

                # Backpressure: Check buffer health before triggering weight update.
                # Weight updates block all rollouts, so we need enough buffer headroom
                # to survive the blocking period without starving.
                buffer_health = await replay_buffer.health_check.call_one(
                    curr_policy_version=training_step
                )
                if not buffer_health["healthy"]:
                    logger.warning(
                        f"[BACKPRESSURE] Buffer low before weight update "
                        f"(surviving={buffer_health['surviving_after_eviction']}, "
                        f"required={buffer_health['required']}). "
                        f"Waiting for more episodes to prevent starvation."
                    )
                    # Wait up to 10 seconds for buffer to recover
                    for _ in range(100):
                        await asyncio.sleep(0.1)
                        buffer_health = await replay_buffer.health_check.call_one(
                            curr_policy_version=training_step
                        )
                        if buffer_health["healthy"]:
                            break
                    else:
                        logger.warning(
                            f"[BACKPRESSURE] Buffer still low after 10s wait. "
                            f"Proceeding with weight update anyway."
                        )
                t.step("backpressure_check")

                await policy.update_weights.fanout(training_step)
                t.step("update_weights")

                if training_step >= 2:
                    await drop_weights(training_step - 1)
                    t.step("drop_weights")

                t.stop()
                restart_tracer = True

                await mlogger.flush.call_one(training_step)

        print(
            f"Reached training limit ({max_steps} steps). Exiting continuous_training loop."
        )

    num_rollout_threads = cfg.get("rollout_threads", 1)
    print(f"Starting OpenEnv GRPO with {num_rollout_threads} rollout threads")
    rollout_tasks = [
        asyncio.create_task(continuous_rollouts()) for _ in range(num_rollout_threads)
    ]
    training_task = asyncio.create_task(continuous_training())

    try:
        await training_task
    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        import traceback

        print(f"Training failed with error: {e}")
        traceback.print_exc()
        raise
    finally:
        print("Shutting down...")
        shutdown_event.set()

        try:
            await asyncio.wait_for(
                asyncio.gather(*rollout_tasks, return_exceptions=True),
                timeout=5,
            )
        except asyncio.TimeoutError:
            for t in rollout_tasks:
                t.cancel()
            await asyncio.gather(*rollout_tasks, return_exceptions=True)

        training_task.cancel()

        print("Cleaning up environment Docker container...")
        try:
            await env_actor.teardown.call_one()
            print("Environment Docker container stopped successfully")
        except Exception as teardown_error:
            print(f"Warning: Error during environment teardown: {teardown_error}")

        await shutdown()


if __name__ == "__main__":

    @parse
    def _main(cfg):
        os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
        os.environ["NCCL_TIMEOUT_MS"] = "60000"
        os.environ["MONARCH_HOSTMESH_V1"] = "1"
        os.environ["TORCHSTORE_RDMA_ENABLED"] = "1"
        asyncio.run(main(cfg))

    _main()
