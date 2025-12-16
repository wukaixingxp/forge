# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Generic OpenEnv GRPO Training Script.

This script can be used for any OpenEnv task by specifying task-specific
functions in the YAML config using !function references.

Usage: python -m apps.openenv.main --config apps/openenv/llama3_8b_julia.yaml
"""

import asyncio
import importlib
import logging
import os
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

# CRITICAL: Add openenv directory to sys.path at module level
# This ensures that when remote actors unpickle function references (e.g., julia_utils functions),
# the module can be imported successfully. This must happen BEFORE any actor definitions.
_openenv_dir = Path(__file__).parent
if str(_openenv_dir) not in sys.path:
    sys.path.insert(0, str(_openenv_dir))

import torch
import torch.nn.functional as F
import torchstore as ts
import yaml
from datasets import load_dataset
from forge.actors._torchstore_utils import (
    get_dcp_whole_state_dict_key,
    get_param_prefix,
)
from forge.actors.generator import Generator
from forge.actors.generic_openenv import GenericOpenEnvActor
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
    """
    Load a function from a string reference like 'julia_utils.build_julia_prompt'.

    Args:
        func_ref: String in format 'module_name.function_name'

    Returns:
        The loaded function
    """
    # Add openenv directory to path
    openenv_dir = Path(__file__).parent
    if str(openenv_dir) not in sys.path:
        sys.path.insert(0, str(openenv_dir))

    # Split module and function name
    module_name, func_name = func_ref.rsplit(".", 1)

    # Import module and get function
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
    """
    Collates a list of batches into a single batch of inputs and targets.
    """
    inputs = []
    targets = []
    for batch_idx, batch in enumerate(batches):
        logger.debug(f"collate Processing batch {batch_idx}, len={len(batch)}")

        request = [e.request_tensor for e in batch]
        request = torch.stack(request)
        logger.debug(f"collate request shape: {request.shape}")

        response = [e.response_tensor for e in batch]
        response = torch.stack(response)
        logger.debug(f"collate response shape: {response.shape}")

        ref_logprobs = [e.ref_logprobs for e in batch]
        ref_logprobs = torch.stack(ref_logprobs)

        if ref_logprobs.dim() > 2:
            ref_logprobs = ref_logprobs.squeeze(0)
        logger.debug(f"collate ref_logprobs shape after stack: {ref_logprobs.shape}")

        advantages = [e.advantage for e in batch]
        advantages = torch.tensor(advantages).unsqueeze(-1)
        logger.debug(f"collate advantages shape: {advantages.shape}")

        pad_id = batch[0].pad_id

        mask = torch.ne(response, pad_id)
        logger.debug(
            f"collate mask shape before checks: {mask.shape}, dtype: {mask.dtype}, type: {type(mask)}"
        )

        if not isinstance(mask, torch.Tensor):
            logger.debug(
                f"collate WARNING: mask is not a tensor, converting from {type(mask)}"
            )
            mask = torch.tensor(mask, dtype=torch.bool)

        if mask.dim() == 0:
            logger.debug(f"collate WARNING: mask is 0D scalar, unsqueezing twice")
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.dim() == 1:
            logger.debug(
                f"collate WARNING: mask is 1D with shape {mask.shape}, unsqueezing to 2D"
            )
            mask = mask.unsqueeze(0)

        logger.debug(f"collate mask final shape: {mask.shape}")
        logger.debug(
            f"collate All shapes - request: {request.shape}, response: {response.shape}, "
            f"ref_logprobs: {ref_logprobs.shape}, advantages: {advantages.shape}, mask: {mask.shape}"
        )

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


def dapo_loss(
    logits: torch.Tensor,
    response: torch.Tensor,
    ref_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    padding_mask: torch.Tensor,
    beta: float = 0.005,
    clip_eps_low: float = 0.2,
    clip_eps_high: float = 0.28,
) -> torch.Tensor:
    """DAPO (Direct Alignment Policy Optimization) loss function."""
    action_log_probs = compute_logprobs(logits, response)

    if beta != 0.0:
        log_ratio = ref_logprobs - action_log_probs
        log_ratio = log_ratio * padding_mask
        k3 = log_ratio.exp() - 1 - log_ratio

    old_action_log_probs = action_log_probs.detach()

    coef_1 = torch.exp(action_log_probs - old_action_log_probs)
    coef_2 = torch.clamp(coef_1, 1 - clip_eps_low, 1 + clip_eps_high)

    per_token_loss1 = coef_1 * advantages
    per_token_loss2 = coef_2 * advantages

    per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
    per_token_loss = per_token_loss * padding_mask

    if beta != 0.0:
        per_token_loss = per_token_loss + beta * k3

    loss = (per_token_loss.sum(dim=1) / padding_mask.sum(dim=1).clamp(min=1.0)).mean()

    return loss


@dataclass
class GenericRewardActor(ForgeActor):
    """Generic reward actor that uses task-specific evaluation function."""

    env_actor: GenericOpenEnvActor
    build_action_fn: Callable
    evaluate_response_fn: Callable
    evaluation_timeout_s: float = 60.0  # Default 60 second timeout per evaluation

    @endpoint
    def setup(self):
        """Ensure the openenv directory is in sys.path for imports."""
        logger.debug("GenericRewardActor.setup Starting setup...")
        openenv_dir = Path(__file__).parent
        if str(openenv_dir) not in sys.path:
            sys.path.insert(0, str(openenv_dir))
        logger.debug(f"GenericRewardActor.setup Timeout set to {self.evaluation_timeout_s}s")
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
            # Build action using task-specific function
            # Pass the target as part of a sample dict
            sample = {"target": target}
            action = self.build_action_fn(response, sample)

            # Execute in environment with timeout protection
            # This prevents infinite loops in generated code from blocking training
            result = await asyncio.wait_for(
                self.env_actor.execute.call_one(action),
                timeout=self.evaluation_timeout_s
            )

            # Evaluate result using task-specific function
            reward = self.evaluate_response_fn(result, response, sample)

            # Record reward metrics
            record_metric(
                "reward/evaluate_response/sum_reward",
                reward,
                Reduce.SUM,
            )
            record_metric(
                "reward/evaluate_response/avg_reward",
                reward,
                Reduce.MEAN,
            )
            record_metric(
                "reward/evaluate_response/std_reward",
                reward,
                Reduce.STD,
            )
            record_metric(
                "reward/evaluate_response/count_calls",
                1,
                Reduce.SUM,
            )

            return reward

        except asyncio.TimeoutError:
            # Code execution exceeded timeout (likely infinite loop)
            logger.warning(
                f"Evaluation timeout after {self.evaluation_timeout_s}s - likely infinite loop in generated code"
            )
            record_metric("reward/evaluate_response/timeout_count", 1, Reduce.SUM)
            return 0.0

        except Exception as e:
            # Other errors (syntax errors, runtime errors, etc.)
            logger.error(f"Evaluation error: {e}")
            record_metric("reward/evaluate_response/error_count", 1, Reduce.SUM)
            return 0.0


@dataclass
class ComputeAdvantages(ForgeActor):
    @endpoint
    def setup(self):
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
    def setup(self):
        """Ensure the openenv directory is in sys.path for imports."""
        openenv_dir = Path(__file__).parent
        if str(openenv_dir) not in sys.path:
            sys.path.insert(0, str(openenv_dir))

        logger.debug("GenericDatasetActor.setup Starting setup...")
        self._tokenizer = get_tokenizer(self.model)
        logger.debug("GenericDatasetActor.setup Tokenizer loaded successfully")

        # Load dataset
        import os

        logger.debug(f"GenericDatasetActor.setup Loading dataset from: {self.path}")
        if os.path.isfile(self.path):
            if self.path.endswith(".parquet"):
                print(
                    "[DEBUG GenericDatasetActor.setup] Loading from local parquet file"
                )
                ds = load_dataset(
                    "parquet",
                    data_files={"train": self.path},
                    split=self.data_split,
                )
            elif self.path.endswith(".json"):
                logger.debug("GenericDatasetActor.setup Loading from local JSON file")
                ds = load_dataset(
                    "json",
                    data_files={"train": self.path},
                    split=self.data_split,
                )
            else:
                raise ValueError(
                    f"Unsupported file format: {self.path}. Only .parquet and .json files are supported."
                )
        else:
            logger.debug("GenericDatasetActor.setup Loading from HF hub or directory")
            ds = load_dataset(
                self.path,
                split=self.data_split,
                streaming=self.streaming,
                revision=self.revision,
            )
        print(
            f"[DEBUG GenericDatasetActor.setup] Dataset loaded successfully, type: {type(ds)}"
        )
        logger.debug(f"GenericDatasetActor.setup Initial dataset size: {len(ds)}")

        # Ensure we have data before proceeding
        if len(ds) == 0:
            raise ValueError(
                f"Dataset is empty after loading from {self.path}. "
                "Please check the dataset file and path."
            )

        # Apply transformation function if provided
        if self.transform_sample_fn:
            logger.debug("GenericDatasetActor.setup Applying transform_sample_fn...")

            def transform_wrapper(sample):
                return self.transform_sample_fn(sample, self._tokenizer)

            logger.debug("GenericDatasetActor.setup Applying filter...")
            original_size = len(ds)
            ds = ds.filter(lambda x: transform_wrapper(x) is not None)
            logger.debug("GenericDatasetActor.setup Filter applied, applying map...")
            filtered_size = len(ds)

            # Critical assertion: ensure filtering didn't remove all samples
            if filtered_size == 0:
                raise ValueError(
                    f"Dataset transform filtered out ALL {original_size} samples! "
                    f"This usually means the transform_sample function is rejecting all samples. "
                    f"Please check:\n"
                    f"  1. Dataset format matches what transform_sample expects\n"
                    f"  2. Required fields exist in the dataset\n"
                    f"  3. transform_sample validation logic is correct\n"
                    f"Dataset path: {self.path}"
                )

            # Warning if too many samples were filtered out
            filtered_out = original_size - filtered_size
            filter_rate = (
                (filtered_out / original_size) * 100 if original_size > 0 else 0
            )
            if filter_rate > 50:
                print(
                    f"WARNING: Transform filtered out {filtered_out}/{original_size} samples ({filter_rate:.1f}%)!"
                )
                print(
                    f"         This seems unusually high. Please verify dataset format."
                )
            else:
                logger.debug(
                    f"GenericDatasetActor.setup Kept {filtered_size}/{original_size} samples ({100-filter_rate:.1f}%)"
                )

            ds = ds.map(transform_wrapper)
            logger.debug("GenericDatasetActor.setup Map applied successfully")
        else:
            print(
                "[DEBUG GenericDatasetActor.setup] No transform_sample_fn provided, skipping transformation"
            )

        logger.debug("GenericDatasetActor.setup Shuffling dataset...")
        ds = ds.shuffle()

        # Final assertion: ensure we still have data after all transformations
        final_size = len(ds)
        if final_size == 0:
            raise ValueError(
                f"Dataset is empty after all transformations! "
                f"This should not happen - please file a bug report. "
                f"Dataset path: {self.path}"
            )

        logger.debug(f"GenericDatasetActor.setup Final dataset size: {final_size}")
        logger.debug("GenericDatasetActor.setup Creating iterator...")
        self._iterator = iter(ds)
        logger.debug("GenericDatasetActor.setup Setup complete!")

    @endpoint
    async def sample(self) -> dict[str, str] | None:
        try:
            sample = next(self._iterator)

            record_metric("dataset/sample/count_samples_generated", 1, Reduce.SUM)

            # Only record sample length if the "request" key exists
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


async def drop_weights(version: int):
    print(f"Dropping weights @ version {version}")
    start_time = time.perf_counter()
    prefix = get_param_prefix(version)
    matching_keys = await ts.keys(prefix)
    dcp_key = get_dcp_whole_state_dict_key(version)
    if dcp_key in matching_keys:
        dcp_handle = await ts.get(dcp_key)
        dcp_handle.drop()
    for key in matching_keys:
        await ts.delete(key)
    elapsed = time.perf_counter() - start_time
    print(f"Dropped weights @ version {version}, took {elapsed:.2f} seconds")


async def main(cfg: DictConfig):
    """Main GRPO training loop for generic OpenEnv tasks."""
    group_size = cfg.group_size
    max_req_tokens = cfg.max_req_tokens
    max_res_tokens = cfg.max_res_tokens

    # Load task-specific functions
    logger.debug("main Loading task-specific functions...")
    task_config = cfg.task

    # Load functions from !function references
    build_action_fn = None
    evaluate_response_fn = None
    transform_sample_fn = None

    logger.debug(f"main task_config.build_action = {task_config.build_action}")
    print(
        f"[DEBUG main] task_config.build_action type = {type(task_config.build_action)}"
    )
    print(
        f"[DEBUG main] isinstance check = {isinstance(task_config.build_action, (tuple, list))}"
    )
    if hasattr(task_config.build_action, "__len__"):
        logger.debug(f"main len = {len(task_config.build_action)}")
        if len(task_config.build_action) > 0:
            logger.debug(f"main] first element = {task_config.build_action[0]}")

    # OmegaConf may convert tuples/tags to lists or ListConfig, so check for all
    if (
        isinstance(task_config.build_action, (tuple, list, ListConfig))
        and len(task_config.build_action) == 2
        and task_config.build_action[0] == "!function"
    ):
        print(
            f"[DEBUG main] Loading build_action_fn from {task_config.build_action[1]}"
        )
        build_action_fn = load_function_from_string(task_config.build_action[1])
        logger.debug(f"main build_action_fn loaded: {build_action_fn}")

    print(
        f"[DEBUG main] task_config.evaluate_response = {task_config.evaluate_response}"
    )
    if (
        isinstance(task_config.evaluate_response, (tuple, list, ListConfig))
        and len(task_config.evaluate_response) == 2
        and task_config.evaluate_response[0] == "!function"
    ):
        print(
            f"[DEBUG main] Loading evaluate_response_fn from {task_config.evaluate_response[1]}"
        )
        evaluate_response_fn = load_function_from_string(
            task_config.evaluate_response[1]
        )
        logger.debug(f"main evaluate_response_fn loaded: {evaluate_response_fn}")

    if hasattr(task_config, "transform_sample"):
        print(
            f"[DEBUG main] task_config.transform_sample = {task_config.transform_sample}"
        )
        if (
            isinstance(task_config.transform_sample, (tuple, list, ListConfig))
            and len(task_config.transform_sample) == 2
            and task_config.transform_sample[0] == "!function"
        ):
            print(
                f"[DEBUG main] Loading transform_sample_fn from {task_config.transform_sample[1]}"
            )
            transform_sample_fn = load_function_from_string(
                task_config.transform_sample[1]
            )
            logger.debug(f"main transform_sample_fn loaded: {transform_sample_fn}")
    else:
        logger.debug("main No transform_sample in task_config")

    logger.debug("main All task-specific functions loaded successfully")

    # Validate configuration for common pitfalls
    max_policy_age = cfg.replay_buffer.get("max_policy_age", None)
    off_by_n = cfg.get("off_by_n", None)

    if max_policy_age == 0 or off_by_n == 0:
        logger.warning(
            f"\n{'='*80}\n"
            f"WARNING: Potential training hang detected in configuration!\n"
            f"{'='*80}\n"
            f"  max_policy_age = {max_policy_age} (from off_by_n = {off_by_n})\n\n"
            f"RISK:\n"
            f"  Setting max_policy_age=0 can cause training to hang if weight updates\n"
            f"  take longer than the time between training steps. This creates a race\n"
            f"  condition where the buffer evicts all episodes before the generator\n"
            f"  finishes updating to the new policy version.\n\n"
            f"RECOMMENDATION:\n"
            f"  Set 'off_by_n: 1' (or higher) in your config to allow episodes from\n"
            f"  the previous policy version to remain in the buffer.\n\n"
            f"  If you see '[HANG DETECTION]' warnings during training, this is the\n"
            f"  likely cause. The training will automatically exit with diagnostics\n"
            f"  after {cfg.get('sample_timeout_s', 300)}s of waiting.\n"
            f"{'='*80}\n"
        )

    # Get env class and action class from task config
    from envs import AutoAction, AutoEnv

    env_name = task_config.env_name
    env_class = AutoEnv.from_name(env_name)
    action_class = AutoAction.from_env(env_name)

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

    # Setup environment using GenericOpenEnvActor
    openenv_config = cfg.get("openenv_config", {})
    docker_image = openenv_config.get("docker_image", "julia-env:latest")
    env_vars = openenv_config.get("env_vars", {})
    container_timeout_s = openenv_config.get("container_timeout_s", 180.0)
    request_timeout_s = openenv_config.get("request_timeout_s", 120.0)
    container_memory_gb = openenv_config.get("container_memory_gb", 4)

    # Set environment variables from config
    if "PORT" not in env_vars:
        env_vars["PORT"] = str(openenv_config.get("port", 8000))
    if "NUM_WORKER" not in env_vars:
        env_vars["NUM_WORKER"] = str(openenv_config.get("num_worker", 4))
    if env_name == "julia" and "JULIA_MAX_WORKERS" not in env_vars:
        env_vars["JULIA_MAX_WORKERS"] = str(openenv_config.get("julia_max_workers", 16))
    # CRITICAL: Set execution timeout for Julia environment server
    # This ensures the timeout inside the Docker container matches our config
    if env_name == "julia" and "JULIA_EXECUTION_TIMEOUT" not in env_vars:
        env_vars["JULIA_EXECUTION_TIMEOUT"] = str(int(request_timeout_s))

    # Handle additional_imports for coding environment
    if env_name == "coding" and "PYTHON_ADDITIONAL_IMPORTS" not in env_vars:
        additional_imports = openenv_config.get("additional_imports", [])
        if additional_imports:
            # Convert list to comma-separated string
            env_vars["PYTHON_ADDITIONAL_IMPORTS"] = ",".join(additional_imports)
            logger.debug(
                f"main Set PYTHON_ADDITIONAL_IMPORTS: {env_vars['PYTHON_ADDITIONAL_IMPORTS']}"
            )

    logger.debug("main Initializing GenericOpenEnvActor...")
    env_actor = await GenericOpenEnvActor.options(
        **cfg.actors.get(f"{env_name}_env", cfg.actors.get("env", {}))
    ).as_actor(
        env_class=env_class,
        action_class=action_class,
        docker_image=docker_image,
        env_vars=env_vars,
        container_timeout_s=container_timeout_s,
        request_timeout_s=request_timeout_s,
        container_memory_gb=container_memory_gb,
    )
    logger.debug("main GenericOpenEnvActor initialized successfully")

    logger.debug("main Starting asyncio.gather for all actors...")
    logger.debug("main - Creating GenericDatasetActor...")
    dataset_task = GenericDatasetActor.options(**cfg.actors.dataset).as_actor(
        path=cfg.dataset.path,
        revision=cfg.dataset.get("revision", "main"),
        data_split=cfg.dataset.get("data_split", "train"),
        streaming=cfg.dataset.get("streaming", False),
        model=cfg.model,
        transform_sample_fn=transform_sample_fn,
    )
    logger.debug("main - Creating Policy...")
    policy_task = Policy.options(**cfg.services.policy).as_service(**cfg.policy)
    logger.debug("main - Creating RLTrainer...")
    trainer_task = RLTrainer.options(**cfg.actors.trainer).as_actor(
        **cfg.trainer, loss=dapo_loss
    )
    logger.debug("main - Creating ReplayBuffer...")
    replay_task = ReplayBuffer.options(**cfg.actors.replay_buffer).as_actor(
        **cfg.replay_buffer, collate=collate
    )
    logger.debug("main - Creating ComputeAdvantages...")
    advantages_task = ComputeAdvantages.options(
        **cfg.actors.compute_advantages
    ).as_actor()
    logger.debug("main - Creating ReferenceModel...")
    ref_model_task = ReferenceModel.options(**cfg.services.ref_model).as_service(
        **cfg.ref_model
    )
    logger.debug("main - Creating GenericRewardActor...")

    # Get evaluation timeout from config (default 60 seconds)
    evaluation_timeout_s = cfg.get("evaluation_timeout_s", 60.0)
    logger.debug(f"main Using evaluation_timeout_s={evaluation_timeout_s}")

    reward_task = GenericRewardActor.options(**cfg.services.reward_actor).as_service(
        env_actor=env_actor,
        build_action_fn=build_action_fn,
        evaluate_response_fn=evaluate_response_fn,
        evaluation_timeout_s=evaluation_timeout_s,
    )

    logger.debug("main All tasks created, now awaiting asyncio.gather...")
    (
        dataloader,
        policy,
        trainer,
        replay_buffer,
        compute_advantages,
        ref_model,
        reward_actor,
    ) = await asyncio.gather(
        dataset_task,
        policy_task,
        trainer_task,
        replay_task,
        advantages_task,
        ref_model_task,
        reward_task,
    )
    logger.debug("main asyncio.gather completed successfully!")

    max_steps = cfg.trainer.training.steps or -1

    print("All services initialized successfully!")
    shutdown_event = asyncio.Event()

    # Initialize torchstore
    logger.debug("main Initializing torchstore...")
    trainer_num_procs = cfg.actors.trainer["procs"]
    trainer_host_mesh_name = cfg.actors.trainer["mesh_name"]
    trainer_hosts = provisioner.get_host_mesh(trainer_host_mesh_name)
    logger.debug(
        f"main trainer_num_procs={trainer_num_procs}, mesh_name={trainer_host_mesh_name}"
    )
    await ts.initialize(
        mesh=trainer_hosts.spawn_procs(per_host={"procs": trainer_num_procs}),
        strategy=ts.LocalRankStrategy(),
    )
    print("Torchstore successfully initialized with local rank strategy")

    # Core RL loops
    async def continuous_rollouts():
        rollout_count = 0
        pad_id = await dataloader.pad_token.call_one()
        while not shutdown_event.is_set():
            t = Tracer("main_perf/continuous_rollouts")
            t.start()
            sample = await dataloader.sample.call_one()
            if sample is None:
                print("Dataloader is empty, exiting continuous rollout")
                return

            t.step("data_loading")

            prompt, target = sample["request"], sample["target"]
            responses: list[Completion] = await policy.generate.route(prompt)
            t.step("policy_generation")

            episodes = []
            input_ids = torch.ones(
                (group_size, max_req_tokens + max_res_tokens),
                dtype=torch.long,
            )

            # Sequential reward evaluation (like GRPO) - avoids timeout amplification
            for i, response in enumerate(responses):
                episode = Episode(
                    episode_id=str(uuid.uuid4()),
                    pad_id=pad_id,
                    request_len=max_req_tokens,
                    response_len=max_res_tokens,
                    target=target,
                    completion=response,
                )

                # Evaluate reward immediately (sequential, not parallel)
                # This prevents asyncio.gather() from waiting for ALL requests
                # when some timeout, reducing 480s timeouts to faster failures
                episode.reward = await reward_actor.evaluate_response.route(
                    prompt=prompt, response=response.text, target=target
                )

                episodes.append(episode)

                # Build input_ids for ref model
                input_ids[i, :max_req_tokens] = episode.request_tensor
                input_ids[i, max_req_tokens:] = episode.response_tensor

            t.step("reward_evaluation")

            ref_logprobs = await ref_model.forward.route(
                input_ids, max_req_tokens, return_logprobs=True
            )
            t.step("reference_model_calculate_logprobs")

            # Defensive check: ensure ref_logprobs is a tensor
            if not isinstance(ref_logprobs, torch.Tensor):
                logger.error(
                    f"ERROR: ref_model.forward.route() returned unexpected type: {type(ref_logprobs)}, "
                    f"value: {ref_logprobs}. Expected torch.Tensor."
                )
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

    async def continuous_training():
        training_step = 0
        restart_tracer = True

        # Progress monitoring: detect training stalls
        last_progress_time = time.time()
        last_logged_step = -1
        stall_warning_threshold_s = 300  # Warn after 5 minutes of no progress
        stall_error_threshold_s = 600  # Error after 10 minutes of no progress

        # Timeout protection: detect when buffer sampling hangs
        sample_timeout_s = cfg.get("sample_timeout_s", 300)  # Default 5 minutes
        sample_wait_start = None
        consecutive_none_count = 0

        while max_steps == -1 or training_step < max_steps:
            if restart_tracer:
                t = Tracer("main_perf/continuous_training")
                t.start()
                restart_tracer = False

            # Check for training stalls (no progress for extended period)
            current_time = time.time()
            time_since_progress = current_time - last_progress_time

            if training_step > last_logged_step:
                # Progress made - reset monitoring
                last_progress_time = current_time
                last_logged_step = training_step
            elif time_since_progress > stall_error_threshold_s:
                # No progress for 10+ minutes - critical stall
                logger.error(
                    f"CRITICAL: Training stalled at step {training_step} for {time_since_progress:.1f}s. "
                    f"No progress since step {last_logged_step}. This may indicate:"
                    f"\n  - Trainer is hung/blocked"
                    f"\n  - Replay buffer is empty or stuck"
                    f"\n  - vLLM cache corruption preventing new episodes"
                    f"\n  - Deadlock in weight synchronization"
                )
                record_metric("training/stall_detected", 1, Reduce.SUM)
                # Force cache reset on policy to recover
                try:
                    logger.info("Attempting emergency vLLM cache reset...")
                    await policy._reset_prefix_cache.fanout()
                    logger.info("Emergency cache reset completed")
                    last_progress_time = current_time  # Reset timer after intervention
                except Exception as e:
                    logger.error(f"Failed to reset vLLM cache: {e}")
                    raise RuntimeError(
                        f"Training stalled for {time_since_progress:.1f}s at step {training_step}. "
                        "Emergency cache reset failed. Manual intervention required."
                    )
            elif time_since_progress > stall_warning_threshold_s:
                # No progress for 5+ minutes - warning
                if int(time_since_progress) % 60 == 0:  # Log once per minute
                    logger.warning(
                        f"Training slow: no progress at step {training_step} for {time_since_progress:.1f}s"
                    )
                    record_metric("training/slow_progress_warnings", 1, Reduce.SUM)

            batch = await replay_buffer.sample.call_one(
                curr_policy_version=training_step
            )
            if batch is None:
                logger.debug("Running out of batch, now waiting")
                await asyncio.sleep(0.1)  # Match GRPO's 100ms (was 1s)
            else:

                t.step("waiting_for_buffer")

                inputs, targets = batch
                await trainer.train_step.call(inputs, targets)
                training_step += 1
                t.step("train_step")

                await trainer.push_weights.call(training_step)
                t.step("push_weights")

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
    num_training_threads = cfg.get("training_threads", 1)
    print(
        f"Starting OpenEnv GRPO with {num_rollout_threads} rollout threads, {num_training_threads} training threads"
    )
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
        print("\nFull traceback:")
        traceback.print_exc()
        raise
    finally:
        print("Shutting down... (this may take a few seconds)")
        shutdown_event.set()

        try:
            await asyncio.wait_for(
                asyncio.gather(*rollout_tasks, return_exceptions=True),
                timeout=5,
            )
        except asyncio.TimeoutError:
            print("Timeout waiting for rollouts; forcing cancellation...")
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
        os.environ["NCCL_TIMEOUT_MS"] = "60000"  # 60 second timeout
        os.environ["MONARCH_HOSTMESH_V1"] = "1"
        os.environ["TORCHSTORE_RDMA_ENABLED"] = "1"
        asyncio.run(main(cfg))

    _main()
