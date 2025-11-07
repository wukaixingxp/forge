# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Usage: python -m apps.grpo.main --config apps/grpo/qwen3_1_7b.yaml

import os

# Limit OpenBLAS and MKL threads to prevent thread exhaustion
# Must be set before importing numpy, scipy, torch, etc.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")

# Limit Go runtime threads for Podman/container management
# This prevents the "failed to create new OS thread" error in Podman
os.environ.setdefault("GOMAXPROCS", "4")
os.environ.setdefault("GOMEMLIMIT", "2GiB")

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable

import torch

# Configure logging to see INFO level messages
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
import torch.nn.functional as F
import torchstore as ts
from datasets import load_dataset
from forge.actors._torchstore_utils import (
    get_dcp_whole_state_dict_key,
    get_param_prefix,
)
from forge.actors.generator import Generator

# from forge.actors.podman_coder import PodmanPythonCoder
from forge.actors.openenv_coder import OpenEnvCoder

from forge.actors.reference_model import ReferenceModel
from forge.actors.replay_buffer import ReplayBuffer
from forge.actors.trainer import RLTrainer
from forge.controller.actor import ForgeActor
from forge.controller.provisioner import init_provisioner, shutdown
from forge.data.rewards import GroundTruthTestReward, ThinkingReward
from forge.data_models.completion import Completion
from forge.observability.metric_actors import get_or_create_metric_logger
from forge.observability.metrics import record_metric, Reduce
from forge.observability.perf_tracker import Tracer

from forge.types import LauncherConfig, ProvisionerConfig
from forge.util.config import parse
from forge.util.ops import compute_logprobs
from monarch.actor import endpoint
from omegaconf import DictConfig
from vllm.transformers_utils.tokenizer import get_tokenizer


@dataclass
class Episode:
    episode_id: str
    pad_id: int
    request_len: int
    response_len: int
    target: Any | None = None
    # Processed data
    completion: Completion | None = None
    ref_logprobs: torch.Tensor | None = None
    reward: float | None = None
    advantage: float | None = None

    @property
    def policy_version(self) -> int | None:
        if self.completion is None:
            return None
        return self.completion.generator_version

    @property
    def request_tensor(self) -> torch.Tensor:
        if self.completion is None:
            return torch.full((self.request_len,), self.pad_id, dtype=torch.long)
        request_tokens: torch.Tensor = self.completion.prompt_ids
        # Use clone() if input is already a tensor to avoid warning
        if isinstance(request_tokens, torch.Tensor):
            tensor = request_tokens.detach().clone()
        else:
            tensor = torch.tensor(request_tokens, dtype=torch.long)
        if tensor.shape[0] < self.request_len:  # left pad
            diff = self.request_len - tensor.shape[0]
            tensor = F.pad(tensor, (diff, 0), value=self.pad_id)
        return tensor

    @property
    def response_tensor(self) -> torch.Tensor:
        if self.completion is None:
            return torch.full((self.response_len,), self.pad_id, dtype=torch.long)
        response_tokens: torch.Tensor = self.completion.token_ids
        # Use clone() if input is already a tensor to avoid warning
        if isinstance(response_tokens, torch.Tensor):
            tensor = response_tokens.detach().clone()
        else:
            tensor = torch.tensor(response_tokens, dtype=torch.long)
        if tensor.shape[0] < self.response_len:  # right pad
            diff = self.response_len - tensor.shape[0]
            tensor = F.pad(tensor, (0, diff), value=self.pad_id)
        return tensor


# Represents the group (G) of episodes in GRPO
Group = list[Episode]

# Represents the Policy Model to collect data from
Policy = Generator


def collate(
    batches: list[Group],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    inputs = []
    targets = []
    for batch in batches:
        request = [e.request_tensor for e in batch]
        request = torch.stack(request)  # [b x s]

        response = [e.response_tensor for e in batch]
        response = torch.stack(response)  # [b x s]

        ref_logprobs = [e.ref_logprobs for e in batch]
        ref_logprobs = torch.stack(ref_logprobs)  # [b x s]
        # Only squeeze last dimension if needed, preserve batch dimension
        if ref_logprobs.dim() > 2:
            ref_logprobs = ref_logprobs.squeeze(-1)

        advantages = [e.advantage for e in batch]
        advantages = torch.tensor(advantages).unsqueeze(-1)  # [b x 1]

        pad_id = batch[0].pad_id
        # Ensure mask is always a 2D tensor [b x s], even for single batch elements
        mask = response != pad_id  # [b x s]
        # Ensure it's a tensor and preserve shape
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=torch.bool)
        # Ensure mask is always 2D
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


def simple_grpo_loss(
    logits: torch.Tensor,
    response: torch.Tensor,
    ref_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    padding_mask: torch.Tensor,
    beta: float = 0.001,
) -> torch.Tensor:
    """
    GRPO Loss Function for on-policy samples with numerical stability improvements

    Args:
        logits: Model logits [batch_size, seq_len, vocab_size]
        response: Response tokens [batch_size, seq_len]
        ref_logprobs: Reference model log probabilities [batch_size, seq_len]
        advantages: Computed advantages [batch_size, 1]
        padding_mask: Mask for valid tokens [batch_size, seq_len]
        beta: KL penalty coefficient (increased for better regularization)

    Note: Since samples are generated by current policy (on-policy),
    the policy gradient uses the standard REINFORCE form.
    ref_logprobs is ONLY used for the KL penalty, not the policy ratio.
    """
    logprobs: torch.Tensor = compute_logprobs(logits, response)

    # Check for NaN/Inf in logprobs
    if torch.isnan(logprobs).any() or torch.isinf(logprobs).any():
        print("WARNING: NaN/Inf detected in logprobs!")
        logprobs = torch.nan_to_num(logprobs, nan=0.0, posinf=0.0, neginf=-100.0)

    # ✅ CORRECT: On-policy REINFORCE gradient
    # This gives gradient: -A · ∇log p_current
    # Forward value: 1.0 * advantages (since exp(0) = 1)
    # Backward gradient: advantages · ∇log p_current
    per_token_policy_loss = torch.exp(logprobs - logprobs.detach()) * advantages

    # ✅ KL divergence penalty with numerical stability
    # Clamp to prevent extreme values while allowing meaningful divergence
    delta = (ref_logprobs - logprobs).clamp(-10, 10)
    kl = torch.exp(delta) - delta - 1

    # ✅ Clamp KL to prevent extreme values
    kl = kl.clamp(-20, 20)

    per_token_loss = -(per_token_policy_loss - beta * kl)

    # ✅ Loss clamping as a safety measure
    per_token_loss = per_token_loss.clamp(-100, 100)

    loss = (
        ((per_token_loss * padding_mask).sum(dim=1))
        / (padding_mask.sum(dim=1).clamp(min=1.0))
    ).mean()

    # Check for NaN/Inf in final loss
    if torch.isnan(loss) or torch.isinf(loss):
        print("WARNING: NaN/Inf detected in final loss!")
        loss = torch.tensor(0.0, device=loss.device, requires_grad=True)

    # ✅ Enhanced logging for debugging
    record_metric("loss/policy_loss", per_token_policy_loss.mean().item(), Reduce.MEAN)
    record_metric("loss/kl_penalty", (beta * kl).mean().item(), Reduce.MEAN)
    record_metric("loss/kl_raw", kl.mean().item(), Reduce.MEAN)
    record_metric("loss/total_loss", loss.item(), Reduce.MEAN)
    record_metric("loss/per_token_loss_mean", per_token_loss.mean().item(), Reduce.MEAN)
    record_metric("loss/per_token_loss_std", per_token_loss.std().item(), Reduce.MEAN)
    record_metric("loss/per_token_loss_min", per_token_loss.min().item(), Reduce.MEAN)
    record_metric("loss/per_token_loss_max", per_token_loss.max().item(), Reduce.MEAN)
    record_metric("loss/logprobs_mean", logprobs.mean().item(), Reduce.MEAN)
    record_metric("loss/ref_logprobs_mean", ref_logprobs.mean().item(), Reduce.MEAN)
    record_metric("loss/delta_mean", delta.mean().item(), Reduce.MEAN)

    return loss


@dataclass
class RewardActor(ForgeActor):
    """Reward actor that uses a list of scoring functions."""

    reward_functions: list[Callable]

    @endpoint
    async def evaluate_response(self, prompt: str, response: str, target: Any) -> float:
        total_rewards = 0.0
        for reward_fn in self.reward_functions:
            # Check if reward_fn is async (returns a coroutine)
            result = reward_fn(prompt, response, target)
            if asyncio.iscoroutine(result):
                reward = await result
            else:
                reward = result
            total_rewards += reward

            # Get a name for the reward function (works for classes, functions, lambdas)
            reward_fn_name = getattr(
                reward_fn, "__name__", reward_fn.__class__.__name__
            )
            # per function reward
            record_metric(
                f"reward/evaluate_response/sum_{reward_fn_name}_reward",
                reward,
                Reduce.SUM,
            )
            record_metric(
                f"reward/evaluate_response/avg_{reward_fn_name}_reward",
                reward,
                Reduce.MEAN,
            )
            record_metric(
                f"reward/evaluate_response/std_{reward_fn_name}_reward",
                reward,
                Reduce.STD,
            )

            # avg total reward
            record_metric(
                "reward/evaluate_response/avg_total_reward",
                reward,
                Reduce.MEAN,
            )

            # count fn calls
            record_metric(
                f"reward/evaluate_response/count_{reward_fn_name}_calls",
                1,
                Reduce.SUM,
            )

        avg_reward = total_rewards / len(self.reward_functions)
        return avg_reward


@dataclass
class ComputeAdvantages(ForgeActor):
    """Compute advantages for GRPO using reward signals with robust normalization."""

    @endpoint
    async def compute(self, group: Group) -> list[float]:
        # ✅ Get RAW rewards as 1D tensor [group_size]
        raw_rewards = torch.tensor([e.reward for e in group], dtype=torch.float32)

        # ✅ CRITICAL: Log group size to verify it's large enough
        group_size = len(group)
        record_metric("advantages/group_size", group_size, Reduce.MEAN)

        # Log raw reward statistics BEFORE any processing
        record_metric("rewards/raw_mean", raw_rewards.mean().item(), Reduce.MEAN)
        record_metric("rewards/raw_std", raw_rewards.std().item(), Reduce.MEAN)
        record_metric("rewards/raw_min", raw_rewards.min().item(), Reduce.MEAN)
        record_metric("rewards/raw_max", raw_rewards.max().item(), Reduce.MEAN)

        # ✅ WARN if std is too low (insufficient diversity)
        if raw_rewards.std().item() < 0.15:
            print(
                f"WARNING: Low reward variance (std={raw_rewards.std().item():.4f}) in group of size {group_size}"
            )
            print(f"         Rewards: {raw_rewards.tolist()}")
            print(
                f"         Consider increasing group_size for better learning signal!"
            )

        # Check for NaN/Inf in rewards
        if torch.isnan(raw_rewards).any() or torch.isinf(raw_rewards).any():
            print(f"WARNING: NaN/Inf detected in rewards: {raw_rewards}")
            raw_rewards = torch.nan_to_num(raw_rewards, nan=0.0, posinf=1.0, neginf=0.0)

        # ✅ Compute normalization statistics
        mean = raw_rewards.mean()
        std = raw_rewards.std(unbiased=False)

        # Log normalization parameters BEFORE normalization
        record_metric("advantages/normalization_mean_before", mean.item(), Reduce.MEAN)
        record_metric("advantages/normalization_std_before", std.item(), Reduce.MEAN)

        # ✅ Prevent division by very small std
        # Use max(0.01, std) to preserve signal when std is reasonable
        if std < 0.01:
            print(f"WARNING: Very low std ({std.item():.6f}), using minimum 0.01")
            std = torch.tensor(0.01)

        # ✅ Normalize advantages (mean should be ~0 after this)
        advantages = (raw_rewards - mean) / std

        # ✅ RELAXED CLIPPING - allow more signal through!
        # Only clip extreme outliers to prevent instability
        advantages = torch.clamp(advantages, min=-10.0, max=10.0)

        # Check for NaN/Inf in advantages
        if torch.isnan(advantages).any() or torch.isinf(advantages).any():
            print(f"WARNING: NaN/Inf detected in advantages: {advantages}")
            advantages = torch.nan_to_num(advantages, nan=0.0, posinf=0.0, neginf=0.0)

        # ✅ Log final advantage statistics (mean should be ~0!)
        record_metric("advantages/mean_after", advantages.mean().item(), Reduce.MEAN)
        record_metric("advantages/std_after", advantages.std().item(), Reduce.MEAN)
        record_metric("advantages/min_after", advantages.min().item(), Reduce.MEAN)
        record_metric("advantages/max_after", advantages.max().item(), Reduce.MEAN)
        record_metric(
            "advantages/range_after",
            (advantages.max() - advantages.min()).item(),
            Reduce.MEAN,
        )

        return advantages.tolist()


@dataclass
class DatasetActor(ForgeActor):
    """Actor wrapper for HuggingFace dataset to provide async interface."""

    path: str = "TIGER-Lab/AceCode-87K"
    revision: str = "main"
    data_split: str = "train"
    streaming: bool = True
    model: str = "Qwen/Qwen3-1.7B"

    @endpoint
    def setup(self):
        self._tokenizer = get_tokenizer(self.model)

        def get_coding_system_prompt():
            """Get system prompt for coding tasks."""
            return """IMPORTANT: You must structure your response in two parts:
1. First, write your detailed reasoning and problem-solving approach inside <think></think> tags
2. Then, provide your final solution after the thinking section

Given a problem description, follow this process:

<think>
- Carefully analyze the problem requirements and constraints
- Break down the problem into smaller components
- Consider multiple approaches and their trade-offs
- Think through edge cases and potential issues
- Plan your algorithm step by step
- Consider time and space complexity
- Think about how to test your solution thoroughly
</think>

Then write a Python function that solves the problem following these guidelines:
1. Write clean, readable, and efficient Python code
2. Add a comprehensive docstring explaining what the function does
3. Handle edge cases appropriately
4. Use only standard library imports unless specified otherwise
5. Ensure your solution is correct and robust

**ALLOWED STANDARD LIBRARY IMPORTS:**
- Core: sys, os, functools, typing, math, random, time, datetime, re, collections, itertools, statistics
- Data: json, csv, struct, base64, dataclasses, copy, heapq, enum
- Strings: string, ast, unicodedata
- Advanced: abc, contextlib, inspect, secrets, uuid, pathlib, io
- Async/Threading: threading, asyncio, concurrent.futures
- Network: socket, urllib.parse

**FORMAT YOUR RESPONSE AS:**

Format your response as:
<think>
[Your detailed reasoning, analysis, and planning here]
</think>

```python
def function_name(parameters):
    \"\"\"Comprehensive docstring explaining the function.\"\"\"
    # Implementation
    pass
```

Provide the final, working solution. Focus on correctness, readability, and efficiency."""

        #         def get_coding_system_prompt():
        #             """Get system prompt for coding tasks."""
        #             return """You are an expert Python programmer who writes clean, efficient, and well-tested code.

        # Given a problem description, write a Python function that solves it following these guidelines:

        # **CODE REQUIREMENTS:**
        # 1. **Write clean and efficient code**: Use clear variable names, proper structure, and Pythonic idioms
        # 2. **Include comprehensive docstrings**: Explain what the function does, parameters, return values, and any important notes
        # 3. **Handle edge cases**: Consider and appropriately handle boundary conditions and potential errors
        # 4. **Ensure correctness**: Your solution should be robust and handle all requirements

        # **CRITICAL RESTRICTIONS (Your code WILL FAIL if you violate these):**

        # **FORBIDDEN KEYWORDS:**
        # - NO `global` keyword (use function parameters/returns instead)
        # - NO `yield` keyword (no generators, use lists instead)
        # - NO `nonlocal` keyword (restructure your code to avoid it)

        # **FORBIDDEN OPERATIONS:**
        # - NO dunder attributes: `__dict__`, `__name__`, `__code__`, etc.
        # - NO dunder methods: `__contains__()`, etc. (use `in` operator instead)
        # - NO `input()` function (all inputs come from function parameters)
        # - NO `locals()` or `globals()` functions
        # - NO nested class definitions

        # **FILE OPERATIONS:**
        # - Use `pathlib` for file paths, NOT `os.path`
        # - Example: `from pathlib import Path; p = Path('/path/to/file')`

        # **ALLOWED STANDARD LIBRARY IMPORTS:**
        # - Core: sys, os, functools, typing, math, random, time, datetime, re, collections, itertools, statistics
        # - Data: json, csv, struct, base64, dataclasses, copy, heapq, enum
        # - Strings: string, ast, unicodedata
        # - Advanced: abc, contextlib, inspect, secrets, uuid, pathlib, io
        # - Async/Threading: threading, asyncio, concurrent.futures
        # - Network: socket, urllib.parse

        # **FORMAT YOUR RESPONSE AS:**

        # ```python
        # def function_name(parameters):
        #     \"\"\"Comprehensive docstring explaining the function.\"\"\"
        #     # Implementation here
        #     pass
        # ```

        # Provide the final, working solution. Focus on correctness, readability, and efficiency."""

        def transform_sample(sample):
            # AceCode format with OSS filtering
            if (
                sample.get("source") != "oss"
                or not sample.get("test_cases")
                or not isinstance(sample.get("test_cases"), list)
                or len(sample.get("test_cases", [])) == 0
            ):
                return None

            system_prompt = get_coding_system_prompt()
            request: str = sample.get("question", sample.get("prompt", ""))
            as_chat = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request},
            ]
            formatted_request = self._tokenizer.apply_chat_template(
                as_chat,
                tokenize=False,
                add_generation_prompt=True,
                # enable_thinking=True,
            )
            test_cases = sample.get("test_cases", [])
            return {
                "request": formatted_request,
                "target": test_cases,  # Use test cases as target for reward function
                "task_id": sample.get("id", ""),
                "source": sample.get("source"),
                "difficulty": sample.get("difficulty", "unknown"),
            }

        # Check if path is a local file
        import os

        if os.path.isfile(self.path) and self.path.endswith(".json"):
            # Load local JSON file
            ds = load_dataset(
                "json",
                data_files=self.path,
                split=self.data_split,
                streaming=self.streaming,
            )
        else:
            # Load from HuggingFace Hub or directory
            ds = load_dataset(
                self.path,
                split=self.data_split,
                streaming=self.streaming,
                revision=self.revision,
            )
        # Filter and transform to coding format
        ds = ds.filter(lambda x: transform_sample(x) is not None)
        ds = ds.map(transform_sample)
        ds = ds.shuffle()
        self._iterator = iter(ds)

    @endpoint
    async def sample(self) -> dict[str, str] | None:
        try:
            sample = next(self._iterator)

            # Record dataset metrics
            record_metric("dataset/sample/count_samples_generated", 1, Reduce.SUM)
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
        return self._tokenizer.pad_token_id


async def drop_weights(version: int):
    print(f"Dropping weights @ version {version}")
    start_time = time.perf_counter()
    prefix = get_param_prefix(version)
    matching_keys = await ts.keys(prefix)
    # TODO: once we have something like `get_meta()` in torchstore, we can just
    # query the type of the object instead of relying on keys.
    dcp_key = get_dcp_whole_state_dict_key(version)
    if dcp_key in matching_keys:
        dcp_handle = await ts.get(dcp_key)
        dcp_handle.drop()
    for key in matching_keys:
        await ts.delete(key)
    elapsed = time.perf_counter() - start_time
    print(f"Dropped weights @ version {version}, took {elapsed:.2f} seconds")


async def main(cfg: DictConfig):
    """Main GRPO training loop with rollout and training processes."""
    group_size = cfg.group_size
    max_req_tokens = cfg.max_req_tokens
    max_res_tokens = cfg.max_res_tokens

    # ---- Global setups ---- #
    provisioner = None
    if cfg.get("provisioner", None) is not None:
        provisioner = await init_provisioner(
            ProvisionerConfig(launcher_config=LauncherConfig(**cfg.provisioner))
        )
    else:
        provisioner = await init_provisioner()

    metric_logging_cfg = cfg.get("metric_logging", {"console": {"log_per_rank": False}})
    mlogger = await get_or_create_metric_logger()
    await mlogger.init_backends.call_one(metric_logging_cfg)

    # ---- Setup services ---- #

    # Setup coding environment with comprehensive standard library imports
    # Based on analysis of 143 numpy, 47 requests, 35 urllib.parse, 31 socket, 31 dataclasses import failures
    coder_actor = await OpenEnvCoder.as_actor(
        port=8432,  # Use default port to avoid command override bug in LocalDockerProvider
        additional_imports=[
            # Core (default)
            "sys",
            "os",
            "functools",
            "typing",
            # Data Science & Numerical
            "numpy",
            # Data Structures & Collections (31 dataclasses, 22 copy, 19 heapq, 17 enum)
            "dataclasses",
            "copy",
            "heapq",
            "enum",
            # String & Text Processing (22 string, 21 ast)
            "string",
            "ast",
            # Data Formats & Serialization (25 json, 15 struct, 10 base64, 5 csv)
            "json",
            "struct",
            "base64",
            "csv",
            # Math & Numbers (12 cmath)
            "cmath",
            # Abstract Base Classes & Patterns (16 abc, 7 contextlib, 7 inspect)
            "abc",
            "contextlib",
            "inspect",
            # Security & Utilities (16 secrets, 4 uuid)
            "secrets",
            "uuid",
            # I/O & Path Operations (6 pathlib, 5 io)
            "pathlib",
            "io",
            # Async & Concurrency (11 threading, 6 asyncio, 3 concurrent.futures)
            "threading",
            "asyncio",
            "concurrent.futures",
            # Network & Web (35 urllib.parse, 31 socket)
            "urllib.parse",
            "socket",
        ],
    )

    # Setup coding reward functions
    ground_truth_reward = GroundTruthTestReward(coder_actor)
    thinking_reward = ThinkingReward()

    (
        dataloader,
        policy,
        trainer,
        replay_buffer,
        compute_advantages,
        ref_model,
        reward_actor,
    ) = await asyncio.gather(
        DatasetActor.options(**cfg.actors.dataset).as_actor(**cfg.dataset),
        Policy.options(**cfg.services.policy).as_service(**cfg.policy),
        RLTrainer.options(**cfg.actors.trainer).as_actor(
            **cfg.trainer, loss=simple_grpo_loss
        ),
        ReplayBuffer.options(**cfg.actors.replay_buffer).as_actor(
            **cfg.replay_buffer, collate=collate
        ),
        ComputeAdvantages.options(**cfg.actors.compute_advantages).as_actor(),
        ReferenceModel.options(**cfg.services.ref_model).as_service(**cfg.ref_model),
        RewardActor.options(**cfg.services.reward_actor).as_service(
            reward_functions=[ground_truth_reward, thinking_reward]
        ),
    )

    # Set max_steps to the configured value, or -1 if not specified or Null
    max_steps = cfg.trainer.training.steps or -1

    print("All services initialized successfully!")
    shutdown_event = asyncio.Event()
    # Here we spawn a torchstore storage volume per trainer process.
    # We initialize after service initialization because torchstore currently
    # requires access to the underlying proc meshes in the local rank strategy.
    # We should be able to hide this in the future.
    # TODO: support multiple host meshes
    trainer_num_procs = cfg.actors.trainer["procs"]
    trainer_host_mesh_name = cfg.actors.trainer["mesh_name"]
    trainer_hosts = provisioner.get_host_mesh(trainer_host_mesh_name)
    await ts.initialize(
        mesh=trainer_hosts.spawn_procs(per_host={"procs": trainer_num_procs}),
        strategy=ts.LocalRankStrategy(),
    )
    print("Torchstore successfully initialized with local rank strategy")

    # ---- Core RL loops ---- #
    async def continuous_rollouts():
        rollout_count = 0
        pad_id = await dataloader.pad_token.call_one()
        while not shutdown_event.is_set():
            try:
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

                # Construct episodes and calculate rewards
                episodes = []
                input_ids = torch.ones(
                    (group_size, max_req_tokens + max_res_tokens),
                    dtype=torch.long,
                )
                for i, response in enumerate(responses):
                    episode = Episode(
                        episode_id=str(uuid.uuid4()),
                        pad_id=pad_id,
                        request_len=max_req_tokens,
                        response_len=max_res_tokens,
                        target=target,
                        completion=response,
                    )
                    episode.reward = await reward_actor.evaluate_response.route(
                        prompt=prompt, response=response.text, target=target
                    )
                    episodes.append(episode)

                    # Build input_ids for reference logprobs
                    input_ids[i, :max_req_tokens] = episode.request_tensor
                    input_ids[i, max_req_tokens:] = episode.response_tensor

                t.step("reward_evaluation")

                ref_logprobs = await ref_model.forward.route(
                    input_ids, max_req_tokens, return_logprobs=True
                )
                t.step("reference_model_calculate_logprobs")

                for i, episode in enumerate(episodes):
                    episode.ref_logprobs = ref_logprobs[i]
                del ref_logprobs, input_ids

                # Calculate advantages and add to replay buffer
                advantages = await compute_advantages.compute.call_one(episodes)
                for episode, advantage in zip(episodes, advantages):
                    episode.advantage = advantage
                    await replay_buffer.add.call_one(episode)

                # Log metrics
                rollout_count += 1
                record_metric(
                    "main/continuous_rollouts/count_rollout_iterations", 1, Reduce.SUM
                )
                t.stop()
            except RuntimeError as e:
                error_msg = str(e).lower()
                # Check if this is a container-related error that couldn't be recovered
                if any(
                    keyword in error_msg
                    for keyword in [
                        "container failed",
                        "could not be recreated",
                        "execution failed after all retry attempts",
                    ]
                ):
                    print("=" * 80)
                    print("CRITICAL ERROR: Container recreation failed")
                    print(f"Error: {e}")
                    print("Triggering graceful shutdown of training...")
                    print("=" * 80)
                    shutdown_event.set()
                    return
                else:
                    # Re-raise non-container errors
                    raise
            except Exception as e:
                print(f"Unexpected error in continuous_rollouts: {e}")
                print("Triggering graceful shutdown of training...")
                shutdown_event.set()
                raise

    async def continuous_training():
        training_step = 0
        restart_tracer = True  # Flag to control when to restart tracer

        while max_steps == -1 or training_step < max_steps:
            # Restart tracer when needed (initial start or after completing a training step)
            # Otherwise, we cannot measure time waiting for buffer
            if restart_tracer:
                t = Tracer("main_perf/continuous_training")
                t.start()
                restart_tracer = False

            batch = await replay_buffer.sample.call_one(
                curr_policy_version=training_step
            )
            if batch is None:
                await asyncio.sleep(0.1)
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

                # Flush metrics every training step to WandB
                await mlogger.flush.call_one(training_step)

        print(
            f"Reached training limit ({max_steps} steps). Exiting continuous_training loop."
        )

    num_rollout_threads = cfg.get("rollout_threads", 1)
    num_training_threads = cfg.get("training_threads", 1)
    print(
        f"Starting GRPO with {num_rollout_threads} rollout threads, {num_training_threads} training threads"
    )
    rollout_tasks = [
        asyncio.create_task(continuous_rollouts()) for _ in range(num_rollout_threads)
    ]
    training_task = asyncio.create_task(continuous_training())

    try:
        await training_task
    except KeyboardInterrupt:
        print("Training interrupted by user")
    finally:
        print("Shutting down...")
        shutdown_event.set()

        try:
            # Give rollouts up to 5s to finish naturally
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

        await shutdown()


if __name__ == "__main__":

    @parse
    def _main(cfg):
        """Main entry point for GRPO training."""
        os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
        os.environ["NCCL_TIMEOUT_MS"] = "60000"  # 60 second timeout
        os.environ["MONARCH_HOSTMESH_V1"] = "1"
        os.environ["TORCHSTORE_RDMA_ENABLED"] = "1"
        # os.environ["FORGE_DISABLE_METRICS"] = "1"
        asyncio.run(main(cfg))

    _main()  # @parse grabs the cfg from CLI
