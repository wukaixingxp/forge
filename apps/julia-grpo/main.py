# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Usage: python -m apps.julia-grpo.main --config apps/julia-grpo/julia_config.yaml

import asyncio
import time
import uuid
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
import torchstore as ts
from datasets import load_dataset
from envs.julia_env import JuliaAction, JuliaEnv
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
        return self.completion.generator_version if self.completion else None

    @property
    def request_tensor(self) -> torch.Tensor:
        tensor: torch.Tensor = self.completion.prompt_ids.to(torch.long)
        if tensor.shape[0] < self.request_len:  # left pad
            diff = self.request_len - tensor.shape[0]
            tensor = F.pad(tensor, (diff, 0), value=self.pad_id)
        return tensor

    @property
    def response_tensor(self) -> torch.Tensor:
        tensor: torch.Tensor = self.completion.token_ids.to(torch.long)
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
    """
    Collates a list of batches into a single batch of inputs and targets.
    Each batch is a list of episodes, and each episode is a dict of tensors.
    """
    inputs = []
    targets = []
    for batch in batches:
        request = [e.request_tensor for e in batch]
        request = torch.stack(request)  # [b x s]

        response = [e.response_tensor for e in batch]
        response = torch.stack(response)  # [b x s]

        ref_logprobs = [e.ref_logprobs for e in batch]
        ref_logprobs = torch.stack(ref_logprobs).squeeze()  # [b x s]

        advantages = [e.advantage for e in batch]
        advantages = torch.tensor(advantages).unsqueeze(-1)  # [b x 1]

        pad_id = batch[0].pad_id
        mask = response != pad_id

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
    beta: float = 0.1,
) -> torch.Tensor:
    logprobs: torch.Tensor = compute_logprobs(logits, response)
    kl = torch.exp(ref_logprobs - logprobs) - (ref_logprobs - logprobs) - 1
    per_token_policy_loss = torch.exp(logprobs - logprobs.detach()) * advantages
    per_token_loss = -(per_token_policy_loss - beta * kl)
    loss = (
        ((per_token_loss * padding_mask).sum(dim=1))
        / (padding_mask.sum(dim=1).clamp(min=1.0))
    ).mean()
    return loss


@dataclass
class JuliaRewardActor(ForgeActor):
    """Reward actor for Julia code execution using GenericOpenEnvActor."""

    julia_env: GenericOpenEnvActor

    @endpoint
    async def evaluate_response(
        self, prompt: str, response: str, target: dict
    ) -> float:
        """
        Evaluate Julia code by executing it with test cases.

        Args:
            prompt: The problem description (not used directly, but available)
            response: The Julia code to evaluate
            target: Dict containing test cases and expected outputs

        Returns:
            Reward score based on test case pass rate
        """
        try:
            # Extract code from markdown code blocks if present
            code = self._extract_code(response)

            # Get test cases from target
            test_cases = target.get("test_cases", [])
            if not test_cases:
                record_metric("reward/julia/no_test_cases", 1, Reduce.SUM)
                return 0.0

            # Execute code with test cases using JuliaAction
            action = JuliaAction(
                code=code,
                test_cases=test_cases,
            )

            result = await self.julia_env.execute.route(action)

            # Calculate reward based on test results
            obs = result.observation
            passed = obs.tests_passed
            total = obs.tests_total

            if total == 0:
                reward = 0.0
            else:
                # Pass rate as reward (0.0 to 1.0)
                reward = passed / total

            # Log metrics
            record_metric("reward/julia/tests_passed", passed, Reduce.SUM)
            record_metric("reward/julia/tests_total", total, Reduce.SUM)
            record_metric("reward/julia/pass_rate", reward, Reduce.MEAN)

            if obs.stderr:
                record_metric("reward/julia/has_errors", 1, Reduce.SUM)

            return reward

        except Exception as e:
            print(f"Error evaluating Julia response: {e}")
            record_metric("reward/julia/evaluation_errors", 1, Reduce.SUM)
            return 0.0

    def _extract_code(self, response: str) -> str:
        """Extract Julia code from markdown code blocks."""
        # Remove markdown code fences if present
        if "```julia" in response:
            start = response.find("```julia") + len("```julia")
            end = response.find("```", start)
            if end != -1:
                return response[start:end].strip()
        elif "```" in response:
            start = response.find("```") + len("```")
            end = response.find("```", start)
            if end != -1:
                return response[start:end].strip()
        return response.strip()


@dataclass
class ComputeAdvantages(ForgeActor):
    @endpoint
    async def compute(self, group: Group) -> list[float]:
        rewards = torch.tensor([[e.reward for e in group]])
        mean = rewards.mean(1, keepdim=True)
        std = rewards.std(1, keepdim=True)
        advantages = (rewards - mean) / (std + 1e-4)
        return advantages.squeeze(0).tolist()


@dataclass
class JuliaDatasetActor(ForgeActor):
    """Actor wrapper for Julia dataset to provide async interface."""

    path: str = "/home/kaiwu/work/amd/amd-submission/julia_trainset.parquet"
    revision: str = "main"
    data_split: str = "train"
    streaming: bool = False
    model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    @endpoint
    def setup(self):
        self._tokenizer = get_tokenizer(self.model)

        def get_julia_code_gen_prompt():
            """Get system prompt for Julia coding tasks."""
            return """You are a precise and pragmatic Julia programmer.

Write a **single Julia function** that correctly solves the problem described below.

Rules:
- The code must be syntactically correct and runnable as is.
- Do not use arrow functions, ternary operators, or modern syntax that may cause issues.
- Use only the Julia standard library.
- Do **not** wrap the code in a module or add a `main` function.
- Do **not** include any test code in your response.
- Do **not** hardcode specific test cases or outputs — the function must work for general inputs.
- The **function name must exactly match** the one used in the provided tests.
- Respond with **only the Julia function** and nothing else (no explanations, no comments, no extra text)
- The function name must exactly match the one used in the provided tests.
- Return only the Julia function.
- character literal should not contain multiple characters.
- take care of object types and mind that spaces matter in julia so cannot add random spaces

Passing tests and clean, compilable code are rewarded. Hardcoding or failing tests is penalized.
FORMAT YOUR RESPONSE AS:

```julia
function <function_name>(<argument_list>)
    <function_body>
end
```
""".strip()

        def transform_sample(sample):
            # Julia dataset format
            if not sample.get("julia_test") or not sample.get("first_test_case"):
                return None

            # julia_test = sample.get("first_test_case", "")
            system_prompt = get_julia_code_gen_prompt()
            request: str = sample.get("julia_prompt", "")

            as_chat = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request},
            ]
            formatted_request = self._tokenizer.apply_chat_template(
                as_chat,
                tokenize=False,
                add_generation_prompt=True,
            )

            return {
                "request": formatted_request,
                "target": sample.get(
                    "julia_test", ""
                ),  # Full test code for reward function
                "task_id": sample.get("task_id", ""),
            }

        # Check if path is a local file
        import os

        import pandas as pd

        if os.path.isfile(self.path) and self.path.endswith(".parquet"):
            # Load local Parquet file
            df = pd.read_parquet(self.path)
            df = df[["julia_prompt", "julia_test", "first_test_case", "task_id"]]
            ds = load_dataset(
                "parquet",
                data_files={"train": self.path},
                split=self.data_split,
            )
        else:
            # Load from HuggingFace Hub or directory
            ds = load_dataset(
                self.path,
                split=self.data_split,
                streaming=self.streaming,
                revision=self.revision,
            )

        # Filter and transform to Julia format
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
    dcp_key = get_dcp_whole_state_dict_key(version)
    if dcp_key in matching_keys:
        dcp_handle = await ts.get(dcp_key)
        dcp_handle.drop()
    for key in matching_keys:
        await ts.delete(key)
    elapsed = time.perf_counter() - start_time
    print(f"Dropped weights @ version {version}, took {elapsed:.2f} seconds")


async def main(cfg: DictConfig):
    """Main GRPO training loop for Julia code generation."""
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

    metric_logging_cfg = cfg.get("metric_logging", {})
    mlogger = await get_or_create_metric_logger(process_name="Controller")
    await mlogger.init_backends.call_one(metric_logging_cfg)

    # ---- Setup services ---- #

    # Setup Julia environment using GenericOpenEnvActor with JuliaEnv
    # This actor provides a sandboxed Julia execution environment via OpenEnv.
    # Get docker image and env vars from config, with sensible defaults
    openenv_config = cfg.get("openenv_config", {})
    docker_image = openenv_config.get("docker_image", "julia-env:latest")
    env_vars = openenv_config.get("env_vars", {})
    container_timeout_s = openenv_config.get("container_timeout_s", 180.0)
    request_timeout_s = openenv_config.get("request_timeout_s", 120.0)
    container_memory_gb = openenv_config.get("container_memory_gb", 4)

    julia_env_actor = await GenericOpenEnvActor.options(
        **cfg.actors.julia_env
    ).as_actor(
        env_class=JuliaEnv,
        action_class=JuliaAction,
        docker_image=docker_image,
        env_vars=env_vars,
        container_timeout_s=container_timeout_s,
        request_timeout_s=request_timeout_s,
        container_memory_gb=container_memory_gb,
    )

    (
        dataloader,
        policy,
        trainer,
        replay_buffer,
        compute_advantages,
        ref_model,
        reward_actor,
    ) = await asyncio.gather(
        JuliaDatasetActor.options(**cfg.actors.dataset).as_actor(**cfg.dataset),
        Policy.options(**cfg.services.policy).as_service(**cfg.policy),
        RLTrainer.options(**cfg.actors.trainer).as_actor(
            **cfg.trainer, loss=simple_grpo_loss
        ),
        ReplayBuffer.options(**cfg.actors.replay_buffer).as_actor(
            **cfg.replay_buffer, collate=collate
        ),
        ComputeAdvantages.options(**cfg.actors.compute_advantages).as_actor(),
        ReferenceModel.options(**cfg.services.ref_model).as_service(**cfg.ref_model),
        JuliaRewardActor.options(**cfg.services.reward_actor).as_service(
            julia_env=julia_env_actor
        ),
    )

    # Set max_steps to the configured value, or -1 if not specified or Null
    max_steps = cfg.trainer.training.steps or -1

    print("All services initialized successfully!")
    shutdown_event = asyncio.Event()

    # Initialize torchstore
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

        while max_steps == -1 or training_step < max_steps:
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

                # Flush metrics every training step
                await mlogger.flush.call_one(training_step)

        print(
            f"Reached training limit ({max_steps} steps). Exiting continuous_training loop."
        )

    num_rollout_threads = cfg.get("rollout_threads", 1)
    num_training_threads = cfg.get("training_threads", 1)
    print(
        f"Starting Julia GRPO with {num_rollout_threads} rollout threads, {num_training_threads} training threads"
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
        print(f"Training failed with error: {e}")
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

        # Explicitly tear down the Julia environment actor to kill the Docker container
        print("Cleaning up Julia environment Docker container...")
        try:
            await julia_env_actor.teardown.call_one()
            print("Julia environment Docker container stopped successfully")
        except Exception as teardown_error:
            print(f"Warning: Error during Julia environment teardown: {teardown_error}")

        await shutdown()


if __name__ == "__main__":

    @parse
    def _main(cfg):
        asyncio.run(main(cfg))

    _main()  # @parse grabs the cfg from CLI
