# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Usage: python -m tests.sandbox.toy_rl.sumdigits --config tests/sandbox/toy_rl/sumdigits.yaml

import asyncio
import random
import uuid
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
import torchstore as ts
from forge.actors._torchstore_utils import get_param_key
from forge.actors.generator import Generator
from forge.actors.replay_buffer import ReplayBuffer
from forge.controller.actor import ForgeActor
from forge.controller.provisioner import shutdown
from forge.losses.grpo_loss import SimpleGRPOLoss
from forge.observability.metric_actors import get_or_create_metric_logger

from forge.observability.metrics import record_metric, Reduce
from forge.util.config import parse
from forge.util.ops import selective_log_softmax
from monarch.actor import endpoint
from omegaconf import DictConfig

from transformers import AutoModelForCausalLM
from vllm.transformers_utils.tokenizer import get_tokenizer


def pad_sequence(
    tensor: torch.Tensor, target_len: int, pad_value: float = 0.0
) -> torch.Tensor:
    diff = target_len - tensor.size(0)
    if diff > 0:
        return F.pad(tensor, (0, diff), value=pad_value)
    return tensor


# TODO: Episode and Group and duplicated and needs clean up.
@dataclass
class Episode:
    # TODO: add adtional layer for multi-turn
    episode_id: str
    request: str
    policy_version: int
    pad_id: int
    request_len: int
    response_len: int
    target: Any | None = None
    # processed data
    response: str | None = None
    request_tokens: list[int] | None = None
    response_tokens: list[int] | None = None
    ref_logprobs: torch.Tensor | None = None
    reward: float | None = None
    advantage: float | None = None
    response_logprobs: torch.Tensor | None = None

    @property
    def max_seq_len(self) -> int:
        """
        Get maximum sequence length for this episode.

        Returns:
            int: Total length (request_len + response_len) before any truncation
        """
        return self.request_len + self.response_len

    @property
    def episode_ids(self) -> torch.Tensor:
        """
        Get complete episode trajectory as concatenated token sequence.

        Returns:
            torch.Tensor: Full sequence [request_tokens + response_tokens].
                         Shape: [request_len + response_len]
        """
        prompt_ids = torch.LongTensor(self.request_tokens)
        token_ids = torch.LongTensor(self.response_tokens)
        ids = torch.cat([prompt_ids, token_ids])
        return ids

    @property
    def input_ids(self) -> torch.Tensor:
        """
        Get model input tokens for next-token prediction.

        Returns:
            torch.Tensor: Episode trajectory with EOS truncated for model input.
                         Shape: [max_seq_len - 1]
        """
        input_ids = self.episode_ids[:-1]  # truncate EOS
        return input_ids

    @property
    def target_ids(self) -> torch.Tensor:
        """
        Get target tokens for next-token prediction training.

        Returns:
            torch.Tensor: Episode trajectory shifted by 1 position (BOS truncated).
                         Aligned with input_ids for teacher forcing.
                         Shape: [max_seq_len - 1]
        """
        target_ids = self.episode_ids[1:]  # truncate BOS
        return target_ids

    @property
    def loss_mask(self) -> torch.Tensor:
        """
        Get mask for computing loss only on response tokens.

        Returns:
            torch.Tensor: Binary mask (0 for prompt, 1 for response) shifted to align
                         with target_ids. Shape: [max_seq_len - 1]
        """
        prompt_ids = torch.LongTensor(self.request_tokens)
        token_ids = torch.LongTensor(self.response_tokens)
        loss_mask = torch.cat(
            [
                torch.zeros(
                    len(prompt_ids), dtype=torch.float32
                ),  # Don't compute loss on prompt
                torch.ones(
                    len(token_ids), dtype=torch.float32
                ),  # Compute loss on response
            ]
        )

        loss_mask = loss_mask[1:]  # Shift to align with target_ids (truncates BOS)
        return loss_mask

    @property
    def sampling_log_probs(self) -> torch.Tensor:
        """
        Get log probabilities from the sampling policy (for importance sampling).

        Returns:
            torch.Tensor: Log probabilities from policy that generated the response,
                         with zeros for prompt positions. Shifted to align with target_ids.
                         Shape: [max_seq_len - 1]
        """
        if self.response_logprobs is None:
            return torch.zeros(self.max_seq_len - 1, dtype=torch.float32)
        prompt_ids = torch.LongTensor(self.request_tokens)
        sampling_log_probs = torch.cat(
            [
                torch.zeros(prompt_ids.shape, dtype=torch.float32),
                self.response_logprobs,
            ]
        )
        sampling_log_probs = sampling_log_probs[1:]  # Shift log probs
        return sampling_log_probs

    @property
    def weighted_advantages(self) -> torch.Tensor:
        """
        Get advantages weighted by loss mask for REINFORCE training.

        Returns:
            torch.Tensor: Advantage values masked to response tokens only.
                         Zero for prompt positions, advantage value for response positions.
                         Shape: [max_seq_len - 1]
        """
        if self.advantage is None:
            return torch.zeros_like(self.loss_mask)
        return self.loss_mask * self.advantage


@dataclass
class Group:
    group_id: str
    episodes: list[Episode]

    @classmethod
    def new_group(
        cls,
        group_id: int,
        group_size: int,
        request: str,
        policy_version: int,
        pad_id: int,
        request_len: int,
        response_len: int,
        target: Any = None,
    ):
        episodes = []
        for _ in range(group_size):
            episodes.append(
                Episode(
                    episode_id=str(uuid.uuid4()),
                    request=request,
                    policy_version=policy_version,
                    pad_id=pad_id,
                    request_len=request_len,
                    response_len=response_len,
                    target=target,
                )
            )
        return cls(str(group_id), episodes)


class RefModel(ForgeActor):
    def __init__(self, model_name, device: torch.device | None = None):
        super().__init__()
        self.model_name = model_name

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()

        self.logger.info(f"Model initialized on {self.device}")

    @endpoint
    async def forward(self, episode: Episode) -> torch.Tensor:
        input_ids = (
            pad_sequence(episode.input_ids, episode.max_seq_len - 1, episode.pad_id)
            .to(self.device)
            .unsqueeze(0)
        )
        target_ids = (
            pad_sequence(episode.target_ids, episode.max_seq_len - 1, episode.pad_id)
            .to(self.device)
            .unsqueeze(0)
        )
        mask = input_ids != episode.pad_id

        with torch.inference_mode():
            logits = self.model(input_ids=input_ids, attention_mask=mask).logits

        return selective_log_softmax(logits, target_ids).squeeze(0)


@dataclass
class Trainer(ForgeActor):
    """Reinforce Loss Trainer implementation for policy optimization."""

    model_name: str = ""
    learning_rate: float = 1e-5
    device: torch.device | None = None
    state_dict_key: str = "model_state_dict"

    def __post_init__(self):
        super().__init__()

    @endpoint
    async def setup(self):
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        ).to(self.device)
        self.model.train()

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.learning_rate
        )
        self.optimizer.zero_grad()

        # beta = 0.01 for quicker convergence
        self.loss = SimpleGRPOLoss(0.01)
        self.logger.info(f"Trainer model initialized on {self.device}")

    @endpoint
    def train_step(self, episodes: list[Episode]) -> float:
        pad_id = episodes[0].pad_id

        # Calculate batch maximum length
        max_seq_len = max(ep.max_seq_len - 1 for ep in episodes)
        batch_input_ids = []
        batch_target_ids = []
        batch_loss_masks = []
        batch_weights = []
        batch_sampling_log_probs = []
        batch_ref_logprobs = []
        for episode in episodes:
            input_ids = pad_sequence(episode.input_ids, max_seq_len, pad_id)
            target_ids = pad_sequence(episode.target_ids, max_seq_len, pad_id)
            loss_mask = pad_sequence(episode.loss_mask, max_seq_len, 0.0)
            sampling_log_probs = pad_sequence(
                episode.sampling_log_probs, max_seq_len, 0.0
            )
            weights = pad_sequence(episode.weighted_advantages, max_seq_len, 0.0)
            ref_logprobs = episode.ref_logprobs

            # Exclude padded response tokens from loss
            valid_mask = target_ids != pad_id
            loss_mask = loss_mask * valid_mask.float()
            weights = weights * valid_mask.float()
            sampling_log_probs = sampling_log_probs * valid_mask.float()

            batch_input_ids.append(input_ids)
            batch_target_ids.append(target_ids)
            batch_loss_masks.append(loss_mask)
            batch_weights.append(weights)
            batch_sampling_log_probs.append(sampling_log_probs)
            batch_ref_logprobs.append(ref_logprobs)

        # Stack into batched tensors
        input_ids = torch.stack(batch_input_ids).to(self.device)
        target_ids = torch.stack(batch_target_ids).to(self.device)
        loss_masks = torch.stack(batch_loss_masks).to(self.device)
        weights = torch.stack(batch_weights).to(self.device)
        sampling_log_probs = torch.stack(batch_sampling_log_probs).to(self.device)
        ref_logprobs = torch.stack(batch_ref_logprobs).to(self.device)

        # Create attention mask
        attention_mask = input_ids != pad_id

        # Forward pass
        logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits

        trainer_log_probs = selective_log_softmax(logits, target_ids)
        # Compute loss only on response tokens
        # loss = self.loss(logits, target_ids, loss_masks, weights, sampling_log_probs)
        loss = self.loss(trainer_log_probs, ref_logprobs, weights, loss_masks)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        return loss.item()

    @endpoint
    async def push_weights(self, policy_version: int) -> None:
        """Push weights to torchstore in HF format."""
        hf_state_dict = self.model.state_dict()
        for name, param in hf_state_dict.items():
            key = get_param_key(policy_version, name)
            await ts.put(key, param)


@dataclass
class RewardActor(ForgeActor):
    """Reward actor that uses a list of scoring functions."""

    @endpoint
    async def evaluate_response(self, prompt: str, response: str, target: str) -> float:
        reward = 1.0 if response.strip() == target else 0.0
        return reward


@dataclass
class SumDigitsDataset:
    def __init__(self, tokenizer, max_samples=1000):
        self.max_numbers = max_samples
        self._tokenizer = tokenizer

    def generate_sample(self, step: int) -> dict[str, str]:
        """Generate a single sample based on training step for progressive difficulty."""
        data = self.generate_one(step)
        answer = str(sum(int(x) for x in data))

        system_prompt = """
        A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
        The assistant only gives very concise answers (just the number, no explanation).
        """
        request: str = f"What is the sum of the digits of {data}"
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
            "question": formatted_request,
            "request": formatted_request,
            "answer": answer,
            "target": answer,
        }

    def generate_one(self, step: int) -> str:
        """Generate number based on training step for curriculum learning."""
        min_val, max_val = 10, 100

        number = random.randint(min_val, max_val)
        return str(number)


@dataclass
class DatasetActor(ForgeActor):
    """Actor wrapper for HuggingFace dataset to provide async interface."""

    model: str = "Qwen/Qwen2.5-0.5B-Instruct"

    @endpoint
    def setup(self):
        self._tokenizer = get_tokenizer(self.model)
        self._dataset = SumDigitsDataset(self._tokenizer)

    @endpoint
    async def sample(self, step: int = 0) -> dict[str, str] | None:
        """Sample with progressive difficulty based on training step."""
        try:
            return self._dataset.generate_sample(step)
        except Exception as e:
            self.logger.error(f"Error generating sample: {e}")
            return None

    @endpoint
    async def pad_token(self):
        return self._tokenizer.pad_token_id


async def main(cfg: DictConfig):
    """Main Sumgits app training loop with rollout and training processes."""
    # Get parameters from config with fallbacks
    group_size = cfg.group_size
    max_req_tokens = cfg.max_req_tokens
    max_res_tokens = cfg.max_res_tokens

    # ---- Setup services ---- #
    print(f"{cfg.policy=}")
    print(f"{cfg.services.policy=}")

    metric_logging_cfg = cfg.get("metric_logging", {"console": {"log_per_rank": False}})
    mlogger = await get_or_create_metric_logger()
    await mlogger.init_backends.call_one(metric_logging_cfg)
    await ts.initialize()
    (
        dataloader,
        policy,
        trainer,
        replay_buffer,
        reward_actor,
        ref_model,
    ) = await asyncio.gather(
        DatasetActor.options(**cfg.actors.dataset).as_actor(**cfg.dataset),
        Generator.options(**cfg.services.policy).as_service(**cfg.policy),
        Trainer.options(**cfg.actors.trainer).as_actor(**cfg.trainer),
        ReplayBuffer.options(**cfg.actors.replay_buffer).as_actor(**cfg.replay_buffer),
        RewardActor.options(**cfg.services.reward_actor).as_service(),
        RefModel.options(**cfg.services.ref_model).as_service(**cfg.ref_model),
    )

    print("All services initialized successfully!")

    # ---- Core RL loops ---- #
    async def continuous_rollouts():
        rollout_count = 0
        pad_id = await dataloader.pad_token.call_one()
        while True:
            # Pass rollout_count for curriculum learning
            sample = await dataloader.sample.call_one(rollout_count)
            if sample is None:
                print("Dataloader is empty, exiting continuous rollout")
                return
            prompt, target = sample["request"], sample["target"]
            responses = await policy.generate.route(prompt)
            assert len(responses) > 0
            version = responses[0].generator_version
            assert version is not None, "Response must indicate a version"
            group = Group.new_group(
                group_id=rollout_count,
                group_size=group_size,
                request=prompt,
                policy_version=version,
                pad_id=pad_id,
                request_len=max_req_tokens,
                response_len=max_res_tokens,
                target=target,
            )

            # TODO: Parallelize the following calculation
            for episode, response in zip(group.episodes, responses):
                episode.request_tokens = response.prompt_ids
                episode.response_tokens = response.token_ids
                episode.response = response.text
                episode.response_logprobs = response.logprobs
                episode.ref_logprobs = await ref_model.forward.route(episode)
                episode.reward = await reward_actor.evaluate_response.route(
                    prompt=prompt, response=response.text, target=target
                )
                episode.advantage = episode.reward  # simple case for now
            for episode in group.episodes:
                await replay_buffer.add.call_one(episode)
            avg_response_len = (
                sum(len(e.response_tokens) for e in group.episodes) / group_size
            )
            record_metric("avg_response_len/rollout", avg_response_len, Reduce.MEAN)
            avg_reward = sum(e.reward for e in group.episodes) / group_size
            record_metric("avg_reward/rollout", avg_reward, Reduce.MEAN)

            rollout_count += 1

    async def continuous_training():
        training_step = 0
        while True:
            batch = await replay_buffer.sample.call_one(
                curr_policy_version=training_step
            )
            if batch is None:
                await asyncio.sleep(0.1)
            else:
                loss = await trainer.train_step.call_one(batch[0])
                training_step += 1
                record_metric("loss/training_step", loss, Reduce.MEAN)
                print(f"loss/training_step: {loss} at training step {training_step}")
                await trainer.push_weights.call(training_step)
                await policy.update_weights.fanout(training_step)
                # NOTE: hard-coded to be on-policy for faster convergence
                await replay_buffer.clear.call()

    print("Starting training loop.")
    # TODO: Start multiple rollouts once all serivces support it
    rollout_task = asyncio.create_task(continuous_rollouts())
    training_task = asyncio.create_task(continuous_training())

    try:
        await asyncio.gather(rollout_task, training_task)
    except KeyboardInterrupt:
        print("Training interrupted by user")
        rollout_task.cancel()
        training_task.cancel()
    finally:
        print("Shutting down...")
        await shutdown()


if __name__ == "__main__":

    @parse
    def _main(cfg):
        asyncio.run(main(cfg))

    _main()  # @parse grabs the cfg from CLI
