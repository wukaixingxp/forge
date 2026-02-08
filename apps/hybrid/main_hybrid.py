# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Usage: python -m apps.grpo.main_hybrid --config apps/hybrid/simple_kv_cache.yaml

import asyncio
import traceback
import uuid

import torch
import yaml
from apps.grpo.data import DatasetActor
from apps.grpo.grading import MathReward, ThinkingReward
from forge.actors.hybrid.policy_actor import HybridPolicyActor
from forge.actors.replay_buffer import ReplayBuffer
from forge.controller.provisioner import init_provisioner, shutdown
from forge.data_models.completion import Completion
from forge.observability.metric_actors import get_or_create_metric_logger
from forge.observability.metrics import record_metric, Reduce
from forge.observability.perf_tracker import Tracer
from forge.rl import collate, ComputeAdvantages, Episode, RewardActor
from forge.rl.loss import DAPOLoss
from forge.types import LauncherConfig, ProvisionerConfig
from forge.util.config import parse
from forge.util.logging import get_logger
from omegaconf import DictConfig, OmegaConf

logger = get_logger("INFO")


async def main(cfg: DictConfig):
    """Main GRPO training loop with hybrid policy actor (training + inference combined)."""
    # Convert OmegaConf config to plain dict
    run_config_for_logging = OmegaConf.to_container(cfg, resolve=True)

    # Log config
    logger.info("=" * 30 + " CONFIGURATION " + "=" * 30)
    logger.info(
        yaml.dump(run_config_for_logging, default_flow_style=False, sort_keys=False)
    )

    # ---- Global setups ---- #
    provisioner = None
    if cfg.get("provisioner", None) is not None:
        # Create launcher config with services and actors for pre-allocation
        launcher_config = LauncherConfig(
            **cfg.provisioner,
            services=cfg.get("services", {}),
            actors=cfg.get("actors", {}),
        )
        provisioner = await init_provisioner(
            ProvisionerConfig(launcher_config=launcher_config)
        )
    else:
        provisioner = await init_provisioner()

    metric_logging_cfg = cfg.get("metric_logging", {})

    mlogger = await get_or_create_metric_logger(process_name="Controller")
    await mlogger.init_backends.call_one(
        backend_config=metric_logging_cfg, run_config=run_config_for_logging
    )

    # ---- Setup loss function ---- #
    loss_fn = DAPOLoss()

    # ---- Setup actors ---- #
    (
        dataloader,
        hybrid_policy,
        replay_buffer,
        compute_advantages,
        reward_actor,
    ) = await asyncio.gather(
        DatasetActor.options(**cfg.actors.dataset).as_actor(**cfg.dataset),
        HybridPolicyActor.options(**cfg.actors.hybrid_policy).as_actor(
            **cfg.hybrid_policy, loss=loss_fn
        ),
        ReplayBuffer.options(**cfg.actors.replay_buffer).as_actor(
            **cfg.replay_buffer, collate=collate
        ),
        ComputeAdvantages.options(**cfg.actors.compute_advantages).as_actor(),
        RewardActor.options(**cfg.services.reward_actor).as_service(
            reward_functions=[MathReward(), ThinkingReward()]
        ),
    )

    group_size = cfg.group_size
    max_req_tokens = cfg.max_req_tokens
    max_res_tokens = cfg.max_res_tokens

    # Set max_steps to the configured value, or -1 if not specified or Null
    max_steps = cfg.hybrid_policy.training.steps or -1

    print("All actors initialized successfully!")
    shutdown_event = asyncio.Event()

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

            prompt, target = sample["request"], sample["target"]

            # Generate using hybrid policy actor (no weight sync needed!)
            responses: list[Completion] = await hybrid_policy.generate.call_one(prompt)
            t.step("policy_generation")

            # Construct episodes and calculate rewards
            episodes = []
            seq_len = max_req_tokens + max_res_tokens

            for i, response in enumerate(responses):
                # Validate logprobs exist
                if response.logprobs is None:
                    raise ValueError(
                        "Completion.logprobs is None. "
                        "Ensure HybridPolicy returns logprobs by setting 'logprobs: 1' in sampling_params config."
                    )

                # Prepare generator_logprobs
                # Shift by -1 to align with next-token prediction
                actual_response_len = response.token_ids.shape[0]
                generator_logprobs = torch.zeros(seq_len, dtype=response.logprobs.dtype)
                generator_logprobs[
                    max_req_tokens : max_req_tokens + actual_response_len
                ] = response.logprobs
                generator_logprobs = torch.roll(generator_logprobs, shifts=-1, dims=0)
                generator_logprobs[-1] = 0.0

                # Prepare loss_mask
                response_mask = torch.zeros(seq_len, dtype=torch.float32)
                response_mask[max_req_tokens : max_req_tokens + actual_response_len] = (
                    1.0
                )
                loss_mask = torch.roll(response_mask, shifts=-1, dims=0)
                loss_mask[-1] = 0.0

                episode = Episode(
                    episode_id=str(uuid.uuid4()),
                    pad_id=pad_id,
                    request_len=max_req_tokens,
                    response_len=max_res_tokens,
                    target=target,
                    request=prompt,
                    response=response.text,
                    completion=response,
                    generator_logprobs=generator_logprobs,
                    loss_mask=loss_mask,
                )

                (
                    episode.reward_breakdown,
                    episode.reward,
                ) = await reward_actor.evaluate_response.route(
                    prompt=prompt, response=response.text, target=target
                )
                episodes.append(episode)

                # Track token-based metrics
                prompt_tokens = episode.completion.prompt_ids.shape[0]
                response_tokens = episode.completion.token_ids.shape[0]

                record_metric("episode/avg_prompt_tokens", prompt_tokens, Reduce.MEAN)
                record_metric("episode/max_prompt_tokens", prompt_tokens, Reduce.MAX)
                record_metric("episode/min_prompt_tokens", prompt_tokens, Reduce.MIN)
                record_metric(
                    "episode/avg_response_tokens", response_tokens, Reduce.MEAN
                )
                record_metric(
                    "episode/max_response_tokens", response_tokens, Reduce.MAX
                )
                record_metric(
                    "episode/min_response_tokens", response_tokens, Reduce.MIN
                )

            # drop episodes if
            # 1> reward std-dev is very small (including all 0s and all 1s)
            # 2> any response was truncated (didn't end with EOS)
            rewards = [e.reward for e in episodes]
            rewards_std = torch.std(torch.tensor(rewards))
            is_low_variance = rewards_std < 1e-3
            num_truncated = sum(
                1 for e in episodes if e.completion.stop_reason == "length"
            )
            is_truncated = num_truncated > 0
            drop = is_low_variance or is_truncated

            n = len(episodes)
            record_metric(
                "main/continuous_rollouts/episodes_dropped/low_variance",
                n if is_low_variance else 0,
                Reduce.SUM,
            )
            record_metric(
                "main/continuous_rollouts/episodes_dropped/truncated",
                num_truncated,
                Reduce.SUM,
            )
            record_metric(
                "main/continuous_rollouts/episodes_dropped/total",
                n if drop else 0,
                Reduce.SUM,
            )

            if drop:
                del episodes
                continue

            t.step("reward_evaluation")

            advantages = await compute_advantages.compute.call_one(episodes)
            for episode, advantage in zip(episodes, advantages):
                episode.advantage = advantage
                await replay_buffer.add.call_one(episode)

                sample = episode.to_dict(
                    exclude=[
                        "completion",
                        "loss_mask",
                        "generator_logprobs",
                        "ref_logprobs",
                    ]
                )
                sample["score"] = sample["reward"]
                record_metric(
                    "main_samples/continuous_rollouts/sample_table",
                    sample,
                    Reduce.SAMPLE,
                )

            rollout_count += 1
            record_metric(
                "main/continuous_rollouts/count_rollout_iterations", 1, Reduce.SUM
            )
            t.stop()

    async def continuous_training():
        training_step = 0
        restart_tracer = True

        while (
            max_steps == -1 or training_step < max_steps
        ) and not shutdown_event.is_set():
            # Restart tracer when needed
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

                # Train using hybrid policy actor (no weight sync needed!)
                await hybrid_policy.train_step.call_one(batch)
                training_step += 1
                t.step("train_step")

                # No weight synchronization needed - model is already shared!
                # This is the key advantage of the hybrid architecture

                t.stop()
                restart_tracer = True

                # Flush metrics every training step
                await mlogger.flush.call_one(training_step)

        if shutdown_event.is_set():
            print("Training stopped due to shutdown event (likely a task failure).")
        else:
            print(
                f"Reached training limit ({max_steps} steps). Exiting continuous_training loop."
            )

    num_rollout_threads = cfg.get("rollout_threads", 1)
    num_training_threads = cfg.get("training_threads", 1)
    print(
        f"Starting GRPO with Hybrid Policy: {num_rollout_threads} rollout threads, {num_training_threads} training threads"
    )
    print("NOTE: Using hybrid architecture - no weight synchronization overhead!")

    rollout_tasks = [
        asyncio.create_task(continuous_rollouts()) for _ in range(num_rollout_threads)
    ]
    training_task = asyncio.create_task(continuous_training())

    # Surface background task failures and trigger shutdown (fail-fast)
    def on_task_done(task: asyncio.Task, name: str):
        if task.cancelled():
            return
        exc = task.exception()
        if exc:
            print(f"ERROR: {name} failed: {type(exc).__name__}: {exc}")
            traceback.print_exception(type(exc), exc, exc.__traceback__)
            shutdown_event.set()

    for i, task in enumerate(rollout_tasks):
        task.add_done_callback(lambda t, i=i: on_task_done(t, f"rollout_task_{i}"))
    training_task.add_done_callback(lambda t: on_task_done(t, "training_task"))

    try:
        await training_task
    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"ERROR: Training task failed: {type(e).__name__}: {e}")
        traceback.print_exc()
    finally:
        print("Shutting down... (this may take a few seconds)")
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
        asyncio.run(main(cfg))

    _main()  # @parse grabs the cfg from CLI
