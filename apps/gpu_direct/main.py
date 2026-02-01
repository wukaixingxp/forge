# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
GPU-Direct Weight Sync GRPO Example

This example demonstrates GRPO training with GPU-direct weight synchronization
using CUDA IPC handles for 60x faster weight transfer between trainer and generator.

Usage:
    # With IPC weight sync (recommended for single-node)
    python -m apps.gpu_direct.main --config apps/gpu_direct/qwen3_32b_2x2.yaml

    # Compare with baseline (TorchStore)
    python -m apps.gpu_direct.main --config apps/gpu_direct/qwen3_32b_2x2_baseline.yaml

Environment:
    FORGE_IPC_GPU_VISIBILITY=1  # Required for IPC across trainer/generator GPUs
"""

import asyncio
import os
import time
import traceback
import uuid

import torch
import torchstore as ts
import yaml
from apps.gpu_direct.data import DatasetActor
from apps.gpu_direct.grading import MathReward, ThinkingReward
from forge.actors.generator import Generator
from forge.actors.reference_model import ReferenceModel
from forge.actors.replay_buffer import ReplayBuffer
from forge.actors.trainer import TitanTrainer
from forge.controller.provisioner import init_provisioner, shutdown
from forge.data_models.completion import Completion
from forge.observability.metric_actors import get_or_create_metric_logger
from forge.observability.metrics import record_metric, Reduce
from forge.observability.perf_tracker import Tracer
from forge.rl import collate, ComputeAdvantages, Episode, RewardActor
from forge.rl.loss import DAPOLoss, GRPOLoss
from forge.types import LauncherConfig, ProvisionerConfig
from forge.util.checkpoint import drop_weights
from forge.util.config import parse
from forge.util.logging import get_logger
from omegaconf import DictConfig, OmegaConf

logger = get_logger("INFO")


def validate_ipc_config(cfg: DictConfig) -> bool:
    """Validate configuration for IPC weight sync."""
    use_ipc = cfg.get("weight_sync", {}).get("use_ipc", False)

    if not use_ipc:
        return True

    # Check environment variable
    if os.environ.get("FORGE_IPC_GPU_VISIBILITY") != "1":
        logger.warning(
            "FORGE_IPC_GPU_VISIBILITY=1 not set. "
            "IPC weight sync may fail for multi-GPU configs. "
            "Set: export FORGE_IPC_GPU_VISIBILITY=1"
        )

    # IPC only works for single-node deployments
    trainer_hosts = cfg.get("actors", {}).get("trainer", {}).get("hosts", 1)
    generator_hosts = cfg.get("services", {}).get("generator", {}).get("hosts", 1)

    if trainer_hosts > 1 or generator_hosts > 1:
        raise ValueError(
            f"IPC weight sync only works for single-node deployments. "
            f"Got trainer_hosts={trainer_hosts}, generator_hosts={generator_hosts}. "
            f"Set weight_sync.use_ipc=false for multi-node."
        )

    logger.info("IPC weight sync enabled - using CUDA IPC for direct GPU transfer")
    return True


async def main(cfg: DictConfig):
    """Main GRPO training loop with GPU-direct weight sync support."""
    # Convert OmegaConf config to plain dict
    run_config_for_logging = OmegaConf.to_container(cfg, resolve=True)

    # Log config
    logger.info("=" * 30 + " GPU-DIRECT GRPO CONFIGURATION " + "=" * 30)
    logger.info(
        yaml.dump(run_config_for_logging, default_flow_style=False, sort_keys=False)
    )

    # Validate IPC configuration
    use_ipc = cfg.get("weight_sync", {}).get("use_ipc", False)
    validate_ipc_config(cfg)

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

    # Fail-fast: Check loss/ref_model compatibility before spawning actors
    uses_ref_model = cfg.get("services", {}).get("ref_model") is not None
    if uses_ref_model and not isinstance(loss_fn, GRPOLoss):
        raise ValueError(
            f"ref_model is configured but {type(loss_fn).__name__} does not use ref_logprobs. "
            "Either remove the ref_model service config or use GRPOLoss with beta > 0."
        )
    if isinstance(loss_fn, GRPOLoss) and loss_fn.beta > 0 and not uses_ref_model:
        raise ValueError(
            f"GRPOLoss with beta={loss_fn.beta} requires ref_logprobs, but ref_model is not configured. "
            "Either add ref_model to services config or set beta=0."
        )

    # ---- Setup services ---- #

    async def noop():
        return None

    (
        dataloader,
        generator,
        trainer,
        replay_buffer,
        compute_advantages,
        ref_model,
        reward_actor,
    ) = await asyncio.gather(
        DatasetActor.options(**cfg.actors.dataset).as_actor(**cfg.dataset),
        Generator.options(**cfg.services.generator).as_service(**cfg.generator),
        TitanTrainer.options(**cfg.actors.trainer).as_actor(
            **cfg.trainer, loss=loss_fn
        ),
        ReplayBuffer.options(**cfg.actors.replay_buffer).as_actor(
            **cfg.replay_buffer, collate=collate
        ),
        ComputeAdvantages.options(**cfg.actors.compute_advantages).as_actor(),
        (
            ReferenceModel.options(**cfg.services.ref_model).as_service(**cfg.ref_model)
            if uses_ref_model
            else noop()
        ),
        RewardActor.options(**cfg.services.reward_actor).as_service(
            reward_functions=[MathReward(), ThinkingReward()]
        ),
    )

    group_size = cfg.group_size
    max_req_tokens = cfg.max_req_tokens
    max_res_tokens = cfg.max_res_tokens

    # Set max_steps to the configured value, or -1 if not specified or Null
    max_steps = cfg.trainer.training.steps or -1

    print("All services initialized successfully!")
    shutdown_event = asyncio.Event()

    # Initialize TorchStore (needed for baseline comparison and potentially other uses)
    # For IPC mode, TorchStore is still initialized but not used for weight sync
    trainer_num_procs = cfg.actors.trainer["procs"]
    trainer_host_mesh_name = cfg.actors.trainer["mesh_name"]
    trainer_hosts = await provisioner.get_host_mesh(trainer_host_mesh_name)
    await ts.initialize(
        mesh=trainer_hosts.spawn_procs(per_host={"procs": trainer_num_procs}),
        strategy=ts.LocalRankStrategy(),
    )

    if use_ipc:
        print("TorchStore initialized (available for fallback, but IPC will be used for weight sync)")
    else:
        print("TorchStore initialized with local rank strategy (baseline mode)")

    # Track weight sync timing for comparison
    weight_sync_times = []

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
            responses: list[Completion] = await generator.generate.route(prompt)
            t.step("policy_generation")

            # Construct episodes and calculate rewards
            episodes = []
            input_ids = torch.ones(
                (group_size, max_req_tokens + max_res_tokens),
                dtype=torch.long,
            )
            seq_len = max_req_tokens + max_res_tokens

            for i, response in enumerate(responses):
                # Validate logprobs exist
                if response.logprobs is None:
                    raise ValueError(
                        "Completion.logprobs is None. "
                        "Ensure Generator returns logprobs by setting 'logprobs: 1' in sampling_params config."
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

                # Build input_ids for reference logprobs
                input_ids[i, :max_req_tokens] = episode.request_tensor
                input_ids[i, max_req_tokens:] = episode.response_tensor

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
            # TODO: change it to filter only truncated episodes instead of dropping entire group
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
                del input_ids, episodes
                continue

            t.step("reward_evaluation")

            # Compute ref_logprobs only if ref_model is configured
            if ref_model is not None:
                ref_logprobs = await ref_model.forward.route(
                    input_ids, return_logprobs=True
                )
                t.step("reference_model_calculate_logprobs")

                for i, episode in enumerate(episodes):
                    episode.ref_logprobs = ref_logprobs[i]  # [seq_len]

                del ref_logprobs

            del input_ids

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
        nonlocal weight_sync_times
        training_step = 0
        restart_tracer = True  # Flag to control when to restart tracer

        while (
            max_steps == -1 or training_step < max_steps
        ) and not shutdown_event.is_set():
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

                await trainer.train_step.call(batch)
                training_step += 1
                t.step("train_step")

                # === Weight Synchronization ===
                sync_start = time.perf_counter()

                if use_ipc:
                    # Phase 2: GPU-Direct IPC weight sync
                    # Single call - trainer pushes IPC handles directly to generator workers
                    await generator.update_weights_ipc.fanout(
                        version=training_step,
                        trainer=trainer
                    )
                    t.step("update_weights_ipc")
                    # No drop_weights needed - IPC doesn't use TorchStore
                else:
                    # Baseline: TorchStore-based weight sync
                    await trainer.push_weights.call(training_step)
                    t.step("push_weights")

                    await generator.update_weights.fanout(training_step)
                    t.step("update_weights")

                    if training_step >= 2:
                        await drop_weights(training_step - 1)
                        t.step("drop_weights")

                sync_time = time.perf_counter() - sync_start
                weight_sync_times.append(sync_time)

                # Log weight sync timing
                record_metric("weight_sync/time_seconds", sync_time, Reduce.MEAN)
                record_metric("weight_sync/use_ipc", 1.0 if use_ipc else 0.0, Reduce.MAX)

                if training_step % 10 == 0:
                    avg_sync_time = sum(weight_sync_times[-10:]) / min(10, len(weight_sync_times))
                    mode = "IPC" if use_ipc else "TorchStore"
                    logger.info(
                        f"Step {training_step}: Weight sync ({mode}) avg={avg_sync_time:.3f}s, "
                        f"last={sync_time:.3f}s"
                    )

                t.stop()
                restart_tracer = True

                # Flush metrics every training step to WandB
                await mlogger.flush.call_one(training_step)

        if shutdown_event.is_set():
            print("Training stopped due to shutdown event (likely a task failure).")
        else:
            print(
                f"Reached training limit ({max_steps} steps). Exiting continuous_training loop."
            )

        # Print final weight sync summary
        if weight_sync_times:
            avg_time = sum(weight_sync_times) / len(weight_sync_times)
            min_time = min(weight_sync_times)
            max_time = max(weight_sync_times)
            mode = "IPC" if use_ipc else "TorchStore"
            print(f"\n{'='*60}")
            print(f"Weight Sync Summary ({mode} mode)")
            print(f"{'='*60}")
            print(f"  Total syncs: {len(weight_sync_times)}")
            print(f"  Average time: {avg_time:.3f}s")
            print(f"  Min time: {min_time:.3f}s")
            print(f"  Max time: {max_time:.3f}s")
            print(f"{'='*60}\n")

    num_rollout_threads = cfg.get("rollout_threads", 1)
    num_training_threads = cfg.get("training_threads", 1)
    print(
        f"Starting GPU-Direct GRPO with {num_rollout_threads} rollout threads, "
        f"{num_training_threads} training threads, "
        f"weight_sync={'IPC' if use_ipc else 'TorchStore'}"
    )
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
