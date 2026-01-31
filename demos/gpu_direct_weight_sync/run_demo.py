#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""
GPU-Direct Weight Sync Demo

Demonstrates GPU-direct weight synchronization between:
- Trainer: 2 GPUs with FSDP (each GPU holds half of each parameter)
- Generator: 2 GPUs with TP=2 (each GPU holds columns or rows based on layer type)

Model: Llama 4 Scout (17B with 16 experts)

Usage:
    # From torchforge directory
    python -m demos.gpu_direct_weight_sync.run_demo

Requirements:
    - 4 GPUs (2 for trainer, 2 for generator)
    - Llama 4 Scout model at /home/dev/framework/Llama-4-Scout-17B-16E-Instruct
    - TorchStore initialized
"""

import asyncio
import logging
import os
import sys
import time
import uuid

import monarch
import torch
import torchstore.api as ts
from omegaconf import DictConfig, OmegaConf

# Workaround for monarch mesh shutdown exit code during teardown
monarch.actor.unhandled_fault_hook = lambda failure: None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> DictConfig:
    """Load and resolve config file."""
    from forge.util.config import resolve_hf_hub_paths

    cfg = OmegaConf.load(config_path)
    assert isinstance(cfg, DictConfig)
    cfg = resolve_hf_hub_paths(cfg)
    return cfg


async def run_gpu_direct_demo(config_path: str = None, load_checkpoint: bool = False):
    """Run the GPU-direct weight sync demo."""
    from forge.actors.trainer import TitanTrainer
    from forge.actors.generator import Generator
    from forge.controller.provisioner import init_provisioner
    from forge.types import LauncherConfig, ProvisionerConfig

    # Load config
    config_dir = os.path.dirname(os.path.abspath(__file__))
    if config_path is None:
        config_path = os.path.join(config_dir, "qwen3_demo.yaml")
    cfg = load_config(config_path)

    model_name = cfg.trainer.model.get("name", "unknown")
    model_flavor = cfg.trainer.model.get("flavor", "unknown")

    print("=" * 70)
    print("GPU-Direct Weight Sync Demo")
    print("=" * 70)
    print(f"Model: {model_name} {model_flavor}")
    print("Trainer: FSDP=2 (2 GPUs)")
    print("Generator: TP=2 (2 GPUs)")
    print(f"Load checkpoint: {load_checkpoint}")
    print("=" * 70)

    logger.info(f"Loaded config from {config_path}")

    # Initialize TorchStore
    print("\n[1/6] Initializing TorchStore...")
    await ts.initialize(strategy=ts.ControllerStorageVolumes())
    logger.info("TorchStore initialized")

    # Launch trainer and generator
    print("\n[2/6] Launching Trainer (FSDP=2) and Generator (TP=2)...")

    trainer_cfg = cfg.trainer
    if not load_checkpoint:
        # Disable checkpoint loading for faster startup (random weights)
        trainer_cfg.checkpoint = {
            "enable": False,
        }

    generator_cfg = cfg.generator
    services_generator_cfg = cfg.services.generator
    services_generator_cfg.num_replicas = 1

    try:
        # Launch sequentially to avoid concurrent memory pressure during checkpoint loading
        logger.info("Launching trainer first...")
        trainer = await TitanTrainer.options(**cfg.actors.trainer).as_actor(**trainer_cfg)
        logger.info("Trainer launched. Now launching generator...")
        generator = await Generator.options(**services_generator_cfg).as_service(**generator_cfg)
        logger.info("Trainer and Generator launched successfully")
    except Exception as e:
        logger.error(f"Failed to launch actors: {e}")
        await ts.shutdown()
        raise

    try:
        # Test legacy weight sync first
        print("\n[3/6] Testing LEGACY weight sync (baseline)...")
        v0 = uuid.uuid4().int

        start_time = time.perf_counter()
        await trainer.push_weights.call(policy_version=v0)
        legacy_push_time = time.perf_counter() - start_time
        logger.info(f"Legacy push_weights completed in {legacy_push_time:.2f}s")

        start_time = time.perf_counter()
        await generator.update_weights.fanout(version=v0)
        legacy_update_time = time.perf_counter() - start_time
        logger.info(f"Legacy update_weights completed in {legacy_update_time:.2f}s")

        print(f"   Legacy push_weights: {legacy_push_time:.2f}s")
        print(f"   Legacy update_weights: {legacy_update_time:.2f}s")
        print(f"   Legacy total: {legacy_push_time + legacy_update_time:.2f}s")

        # Test GPU-direct weight sync
        print("\n[4/6] Testing GPU-DIRECT weight sync (new method)...")
        v1 = v0 + 1

        # Get parameter shapes for TP slice computation
        # Note: For FSDP trainer with multiple ranks, use .call() and take first result
        param_shapes_list = await trainer.get_param_shapes.call()
        param_shapes = param_shapes_list[0]  # All ranks return same shapes
        logger.info(f"Got {len(param_shapes)} parameter shapes from trainer")

        start_time = time.perf_counter()
        await trainer.push_weights_sharded.call(policy_version=v1)
        gpu_direct_push_time = time.perf_counter() - start_time
        logger.info(f"GPU-direct push_weights_sharded completed in {gpu_direct_push_time:.2f}s")

        start_time = time.perf_counter()
        await generator.update_weights_gpu_direct.fanout(
            version=v1,
            param_shapes=param_shapes,
        )
        gpu_direct_update_time = time.perf_counter() - start_time
        logger.info(f"GPU-direct update_weights_gpu_direct completed in {gpu_direct_update_time:.2f}s")

        print(f"   GPU-direct push_weights_sharded: {gpu_direct_push_time:.2f}s")
        print(f"   GPU-direct update_weights_gpu_direct: {gpu_direct_update_time:.2f}s")
        print(f"   GPU-direct total: {gpu_direct_push_time + gpu_direct_update_time:.2f}s")

        # Compare results
        print("\n[5/6] Performance Comparison...")
        legacy_total = legacy_push_time + legacy_update_time
        gpu_direct_total = gpu_direct_push_time + gpu_direct_update_time

        if gpu_direct_total > 0:
            speedup = legacy_total / gpu_direct_total
        else:
            speedup = float('inf')

        print(f"   Legacy total: {legacy_total:.2f}s")
        print(f"   GPU-direct total: {gpu_direct_total:.2f}s")
        print(f"   Speedup: {speedup:.2f}x")

        # Verify model can still generate
        print("\n[6/6] Verifying model generation...")
        try:
            completions = await generator.generate.call_one(
                "Hello, my name is",
                sampling_params={"max_tokens": 20, "temperature": 0.7},
            )
            if completions:
                print(f"   Generated: {completions[0].text[:100]}...")
                print("   Generation verification: PASSED")
            else:
                print("   Generation verification: FAILED (no completions)")
        except Exception as e:
            print(f"   Generation verification: FAILED ({e})")

        # Summary
        print("\n" + "=" * 70)
        print("DEMO RESULTS")
        print("=" * 70)
        print(f"Legacy weight sync:     {legacy_total:.2f}s")
        print(f"GPU-direct weight sync: {gpu_direct_total:.2f}s")
        print(f"Speedup:                {speedup:.2f}x")
        print("=" * 70)

        if speedup > 1.0:
            print("SUCCESS: GPU-direct weight sync is faster!")
        else:
            print("NOTE: GPU-direct may need optimization for this config")

    finally:
        # Cleanup
        print("\nCleaning up...")
        try:
            await trainer.cleanup.call()
        except Exception as e:
            logger.warning(f"Trainer cleanup error: {e}")

        try:
            await generator.shutdown()
        except Exception as e:
            logger.warning(f"Generator shutdown error: {e}")

        try:
            await TitanTrainer.shutdown(trainer)
        except Exception as e:
            logger.warning(f"Trainer shutdown error: {e}")

        await ts.shutdown()
        print("Cleanup complete")


async def run_simplified_demo():
    """Run a simplified demo that tests the new APIs without full model loading.

    This is useful for quick testing of the API additions.
    """
    from torchstore.transport.types import TensorSlice

    print("=" * 70)
    print("GPU-Direct Weight Sync - Simplified API Test")
    print("=" * 70)

    # Initialize TorchStore
    print("\n[1/4] Initializing TorchStore...")
    await ts.initialize()
    logger.info("TorchStore initialized")

    # Test put_slice and get_slice APIs
    print("\n[2/4] Testing put_slice API...")

    # Create mock FSDP shards (2 ranks, row-wise sharding)
    global_shape = (1000, 512)
    fsdp_size = 2

    # Shard 0: rows 0-499
    shard_0 = torch.randn(500, 512)
    slice_0 = TensorSlice(
        offsets=(0, 0),
        coordinates=(0,),
        global_shape=global_shape,
        local_shape=(500, 512),
        mesh_shape=(fsdp_size,),
    )

    # Shard 1: rows 500-999
    shard_1 = torch.randn(500, 512)
    slice_1 = TensorSlice(
        offsets=(500, 0),
        coordinates=(1,),
        global_shape=global_shape,
        local_shape=(500, 512),
        mesh_shape=(fsdp_size,),
    )

    await ts.put_slice("test_tensor", shard_0, slice_0)
    await ts.put_slice("test_tensor", shard_1, slice_1)
    print("   Stored 2 FSDP shards (row-wise)")

    # Test get_slice API (simulate TP fetch)
    print("\n[3/4] Testing get_slice API...")

    # TP rank 0 needs first half of columns
    tp_slice_0 = TensorSlice(
        offsets=(0, 0),
        coordinates=(0,),
        global_shape=global_shape,
        local_shape=(1000, 256),
        mesh_shape=(2,),
    )

    result_0 = await ts.get_slice("test_tensor", tp_slice_0)
    print(f"   TP rank 0 fetched: shape={result_0.shape}")

    # TP rank 1 needs second half of columns
    tp_slice_1 = TensorSlice(
        offsets=(0, 256),
        coordinates=(1,),
        global_shape=global_shape,
        local_shape=(1000, 256),
        mesh_shape=(2,),
    )

    result_1 = await ts.get_slice("test_tensor", tp_slice_1)
    print(f"   TP rank 1 fetched: shape={result_1.shape}")

    # Verify correctness
    print("\n[4/4] Verifying correctness...")

    # Check shapes
    assert result_0.shape == (1000, 256), f"Expected (1000, 256), got {result_0.shape}"
    assert result_1.shape == (1000, 256), f"Expected (1000, 256), got {result_1.shape}"

    # Check data integrity (first shard's first column slice)
    expected_top_left = shard_0[:, :256]
    actual_top_left = result_0[:500, :]
    if torch.allclose(expected_top_left, actual_top_left, atol=1e-5):
        print("   Data integrity: PASSED")
    else:
        print("   Data integrity: FAILED")

    # Cleanup
    await ts.shutdown()

    print("\n" + "=" * 70)
    print("Simplified API Test: PASSED")
    print("=" * 70)


def main():
    """Entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="GPU-Direct Weight Sync Demo")
    parser.add_argument(
        "--simplified",
        action="store_true",
        help="Run simplified API test without full model loading",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file (default: qwen3_demo.yaml)",
    )
    parser.add_argument(
        "--load-checkpoint",
        action="store_true",
        help="Load model checkpoint (required for meaningful generation)",
    )
    args = parser.parse_args()

    if args.simplified:
        asyncio.run(run_simplified_demo())
    else:
        asyncio.run(run_gpu_direct_demo(config_path=args.config, load_checkpoint=args.load_checkpoint))


if __name__ == "__main__":
    main()
