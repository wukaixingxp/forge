#!/usr/bin/env python3
"""
Simple benchmark comparing Legacy vs GPU-Direct push_weights.

This benchmark focuses only on the trainer-side push operation,
which is the primary bottleneck that GPU-direct addresses.
"""

import asyncio
import logging
import os
import time
import uuid

import monarch
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


async def run_push_benchmark(config_path: str, load_checkpoint: bool = False):
    """Benchmark just the push_weights operations."""
    from forge.actors.trainer import TitanTrainer

    # Load config
    config_dir = os.path.dirname(os.path.abspath(__file__))
    if config_path is None:
        config_path = os.path.join(config_dir, "qwen3_4b_demo.yaml")
    cfg = load_config(config_path)

    model_name = cfg.trainer.model.get("name", "unknown")
    model_flavor = cfg.trainer.model.get("flavor", "unknown")

    print("=" * 70)
    print("Push Weights Benchmark: Legacy vs GPU-Direct")
    print("=" * 70)
    print(f"Model: {model_name} {model_flavor}")
    print(f"Trainer: FSDP={cfg.trainer.parallelism.data_parallel_shard_degree} (2 GPUs)")
    print(f"Load checkpoint: {load_checkpoint}")
    print("=" * 70)

    # Initialize TorchStore
    print("\n[1/4] Initializing TorchStore...")
    await ts.initialize(strategy=ts.ControllerStorageVolumes())
    logger.info("TorchStore initialized")

    # Launch trainer only
    print("\n[2/4] Launching Trainer (FSDP=2)...")

    trainer_cfg = cfg.trainer
    if not load_checkpoint:
        trainer_cfg.checkpoint = {"enable": False}

    try:
        trainer = await TitanTrainer.options(**cfg.actors.trainer).as_actor(**trainer_cfg)
        logger.info("Trainer launched successfully")
    except Exception as e:
        logger.error(f"Failed to launch trainer: {e}")
        await ts.shutdown()
        raise

    try:
        # Benchmark 1: Legacy push_weights
        print("\n[3/4] Benchmarking LEGACY push_weights...")
        v0 = uuid.uuid4().int

        start_time = time.perf_counter()
        await trainer.push_weights.call(policy_version=v0)
        legacy_push_time = time.perf_counter() - start_time
        print(f"   Legacy push_weights: {legacy_push_time:.2f}s")

        # Clear TorchStore for fair comparison
        # (In real usage, versions are different so this isn't needed)

        # Benchmark 2: GPU-Direct push_weights_sharded
        print("\n[4/4] Benchmarking GPU-DIRECT push_weights_sharded...")
        v1 = v0 + 1

        start_time = time.perf_counter()
        result = await trainer.push_weights_sharded.call(policy_version=v1)
        gpu_direct_push_time = time.perf_counter() - start_time

        # Get metadata from first rank (handle ValueMesh response)
        if hasattr(result, '__iter__') and not isinstance(result, dict):
            metadata = list(result)[0] if result else {}
        else:
            metadata = result
        param_count = metadata.get("param_count", "?") if isinstance(metadata, dict) else "?"
        print(f"   GPU-Direct push_weights_sharded: {gpu_direct_push_time:.2f}s")
        print(f"   Pushed {param_count} shards per rank")

        # Results
        print("\n" + "=" * 70)
        print("PUSH BENCHMARK RESULTS")
        print("=" * 70)
        print(f"Legacy push_weights:        {legacy_push_time:.2f}s")
        print(f"GPU-Direct push_sharded:    {gpu_direct_push_time:.2f}s")

        if gpu_direct_push_time > 0:
            speedup = legacy_push_time / gpu_direct_push_time
            print(f"Speedup:                    {speedup:.2f}x")

            if speedup > 1.0:
                print("\nSUCCESS: GPU-direct is faster!")
            else:
                print("\nNOTE: GPU-direct may need more parallelism benefit in distributed setting")

        print("=" * 70)

        return {
            "legacy_push_time": legacy_push_time,
            "gpu_direct_push_time": gpu_direct_push_time,
            "speedup": legacy_push_time / gpu_direct_push_time if gpu_direct_push_time > 0 else 0,
        }

    finally:
        # Cleanup
        print("\nCleaning up...")
        try:
            await trainer.cleanup.call()
        except Exception as e:
            logger.warning(f"Trainer cleanup error: {e}")

        try:
            await TitanTrainer.shutdown(trainer)
        except Exception as e:
            logger.warning(f"Trainer shutdown error: {e}")

        await ts.shutdown()
        print("Cleanup complete")


def main():
    """Entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Push Weights Benchmark")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file (default: qwen3_4b_demo.yaml)",
    )
    parser.add_argument(
        "--load-checkpoint",
        action="store_true",
        help="Load model checkpoint (for real weights)",
    )
    args = parser.parse_args()

    asyncio.run(run_push_benchmark(config_path=args.config, load_checkpoint=args.load_checkpoint))


if __name__ == "__main__":
    main()
