#!/usr/bin/env python3
"""
GPU-Direct only benchmark - tests push_weights_sharded without legacy comparison.
For large models where legacy is too slow.
"""

import asyncio
import logging
import os
import time
import uuid

import monarch
import torchstore.api as ts
from omegaconf import DictConfig, OmegaConf

monarch.actor.unhandled_fault_hook = lambda failure: None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> DictConfig:
    from forge.util.config import resolve_hf_hub_paths
    cfg = OmegaConf.load(config_path)
    assert isinstance(cfg, DictConfig)
    cfg = resolve_hf_hub_paths(cfg)
    return cfg


async def run_gpu_direct_benchmark(config_path: str, load_checkpoint: bool = False):
    """Benchmark GPU-direct push_weights_sharded only."""
    from forge.actors.trainer import TitanTrainer

    config_dir = os.path.dirname(os.path.abspath(__file__))
    if config_path is None:
        config_path = os.path.join(config_dir, "qwen3_demo.yaml")
    cfg = load_config(config_path)

    model_name = cfg.trainer.model.get("name", "unknown")
    model_flavor = cfg.trainer.model.get("flavor", "unknown")

    print("=" * 70)
    print("GPU-Direct Weight Sync Benchmark (Large Model)")
    print("=" * 70)
    print(f"Model: {model_name} {model_flavor}")
    print(f"Trainer: FSDP={cfg.trainer.parallelism.data_parallel_shard_degree}")
    print(f"Load checkpoint: {load_checkpoint}")
    print("=" * 70)

    print("\n[1/3] Initializing TorchStore...")
    await ts.initialize(strategy=ts.ControllerStorageVolumes())

    print("\n[2/3] Launching Trainer...")
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
        print("\n[3/3] Benchmarking GPU-DIRECT push_weights_sharded...")
        v1 = uuid.uuid4().int

        start_time = time.perf_counter()
        result = await trainer.push_weights_sharded.call(policy_version=v1)
        gpu_direct_push_time = time.perf_counter() - start_time

        # Get metadata
        if hasattr(result, '__iter__') and not isinstance(result, dict):
            metadata = list(result)[0] if result else {}
        else:
            metadata = result
        param_count = metadata.get("param_count", "?") if isinstance(metadata, dict) else "?"

        print("\n" + "=" * 70)
        print("GPU-DIRECT BENCHMARK RESULTS")
        print("=" * 70)
        print(f"Model: {model_name} {model_flavor}")
        print(f"GPU-Direct push_weights_sharded: {gpu_direct_push_time:.2f}s")
        print(f"Parameters pushed per rank: {param_count}")
        print("=" * 70)

        return {"gpu_direct_push_time": gpu_direct_push_time, "param_count": param_count}

    finally:
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
    import argparse
    parser = argparse.ArgumentParser(description="GPU-Direct Only Benchmark")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--load-checkpoint", action="store_true")
    args = parser.parse_args()
    asyncio.run(run_gpu_direct_benchmark(config_path=args.config, load_checkpoint=args.load_checkpoint))


if __name__ == "__main__":
    main()
