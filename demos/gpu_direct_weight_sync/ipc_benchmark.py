#!/usr/bin/env python3
"""
Phase 2 CUDA IPC Weight Sync Benchmark

This benchmark tests the Phase 2 optimization that bypasses TorchStore entirely
by using CUDA IPC handles for GPU-direct weight transfer from trainer to generator.

Key optimizations over Phase 1:
1. Skip TorchStore - direct trainer -> worker communication
2. Skip Python serialization - use 66-byte CUDA IPC handles
3. Skip state_dict() on trainer - access parameters directly

Requirements:
- Single-node deployment (CUDA IPC is intra-node only)
- Trainer and generator must be on same physical machine
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


async def run_ipc_benchmark(
    config_path: str = None,
    num_iterations: int = 3,
    load_checkpoint: bool = False,
    compare_baseline: bool = False,
):
    """Run Phase 2 IPC weight sync benchmark.

    Args:
        config_path: Path to config file
        num_iterations: Number of benchmark iterations
        load_checkpoint: Whether to load a real checkpoint
        compare_baseline: Also run Phase 1 for comparison
    """
    from forge.actors.trainer import TitanTrainer
    from forge.actors.generator import Generator
    from torchstore.transport.cuda_ipc import cuda_ipc_available

    # Load config
    config_dir = os.path.dirname(os.path.abspath(__file__))
    if config_path is None:
        config_path = os.path.join(config_dir, "qwen3_4b_1x1.yaml")
    cfg = load_config(config_path)

    model_name = cfg.trainer.model.get("name", "unknown")
    model_flavor = cfg.trainer.model.get("flavor", "unknown")
    trainer_gpus = cfg.actors.trainer.get("procs", 1)
    generator_gpus = cfg.services.generator.get("procs", 1)

    # Check CUDA IPC availability
    ipc_available = cuda_ipc_available()

    print("=" * 70)
    print("Phase 2 CUDA IPC Weight Sync Benchmark")
    print("=" * 70)
    print(f"Model: {model_name} {model_flavor}")
    print(f"Trainer: {trainer_gpus} GPU(s)")
    print(f"Generator: {generator_gpus} GPU(s)")
    print(f"Iterations: {num_iterations}")
    print(f"Load checkpoint: {load_checkpoint}")
    print(f"CUDA IPC available: {ipc_available}")
    print(f"Compare baseline: {compare_baseline}")
    print("=" * 70)

    if not ipc_available:
        print("\nERROR: CUDA IPC is not available on this system!")
        print("Phase 2 requires CUDA IPC for GPU-direct transfers.")
        return None

    # Initialize TorchStore (needed for baseline comparison)
    print("\n[1/6] Initializing TorchStore...")
    await ts.initialize()
    logger.info("TorchStore initialized")

    # Launch trainer
    print("\n[2/6] Launching Trainer...")
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

    # Launch generator
    print("\n[3/6] Launching Generator...")
    generator_cfg = dict(cfg.generator)
    services_generator_cfg = cfg.services.generator
    services_generator_cfg.num_replicas = 1
    # Disable prefetch for fair comparison
    generator_cfg["prefetch_weights_to_shm"] = False

    try:
        generator = await Generator.options(**services_generator_cfg).as_service(**generator_cfg)
        logger.info("Generator launched successfully")
    except Exception as e:
        logger.error(f"Failed to launch generator: {e}")
        await trainer.cleanup.call()
        await TitanTrainer.shutdown(trainer)
        await ts.shutdown()
        raise

    baseline_results = None
    ipc_results = None

    try:
        # Run baseline comparison first if requested
        if compare_baseline:
            print(f"\n[4/6] Running {num_iterations} Phase 1 (baseline) iterations...")
            baseline_push_times = []
            baseline_update_times = []
            baseline_total_times = []

            for i in range(num_iterations):
                version = uuid.uuid4().int

                # Measure push time (trainer -> TorchStore)
                start = time.perf_counter()
                await trainer.push_weights.call(policy_version=version)
                push_time = time.perf_counter() - start

                # Measure update time (TorchStore -> generator)
                start = time.perf_counter()
                await generator.update_weights.fanout(version=version)
                update_time = time.perf_counter() - start

                total_time = push_time + update_time
                baseline_push_times.append(push_time)
                baseline_update_times.append(update_time)
                baseline_total_times.append(total_time)

                print(f"   Baseline {i+1}: push={push_time:.2f}s, update={update_time:.2f}s, total={total_time:.2f}s")

            baseline_results = {
                "push_times": baseline_push_times,
                "update_times": baseline_update_times,
                "total_times": baseline_total_times,
                "push_avg": sum(baseline_push_times) / len(baseline_push_times),
                "update_avg": sum(baseline_update_times) / len(baseline_update_times),
                "total_avg": sum(baseline_total_times) / len(baseline_total_times),
            }
        else:
            print("\n[4/6] Skipping baseline (use --compare-baseline to enable)")

        # Run Phase 2 IPC benchmark
        print(f"\n[5/6] Running {num_iterations} Phase 2 (IPC) iterations...")
        ipc_total_times = []
        ipc_handle_times = []
        ipc_send_times = []

        for i in range(num_iterations):
            version = uuid.uuid4().int

            # Measure IPC push (trainer -> generator directly)
            start = time.perf_counter()
            result = await generator.update_weights_ipc.fanout(
                version=version,
                trainer=trainer,
            )
            total_time = time.perf_counter() - start

            # Extract timing details from result
            # The result is a list of dicts from each replica
            if result and len(result) > 0:
                r = result[0]
                handle_time = r.get("handle_creation_time", 0)
                send_time = r.get("send_time", 0)
            else:
                handle_time = 0
                send_time = 0

            ipc_total_times.append(total_time)
            ipc_handle_times.append(handle_time)
            ipc_send_times.append(send_time)

            print(f"   IPC {i+1}: total={total_time:.2f}s (handles={handle_time:.2f}s, send={send_time:.2f}s)")

        ipc_results = {
            "total_times": ipc_total_times,
            "handle_times": ipc_handle_times,
            "send_times": ipc_send_times,
            "total_avg": sum(ipc_total_times) / len(ipc_total_times),
            "handle_avg": sum(ipc_handle_times) / len(ipc_handle_times),
            "send_avg": sum(ipc_send_times) / len(ipc_send_times),
        }

        # Print results
        print("\n[6/6] Results...")
        print("\n" + "=" * 70)
        print("PHASE 2 IPC BENCHMARK RESULTS")
        print("=" * 70)
        print(f"Configuration: {trainer_gpus} trainer GPU -> {generator_gpus} generator GPU")
        print(f"Model: {model_name} {model_flavor}")

        if baseline_results:
            print("-" * 70)
            print("Phase 1 (Baseline via TorchStore):")
            print(f"  Push:   avg={baseline_results['push_avg']:.2f}s")
            print(f"  Update: avg={baseline_results['update_avg']:.2f}s")
            print(f"  Total:  avg={baseline_results['total_avg']:.2f}s")

        print("-" * 70)
        print("Phase 2 (CUDA IPC Direct):")
        print(f"  Handle creation: avg={ipc_results['handle_avg']:.2f}s")
        print(f"  IPC send:        avg={ipc_results['send_avg']:.2f}s")
        print(f"  Total:           avg={ipc_results['total_avg']:.2f}s")

        if baseline_results:
            speedup = baseline_results['total_avg'] / ipc_results['total_avg']
            print("-" * 70)
            print(f"SPEEDUP: {speedup:.1f}x faster than Phase 1")
            print(f"         ({baseline_results['total_avg']:.1f}s -> {ipc_results['total_avg']:.1f}s)")

        print("=" * 70)

        return {
            "config": f"{trainer_gpus}x{generator_gpus}",
            "model": f"{model_name} {model_flavor}",
            "baseline": baseline_results,
            "ipc": ipc_results,
        }

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


def main():
    """Entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Phase 2 CUDA IPC Weight Sync Benchmark")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file (default: qwen3_4b_1x1.yaml)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Number of weight sync iterations (default: 3)",
    )
    parser.add_argument(
        "--load-checkpoint",
        action="store_true",
        help="Load model checkpoint (for real weights)",
    )
    parser.add_argument(
        "--compare-baseline",
        action="store_true",
        help="Also run Phase 1 baseline for comparison",
    )
    args = parser.parse_args()

    asyncio.run(run_ipc_benchmark(
        config_path=args.config,
        num_iterations=args.iterations,
        load_checkpoint=args.load_checkpoint,
        compare_baseline=args.compare_baseline,
    ))


if __name__ == "__main__":
    main()
