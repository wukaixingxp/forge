#!/usr/bin/env python3
"""
Baseline Weight Sync Benchmark: 1 Trainer GPU + 1 Generator GPU

This benchmark measures the end-to-end weight synchronization time
for a single GPU trainer pushing weights to a single GPU generator.

This establishes a baseline for optimizing weight sync performance.

Transport Types:
- MonarchRPC: Default, CPU-staged transfers (slowest)
- MonarchRDMA: Monarch RDMA (requires InfiniBand)
- TorchCommsRDMA: TorchComms RDMA (requires InfiniBand + torchcomms)
"""

import asyncio
import logging
import os
import time
import uuid

import monarch
import torchstore.api as ts
from torchstore.transport import TransportType
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


def get_transport_type(transport_name: str) -> TransportType:
    """Convert transport name string to TransportType enum."""
    transport_map = {
        "auto": TransportType.Unset,  # Auto-detect best available
        "monarchrpc": TransportType.MonarchRPC,
        "monarchrdma": TransportType.MonarchRDMA,
        "torchcommsrdma": TransportType.TorchCommsRDMA,
        "cudaipc": TransportType.CudaIPC,
    }
    name_lower = transport_name.lower().replace("-", "").replace("_", "")
    if name_lower not in transport_map:
        valid = ", ".join(transport_map.keys())
        raise ValueError(f"Unknown transport '{transport_name}'. Valid options: {valid}")
    return transport_map[name_lower]


def check_transport_availability() -> dict[str, bool]:
    """Check which transports are available on this system."""
    from torchstore.transport.monarch_rdma import monarch_rdma_transport_available
    from torchstore.transport.torchcomms.cache import torchcomms_rdma_available
    from torchstore.transport.cuda_ipc import cuda_ipc_available

    return {
        "MonarchRPC": True,  # Always available
        "MonarchRDMA": monarch_rdma_transport_available(),
        "TorchCommsRDMA": torchcomms_rdma_available(),
        "CudaIPC": cuda_ipc_available(),
    }


async def run_baseline_benchmark(
    config_path: str = None,
    num_iterations: int = 3,
    load_checkpoint: bool = False,
    transport: str = "auto",
    prefetch: bool = True,
):
    """Run baseline weight sync benchmark."""
    from forge.actors.trainer import TitanTrainer
    from forge.actors.generator import Generator

    # Load config
    config_dir = os.path.dirname(os.path.abspath(__file__))
    if config_path is None:
        config_path = os.path.join(config_dir, "qwen3_4b_1x1.yaml")
    cfg = load_config(config_path)

    model_name = cfg.trainer.model.get("name", "unknown")
    model_flavor = cfg.trainer.model.get("flavor", "unknown")
    trainer_gpus = cfg.actors.trainer.get("procs", 1)
    generator_gpus = cfg.services.generator.get("procs", 1)

    # Resolve transport type
    transport_type = get_transport_type(transport)
    transport_availability = check_transport_availability()

    # If auto, determine best available
    if transport_type == TransportType.Unset:
        if transport_availability["TorchCommsRDMA"]:
            transport_type = TransportType.TorchCommsRDMA
        elif transport_availability["MonarchRDMA"]:
            transport_type = TransportType.MonarchRDMA
        elif transport_availability["CudaIPC"]:
            transport_type = TransportType.CudaIPC
        else:
            transport_type = TransportType.MonarchRPC

    print("=" * 70)
    print("Baseline Weight Sync Benchmark")
    print("=" * 70)
    print(f"Model: {model_name} {model_flavor}")
    print(f"Trainer: {trainer_gpus} GPU(s)")
    print(f"Generator: {generator_gpus} GPU(s)")
    print(f"Iterations: {num_iterations}")
    print(f"Load checkpoint: {load_checkpoint}")
    print(f"Transport: {transport_type.name}")
    print(f"Prefetch to SHM: {prefetch}")
    print("-" * 70)
    print("Transport Availability:")
    for name, available in transport_availability.items():
        status = "AVAILABLE" if available else "NOT AVAILABLE"
        print(f"  {name}: {status}")
    print("=" * 70)

    # Initialize TorchStore with configured transport
    print("\n[1/5] Initializing TorchStore...")
    strategy = ts.ControllerStorageVolumes(default_transport_type=transport_type)
    await ts.initialize(strategy=strategy)
    logger.info(f"TorchStore initialized with {transport_type.name} transport")

    # Launch trainer
    print("\n[2/5] Launching Trainer...")
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
    print("\n[3/5] Launching Generator...")
    generator_cfg = dict(cfg.generator)
    services_generator_cfg = cfg.services.generator
    services_generator_cfg.num_replicas = 1

    # Control prefetch behavior
    generator_cfg["prefetch_weights_to_shm"] = prefetch

    try:
        generator = await Generator.options(**services_generator_cfg).as_service(**generator_cfg)
        logger.info("Generator launched successfully")
    except Exception as e:
        logger.error(f"Failed to launch generator: {e}")
        await trainer.cleanup.call()
        await TitanTrainer.shutdown(trainer)
        await ts.shutdown()
        raise

    try:
        # Run weight sync iterations
        print(f"\n[4/5] Running {num_iterations} weight sync iterations...")

        push_times = []
        update_times = []
        total_times = []

        for i in range(num_iterations):
            version = uuid.uuid4().int

            # Measure push time
            start = time.perf_counter()
            await trainer.push_weights.call(policy_version=version)
            push_time = time.perf_counter() - start

            # Measure update time
            start = time.perf_counter()
            await generator.update_weights.fanout(version=version)
            update_time = time.perf_counter() - start

            total_time = push_time + update_time

            push_times.append(push_time)
            update_times.append(update_time)
            total_times.append(total_time)

            print(f"   Iteration {i+1}: push={push_time:.2f}s, update={update_time:.2f}s, total={total_time:.2f}s")

        # Calculate statistics
        print("\n[5/5] Results...")

        avg_push = sum(push_times) / len(push_times)
        avg_update = sum(update_times) / len(update_times)
        avg_total = sum(total_times) / len(total_times)

        min_push = min(push_times)
        min_update = min(update_times)
        min_total = min(total_times)

        max_push = max(push_times)
        max_update = max(update_times)
        max_total = max(total_times)

        print("\n" + "=" * 70)
        print("BASELINE BENCHMARK RESULTS")
        print("=" * 70)
        print(f"Configuration: {trainer_gpus} trainer GPU -> {generator_gpus} generator GPU")
        print(f"Model: {model_name} {model_flavor}")
        print(f"Transport: {transport_type.name}")
        print("-" * 70)
        print(f"Push weights:")
        print(f"  Average: {avg_push:.2f}s")
        print(f"  Min:     {min_push:.2f}s")
        print(f"  Max:     {max_push:.2f}s")
        print("-" * 70)
        print(f"Update weights:")
        print(f"  Average: {avg_update:.2f}s")
        print(f"  Min:     {min_update:.2f}s")
        print(f"  Max:     {max_update:.2f}s")
        print("-" * 70)
        print(f"Total weight sync:")
        print(f"  Average: {avg_total:.2f}s")
        print(f"  Min:     {min_total:.2f}s")
        print(f"  Max:     {max_total:.2f}s")
        print("=" * 70)

        return {
            "config": f"{trainer_gpus}x{generator_gpus}",
            "model": f"{model_name} {model_flavor}",
            "transport": transport_type.name,
            "push_avg": avg_push,
            "update_avg": avg_update,
            "total_avg": avg_total,
            "push_times": push_times,
            "update_times": update_times,
            "total_times": total_times,
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

    parser = argparse.ArgumentParser(description="Baseline Weight Sync Benchmark")
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
        "--transport",
        type=str,
        default="auto",
        choices=["auto", "MonarchRPC", "MonarchRDMA", "TorchCommsRDMA", "CudaIPC"],
        help="Transport type (default: auto - detect best available)",
    )
    parser.add_argument(
        "--no-prefetch",
        action="store_true",
        help="Disable prefetch to shared memory (direct TorchStore fetch instead)",
    )
    args = parser.parse_args()

    asyncio.run(run_baseline_benchmark(
        config_path=args.config,
        num_iterations=args.iterations,
        load_checkpoint=args.load_checkpoint,
        transport=args.transport,
        prefetch=not args.no_prefetch,
    ))


if __name__ == "__main__":
    main()
