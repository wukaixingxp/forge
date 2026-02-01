#!/usr/bin/env python3
"""
Profile weight sync to understand GPU communication patterns.

This script profiles:
1. Where tensors are allocated (CPU vs GPU)
2. Memory transfers (CPU<->GPU copies)
3. RDMA operations
4. Time spent in each phase

Usage:
    # With PyTorch profiler (generates Chrome trace)
    python -m demos.gpu_direct_weight_sync.profile_weight_sync --profiler pytorch

    # With Nsight Systems (run with nsys)
    nsys profile -o weight_sync_trace python -m demos.gpu_direct_weight_sync.profile_weight_sync --profiler nsys

    # Simple timing only
    python -m demos.gpu_direct_weight_sync.profile_weight_sync --profiler timing
"""

import argparse
import asyncio
import logging
import os
import time
import uuid
from contextlib import contextmanager

import torch
import monarch
import torchstore.api as ts
from omegaconf import DictConfig, OmegaConf

monarch.actor.unhandled_fault_hook = lambda failure: None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> DictConfig:
    from forge.util.config import resolve_hf_hub_paths
    cfg = OmegaConf.load(config_path)
    assert isinstance(cfg, DictConfig)
    cfg = resolve_hf_hub_paths(cfg)
    return cfg


@contextmanager
def pytorch_profiler(output_dir: str):
    """PyTorch profiler context with detailed GPU tracing."""
    os.makedirs(output_dir, exist_ok=True)

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
    ) as prof:
        yield prof

    # Export traces
    trace_file = os.path.join(output_dir, "weight_sync_trace.json")
    prof.export_chrome_trace(trace_file)
    print(f"\n=== PyTorch Profiler Results ===")
    print(f"Trace saved to: {trace_file}")
    print(f"Open in Chrome: chrome://tracing")
    print(f"\n=== Key Events ===")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

    # Memory events
    print(f"\n=== Memory Events ===")
    print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))


@contextmanager
def nsys_profiler():
    """Nsight Systems profiler - just markers, actual profiling done by nsys CLI."""
    # Add NVTX markers for nsys
    try:
        import nvtx
        has_nvtx = True
    except ImportError:
        has_nvtx = False
        print("Warning: nvtx not available. Install with: pip install nvtx")

    class NsysContext:
        def __init__(self):
            self.has_nvtx = has_nvtx

        def range_push(self, name):
            if self.has_nvtx:
                nvtx.push_range(name)
            else:
                torch.cuda.nvtx.range_push(name)

        def range_pop(self):
            if self.has_nvtx:
                nvtx.pop_range()
            else:
                torch.cuda.nvtx.range_pop()

    ctx = NsysContext()
    yield ctx
    print("\n=== Nsight Systems ===")
    print("Run with: nsys profile -o trace python -m demos.gpu_direct_weight_sync.profile_weight_sync --profiler nsys")
    print("View with: nsys-ui trace.nsys-rep")


@contextmanager
def timing_profiler():
    """Simple timing profiler."""
    timings = {}

    class TimingContext:
        def __init__(self):
            self.timings = timings
            self.start_times = {}

        def start(self, name):
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            self.start_times[name] = time.perf_counter()

        def stop(self, name):
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            elapsed = time.perf_counter() - self.start_times[name]
            self.timings[name] = elapsed
            return elapsed

    ctx = TimingContext()
    yield ctx

    print("\n=== Timing Results ===")
    total = sum(timings.values())
    for name, elapsed in sorted(timings.items(), key=lambda x: -x[1]):
        pct = (elapsed / total * 100) if total > 0 else 0
        print(f"  {name}: {elapsed:.3f}s ({pct:.1f}%)")
    print(f"  TOTAL: {total:.3f}s")


async def profile_put_operation(trainer, policy_version, prof_ctx):
    """Profile the push_weights operation."""
    print("\n[Profiling PUT operation...]")

    if hasattr(prof_ctx, 'start'):
        prof_ctx.start("push_weights_total")
    elif hasattr(prof_ctx, 'range_push'):
        prof_ctx.range_push("push_weights_total")

    await trainer.push_weights.call(policy_version=policy_version)

    if hasattr(prof_ctx, 'stop'):
        prof_ctx.stop("push_weights_total")
    elif hasattr(prof_ctx, 'range_pop'):
        prof_ctx.range_pop()


async def profile_get_operation(store_client, key, prof_ctx):
    """Profile the get operation."""
    print("\n[Profiling GET operation...]")

    if hasattr(prof_ctx, 'start'):
        prof_ctx.start("get_total")

    # Get a single tensor to profile
    result = await store_client.get(key)

    if hasattr(prof_ctx, 'stop'):
        prof_ctx.stop("get_total")

    return result


async def run_profiling(config_path: str, profiler_type: str):
    """Main profiling routine."""
    from forge.actors.trainer import TitanTrainer

    config_dir = os.path.dirname(os.path.abspath(__file__))
    if config_path is None:
        config_path = os.path.join(config_dir, "qwen3_4b_demo.yaml")
    cfg = load_config(config_path)

    print("=" * 70)
    print(f"Weight Sync Profiling ({profiler_type})")
    print("=" * 70)
    print(f"Model: {cfg.trainer.model.get('name')} {cfg.trainer.model.get('flavor')}")
    print(f"FSDP: {cfg.trainer.parallelism.data_parallel_shard_degree}")
    print("=" * 70)

    # Select profiler
    if profiler_type == "pytorch":
        profiler_ctx = pytorch_profiler("/tmp/weight_sync_profile")
    elif profiler_type == "nsys":
        profiler_ctx = nsys_profiler()
    else:
        profiler_ctx = timing_profiler()

    with profiler_ctx as prof_ctx:
        # Initialize TorchStore
        print("\n[1/4] Initializing TorchStore...")
        await ts.initialize(strategy=ts.ControllerStorageVolumes())

        # Launch trainer
        print("\n[2/4] Launching Trainer...")
        trainer_cfg = cfg.trainer
        trainer_cfg.checkpoint = {"enable": False}  # Skip checkpoint for profiling

        trainer = await TitanTrainer.options(**cfg.actors.trainer).as_actor(**trainer_cfg)

        try:
            # Profile PUT
            print("\n[3/4] Profiling push_weights...")
            v0 = uuid.uuid4().int

            # Warm up
            print("  Warmup run...")
            await trainer.push_weights.call(policy_version=v0)

            # Profiled run
            print("  Profiled run...")
            v1 = v0 + 1

            if profiler_type == "pytorch":
                # Use PyTorch profiler
                await profile_put_operation(trainer, v1, prof_ctx)
            else:
                await profile_put_operation(trainer, v1, prof_ctx)

            print("\n[4/4] Profiling complete!")

        finally:
            print("\nCleaning up...")
            try:
                await trainer.cleanup.call()
            except:
                pass
            try:
                await TitanTrainer.shutdown(trainer)
            except:
                pass
            await ts.shutdown()


def main():
    parser = argparse.ArgumentParser(description="Profile Weight Sync")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--profiler",
        type=str,
        choices=["pytorch", "nsys", "timing"],
        default="timing",
        help="Profiler type: pytorch (Chrome trace), nsys (Nsight Systems), timing (simple)"
    )
    args = parser.parse_args()

    # Set environment for better profiling
    os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "0")

    asyncio.run(run_profiling(config_path=args.config, profiler_type=args.profiler))


if __name__ == "__main__":
    main()
