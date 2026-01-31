#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""
Benchmark: GPU-Direct Weight Sync vs Traditional Approach

This benchmark compares:
1. Traditional: Store full tensor -> Fetch full tensor -> Slice on consumer
2. GPU-Direct: Store shards directly -> Fetch only needed slices

Simulates real FSDP->TP weight sync scenario with realistic tensor sizes.
"""

import asyncio
import time
from dataclasses import dataclass
from typing import Optional

import torch
import torchstore.api as ts
from torchstore.transport.types import TensorSlice


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    method: str
    store_time_ms: float
    fetch_time_ms: float
    total_time_ms: float
    stored_bytes: int
    fetched_bytes: int

    def __str__(self):
        return (
            f"{self.method}:\n"
            f"  Store: {self.store_time_ms:.2f}ms ({self.stored_bytes / 1e6:.1f}MB)\n"
            f"  Fetch: {self.fetch_time_ms:.2f}ms ({self.fetched_bytes / 1e6:.1f}MB)\n"
            f"  Total: {self.total_time_ms:.2f}ms"
        )


async def benchmark_traditional(
    tensor_key: str,
    global_shape: tuple[int, int],
    fsdp_size: int,
    tp_size: int,
    target_device: str = "cpu",
) -> BenchmarkResult:
    """Benchmark traditional approach: gather + store full + fetch full + slice.

    Traditional flow:
    1. FSDP ranks all_gather to full tensor (simulated)
    2. Store the full tensor
    3. Fetch the full tensor
    4. Slice to get TP shard (simulated)
    """
    # Create full tensor (simulating result of all_gather)
    full_tensor = torch.randn(*global_shape, dtype=torch.bfloat16)

    # Store full tensor
    start = time.perf_counter()
    slice_spec = TensorSlice(
        offsets=(0, 0),
        coordinates=(0,),
        global_shape=global_shape,
        local_shape=global_shape,
        mesh_shape=(1,),  # Single shard = full tensor
    )
    await ts.put_slice(tensor_key, full_tensor, slice_spec)
    store_time = (time.perf_counter() - start) * 1000

    # Fetch full tensor
    start = time.perf_counter()
    fetched = await ts.get_slice(tensor_key, slice_spec, target_device=target_device)
    fetch_time = (time.perf_counter() - start) * 1000

    # Simulate TP slicing (column-parallel)
    tp_cols = global_shape[1] // tp_size
    _ = fetched[:, :tp_cols]  # Simulated slice for TP rank 0

    # Cleanup
    await ts.delete(tensor_key)

    stored_bytes = full_tensor.numel() * full_tensor.element_size()
    fetched_bytes = fetched.numel() * fetched.element_size()

    return BenchmarkResult(
        method="Traditional (gather+full)",
        store_time_ms=store_time,
        fetch_time_ms=fetch_time,
        total_time_ms=store_time + fetch_time,
        stored_bytes=stored_bytes,
        fetched_bytes=fetched_bytes,
    )


async def benchmark_gpu_direct(
    tensor_key: str,
    global_shape: tuple[int, int],
    fsdp_size: int,
    tp_size: int,
    target_device: str = "cpu",
) -> BenchmarkResult:
    """Benchmark GPU-direct approach: store shards + fetch slices.

    GPU-direct flow:
    1. Each FSDP rank stores its shard directly (no all_gather) - IN PARALLEL
    2. Fetch only the slice needed for this TP rank
    """
    rows, cols = global_shape
    shard_rows = rows // fsdp_size

    # Prepare shards and slice specs
    shards = []
    slice_specs = []
    stored_bytes = 0
    for fsdp_rank in range(fsdp_size):
        shard = torch.randn(shard_rows, cols, dtype=torch.bfloat16)
        slice_spec = TensorSlice(
            offsets=(fsdp_rank * shard_rows, 0),
            coordinates=(fsdp_rank,),
            global_shape=global_shape,
            local_shape=(shard_rows, cols),
            mesh_shape=(fsdp_size,),
        )
        shards.append(shard)
        slice_specs.append(slice_spec)
        stored_bytes += shard.numel() * shard.element_size()

    # Store FSDP shards IN PARALLEL (simulating distributed FSDP ranks)
    start = time.perf_counter()
    await asyncio.gather(*[
        ts.put_slice(tensor_key, shards[i], slice_specs[i])
        for i in range(fsdp_size)
    ])
    store_time = (time.perf_counter() - start) * 1000

    # Fetch TP slice (only what TP rank 0 needs)
    tp_cols = cols // tp_size
    tp_slice = TensorSlice(
        offsets=(0, 0),
        coordinates=(0,),
        global_shape=global_shape,
        local_shape=(rows, tp_cols),
        mesh_shape=(tp_size,),
    )

    start = time.perf_counter()
    fetched = await ts.get_slice(tensor_key, tp_slice, target_device=target_device)
    fetch_time = (time.perf_counter() - start) * 1000

    # Cleanup
    await ts.delete(tensor_key)

    fetched_bytes = fetched.numel() * fetched.element_size()

    return BenchmarkResult(
        method="GPU-Direct (shards+slices)",
        store_time_ms=store_time,
        fetch_time_ms=fetch_time,
        total_time_ms=store_time + fetch_time,
        stored_bytes=stored_bytes,
        fetched_bytes=fetched_bytes,
    )


async def run_benchmark(
    hidden_dim: int = 4096,
    num_params: int = 10,
    fsdp_size: int = 2,
    tp_size: int = 2,
    warmup_runs: int = 2,
    benchmark_runs: int = 5,
) -> None:
    """Run the full benchmark comparing traditional vs GPU-direct."""

    print("=" * 70)
    print("GPU-Direct Weight Sync Benchmark")
    print("=" * 70)
    print(f"Hidden dimension: {hidden_dim}")
    print(f"Number of params: {num_params}")
    print(f"FSDP size: {fsdp_size}")
    print(f"TP size: {tp_size}")
    print(f"Warmup runs: {warmup_runs}")
    print(f"Benchmark runs: {benchmark_runs}")
    print("=" * 70)

    # Initialize TorchStore
    print("\n[1/5] Initializing TorchStore...")
    await ts.initialize()

    # Determine target device
    target_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Target device: {target_device}")

    # Global shape (simulating a weight matrix)
    global_shape = (hidden_dim, hidden_dim)
    bytes_per_param = hidden_dim * hidden_dim * 2  # bfloat16
    total_model_bytes = bytes_per_param * num_params

    print(f"\nParameter size: {bytes_per_param / 1e6:.1f}MB each")
    print(f"Total model size: {total_model_bytes / 1e9:.2f}GB ({num_params} params)")

    # Warmup
    print(f"\n[2/5] Warmup ({warmup_runs} runs each)...")
    for i in range(warmup_runs):
        await benchmark_traditional(f"warmup_trad_{i}", global_shape, fsdp_size, tp_size, target_device)
        await benchmark_gpu_direct(f"warmup_gpu_{i}", global_shape, fsdp_size, tp_size, target_device)
    print("   Warmup complete")

    # Benchmark traditional approach
    print(f"\n[3/5] Benchmarking Traditional approach ({benchmark_runs} runs)...")
    trad_results = []
    for i in range(benchmark_runs):
        result = await benchmark_traditional(f"bench_trad_{i}", global_shape, fsdp_size, tp_size, target_device)
        trad_results.append(result)
        print(f"   Run {i+1}: {result.total_time_ms:.2f}ms")

    # Benchmark GPU-direct approach
    print(f"\n[4/5] Benchmarking GPU-Direct approach ({benchmark_runs} runs)...")
    gpu_results = []
    for i in range(benchmark_runs):
        result = await benchmark_gpu_direct(f"bench_gpu_{i}", global_shape, fsdp_size, tp_size, target_device)
        gpu_results.append(result)
        print(f"   Run {i+1}: {result.total_time_ms:.2f}ms")

    # Calculate statistics
    print("\n[5/5] Computing results...")

    def avg(results, attr):
        return sum(getattr(r, attr) for r in results) / len(results)

    trad_avg_total = avg(trad_results, "total_time_ms")
    trad_avg_store = avg(trad_results, "store_time_ms")
    trad_avg_fetch = avg(trad_results, "fetch_time_ms")

    gpu_avg_total = avg(gpu_results, "total_time_ms")
    gpu_avg_store = avg(gpu_results, "store_time_ms")
    gpu_avg_fetch = avg(gpu_results, "fetch_time_ms")

    # Compute speedup
    speedup = trad_avg_total / gpu_avg_total if gpu_avg_total > 0 else float('inf')

    # Per-param estimates for full model sync
    trad_full_model_ms = trad_avg_total * num_params
    gpu_full_model_ms = gpu_avg_total * num_params

    # Print results
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS (average over {} runs)".format(benchmark_runs))
    print("=" * 70)

    print("\nTraditional (gather + store full + fetch full):")
    print(f"  Store time:  {trad_avg_store:.2f}ms")
    print(f"  Fetch time:  {trad_avg_fetch:.2f}ms")
    print(f"  Total:       {trad_avg_total:.2f}ms per param")
    print(f"  Full model:  {trad_full_model_ms/1000:.2f}s (estimated)")

    print("\nGPU-Direct (store shards + fetch slices):")
    print(f"  Store time:  {gpu_avg_store:.2f}ms")
    print(f"  Fetch time:  {gpu_avg_fetch:.2f}ms")
    print(f"  Total:       {gpu_avg_total:.2f}ms per param")
    print(f"  Full model:  {gpu_full_model_ms/1000:.2f}s (estimated)")

    print("\n" + "-" * 70)
    print(f"SPEEDUP: {speedup:.2f}x")
    print("-" * 70)

    # Memory savings analysis
    tp_cols = hidden_dim // tp_size
    trad_fetch_bytes = hidden_dim * hidden_dim * 2  # Full tensor
    gpu_fetch_bytes = hidden_dim * tp_cols * 2  # Only TP slice
    memory_reduction = (1 - gpu_fetch_bytes / trad_fetch_bytes) * 100

    print(f"\nMemory transfer reduction: {memory_reduction:.1f}%")
    print(f"  Traditional fetches: {trad_fetch_bytes / 1e6:.1f}MB per param")
    print(f"  GPU-Direct fetches:  {gpu_fetch_bytes / 1e6:.1f}MB per param")

    print("\n" + "=" * 70)
    if speedup > 1.0:
        print(f"SUCCESS: GPU-Direct is {speedup:.2f}x faster!")
    else:
        print("NOTE: Results may vary based on storage backend and network")
    print("=" * 70)

    # Cleanup
    await ts.shutdown()


def main():
    """Entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="GPU-Direct Weight Sync Benchmark")
    parser.add_argument(
        "--hidden-dim", type=int, default=4096,
        help="Hidden dimension for weight matrices (default: 4096)"
    )
    parser.add_argument(
        "--num-params", type=int, default=10,
        help="Number of parameters to simulate (default: 10)"
    )
    parser.add_argument(
        "--fsdp-size", type=int, default=2,
        help="FSDP world size (default: 2)"
    )
    parser.add_argument(
        "--tp-size", type=int, default=2,
        help="Tensor parallel size (default: 2)"
    )
    parser.add_argument(
        "--warmup", type=int, default=2,
        help="Number of warmup runs (default: 2)"
    )
    parser.add_argument(
        "--runs", type=int, default=5,
        help="Number of benchmark runs (default: 5)"
    )
    args = parser.parse_args()

    asyncio.run(run_benchmark(
        hidden_dim=args.hidden_dim,
        num_params=args.num_params,
        fsdp_size=args.fsdp_size,
        tp_size=args.tp_size,
        warmup_runs=args.warmup,
        benchmark_runs=args.runs,
    ))


if __name__ == "__main__":
    main()
