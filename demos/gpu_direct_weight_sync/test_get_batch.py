#!/usr/bin/env python3
"""Test put_batch and get_batch performance vs individual operations."""

import asyncio
import logging
import time

import torch
import torchstore as ts

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_batching():
    """Test put_batch/get_batch vs sequential put/get."""
    # Initialize TorchStore
    await ts.initialize()
    logger.info("TorchStore initialized")

    # Create test tensors
    num_params = 50  # Use fewer params for quick test
    tensor_size = (1024, 1024)  # ~4MB per tensor

    logger.info(f"Testing with {num_params} tensors of shape {tensor_size}")

    # Generate test data
    test_tensors = {
        f"test_param_{i}": torch.randn(tensor_size, device="cuda:0")
        for i in range(num_params)
    }
    keys = list(test_tensors.keys())

    # ==================== PUT TESTS ====================
    print("\n" + "=" * 60)
    print("PUT TESTS")
    print("=" * 60)

    # Test 1: Sequential put
    logger.info("\n=== Test 1: Sequential put ===")
    seq_put_start = time.perf_counter()
    for key, tensor in test_tensors.items():
        await ts.put(f"seq_{key}", tensor)
    seq_put_time = time.perf_counter() - seq_put_start
    logger.info(f"Sequential put: {seq_put_time:.2f}s ({seq_put_time/num_params*1000:.1f}ms per param)")

    # Test 2: Batched put
    logger.info("\n=== Test 2: Batched put ===")
    batch_tensors = {f"batch_{key}": tensor for key, tensor in test_tensors.items()}
    batch_put_start = time.perf_counter()
    await ts.put_batch(batch_tensors)
    batch_put_time = time.perf_counter() - batch_put_start
    logger.info(f"Batched put: {batch_put_time:.2f}s ({batch_put_time/num_params*1000:.1f}ms per param)")

    # ==================== GET TESTS ====================
    print("\n" + "=" * 60)
    print("GET TESTS")
    print("=" * 60)

    # Test 3: Sequential get
    logger.info("\n=== Test 3: Sequential get ===")
    seq_keys = [f"seq_{key}" for key in keys]
    seq_get_start = time.perf_counter()
    for key in seq_keys:
        _ = await ts.get(key)
    seq_get_time = time.perf_counter() - seq_get_start
    logger.info(f"Sequential get: {seq_get_time:.2f}s ({seq_get_time/num_params*1000:.1f}ms per param)")

    # Test 4: Batched get
    logger.info("\n=== Test 4: Batched get ===")
    batch_keys = [f"batch_{key}" for key in keys]
    batch_get_start = time.perf_counter()
    results = await ts.get_batch(batch_keys)
    batch_get_time = time.perf_counter() - batch_get_start
    logger.info(f"Batched get: {batch_get_time:.2f}s ({batch_get_time/num_params*1000:.1f}ms per param)")

    # ==================== RESULTS ====================
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"PUT Operations ({num_params} tensors):")
    print(f"  Sequential: {seq_put_time:.2f}s")
    print(f"  Batched:    {batch_put_time:.2f}s")
    print(f"  Speedup:    {seq_put_time/batch_put_time:.1f}x")
    print()
    print(f"GET Operations ({num_params} tensors):")
    print(f"  Sequential: {seq_get_time:.2f}s")
    print(f"  Batched:    {batch_get_time:.2f}s")
    print(f"  Speedup:    {seq_get_time/batch_get_time:.1f}x")
    print("=" * 60)

    # Cleanup
    await ts.shutdown()


if __name__ == "__main__":
    asyncio.run(test_batching())
