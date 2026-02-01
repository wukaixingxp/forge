#!/usr/bin/env python3
"""Test get_batch performance vs individual gets."""

import asyncio
import logging
import time

import torch
import torchstore as ts

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_batching():
    """Test get_batch vs sequential get."""
    # Initialize TorchStore
    await ts.initialize()
    logger.info("TorchStore initialized")

    # Create test tensors
    num_params = 50  # Use fewer params for quick test
    tensor_size = (1024, 1024)  # ~4MB per tensor

    logger.info(f"Creating {num_params} test tensors of shape {tensor_size}")

    # Put tensors
    keys = []
    put_start = time.perf_counter()
    for i in range(num_params):
        key = f"test_param_{i}"
        keys.append(key)
        tensor = torch.randn(tensor_size, device="cuda:0")
        await ts.put(key, tensor)
    put_time = time.perf_counter() - put_start
    logger.info(f"Put {num_params} tensors in {put_time:.2f}s")

    # Test 1: Sequential get
    logger.info("\n=== Test 1: Sequential get ===")
    seq_start = time.perf_counter()
    for key in keys:
        _ = await ts.get(key)
    seq_time = time.perf_counter() - seq_start
    logger.info(f"Sequential get: {seq_time:.2f}s ({seq_time/num_params*1000:.1f}ms per param)")

    # Test 2: Batched get
    logger.info("\n=== Test 2: Batched get ===")
    batch_start = time.perf_counter()
    results = await ts.get_batch(keys)
    batch_time = time.perf_counter() - batch_start
    logger.info(f"Batched get: {batch_time:.2f}s ({batch_time/num_params*1000:.1f}ms per param)")

    # Results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Sequential get: {seq_time:.2f}s")
    print(f"Batched get:    {batch_time:.2f}s")
    print(f"Speedup:        {seq_time/batch_time:.1f}x")
    print("=" * 60)

    # Cleanup
    await ts.shutdown()


if __name__ == "__main__":
    asyncio.run(test_batching())
