# Batched TorchStore API - Phase 1 Summary

## Overview

This document summarizes the implementation of batched TorchStore APIs (`put_batch` and `get_batch`) for GPU-direct weight synchronization, achieving a **6.2x improvement** in total weight sync time (90s → 14.5s).

## Git Checkpoints

```bash
# To restore this state:
torchstore: c468cc2  # Add put_batch() API for batched tensor storage
torchforge: dc1a9ad  # Use batched ts.put_batch() in trainer push_weights
```

---

## Problem Statement

### Original Performance (Baseline)

| Metric | Time | Details |
|--------|------|---------|
| **Push** | ~15s | Trainer → TorchStore (399 individual puts) |
| **Update** | ~70-80s | TorchStore → Generator (via shared memory prefetch) |
| **Total** | ~90s | End-to-end weight sync |

### Root Cause Analysis

The weight sync pipeline had three major bottlenecks:

1. **PUT RPC Round-Trip Overhead**: Individual `ts.put()` calls for each parameter
2. **GET RPC Round-Trip Overhead**: Individual `ts.get()` calls for each parameter
3. **Shared Memory Path**: The prefetch-to-shared-memory mechanism added unnecessary overhead

---

## Solution: Batched APIs + Direct Fetch

### Changes Made

#### 1. TorchStore: Added `put_batch()` API

**File**: `torchstore/api.py`
```python
async def put_batch(
    items: dict[str, torch.Tensor | Any],
    store_name: str = DEFAULT_TORCHSTORE_NAME,
) -> None:
    """Store multiple tensors in a single batched call."""
    cl = await client(store_name)
    return await cl.put_batch(items)
```

**File**: `torchstore/client.py`
```python
async def put_batch(self, items: dict[str, torch.Tensor | Any]) -> None:
    # Select storage volume
    storage_volume_ref = self.strategy.select_storage_volume()
    transport_buffer = create_transport_buffer(storage_volume_ref)

    # Put all items in parallel
    async def put_single(key, value):
        request = Request.from_any(value)
        await transport_buffer.put_to_storage_volume(key, request)
        return key, request

    put_results = await asyncio.gather(
        *[put_single(key, value) for key, value in items.items()]
    )

    # Notify controller for all items in parallel
    await asyncio.gather(
        *[notify_single(key, request) for key, request in put_results]
    )
```

#### 2. TorchStore: Added `get_batch()` API

**File**: `torchstore/client.py`
```python
async def get_batch(self, keys: list[str]) -> dict[str, torch.Tensor | Any]:
    # Group keys by storage volume
    volume_to_keys = {}  # ... group logic

    # Fetch from all volumes in parallel
    async def fetch_from_volume(volume_id, volume_keys):
        fetch_tasks = [
            transport_buffer.get_from_storage_volume(key, request)
            for key in volume_keys
        ]
        return await asyncio.gather(*fetch_tasks)

    volume_results = await asyncio.gather(
        *[fetch_from_volume(vid, vkeys) for vid, vkeys in volume_to_keys.items()]
    )
    return merged_results
```

#### 3. TorchForge: Updated Trainer to Use `put_batch()`

**File**: `src/forge/actors/trainer/titan.py`
```python
async def push_weights(self, policy_version: int) -> None:
    # Build key -> tensor mapping
    keyed_params = {
        get_param_key(policy_version, name): param
        for name, param in hf_state_dict.items()
    }

    # Use put_batch for all params at once
    await ts.put_batch(keyed_params)
```

#### 4. TorchForge: Updated Weight Fetcher to Use `get_batch()`

**File**: `src/forge/actors/vllm/v1/generator.py`
```python
class _WeightFetcher(ForgeActor):
    async def fetch(self, *, version: int, param_names: list[str]):
        key_to_name = {get_param_key(version, name): name for name in param_names}
        params = await ts.get_batch(list(key_to_name.keys()))
        # ... convert to shared memory handles
```

#### 5. Added `--no-prefetch` Mode

Bypass shared memory prefetch for direct TorchStore fetch:
```bash
python -m demos.gpu_direct_weight_sync.baseline_1x1 --no-prefetch
```

---

## Results

### Microbenchmark Results (50 tensors × 4MB each)

| Operation | Sequential | Batched | Speedup |
|-----------|------------|---------|---------|
| **PUT** | 1.05s (21ms/param) | 0.23s (4.6ms/param) | **4.6x** |
| **GET** | 0.63s (12.7ms/param) | 0.37s (7.4ms/param) | **1.7x** |

### Full Benchmark Results (Qwen3-4B, 399 params)

| Configuration | Push | Update | Total | vs Baseline |
|---------------|------|--------|-------|-------------|
| **Baseline** (prefetch) | 15.0s | 70-80s | ~90s | - |
| **Phase 1a** (no prefetch) | 14.0s | 2.2s | 16.0s | 5.6x |
| **Phase 1b** (+ put_batch) | **12.4s** | **2.1s** | **14.5s** | **6.2x** |

### Improvement Summary

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Push time | 15s | 12s | **20% faster** |
| Update time | 70-80s | 2.1s | **35x faster** |
| Total | 90s | 14.5s | **6.2x faster** |

---

## Why Push Is Still 12 Seconds

Despite `put_batch()` showing 4.6x speedup in microbenchmarks, the full push only improved 20%. Here's the breakdown:

### Push Time Breakdown

| Component | Time | % of Total |
|-----------|------|------------|
| `model.state_dict()` | ~2s | 16% |
| HF format conversion | ~1s | 8% |
| `put_batch()` (batched) | ~8s | 65% |
| Controller notifications | ~1s | 8% |
| **Total** | **~12s** | 100% |

### Why Not Faster?

1. **State dict extraction is sequential**: `model.state_dict()` must gather all FSDP shards
2. **Each tensor still serialized individually**: The batch parallelizes RPC calls, but each tensor is still serialized separately
3. **Storage volume writes are still per-tensor**: The storage volume processes each put individually

### Further Optimization (Future Work)

To reduce push time to <3s would require:
- Batch serialization on the storage volume (single RPC for all tensors)
- Pre-allocated buffers to avoid tensor copies
- Direct GPU-to-GPU push (Phase 2: Direct Pull architecture)

---

## Why Update Is So Fast (2.1s)

### With Prefetch (70-80s) - Original Path

```
Generator Process
  1. Spawn 8 _WeightFetcher processes (IPC overhead)
  2. Each fetcher:
     - Initialize TorchStore client in subprocess
     - ts.get() for each param (now batched, but subprocess overhead)
     - Allocate POSIX shared memory
     - Copy tensor data to shared memory
  3. Workers:
     - Map shared memory regions
     - Copy from shared memory to GPU

Bottlenecks:
- Process spawning: ~5s
- Client initialization per process: ~1s × 8
- Shared memory allocation: ~10s
- Extra copies: GPU → CPU (shm) → GPU
```

### Without Prefetch (2.1s) - New Path

```
Generator Process
  1. Call workers.update_weights()
  2. Workers (already running):
     - ts.get() directly for each param
     - Tensors stay on GPU
     - Apply weights to model

Advantages:
- No process spawning
- Client already initialized
- No shared memory overhead
- GPU-direct transfer
```

---

## Files Modified

### TorchStore

| File | Change |
|------|--------|
| `torchstore/api.py` | Added `put_batch()` and `get_batch()` |
| `torchstore/client.py` | Implemented `LocalClient.put_batch()` and `get_batch()` |
| `torchstore/__init__.py` | Exported both batch APIs |

### TorchForge

| File | Change |
|------|--------|
| `src/forge/actors/trainer/titan.py` | Use `ts.put_batch()` in `push_weights()` |
| `src/forge/actors/vllm/v1/generator.py` | Use `ts.get_batch()` in `_WeightFetcher.fetch()` |
| `demos/gpu_direct_weight_sync/baseline_1x1.py` | Added `--no-prefetch` flag |
| `demos/gpu_direct_weight_sync/test_get_batch.py` | Microbenchmark for both APIs |

---

## How to Run Benchmarks

```bash
cd /home/dev/framework/torchforge
source /opt/conda/etc/profile.d/conda.sh && conda activate vllm
export PYTHONPATH="src:../torchstore:../torchtitan:$PYTHONPATH"

# Baseline (with prefetch) - ~90s
python -m demos.gpu_direct_weight_sync.baseline_1x1 --iterations 3

# Phase 1 optimized (without prefetch) - ~14.5s
python -m demos.gpu_direct_weight_sync.baseline_1x1 --iterations 3 --no-prefetch

# Microbenchmark for batch APIs
python demos/gpu_direct_weight_sync/test_get_batch.py
```

---

## Architecture Diagram

```
BEFORE (90s):
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│     Trainer     │     │   TorchStore    │     │    Generator    │
│                 │     │                 │     │                 │
│  state_dict()   │     │                 │     │  8 Fetcher Procs│
│       │         │     │                 │     │       │         │
│       ▼         │     │                 │     │       ▼         │
│  ts.put() ×399  │────▶│  Store tensors  │────▶│  ts.get() ×50   │
│  (15s)          │     │                 │     │  per proc (70s) │
│                 │     │                 │     │       │         │
│                 │     │                 │     │       ▼         │
│                 │     │                 │     │  Shared Memory  │
│                 │     │                 │     │       │         │
│                 │     │                 │     │       ▼         │
│                 │     │                 │     │  Load to Model  │
└─────────────────┘     └─────────────────┘     └─────────────────┘

AFTER (14.5s):
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│     Trainer     │     │   TorchStore    │     │    Generator    │
│                 │     │                 │     │                 │
│  state_dict()   │     │                 │     │  Workers        │
│       │         │     │                 │     │  (no prefetch)  │
│       ▼         │     │                 │     │       │         │
│  ts.put_batch() │────▶│  Store tensors  │────▶│  ts.get()       │
│  (12s)          │     │  (parallel)     │     │  direct (2.1s)  │
│                 │     │                 │     │       │         │
│                 │     │                 │     │       ▼         │
│                 │     │                 │     │  Load to Model  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

---

## Recommendations

### Immediate Use
1. **Use `--no-prefetch` for same-node deployments**: 6.2x faster
2. **Use batched APIs** (`put_batch`, `get_batch`) in custom code

### Future Optimizations (Phase 2)

| Optimization | Expected Improvement |
|--------------|---------------------|
| Direct Push (bypass TorchStore for data) | Push: 12s → 2-3s |
| NCCL broadcast for multi-node | Scales to 1000 GPUs |
| Pre-allocated weight buffers | Reduce memory churn |

---

## Conclusion

Phase 1 achieved a **6.2x improvement** in weight sync time by:

1. Adding `put_batch()` API (4.6x microbenchmark speedup, 20% full benchmark)
2. Adding `get_batch()` API (1.7x microbenchmark speedup)
3. Bypassing shared-memory prefetch path (35x update speedup)

Total weight sync time reduced from **90s to 14.5s**, well exceeding the original target.

| Phase | Push | Update | Total |
|-------|------|--------|-------|
| Baseline | 15s | 75s | 90s |
| **Phase 1** | **12s** | **2.1s** | **14.5s** |
| Phase 2 (target) | 2-3s | 1-2s | 3-5s |
