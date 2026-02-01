# Phase 2: CUDA IPC Direct Weight Transfer

## Overview

Phase 2 implements GPU-direct weight synchronization by bypassing TorchStore entirely
and using CUDA IPC handles for cross-process GPU memory access. This achieves a
**19-26x speedup** over the original baseline.

## Git Checkpoints

```bash
# To restore this state:
torchstore: 1e75533  # Add batched controller notification for put_batch
torchforge: 022b6fd  # Final checkpoint update for Phase 2

# Key commits:
# torchforge: 063fce1  # Implement Phase 2: CUDA IPC direct weight transfer
# torchforge: 9967369  # Phase 1 docs update
```

---

## Performance Results

### Benchmark: Qwen3-4B (399 parameters, ~8GB weights)

| Phase | Push | Update | Total | vs Baseline |
|-------|------|--------|-------|-------------|
| Baseline (prefetch) | 15s | 75s | **90s** | - |
| Phase 1 (no prefetch) | 12s | 2.1s | **14.5s** | 6.2x |
| **Phase 2 (IPC)** | N/A | 3.5-4.7s | **3.5-4.7s** | **19-26x** |

### Detailed Phase 2 Timing

| Operation | First Run | Subsequent | Notes |
|-----------|-----------|------------|-------|
| Handle creation | 0.43s | 0.01s | CUDA IPC warmup effect |
| IPC send + receive | 3.5s | 3.5s | Consistent |
| **Total** | **5.8s** | **3.5s** | - |

---

## Architecture

```
BEFORE (Phase 1 - 14.5s):
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│     Trainer     │     │   TorchStore    │     │    Generator    │
│                 │     │                 │     │                 │
│  state_dict()   │     │                 │     │  Workers        │
│       │         │     │                 │     │       │         │
│       ▼         │     │                 │     │       ▼         │
│  ts.put_batch() │────▶│  Store tensors  │────▶│  ts.get()       │
│  (serialize)    │     │  (RPC)          │     │  (deserialize)  │
│                 │     │                 │     │       │         │
│                 │     │                 │     │       ▼         │
│                 │     │                 │     │  Load to Model  │
└─────────────────┘     └─────────────────┘     └─────────────────┘

AFTER (Phase 2 - 3.5s):
┌─────────────────┐                          ┌─────────────────┐
│     Trainer     │                          │    Generator    │
│                 │                          │                 │
│  model.params   │                          │  Workers        │
│       │         │                          │       │         │
│       ▼         │                          │       ▼         │
│  create_ipc_    │  ───66-byte handles───▶  │  reconstruct_   │
│  handle()       │   (no serialization)     │  tensor()       │
│                 │                          │       │         │
│                 │                          │       ▼         │
│                 │                          │  Load to Model  │
└─────────────────┘                          └─────────────────┘
```

---

## Key Optimizations

### 1. Skip TorchStore
- No RPC round-trips to controller/storage volumes
- No metadata tracking overhead
- Direct trainer → worker communication

### 2. Skip Python Serialization
- CUDA IPC handles are only **66 bytes** each
- vs ~8MB per parameter with pickle serialization
- 399 params × 66 bytes = 26KB total metadata (vs ~3.2GB serialized)

### 3. Skip state_dict()
- Access `model.named_parameters()` directly
- No FSDP all_gather operation
- No HuggingFace name conversion overhead

### 4. GPU-Direct Memory Access
- Receiver reconstructs tensor from trainer's GPU memory
- Uses PyTorch's `rebuild_cuda_tensor()`
- Leverages NVLink bandwidth when available

---

## Implementation

### Files Modified

| File | Change |
|------|--------|
| `src/forge/actors/trainer/titan.py` | Added `push_weights_ipc()` endpoint |
| `src/forge/actors/vllm/v1/forge_executor.py` | Added `receive_weights_ipc()` endpoint |
| `src/forge/actors/vllm/v1/generator.py` | Added `update_weights_ipc()` coordinator |
| `demos/gpu_direct_weight_sync/ipc_benchmark.py` | Phase 2 benchmark script |

### Key Code Paths

**Trainer (`push_weights_ipc`):**
```python
for name, param in model.named_parameters():
    tensor = param.data
    handle = create_ipc_handle(tensor)  # 66 bytes
    ipc_handles[hf_name] = handle

await generator_workers.receive_weights_ipc.call(
    policy_version=version,
    ipc_handles=ipc_handles,
)
```

**Worker (`receive_weights_ipc`):**
```python
for name, handle in ipc_handles.items():
    tensor = handle.reconstruct_tensor()  # GPU-direct
    tensor = tensor.clone()  # Own the data
    model.load_weights([(name, tensor)])
```

**Generator (`update_weights_ipc`):**
```python
await self.llm.pause_generation(...)
result = await trainer.push_weights_ipc.call_one(
    policy_version=version,
    generator_workers=self.workers,
)
await self.llm.resume_generation()
```

---

## Usage

### Run Benchmark

```bash
cd /home/dev/framework/torchforge
source /opt/conda/etc/profile.d/conda.sh && conda activate vllm
export PYTHONPATH="src:../torchstore:../torchtitan:$PYTHONPATH"

# Phase 2 only
python -m demos.gpu_direct_weight_sync.ipc_benchmark --iterations 3

# Compare with Phase 1 baseline
python -m demos.gpu_direct_weight_sync.ipc_benchmark --iterations 3 --compare-baseline
```

### In Application Code

```python
# Old way (Phase 1 via TorchStore):
await trainer.push_weights.call(policy_version=version)
await generator.update_weights.fanout(version=version)

# New way (Phase 2 via IPC):
await generator.update_weights_ipc.fanout(version=version, trainer=trainer)
```

---

## Requirements

- **Single-node deployment**: CUDA IPC is intra-node only
- **Same physical machine**: Trainer and generator must share GPU access
- **CUDA-capable GPUs**: Required for IPC handle creation

---

## Limitations

1. **Single-node only**: CUDA IPC doesn't work across network
2. **No FSDP support yet**: Currently requires single trainer GPU (no PP)
3. **Memory lifetime**: Trainer must keep tensors alive until transfer completes

---

## Future Work

| Optimization | Expected Impact |
|--------------|-----------------|
| NCCL broadcast for multi-node | Enable distributed deployments |
| Async IPC with overlap | Hide transfer latency during generation |
| Pre-allocated buffers | Reduce memory allocation overhead |
| FSDP shard-aware IPC | Support multi-GPU trainers |

---

## Summary

Phase 2 achieves **19-26x speedup** over baseline by:

1. ✅ Bypassing TorchStore entirely
2. ✅ Using 66-byte CUDA IPC handles instead of serialization
3. ✅ Accessing model parameters directly (no state_dict)
4. ✅ Enabling GPU-direct memory access

| Metric | Baseline | Phase 1 | Phase 2 |
|--------|----------|---------|---------|
| Total time | 90s | 14.5s | **3.5-4.7s** |
| Speedup | - | 6.2x | **19-26x** |
