# Phase 2: CUDA IPC Direct Weight Transfer

## Overview

Phase 2 implements GPU-direct weight synchronization by bypassing TorchStore entirely
and using CUDA IPC handles for cross-process GPU memory access. Now supports both
1x1 (single GPU) and 2x2 (FSDP + TP) configurations.

---

## Performance Results

### Benchmark: Qwen3-4B (399 parameters, ~8GB weights)

#### 1x1 Configuration (1 trainer GPU, 1 generator GPU)

| Phase | Push | Update | Total | vs Baseline |
|-------|------|--------|-------|-------------|
| Baseline (prefetch) | 15s | 75s | **90s** | - |
| Phase 1 (no prefetch) | 12s | 2.1s | **14.5s** | 6.2x |
| **Phase 2 (IPC)** | N/A | 3.5-4.7s | **3.5-4.7s** | **19-26x** |

#### 2x2 Configuration (FSDP=2 trainer, TP=2 generator)

| Phase | Total | vs Baseline |
|-------|-------|-------------|
| Baseline (TorchStore) | ~10s | - |
| **Phase 2 (IPC)** | **1.29s** | **~7.7x** |

### Detailed Phase 2 Timing (2x2 Config)

| Operation | Time | Notes |
|-----------|------|-------|
| Handle creation | 0.04s | Create IPC handles for 399 params |
| IPC send + receive | 0.30s | Transfer handles to both TP workers |
| Direct param copy | ~0.95s | Copy weights to model parameters |
| **Total** | **1.29s** | Average across iterations |

---

## Architecture

```
BEFORE (Phase 1 via TorchStore):
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│     Trainer     │     │   TorchStore    │     │    Generator    │
│   (FSDP x2)     │     │                 │     │   (TP x2)       │
│       │         │     │                 │     │       │         │
│  state_dict()   │     │                 │     │  Workers        │
│  (all_gather)   │     │                 │     │       │         │
│       │         │     │                 │     │       ▼         │
│       ▼         │     │                 │     │  ts.get()       │
│  ts.put_batch() │────▶│  Store tensors  │────▶│  (deserialize)  │
│  (serialize)    │     │  (RPC)          │     │       │         │
│                 │     │                 │     │       ▼         │
│                 │     │                 │     │  Load to Model  │
└─────────────────┘     └─────────────────┘     └─────────────────┘

AFTER (Phase 2 - Direct IPC):
┌─────────────────┐                          ┌─────────────────┐
│     Trainer     │                          │    Generator    │
│   (FSDP x2)     │                          │   (TP x2)       │
│       │         │                          │       │         │
│  state_dict()   │                          │  Worker 0 (TP0) │
│  (all_gather)   │                          │  Worker 1 (TP1) │
│       │         │                          │       │         │
│       ▼         │                          │       ▼         │
│  create_ipc_    │  ───66-byte handles───▶  │  reconstruct_   │
│  handle()       │   (parallel to workers)  │  tensor()       │
│                 │                          │       │         │
│                 │                          │       ▼         │
│                 │                          │  Direct param   │
│                 │                          │  .data.copy_()  │
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

### 3. GPU-Direct Memory Access
- Receiver reconstructs tensor from trainer's GPU memory
- Uses PyTorch's `rebuild_cuda_tensor()`
- Leverages NVLink bandwidth when available

### 4. Direct Parameter Update (NEW for TP support)
- Bypass vLLM's `model.load_weights()` which doesn't support TP weight updates
- Map HF names to vLLM merged parameters (qkv_proj, gate_up_proj)
- Slice full tensors for each TP rank
- Direct copy via `param.data.copy_()`

---

## Implementation

### Files Modified

| File | Change |
|------|--------|
| `src/forge/actors/trainer/titan.py` | `push_weights_ipc()` - creates IPC handles and sends to workers |
| `src/forge/actors/vllm/v1/forge_executor.py` | `receive_weights_ipc_sliced()` - reconstructs and loads weights |
| `src/forge/actors/vllm/v1/generator.py` | `update_weights_ipc()` - coordinates IPC weight sync |
| `src/forge/actors/vllm/v1/monarch_executor.py` | `FORGE_IPC_GPU_VISIBILITY` - GPU visibility for IPC |
| `demos/gpu_direct_weight_sync/ipc_benchmark.py` | Phase 2 benchmark script |
| `demos/gpu_direct_weight_sync/qwen3_4b_2x2.yaml` | 2x2 config (FSDP + TP) |

### Key Code Paths

**Trainer (`push_weights_ipc`):**
```python
# For TP > 1, send full tensors (workers handle slicing)
for hf_name, tensor in hf_state_dict.items():
    handle = create_ipc_handle(tensor)  # 66 bytes
    for tp_rank in range(tp_size):
        ipc_handles_per_rank[tp_rank][hf_name] = handle

await generator_workers.receive_weights_ipc_sliced.call(
    policy_version=version,
    ipc_handles_per_rank=ipc_handles_per_rank,
)
```

**Worker (`receive_weights_ipc_sliced`):**
```python
# Build mapping from HF names to vLLM merged params
param_map = self._build_param_map(model)

for name, handle in ipc_handles.items():
    tensor = handle.reconstruct_tensor()  # GPU-direct
    tensor = tensor.clone()  # Own the data

    if name in param_map:
        mapping = param_map[name]
        if isinstance(mapping, tuple):  # Merged weight (qkv_proj, gate_up_proj)
            self._copy_to_merged_param(merge_type, tensor, param, tp_rank, tp_size)
        else:
            # Slice for TP if needed, then direct copy
            if tensor.shape != param.shape:
                tensor = self._slice_for_tp(name, tensor, param.shape, tp_rank, tp_size)
            param.data.copy_(tensor)
```

---

## Usage

### Run Benchmark

```bash
cd /home/dev/framework/torchforge
source /opt/conda/etc/profile.d/conda.sh && conda activate vllm
export PYTHONPATH="src:../torchstore:../torchtitan:$PYTHONPATH"

# For FSDP + TP configs, enable IPC GPU visibility
export FORGE_IPC_GPU_VISIBILITY=1

# 1x1 config (single GPU trainer/generator)
python -m demos.gpu_direct_weight_sync.ipc_benchmark --iterations 3

# 2x2 config (FSDP=2 trainer, TP=2 generator)
python -m demos.gpu_direct_weight_sync.ipc_benchmark \
    --config demos/gpu_direct_weight_sync/qwen3_4b_2x2.yaml \
    --iterations 3
```

### In Application Code

```python
# Phase 2 via IPC (works for both 1x1 and 2x2):
await generator.update_weights_ipc.fanout(version=version, trainer=trainer)
```

---

## Requirements

- **Single-node deployment**: CUDA IPC is intra-node only
- **Same physical machine**: Trainer and generator must share GPU access
- **CUDA-capable GPUs**: Required for IPC handle creation
- **GPU visibility**: For multi-GPU configs, set `FORGE_IPC_GPU_VISIBILITY=1`
  - This allows generator workers to access trainer's GPU memory via IPC
- **/dev/shm space**: CUDA IPC uses shared memory for handle coordination
  - Clean up stale files if `/dev/shm` fills up: `rm -f /dev/shm/cuda.shm.* /dev/shm/torch_*`

---

## Configurations Tested

| Config | Trainer | Generator | Status |
|--------|---------|-----------|--------|
| 1x1 | 1 GPU | 1 GPU (TP=1) | ✅ Working |
| 2x2 | 2 GPU (FSDP) | 2 GPU (TP=2) | ✅ Working |

---

## Known Limitations

1. **Single-node only**: CUDA IPC doesn't work across network
2. **PP not supported**: Pipeline parallel trainers not yet supported
3. **Memory lifetime**: Trainer must keep tensors alive until transfer completes
4. **GPU visibility required**: Workers must see trainer's GPUs for IPC to work

---

## Summary

Phase 2 achieves significant speedups by:

1. ✅ Bypassing TorchStore entirely
2. ✅ Using 66-byte CUDA IPC handles instead of serialization
3. ✅ GPU-direct memory access via `rebuild_cuda_tensor()`
4. ✅ Direct parameter updates bypassing vLLM's weight loader
5. ✅ Supporting both single-GPU and multi-GPU (FSDP + TP) configs

| Config | Baseline | Phase 2 | Speedup |
|--------|----------|---------|---------|
| 1x1 | 90s | 3.5-4.7s | **19-26x** |
| 2x2 | ~10s | **1.29s** | **~7.7x** |
