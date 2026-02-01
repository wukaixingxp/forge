# GPU-Direct Weight Sync - Session Checkpoint

**Date:** 2026-02-01
**Status:** Phase 2 Complete (CUDA IPC for 2x2 FSDP+TP)

## Quick Context

We optimized weight synchronization between trainer and generator in online RL from **78s → 1.3s** (60x improvement).

### Results Summary

| Phase | Technique | Time | Speedup |
|-------|-----------|------|---------|
| Baseline | Individual RPCs via MonarchRPC | 78s | — |
| Phase 1 | Batched TorchStore APIs | 12.5s | 6.2x |
| Phase 2 | CUDA IPC (2x2 FSDP+TP) | 1.3s | 60x |

## Key Files to Load

### Essential (load these first)
1. **This file** - `checkpoint-gpu-direct.md`
2. **`phase2_summary.md`** - Detailed Phase 2 implementation notes
3. **`summary.md`** - Full technical documentation

### Source Code (reference as needed)
- `src/forge/actors/trainer/titan.py` - `push_weights_ipc()`, `push_weights_batched()`
- `src/forge/actors/vllm/v1/forge_executor.py` - `receive_weights_ipc_sliced()`, TP slicing
- `src/forge/actors/vllm/v1/generator.py` - Generator-side weight sync endpoints
- `src/forge/actors/vllm/v1/monarch_executor.py` - `FORGE_IPC_GPU_VISIBILITY` handling

### TorchStore Changes (in ~/kai/torchstore)
- `torchstore/api.py` - `get_batch()`, `put_batch()`, slice APIs
- `torchstore/transport/cuda_ipc.py` - CUDA IPC transport implementation

## How to Run Benchmarks

```bash
cd /home/dev/framework/torchforge
source /opt/conda/etc/profile.d/conda.sh && conda activate vllm

# Set environment
export PYTHONPATH="src:../torchstore:../torchtitan:$PYTHONPATH"
export FORGE_IPC_GPU_VISIBILITY=1

# 2x2 FSDP+TP benchmark (Phase 2)
python -m demos.gpu_direct_weight_sync.baseline_1x1 \
    --config demos/gpu_direct_weight_sync/qwen3_4b_2x2.yaml \
    --iterations 3 \
    --use-ipc

# 1x1 simple benchmark
python -m demos.gpu_direct_weight_sync.baseline_1x1 \
    --config demos/gpu_direct_weight_sync/qwen3_4b_demo.yaml \
    --iterations 3
```

## Architecture Overview

```
Trainer (FSDP)                    Generator (TP)
┌─────────────────┐              ┌─────────────────┐
│ GPU 0 (shard 0) │──┐           │ GPU 2 (TP rank 0)│
│ GPU 1 (shard 1) │──┼─ CUDA IPC ┼│ GPU 3 (TP rank 1)│
└─────────────────┘  │           └─────────────────┘
                     │
              IPC handles sent via RPC
              Actual data: GPU-direct
```

## Key Technical Details

### CUDA IPC Flow
1. Trainer exports IPC handles for each parameter tensor
2. Handles sent to generator via RPC (small, ~66 bytes each)
3. Generator reconstructs tensors from handles (GPU-direct, no copy)
4. Generator slices for TP and copies to model parameters

### vLLM Merged Weight Mapping
```
HuggingFace → vLLM:
  q_proj, k_proj, v_proj → qkv_proj (concatenated)
  gate_proj, up_proj → gate_up_proj (concatenated)
```

### Critical Environment Variables
- `FORGE_IPC_GPU_VISIBILITY=1` - Allow all GPUs visible for IPC
- `TORCHSTORE_TRACE_TRANSFERS=1` - Debug logging for transfers

## Git Branches

Your fork branches (in ~/kai/):
- `~/kai/forge` branch: `gpu-direct` (17 commits)
- `~/kai/torchstore` branch: `gpu-direct` (1 squashed commit)

Framework repo (in /home/dev/framework/):
- `torchforge` - 16 commits ahead of origin/main
- `torchstore` - 6 commits ahead of origin/main

## Potential Next Steps

1. **Phase 3: TorchComms RDMA** - For multi-node (requires InfiniBand)
2. **Quantization-aware sync** - Skip FP8 scales (already unchanged)
3. **Incremental sync** - Only sync changed layers
4. **Integration into main training loop** - Replace baseline weight sync

## Validation

Weight sync correctness was validated:
- Added noise to trainer weights
- Synced to generator
- Verified weights changed (399 weights updated correctly)
- Only FP8 quantization scales unchanged (expected)

## Files Created This Session

```
demos/gpu_direct_weight_sync/
├── blog.md                 # Blog post about the optimization journey
├── checkpoint-gpu-direct.md # This file
├── phase2_summary.md       # Phase 2 detailed notes
├── summary.md              # Full technical documentation
├── DESIGN.md               # Architecture and design decisions
├── baseline_1x1.py         # Main benchmark script
├── qwen3_4b_2x2.yaml       # 2x2 FSDP+TP config
└── qwen3_4b_demo.yaml      # 1x1 simple config
```
