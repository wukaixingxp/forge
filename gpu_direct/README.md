# GPU-Direct Weight Sync GRPO

End-to-end GRPO training example with GPU-direct weight synchronization using CUDA IPC handles.

## Performance Comparison

| Method | Weight Sync Time | Speedup |
|--------|-----------------|---------|
| TorchStore (baseline) | ~10s | 1x |
| **CUDA IPC (this)** | **~1.3s** | **~8x** |

*Results from Qwen3-4B with FSDP=2 trainer + TP=2 generator*

## Quick Start

```bash
cd /home/dev/framework/torchforge
source /opt/conda/etc/profile.d/conda.sh && conda activate vllm

# Set environment
export PYTHONPATH="src:../torchstore:../torchtitan:$PYTHONPATH"
export FORGE_IPC_GPU_VISIBILITY=1  # Required for IPC

# Run with IPC weight sync (recommended)
python -m apps.gpu_direct.main --config apps/gpu_direct/qwen3_4b_2x2.yaml

# Compare with baseline (TorchStore)
python -m apps.gpu_direct.main --config apps/gpu_direct/qwen3_32b_2x2_baseline.yaml
```

## Configurations

| Config | Model | Trainer | Generator | Weight Sync |
|--------|-------|---------|-----------|-------------|
| `qwen3_4b_2x2.yaml` | Qwen3-4B | FSDP=2 | TP=2 | IPC |
| `qwen3_32b_2x2.yaml` | Qwen3-32B | FSDP=2 | TP=2 | IPC |
| `qwen3_32b_2x2_baseline.yaml` | Qwen3-32B | FSDP=2 | TP=2 | TorchStore |
| `qwen3_30b_moe_2x2.yaml` | Qwen3-30B-A3B (MoE) | FSDP=2 | TP=2 | IPC |

## Architecture

```
Trainer (FSDP=2)                    Generator (TP=2)
┌─────────────────┐                 ┌─────────────────┐
│ GPU 0 (shard 0) │                 │ GPU 2 (TP rank 0)│
│ GPU 1 (shard 1) │                 │ GPU 3 (TP rank 1)│
└────────┬────────┘                 └────────┬────────┘
         │                                   │
         │    CUDA IPC Handles (66 bytes)    │
         └───────────────────────────────────┘
                    GPU-Direct Transfer
```

### How IPC Weight Sync Works

1. **Trainer** exports CUDA IPC handles for each parameter tensor (66 bytes each)
2. **Handles sent** to generator workers via RPC (tiny payload)
3. **Generator** reconstructs tensors from handles using `rebuild_cuda_tensor()`
4. **Direct copy** to model parameters - no serialization, no intermediate storage

### Comparison with TorchStore (Baseline)

| Aspect | TorchStore | CUDA IPC |
|--------|------------|----------|
| Data path | GPU → CPU → Store → CPU → GPU | GPU → GPU (direct) |
| Serialization | Full tensor pickle | 66-byte handle |
| Intermediate storage | Yes (TorchStore) | No |
| Multi-node support | Yes | Single-node only |

## Configuration Options

### Weight Sync Configuration

```yaml
weight_sync:
  use_ipc: true   # Enable CUDA IPC (default: false)
```

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `FORGE_IPC_GPU_VISIBILITY=1` | Yes (for IPC) | Allow all GPUs visible for IPC handles |

## Requirements

- **Single-node deployment**: CUDA IPC is intra-node only
- **4+ GPUs**: 2 for trainer (FSDP), 2 for generator (TP)
- **CUDA-capable GPUs**: Required for IPC handle creation
- **PyTorch 2.9+**: For CUDA IPC tensor operations

## Metrics

The app logs weight sync timing to both console and WandB:

- `weight_sync/time_seconds` - Time per weight sync operation
- `weight_sync/use_ipc` - 1.0 if IPC mode, 0.0 if TorchStore

## Troubleshooting

### IPC handles fail to reconstruct

```bash
# Clean up stale CUDA shared memory
rm -f /dev/shm/cuda.shm.* /dev/shm/torch_*
```

### GPU visibility issues

```bash
# Ensure all GPUs are visible
export FORGE_IPC_GPU_VISIBILITY=1
nvidia-smi  # Should show all GPUs
```

### Out of memory

- Reduce `max_model_len` in generator config
- Reduce `local_batch_size`
- Use smaller model (qwen3_4b_2x2.yaml for testing)

## Files

```
apps/gpu_direct/
├── __init__.py
├── main.py                      # Main training loop with IPC support
├── data.py                      # GSM8K dataset actor
├── grading.py                   # Math and thinking reward functions
├── README.md                    # This file
├── qwen3_4b_2x2.yaml           # Quick test config (IPC)
├── qwen3_32b_2x2.yaml          # Production config (IPC)
├── qwen3_32b_2x2_baseline.yaml # Baseline comparison (TorchStore)
└── qwen3_30b_moe_2x2.yaml      # MoE model config (IPC)
```

## Related Documentation

- `demos/gpu_direct_weight_sync/summary.md` - Full technical documentation
- `demos/gpu_direct_weight_sync/phase2_summary.md` - Phase 2 implementation details
- `src/forge/actors/trainer/titan.py` - `push_weights_ipc()` implementation
- `src/forge/actors/vllm/v1/generator.py` - `update_weights_ipc()` implementation
