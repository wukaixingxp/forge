# Benchmark Logs

Raw logs from GPU-Direct Weight Sync E2E benchmarks on Qwen3 models.

## Environment & Reproduction

### Hardware
- 4x NVIDIA H200 (143GB each)
- NVLink interconnect

### Software Versions
- **torchforge commit:** `e37e489f80241d8637befa73288e18d3865a377e`
- **Repository:** `git@github.com:meta-pytorch/torchforge.git`
- **Date:** 2026-02-01
- **Python:** 3.12
- **PyTorch:** 2.9+ (nightly)
- **vLLM:** 0.13.0

### How to Reproduce

```bash
# 1. Clone and checkout the correct commit
git clone git@github.com:meta-pytorch/torchforge.git
cd torchforge
git checkout e37e489f80241d8637befa73288e18d3865a377e

# 2. Activate environment (conda or venv with dependencies)
conda activate vllm  # or your environment with forge installed
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"

# 3. Run IPC benchmarks (requires FORGE_IPC_GPU_VISIBILITY=1)
export FORGE_IPC_GPU_VISIBILITY=1

# 2x1 IPC (FSDP=2, TP=1) - Recommended config
python -m apps.gpu_direct.main --config apps/gpu_direct/qwen3_4b_fsdp2_tp1.yaml 2>&1 | tee benchmark_logs/qwen3_4b_2x1_ipc.log

# 1x1 IPC (FSDP=1, TP=1) - Simplest config
python -m apps.gpu_direct.main --config apps/gpu_direct/qwen3_4b_1x1.yaml 2>&1 | tee benchmark_logs/qwen3_4b_1x1_ipc.log

# 2x2 IPC (FSDP=2, TP=2) - Full parallelism
python -m apps.gpu_direct.main --config apps/gpu_direct/qwen3_4b_2x2.yaml 2>&1 | tee benchmark_logs/qwen3_4b_2x2_ipc.log

# 4. Run TorchStore baseline benchmarks (no FORGE_IPC_GPU_VISIBILITY needed)
unset FORGE_IPC_GPU_VISIBILITY

# 2x1 Baseline
python -m apps.gpu_direct.main --config apps/gpu_direct/qwen3_4b_fsdp2_tp1_baseline.yaml 2>&1 | tee benchmark_logs/qwen3_4b_2x1_baseline.log

# 1x1 Baseline
python -m apps.gpu_direct.main --config apps/gpu_direct/qwen3_4b_1x1_baseline.yaml 2>&1 | tee benchmark_logs/qwen3_4b_1x1_baseline.log

# 2x2 Baseline
python -m apps.gpu_direct.main --config apps/gpu_direct/qwen3_4b_2x2_baseline.yaml 2>&1 | tee benchmark_logs/qwen3_4b_2x2_baseline.log
```

### Config Files
```
apps/gpu_direct/
├── qwen3_4b_1x1.yaml              # IPC, FSDP=1, TP=1
├── qwen3_4b_1x1_baseline.yaml     # TorchStore, FSDP=1, TP=1
├── qwen3_4b_fsdp2_tp1.yaml        # IPC, FSDP=2, TP=1 (recommended)
├── qwen3_4b_fsdp2_tp1_baseline.yaml
├── qwen3_4b_2x2.yaml              # IPC, FSDP=2, TP=2
├── qwen3_4b_2x2_baseline.yaml
├── qwen3_32b_fsdp2_tp1.yaml       # 32B model configs
└── qwen3_32b_fsdp2_tp1_baseline.yaml
```

### Key Config Differences

**IPC configs (`use_ipc: true`):**
```yaml
weight_sync:
  use_ipc: true  # Enable CUDA IPC direct transfer
```

**Baseline configs (`use_ipc: false`):**
```yaml
weight_sync:
  use_ipc: false  # Use TorchStore

generator:
  prefetch_weights_to_shm: false  # Disable shared memory prefetch
```

## Available Logs

| File | Model | Config | Mode | weight_sync/time_seconds |
|------|-------|--------|------|--------------------------|
| `qwen3_4b_1x1_ipc.log` | 4B | FSDP=1, TP=1 | IPC | **10.33s** |
| `qwen3_4b_1x1_baseline.log` | 4B | FSDP=1, TP=1 | TorchStore | **50.10s** |
| `qwen3_4b_2x1_baseline.log` | 4B | FSDP=2, TP=1 | TorchStore | **45.47s** |
| `qwen3_4b_2x2_baseline.log` | 4B | FSDP=2, TP=2 | TorchStore | **65.08s** |
| `qwen3_4b_2x2_ipc.log` | 4B | FSDP=2, TP=2 | IPC | (incomplete) |
| `qwen3_32b_2x1_ipc_incomplete.log` | 32B | FSDP=2, TP=1 | IPC | (timed out) |

## Missing Logs

The following runs were observed during the session but logs were not saved:

| Config | Mode | Observed Result | Source |
|--------|------|-----------------|--------|
| 2x1 (FSDP=2, TP=1) | IPC | **9.11s** | e2e_test_summary.md |
| 2x2 (FSDP=2, TP=2) | IPC | **12.80s** | e2e_test_summary.md |

## Key Metrics in Logs

```
weight_sync/time_seconds           - Total weight sync time
weight_sync/use_ipc                - 1.0 = IPC, 0.0 = TorchStore
generator_perf/update_weights_ipc/pause_generation_duration_s
generator_perf/update_weights_ipc/worker_load_weights_duration_s
main_perf/continuous_training/push_weights/duration_avg_s
main_perf/continuous_training/update_weights/duration_avg_s
rl_trainer_perf/step/forward_backward/duration_avg_s
```

## How to Read Logs

```bash
# Get weight sync time
grep "weight_sync/time_seconds" <logfile>

# Get IPC breakdown
grep "update_weights_ipc" <logfile>

# Get baseline breakdown
grep "push_weights\|update_weights" <logfile>
```

## Summary Table

| Config | TorchStore (Baseline) | IPC | Speedup |
|--------|----------------------|-----|---------|
| 1x1 | 50.1s | 10.3s | 4.9x |
| 2x1 | 45.5s | 9.1s | 5.0x |
| 2x2 | 65.1s | 12.8s | 5.1x |

**Note:** "Baseline" here refers to TorchStore, NOT the original per-tensor RPC approach (78s) from Phase 1 of the optimization work.
