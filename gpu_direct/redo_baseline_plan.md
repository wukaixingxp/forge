# Plan: Redo All Baseline Benchmarks with Real Weights

## Goal

Reproduce the complete optimization journey with accurate, reproducible benchmarks:

| Stage | Technique | Expected Sync Time |
|-------|-----------|-------------------|
| Baseline | Individual RPCs via MonarchRPC | ~78s |
| Phase 1 | Batched APIs | ~12.5s |
| Phase 2 | CUDA IPC | ~1.3-2s |

**Problem:** Our current "baseline" uses TorchStore which already has internal optimizations. We need to benchmark against the **true original baseline** (per-tensor MonarchRPC calls).

---

## Phase 0: Environment Setup

### 0.1 True Baseline Environment (Main Branch Forge)

The true baseline uses the **main branch forge** at `~/kai/forge` with no IPC/batching optimizations.

**Baseline Forge Location:** `~/kai/forge`
**Baseline Commit:** `1ad7e7c` (main branch)
```
1ad7e7c Fix appending index to replicas on recovery (#747)
f1bedc1 Create a HostMesh per replica
cd9e295 [vllm] Upgrade vllm version to v0.13.0 (#737)
```

**Baseline Conda Environment:** `baseline` (already exists)

### 0.2 Setup Baseline Environment

```bash
# Activate baseline environment
conda activate baseline

# Install main branch forge (no optimizations)
pip install -e ~/kai/forge

# Verify installation
python -c "import forge; print('Baseline forge installed')"
```

### 0.3 Setup Optimized Environment (Current)

```bash
# Use existing vllm environment with IPC optimizations
conda activate vllm
export PYTHONPATH="/home/dev/framework/torchforge/src:$PYTHONPATH"

# Verify IPC-enabled forge
python -c "from forge.actors.vllm.v1.forge_executor import ForgeWorkerWrapper; print('IPC forge available')"
```

### 0.4 Available GRPO Configs in Baseline

```bash
~/kai/forge/apps/grpo/
├── llama3_8b.yaml
├── qwen3_1_7b.yaml
└── qwen3_8b.yaml
```

**Note:** Need to create `qwen3_4b.yaml` config for baseline testing (copy from qwen3_1_7b.yaml and modify).

---

## Phase 1: Benchmark Matrix

We need to run **12 benchmarks** total:

| # | Model | Config | Sync Method | Environment |
|---|-------|--------|-------------|-------------|
| 1 | Qwen3-4B | 1x1 | Per-tensor RPC (Original) | forge_baseline |
| 2 | Qwen3-4B | 1x1 | Batched RPC (Phase 1) | forge_baseline + batch patch |
| 3 | Qwen3-4B | 1x1 | TorchStore | vllm (current) |
| 4 | Qwen3-4B | 1x1 | CUDA IPC (Phase 2) | vllm (current) |
| 5 | Qwen3-4B | 2x1 | Per-tensor RPC (Original) | forge_baseline |
| 6 | Qwen3-4B | 2x1 | Batched RPC (Phase 1) | forge_baseline + batch patch |
| 7 | Qwen3-4B | 2x1 | TorchStore | vllm (current) |
| 8 | Qwen3-4B | 2x1 | CUDA IPC (Phase 2) | vllm (current) |
| 9 | Qwen3-4B | 2x2 | Per-tensor RPC (Original) | forge_baseline |
| 10 | Qwen3-4B | 2x2 | Batched RPC (Phase 1) | forge_baseline + batch patch |
| 11 | Qwen3-4B | 2x2 | TorchStore | vllm (current) |
| 12 | Qwen3-4B | 2x2 | CUDA IPC (Phase 2) | vllm (current) |

---

## Phase 2: Investigation - Find True Baseline Code

### 2.1 Key Commits Identified

**TorchStore optimization history:**
```
1e75533 Add batched controller notification for put_batch  <- Phase 1 (batch notification)
c468cc2 Add put_batch() API for batched tensor storage     <- Phase 1 (batched push)
84db8ba Export get_batch from torchstore module
f8c8730 Add get_batch() API for batched tensor retrieval   <- Phase 1 (batched fetch)
4b0306c Add CudaIPC transport for intra-node GPU-direct    <- Phase 2 (IPC)
2c1c639 Add slice APIs for GPU-direct weight sync
a6716ec fix rdma scalar tensor move (#112)                 <- BASELINE (before optimizations)
```

**TorchForge optimization history:**
```
e37e489 Update Phase 2 documentation
d70a2bc Implement Phase 2 CUDA IPC for 2x2 FSDP+TP
ac798e9 Add FSDP support to Phase 2 IPC
063fce1 Implement Phase 2: CUDA IPC direct weight transfer  <- Phase 2
dc1a9ad Use batched ts.put_batch() in trainer push_weights  <- Phase 1 (uses batch)
224c1c2 Use batched ts.get_batch() in _WeightFetcher        <- Phase 1 (uses batch)
a3ccdd6 Add transport selection to weight sync benchmark
a1725c7 Add GPU-direct weight sync demo and benchmarks      <- BASELINE
```

**Baseline commits (before ANY optimization):**
- TorchStore: `a6716ec` or earlier
- TorchForge: Before `224c1c2` (the commit that started using batched APIs)

### 2.2 Questions Answered

1. **Where is the per-tensor RPC code?**
   - TorchStore still has `push()` and `get()` methods (single tensor)
   - The baseline used these per-tensor methods in a loop
   - Batched methods (`put_batch`, `get_batch`) were added later

2. **What commit introduced batched APIs?**
   - TorchStore: `f8c8730` (get_batch) and `c468cc2` (put_batch)
   - TorchForge: `224c1c2` (use get_batch) and `dc1a9ad` (use put_batch)

3. **Can we disable batching via config?**
   - Need to check if batch_size=1 or similar option exists
   - Alternative: checkout pre-batch commits to get true baseline

### 2.2 Code Locations to Check

```bash
# TorchStore - weight push/pull logic
/home/dev/framework/torchstore/torchstore/client.py
/home/dev/framework/torchstore/torchstore/api.py

# Forge - weight sync logic
/home/dev/framework/torchforge/src/forge/actors/trainer/titan.py  # push_weights()
/home/dev/framework/torchforge/src/forge/actors/vllm/v1/generator.py  # update_weights()

# Check for batch_size parameters
grep -r "batch_size" /home/dev/framework/torchstore/
grep -r "batch_size" /home/dev/framework/torchforge/src/forge/
```

---

## Phase 3: Create Baseline Benchmark Script

### 3.1 Option A: Find and Use Original Code

If we can find the pre-optimization commit:

```bash
#!/bin/bash
# baseline_benchmark.sh

# Activate baseline environment
conda activate forge_baseline
cd /home/dev/baseline_test/torchforge_baseline

# Run per-tensor RPC baseline
python -m apps.gpu_direct.main \
  --config apps/gpu_direct/qwen3_4b_1x1_baseline.yaml \
  2>&1 | tee /home/dev/framework/torchforge/apps/gpu_direct/benchmark_logs/qwen3_4b_1x1_per_tensor_rpc.log
```

### 3.2 Option B: Create a "Slow Path" Config

If batching is controlled by config, create unbatched configs:

```yaml
# qwen3_4b_1x1_unbatched.yaml
weight_sync:
  use_ipc: false
  batch_size: 1  # Force per-tensor transfers
  disable_batching: true  # If this option exists
```

### 3.3 Option C: Patch TorchStore to Disable Batching

The forge code has fallback paths! From `titan.py`:
```python
# Line 370-380
await ts.put_batch(keyed_params)  # Try batched first
# Fallback for older torchstore without put_batch
logger.warning("ts.put_batch not available, falling back to individual puts")
batch_size = 100  # Still batched in fallback!
```

**To get TRUE per-tensor baseline:**

Option C1: Rename batch methods in torchstore (force fallback):
```bash
cd /home/dev/framework/torchstore
# Temporarily rename put_batch/get_batch to force fallback
git stash
# Edit torchstore/api.py to rename put_batch -> _put_batch_disabled
```

Option C2: Patch forge to use batch_size=1:
```python
# In titan.py, change fallback batch_size from 100 to 1
batch_size = 1  # True per-tensor baseline
```

Option C3: Create a baseline branch:
```bash
cd /home/dev/framework/torchstore
git checkout a6716ec -b baseline_no_batch

cd /home/dev/framework/torchforge
git checkout 224c1c2~1 -b baseline_no_batch
```

---

## Phase 4: Execution Plan

### Step 1: Create Qwen3-4B Config for Baseline

```bash
# Copy and modify existing config
cd ~/kai/forge
cp apps/grpo/qwen3_1_7b.yaml apps/grpo/qwen3_4b.yaml

# Edit qwen3_4b.yaml:
# - Change model: "Qwen/Qwen3-4B"
# - Change trainer.model.flavor: 4B
# - Set trainer.training.steps: 10
```

### Step 2: Run TRUE Baseline (Main Branch Forge, ~78s expected)

```bash
# Activate baseline environment with main branch forge
conda activate baseline
pip install -e ~/kai/forge
cd ~/kai/forge

# Run GRPO for 10 steps - this is the TRUE baseline
python -m apps.grpo.main --config apps/grpo/qwen3_4b.yaml \
  2>&1 | tee /home/dev/framework/torchforge/apps/gpu_direct/benchmark_logs/true_baseline_main_branch.log

# Expected: ~78s per weight sync (per-tensor MonarchRPC)
```

### Step 3: Run Optimized Benchmarks (IPC-enabled Forge)

```bash
# Switch to optimized environment
conda activate vllm
export PYTHONPATH="/home/dev/framework/torchforge/src:$PYTHONPATH"
cd /home/dev/framework/torchforge

# Run TorchStore baseline (current "baseline" ~45-65s)
python -m apps.gpu_direct.main --config apps/gpu_direct/qwen3_4b_fsdp2_tp1_baseline.yaml \
  2>&1 | tee benchmark_logs/torchstore_2x1.log

# Run CUDA IPC (Phase 2, ~9s expected)
export FORGE_IPC_GPU_VISIBILITY=1
python -m apps.gpu_direct.main --config apps/gpu_direct/qwen3_4b_fsdp2_tp1.yaml \
  2>&1 | tee benchmark_logs/ipc_2x1.log
```

### Step 4: Run All 6 Configurations

```bash
# === TRUE BASELINE (main branch forge) ===
conda activate baseline
cd ~/kai/forge

# 1x1 baseline
python -m apps.grpo.main --config apps/grpo/qwen3_4b_1x1.yaml \
  2>&1 | tee /home/dev/framework/torchforge/apps/gpu_direct/benchmark_logs/true_baseline_1x1.log

# 2x1 baseline
python -m apps.grpo.main --config apps/grpo/qwen3_4b_2x1.yaml \
  2>&1 | tee /home/dev/framework/torchforge/apps/gpu_direct/benchmark_logs/true_baseline_2x1.log

# 2x2 baseline
python -m apps.grpo.main --config apps/grpo/qwen3_4b_2x2.yaml \
  2>&1 | tee /home/dev/framework/torchforge/apps/gpu_direct/benchmark_logs/true_baseline_2x2.log

# === IPC OPTIMIZED (this repo) ===
conda activate vllm
export PYTHONPATH="/home/dev/framework/torchforge/src:$PYTHONPATH"
export FORGE_IPC_GPU_VISIBILITY=1
cd /home/dev/framework/torchforge

# 1x1 IPC
python -m apps.gpu_direct.main --config apps/gpu_direct/qwen3_4b_1x1.yaml \
  2>&1 | tee benchmark_logs/ipc_1x1.log

# 2x1 IPC
python -m apps.gpu_direct.main --config apps/gpu_direct/qwen3_4b_fsdp2_tp1.yaml \
  2>&1 | tee benchmark_logs/ipc_2x1.log

# 2x2 IPC
python -m apps.gpu_direct.main --config apps/gpu_direct/qwen3_4b_2x2.yaml \
  2>&1 | tee benchmark_logs/ipc_2x2.log
```

### Step 5: Extract and Compare Results

```bash
# Extract metrics from all logs
cd /home/dev/framework/torchforge/apps/gpu_direct

echo "=== TRUE BASELINE (main branch, per-tensor RPC) ===" > benchmark_logs/summary.txt
for log in benchmark_logs/true_baseline_*.log; do
  echo "$log:" >> benchmark_logs/summary.txt
  grep "weight_sync" "$log" | head -1 >> benchmark_logs/summary.txt
done

echo "" >> benchmark_logs/summary.txt
echo "=== IPC OPTIMIZED ===" >> benchmark_logs/summary.txt
for log in benchmark_logs/ipc_*.log; do
  echo "$log:" >> benchmark_logs/summary.txt
  grep "weight_sync/time_seconds" "$log" | head -1 >> benchmark_logs/summary.txt
done

cat benchmark_logs/summary.txt
```

### Step 6: Update Documentation

```bash
# Update blog.md with accurate numbers from real benchmarks
# Update README.md with reproduction steps
# Commit all logs and documentation
```

---

## Phase 5: Expected Results

| Stage | Environment | 1x1 | 2x1 | 2x2 | Notes |
|-------|-------------|-----|-----|-----|-------|
| **TRUE Baseline** | `~/kai/forge` (main) | ~78s | ~80s | ~85s | Per-tensor MonarchRPC |
| TorchStore | This repo (use_ipc=false) | ~50s | ~45s | ~65s | Internal batching |
| **CUDA IPC** | This repo (use_ipc=true) | ~10s | ~9s | ~13s | GPU-direct transfer |

**Speedup: TRUE Baseline → IPC = ~78s → ~9s = 8.7x improvement**

---

## Open Questions (Answered)

1. **Is the per-tensor RPC code still available?**
   - ✅ YES - Main branch forge at `~/kai/forge` uses per-tensor RPC
   - Baseline conda environment already exists

2. **What's the relationship between "Batched RPC" and "TorchStore"?**
   - TorchStore is the storage system used by main branch forge
   - Our optimized forge adds batched APIs (`put_batch`, `get_batch`) on top
   - The "78s baseline" is main branch forge with TorchStore but NO batching

3. **Hardware consistency:**
   - All benchmarks will run on same hardware: 4x H200 (143GB each)
   - Same model: Qwen3-4B
   - Same config: 10 training steps

---

## Files to Create/Modify

1. `benchmark_logs/baseline_per_tensor_*.log` - New baseline logs
2. `benchmark_logs/batched_rpc_*.log` - Phase 1 logs
3. `benchmark_logs/torchstore_*.log` - TorchStore logs
4. `benchmark_logs/ipc_*.log` - Phase 2 logs
5. `blog.md` - Update with accurate numbers
6. `README.md` - Update reproduction steps

---

## Success Criteria

- [ ] Per-tensor RPC baseline reproduces ~78s sync time
- [ ] Batched RPC shows ~6x improvement over per-tensor
- [ ] CUDA IPC shows ~60x improvement over per-tensor
- [ ] All benchmarks use real Qwen3-4B weights
- [ ] All benchmarks run on same hardware (4x H200)
- [ ] All results are reproducible with documented commands
- [ ] Logs saved in benchmark_logs/ with clear naming
