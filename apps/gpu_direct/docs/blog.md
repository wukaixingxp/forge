# Optimizing RL Weight Sync: 4.3x Faster Training Steps with CUDA IPC

Weight synchronization between trainer and generator was killing our RL training loop. Here's how we fixed it.

---

## TL;DR: 4.3x Faster Training Steps

![Training Breakdown](diagrams/00-tldr-training-breakdown.excalidraw)

**Steady-state training step comparison on Qwen3-4B (4x H200 GPUs):**

```
BASELINE (TorchStore RPC) - Step 2:
├── train_step         0.4s  █
├── update_weights    31.2s  ███████████████████████████████  ← BOTTLENECK
│   ├── pause_gen      7.6s  ████████
│   └── worker_load   23.6s  ████████████████████████  (RPC transfer)
└── TOTAL:            47.6s

CUDA IPC - Step 2:
├── train_step         0.4s  █
├── update_weights    10.6s  ███████████  ← 3x FASTER
│   ├── pause_gen      9.7s  ██████████
│   └── worker_load    0.9s  █  (GPU-direct transfer, 26x faster!)
└── TOTAL:            11.0s  ← 4.3x FASTER
```

![Step 2 Breakdown](diagrams/04-step2-breakdown.excalidraw)

### The Numbers (Verified from Benchmark Logs)

| Metric | Baseline | IPC | Speedup |
|--------|----------|-----|---------|
| **Total step time** | 47.6s | 11.0s | **4.3x** |
| Weight sync | 31.2s | 10.6s | **2.9x** |
| └─ pause_generation | 7.6s | 9.7s | ~same |
| └─ worker_load (transfer) | 23.6s | 0.9s | **26x** |
| train_step | 0.4s | 0.4s | ~same |

**Key insight:** The actual data transfer (`worker_load_weights`) went from 23.6s to 0.9s — **26x faster**. The `pause_generation` time is unavoidable (waiting for in-flight requests), but the GPU-direct transfer is nearly instant.

---

## Why Weight Sync Matters

Online reinforcement learning has a fundamental loop: **train → sync → generate → repeat**.

```
┌─────────────┐                      ┌─────────────┐
│   Trainer   │ ──── sync weights ───▶│  Generator  │
│  (learns)   │                      │ (explores)  │
└─────────────┘                      └─────────────┘
      ▲                                     │
      └──────────── new experience ─────────┘
```

During weight sync, nothing productive happens. The trainer can't update weights it's sending. The generator can't use weights it hasn't received. At 31-51 seconds per sync with the baseline approach (depending on parallelism config), weight sync was dominating our training time.

---

## The Three Approaches

![Data Flow Comparison](diagrams/01-data-flow-comparison.excalidraw)

### 1. Per-Tensor RPC (Baseline) — Slow

```python
# ~400 individual RPC calls, one per tensor
for name, tensor in model.named_parameters():
    await storage.put(name, tensor)  # ~35ms overhead per call
```

**Data path:** `GPU → CPU → Serialize → Network → Deserialize → CPU → GPU`

**Result:** 14s push + 24s fetch = **~38s total** (requires extended timeout)

The per-tensor fetch exceeds Monarch's default 30s timeout. Set `HYPERACTOR_MESSAGE_DELIVERY_TIMEOUT=120s` to complete successfully.

### 2. Batched RPC (Phase 1) — 1.7x Faster

```python
# ~8 RPC calls instead of 800
batches = chunk(model.named_parameters(), batch_size=100)
for batch in batches:
    await storage.put_batch(batch)
```

**Insight:** Each RPC has fixed overhead (serialization, round-trip, dispatch). Batching pays this tax once per batch instead of once per tensor.

**Result:** 14s push + 8s fetch = **~22s total**

**Code:** Available in `batch_fetch` branch of `~/kai/forge`

### 3. CUDA IPC (Phase 2) — 2.9x Faster

```python
# Zero-copy GPU-to-GPU transfer
handle = tensor.storage()._share_cuda_()  # 66 bytes
await generator.receive_handle(handle, shape, dtype)
# Generator reconstructs tensor directly from trainer's GPU memory
```

**Data path:** `GPU ═══════════ GPU` (shared memory, no CPU copies)

**Result:** **~10.6s total** (9.7s pause + 0.9s transfer)

---

## CUDA IPC: Challenges We Solved

### Challenge 1: GPU Visibility

CUDA IPC handles reference specific GPU memory. When the generator starts with `CUDA_VISIBLE_DEVICES=2,3`, it can't see GPUs 0,1 where trainer memory lives.

```python
# Solution: Give all processes visibility for IPC
if os.environ.get("FORGE_IPC_GPU_VISIBILITY") == "1":
    cuda_devices = ",".join(str(i) for i in range(torch.cuda.device_count()))
```

### Challenge 2: vLLM's Merged Weights

vLLM merges certain weights for efficiency:

```
HuggingFace:              vLLM:
  q_proj.weight    ─┐
  k_proj.weight    ─┼──▶  qkv_proj.weight
  v_proj.weight    ─┘
```

We needed a translation layer to map HuggingFace names to vLLM's merged parameters.

### Challenge 3: Tensor Parallel Slicing

With TP=2, each generator GPU holds half of certain weight matrices. The trainer sends full weights; the generator must slice correctly for its rank.

### Challenge 4: GQA (Grouped Query Attention) Slicing

Modern models like Qwen3 use GQA where Q has more heads than K/V:

```
Qwen3-4B head configuration:
  Q: 16 attention heads → 2048 output dim
  K: 4 KV heads         → 512 output dim  ← Different!
  V: 4 KV heads         → 512 output dim

vLLM merges these into qkv_proj.weight:
  Total: [3072, 2048] = [Q + K + V, hidden_size]

With TP=2, each rank gets:
  TP rank 0: Q[0:1024], K[0:256], V[0:256] → [1536, 2048]
  TP rank 1: Q[1024:2048], K[256:512], V[256:512] → [1536, 2048]
```

The naive approach (`part_size = total_qkv_size // 3`) assumes equal Q/K/V sizes and fails for GQA models. We fixed this by querying vLLM's model config:

```python
# Get head counts from vLLM config
num_attention_heads = model_config.get_num_attention_heads(parallel_config)  # 16
num_kv_heads = model_config.get_num_kv_heads(parallel_config)  # 4
head_dim = model_config.get_head_size()  # 128

# Calculate correct sizes per TP rank
q_size_per_rank = (num_attention_heads // tp_size) * head_dim  # 1024
kv_size_per_rank = (num_kv_heads // tp_size) * head_dim  # 256
```

### Challenge 5: The /dev/shm Ghost Problem

CUDA IPC stores handles (66 bytes each) in `/dev/shm`. Crashed processes leave orphaned handles. After enough experiments:

```bash
$ df -h /dev/shm
tmpfs  189G  189G  0  100%  /dev/shm  # Full!

$ rm -f /dev/shm/cuda.shm.* /dev/shm/torch_*  # Fix
```

---

## Deep Dive: FSDP and Weight Sync

The complexity of weight synchronization increases significantly with parallelism. This section explains what happens under the hood for each configuration.

![FSDP Weight Sync Flow](diagrams/05-fsdp-weight-sync.excalidraw)

### Understanding FSDP2 (Fully Sharded Data Parallel)

FSDP2 shards model weights across multiple GPUs to reduce per-GPU memory usage. For a model with N parameters:

```
Without FSDP (1 GPU):
┌─────────────────────────────────────────────┐
│              GPU 0 (100% of model)          │
│  [param_0, param_1, param_2, ..., param_N]  │
└─────────────────────────────────────────────┘

With FSDP=2 (2 GPUs):
┌─────────────────────┐  ┌─────────────────────┐
│   GPU 0 (50%)       │  │   GPU 1 (50%)       │
│  [shard_0, shard_2] │  │  [shard_1, shard_3] │
└─────────────────────┘  └─────────────────────┘
```

Each parameter becomes a `DTensor` with only its local shard stored. When you call `model.state_dict()`, you get sharded DTensors, NOT full tensors.

### The All-Gather Problem (Baseline Only)

The **baseline** approach requires reconstructing full tensors from shards using an `all_gather` collective operation:

```
Baseline: DTensor.full_tensor() triggers all_gather:

GPU 0: [shard_0] ──┐              ┌──▶ [full_tensor] on GPU 0
                   ├──all_gather──┤
GPU 1: [shard_1] ──┘              └──▶ [full_tensor] on GPU 1

Result: ALL FSDP ranks now have the complete tensor
Peak memory: 2x per rank during gather (shard + full)
```

**Critical insight:** CUDA IPC does NOT require all_gather! IPC handles work directly with the local DTensor shard, bypassing the expensive collective entirely. This is the primary source of the 26x speedup in data transfer.

### Configuration 1x1: Simple Direct Transfer

**Setup:** Trainer 1 GPU, Generator 1 GPU (no parallelism)

```
┌────────────────┐              ┌────────────────┐
│    Trainer     │              │   Generator    │
│     GPU 0      │              │     GPU 1      │
│                │              │                │
│  Full Model    │──IPC Handle─▶│  Full Model    │
│  [4.4B params] │   (66 bytes) │  (vLLM loads)  │
└────────────────┘              └────────────────┘

Data path: param.data → create_ipc_handle → send handle → reconstruct
No gathering needed - model is already complete on one GPU
```

**Baseline flow:**
1. `model.named_parameters()` → iterate tensors
2. `ts.put(key, tensor)` → serialize + RPC to TorchStore
3. Generator: `ts.get(key)` → deserialize + copy to GPU

**IPC flow:**
1. `model.named_parameters()` → iterate tensors
2. `create_ipc_handle(tensor)` → 66-byte handle
3. Send handle to generator (no serialization)
4. Generator: `handle.reconstruct_tensor()` → direct GPU memory access

### Configuration 2x1: FSDP Trainer, Full Generator

**Setup:** Trainer FSDP=2 (2 GPUs), Generator 1 GPU

![2x1 Configuration](diagrams/06-config-2x1.excalidraw)

```
BASELINE (with all_gather):              IPC (no all_gather):
┌────────────────────────────┐           ┌────────────────────────────┐
│      TRAINER (FSDP=2)      │           │      TRAINER (FSDP=2)      │
│  GPU 0      GPU 1          │           │  GPU 0      GPU 1          │
│  [shard_0]  [shard_1]      │           │  [shard_0]  [shard_1]      │
│      │          │          │           │      │                     │
│      └──all_gather─┘       │           │      │ (no gather!)        │
│            │               │           │      │                     │
│     [full_tensor]          │           │  IPC handle for shard_0    │
│            │               │           │      │                     │
│     serialize + RPC        │           │      │ direct GPU access   │
└────────────┬───────────────┘           └──────┬─────────────────────┘
             │                                   │
             ▼                                   ▼
┌────────────────────────────┐           ┌────────────────────────────┐
│    GENERATOR (TP=1)        │           │    GENERATOR (TP=1)        │
│    deserialize + copy      │           │    reconstruct + gather    │
│    Full Model (4.4B)       │           │    Full Model (4.4B)       │
└────────────────────────────┘           └────────────────────────────┘
```

**Baseline flow (32.0s total):**
1. `model.state_dict()` → returns DTensors (sharded)
2. For each DTensor: `param.full_tensor()` → **all_gather** (GPU collective)
3. Only rank 0: `ts.put_batch(gathered_tensors)` → serialize + RPC
4. Generator: `ts.get_batch(keys)` → deserialize + GPU copy

**IPC flow (9.6s total) — NO all_gather on trainer!**
1. `model.state_dict()` → returns DTensors (sharded)
2. DTensor operations proxy to `._local_tensor` (local shard)
3. `create_ipc_handle(dtensor)` → 66-byte handle pointing to **local shard GPU memory**
4. Send handles → Generator reconstructs tensors directly from trainer GPU shards
5. Generator combines shards into full model internally

**Key insight:** IPC bypasses the expensive `full_tensor()` all_gather entirely! The DTensor's `_typed_storage()` method returns the local shard's storage, allowing IPC to work directly with sharded memory. This eliminates both:
- The all_gather collective (O(model_size) GPU communication)
- Serialization/RPC overhead

**Timing breakdown (from 2x1 IPC benchmark):**
- Handle creation: **0.038s** (proves no all_gather — would take seconds otherwise)
- Total push: **0.34s**
- pause_generation: **7.3s**
- worker_load: **1.7s**
- **Total: 9.6s** (vs 32.0s baseline = **3.3x faster**)

### Configuration 2x2: FSDP Trainer, TP Generator

**Setup:** Trainer FSDP=2 (2 GPUs), Generator TP=2 (2 GPUs)

![2x2 Configuration](diagrams/07-config-2x2.excalidraw)

This is the most complex configuration. The baseline must gather on trainer, while IPC can work with shards directly:

```
BASELINE (with all_gather):              IPC (no all_gather on trainer):
┌────────────────────────────┐           ┌────────────────────────────┐
│      TRAINER (FSDP=2)      │           │      TRAINER (FSDP=2)      │
│  GPU 0      GPU 1          │           │  GPU 0      GPU 1          │
│  [shard_0]  [shard_1]      │           │  [shard_0]  [shard_1]      │
│      │          │          │           │      │          │          │
│      └──all_gather─┘       │           │  IPC handle  IPC handle    │
│            │               │           │  (shard_0)   (shard_1)     │
│     [full_tensor]          │           │      │          │          │
│            │               │           │      └────┬─────┘          │
│     serialize + RPC        │           │           │                │
└────────────┬───────────────┘           └───────────┼────────────────┘
             │                                       │
             ▼                                       ▼
┌────────────────────────────┐           ┌────────────────────────────┐
│    GENERATOR (TP=2)        │           │    GENERATOR (TP=2)        │
│  TP0: deserialize + slice  │           │  Both ranks reconstruct    │
│  TP1: deserialize + slice  │           │  from BOTH shards, then    │
│                            │           │  combine + slice for TP    │
└────────────────────────────┘           └────────────────────────────┘
```

**The slicing challenge (GQA models):**

For GQA (Grouped Query Attention) models like Qwen3, Q/K/V have different sizes:
- Q: 16 attention heads → 2048 output dim
- K: 4 KV heads → 512 output dim
- V: 4 KV heads → 512 output dim

vLLM merges these into `qkv_proj.weight` with shape `[3072, 2048]`.

With TP=2, each generator GPU gets half:
```
Full qkv_proj:      TP rank 0:         TP rank 1:
[Q: 2048]           [Q: 1024]          [Q: 1024]
[K: 512 ]     →     [K: 256 ]          [K: 256 ]
[V: 512 ]           [V: 256 ]          [V: 256 ]
[total: 3072]       [total: 1536]      [total: 1536]
```

**Baseline flow (51.2s total):**
1. Trainer: `full_tensor()` → **explicit all_gather** (GPU collective)
2. Push full tensors to TorchStore (serialize + RPC)
3. Generator workers: fetch full tensors, slice for TP rank

**IPC flow (15.8s total) — NO all_gather on trainer!**
1. Trainer: `model.state_dict()` → DTensors (sharded)
2. `create_ipc_handle(dtensor)` → handles point to **local shard memory**
3. Send handles from all FSDP ranks to generator
4. Generator: reconstruct from shards → combine → slice for TP rank

**Timing breakdown (from 2x2 IPC benchmark):**
- pause_generation: **10.5s**
- worker_load: **4.7s**
- **Total: 15.8s** (vs 51.2s baseline = **3.2x faster**)

---

## Configuration Comparison

| Config | Baseline | IPC | Speedup | Transfer Speedup |
|--------|----------|-----|---------|------------------|
| 1x1 (FSDP=1, TP=1) | 31.2s | 10.6s | **2.9x** | 26x (23.6s → 0.9s) |
| 2x1 (FSDP=2, TP=1) | 32.0s | 9.6s | **3.3x** | 15x (26.4s → 1.7s) |
| 2x2 (FSDP=2, TP=2) | 51.2s | 15.8s | **3.2x** | 9x (39.9s → 4.7s) |

*Note: Times shown are `update_weights` duration. Transfer speedup compares `worker_load_weights` times.*

**Why IPC is dramatically faster:**
1. **No all_gather on trainer** — IPC works directly with FSDP shards
2. **No serialization** — 66-byte handles vs multi-GB tensor data
3. **GPU-direct memory access** — NVLink/PCIe P2P, no CPU involvement

**Why baseline 2x2 is slowest:** Two expensive operations:
1. Trainer-side all_gather for FSDP (GPU collective)
2. More parameters to transfer (full model to 2 generator GPUs)

**Recommendation:**
- **FSDP=2, TP=1** (9.6s) — Best sync speed, use when generator fits on 1 GPU
- **FSDP=2, TP=2** (15.8s) — Use when model requires tensor parallelism for memory

---

## Understanding pause_generation

The generator must pause and wait for in-flight requests before updating weights:

```
IPC Breakdown (10.6s total):
├── pause_generation  9.7s  ████████████████████████████  (unavoidable)
└── IPC transfer      0.9s  ███
```

This ~10s pause is **unavoidable** regardless of transfer method. It's the time to:
1. Wait for in-flight generation requests to complete
2. Clear the KV cache before loading new weights

The pause scales with `max_tokens / generation_speed`:
- 512 tokens @ 50 tok/s ≈ 10s
- 128 tokens @ 50 tok/s ≈ 2.5s

---

## Key Takeaways

1. **Measure first.** We assumed bandwidth was the bottleneck. It was RPC overhead.

2. **Batch aggressively.** One RPC with 100 tensors beats 100 RPCs with 1 tensor.

3. **Eliminate copies.** GPU-direct transfers skip CPU entirely.

4. **Test real configs.** Our 1x1 prototype worked. Production 2x2 surfaced three new bugs.

5. **The pause is irreducible.** After optimizing transfer, generator pause becomes the floor.

---

## Reproducing Our Results

### Environment Setup

```bash
# Required: Increase Monarch timeout (default 30s is too short for baseline)
export HYPERACTOR_MESSAGE_DELIVERY_TIMEOUT=120s
export HYPERACTOR_HOST_SPAWN_READY_TIMEOUT=120s
```

### Per-Tensor Baseline (slow, requires extended timeout)

```bash
cd ~/kai/forge && git checkout main
conda activate baseline
python -m apps.grpo.main --config apps/grpo/qwen3_4b_1x1.yaml
# Expected: Push ~14s, Fetch ~24s, Total ~38s
```

### Batched RPC (Phase 1)

```bash
cd ~/kai/forge && git checkout batch_fetch
conda activate baseline
python -m apps.grpo.main --config apps/grpo/qwen3_4b_1x1.yaml
# Expected: Push ~14s, Fetch ~8s, Total ~22s
```

### CUDA IPC (Phase 2)

```bash
cd /home/dev/framework/torchforge
conda activate vllm
export PYTHONPATH="src:$PYTHONPATH"
export FORGE_IPC_GPU_VISIBILITY=1
python -m apps.gpu_direct.main --config apps/gpu_direct/qwen3_4b_1x1.yaml
# Expected: Total ~10.6s (including ~9.7s pause)
```

---

## Architecture Diagrams

See the `diagrams/` folder:
- [00-tldr-training-breakdown.excalidraw](diagrams/00-tldr-training-breakdown.excalidraw) — Training step breakdown
- [01-data-flow-comparison.excalidraw](diagrams/01-data-flow-comparison.excalidraw) — Data flow for all three approaches
- [02-timeline-comparison.excalidraw](diagrams/02-timeline-comparison.excalidraw) — Timeline with verified numbers
- [03-rl-training-loop.excalidraw](diagrams/03-rl-training-loop.excalidraw) — Where weight sync fits in RL
- [05-fsdp-weight-sync.excalidraw](diagrams/05-fsdp-weight-sync.excalidraw) — FSDP all_gather + IPC flow
- [06-config-2x1.excalidraw](diagrams/06-config-2x1.excalidraw) — Config 2x1: FSDP trainer → Full generator
- [07-config-2x2.excalidraw](diagrams/07-config-2x2.excalidraw) — Config 2x2: FSDP trainer → TP generator (GQA)

---

## Raw Benchmark Data

All benchmark logs are in `apps/gpu_direct/benchmark_logs/`:

| Log File | What It Measures |
|----------|------------------|
| `baseline_step2_1x1.log` | 1x1 Baseline: 31.2s update_weights |
| `ipc_1x1.log` | 1x1 IPC: 10.6s (9.7s pause + 0.9s transfer) |
| `qwen3_4b_2x1_baseline.log` | 2x1 Baseline: 32.0s update_weights |
| `ipc_2x1.log` | 2x1 IPC: 9.6s (7.3s pause + 1.7s transfer) |
| `qwen3_4b_2x2_baseline.log` | 2x2 Baseline: 51.2s update_weights |
| `2x2_ipc_benchmark_clean.log` | 2x2 IPC: 15.8s (10.5s pause + 4.7s transfer) |
| `phase1_batched_rpc_1x1.log` | Batched RPC: 14s push, 8s fetch = ~22s |

---

*Thanks to the TorchForge team for making this work possible.*
