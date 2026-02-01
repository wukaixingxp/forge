# Optimizing RL Weight Sync: 3.6x Faster with CUDA IPC

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

During weight sync, nothing productive happens. The trainer can't update weights it's sending. The generator can't use weights it hasn't received. At ~38 seconds per sync with the baseline approach (requiring extended timeouts), weight sync was dominating our training time.

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

### 3. CUDA IPC (Phase 2) — 3.6x Faster

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

### Challenge 4: The /dev/shm Ghost Problem

CUDA IPC stores handles (66 bytes each) in `/dev/shm`. Crashed processes leave orphaned handles. After enough experiments:

```bash
$ df -h /dev/shm
tmpfs  189G  189G  0  100%  /dev/shm  # Full!

$ rm -f /dev/shm/cuda.shm.* /dev/shm/torch_*  # Fix
```

---

## Configuration Comparison

| Config | Baseline | IPC | Speedup |
|--------|----------|-----|---------|
| 1x1 (FSDP=1, TP=1) | ~38s | 10.6s | **3.6x** |
| 2x1 (FSDP=2, TP=1) | 45.5s | 9.1s | **5.0x** |
| 2x2 (FSDP=2, TP=2) | 65.1s | 12.8s | **5.1x** |

**Recommended:** FSDP=2, TP=1 with IPC — 9.1s sync, best balance of memory efficiency and speed.

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

---

## Raw Benchmark Data

All benchmark logs are in `apps/gpu_direct/benchmark_logs/`:

| Log File | What It Measures |
|----------|------------------|
| `baseline_step2_1x1.log` | Per-tensor RPC: 14s push, 24s fetch = ~38s |
| `phase1_batched_rpc_1x1.log` | Batched RPC: 14s push, 8s fetch = ~22s |
| `ipc_1x1.log` | IPC: 10.6s total (9.7s pause + 0.9s transfer) |
| `ipc_2x1.log` | IPC with FSDP=2: 9.6s total |

---

*Thanks to the TorchForge team for making this work possible.*
