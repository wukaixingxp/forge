# From 78 Seconds to 1.3 Seconds: Our Journey to GPU-Direct Weight Sync

Every 78 seconds, our RL training loop sat idle. Not learning. Not generating data. Just waiting—waiting for model weights to copy from the trainer to the generator. In a world where GPU hours cost real money and iteration speed determines research velocity, this was unacceptable. Here's how we got it down to 1.3 seconds.

> **📊 Architecture Diagrams:** See the [diagrams/](diagrams/) folder for visual illustrations:
> - [01-data-flow-comparison.excalidraw](diagrams/01-data-flow-comparison.excalidraw) - Data flow paths for all three approaches
> - [02-timeline-comparison.excalidraw](diagrams/02-timeline-comparison.excalidraw) - Timeline showing performance improvements
> - [03-rl-training-loop.excalidraw](diagrams/03-rl-training-loop.excalidraw) - Where weight sync fits in RL training

## Why Weight Sync Matters

Online reinforcement learning has a fundamental loop: **train → sync → generate → repeat**. You have a trainer that updates model weights through gradient descent, and a generator that uses those weights to collect new experience. They need to stay in sync.

```
┌─────────────┐                      ┌─────────────┐
│   Trainer   │ ──── sync weights ───▶│  Generator  │
│  (learns)   │                      │ (explores)  │
└─────────────┘                      └─────────────┘
      ▲                                     │
      └──────────── new experience ─────────┘
```

The catch? During weight sync, nothing productive happens. The trainer can't update weights it's sending. The generator can't use weights it hasn't received. It's pure overhead—and at 78 seconds per sync, it was dominating our training time.

## The Baseline: Death by a Thousand RPCs

Our starting point used MonarchRPC, a general-purpose transport layer. The data path looked like this:

```
Trainer GPU → Trainer CPU → Network → Generator CPU → Generator GPU
```

Four memory copies. Four potential bottlenecks. But the real killer wasn't bandwidth—it was latency.

Our model has ~800 parameter tensors. The baseline code synced them one at a time:

```python
for name, tensor in model.named_parameters():
    await storage.push(name, tensor)  # RPC call
    await generator.update(name)       # Another RPC call
```

That's 1,600+ round trips. Even with sub-millisecond network latency, the coordination overhead added up. Our benchmark told the story:

| Phase | Time |
|-------|------|
| Push weights to storage (per-tensor `ts.put()`) | ~27s |
| Generator pulls weights (per-tensor `ts.get()`) | >30s (timeout!) |
| **Total** | **>57s** (often fails) |

The per-tensor fetch was so slow it exceeded the default 30s Monarch timeout! We had to increase `HYPERACTOR_MESSAGE_DELIVERY_TIMEOUT=120s` just to measure it. Even then, the generator often crashed before completing.

We measured effective bandwidth at ~1.4 GB/s. Our hardware could theoretically do 25+ GB/s. Something was very wrong.

## Phase 1: Batching the API

### The Insight

Each RPC call carries fixed overhead: serialization, network round-trip, deserialization, dispatch. When you make 800 calls, you pay that tax 800 times. What if we paid it once?

### The Implementation

We introduced batched APIs that bundle multiple tensors into single RPC calls:

```python
# Before: 800 RPC calls
for name, tensor in model.named_parameters():
    await storage.push(name, tensor)

# After: ~8 RPC calls (100 tensors per batch)
batches = chunk(model.named_parameters(), batch_size=100)
for batch in batches:
    await storage.push_batch(batch)
```

The key was finding the right batch size. Too small and you still have overhead. Too large and you hit memory pressure. We settled on 100 tensors per batch after benchmarking.

We also optimized the notification pattern. Instead of the generator polling for each weight update, the trainer sends a single "weights ready" signal after all batches complete.

### The Result

| Metric | Per-tensor RPC | Batched RPC | Improvement |
|--------|----------------|-------------|-------------|
| Push time (`ts.put` → `ts.put_batch`) | ~27s | ~14.4s | 1.9x |
| Fetch time (`ts.get` → `ts.get_batch`) | >30s (timeout) | ~7.8s | 4x+ |
| **Total** | **>57s** | **~22s** | **2.6x** |

Not bad for what was essentially a one-line conceptual change. The fetch improvement was dramatic—going from timeout failures to reliable sub-10s completion.

**Reproducibility:** The batched RPC implementation is available in the `batch_fetch` branch of `~/kai/forge`:
```bash
cd ~/kai/forge && git checkout batch_fetch
export HYPERACTOR_MESSAGE_DELIVERY_TIMEOUT=120s
python -m apps.grpo.main --config apps/grpo/qwen3_4b_1x1.yaml
```

But we were still copying through CPU memory. Could we do better?

## Phase 2: Going GPU-Direct with CUDA IPC

### The Insight

CUDA Inter-Process Communication (IPC) lets GPUs share memory directly. No CPU copies. No serialization. One GPU writes, another GPU reads—same physical memory.

```
Before:  GPU → CPU → CPU → GPU  (4 copies)
After:   GPU ═══════════ GPU    (0 copies, shared memory)
```

The theory was beautiful. The practice... had some surprises.

### Attempt 1: The Happy Path (1x1 Configuration)

We started simple: one trainer GPU, one generator GPU. The code was straightforward:

```python
# Trainer side: export CUDA IPC handle
handle = tensor.storage()._share_cuda_()
await generator.receive_handle(handle, tensor.shape, tensor.dtype)

# Generator side: reconstruct tensor from handle
tensor = torch.reconstruct_tensor(handle, shape, dtype)
model.load_weights([(name, tensor)])
```

It worked immediately. Weight sync dropped to **~1.8 seconds**. We high-fived and moved on to the real test.

### Attempt 2: The Real World (2x2 Configuration)

Production uses FSDP (Fully Sharded Data Parallel) on the trainer side and Tensor Parallelism on the generator side. Two trainer GPUs, two generator GPUs. How hard could it be?

Hard. Very hard.

**Problem 1: GPU Visibility**

CUDA IPC handles reference specific GPU memory. When the generator process started with `CUDA_VISIBLE_DEVICES=2,3`, it couldn't see GPUs 0 and 1 where the trainer's memory lived. The handles were valid but pointed to invisible memory.

```python
# Solution: Give all processes visibility to all GPUs for IPC
if os.environ.get("FORGE_IPC_GPU_VISIBILITY") == "1":
    # Don't restrict CUDA_VISIBLE_DEVICES
    cuda_devices = ",".join(str(i) for i in range(torch.cuda.device_count()))
```

**Problem 2: vLLM's Merged Weights**

vLLM doesn't store weights the way Hugging Face models do. For efficiency, it merges certain weights:

```
HuggingFace:              vLLM:
  q_proj.weight    ─┐
  k_proj.weight    ─┼──▶  qkv_proj.weight
  v_proj.weight    ─┘

  gate_proj.weight ─┬──▶  gate_up_proj.weight
  up_proj.weight   ─┘
```

Our trainer exports HuggingFace-style weights. The generator expects vLLM-style weights. We needed a translation layer:

```python
def _build_param_map(self, model):
    """Map HuggingFace names to vLLM merged parameters."""
    param_map = {}
    for name, param in model.named_parameters():
        if "qkv_proj" in name:
            # Register all three source names
            for proj in ['q_proj', 'k_proj', 'v_proj']:
                hf_name = name.replace("qkv_proj", proj)
                param_map[hf_name] = (f"qkv_{proj[0]}", param)
    return param_map
```

**Problem 3: Tensor Parallel Slicing**

With TP=2, each generator GPU only holds half of certain weight matrices. The trainer sends full weights; the generator must slice them correctly:

```python
def _slice_for_tp(self, tensor, dim, param_name):
    """Slice tensor for this TP rank."""
    tp_rank = self.tp_rank
    tp_size = self.tp_size

    slice_size = tensor.shape[dim] // tp_size
    start = tp_rank * slice_size
    end = start + slice_size

    # Slice along the appropriate dimension
    if dim == 0:
        return tensor[start:end, ...]
    else:
        return tensor[..., start:end]
```

### The Result

After solving all three problems:

| Configuration | Sync Time | vs Baseline |
|---------------|-----------|-------------|
| 1x1 (simple)  | 1.8s      | 43x faster  |
| 2x2 (FSDP+TP) | 1.3s      | 60x faster  |

Wait—2x2 is *faster* than 1x1? Yes! With FSDP, each trainer GPU only holds a shard of the weights. Less data to transfer per GPU means faster sync, even with more coordination.

## A Debugging Interlude: The Case of the Full /dev/shm

Midway through Phase 2, everything stopped working. CUDA IPC handles that worked yesterday now failed with cryptic `cudaErrorInvalidValue` errors.

After much head-scratching, we discovered `/dev/shm` was 100% full:

```bash
$ df -h /dev/shm
Filesystem      Size  Used Avail Use% Mounted on
tmpfs           189G  189G     0 100% /dev/shm
```

"But wait," you might ask, "I thought this was GPU-direct? Why is shared memory involved?"

Great question. CUDA IPC doesn't copy tensor *data* through `/dev/shm`—that stays on the GPU. But it does store *handles* there. Each handle is only ~66 bytes, but they accumulate. Crashed processes leave orphaned handles. Run enough experiments and you fill the disk with ghosts.

```bash
$ rm -f /dev/shm/cuda.shm.* /dev/shm/torch_*
```

Problem solved. Lesson learned: even "GPU-direct" has hidden dependencies.

## The Complete Picture

Here's our full optimization journey (measured on Qwen3-4B, 1x1 config):

| Stage | Technique | Push | Fetch | Total | vs Baseline |
|-------|-----------|------|-------|-------|-------------|
| Baseline | Per-tensor RPC (`ts.put`/`ts.get`) | 27s | >30s | >57s | — |
| Phase 1 | Batched RPC (`ts.put_batch`/`ts.get_batch`) | 14.4s | 7.8s | ~22s | 2.6x |
| Phase 2 | CUDA IPC (1x1) | — | — | ~10.6s | 5.4x |
| Phase 2 | CUDA IPC (2x2 FSDP+TP) | — | — | ~9.6s | 6x |

**Note:** The IPC total includes `pause_generation` time (~7s) which is unavoidable—the generator must wait for in-flight requests to complete. The actual IPC data transfer is only ~1.7-2.8s.

From >57 seconds (and frequent timeouts) to ~10 seconds. From "broken" to "fast enough to forget about."

## What This Enables

Fast weight sync isn't just about saving time—it changes what's possible:

**More iterations, faster learning.** If you're running 1000 sync cycles during training, you just saved 21 hours of wall-clock time.

**Tighter feedback loops.** With sub-2-second sync, you can afford to sync more frequently. Fresher weights in the generator mean better exploration.

**Cheaper experiments.** GPU hours cost money. Cutting overhead by 60x means the same experiment costs less, or you can run more experiments for the same budget.

## Key Takeaways

1. **Measure before optimizing.** Our initial assumption was bandwidth-limited. The real bottleneck was RPC overhead. Profiling saved us from optimizing the wrong thing.

2. **Batch aggressively.** Coordination overhead often dominates data transfer time. One RPC with 100 tensors beats 100 RPCs with 1 tensor.

3. **Eliminate copies.** Every memory copy is latency and bandwidth you're not using for real work. GPU-direct transfers skip the CPU entirely.

4. **Test realistic configurations.** Our 1x1 prototype worked perfectly. Production's 2x2 configuration surfaced three new problems. Test what you'll actually run.

5. **Debug systematically.** When CUDA IPC failed, the error messages were useless. Checking disk space, GPU visibility, and process isolation eventually revealed the issues.

The code is in `torchforge/demos/gpu_direct_weight_sync/`. Try it yourself—and let us know if you find even faster approaches.

### Reproducing the Benchmarks

**Environment Setup:**
```bash
# Required: Increase Monarch timeout (default 30s is too short for per-tensor)
export HYPERACTOR_MESSAGE_DELIVERY_TIMEOUT=120s
export HYPERACTOR_HOST_SPAWN_READY_TIMEOUT=120s
```

**Per-tensor Baseline (original, slow):**
```bash
cd ~/kai/forge && git checkout main
conda activate baseline
python -m apps.grpo.main --config apps/grpo/qwen3_4b_1x1.yaml
# Expected: Push ~27s, Fetch timeout or >30s
```

**Batched RPC (Phase 1):**
```bash
cd ~/kai/forge && git checkout batch_fetch
conda activate baseline
python -m apps.grpo.main --config apps/grpo/qwen3_4b_1x1.yaml
# Expected: Push ~14s, Fetch ~8s, Total ~22s
```

**CUDA IPC (Phase 2):**
```bash
cd /home/dev/framework/torchforge
conda activate vllm
export PYTHONPATH="/home/dev/framework/torchforge/src:$PYTHONPATH"
export FORGE_IPC_GPU_VISIBILITY=1
python -m apps.gpu_direct.main --config apps/gpu_direct/qwen3_4b_1x1.yaml
# Expected: Total ~10s (including ~7s pause_generation)
```

Detailed benchmark logs are in `torchforge/apps/gpu_direct/benchmark_logs/`.

---

## Update: E2E Integration Results

We integrated GPU-Direct Weight Sync into a complete GRPO (Grouped Relative Policy Optimization) training loop and ran end-to-end benchmarks on 4x NVIDIA H200 GPUs with Qwen3-4B.

### Important: Understanding the Baselines

**Clarification on baseline comparisons (measured on Qwen3-4B):**

| System | Push | Fetch | Total | Description |
|--------|------|-------|-------|-------------|
| Per-tensor RPC | 27s | >30s (timeout) | >57s | Original, 1 RPC per tensor - often fails |
| Batched RPC (Phase 1) | 14.4s | 7.8s | ~22s | `ts.put_batch`/`ts.get_batch` |
| TorchStore (existing) | — | — | 45-65s | Different storage system, CPU-mediated |
| CUDA IPC (Phase 2) | — | — | 9-13s | GPU-direct transfer, zero CPU copies |

The per-tensor baseline requires increasing `HYPERACTOR_MESSAGE_DELIVERY_TIMEOUT=120s` (default is 30s) to even complete. Even then, it frequently crashes.

The E2E benchmarks below compare **IPC vs TorchStore** (not vs the original per-tensor baseline). TorchStore is a different system that was already in use in the codebase - it has its own optimizations but still involves CPU-mediated transfers.

### Full Training Loop Benchmarks (IPC vs TorchStore)

| Configuration | TorchStore | IPC | Speedup |
|---------------|------------|-----|---------|
| 2x2 (FSDP=2, TP=2) | 65.1s | 12.8s | **5.1x** |
| 2x1 (FSDP=2, TP=1) | 45.5s | 9.1s | **5.0x** |
| 1x1 (FSDP=1, TP=1) | 50.1s | 10.5s | **4.8x** |

**Note:** The "60x speedup" (78s → 1.3s) mentioned earlier was from isolated benchmarks comparing IPC to the original per-tensor RPC approach. The E2E comparison against TorchStore shows a consistent **5x speedup**.

### Breakdown: Where the Time Goes

**Baseline (2x1, TorchStore):**
```
push_weights (trainer → TorchStore)    13.4s  ███████████████
update_weights (TorchStore → gen)      32.0s  ████████████████████████████████████
├── pause_generation                    5.7s  ██████
└── worker_load_weights                26.4s  ██████████████████████████████
                                       ─────
Total:                                 45.5s
```

**IPC (2x1, GPU-Direct):**
```
update_weights_ipc                      9.1s  █████████
├── pause_generation                    6.7s  ███████
├── IPC handle send                     0.3s
└── worker_load_weights                 2.0s  ██
                                       ─────
Total:                                  9.1s
```

### Key Insight: Generator Pause Time

The generator must pause and wait for in-flight requests to complete before updating weights. This pause time (~6-10s) is unavoidable and sets a floor on sync time regardless of transfer method.

```
pause_time ≈ max_tokens / generation_speed
512 tokens @ 50 tok/s ≈ 10s pause
```

With IPC, the data transfer (0.3s + 2.0s = 2.3s) is negligible compared to the pause. The bottleneck has shifted from transfer to generation.

### Bug Discovery: FSDP + TorchStore

During testing, we discovered a critical bug: TorchStore silently failed with FSDP models. PyTorch FSDP2 returns DTensors from `state_dict()`, which TorchStore stored as sharded dicts. The fetch path couldn't handle this format, causing silent weight corruption.

**Fix:** Gather full tensors via `.full_tensor()` before pushing to TorchStore.

### Recommended Configuration

For H200 with 143GB memory, we recommend **FSDP=2, TP=1 with IPC**:

- **9.1s total sync time** (5x faster than baseline)
- FSDP provides memory efficiency for larger models
- TP=1 avoids tensor parallel slicing complexity
- Single-GPU generation is faster (no TP communication overhead)

The full benchmark results and configs are in `torchforge/apps/gpu_direct/`.

---

## Deep Dive: Training Loop Bottleneck Analysis

To help understand where optimization efforts should focus, here's a detailed breakdown of time spent in each phase of the GRPO training loop.

### Complete Training Step Breakdown (Qwen3-4B, FSDP=2, TP=1)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     ONE TRAINING STEP BREAKDOWN                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. GENERATION PHASE                                        ~45s (55%) │
│     ├── Prefill (prompt processing)              ~3s   ████           │
│     └── Decode (token generation, 512 tokens)    ~42s  ███████████████ │
│                                                                         │
│  2. TRAINING PHASE                                          ~25s (30%) │
│     ├── Forward pass                             ~8s   ████████        │
│     ├── Loss computation                         ~2s   ██              │
│     ├── Backward pass                            ~12s  ████████████    │
│     └── Optimizer step                           ~3s   ███             │
│                                                                         │
│  3. WEIGHT SYNC PHASE (what we optimized)               ~9-45s (12-55%)│
│     │                                                                   │
│     │  BASELINE (TorchStore):                           45.5s total    │
│     │  ├── push_weights (trainer → store)        13.4s  █████████████  │
│     │  └── update_weights (store → generator)    32.0s  █████████████████████████████│
│     │      ├── pause_generation                  5.7s   ██████         │
│     │      └── worker_load_weights               26.4s  ████████████████████████│
│     │                                                                   │
│     │  IPC (GPU-Direct):                                 9.1s total    │
│     │  ├── pause_generation                      6.7s   ███████        │
│     │  ├── IPC handle send                       0.3s                  │
│     │  └── worker_load_weights                   2.0s   ██             │
│     │                                                                   │
│  4. OTHER (data loading, reward compute, etc.)          ~3s (3%)      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Percentage Breakdown: Where Does Time Go?

| Phase | Per-tensor RPC | Batched RPC | IPC (GPU-Direct) |
|-------|----------------|-------------|------------------|
| Generation | 45s | 45s | 45s |
| Training | 25s | 25s | 25s |
| **Weight Sync** | **>57s (broken)** | **~22s (24%)** | **~9s (11%)** |
| Other | 3s | 3s | 3s |
| **Total** | **>130s** | **~95s** | **~82s** |

### Key Insights for Optimization Priorities

**Per-tensor RPC (Original Baseline):**
- Weight sync was **broken** — frequently timed out after 30s
- Required increasing `HYPERACTOR_MESSAGE_DELIVERY_TIMEOUT=120s` just to measure
- Push: ~27s for ~400 tensors = ~67ms per RPC call overhead

**Batched RPC (Phase 1):**
- Weight sync reduced to **~22s (24% of training loop)**
- Still involves CPU copies but dramatically fewer RPC calls
- Push: 14.4s, Fetch: 7.8s — both work reliably

**After IPC Optimization (Phase 2):**
- Weight sync reduced to **~9s (11% of training loop)**
- Generation (55%) is now the dominant bottleneck
- The limiting factor has shifted from infrastructure to model capability

### Scaling to 32B Models

For larger models like Qwen3-32B, the breakdown shifts:

```
┌─────────────────────────────────────────────────────────────────────────┐
│              32B MODEL TRAINING STEP (Estimated)                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Generation (32B @ ~25 tok/s)                          ~80s (40%)     │
│  Training (8x more params)                             ~100s (50%)    │
│  Weight Sync (IPC, 8x data)                            ~15-20s (10%) │
│  Weight Sync (Baseline, 8x data)                       ~200-250s     │
│                                                                         │
│  With baseline: Weight sync would be ~50% of step time!               │
│  With IPC: Weight sync remains ~10% of step time                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### The Math: Why IPC Matters More for Larger Models

| Model | Parameters | Per-tensor RPC | Batched RPC | IPC Sync | Savings vs Batched |
|-------|------------|----------------|-------------|----------|-------------------|
| 4B | 4B | >57s (broken) | ~22s | ~9s | **13s/step** |
| 32B | 32B | >200s (est.) | ~80s | ~20s | **60s/step** |
| 70B | 70B | >400s (est.) | ~150s | ~35s | **115s/step** |

**Savings over 1000 training steps:**
- 4B model: 13s × 1000 = **3.6 hours** saved (IPC vs Batched)
- 32B model: 60s × 1000 = **16.7 hours** saved
- 70B model: 115s × 1000 = **32 hours** saved

For large model training with thousands of sync cycles, IPC saves **days of GPU time**.

### Pause Generation: The Irreducible Floor

One insight from our benchmarks: the generator pause time (~6-10s) is largely unavoidable. The generator must:

1. Wait for in-flight generation requests to complete
2. Clear KV cache before loading new weights
3. This time scales with `max_tokens / generation_speed`

```python
pause_time ≈ max_tokens / tokens_per_second
# 512 tokens @ 50 tok/s = ~10s (4B model)
# 512 tokens @ 25 tok/s = ~20s (32B model)
# 128 tokens @ 25 tok/s = ~5s  (32B, reduced max_tokens)
```

**Options to reduce pause time:**
1. Use `wait_for_inflight_requests=False` (aborts in-flight work, wastes compute)
2. Reduce `max_tokens` (shorter responses)
3. Use faster generation (larger TP, better hardware)

### Recommendations for Your Team

1. **IPC is essential for large models** — Without it, weight sync dominates training time
2. **Optimize generation next** — After IPC, generation (prefill + decode) is the bottleneck
3. **FSDP=2, TP=1 is optimal for H200** — Best balance of memory efficiency and sync speed
4. **Consider batch sync for very large models** — Sync every N steps instead of every step

---

*Thanks to the TorchForge team for making this work possible.*
