# From 78 Seconds to 1.3 Seconds: Our Journey to GPU-Direct Weight Sync

Every 78 seconds, our RL training loop sat idle. Not learning. Not generating data. Just waiting—waiting for model weights to copy from the trainer to the generator. In a world where GPU hours cost real money and iteration speed determines research velocity, this was unacceptable. Here's how we got it down to 1.3 seconds.

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
| Push weights to storage | ~15s |
| Generator pulls weights | ~63s |
| **Total** | **~78s** |

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

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Push time | ~15s | ~2.5s | 6x |
| Update time | ~63s | ~10s | 6.3x |
| **Total** | **~78s** | **~12.5s** | **6.2x** |

Not bad for what was essentially a one-line conceptual change. But we were still copying through CPU memory. Could we do better?

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

Here's our full optimization journey:

| Stage | Technique | Sync Time | Improvement |
|-------|-----------|-----------|-------------|
| Baseline | Individual RPCs via MonarchRPC | 78s | — |
| Phase 1 | Batched APIs | 12.5s | 6.2x |
| Phase 2 | CUDA IPC (1x1) | 1.8s | 43x |
| Phase 2 | CUDA IPC (2x2 FSDP+TP) | 1.3s | 60x |

From 78 seconds to 1.3 seconds. From "go get coffee" to "barely noticeable."

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

---

*Thanks to the TorchForge team for making this work possible.*
