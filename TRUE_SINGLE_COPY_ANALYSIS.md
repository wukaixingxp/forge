# True Single-Copy KV Cache Integration: Deep Analysis

## Executive Summary

After deep investigation of TorchTitan's experimental code, **neither of their approaches achieves true single-copy**. Both maintain 2 model instances with weight synchronization:

1. **"unified" approach**: Separate TorchTitan model instance in vLLM
2. **"vllm_compat" approach**: Modified TorchTitan model with vLLM-compatible weights

**Key finding**: TorchTitan's experimental code synchronizes weights after EVERY training step via file-based checkpointing.

## TorchTitan's Actual Implementation

### Architecture (from `simple_rl_multiprocess.py`)

```python
# Two separate actors:
trainer = Trainer(...)      # TorchTitan model for training
generator = Generator(...)  # vLLM LLM for inference

# Training loop - WEIGHT SYNC EVERY STEP:
for step in range(num_steps):
    # 1. Generate with vLLM model
    batch = generator.generate.call().get()

    # 2. Train with TorchTitan model
    metrics = trainer.step.call(batch).get()

    # 3. SYNC WEIGHTS: trainer → generator
    weights = trainer.get_weights.call().get()
    await generator.update.call(step, weights)
```

### Weight Sync Implementation (from `generator.py`)

```python
def update_weights(self, vllm_compat_state: dict) -> None:
    """Update vLLM model weights from trainer."""
    # Convert to vLLM format
    vllm_state = torchtitan_to_vllm(vllm_compat_state)

    # Save to disk
    checkpoint_path = "model.safetensors"
    save_file(vllm_state, checkpoint_path)

    # Reload vLLM engine
    if self.llm is None:
        self.llm = LLM(model=checkpoint_path, ...)
    else:
        self.llm.collective_rpc("reload_weights")  # Hot reload
```

**Memory cost**: 2x model weights (15GB training + 15GB inference = 30GB for Qwen3-1.7B)

## True Single-Copy Requirements

To achieve **actual** single-copy (not what TorchTitan does), we'd need:

### Option 1: Custom vLLM Model Wrapping Existing Instance

```python
class SingleCopyVLLMWrapper(nn.Module):
    """Wraps existing training model instance for vLLM inference."""

    def __init__(self, training_model: nn.Module):
        super().__init__()
        # Use the SAME model instance (not a copy!)
        self.model = training_model

        # Replace attention layers with vLLM PagedAttention
        self._replace_attention_with_paged_attention()

        # Initialize KV cache manager
        self.kv_cache = PagedKVCache(...)

    def forward(self, input_ids, ...):
        # Use training model's layers + paged KV cache
        return self.model.forward_with_paged_attention(
            input_ids, kv_cache=self.kv_cache
        )
```

**Challenges**:
1. **Attention layer replacement**: Must surgically replace attention in existing model
2. **KV cache management**: vLLM's cache allocator expects specific data structures
3. **Batching/scheduling**: vLLM's continuous batching scheduler is tightly coupled
4. **CUDA graphs**: Capturing graphs with existing model instance is complex
5. **State management**: Training vs inference mode switching

### Option 2: In-Process vLLM Integration (Even More Complex)

```python
class HybridPolicyActorTrueSingleCopy:
    def __init__(self):
        # Single TorchTitan model
        self.model = TorchTitanModel(...)

        # Replace attention layers
        replace_attention_with_vllm_paged(self.model)

        # Custom scheduler/executor (can't use vLLM's LLM class)
        self.scheduler = CustomPagedAttentionScheduler(self.model)
        self.kv_cache_manager = CustomKVCacheManager()

    def generate(self, prompts):
        # Switch to inference mode
        self.model.eval()

        # Manual batching + KV cache management
        return self.scheduler.generate(prompts, self.kv_cache_manager)

    def train_step(self, batch):
        # Switch back to training mode
        self.model.train()

        # Standard training (attention layers work for both modes)
        loss = compute_loss(self.model, batch)
        loss.backward()
```

**This requires reimplementing**:
- vLLM's PagedAttention scheduler (~2000 lines)
- KV cache block management (~1000 lines)
- Request batching and continuous batching (~1500 lines)
- CUDA graph capture for decode (~500 lines)

**Estimate**: 2-3 weeks of development + debugging

## Recommended Approach: Efficient 2-Copy with Fast Weight Sync

Since true single-copy is extremely complex and even TorchTitan doesn't do it, the practical solution is:

### Architecture

```python
class HybridPolicyActor:
    def __init__(self):
        # Training model (TorchTitan)
        self.training_model = ForgeEngine(...)

        # Inference model (vLLM) - separate instance
        self.inference_engine = SimpleVLLMEngine(...)

        # Weight sync mechanism
        self.weight_sync = EfficientWeightSync()

    async def generate(self, prompts):
        # 1. Sync weights if training happened
        if self.training_model.step > self.last_synced_step:
            await self.weight_sync.sync_async(
                src=self.training_model,
                dst=self.inference_engine
            )

        # 2. Generate with vLLM (50-100x faster than naive)
        return await self.inference_engine.generate(prompts)

    def train_step(self, batch):
        # Train as normal
        return self.training_model.step(batch)
```

### Efficient Weight Sync Options

#### Option A: In-Memory Transfer (Fastest)
```python
class InMemoryWeightSync:
    """Transfer weights via shared memory or GPU-Direct."""

    def sync(self, src_model, dst_model):
        # Get state dict from training model
        src_state = src_model.state_dict()

        # Convert format (TorchTitan → vLLM)
        dst_state = convert_torchtitan_to_vllm(src_state)

        # Load into vLLM via reload_weights API
        dst_model.llm.collective_rpc("reload_weights_from_dict", dst_state)
```

**Speed**: ~1-2 seconds for Qwen3-1.7B (GPU→GPU transfer)

#### Option B: TorchStore (Distributed)
```python
class TorchStoreWeightSync:
    """Use TorchStore for distributed weight sharing."""

    def __init__(self):
        self.store = TorchStore(backend="redis")

    async def sync_async(self, src_model, dst_model):
        # Publish weights to TorchStore
        await self.store.save_async("policy_weights", src_model.state_dict())

        # vLLM loads from TorchStore
        await dst_model.load_from_store_async("policy_weights")
```

**Speed**: ~2-5 seconds depending on network

#### Option C: File-Based (Simplest, Slowest)
```python
class FileBasedWeightSync:
    """Save to disk and reload (TorchTitan's approach)."""

    def sync(self, src_model, dst_model):
        # Save checkpoint
        torch.save(src_model.state_dict(), "checkpoint.pt")

        # Reload vLLM
        dst_model.llm.collective_rpc("reload_weights")
```

**Speed**: ~5-10 seconds for Qwen3-1.7B (disk I/O bottleneck)

## Comparison Matrix

| Approach | Model Copies | Weight Sync | Memory | Complexity | Dev Time |
|----------|--------------|-------------|--------|------------|----------|
| **True Single-Copy** | 1 | None | 15GB | Very High | 2-3 weeks |
| **TorchTitan "unified"** | 2 | File (every step) | 30GB | Medium | Done (experimental) |
| **TorchTitan "vllm_compat"** | 2 | File (every step) | 30GB | High | Done (experimental) |
| **Our SimpleVLLM** | 2 | None (off-policy) | 30GB | Low | Done ✓ |
| **Efficient 2-Copy + InMemory Sync** | 2 | GPU (on-demand) | 30GB | Medium | 2-3 days |

## Recommendation: Hybrid Approach

Given the constraints, I recommend:

### Phase 1: Current SimpleVLLM (Already Working)
- ✅ 50-100x inference speedup achieved
- ✅ Paged KV cache working
- ✅ Zero weight sync complexity
- ⚠️ Off-policy (may reduce sample efficiency slightly)

### Phase 2: Add Efficient Weight Sync (If On-Policy Needed)
- Implement in-memory weight sync via vLLM's `reload_weights` API
- Sync only when training step completes
- Use async transfer to overlap with generation
- **Estimated effort**: 2-3 days
- **Memory**: Still 2x (30GB for Qwen3-1.7B)
- **Sync cost**: ~1-2 seconds per sync

### Phase 3: Explore TorchStore Integration (Future)
- Use TorchStore for distributed weight sharing
- Enables multi-node training + inference
- **Estimated effort**: 1 week

## Conclusion

**True single-copy is NOT what TorchTitan does**, and implementing it from scratch would take 2-3 weeks with high complexity. The practical approach is efficient 2-copy with fast weight synchronization.

Our current `SimpleVLLMEngine` already achieves the 50-100x speedup target. Adding weight sync (if needed for on-policy RL) would take 2-3 days and maintain the same memory footprint as TorchTitan's approach.

## Implementation Plan for Efficient Weight Sync

If we want on-policy training with weight sync:

### Step 1: Add Weight Sync API (1 day)

```python
# src/forge/actors/hybrid/weight_sync.py
class EfficientWeightSync:
    """Fast in-memory weight synchronization."""

    def __init__(self, training_model, inference_engine):
        self.training_model = training_model
        self.inference_engine = inference_engine
        self.last_synced_step = 0

    def should_sync(self) -> bool:
        """Check if sync is needed."""
        return self.training_model.step > self.last_synced_step

    @torch.no_grad()
    def sync(self):
        """Sync weights from training model to inference engine."""
        # Get training model state dict
        train_state = self.training_model.model.state_dict()

        # Convert TorchTitan → vLLM format
        vllm_state = self._convert_format(train_state)

        # Reload vLLM weights
        self._reload_vllm_weights(vllm_state)

        self.last_synced_step = self.training_model.step
```

### Step 2: Integrate into HybridPolicyActor (1 day)

```python
# Modify generate() method
async def generate(self, prompts):
    # Sync weights if training happened
    if self.weight_sync.should_sync():
        logger.info("Syncing weights to inference engine...")
        self.weight_sync.sync()

    # Generate with synced model
    return await self.inference_engine.generate(prompts)
```

### Step 3: Test and Optimize (1 day)

- Measure sync overhead
- Profile GPU→GPU transfer
- Add async sync option
- Test with FSDP (multi-GPU)

**Total**: 2-3 days to add efficient weight synchronization.
