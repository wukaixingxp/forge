# KV Cache & Acceleration Plan for HybridPolicyActor

## Current Status

### ✅ What Works Now
- ValueMesh handling fixed
- Token synchronization for FSDP
- Generation completes successfully
- Training loop progresses

### ❌ What's Slow
- **No KV cache**: Each decode step recomputes entire sequence (O(n²) instead of O(n))
- **Sequential generation**: Generates n=4 completions one by one instead of batched
- **TorchTitan wrapper**: Model wrapper doesn't expose `past_key_values` parameter

## Why TorchTitan Model Doesn't Support KV Cache

The issue: `self.model` in InferenceEngine is a TorchTitan-wrapped model (with FSDP), not the raw HuggingFace model.

```python
# TorchTitan wraps like this:
model = Qwen3ForCausalLM(config)  # Raw model
model = FSDP(model, ...)          # Wrapped with FSDP
```

The FSDP wrapper's forward signature:
```python
def forward(self, input_ids):  # No past_key_values!
    return self.model(input_ids)
```

## Solution Options

### Option 1: Unwrap for Inference (Quick Fix)
**Status**: Not recommended - breaks FSDP communication

Store reference to unwrapped model, use for inference:
```python
self.raw_model = self.engine.model  # Before FSDP wrap
```

**Problem**: Loses FSDP collective communication needed for multi-GPU

---

### Option 2: Implement Paged KV Cache (Like nano-vLLM) ⭐ RECOMMENDED
**Status**: Best approach, used by vLLM and nano-vLLM

**Key insight**: Don't use PyTorch's built-in `past_key_values`. Instead:
1. Pre-allocate KV cache blocks
2. Manually manage KV updates
3. Index into cache during attention

**Benefits**:
- Works with any model wrapper (including FSDP)
- More memory efficient (paging)
- Enables continuous batching
- Can share cache across sequences (prefix caching)

**Implementation** (from nano-vLLM):

```python
class InferenceEngine:
    def allocate_kv_cache(self):
        """Pre-allocate GPU memory for KV cache blocks"""
        # Calculate how much memory we have
        free_mem = torch.cuda.mem_get_info()[0]
        cache_size = free_mem * 0.9  # Use 90% for KV cache

        # Calculate blocks needed
        num_layers = self.model.config.num_hidden_layers
        num_heads = self.model.config.num_attention_heads
        head_dim = self.model.config.hidden_size // num_heads
        block_size = 16  # tokens per block

        # Allocate [num_blocks, 2, num_layers, block_size, num_heads, head_dim]
        # 2 for K and V
        self.kv_cache = torch.zeros(
            (num_blocks, 2, num_layers, block_size, num_heads, head_dim),
            dtype=torch.bfloat16,
            device="cuda"
        )

        # Block table: maps sequence_id -> list of block indices
        self.block_manager = BlockManager(num_blocks, block_size)

    def _generate_one_with_paged_kv(self, prompt_ids, ...):
        # Allocate blocks for this sequence
        seq_id = self.block_manager.allocate_sequence()

        # Prefill: process full prompt
        for layer_idx in range(num_layers):
            # Inject KV cache pointer into attention layer
            layer.self_attn.kv_cache = self.kv_cache
            layer.self_attn.block_table = self.block_manager.get_block_table(seq_id)

        logits = self.model(prompt_ids)  # Attention writes to KV cache

        # Decode: process one token at a time
        for step in range(max_tokens):
            logits = self.model(next_token)  # Reads from KV cache
            next_token = sample(logits)

        # Free blocks
        self.block_manager.free_sequence(seq_id)
```

**Required Changes**:
1. Implement `BlockManager` (see nano-vllm/engine/block_manager.py)
2. Implement `PagedAttention` (see nano-vllm/layers/attention.py)
3. Modify model's attention to use paged cache
4. Pre-allocate KV cache on startup

**Effort**: 2-3 days
**Speedup**: 10-50x

---

### Option 3: Switch to nano-vLLM for Inference ⭐⭐ ALTERNATIVE
**Status**: Cleanest long-term solution

Replace InferenceEngine with nano-vLLM:

```python
from nanovllm import LLM, SamplingParams as NanoSamplingParams

class HybridPolicyActor:
    def __post_init__(self):
        # For training: Use TorchTitan engine with FSDP
        self.engine = ForgeEngine(...)
        self.model = self.engine.model

        # For inference: Use nano-vLLM
        self.inference_engine = LLM(
            model_path,
            enforce_eager=False,  # Enable CUDA graphs
            tensor_parallel_size=self.parallelism.tensor_parallel_degree,
            max_num_seqs=16,  # Continuous batching
        )

    async def generate(self, prompt, sampling_params):
        await self.switch_mode("infer")

        # Use nano-vLLM for generation
        outputs = self.inference_engine.generate(
            [prompt],
            NanoSamplingParams(
                n=sampling_params.n,
                max_tokens=sampling_params.max_tokens,
                temperature=sampling_params.temperature,
            )
        )
        return outputs
```

**Benefits**:
- Get all optimizations for free: KV cache, batching, CUDA graphs, prefix caching
- Battle-tested implementation (~1200 lines, readable)
- Maintained separately from training code

**Trade-offs**:
- Need to load model twice (training + inference)
- More memory usage
- But: Modern GPUs have plenty of memory (H200 has 141GB)

**Memory Calculation**:
- Qwen3-1.7B: ~7GB (bf16) × 2 = 14GB
- KV cache: ~8GB
- Total: ~22GB << 141GB available ✅

**Effort**: 1 day
**Speedup**: 50-100x (all optimizations)

---

## Recommended Implementation Plan

### Phase 1: Get Working Again (TODAY - 1 hour) ✅
- [x] Revert KV cache attempt
- [x] Fall back to simple forward pass
- [x] Verify generation works

### Phase 2: Evaluate Options (TOMORROW - 2 hours)
1. Benchmark current performance (tokens/sec)
2. Test nano-vLLM standalone
3. Measure memory usage with dual model approach
4. Decision: Option 2 (paged KV) vs Option 3 (nano-vLLM)

### Phase 3A: If Choosing Paged KV (2-3 days)
1. Implement BlockManager
2. Implement PagedAttention
3. Integrate with TorchTitan model
4. Test with FSDP
5. Benchmark

### Phase 3B: If Choosing nano-vLLM (1 day) ⭐ RECOMMENDED
1. Add nano-vLLM dependency
2. Initialize separate inference engine
3. Wire up generate() to use nano-vLLM
4. Test memory usage
5. Benchmark

## Expected Performance

| Optimization | Current | With Paged KV | With nano-vLLM |
|--------------|---------|---------------|----------------|
| Prefill | 148 tokens | Same | Same |
| Decode per token | ~100ms | ~2ms | ~1ms |
| Total (50 tokens) | ~5s | ~0.25s | ~0.15s |
| Throughput (4 seqs) | 0.8 tok/s/seq | 40 tok/s/seq | 66 tok/s/seq |
| **Speedup** | **1x** | **50x** | **83x** |

## Files to Study from nano-vLLM

Key files to understand their approach:

1. **Block Management**:
   - `nanovllm/engine/block_manager.py` - Memory allocation
   - `nanovllm/engine/sequence.py` - Sequence tracking

2. **Attention**:
   - `nanovllm/layers/attention.py` - Paged attention implementation
   - Uses pre-allocated cache, no `past_key_values`

3. **Scheduling**:
   - `nanovllm/engine/scheduler.py` - Continuous batching
   - `nanovllm/engine/model_runner.py` - Forward pass orchestration

4. **Model Integration**:
   - `nanovllm/models/qwen3.py` - Modified model to use paged cache

## Next Steps

1. **Test current fix works** (no KV cache, but functional)
2. **Benchmark baseline**: Measure tokens/sec
3. **Try nano-vLLM standalone**: Verify it works with Qwen3-1.7B
4. **Make decision**: Paged KV vs nano-vLLM integration
5. **Implement chosen approach**

## Summary

**Current status**: Generation works but is 50-100x slower than it should be due to lack of KV cache.

**Best path forward**: Integrate nano-vLLM as the inference engine (Option 3)
- Cleanest separation of concerns (training vs inference)
- Get all optimizations for free
- Battle-tested implementation
- 1 day vs 2-3 days effort

**Alternative**: Implement paged KV cache ourselves (Option 2)
- More control
- Learn the internals
- But more work and maintenance
