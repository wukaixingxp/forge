# Simple KV Cache Integration Plan (Nano-vLLM Approach)

## Executive Summary

Nano-vLLM provides a **much simpler** path to true single-copy KV cache integration:
- Only ~1,200 lines of code (vs vLLM's ~50,000)
- **Direct KV cache assignment** to model layers (no complex scheduler)
- Uses `flash_attn_with_kvcache` for efficient decode
- Can wrap existing TorchTitan model!

**Key insight**: Nano-vLLM allocates KV cache buffers and **assigns them directly to attention layers**. We can do the same with our training model!

## Nano-vLLM Architecture Analysis

### 1. KV Cache Allocation (model_runner.py:99-117)

```python
def allocate_kv_cache(self):
    # Calculate available GPU memory
    free, total = torch.cuda.mem_get_info()

    # Allocate single large tensor for all KV cache
    # Shape: [2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]
    self.kv_cache = torch.empty(2, num_layers, num_blocks, block_size, num_kv_heads, head_dim)

    # Assign cache buffers to each attention layer
    layer_id = 0
    for module in self.model.modules():
        if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
            module.k_cache = self.kv_cache[0, layer_id]  # Share reference!
            module.v_cache = self.kv_cache[1, layer_id]
            layer_id += 1
```

**Key**: KV cache is a **view into the same memory** used by all layers. Training and inference share it!

### 2. Attention Layer (attention.py:43-75)

```python
class Attention(nn.Module):
    def __init__(self, num_heads, head_dim, scale, num_kv_heads):
        super().__init__()
        self.k_cache = self.v_cache = torch.tensor([])  # Placeholder

    def forward(self, q, k, v):
        context = get_context()  # Get inference metadata

        # Store new KV in cache
        if self.k_cache.numel() and self.v_cache.numel():
            store_kvcache(k, v, self.k_cache, self.v_cache, context.slot_mapping)

        if context.is_prefill:
            # Prefill: use flash attention
            o = flash_attn_varlen_func(q, k, v, ...)
        else:
            # Decode: use cached KV
            o = flash_attn_with_kvcache(q.unsqueeze(1), self.k_cache, self.v_cache,
                                        cache_seqlens=context.context_lens, ...)
        return o
```

**Key**: Same attention layer works for both training (no cache) and inference (with cache)!

### 3. Block Manager (block_manager.py:26-112)

```python
class BlockManager:
    """Simple paged KV cache with prefix caching."""

    def __init__(self, num_blocks, block_size):
        self.blocks = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id = dict()  # Prefix cache lookup
        self.free_block_ids = deque(range(num_blocks))

    def allocate(self, seq: Sequence):
        """Allocate blocks for sequence (with prefix cache hit detection)."""
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, prefix_hash)
            block_id = self.hash_to_block_id.get(h, -1)

            if block_id == -1:  # Cache miss
                block_id = self.free_block_ids.popleft()
            else:  # Cache hit!
                seq.num_cached_tokens += self.block_size

            seq.block_table.append(block_id)
```

**Only ~113 lines** vs ~1000 in vLLM!

## True Single-Copy Implementation Plan

### Architecture

```python
class HybridPolicyActorSingleCopy:
    """Single model copy with nano-vLLM style KV cache."""

    def __init__(self):
        # Single TorchTitan model
        self.engine = ForgeEngine(...)
        self.model = self.engine.model_parts[0]

        # Replace attention layers with KV-cache enabled versions
        self._replace_attention_layers()

        # Allocate KV cache and assign to layers
        self.kv_cache_manager = NanoStyleKVCache(
            model=self.model,
            num_blocks=1000,
            block_size=16
        )

        # Simple block manager
        self.block_manager = BlockManager(num_blocks=1000, block_size=16)

        # Scheduler (simplified - no continuous batching)
        self.scheduler = SimpleScheduler()

    def _replace_attention_layers(self):
        """Replace TorchTitan attention with nano-vLLM style attention."""
        for name, module in self.model.named_modules():
            if isinstance(module, TorchTitanAttention):
                # Replace with KV-cache enabled attention
                new_attn = NanoStyleAttention(
                    num_heads=module.n_heads,
                    head_dim=module.head_dim,
                    num_kv_heads=module.n_kv_heads
                )
                # Preserve weights
                new_attn.load_state_dict(module.state_dict(), strict=False)
                # Replace in parent
                parent = self._get_parent_module(name)
                setattr(parent, name.split('.')[-1], new_attn)

    @torch.inference_mode()
    async def generate(self, prompts):
        # Switch to inference mode
        self.model.eval()

        # Allocate blocks for sequences
        sequences = [Sequence(tokenize(p)) for p in prompts]
        for seq in sequences:
            self.block_manager.allocate(seq)

        # Set inference context
        with inference_context(sequences, self.block_manager):
            # Generate using model with KV cache
            outputs = self._generate_loop(sequences)

        # Free blocks
        for seq in sequences:
            self.block_manager.deallocate(seq)

        return outputs

    def train_step(self, batch):
        # Switch to training mode
        self.model.train()

        # Training (no KV cache context set, so cache is bypassed)
        loss = self.engine.step(batch)
        return loss
```

### Implementation Steps

#### Phase 1: Nano-Style Attention Layer (2 days)

Create `src/forge/actors/hybrid/nano_attention.py`:

```python
class NanoStyleAttention(nn.Module):
    """Attention layer supporting both training and cached inference."""

    def __init__(self, num_heads, head_dim, num_kv_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads

        # QKV projections (preserved from original)
        self.q_proj = nn.Linear(...)
        self.k_proj = nn.Linear(...)
        self.v_proj = nn.Linear(...)
        self.o_proj = nn.Linear(...)

        # KV cache buffers (assigned by cache manager)
        self.k_cache = torch.tensor([])
        self.v_cache = torch.tensor([])

    def forward(self, x, rope_cache=None, positions=None):
        # Get context (None during training)
        context = get_inference_context()

        # Project Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Apply RoPE
        q, k = apply_rotary_emb(q, k, rope_cache, positions)

        if context is None:
            # Training mode: use standard flash attention
            o = flash_attn_func(q, k, v, causal=True)
        else:
            # Inference mode: use KV cache
            if self.k_cache.numel():
                store_kvcache(k, v, self.k_cache, self.v_cache, context.slot_mapping)

            if context.is_prefill:
                o = flash_attn_varlen_func(q, k, v, ...)
            else:
                o = flash_attn_with_kvcache(q, self.k_cache, self.v_cache, ...)

        return self.o_proj(o)
```

#### Phase 2: KV Cache Manager (1 day)

Create `src/forge/actors/hybrid/nano_kv_cache.py`:

```python
class NanoStyleKVCache:
    """Manages KV cache allocation and assignment to layers."""

    def __init__(self, model, num_blocks, block_size):
        self.num_blocks = num_blocks
        self.block_size = block_size

        # Count attention layers
        self.num_layers = sum(1 for m in model.modules()
                             if isinstance(m, NanoStyleAttention))

        # Get dimensions from first attention layer
        first_attn = next(m for m in model.modules()
                         if isinstance(m, NanoStyleAttention))
        self.num_kv_heads = first_attn.num_kv_heads
        self.head_dim = first_attn.head_dim

        # Allocate single large KV cache
        # Shape: [2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]
        self.kv_cache = torch.empty(
            2, self.num_layers, num_blocks, block_size,
            self.num_kv_heads, self.head_dim,
            dtype=torch.bfloat16, device='cuda'
        )

        # Assign cache to each attention layer
        layer_id = 0
        for module in model.modules():
            if isinstance(module, NanoStyleAttention):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def clear(self):
        """Clear KV cache."""
        self.kv_cache.zero_()
```

#### Phase 3: Simple Block Manager (1 day)

Copy nano-vLLM's `block_manager.py` with minimal changes:

```python
# src/forge/actors/hybrid/block_manager.py
# Copy from nano-vLLM with adjustments for our Sequence type
class Block:
    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

class BlockManager:
    def __init__(self, num_blocks, block_size):
        self.blocks = [Block(i) for i in range(num_blocks)]
        self.free_block_ids = deque(range(num_blocks))
        # ... (rest of nano-vLLM implementation)
```

#### Phase 4: Inference Context (1 day)

Create context manager for inference metadata:

```python
# src/forge/actors/hybrid/inference_context.py
from contextvars import ContextVar

_inference_context: ContextVar[InferenceContext | None] = ContextVar('inference_context', default=None)

class InferenceContext:
    """Inference metadata passed to attention layers."""

    def __init__(self, sequences, block_manager):
        self.is_prefill = True
        self.slot_mapping = None
        self.block_tables = None
        self.context_lens = None
        # ... (prepare from sequences)

def get_inference_context():
    return _inference_context.get()

@contextmanager
def inference_context(sequences, block_manager):
    ctx = InferenceContext(sequences, block_manager)
    token = _inference_context.set(ctx)
    try:
        yield ctx
    finally:
        _inference_context.reset(token)
```

#### Phase 5: Simple Scheduler (2 days)

Create simplified scheduler (no continuous batching):

```python
class SimpleScheduler:
    """Simplified scheduler for single-batch generation."""

    def __init__(self, block_manager):
        self.block_manager = block_manager

    def schedule(self, sequences):
        """Allocate blocks and prepare for generation."""
        for seq in sequences:
            self.block_manager.allocate(seq)

    def step(self, sequences):
        """Single decode step."""
        # Prepare context
        context = InferenceContext(sequences, self.block_manager)
        return context
```

#### Phase 6: Integration (1 day)

Integrate into `HybridPolicyActor`:

```python
# Modify __init__
if self.inference.use_simple_kv_cache:
    from forge.actors.hybrid.nano_attention import replace_attention_with_nano
    from forge.actors.hybrid.nano_kv_cache import NanoStyleKVCache
    from forge.actors.hybrid.block_manager import BlockManager

    # Replace attention layers
    replace_attention_with_nano(self.model)

    # Setup KV cache
    self.kv_cache = NanoStyleKVCache(self.model, num_blocks=1000, block_size=16)
    self.block_manager = BlockManager(num_blocks=1000, block_size=16)

    logger.info("Using simple KV cache (single model copy)")
```

## Complexity Comparison

| Component | Lines of Code | Complexity |
|-----------|---------------|------------|
| Nano-style attention layer | ~150 | Low |
| KV cache manager | ~50 | Low |
| Block manager | ~120 | Low |
| Inference context | ~100 | Low |
| Simple scheduler | ~80 | Low |
| Integration code | ~100 | Low |
| **Total** | **~600** | **Low-Medium** |

Compare to true single-copy with full vLLM: ~5,300 lines, Very High complexity.

## Expected Performance

### Memory
- **Single model copy**: 15GB for Qwen3-1.7B (vs 30GB for 2-copy)
- **KV cache**: ~8GB for 1000 blocks × 16 tokens
- **Total**: ~23GB (vs 30GB current)

### Speed
- **Decode**: 10-20x faster than naive (flash_attn_with_kvcache)
- **Not as fast as vLLM**: Missing continuous batching, CUDA graphs
- **Good enough**: 50-100 tok/s single sequence, 200-400 tok/s batched

### Limitations
1. **No continuous batching**: Can't dynamically add/remove sequences
2. **No CUDA graphs for decode**: Slightly slower than vLLM
3. **Simpler scheduler**: Less efficient GPU utilization

## Implementation Timeline

| Phase | Task | Time | Difficulty |
|-------|------|------|------------|
| 1 | Nano-style attention layer | 2 days | Medium |
| 2 | KV cache manager | 1 day | Low |
| 3 | Block manager (copy nano-vLLM) | 1 day | Low |
| 4 | Inference context | 1 day | Low |
| 5 | Simple scheduler | 2 days | Medium |
| 6 | Integration + testing | 1 day | Low |
| **Total** | | **8 days** | **Low-Medium** |

## Advantages vs Other Approaches

| Approach | Model Copies | Memory | Speed | Complexity | Dev Time |
|----------|--------------|--------|-------|------------|----------|
| **Nano-vLLM style (this plan)** | **1** | **23GB** | **10-20x** | **Low-Med** | **8 days** |
| SimpleVLLM (current) | 2 | 30GB | 50-100x | Low | Done ✓ |
| True single-copy (full vLLM) | 1 | 23GB | 50-100x | Very High | 3 weeks |
| TorchTitan vllm_compat | 2 | 30GB | 50-100x | High | Done (buggy) |

## Recommendation

**Implement nano-vLLM style approach**:

✅ **True single-copy** (15GB model + 8GB cache = 23GB)
✅ **Reasonable complexity** (~600 lines, 8 days)
✅ **Good speedup** (10-20x, good enough for RL)
✅ **Clean architecture** (minimal changes to training code)
⚠️ **Not as fast as vLLM** (missing continuous batching/CUDA graphs)
⚠️ **But much simpler** (600 lines vs 5,300 lines)

This is the **sweet spot** between complexity and performance:
- Avoids 2x memory overhead
- Achieves good (but not maximal) speedup
- Maintains code simplicity
- Can be implemented in 8 days

## Next Steps

1. ✅ **Phase 0**: Validate approach with nano-vLLM example (0.5 day)
2. **Phase 1-6**: Implement as outlined above (8 days)
3. **Testing**: Verify training + inference work correctly (2 days)
4. **Optimization**: Add CUDA graphs if needed (optional, +2 days)

**Total with validation**: 10-12 days for true single-copy with reasonable speedup.
