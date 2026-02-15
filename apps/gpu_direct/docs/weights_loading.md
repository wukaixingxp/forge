# IPC Weight Loading: Deep Dive

How weights transfer from an FSDP trainer (2 GPUs) to a TP vLLM generator (2 GPUs) via CUDA IPC handles.

## Overview

The IPC weight sync bypasses TorchStore entirely. Instead of serializing full tensors through RPC, the trainer exports 66-byte CUDA IPC handles that point to GPU memory, and generator workers reconstruct tensors directly from those handles.

**We do NOT use vLLM's `model.load_weights`.** The FSDP+TP path uses `param.data.copy_()` with manual name mapping, merging, and TP slicing. This section explains why, whether that's correct, and where it breaks.

## The Full 2x2 Data Flow

```
Trainer GPU 0 (FSDP rank 0)     Trainer GPU 1 (FSDP rank 1)
┌─────────────────────────┐     ┌─────────────────────────┐
│ DTensor._local_tensor   │     │ DTensor._local_tensor   │
│ = top half of each param│     │ = bottom half of each   │
│                         │     │   param                 │
│ create_ipc_handle()     │     │ create_ipc_handle()     │
│ → 66-byte handle        │     │ → 66-byte handle        │
└───────────┬─────────────┘     └───────────┬─────────────┘
            │ get_shard_ipc_handles()        │
            └──────────┬─────────────────────┘
                       │ handles + metadata (offsets, shapes)
                       ▼
              Generator (orchestrator)
              update_weights_ipc()
              ┌────────────────────────┐
              │ Overlaps handle        │
              │ collection with        │
              │ pause_generation       │
              │                        │
              │ Then sends ALL shard   │
              │ handles to ALL workers │
              └───────────┬────────────┘
                          │ receive_shards_ipc()
            ┌─────────────┴─────────────┐
            ▼                           ▼
    Worker GPU 2 (TP rank 0)    Worker GPU 3 (TP rank 1)
    ┌─────────────────────┐     ┌─────────────────────────┐
    │ 1. reconstruct from │     │ 1. reconstruct from     │
    │    both IPC handles │     │    both IPC handles     │
    │ 2. _combine_shards  │     │ 2. _combine_shards     │
    │    → full tensor    │     │    → full tensor        │
    │ 3. _slice_for_tp    │     │ 3. _slice_for_tp       │
    │    or _copy_to_     │     │    or _copy_to_         │
    │    merged_param     │     │    merged_param         │
    │ 4. param.data.copy_ │     │ 4. param.data.copy_    │
    └─────────────────────┘     └─────────────────────────┘
```

## Step-by-Step Trace

### Step 1: Trainer creates IPC handles

**File:** `src/forge/actors/trainer/titan.py` → `get_shard_ipc_handles()`

Each FSDP rank holds a shard of the model as DTensors. The method:

1. Calls `model.state_dict()` → returns DTensors (NOT full tensors)
2. Converts native names to HF names via `sd_adapter.to_hf()` or `_native_to_hf_name()`
3. For each DTensor parameter:
   - Extracts `param._local_tensor` (the local FSDP shard, NO all_gather)
   - Computes `offsets` via `_compute_local_shape_and_global_offset()` (where this shard fits globally)
   - Creates a 66-byte IPC handle via `create_ipc_handle(local_shard)`
4. Returns `{handles, metadata, fsdp_rank, fsdp_size}`

**Example for `model.layers.0.self_attn.q_proj.weight`** (Qwen3-4B, shape `[2048, 2048]`):
- FSDP rank 0: local shard shape `[1024, 2048]`, offsets `(0, 0)`
- FSDP rank 1: local shard shape `[1024, 2048]`, offsets `(1024, 0)`

### Step 2: Generator overlaps handle collection with pause

**File:** `src/forge/actors/vllm/v1/generator.py` → `update_weights_ipc()`

```python
handle_task = asyncio.create_task(_collect_handles())  # runs on trainer
pause_task = asyncio.create_task(self.llm.pause_generation(...))  # runs on generator
shard_results, _ = await asyncio.gather(handle_task, pause_task)
```

Handle collection runs on trainer GPUs (doesn't touch generator model), so it safely overlaps with `pause_generation` which waits for in-flight requests on the generator. This hides handle creation time behind the ~9.7s pause.

### Step 3: Workers receive and process shards

**File:** `src/forge/actors/vllm/v1/forge_executor.py` → `receive_shards_ipc()`

Each TP worker processes ALL parameters through these substeps:

#### 3a. Reconstruct tensors from IPC handles

```python
for fsdp_rank in range(fsdp_size):
    shard_tensor = handle.reconstruct_tensor().to(target_device).clone()
```

`reconstruct_tensor()` uses the 66-byte handle to map the trainer's GPU memory. The `.clone()` is critical - it copies the data so the trainer can free its memory after the sync.

#### 3b. Combine FSDP shards → full tensor

**Method:** `_combine_shards(shards, offsets_list, global_shape)`

```python
indexed_shards = list(zip(offsets_list, shards))
indexed_shards.sort(key=lambda x: x[0][0])  # sort by dim-0 offset
full_tensor = torch.cat(sorted_shards, dim=0)
```

Sorts shards by their offset in dimension 0 and concatenates. FSDP always shards on dim 0 for standard parameters.

**Example (q_proj):**
- Shard 0: shape `[1024, 2048]`, offset `(0, 0)`
- Shard 1: shape `[1024, 2048]`, offset `(1024, 0)`
- Combined: shape `[2048, 2048]` ← full q_proj weight

#### 3c. Route to correct loading method

The `_build_param_map()` method creates a mapping from HF names to vLLM parameters:

```python
param_map = {
    # Direct mappings (non-merged params)
    "model.layers.0.self_attn.o_proj.weight": <param>,
    "model.layers.0.mlp.down_proj.weight": <param>,

    # Merged mappings (q/k/v → qkv_proj)
    "model.layers.0.self_attn.q_proj.weight": ("qkv_proj_q", <qkv_param>),
    "model.layers.0.self_attn.k_proj.weight": ("qkv_proj_k", <qkv_param>),
    "model.layers.0.self_attn.v_proj.weight": ("qkv_proj_v", <qkv_param>),

    # Merged mappings (gate/up → gate_up_proj)
    "model.layers.0.mlp.gate_proj.weight": ("gate_up_proj_gate", <gate_up_param>),
    "model.layers.0.mlp.up_proj.weight": ("gate_up_proj_up", <gate_up_param>),
}
```

For each parameter, routing depends on the mapping type:

| Mapping type | Method | What it does |
|-------------|--------|-------------|
| `tuple` (merged) | `_copy_to_merged_param()` | TP-slice source, compute offset in merged param, copy |
| `param` (direct, shapes match) | `param.data.copy_()` | Direct copy, no slicing needed |
| `param` (direct, shapes differ) | `_slice_for_tp()` then `copy_()` | Slice for TP rank, then copy |

#### 3d. Merged parameter handling (QKV)

**Method:** `_copy_to_merged_param(merge_type, tensor, param, tp_rank, tp_size)`

This is the most complex part. vLLM internally merges separate Q, K, V projections into a single `qkv_proj` parameter. We receive separate Q, K, V tensors from the trainer and must place them at the correct offsets.

**vLLM's qkv_proj memory layout** (per TP rank):
```
┌─────────────────────────────────────────────────────┐
│  Q heads for this rank  │  K heads  │  V heads      │
│  [num_q_heads * head_d] │ [num_kv * │ [num_kv *     │
│                         │  head_d]  │  head_d]      │
└─────────────────────────────────────────────────────┘
```

**For Qwen3-4B** (32 Q heads, 8 KV heads, head_dim=64, TP=2):
```
Per TP rank qkv_proj shape: [1536, 2048]
  Q: 16 heads × 64 = 1024  (offset 0)
  K:  4 heads × 64 =  256  (offset 1024)
  V:  4 heads × 64 =  256  (offset 1280)
  Total: 1536
```

**Our approach** (the heuristic):
```python
# For Q: start at 0
start_idx = 0

# For K: Q_size = total - 2*K_size (since K=V)
q_size_per_tp = total_qkv_size - 2 * sliced_kv_size
start_idx = q_size_per_tp  # = 1536 - 2*256 = 1024 ✓

# For V: after Q and K
start_idx = q_size_per_tp + sliced_kv_size  # = 1024 + 256 = 1280 ✓
```

The heuristic `Q = total - 2*K` relies on the fact that **K and V are always the same size** in all known transformer architectures. This is true for standard attention (Q=K=V), GQA (Q > K=V), and MQA (Q >> K=V).

**TP slicing of source tensors:**
```python
src_part_size = src_shape[0] // tp_size
tensor = tensor[tp_rank * src_part_size : (tp_rank + 1) * src_part_size]
```

This divides the full Q/K/V weight evenly among TP ranks. For Q with 32 heads, each TP rank gets 16 heads. For K/V with 8 heads, each gets 4 heads.

#### 3e. Merged parameter handling (gate_up_proj)

vLLM merges `gate_proj` and `up_proj` into a single `gate_up_proj`:

```
┌───────────────────────────────────┐
│   gate_proj    │    up_proj       │
│  [inter/tp, h] │  [inter/tp, h]  │
└───────────────────────────────────┘
```

The code splits 50/50:
```python
part_size = total_size // 2  # gate and up are always the same size
start_idx = 0 if part == 'gate' else part_size
```

#### 3f. Non-merged parameter TP slicing

**Method:** `_slice_for_tp(name, tensor, target_shape, tp_rank, tp_size)`

Uses pattern matching on parameter names:

| Pattern | Parallel type | Shard dim | Example |
|---------|--------------|-----------|---------|
| `q_proj`, `k_proj`, `v_proj`, `qkv_proj` | Column | dim 0 (output) | `tensor[rank*size:(rank+1)*size, :]` |
| `gate_proj`, `up_proj`, `gate_up_proj` | Column | dim 0 (output) | `tensor[rank*size:(rank+1)*size, :]` |
| `o_proj`, `down_proj` | Row | dim 1 (input) | `tensor[:, rank*size:(rank+1)*size]` |
| `embed_tokens`, `lm_head` | Vocab | dim 0 (vocab) | `tensor[rank*size:(rank+1)*size, :]` |
| `layernorm`, `norm` | Replicated | none | tensor unchanged |

## Comparison: Our Approach vs vLLM's `model.load_weights`

### What vLLM's `model.load_weights` does

vLLM's approach (used in `receive_weights_ipc` for the non-FSDP path):

1. **Model-specific routing**: Each model (Llama, Qwen3, etc.) has a `load_weights()` method with a `stacked_params_mapping` that maps HF names to merged vLLM params:
   ```python
   stacked_params_mapping = [
       (".qkv_proj", ".q_proj", "q"),
       (".qkv_proj", ".k_proj", "k"),
       (".qkv_proj", ".v_proj", "v"),
       (".gate_up_proj", ".gate_proj", 0),
       (".gate_up_proj", ".up_proj", 1),
   ]
   ```

2. **Per-parameter weight loaders**: Each vLLM parameter type (`QKVParallelLinear`, `RowParallelLinear`, etc.) has a `weight_loader` method that knows the model config:
   ```python
   # QKVParallelLinear knows exact head counts
   shard_offset = {"q": 0, "k": num_q_heads*head_dim, "v": (num_q_heads+num_kv_heads)*head_dim}
   shard_size = {"q": num_q_heads*head_dim, "k": num_kv_heads*head_dim, "v": num_kv_heads*head_dim}
   ```

3. **KV head replication**: When `tp_size >= total_num_kv_heads`, vLLM replicates KV heads:
   ```python
   # parameter.py line 194
   shard_id = self.tp_rank if shard_id == "q" else self.tp_rank // num_kv_head_replicas
   ```

4. **Automatic TP slicing**: The weight loader slices the source tensor for the correct TP rank automatically using `loaded_weight.narrow(output_dim, shard_id * shard_size, shard_size)`.

### What our approach does

The FSDP+TP path (`receive_shards_ipc`) reimplements this logic:

1. **Generic name mapping**: `_build_param_map()` discovers merged params by scanning vLLM model parameters for `qkv_proj` and `gate_up_proj` patterns. Model-agnostic.

2. **Heuristic offsets**: `_copy_to_merged_param()` computes QKV offsets using `Q = total - 2*K` instead of knowing exact head counts.

3. **Pattern-based TP slicing**: `_slice_for_tp()` matches parameter names to determine column/row/vocab parallel type.

### Why not use `model.load_weights` for the FSDP+TP path?

`model.load_weights` expects **full tensors** with HF-style names. For the FSDP+TP path, we need to:

1. **Combine shards first** - `model.load_weights` doesn't know about FSDP shards
2. **Handle per-parameter** - We need to combine shards, then load. `model.load_weights` processes weights as a stream and doesn't support this two-phase approach.

A potential improvement would be to combine shards into full tensors first, then pass them through `model.load_weights`. This would leverage vLLM's battle-tested weight loading logic and handle all edge cases. The trade-off is that `model.load_weights` does more work per parameter (name resolution, quantization handling) than direct `param.data.copy_()`.

## Correctness Analysis

### What works correctly

| Scenario | Status | Notes |
|----------|--------|-------|
| Standard attention (Q=K=V heads) | **Correct** | All sizes equal, heuristic is trivially correct |
| GQA (Q > K=V), e.g., Qwen3, Llama3 | **Correct** | K=V assumption holds, verified with Qwen3-4B (32Q/8KV) |
| Column parallel (q/k/v/gate/up) | **Correct** | Slices dim 0 by target_shape |
| Row parallel (o_proj, down_proj) | **Correct** | Slices dim 1 by target_shape |
| Vocab parallel (embed_tokens, lm_head) | **Correct** | Slices dim 0, handles padding via target_shape |
| Replicated (norm, layernorm) | **Correct** | No slicing, shapes match |
| Tied weights (lm_head = embed_tokens) | **Correct** | `_build_param_map` maps lm_head to embed_tokens if not found separately |
| FSDP dim-0 sharding | **Correct** | `_combine_shards` sorts by offset[0] and concatenates dim 0 |

### Known issues and edge cases

#### 1. KV Head Replication (tp_size > num_kv_heads)

**Severity: HIGH for affected configs**

When `tp_size >= total_num_kv_heads`, vLLM replicates KV heads across TP ranks. Our code divides KV by `tp_size` which produces incorrect (too small) slices.

**Example:** A model with 2 KV heads and TP=4:
- vLLM: Each rank gets 1 KV head (replicated 2x). `num_kv_head_replicas = 2`
- Our code: `kv_size / 4 = 2 * head_dim / 4 = 0.5 * head_dim` → wrong

**Affected models:** Models with very few KV heads relative to TP size. NOT affected for common configs:
- Qwen3-4B (8 KV heads, TP=2): 4 per rank → OK
- Llama3-8B (8 KV heads, TP=2): 4 per rank → OK
- Llama3-8B (8 KV heads, TP=4): 2 per rank → OK
- Llama3-8B (8 KV heads, TP=8): 1 per rank → OK (8 divides 8)
- MQA model (1 KV head, TP=2): **BROKEN** (replication needed)

**Fix:** Read `num_kv_heads` and `head_size` from the model config via `vllm_config.model_config.hf_config` and compute exact offsets instead of using the heuristic. Or, combine shards into full tensors and pass through `model.load_weights`.

#### 2. Non-dim-0 FSDP Sharding

**Severity: LOW**

`_combine_shards` only concatenates along dim 0. FSDP2 with DTensor almost always uses `Shard(0)` placement, but custom placements could use other dimensions.

**Mitigation:** The offsets metadata from `_compute_local_shape_and_global_offset` already encodes the correct dimension. A more robust implementation would use the offsets to determine the shard dimension rather than hardcoding dim 0.

#### 3. MoE Expert Parallel

**Severity: MEDIUM for MoE models**

`_slice_for_tp` doesn't handle expert-parallel patterns. MoE models like Qwen3-30B-A3B (MoE) or DeepSeek have expert weights that may be distributed differently.

**Missing patterns:**
- `experts.*.gate_proj` / `experts.*.up_proj` / `experts.*.down_proj`
- `router.weight` (replicated)
- Expert parallel vs tensor parallel interaction

#### 4. Bias Terms

**Severity: LOW**

`_slice_for_tp` and `_copy_to_merged_param` handle weight tensors but not bias terms. Most modern LLMs (Llama, Qwen3, Mistral) have `bias=False` for attention and MLP, so this rarely matters.

**Exception:** Qwen2 (not Qwen3) uses attention bias. If running Qwen2 with TP>1, bias terms would not be correctly sliced.

#### 5. Quantized Models

**Severity: HIGH for quantized models**

Our code does raw `param.data.copy_()` which doesn't handle quantization packing. vLLM's `weight_loader` has special handling for:
- FP8 scale remapping
- GPTQ/AWQ packed weights
- Block quantization scales

**Mitigation:** The current implementation targets BF16/FP16 training where quantization is not involved. Quantized model weights would need the vLLM weight loader path.

## Model Compatibility Matrix

| Model | Architecture | GQA | TP=2 | TP=4 | TP=8 | Notes |
|-------|-------------|-----|------|------|------|-------|
| Qwen3-4B | Qwen3 | 32Q/8KV | **OK** | **OK** | **OK** | Tested, primary target |
| Qwen3-32B | Qwen3 | 64Q/8KV | **OK** | **OK** | **OK** | Same arch as 4B |
| Llama3-8B | Llama | 32Q/8KV | **OK** | **OK** | **OK** | Needs llama3 config |
| Llama3-70B | Llama | 64Q/8KV | **OK** | **OK** | **OK** | Same arch |
| Llama3.1-405B | Llama | 128Q/8KV | **OK** | **OK** | **WARN** | TP=16 would need KV replication |
| Mistral-7B | Llama-like | 32Q/8KV | **OK** | **OK** | **OK** | Uses Llama arch in vLLM |
| Gemma-2-9B | Gemma | 16Q/8KV | **OK** | **OK** | **OK** | Standard GQA |
| Phi-3 | Phi | 32Q/32KV | **OK** | **OK** | **OK** | Standard attention (Q=K=V) |
| DeepSeek-V2 | MoE | GQA+MoE | **WARN** | **WARN** | **WARN** | MoE expert parallel not handled |
| Qwen3-30B-A3B | MoE | GQA+MoE | **WARN** | **WARN** | **WARN** | MoE expert parallel not handled |

**OK** = Works correctly with current implementation
**WARN** = May have issues (MoE patterns, KV replication)

## Recommendations

### Short-term (for public release)

1. **Document supported configs**: Only Qwen3-4B configs are tested. Add explicit configs for Llama3-8B.
2. **Add runtime validation**: After weight sync, compare a few parameter checksums between trainer and generator.
3. **Guard against KV replication**: Add a check in `receive_shards_ipc` that errors if `tp_size > total_num_kv_heads`.

### Medium-term (robustness)

4. **Use `model.load_weights` after shard combination**: Combine FSDP shards into full tensors, then pass through vLLM's `model.load_weights`. This would handle all edge cases (KV replication, quantization, model-specific logic) and eliminate the manual name mapping, merging, and TP slicing code.

   ```python
   # Proposed approach
   for param_name in param_names:
       shards = [reconstruct(handle) for handle in shard_handles]
       full_tensor = combine_shards(shards, offsets)
       full_weights.append((param_name, full_tensor))

   # Let vLLM handle everything: merging, TP slicing, KV replication
   model.load_weights(full_weights)
   ```

   This is the most impactful improvement - it would make IPC weight sync work with ANY model that vLLM supports, automatically.

5. **Read model config for head counts**: If keeping manual loading, read `num_attention_heads`, `num_key_value_heads`, and `head_dim` from `vllm_config.model_config.hf_config` to compute exact QKV offsets instead of the heuristic.

### Long-term (optimization)

6. **Pipeline shard processing**: Instead of waiting for all shards to be combined before starting TP slicing, process each parameter as soon as its shards arrive.
7. **Avoid clone() for same-GPU**: When trainer and generator share the same GPU (e.g., 1x1 config), the clone is unnecessary overhead.
