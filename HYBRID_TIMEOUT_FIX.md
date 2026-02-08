# HybridPolicyActor Timeout Fix

## Problem

The training example with HybridPolicyActor was hanging after initialization when using FSDP with `procs: 2`. The logs showed:

```
All services initialized successfully!
🚀 Using HybridPolicyActor - zero weight sync overhead between train/infer!
Starting GRPO with 1 rollout threads, 1 training threads
[... warnings ...]
[HANG - no further progress]
```

## Root Cause

**FSDP Collective Operation Deadlock**

When using FSDP with multiple ranks (`procs: 2`), the HybridPolicyActor has two problems:

1. **Divergent Token Sampling**: Each rank independently sampled tokens during autoregressive generation, leading to different sequences on different ranks. This caused FSDP collective operations to deadlock because:
   - Rank 0 generated tokens: [123, 456, 789]
   - Rank 1 generated tokens: [111, 222, 333]
   - When calling `model(input_ids)`, FSDP requires all ranks to have the SAME input, but they had different tokens!

2. **Incorrect Return Value Handling**: The `generate()` endpoint returned results from all ranks, but with FSDP, only rank 0's results should be returned (other ranks participate in computation but don't return data).

## Fixes Applied

### Fix 1: Unwrap Multi-Process Actor Return Values

**File**: `apps/grpo/main_hybrid.py`

**Change**: When calling `.call()` on a multi-process actor, Monarch returns results from all ranks as a tuple. We need to unwrap and take rank 0's result:

```python
# .call() returns a tuple of results from all ranks, we take the first (rank 0)
responses_tuple = await hybrid_policy.generate.call(prompt)
responses: list[Completion] = responses_tuple[0] if isinstance(responses_tuple, tuple) else responses_tuple
```

**Why this is needed**:
- With `procs: 1`: `.call()` returns `(result,)` - a single-element tuple
- With `procs: 2`: `.call()` returns `(result_from_rank_0, result_from_rank_1)`
- We always want rank 0's result since that's where actual completions are returned

### Fix 2: Synchronize Token Sampling Across Ranks

**File**: `src/forge/actors/hybrid/inference_engine.py`

**Change**: Added `torch.distributed.broadcast()` to synchronize sampled tokens:

```python
# Sample next token
probs = F.softmax(next_token_logits, dim=-1)
next_token = torch.multinomial(probs, num_samples=1)  # [1, 1]

# Synchronize sampled token across FSDP ranks
# All ranks must use the same token for the next forward pass
if torch.distributed.is_initialized():
    # Broadcast from rank 0 to all other ranks
    torch.distributed.broadcast(next_token, src=0)
```

**Why this works**: Now all ranks use the exact same sequence of tokens, so FSDP collective operations succeed.

### Fix 3: Simplified Return Logic

**File**: `src/forge/actors/hybrid/policy_actor.py`

**Change**: All ranks return completions (they're identical due to token synchronization), and the caller unwraps the tuple:

```python
# All ranks must call generate() to participate in FSDP forward passes
# Token sampling is synchronized via broadcast in inference engine
params = sampling_params or self.sampling_params
completions = self.inference_engine.generate(prompt, params)

record_metric("hybrid_policy/generate/count_requests", 1, Reduce.SUM)
record_metric(
    "hybrid_policy/generate/count_sequences_completed",
    len(completions),
    Reduce.SUM,
)

t.stop()
return completions
```

**Why this works**:
- All ranks participate in forward passes (satisfies FSDP requirements)
- All ranks return identical completions (thanks to synchronized token sampling)
- The caller unwraps the tuple and takes rank 0's result (see Fix 1)

## Testing

### Test with 2 GPUs (FSDP)

Run the original config with the fixes:

```bash
python -m apps.grpo.main_hybrid --config apps/grpo/qwen3_1_7b_hybrid.yaml
```

Expected behavior:
- Initialization completes
- Rollout thread starts generating responses
- Training thread starts training
- No more timeouts or hangs

### Test with 1 GPU (No FSDP - Simpler)

For faster debugging without FSDP complexity:

```bash
python -m apps.grpo.main_hybrid --config apps/grpo/qwen3_1_7b_hybrid_1gpu.yaml
```

This config:
- Uses only 1 GPU (`procs: 1`)
- Reduced batch sizes and sequence lengths
- Only 10 training steps
- Compile disabled for faster startup

## Key Insights

1. **FSDP Requires Synchronized Inputs**: When using FSDP, all ranks MUST call forward() with identical inputs. Random sampling must be synchronized via broadcast.

2. **Only Rank 0 Returns Data**: For multi-rank actors with FSDP, endpoints should return data only from rank 0, while all ranks participate in computation.

3. **ReferenceModel Does This Correctly**: The ReferenceModel actor handles this correctly by:
   - All ranks running forward()
   - Returning DTensor.full_tensor() which handles rank coordination
   - But for generation, we need explicit broadcast because each token depends on the previous one

## Related Files

- `src/forge/actors/hybrid/policy_actor.py` - HybridPolicyActor with generate() endpoint
- `src/forge/actors/hybrid/inference_engine.py` - InferenceEngine with token sampling
- `apps/grpo/main_hybrid.py` - Main training loop
- `apps/grpo/qwen3_1_7b_hybrid.yaml` - 2 GPU FSDP config
- `apps/grpo/qwen3_1_7b_hybrid_1gpu.yaml` - 1 GPU test config (NEW)

## Verification

After applying fixes, you should see:

```
All services initialized successfully!
🚀 Using HybridPolicyActor - zero weight sync overhead between train/infer!
Torchstore initialized (not used for weight sync in hybrid mode)
Starting GRPO with 1 rollout threads, 1 training threads
[rank0]:W0208 00:46:47.944000 76319 ...
[... generation happens ...]
[... training happens ...]
[... metrics logged ...]
```

The training loop should progress through rollouts and training steps without hanging.
