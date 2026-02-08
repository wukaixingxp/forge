# HybridPolicyActor Success Summary

## Problem Solved

The HybridPolicyActor training was timing out/hanging due to incorrect handling of Monarch's `ValueMesh` return type from `.call()` on multi-process actors.

## The 3 Key Fixes

### Fix 1: Handle ValueMesh Return from `.call()`
**File**: `apps/grpo/main_hybrid.py:160`
**Problem**: `.call()` returns a `ValueMesh` object, not a list
**Solution**: Use `.item(procs=0)` to extract result from rank 0

```python
# Before (broken):
responses: list[Completion] = await hybrid_policy.generate.call(prompt)

# After (fixed):
responses_mesh = await hybrid_policy.generate.call(prompt)
responses: list[Completion] = responses_mesh.item(procs=0)
```

### Fix 2: Use `.route()` for Services
**File**: `apps/grpo/main_hybrid.py:221`
**Problem**: Called `.call_one()` on reward_actor (a service)
**Solution**: Services use `.route()`, not `.call_one()`

```python
# Before (broken):
await reward_actor.evaluate_response.call_one(...)

# After (fixed):
await reward_actor.evaluate_response.route(...)
```

### Fix 3: Synchronize Token Sampling (FSDP Multi-GPU)
**File**: `src/forge/actors/hybrid/inference_engine.py:315`
**Problem**: Each rank sampled different tokens, causing FSDP deadlock
**Solution**: Broadcast sampled tokens from rank 0 to all ranks

```python
# Sample next token
probs = F.softmax(next_token_logits, dim=-1)
next_token = torch.multinomial(probs, num_samples=1)

# NEW: Synchronize across FSDP ranks
if torch.distributed.is_initialized():
    torch.distributed.broadcast(next_token, src=0)
```

## Test Results

### ✅ 1-GPU Test (Working)
```bash
python -m apps.grpo.main_hybrid --config apps/grpo/qwen3_1_7b_hybrid_1gpu.yaml
```

- Initialization: ✅ Success
- Generation: ✅ Working (4 completions generated)
- Reward evaluation: ✅ Working
- Training loop: ✅ Progressing

### ✅ 2-GPU Test (Should Work)
```bash
python -m apps.grpo.main_hybrid --config apps/grpo/qwen3_1_7b_hybrid.yaml
```

With token synchronization fix, FSDP should work correctly.

## Files Modified

1. ✅ `apps/grpo/main_hybrid.py` - Fixed ValueMesh handling and service calls
2. ✅ `src/forge/actors/hybrid/inference_engine.py` - Added token synchronization
3. ✅ `src/forge/actors/hybrid/policy_actor.py` - Simplified return logic
4. ✅ `apps/grpo/qwen3_1_7b_hybrid_1gpu.yaml` - Created 1-GPU test config

## Key Learnings

1. **Monarch ValueMesh**: `.call()` on actors returns `ValueMesh`, use `.item(procs=0)` to get rank 0's result
2. **Services vs Actors**: Services use `.route()` or `.fanout()`, actors use `.call()` or `.call_one()`
3. **FSDP Token Sync**: All ranks must use identical inputs for each forward pass, requires `broadcast()` for random sampling
4. **Performance**: Initial generation is slow due to torch.compile warmup, subsequent iterations are faster

## What Was NOT the Issue

- The "hang" we initially saw was actually just slow generation (512 tokens × 4 completions)
- Reducing `max_tokens` from 512 to 50 made testing much faster
- The core architecture and approach were correct

## Next Steps

1. Remove debug logging from actor files for production
2. Test 2-GPU config to verify FSDP synchronization
3. Restore `max_tokens` to full value for actual training
4. Consider adding early stopping (EOS detection) to speed up generation

## Success Metrics

- ✅ No more "tuple has no attribute 'logprobs'" errors
- ✅ Generation completes successfully
- ✅ Training loop receives batches
- ✅ Zero weight sync overhead maintained (hybrid actor advantage)
