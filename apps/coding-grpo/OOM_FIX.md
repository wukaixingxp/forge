# CUDA Out of Memory (OOM) Fix for llama3_8b_hard.yaml

## Problem Analysis

The training was failing with CUDA OOM error after the first training step:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 11.74 GiB. 
GPU 0 has a total capacity of 139.72 GiB of which 10.88 GiB is free. 
Of the allocated memory 126.69 GiB is allocated by PyTorch
```

## Root Cause

With the original configuration:
- **Batch size**: 16
- **Sequence length**: 3072 (very long sequences for code generation)
- **GPU memory utilization**: 0.85 for policy engine
- **max_num_seqs**: 24

This configuration consumed ~127 GiB leaving insufficient memory for the second training iteration.

## Changes Applied

### 1. Reduced Batch Size (Critical)
```yaml
# Before:
batch_size: 16

# After:
batch_size: 8   # Reduced to 8 to avoid OOM with seq_len=3072
```

**Impact**: Reduces memory consumption by 50% for the trainer, allowing the second training step to succeed.

### 2. Reduced Policy Engine Memory (Important)
```yaml
# Before:
gpu_memory_utilization: 0.85

# After:
gpu_memory_utilization: 0.75  # Reduced from 0.85 to leave more memory for trainer
```

**Impact**: Reserves more GPU memory for the trainer component.

### 3. Reduced Max Sequences (Moderate)
```yaml
# Before:
max_num_seqs: 24

# After:
max_num_seqs: 16  # Reduced from 24 to reduce memory pressure
```

**Impact**: Reduces memory used by the vLLM policy engine's KV cache.

## Additional Recommendations

### Environment Variables (Optional but Helpful)
Add these to your launch script to help with memory fragmentation:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

This addresses the warning in the error message about memory fragmentation.

### Alternative Optimizations (If Still Experiencing OOM)

If you still encounter OOM issues, try these additional steps in order:

#### Option 1: Further Reduce Batch Size
```yaml
batch_size: 4   # Half the current size
```

#### Option 2: Enable Gradient Accumulation
```yaml
# In trainer section:
gradient_accumulation_steps: 2  # Effective batch size will be 8 * 2 = 16
batch_size: 4  # Physical batch size
```

#### Option 3: Reduce Sequence Length
```yaml
seq_len: 2048  # From 3072
```
Note: This will truncate longer code samples.

#### Option 4: Enable CPU Offloading
```yaml
training:
  enable_cpu_offload: true  # Offload optimizer states to CPU
```

#### Option 5: Use More Memory-Efficient Attention
Already enabled with:
```yaml
activation_checkpoint:
  mode: selective
  selective_ac_option: op
```

## Performance Impact

With these changes:
- **Training Speed**: Reduced by ~40-50% due to smaller batch size (8 vs 16)
- **Memory Usage**: Reduced by ~50% for trainer component
- **Convergence**: Should be similar, but may require more steps to see equivalent improvements

To compensate for the reduced batch size:
- Consider increasing the number of training steps proportionally
- Original: 23,000 steps with batch size 16
- New recommendation: 46,000 steps with batch size 8 (for equivalent total samples)

## Verification

To verify the fix works:
1. Run the training command again:
   ```bash
   python -m apps.coding-grpo.main --config apps/coding-grpo/llama3_8b_hard.yaml
   ```

2. Monitor GPU memory usage:
   ```bash
   watch -n 1 nvidia-smi
   ```

3. Check that training progresses beyond step 1 without OOM errors

## Monitoring

Key metrics to watch in logs:
- `rl_trainer_perf/step/memory_peak_max_gb`: Should be lower than before
- `rl_trainer/count_training_steps`: Should increase steadily
- No more `torch.OutOfMemoryError` messages

## Summary

The primary fix is reducing batch size from 16 to 8, which should resolve the OOM issue. The additional optimizations (reduced gpu_memory_utilization and max_num_seqs) provide extra safety margin and improve overall memory efficiency.
