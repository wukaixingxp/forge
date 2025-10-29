# GRPO Training Fixes - Complete Summary

## 🔍 Original Problems Identified

Based on WandB analysis, the following critical issues were detected:

1. **Extreme Negative Policy Loss**: Loss dropping to **-1000**
2. **Near-Zero KL Penalty**: ~0.00004 (no meaningful regularization)
3. **Low Reward Signals**: Stuck in 0.1-0.6 range
4. **Learning Rate**: Was very small (1e-7 to 2e-8 in practice)

## ✅ Fixes Applied to `/home/kaiwu/work/kaiwu/forge/apps/coding-grpo/main.py`

### 1. **Increased KL Coefficient (Beta)**
```python
# Before: beta = 0.01
# After:  beta = 0.05
```
- **Impact**: Should increase KL penalty from near-zero to meaningful values
- **Purpose**: Better regularization to prevent policy from diverging too much from reference model

### 2. **Improved Numerical Stability**

Added comprehensive NaN/Inf detection and handling:

```python
# In loss computation:
if torch.isnan(logprobs).any() or torch.isinf(logprobs).any():
    print("WARNING: NaN/Inf detected in logprobs!")
    logprobs = torch.nan_to_num(logprobs, nan=0.0, posinf=0.0, neginf=-100.0)

# In final loss:
if torch.isnan(loss) or torch.isinf(loss):
    print("WARNING: NaN/Inf detected in final loss!")
    loss = torch.tensor(0.0, device=loss.device, requires_grad=True)

# In rewards and advantages:
if torch.isnan(rewards).any() or torch.isinf(rewards).any():
    print(f"WARNING: NaN/Inf detected in rewards: {rewards}")
    rewards = torch.nan_to_num(rewards, nan=0.0, posinf=1.0, neginf=0.0)
```

### 3. **Enhanced Advantage Computation**

#### Adaptive Minimum Std
```python
# Before: Fixed minimum std = 0.1
# After:  Adaptive std based on reward scale
min_std = max(0.01, rewards.abs().mean().item() * 0.1)
```
- **Benefit**: Scales normalization based on actual reward magnitude
- **Prevents**: Division by very small values when rewards are small

#### Increased Advantage Clipping Range
```python
# Before: advantages.clamp(-2.0, 2.0)
# After:  advantages.clamp(-5.0, 5.0)
```
- **Benefit**: Allows more gradient signal to flow through
- **Prevents**: Over-aggressive clipping that removes learning signal

#### Better Normalization
```python
# Using unbiased=False for more stable std calculation
std = rewards.std(1, keepdim=True, unbiased=False)
```

### 4. **Comprehensive Debug Logging**

Added extensive metrics for debugging:

```python
# Loss metrics:
- loss/kl_raw                   # Raw KL before beta scaling
- loss/per_token_loss_mean      # Mean per-token loss
- loss/per_token_loss_std       # Std dev of per-token loss
- loss/per_token_loss_min       # Minimum per-token loss
- loss/per_token_loss_max       # Maximum per-token loss
- loss/logprobs_mean            # Mean logprobs from current policy
- loss/ref_logprobs_mean        # Mean logprobs from reference model
- loss/delta_mean               # Mean KL delta

# Reward metrics:
- rewards/raw_mean              # Mean raw rewards before normalization
- rewards/raw_std               # Std dev of raw rewards
- rewards/raw_min               # Minimum raw reward
- rewards/raw_max               # Maximum raw reward

# Advantage metrics:
- advantages/normalization_std  # Std used for normalization
- advantages/normalization_mean # Mean used for normalization
```

## 📊 Configuration Already Correct

In `/home/kaiwu/work/kaiwu/forge/apps/coding-grpo/qwen3_8b.yaml`:

### Learning Rate ✅
```yaml
optimizer:
  name: AdamW
  lr: 1e-5          # ✅ Within recommended 5e-6 to 1e-5 range
  eps: 1e-8
```

### Gradient Clipping ✅
```yaml
training:
  max_norm: 1.0     # ✅ Proper gradient clipping already enabled
```

## 🎯 Expected Improvements

After these changes, you should see:

1. **KL Penalty increases** from ~0.00004 to meaningful values (0.001-0.01)
2. **Policy Loss stabilizes** to reasonable range (not -1000)
3. **Rewards improve** over training steps
4. **No NaN/Inf crashes** with safety fallbacks
5. **Better gradient flow** with relaxed advantage clipping

## 📈 Monitoring Recommendations

Watch these new metrics in WandB:

### Primary Metrics to Track:
- `loss/kl_penalty` - Should be non-zero and relatively stable
- `loss/kl_raw` - Raw KL divergence (before beta scaling)
- `loss/policy_loss` - Should be in reasonable range (-10 to 10)
- `loss/per_token_loss_mean` - Overall loss magnitude

### Debugging Metrics:
- `loss/per_token_loss_min/max` - Check for extreme values
- `rewards/raw_*` - Monitor reward distribution
- `advantages/normalization_*` - Verify normalization is working

### Warning Signs to Watch For:
- If `loss/kl_penalty` is still near zero → Increase beta further (try 0.1)
- If `loss/per_token_loss_max` > 50 → Model is unstable, check rewards
- If console shows "WARNING: NaN/Inf detected" → Investigate reward function

## 🔧 Further Tuning Options

If issues persist, try these adjustments:

### Increase KL Coefficient Further
```python
# In main.py, line 172
beta: float = 0.1  # Increase from 0.05 to 0.1
```

### Adjust Advantage Clipping
```python
# In main.py, line 332
advantages = torch.clamp(advantages, min=-10.0, max=10.0)  # More signal
# OR
advantages = torch.clamp(advantages, min=-2.0, max=2.0)   # More stability
```

### Try Different Learning Rate
```yaml
# In qwen3_8b.yaml, line 56
lr: 5e-6  # Try lower if training is unstable
```

## 📝 Testing Protocol

1. **Start Training**: Run with current config
2. **Check Within First 10 Steps**:
   - `loss/kl_penalty` should be > 0.001
   - `loss/policy_loss` should be between -50 and 50
   - No NaN/Inf warnings in console
3. **Check After 100 Steps**:
   - `rewards/raw_mean` should show improvement
   - `loss/total_loss` should be decreasing
4. **If Problems Persist**: Adjust beta or advantage clipping as noted above

## 🔄 Rollback Instructions

If changes cause issues, the key parameters to revert:

```python
# Revert to previous stable values:
beta = 0.01                              # From 0.05
advantages = torch.clamp(advantages, min=-2.0, max=2.0)  # From [-5, 5]
```

---

**Last Updated**: 2025-10-29
**Config File**: `/home/kaiwu/work/kaiwu/forge/apps/coding-grpo/qwen3_8b.yaml`
**Main File**: `/home/kaiwu/work/kaiwu/forge/apps/coding-grpo/main.py`
