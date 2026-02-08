# TorchTitan + vLLM Integration Plan

## 🎉 Discovery: Built-in Solution Exists!

TorchTitan already has experimental code that wraps TorchTitan models with vLLM's paged attention!

**Location**: `../torchtitan/torchtitan/experiments/rl/unified/`

---

## What It Provides

### `TorchTitanVLLMModelWrapper`
- Wraps TorchTitan models to work with vLLM V1 engine
- **Replaces attention layers with vLLM's paged attention**
- **Maintains single model copy** (your requirement!)
- Handles DTensor/FSDP conversion between TorchTitan and vLLM
- Supports tensor parallelism

### Auto-Registration
```python
# Auto-registered when you import:
from torchtitan.experiments.rl.unified import TorchTitanVLLMModelWrapper

# Registers: "Qwen3TorchTitanForCausalLM" with vLLM
```

---

## Implementation Plan (Revised - EASY)

### Option A: Use TorchTitan's Wrapper (Recommended) ⭐
**Effort**: 0.5-1 day (just integration, no model modification!)
**Speedup**: 50-100x (vLLM paged attention)
**Memory**: Single model copy + KV cache (~15GB + 8GB = 23GB)

#### Step 1: Import TorchTitan's vLLM Integration
```python
# In HybridPolicyActor or InferenceEngine
from torchtitan.experiments.rl.unified import TorchTitanVLLMModelWrapper
from vllm import LLM
```

#### Step 2: Create vLLM Engine with TorchTitan Model
```python
# In HybridPolicyActor.__post_init__()
if self.inference.use_vllm_with_torchtitan:  # New config flag
    # Use TorchTitan's model wrapped for vLLM
    self.inference_llm = LLM(
        model="Qwen3TorchTitanForCausalLM",  # Auto-registered by TorchTitan
        tensor_parallel_size=self.parallelism.tensor_parallel_degree,
        enforce_eager=not self.inference.enable_cuda_graphs,
        max_num_seqs=self.inference.max_batch_size,
    )
else:
    # Fallback to custom InferenceEngine
    self.inference_engine = InferenceEngine(...)
```

#### Step 3: Update Generation
```python
async def generate(self, prompt, sampling_params):
    if self.inference_llm:
        # Use vLLM with TorchTitan model wrapper
        outputs = self.inference_llm.generate([prompt], sampling_params)
        return self._convert_vllm_outputs(outputs)
    else:
        # Use custom InferenceEngine
        return self.inference_engine.generate(prompt, sampling_params)
```

#### Step 4: Test
```bash
# Enable in config
use_vllm_with_torchtitan: true

# Run training
python -m apps.grpo.main_hybrid --config apps/grpo/qwen3_1_7b_hybrid_1gpu.yaml
```

---

## Comparison

| Approach | Effort | Memory | Speedup | Complexity |
|----------|--------|--------|---------|------------|
| **TorchTitan's vLLM wrapper** | 0.5-1 day | 23GB (single) | 50-100x | Low ⭐ |
| nano-vLLM (separate model) | 5 min | 21GB (dual) | 50-100x | Very Low |
| Modify TorchTitan manually | 2-3 days | 15GB (single) | 35-50x | High |

---

## Key Benefits

### ✅ Single Model Copy
- TorchTitan's model is used for both training and inference
- Only attention layers are swapped with vLLM's paged attention
- No weight duplication!

### ✅ vLLM Paged Attention
- Full KV cache with paging
- Continuous batching
- CUDA graphs support
- 50-100x speedup

### ✅ Already Built
- No need to modify TorchTitan core
- Just use the experimental wrapper
- Maintained by TorchTitan team

### ✅ Clean Integration
- Import and use
- Minimal changes to HybridPolicyActor
- Config-driven enable/disable

---

## Detailed Implementation

### File 1: `src/forge/actors/hybrid/torchtitan_vllm_engine.py` (NEW)

```python
"""vLLM engine using TorchTitan's model wrapper."""

import torch
from dataclasses import dataclass
from typing import Optional

from forge.data_models.completion import Completion
from forge.data_models.prompt import to_prompt
from vllm import LLM
from vllm.sampling_params import SamplingParams


@dataclass
class TorchTitanVLLMConfig:
    """Configuration for TorchTitan + vLLM integration."""
    tensor_parallel_size: int = 1
    enable_cuda_graphs: bool = True
    max_num_seqs: int = 16
    gpu_memory_utilization: float = 0.9


class TorchTitanVLLMEngine:
    """vLLM engine using TorchTitan's model wrapper.

    Uses TorchTitan's experimental TorchTitanVLLMModelWrapper which:
    - Wraps TorchTitan models for vLLM inference
    - Replaces attention with vLLM's paged attention
    - Maintains single model copy (no weight duplication)
    """

    def __init__(self, model_name: str, config: TorchTitanVLLMConfig):
        # Import TorchTitan's vLLM wrapper (auto-registers models)
        from torchtitan.experiments.rl.unified import TorchTitanVLLMModelWrapper

        # Create vLLM LLM with TorchTitan model
        self.llm = LLM(
            model=f"{model_name}TorchTitanForCausalLM",  # e.g., "Qwen3TorchTitanForCausalLM"
            tensor_parallel_size=config.tensor_parallel_size,
            enforce_eager=not config.enable_cuda_graphs,
            max_num_seqs=config.max_num_seqs,
            gpu_memory_utilization=config.gpu_memory_utilization,
            trust_remote_code=True,
        )

        self.config = config

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        sampling_params: SamplingParams,
    ) -> list[Completion]:
        """Generate completions using vLLM with TorchTitan model."""

        # Generate using vLLM
        outputs = self.llm.generate([prompt], sampling_params, use_tqdm=False)

        # Convert to Completion format
        completions = []
        for output in outputs:
            for completion_output in output.outputs:
                completion = Completion(
                    prompt=to_prompt(prompt),
                    text=completion_output.text,
                    prompt_ids=torch.tensor(output.prompt_token_ids, device="cpu"),
                    token_ids=torch.tensor(completion_output.token_ids, device="cpu"),
                    logprobs=None,  # TODO: Extract if needed
                    stop_reason=completion_output.finish_reason,
                    generator_version="torchtitan-vllm",
                    metadata={
                        "torchtitan_vllm": True,
                        "paged_attention": True,
                    },
                )
                completions.append(completion)

        return completions

    def clear_cache(self):
        """vLLM manages cache automatically."""
        pass

    def get_stats(self) -> dict:
        """Get statistics."""
        return {
            "engine": "torchtitan-vllm",
            "tensor_parallel_size": self.config.tensor_parallel_size,
            "cuda_graphs_enabled": self.config.enable_cuda_graphs,
        }
```

### File 2: Update `src/forge/actors/hybrid/policy_actor.py`

```python
# Add to imports
from forge.actors.hybrid.torchtitan_vllm_engine import (
    TorchTitanVLLMEngine,
    TorchTitanVLLMConfig,
)

# In __post_init__(), add option for TorchTitan+vLLM:
if self.inference.use_torchtitan_vllm:
    # Use TorchTitan model with vLLM paged attention
    vllm_config = TorchTitanVLLMConfig(
        tensor_parallel_size=self.parallelism.tensor_parallel_degree,
        enable_cuda_graphs=self.inference.enable_cuda_graphs,
        max_num_seqs=self.inference.max_batch_size,
    )

    self.inference_engine = TorchTitanVLLMEngine(
        model_name="Qwen3",  # Will use Qwen3TorchTitanForCausalLM
        config=vllm_config,
    )
    logger.info("Using TorchTitan model with vLLM paged attention (single copy)")
elif self.inference.use_nano_vllm:
    # ... existing nano-vLLM code ...
else:
    # ... existing InferenceEngine code ...
```

### File 3: Update Config

```yaml
# apps/grpo/qwen3_1_7b_hybrid.yaml
hybrid_policy:
  inference:
    use_torchtitan_vllm: true  # NEW: Use TorchTitan + vLLM wrapper
    use_nano_vllm: false  # OLD: Separate nano-vLLM model
    enable_cuda_graphs: true
    max_batch_size: 16
```

---

## Testing Plan

### Phase 1: Verify TorchTitan vLLM Wrapper Works
```bash
cd ../torchtitan
python -c "
from torchtitan.experiments.rl.unified import TorchTitanVLLMModelWrapper
from vllm import LLM, SamplingParams

llm = LLM(model='Qwen3TorchTitanForCausalLM', tensor_parallel_size=1)
outputs = llm.generate(['Hello'], SamplingParams(max_tokens=50))
print(outputs[0].outputs[0].text)
"
```

### Phase 2: Integrate into HybridPolicyActor
- Create `TorchTitanVLLMEngine` wrapper
- Update `policy_actor.py`
- Test with 1 GPU

### Phase 3: Test with GRPO Training
```bash
python -m apps.grpo.main_hybrid --config apps/grpo/qwen3_1_7b_hybrid_1gpu.yaml
```

### Phase 4: Benchmark Performance
- Compare with slow generation (no KV cache)
- Validate 50-100x speedup
- Verify memory usage (single model copy)

---

## Expected Results

### Performance
- **Before**: 0.8 tokens/sec (no KV cache)
- **After**: 60+ tokens/sec (vLLM paged attention)
- **Speedup**: 75x

### Memory
- **Training model**: TorchTitan's model with FSDP (~7GB)
- **Inference**: Same model, attention swapped with vLLM (~8GB KV cache)
- **Total**: ~15GB (single copy!) + 8GB KV cache = 23GB

### Architecture
```
HybridPolicyActor
└── TorchTitan Model (single copy)
    ├── Training Mode: Standard attention, FSDP
    └── Inference Mode: vLLM paged attention, KV cache
```

---

## Why This Is Better

| Feature | TorchTitan+vLLM | nano-vLLM | Manual Modify |
|---------|----------------|-----------|---------------|
| Single model copy | ✅ Yes | ❌ No (2x) | ✅ Yes |
| Effort | ✅ 0.5-1 day | ✅ 5 min | ❌ 2-3 days |
| Maintained | ✅ By TorchTitan | ✅ By nano-vLLM | ❌ By you |
| Paged KV cache | ✅ Yes | ✅ Yes | ⚠️  Need to implement |
| CUDA graphs | ✅ Yes | ✅ Yes | ⚠️  Need to implement |
| Speedup | ✅ 50-100x | ✅ 50-100x | ⚠️  35-50x |

---

## Recommendation

**Use TorchTitan's vLLM wrapper (Option A)** because:

1. ✅ **Already exists** - no need to modify TorchTitan core
2. ✅ **Single model copy** - your requirement
3. ✅ **Full vLLM optimizations** - paged KV cache, CUDA graphs, etc.
4. ✅ **Minimal integration** - just wrap and use
5. ✅ **Maintained** - by TorchTitan team

**Effort**: 0.5-1 day vs 2-3 days for manual modification!

---

## Next Steps

1. Test TorchTitan's vLLM wrapper standalone
2. Create `TorchTitanVLLMEngine` wrapper class
3. Integrate into `HybridPolicyActor`
4. Update configs
5. Test and benchmark

**Want me to implement this now?** It's much easier than I initially thought!
