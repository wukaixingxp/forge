# Hybrid Actor Solutions

This directory contains configurations for two hybrid training+inference solutions that eliminate weight synchronization overhead between training and generation.

## Solutions Overview

| Solution | Model Copies | Memory | Speed | Complexity |
|----------|--------------|--------|-------|------------|
| **Simple KV Cache** | 1 | ~23GB | 10-20x | Low |
| **SimpleVLLM** | 2 | ~30GB | 50-100x | Low |

## 1. Simple KV Cache (`simple_kv_cache.yaml`)

**Single model copy** with nano-vLLM style KV cache.

### Features
- Training and inference share the **same model in GPU memory**
- Context-based mode switching (no weight copying)
- Paged KV cache with automatic prefix caching
- Block-based memory management

### Usage
```bash
python -m apps.hybrid.main_hybrid --config apps/hybrid/simple_kv_cache.yaml
```

### Memory Breakdown
- Model (Qwen3-1.7B FSDP): 15GB
- KV Cache (256 blocks × 256 tokens): 8GB
- **Total: ~23GB**

### Performance
- **10-20x faster** than naive autoregressive generation
- Mode switch overhead: ~10-50ms
- Ideal for: Memory-constrained setups, simpler codebase

### Implementation
- `src/forge/actors/hybrid/simple_kv_cache_engine.py`
- `src/forge/actors/hybrid/nano_style_attention.py`
- `src/forge/actors/hybrid/block_manager.py`
- ~600 lines of logic

## 2. SimpleVLLM (`simple_vllm.yaml`)

**Separate vLLM instance** using standard HuggingFace checkpoint loading.

### Features
- Training model (FSDP) + separate vLLM inference model
- Full vLLM optimizations (paged KV cache, CUDA graphs, continuous batching)
- Standard vLLM API compatibility

### Usage
```bash
python -m apps.hybrid.main_hybrid --config apps/hybrid/simple_vllm.yaml
```

### Memory Breakdown
- Training model (Qwen3-1.7B FSDP): 15GB
- vLLM inference model: 15GB
- **Total: ~30GB**

### Performance
- **50-100x faster** than naive autoregressive generation
- Full vLLM performance with CUDA graphs
- Ideal for: Maximum inference speed, GPU memory available

### Implementation
- `src/forge/actors/hybrid/simple_vllm_adapter.py`
- Uses standard vLLM library

## Choosing a Solution

**Use Simple KV Cache when:**
- GPU memory is limited (<32GB)
- You want simpler, more maintainable code
- 10-20x speedup is sufficient
- You prefer single model copy in memory

**Use SimpleVLLM when:**
- GPU memory is abundant (>32GB)
- You need maximum inference speed
- You want full vLLM feature compatibility
- You're comfortable with two model copies

## Implementation Details

Both solutions use `HybridPolicyActor` which combines:
- **Training Mode**: TorchTitan ForgeEngine with FSDP
- **Inference Mode**: Lightweight inference engine (Simple KV Cache or SimpleVLLM)

The key innovation is eliminating expensive weight synchronization (~1-3 seconds) by keeping the model in GPU memory and just switching modes (~10-50ms).
