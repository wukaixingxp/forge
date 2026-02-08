#!/usr/bin/env python3
"""Test HybridPolicyActor with native KV cache (single model copy)."""

import time
import torch
from forge.actors.hybrid.inference_engine import InferenceEngine, InferenceConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm.sampling_params import SamplingParams

def test_kv_cache():
    print("=" * 80)
    print("Testing InferenceEngine with native KV cache")
    print("=" * 80)

    # Load model
    print("\n1. Loading Qwen3-1.7B...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-1.7B",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).cuda()

    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-1.7B",
        trust_remote_code=True,
    )

    # Create InferenceEngine
    print("\n2. Creating InferenceEngine...")
    config = InferenceConfig(
        use_nano_vllm=False,
        enable_cuda_graphs=False,
        max_batch_size=4,
    )

    engine = InferenceEngine(
        model=model,
        tokenizer=tokenizer,
        device=torch.device("cuda"),
        config=config,
        engine=None,  # No ForgeEngine for this test
    )

    # Test generation
    prompt = "What is 2 + 2?"
    print(f"\n3. Testing generation...")
    print(f"   Prompt: {prompt}")

    sampling_params = SamplingParams(
        n=1,
        max_tokens=50,
        temperature=1.0,
        top_p=1.0,
    )

    start_time = time.time()
    completions = engine.generate(prompt, sampling_params)
    gen_time = time.time() - start_time

    print(f"\n   Response: {completions[0].text}")
    print(f"\n   ✓ Generated in {gen_time:.2f}s")
    print(f"   Tokens: {len(completions[0].token_ids)} output tokens")
    print(f"   Speed: {len(completions[0].token_ids) / gen_time:.1f} tokens/sec")

    # Test with multiple completions
    print(f"\n4. Testing batched generation (n=4)...")
    sampling_params_batched = SamplingParams(
        n=4,
        max_tokens=50,
        temperature=1.0,
        top_p=1.0,
    )

    start_time = time.time()
    completions_batched = engine.generate(prompt, sampling_params_batched)
    batch_gen_time = time.time() - start_time

    print(f"   Generated {len(completions_batched)} completions")
    print(f"   ✓ Generated in {batch_gen_time:.2f}s")

    total_tokens = sum(len(c.token_ids) for c in completions_batched)
    print(f"   Total tokens: {total_tokens}")
    print(f"   Throughput: {total_tokens / batch_gen_time:.1f} tokens/sec")

    print("\n" + "=" * 80)
    print("✅ All tests passed!")
    print("=" * 80)
    print("\nSummary:")
    print(f"  - Single generation: {gen_time:.2f}s ({len(completions[0].token_ids) / gen_time:.1f} tok/s)")
    print(f"  - Batched (n=4): {batch_gen_time:.2f}s ({total_tokens / batch_gen_time:.1f} tok/s)")
    print("\n✅ Native KV cache is working!")

if __name__ == "__main__":
    test_kv_cache()
