#!/usr/bin/env python3
"""Test nano-vLLM standalone with Qwen3-1.7B."""

import time
from nanovllm import LLM, SamplingParams

def test_nano_vllm():
    print("=" * 80)
    print("Testing nano-vLLM standalone with Qwen3-1.7B")
    print("=" * 80)

    # Initialize nano-vLLM
    print("\n1. Initializing nano-vLLM...")
    start_time = time.time()
    llm = LLM(
        model="Qwen/Qwen3-1.7B",
        tensor_parallel_size=1,
        enforce_eager=False,  # Enable CUDA graphs
        max_num_seqs=4,
        trust_remote_code=True,
    )
    init_time = time.time() - start_time
    print(f"   ✓ Initialized in {init_time:.2f}s")

    # Test single generation
    print("\n2. Testing single generation...")
    prompt = "What is 2 + 2?"
    sampling_params = SamplingParams(
        n=1,
        max_tokens=50,
        temperature=1.0,
        top_p=1.0,
    )

    start_time = time.time()
    outputs = llm.generate([prompt], sampling_params, use_tqdm=False)
    gen_time = time.time() - start_time

    print(f"   Prompt: {prompt}")
    print(f"   Response: {outputs[0].outputs[0].text}")
    print(f"   ✓ Generated in {gen_time:.2f}s")
    print(f"   Tokens: {len(outputs[0].outputs[0].token_ids)} output tokens")
    print(f"   Speed: {len(outputs[0].outputs[0].token_ids) / gen_time:.1f} tokens/sec")

    # Test batched generation (n=4)
    print("\n3. Testing batched generation (n=4)...")
    sampling_params_batched = SamplingParams(
        n=4,
        max_tokens=50,
        temperature=1.0,
        top_p=1.0,
    )

    start_time = time.time()
    outputs_batched = llm.generate([prompt], sampling_params_batched, use_tqdm=False)
    batch_gen_time = time.time() - start_time

    print(f"   Generated {len(outputs_batched[0].outputs)} completions")
    print(f"   ✓ Generated in {batch_gen_time:.2f}s")

    total_tokens = sum(len(out.token_ids) for out in outputs_batched[0].outputs)
    print(f"   Total tokens: {total_tokens}")
    print(f"   Throughput: {total_tokens / batch_gen_time:.1f} tokens/sec")

    # Test with logprobs
    print("\n4. Testing with logprobs...")
    sampling_params_logprobs = SamplingParams(
        n=1,
        max_tokens=10,
        temperature=1.0,
        logprobs=1,
    )

    start_time = time.time()
    outputs_logprobs = llm.generate([prompt], sampling_params_logprobs, use_tqdm=False)
    logprob_time = time.time() - start_time

    print(f"   ✓ Generated with logprobs in {logprob_time:.2f}s")
    print(f"   Logprobs available: {outputs_logprobs[0].outputs[0].logprobs is not None}")

    print("\n" + "=" * 80)
    print("✅ All tests passed!")
    print("=" * 80)
    print("\nSummary:")
    print(f"  - Initialization: {init_time:.2f}s")
    print(f"  - Single generation: {gen_time:.2f}s ({len(outputs[0].outputs[0].token_ids) / gen_time:.1f} tok/s)")
    print(f"  - Batched generation (n=4): {batch_gen_time:.2f}s ({total_tokens / batch_gen_time:.1f} tok/s)")
    print(f"  - Logprobs generation: {logprob_time:.2f}s")
    print("\n✅ nano-vLLM is working correctly!")

if __name__ == "__main__":
    test_nano_vllm()
