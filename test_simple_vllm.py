#!/usr/bin/env python3
"""Test simple vLLM adapter."""

import time
import sys

def test_simple_vllm():
    print("=" * 80)
    print("Testing Simple vLLM Adapter")
    print("=" * 80)

    # Test 1: Import simple vLLM adapter
    print("\n1. Importing simple vLLM adapter...")
    try:
        from src.forge.actors.hybrid.simple_vllm_adapter import SimpleVLLMEngine, SimpleVLLMConfig
        print("   ✓ Successfully imported SimpleVLLMEngine")
    except ImportError as e:
        print(f"   ✗ Failed to import: {e}")
        return False

    # Test 2: Check if vLLM is available
    print("\n2. Checking if vLLM is installed...")
    try:
        import vllm
        print(f"   ✓ vLLM version: {vllm.__version__}")
    except ImportError:
        print("   ✗ vLLM not installed")
        return False

    # Test 3: Create simple vLLM engine
    print("\n3. Creating simple vLLM engine with Qwen/Qwen3-0.6B...")
    try:
        from vllm import SamplingParams

        config = SimpleVLLMConfig(
            model_name="Qwen/Qwen3-0.6B",
            tensor_parallel_size=1,
            enable_cuda_graphs=True,
            max_num_seqs=4,
            gpu_memory_utilization=0.5,  # Use only 50% to leave room for other tests
        )

        engine = SimpleVLLMEngine(config=config)
        print("   ✓ Engine created successfully")
    except Exception as e:
        print(f"   ✗ Failed to create engine: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 4: Generate text (n=1)
    print("\n4. Generating single completion...")
    prompt = "What is 2 + 2?"
    sampling_params = SamplingParams(
        n=1,
        max_tokens=50,
        temperature=1.0,
        top_p=1.0,
        logprobs=1,
    )

    try:
        start_time = time.time()
        completions = engine.generate(prompt, sampling_params)
        gen_time = time.time() - start_time

        print(f"   Prompt: {prompt}")
        print(f"   Response: {completions[0].text}")
        print(f"   ✓ Generated in {gen_time:.2f}s")
        print(f"   Tokens: {len(completions[0].token_ids)} output tokens")
        print(f"   Speed: {len(completions[0].token_ids) / gen_time:.1f} tokens/sec")

        # Verify logprobs
        if completions[0].logprobs is not None:
            print(f"   Logprobs shape: {completions[0].logprobs.shape}")
        else:
            print("   ⚠ Logprobs not returned")

    except Exception as e:
        print(f"   ✗ Failed to generate: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 5: Test batched generation (n=4)
    print("\n5. Testing batched generation (n=4)...")
    sampling_params_batched = SamplingParams(
        n=4,
        max_tokens=50,
        temperature=1.0,
        top_p=1.0,
        logprobs=1,
    )

    try:
        start_time = time.time()
        completions_batched = engine.generate(prompt, sampling_params_batched)
        batch_gen_time = time.time() - start_time

        print(f"   Generated {len(completions_batched)} completions")
        print(f"   ✓ Generated in {batch_gen_time:.2f}s")

        total_tokens = sum(len(comp.token_ids) for comp in completions_batched)
        print(f"   Total tokens: {total_tokens}")
        print(f"   Throughput: {total_tokens / batch_gen_time:.1f} tokens/sec")

        # Verify all completions have logprobs
        has_logprobs = all(comp.logprobs is not None for comp in completions_batched)
        if has_logprobs:
            print(f"   ✓ All completions have logprobs")
        else:
            print("   ⚠ Some completions missing logprobs")

    except Exception as e:
        print(f"   ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 6: Get engine stats
    print("\n6. Checking engine statistics...")
    try:
        stats = engine.get_stats()
        print(f"   Engine: {stats['engine']}")
        print(f"   Model: {stats['model_name']}")
        print(f"   TP size: {stats['tensor_parallel_size']}")
        print(f"   CUDA graphs: {stats['cuda_graphs_enabled']}")
        print(f"   Paged attention: {stats['paged_attention']}")
        print("   ✓ Statistics retrieved")
    except Exception as e:
        print(f"   ✗ Failed to get stats: {e}")
        return False

    print("\n" + "=" * 80)
    print("✅ All tests passed!")
    print("=" * 80)
    print("\nSummary:")
    print(f"  - Single generation: {gen_time:.2f}s ({len(completions[0].token_ids) / gen_time:.1f} tok/s)")
    print(f"  - Batched (n=4): {batch_gen_time:.2f}s ({total_tokens / batch_gen_time:.1f} tok/s)")
    print("\n✅ Simple vLLM adapter is working!")

    return True


if __name__ == "__main__":
    success = test_simple_vllm()
    sys.exit(0 if success else 1)
