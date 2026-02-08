#!/usr/bin/env python3
"""Test TorchTitan + vLLM integration standalone."""

import time
import sys
from pathlib import Path

# Add torchtitan source to path
torchtitan_path = Path(__file__).parent.parent / "torchtitan"
if torchtitan_path.exists():
    sys.path.insert(0, str(torchtitan_path))
    print(f"Added TorchTitan source to path: {torchtitan_path}")

def test_torchtitan_vllm():
    print("=" * 80)
    print("Testing TorchTitan + vLLM Integration")
    print("=" * 80)

    # Test 1: Import TorchTitan's vLLM wrapper
    print("\n1. Importing TorchTitan vLLM wrapper...")
    try:
        from torchtitan.experiments.rl.unified import TorchTitanVLLMModelWrapper
        print("   ✓ Successfully imported TorchTitanVLLMModelWrapper")
    except ImportError as e:
        print(f"   ✗ Failed to import: {e}")
        print("\n   This is expected - TorchTitan's experimental vLLM wrapper may not be installed.")
        print("   Checking if vLLM is available...")
        return False

    # Test 2: Check if models are auto-registered
    print("\n2. Checking if Qwen3TorchTitanForCausalLM is registered...")
    try:
        from vllm.model_executor.models.registry import ModelRegistry
        registered_models = ModelRegistry.get_supported_archs()
        if "Qwen3TorchTitanForCausalLM" in registered_models:
            print("   ✓ Qwen3TorchTitanForCausalLM is registered")
        else:
            print(f"   ✗ Not found. Registered models: {registered_models[:10]}...")
            return False
    except Exception as e:
        print(f"   ⚠  Could not check registry: {e}")

    # Test 3: Try to create vLLM LLM with TorchTitan model
    print("\n3. Creating vLLM LLM with TorchTitan model...")
    try:
        from vllm import LLM, SamplingParams

        llm = LLM(
            model="Qwen3TorchTitanForCausalLM",
            tensor_parallel_size=1,
            enforce_eager=True,  # Disable CUDA graphs for testing
            max_num_seqs=4,
            trust_remote_code=True,
        )
        print("   ✓ vLLM LLM created successfully")
    except Exception as e:
        print(f"   ✗ Failed to create LLM: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 4: Generate text
    print("\n4. Generating text...")
    prompt = "What is 2 + 2?"
    sampling_params = SamplingParams(
        n=1,
        max_tokens=50,
        temperature=1.0,
        top_p=1.0,
    )

    try:
        start_time = time.time()
        outputs = llm.generate([prompt], sampling_params, use_tqdm=False)
        gen_time = time.time() - start_time

        print(f"   Prompt: {prompt}")
        print(f"   Response: {outputs[0].outputs[0].text}")
        print(f"   ✓ Generated in {gen_time:.2f}s")
        print(f"   Tokens: {len(outputs[0].outputs[0].token_ids)} output tokens")
        print(f"   Speed: {len(outputs[0].outputs[0].token_ids) / gen_time:.1f} tokens/sec")
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
    )

    try:
        start_time = time.time()
        outputs_batched = llm.generate([prompt], sampling_params_batched, use_tqdm=False)
        batch_gen_time = time.time() - start_time

        print(f"   Generated {len(outputs_batched[0].outputs)} completions")
        print(f"   ✓ Generated in {batch_gen_time:.2f}s")

        total_tokens = sum(len(out.token_ids) for out in outputs_batched[0].outputs)
        print(f"   Total tokens: {total_tokens}")
        print(f"   Throughput: {total_tokens / batch_gen_time:.1f} tokens/sec")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 80)
    print("✅ All tests passed!")
    print("=" * 80)
    print("\nSummary:")
    print(f"  - Single generation: {gen_time:.2f}s ({len(outputs[0].outputs[0].token_ids) / gen_time:.1f} tok/s)")
    print(f"  - Batched (n=4): {batch_gen_time:.2f}s ({total_tokens / batch_gen_time:.1f} tok/s)")
    print("\n✅ TorchTitan + vLLM integration is working!")

    return True


if __name__ == "__main__":
    success = test_torchtitan_vllm()
    sys.exit(0 if success else 1)
