#!/usr/bin/env python3
"""Test that FSDP model can extract unwrapped module for KV cache."""

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import AutoModelForCausalLM, AutoTokenizer

def test_fsdp_unwrap():
    print("=" * 80)
    print("Testing FSDP unwrapped module access for KV cache")
    print("=" * 80)

    # Initialize distributed (required for FSDP)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    print(f"\nRank {rank}/{world_size}")

    # Load model
    print(f"\n[Rank {rank}] Loading Qwen3-1.7B...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-1.7B",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).cuda(rank)

    # Wrap with FSDP
    print(f"\n[Rank {rank}] Wrapping with FSDP...")
    fsdp_model = FSDP(model, device_id=rank)

    print(f"\n[Rank {rank}] FSDP model type: {type(fsdp_model)}")
    print(f"[Rank {rank}] Has _fsdp_wrapped_module: {hasattr(fsdp_model, '_fsdp_wrapped_module')}")

    # Try to access unwrapped module
    if hasattr(fsdp_model, '_fsdp_wrapped_module'):
        unwrapped = fsdp_model._fsdp_wrapped_module
        print(f"[Rank {rank}] ✓ Successfully accessed unwrapped module: {type(unwrapped)}")

        # Test if unwrapped module supports use_cache
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B", trust_remote_code=True)
        input_ids = tokenizer.encode("Hello", return_tensors="pt").cuda(rank)

        print(f"\n[Rank {rank}] Testing use_cache on unwrapped module...")
        with torch.no_grad():
            try:
                output = unwrapped(input_ids, use_cache=True)
                has_kv = hasattr(output, 'past_key_values') and output.past_key_values is not None
                print(f"[Rank {rank}] ✓ use_cache works! has_past_key_values: {has_kv}")

                if has_kv:
                    print(f"[Rank {rank}] ✓ KV cache shape: {output.past_key_values[0][0].shape}")
            except Exception as e:
                print(f"[Rank {rank}] ✗ Error: {e}")
    else:
        print(f"[Rank {rank}] ✗ Cannot access _fsdp_wrapped_module")

    print(f"\n[Rank {rank}] Test complete!")

    dist.destroy_process_group()

if __name__ == "__main__":
    test_fsdp_unwrap()
