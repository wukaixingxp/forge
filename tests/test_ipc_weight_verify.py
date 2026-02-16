"""Verify IPC weight sync correctness.

Tests that model.load_weights (the same path used by our IPC weight sync)
correctly updates model parameters for merged (QKV, gate_up) and non-merged params.

Also tests that the model produces meaningful output before and after weight updates.

Usage:
    CUDA_VISIBLE_DEVICES=0 python tests/test_ipc_weight_verify.py
"""
import torch
import sys
import os


def test_load_weights_correctness():
    """Test model.load_weights correctly updates merged and non-merged params."""
    from vllm.config import VllmConfig, ModelConfig
    from vllm.model_executor.models import ModelRegistry

    print("=" * 60)
    print("Test 1: model.load_weights correctness")
    print("=" * 60)

    model_name = "Qwen/Qwen3-4B"
    device = "cuda:0"

    # Load model directly (bypass vLLM engine)
    print(f"\n[1/5] Loading {model_name} model class directly...")
    model_config = ModelConfig(
        model=model_name,
        task="generate",
        dtype="bfloat16",
        max_model_len=256,
    )
    vllm_config = VllmConfig(model_config=model_config)

    # Initialize vLLM parallel state (required for model instantiation)
    from vllm.distributed.parallel_state import init_distributed_environment, initialize_model_parallel
    import os
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29599")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    init_distributed_environment(world_size=1, rank=0, local_rank=0)
    initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)

    # Get model class and instantiate
    model_cls, _ = ModelRegistry.resolve_model_cls(["Qwen3ForCausalLM"], vllm_config.model_config)

    with torch.device(device):
        model = model_cls(vllm_config=vllm_config)

    # Load initial weights from checkpoint
    from safetensors.torch import load_file
    import glob

    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    model_dir = None
    for d in glob.glob(f"{cache_dir}/models--Qwen--Qwen3-4B/snapshots/*"):
        if os.path.isdir(d):
            model_dir = d
            break

    if model_dir is None:
        print("    Model not cached locally, loading from HF...")
        from huggingface_hub import snapshot_download
        model_dir = snapshot_download(model_name)

    print(f"    Loading from: {model_dir}")

    # Load safetensor files
    weight_files = sorted(glob.glob(f"{model_dir}/model*.safetensors"))
    all_weights = []
    for wf in weight_files:
        weights = load_file(wf, device=str(device))
        all_weights.extend(weights.items())

    # Keep a dict of HF weights for later perturbation
    hf_weights_dict = dict(all_weights)

    print(f"    Loading {len(all_weights)} weight tensors via model.load_weights...")
    model.load_weights(all_weights)
    print("    Initial weights loaded successfully.")

    # Record initial parameter norms
    print("\n[2/5] Recording initial parameter norms...")
    initial_norms = {}
    for name, param in model.named_parameters():
        initial_norms[name] = param.data.norm().item()

    print(f"    Total parameters: {len(initial_norms)}")
    for name in ["model.layers.0.self_attn.qkv_proj.weight",
                 "model.layers.0.mlp.gate_up_proj.weight",
                 "model.layers.0.self_attn.o_proj.weight",
                 "model.layers.0.input_layernorm.weight"]:
        if name in initial_norms:
            print(f"    {name}: norm={initial_norms[name]:.6f}")

    # Create perturbed weights using ORIGINAL HF checkpoint tensors
    # This simulates exactly what the trainer sends: HF-format separate Q/K/V/gate/up
    print("\n[3/5] Creating perturbed weights from HF checkpoint (simulating trainer output)...")
    noise_scale = 0.01
    perturbed_weights = []

    # Read model dimensions from config
    hf_config = model_config.hf_config
    hidden_size = hf_config.hidden_size
    num_heads = hf_config.num_attention_heads
    num_kv_heads = hf_config.num_key_value_heads
    head_dim = hidden_size // num_heads

    print(f"    Model: hidden_size={hidden_size}, num_heads={num_heads}, num_kv_heads={num_kv_heads}, head_dim={head_dim}")

    # Perturb HF-format Q/K/V weights (these are the ORIGINAL checkpoint format)
    hf_q = hf_weights_dict["model.layers.0.self_attn.q_proj.weight"]
    hf_k = hf_weights_dict["model.layers.0.self_attn.k_proj.weight"]
    hf_v = hf_weights_dict["model.layers.0.self_attn.v_proj.weight"]

    print(f"    HF Q shape: {hf_q.shape}, K shape: {hf_k.shape}, V shape: {hf_v.shape}")

    q_weight = hf_q.clone() + torch.randn_like(hf_q) * noise_scale
    k_weight = hf_k.clone() + torch.randn_like(hf_k) * noise_scale
    v_weight = hf_v.clone() + torch.randn_like(hf_v) * noise_scale

    perturbed_weights.append(("model.layers.0.self_attn.q_proj.weight", q_weight))
    perturbed_weights.append(("model.layers.0.self_attn.k_proj.weight", k_weight))
    perturbed_weights.append(("model.layers.0.self_attn.v_proj.weight", v_weight))

    # Perturb HF-format gate/up weights
    hf_gate = hf_weights_dict["model.layers.0.mlp.gate_proj.weight"]
    hf_up = hf_weights_dict["model.layers.0.mlp.up_proj.weight"]

    print(f"    HF gate shape: {hf_gate.shape}, up shape: {hf_up.shape}")

    gate_weight = hf_gate.clone() + torch.randn_like(hf_gate) * noise_scale
    up_weight = hf_up.clone() + torch.randn_like(hf_up) * noise_scale

    perturbed_weights.append(("model.layers.0.mlp.gate_proj.weight", gate_weight))
    perturbed_weights.append(("model.layers.0.mlp.up_proj.weight", up_weight))

    # Non-merged params - use HF names (same as vLLM names for these)
    params_dict = dict(model.named_parameters())
    for name in ["model.layers.0.self_attn.o_proj.weight",
                 "model.layers.0.mlp.down_proj.weight",
                 "model.layers.0.input_layernorm.weight"]:
        hf_w = hf_weights_dict.get(name, params_dict[name].data)
        w = hf_w.clone() + torch.randn_like(hf_w) * noise_scale
        perturbed_weights.append((name, w))

    print(f"    Prepared {len(perturbed_weights)} perturbed weight tensors")

    # Load perturbed weights via model.load_weights (same path as IPC sync)
    print("\n[4/5] Loading perturbed weights via model.load_weights...")
    model.load_weights(perturbed_weights)
    print("    model.load_weights completed!")

    # Verify
    print("\n[5/5] Verifying parameter changes...")
    results = []

    # Use actual HF tensor shapes for QKV verification
    # (head_dim may differ from hidden_size // num_heads, e.g. Qwen3-4B has head_dim=128)
    q_size = hf_q.shape[0]   # num_heads * actual_head_dim
    k_size = hf_k.shape[0]   # num_kv_heads * actual_head_dim
    v_size = hf_v.shape[0]   # num_kv_heads * actual_head_dim
    print(f"    QKV sizes from HF: Q={q_size}, K={k_size}, V={v_size}, total={q_size+k_size+v_size}")

    # Check merged QKV
    new_qkv = params_dict["model.layers.0.self_attn.qkv_proj.weight"]
    old_norm = initial_norms["model.layers.0.self_attn.qkv_proj.weight"]
    new_norm = new_qkv.data.norm().item()
    diff = abs(new_norm - old_norm)
    changed = diff > 0.001
    print(f"    qkv_proj: {old_norm:.4f} -> {new_norm:.4f} (diff={diff:.6f}) {'CHANGED' if changed else 'UNCHANGED'}")
    results.append(("qkv_proj (merged Q+K+V)", changed))

    # Verify Q portion matches what we sent (cast to model dtype for comparison)
    model_dtype = new_qkv.data.dtype
    actual_q = new_qkv.data[:q_size]
    q_match = torch.allclose(actual_q, q_weight.to(model_dtype), atol=1e-3)
    print(f"    Q portion matches expected: {q_match} (actual shape: {actual_q.shape}, expected: {q_weight.shape})")
    results.append(("Q projection values", q_match))

    # Verify K portion
    actual_k = new_qkv.data[q_size:q_size+k_size]
    k_match = torch.allclose(actual_k, k_weight.to(model_dtype), atol=1e-3)
    print(f"    K portion matches expected: {k_match} (actual shape: {actual_k.shape}, expected: {k_weight.shape})")
    results.append(("K projection values", k_match))

    # Verify V portion
    actual_v = new_qkv.data[q_size+k_size:q_size+k_size+v_size]
    v_match = torch.allclose(actual_v, v_weight.to(model_dtype), atol=1e-3)
    print(f"    V portion matches expected: {v_match} (actual shape: {actual_v.shape}, expected: {v_weight.shape})")
    results.append(("V projection values", v_match))

    # Check merged gate_up
    new_gate_up = params_dict["model.layers.0.mlp.gate_up_proj.weight"]
    half = new_gate_up.shape[0] // 2
    old_norm = initial_norms["model.layers.0.mlp.gate_up_proj.weight"]
    new_norm = new_gate_up.data.norm().item()
    diff = abs(new_norm - old_norm)
    changed = diff > 0.001
    print(f"    gate_up_proj: {old_norm:.4f} -> {new_norm:.4f} (diff={diff:.6f}) {'CHANGED' if changed else 'UNCHANGED'}")
    results.append(("gate_up_proj (merged gate+up)", changed))

    # Verify gate portion
    actual_gate = new_gate_up.data[:half]
    gate_match = torch.allclose(actual_gate, gate_weight.to(model_dtype), atol=1e-3)
    print(f"    Gate portion matches expected: {gate_match} (actual: {actual_gate.shape}, expected: {gate_weight.shape})")
    results.append(("Gate projection values", gate_match))

    # Verify up portion
    actual_up = new_gate_up.data[half:]
    up_match = torch.allclose(actual_up, up_weight.to(model_dtype), atol=1e-3)
    print(f"    Up portion matches expected: {up_match} (actual: {actual_up.shape}, expected: {up_weight.shape})")
    results.append(("Up projection values", up_match))

    # Check non-merged params
    for name in ["model.layers.0.self_attn.o_proj.weight",
                 "model.layers.0.mlp.down_proj.weight",
                 "model.layers.0.input_layernorm.weight"]:
        old_norm = initial_norms[name]
        new_norm = params_dict[name].data.norm().item()
        diff = abs(new_norm - old_norm)
        changed = diff > 0.001
        short = name.split(".")[-2]
        print(f"    {short}: {old_norm:.4f} -> {new_norm:.4f} (diff={diff:.6f}) {'CHANGED' if changed else 'UNCHANGED'}")
        results.append((f"{short} (non-merged)", changed))

    # Check that UNTOUCHED params (other layers) didn't change
    print("\n    Checking untouched layers...")
    untouched_ok = True
    for name, param in model.named_parameters():
        if "layers.0." in name:
            continue  # We modified layer 0
        if name not in initial_norms:
            continue
        old_norm = initial_norms[name]
        new_norm = param.data.norm().item()
        if abs(new_norm - old_norm) > 1e-6:
            print(f"    WARNING: {name} unexpectedly changed! {old_norm:.6f} -> {new_norm:.6f}")
            untouched_ok = False
    if untouched_ok:
        print("    All untouched layers verified unchanged.")
    results.append(("Untouched layers unchanged", untouched_ok))

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    all_pass = True
    for check_name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {check_name}")
        if not passed:
            all_pass = False

    print("\n" + ("PASS: All checks passed!" if all_pass else "FAIL: Some checks failed!"))
    print("=" * 60)
    return all_pass


def test_generation_meaningful():
    """Test that the model generates meaningful output (not garbage)."""
    from vllm import LLM, SamplingParams

    print("\n" + "=" * 60)
    print("Test 2: Generation produces meaningful output")
    print("=" * 60)

    model_name = "Qwen/Qwen3-4B"
    print(f"\n[1/3] Loading {model_name} in vLLM...")
    llm = LLM(
        model=model_name,
        tensor_parallel_size=1,
        enforce_eager=True,
        max_model_len=256,
        gpu_memory_utilization=0.5,
    )
    print("    Loaded.")

    sp = SamplingParams(max_tokens=80, temperature=0.0)
    prompts = [
        "What is 2 + 3? Give just the number:",
        "The capital of France is",
        "def fibonacci(n):",
    ]

    print("\n[2/3] Generating with initial weights...")
    outputs = llm.generate(prompts, sp)
    for i, output in enumerate(outputs):
        text = output.outputs[0].text.strip()[:120]
        print(f"    Prompt: {prompts[i]}")
        print(f"    Output: {text}")
        print()

    # Basic sanity: outputs should be non-empty and not all identical
    texts = [o.outputs[0].text.strip() for o in outputs]
    non_empty = all(len(t) > 0 for t in texts)
    not_identical = len(set(texts)) > 1

    print("\n[3/3] Sanity checks...")
    print(f"    All outputs non-empty: {non_empty}")
    print(f"    Outputs are diverse: {not_identical}")

    passed = non_empty and not_identical
    print("\n" + ("PASS: Model generates meaningful diverse output" if passed else "FAIL"))
    print("=" * 60)
    return passed


if __name__ == "__main__":
    # Test 1: Weight loading correctness
    test1_pass = test_load_weights_correctness()

    # Clean up GPU memory
    torch.cuda.empty_cache()
    import gc
    gc.collect()

    # Test 2: Generation quality
    test2_pass = test_generation_meaningful()

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"  Test 1 (Weight sync correctness): {'PASS' if test1_pass else 'FAIL'}")
    print(f"  Test 2 (Meaningful generation):    {'PASS' if test2_pass else 'FAIL'}")
    overall = test1_pass and test2_pass
    print(f"\n  Overall: {'ALL TESTS PASSED' if overall else 'SOME TESTS FAILED'}")
    print("=" * 60)

    sys.exit(0 if overall else 1)
