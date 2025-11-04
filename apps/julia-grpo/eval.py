# -*- coding: utf-8 -*-
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Evaluation script for Julia code generation models.
Evaluates both baseline and fine-tuned models on julia_testset.parquet
"""

import argparse
import os
import re
import sys
from pathlib import Path

import pandas as pd
import requests
import torch
from datasets import Dataset
from tqdm import tqdm
from unsloth import FastLanguageModel

# Setup paths for OpenEnv
sys.path.insert(0, "./src")
working_directory = str(Path.cwd().parent.absolute() / "OpenEnv")

from envs.julia_env import JuliaEnv
from envs.julia_env.models import JuliaAction

# Julia code generation prompt
julia_code_gen_prompt = """
You are a precise and pragmatic Julia programmer.

Write a **single Julia function** that correctly solves the problem described below.

Rules:
- The code must be syntactically correct and runnable as is.
- Do not use arrow functions, ternary operators, or modern syntax that may cause issues.
- Use only the Julia standard library.
- Do **not** wrap the code in a module or add a `main` function.
- Do **not** include any test code in your response.
- Do **not** hardcode specific test cases or outputs — the function must work for general inputs.
- The **function name must exactly match** the one used in the provided tests.
- Respond with **only the Julia function** and nothing else (no explanations, no comments, no extra text)
- The function name must exactly match the one used in the provided tests.
- Return only the Julia function.
- character literal should not contain multiple characters.
- take care of object types and mind that spaces matter in julia so cannot add random spaces

Passing tests and clean, compilable code are rewarded. Hardcoding or failing tests is penalized.

Test Reference (for context only, do not include in the output):
{julia_test}

Code:
""".strip()


def remove_ticks(text):
    """Remove markdown code fences from generated text"""
    text = re.sub(r"^```julia\s*\n?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n?```\s*$", "", text)
    return text


def load_model(model_name, lora_path=None, max_seq_length=2000):
    """Load model with optional LoRA weights"""
    print(f"Loading model: {model_name}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        fast_inference=True,
        load_in_4bit=False,
        max_seq_length=max_seq_length,
        gpu_memory_utilization=0.85,
    )

    if lora_path:
        print(f"Loading LoRA weights from: {lora_path}")
        model = FastLanguageModel.get_peft_model(
            model,
            r=10,  # Must match training config
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_alpha=10,
            use_gradient_checkpointing="unsloth",
            random_state=3407,
        )

    return model, tokenizer


def setup_julia_env(localhost="http://localhost:8000"):
    """Setup connection to JuliaEnv server"""
    print(f"Connecting to Julia environment at {localhost}")

    try:
        response = requests.get(f"{localhost}/health", timeout=2)
        print(f"Julia environment health check: {response.content}")
    except Exception as e:
        print(f"Warning: Could not connect to Julia environment: {e}")
        print("Make sure the Julia environment server is running:")
        print("  docker run -d -p 8000:8000 --name julia-env-server julia-env:latest")
        raise

    openenv_process = JuliaEnv(base_url=localhost)
    result = openenv_process.reset()
    print(f"Julia environment initialized: {result.observation}")

    return openenv_process


def evaluate_code(openenv_process, core_code, test_code):
    """Evaluate a single piece of generated code"""
    try:
        result = openenv_process.reset()
        action = JuliaAction(core_code=core_code, test_code=test_code)
        result = openenv_process.step(action)

        reward = result.reward if result.reward is not None else 0.0
        passed = reward > 0.5  # Consider passed if reward > 0.5

        return {
            "reward": reward,
            "passed": passed,
            "observation": result.observation,
            "error": None,
        }
    except Exception as e:
        return {"reward": 0.0, "passed": False, "observation": None, "error": str(e)}


def run_evaluation(
    model_name,
    test_file,
    output_file,
    lora_path=None,
    localhost="http://localhost:8000",
    max_seq_length=2000,
    temperature=0.8,
    top_p=0.95,
    max_tokens=1024,
    num_samples=None,
):
    """Run full evaluation"""

    # Load model
    model, tokenizer = load_model(model_name, lora_path, max_seq_length)

    # Setup Julia environment
    openenv_process = setup_julia_env(localhost)

    # Load test dataset
    print(f"Loading test dataset from: {test_file}")
    df = pd.read_parquet(test_file)

    if num_samples:
        df = df.head(num_samples)
        print(f"Evaluating on {num_samples} samples")

    df = df[["julia_prompt", "julia_test", "task_id", "first_test_case"]]
    dataset = Dataset.from_pandas(df)

    # Prepare prompts
    MAX_LEN = max_seq_length
    dataset = dataset.map(
        lambda x: {
            "prompt": [
                {
                    "role": "system",
                    "content": julia_code_gen_prompt.format(
                        julia_test=x["first_test_case"]
                    )[:MAX_LEN],
                },
                {"role": "user", "content": x["julia_prompt"][:MAX_LEN]},
            ],
        }
    )

    # Setup sampling parameters
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    # Run evaluation
    results = []
    total_reward = 0.0
    total_passed = 0

    print(f"\nStarting evaluation on {len(dataset)} examples...")

    for i in tqdm(range(len(dataset)), desc="Evaluating Julia Code"):
        task_id = dataset[i]["task_id"]
        julia_test = dataset[i]["julia_test"]
        message = dataset[i]["prompt"]

        # Generate code
        text = tokenizer.apply_chat_template(
            message,
            add_generation_prompt=True,
            tokenize=False,
        )

        if lora_path:
            response = model.fast_generate(
                text,
                sampling_params=sampling_params,
                lora_request=model.load_lora(lora_path),
            )
        else:
            response = model.fast_generate(
                text,
                sampling_params=sampling_params,
            )

        julia_code = response[0].outputs[0].text
        julia_code = remove_ticks(julia_code)

        # Evaluate code
        eval_result = evaluate_code(openenv_process, julia_code, julia_test)

        # Track metrics
        total_reward += eval_result["reward"]
        if eval_result["passed"]:
            total_passed += 1

        # Store result
        results.append(
            {
                "task_id": task_id,
                "julia_code": julia_code,
                "julia_test": julia_test,
                "reward": eval_result["reward"],
                "passed": eval_result["passed"],
                "error": eval_result["error"],
                "observation": str(eval_result["observation"]),
            }
        )

    # Calculate final metrics
    num_examples = len(results)
    avg_reward = total_reward / num_examples
    pass_rate = (total_passed / num_examples) * 100

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Model: {model_name}")
    if lora_path:
        print(f"LoRA: {lora_path}")
    print(f"Test set: {test_file}")
    print(f"Number of examples: {num_examples}")
    print(f"Pass rate: {pass_rate:.2f}% ({total_passed}/{num_examples})")
    print(f"Average reward: {avg_reward:.4f}")
    print("=" * 60)

    # Save results
    result_df = pd.DataFrame(results)
    result_df.to_parquet(output_file, index=False)
    print(f"\n✅ Results saved to {output_file}")

    # Save summary
    summary = {
        "model_name": model_name,
        "lora_path": lora_path,
        "test_file": test_file,
        "num_examples": num_examples,
        "total_passed": total_passed,
        "pass_rate": pass_rate,
        "avg_reward": avg_reward,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }

    summary_file = output_file.replace(".parquet", "_summary.json")
    import json

    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"✅ Summary saved to {summary_file}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Julia code generation models"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="Model name or path",
    )
    parser.add_argument(
        "--lora_path", type=str, default=None, help="Path to LoRA weights (optional)"
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default="julia_testset.parquet",
        help="Path to test dataset",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output file path (default: auto-generated)",
    )
    parser.add_argument(
        "--localhost",
        type=str,
        default="http://localhost:8000",
        help="Julia environment server URL",
    )
    parser.add_argument(
        "--max_seq_length", type=int, default=2000, help="Maximum sequence length"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.8, help="Sampling temperature"
    )
    parser.add_argument(
        "--top_p", type=float, default=0.95, help="Nucleus sampling top_p"
    )
    parser.add_argument(
        "--max_tokens", type=int, default=1024, help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to evaluate (default: all)",
    )

    args = parser.parse_args()

    # Auto-generate output filename if not provided
    if args.output_file is None:
        model_name_slug = args.model_name.replace("/", "_").replace("-", "_")
        if args.lora_path:
            args.output_file = f"eval_results_{model_name_slug}_lora.parquet"
        else:
            args.output_file = f"eval_results_{model_name_slug}_baseline.parquet"

    # Run evaluation
    run_evaluation(
        model_name=args.model_name,
        test_file=args.test_file,
        output_file=args.output_file,
        lora_path=args.lora_path,
        localhost=args.localhost,
        max_seq_length=args.max_seq_length,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        num_samples=args.num_samples,
    )


if __name__ == "__main__":
    main()
