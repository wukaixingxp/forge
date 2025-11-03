#!/usr/bin/env python3
"""
GRPO Training Script for Python Code Generation with Deep Thinking
Train Llama models using Group Relative Policy Optimization with deep thinking encouragement:
- Always encourage deep thinking with <think></think> tags
- Reward high-quality thinking tokens
- Evaluation: Use ground truth test cases from AceCode OSS subset
"""

import argparse
import asyncio
import json
import os
import re

import traceback
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from peft import get_peft_model, LoraConfig, TaskType

warnings.filterwarnings("ignore")

import gc
import uuid

import aiohttp
import requests

import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer


# Convenience functions for external use
async def execute_code_remotely(
    code: str, server_url: str = "http://127.0.0.1:8222"
) -> dict:
    """
    Execute code on a remote execution server asynchronously.
    """
    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        ) as session:
            async with session.post(
                f"{server_url}/execute", json={"code": code}
            ) as response:
                response.raise_for_status()
                return await response.json()
    except Exception as e:
        return {
            "execution_id": str(uuid.uuid4()),
            "output": "",
            "success": False,
            "error": str(e),
        }


# Synchronous wrapper for backward compatibility
def execute_code_remotely_sync(
    code: str, server_url: str = "http://127.0.0.1:8222"
) -> dict:
    """
    Synchronous wrapper for execute_code_remotely.
    """
    return asyncio.run(execute_code_remotely(code, server_url))


# =====================================================================


# ============================================================================
# Deep Thinking System Prompt
# ============================================================================


def get_system_prompt():
    """Get system prompt that encourages deep thinking."""

    return """You are an expert Python programmer who writes clean, efficient, and well-tested code.

IMPORTANT: You must structure your response in two parts:
1. First, write your detailed reasoning and problem-solving approach inside <think></think> tags
2. Then, provide your final solution after the thinking section

Given a problem description, follow this process:

<think>
- Carefully analyze the problem requirements and constraints
- Break down the problem into smaller components
- Consider multiple approaches and their trade-offs
- Think through edge cases and potential issues
- Plan your algorithm step by step
- Consider time and space complexity
- Think about how to test your solution thoroughly
</think>

Then write a Python function that solves the problem following these guidelines:
1. Write clean, readable, and efficient Python code
2. Add a comprehensive docstring explaining what the function does
3. Handle edge cases appropriately
4. Use only standard library imports unless specified otherwise
5. Ensure your solution is correct and robust

Format your response as:
<think>
[Your detailed reasoning, analysis, and planning here]
</think>

```python
def function_name(parameters):
    \"\"\"Comprehensive docstring explaining the function.\"\"\"
    # Implementation
    pass
```"""


# ============================================================================
# Utility Functions with Thinking Support
# ============================================================================


def extract_thinking_content(text: str) -> Optional[str]:
    """Extract content from <think></think> tags."""
    pattern = r"<think>(.*?)</think>"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else None


def remove_thinking_tags(text: str) -> str:
    """Remove <think></think> tags and their content from text."""
    pattern = r"<think>.*?</think>"
    return re.sub(pattern, "", text, flags=re.DOTALL | re.IGNORECASE).strip()


def extract_python_code(text: str) -> Optional[str]:
    """Extract Python code from markdown code blocks or raw text, ignoring thinking sections."""
    # First remove thinking sections
    text = remove_thinking_tags(text)

    patterns = [r"```python\n(.*?)\n```", r"```py\n(.*?)\n```", r"```\n(.*?)\n```"]

    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()

    return text.strip()


# ============================================================================
# Simple Reward Functions with Deep Thinking Support
# ============================================================================


class RewardFunctions:
    """Collection of reward functions for evaluating Python code quality with deep thinking."""

    @staticmethod
    def thinking_reward(prompts, completions, **kwargs) -> List[float]:
        """Simple binary thinking reward: +20 for thinking content, -20 for none."""
        rewards = []
        for completion in completions:
            content = completion[0]["content"]
            has_thinking = (
                "<think>" in content.lower() and "</think>" in content.lower()
            )

            if not has_thinking:
                # No thinking tags at all
                rewards.append(-20.0)
                continue

            # Extract thinking content
            thinking_content = extract_thinking_content(content)
            if thinking_content and thinking_content.strip():
                # Has actual thinking content - reward
                rewards.append(20.0)
            else:
                # Has thinking tags but no content - penalize
                rewards.append(-20.0)

        return rewards

    @staticmethod
    def ground_truth_test_reward(prompts, completions, **kwargs) -> List[float]:
        """Reward based on ground truth test cases from OSS subset with concurrent execution."""
        return asyncio.run(
            RewardFunctions._ground_truth_test_reward_async(
                prompts, completions, **kwargs
            )
        )

    @staticmethod
    async def _ground_truth_test_reward_async(
        prompts, completions, **kwargs
    ) -> List[float]:
        """Async implementation of ground_truth_test_reward for concurrent execution."""

        # Get test_cases from dataset items passed via kwargs
        all_test_cases = kwargs.get("test_cases", [])
        num_generations = kwargs.get("num_generations", 4)  # Default from GRPO config

        async def process_completion(i: int, completion) -> float:
            """Process a single completion and return its reward."""
            text = remove_thinking_tags(completion[0]["content"])
            code = extract_python_code(text)
            if not code:
                return -10.0  # Strong penalty for no code

            # Map completion index to correct test_cases
            example_test_cases = []
            if i < len(all_test_cases) and all_test_cases[i]:
                example_test_cases = all_test_cases[i]
            elif len(all_test_cases) > 0:
                # Fallback: map to original prompt's test cases if repeated structure is wrong
                prompt_index = i // num_generations
                if prompt_index < len(all_test_cases) and all_test_cases[prompt_index]:
                    example_test_cases = all_test_cases[prompt_index]

            if not example_test_cases:
                return -5.0  # Penalty for no test cases

            # Test against ground truth test cases
            try:
                # Create proper test script with individual test case validation
                common_imports = """
import math
import re
import sys
import os
import random
import itertools
import functools
import collections
from typing import List, Dict, Set, Tuple, Optional, Any, Union
from collections import defaultdict, Counter, deque
"""
                test_script = f"""
{common_imports}

{code}

# Ground truth test cases validation
passed = 0
total = {len(example_test_cases)}
failed_tests = []

"""
                # Add each test case with proper error handling
                for j, test_case in enumerate(example_test_cases):
                    # Clean the test case - remove extra whitespace and ensure it's a valid assertion
                    test_case = test_case.strip()
                    test_num = j + 1
                    test_script += f"""
try:
    {test_case}
    passed += 1
    print("Test {test_num} PASSED")
except Exception as e:
    failed_tests.append("Test {test_num} FAILED: " + str(e))
    print("Test {test_num} FAILED: " + str(e))
"""

                test_script += f"""
success_rate = passed / total if total > 0 else 0.0
print("PASSED:" + str(passed))
print("TOTAL:" + str(total))
print("SUCCESS_RATE:" + str(success_rate))

if failed_tests:
    print("FAILED_TESTS:")
    for failed in failed_tests[:3]:  # Show first 3 failures
        print("  " + failed)
"""

                # Execute code remotely (now async)
                results = await execute_code_remotely(test_script)
                print(f"Test script execution results: {results}")

                if results.get("executed", False):
                    # Parse results from output
                    passed = 0
                    total = len(example_test_cases)

                    for line in results["output"].split("\n"):
                        if line.startswith("PASSED:"):
                            try:
                                passed = int(line.split(":")[1].strip())
                            except (ValueError, IndexError):
                                pass
                        elif line.startswith("TOTAL:"):
                            try:
                                total = int(line.split(":")[1].strip())
                            except (ValueError, IndexError):
                                pass

                    success_rate = passed / total if total > 0 else 0.0
                    print(f"Success rate: {success_rate:.2f}")
                    print(f"Passed: {passed}, Total: {total}")

                    # Improved reward based on success rate with better granularity
                    if success_rate == 1.0:
                        reward = 20.0  # Perfect score
                    elif success_rate >= 0.8:
                        reward = 15.0  # Very good
                    elif success_rate >= 0.6:
                        reward = 10.0  # Good
                    elif success_rate >= 0.4:
                        reward = 5.0  # Fair
                    elif success_rate >= 0.2:
                        reward = 2.0  # Poor but some progress
                    elif success_rate > 0.0:
                        reward = -2.0  # Very poor but at least some test passed
                    else:
                        reward = -8.0  # Complete failure - no tests passed

                    return reward
                else:
                    # Execution failed - check if it's a syntax error or runtime error
                    error_str = results.get("error", "")
                    if "SyntaxError" in error_str:
                        return -15.0  # Syntax error penalty
                    elif results.get("timeout", False):
                        return -12.0  # Timeout penalty
                    else:
                        return -10.0  # General execution failure

            except Exception as e:
                print(f"Error in testing framework: {e}")
                return -10.0  # Error in testing framework itself

        # Execute all completions concurrently
        tasks = [
            process_completion(i, completion)
            for i, completion in enumerate(completions)
        ]

        rewards = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions that occurred
        final_rewards = []
        for i, reward in enumerate(rewards):
            if isinstance(reward, Exception):
                print(f"Exception in completion {i}: {reward}")
                final_rewards.append(-10.0)  # Default penalty for exceptions
            else:
                final_rewards.append(reward)

        return final_rewards


# ============================================================================
# Dataset Preparation
# ============================================================================


def prepare_dataset(
    dataset_name: str,
    num_samples: Optional[int] = None,
    cache_dir: Optional[str] = None,
) -> Dataset:
    """Load and prepare the AceCode OSS subset for GRPO training with deep thinking."""
    # Only load on main process in distributed setting
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if local_rank == 0:
        print(f"Loading dataset: {dataset_name}")

    dataset = load_dataset(dataset_name, split="train", cache_dir=cache_dir)

    # Filter to OSS subset only
    def is_oss_with_tests(example):
        return (
            example.get("source") == "oss"
            and example.get("test_cases")
            and isinstance(example.get("test_cases"), list)
            and len(example.get("test_cases", [])) > 0
        )

    dataset = dataset.filter(is_oss_with_tests)

    if local_rank == 0:
        print(f"Filtered to OSS subset with test cases: {len(dataset)} examples")

    if num_samples:
        dataset = dataset.select(range(min(num_samples, len(dataset))))
        if local_rank == 0:
            print(f"Selected {len(dataset)} samples")

    def format_example(example):
        # Get question and test cases
        question = example.get("question", example.get("prompt", ""))
        test_cases = example.get("test_cases", [])

        # Use deep thinking system prompt
        system_prompt = get_system_prompt()

        return {
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            "task_id": example.get("id", ""),
            "test_cases": test_cases,  # Store test cases for reward function
            "source": example.get("source"),
            "difficulty": example.get("difficulty", "unknown"),
        }

    dataset = dataset.map(format_example)

    if local_rank == 0:
        print(f"Dataset prepared. Final samples: {len(dataset)}")

        # Show some stats
        difficulties = [ex.get("difficulty", "unknown") for ex in dataset]
        from collections import Counter

        difficulty_counts = Counter(difficulties)
        print(f"Difficulty distribution: {dict(difficulty_counts)}")

    return dataset


# ============================================================================
# Training Function
# ============================================================================


def get_lora_config():
    """Get LoRA configuration optimized for Llama models."""
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=64,
        lora_alpha=256,
        lora_dropout=0.05,
        bias="none",
        use_rslora=True,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )


def train_model(args):
    """Main training function with staged reward approach."""

    # Set device and print system info
    if torch.cuda.is_available() and not args.cpu:
        device = "cuda"
        print(f"Using {torch.cuda.device_count()} GPU(s)")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            print(
                f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB"
            )
    else:
        device = "cpu"
        print("Using CPU")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset
    train_dataset = prepare_dataset(args.dataset_name, args.num_samples, args.cache_dir)

    # GRPOTrainer will handle model and tokenizer initialization

    print(f"\n🎯 DEEP THINKING APPROACH WITH OSS TEST CASES:")
    print(f"  📊 Clean 2-reward system:")
    print(
        f"     - Binary thinking reward (±20): Simple presence/absence of thinking content"
    )
    print(
        f"     - Ground truth test reward (+20 to -10): Based on OSS test case success rate"
    )
    print(f"     - OSS subset only: Real test cases from open source problems")
    print(f"     - Deep thinking system prompt: Always encourages detailed reasoning")
    print(f"     - Quality emerges naturally: No complex quality analysis needed")
    print(
        f"  - Effective batch size: {args.batch_size * args.gradient_accumulation_steps * torch.cuda.device_count()}"
    )

    # Training arguments optimized for multi-GPU full fine-tuning
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        use_vllm=args.use_vllm,
        # Training hyperparameters
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_epochs,
        # GRPO specific
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        # Optimization
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler,
        optim=args.optimizer,
        adam_beta1=0.9,
        adam_beta2=0.999,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        # Logging and saving
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        # Performance optimizations
        bf16=torch.cuda.is_available() and not args.no_bf16,
        tf32=torch.cuda.is_available() and not args.no_tf32,
        dataloader_num_workers=args.num_workers,
        gradient_checkpointing=args.gradient_checkpointing,
        # Multi-GPU settings
        ddp_find_unused_parameters=False,
        # Disable wandb
        report_to="none" if args.no_wandb else "wandb",
        run_name=(
            args.run_name
            if args.run_name
            else f"grpo_thinking_{args.model_name.split('/')[-1]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        ),
    )

    # Initialize LoRA configuration
    lora_config = get_lora_config()
    print(f"\nLoRA Configuration:")
    print(f"  - Rank (r): {lora_config.r}")
    print(f"  - Alpha: {lora_config.lora_alpha}")
    print(f"  - Dropout: {lora_config.lora_dropout}")
    print(f"  - Target modules: {lora_config.target_modules}")
    print(f"  - Use RSLoRA: {lora_config.use_rslora}")

    # Set up all reward functions from the start
    reward_functions = []

    # Core reward functions: Deep thinking + Ground truth testing
    # 1. Binary thinking reward (±20) - Simple thinking presence check
    if not args.disable_thinking_reward:
        reward_functions.append(RewardFunctions.thinking_reward)

    # 2. Ground truth test reward (+20 to -10) - OSS test case success
    if not args.disable_ground_truth_reward:
        # Use the ground truth test reward function directly - test_cases come from dataset items via kwargs
        reward_functions.append(RewardFunctions.ground_truth_test_reward)

    print(f"🎯 Using {len(reward_functions)} core reward functions for clean training")

    # Initialize trainer with LoRA and all reward functions from start
    trainer = GRPOTrainer(
        model=args.model_name,
        reward_funcs=reward_functions,
        args=training_args,
        train_dataset=train_dataset,
        peft_config=lora_config,
    )

    print("\nStarting GRPO training with deep thinking (OSS ground truth tests)...")
    print("=" * 50)

    try:
        # Train
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

        print("\n✅ Training completed successfully!")

        # Save final model
        final_model_path = os.path.join(args.output_dir, "final_model")
        trainer.save_model(final_model_path)
        print(f"Model saved to: {final_model_path}")

        # Save training metrics
        if hasattr(trainer.state, "log_history"):
            metrics_path = os.path.join(args.output_dir, "training_metrics.json")
            with open(metrics_path, "w") as f:
                json.dump(trainer.state.log_history, f, indent=2)
            print(f"Training metrics saved to: {metrics_path}")

    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        checkpoint_path = os.path.join(args.output_dir, "interrupted_checkpoint")
        trainer.save_model(checkpoint_path)
        print(f"Checkpoint saved to: {checkpoint_path}")

    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        traceback.print_exc()

    finally:
        # Clean up GPU memory
        if torch.cuda.is_available():
            del trainer
            torch.cuda.empty_cache()
            gc.collect()
            print("\n🧹 GPU memory cleared")


# ============================================================================
# Main Entry Point
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Train Llama models for Python code generation using GRPO with Deep Thinking (OSS Ground Truth Tests)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model arguments
    parser.add_argument(
        "--model-name",
        type=str,
        default="meta-llama/Llama-3.2-3B-Instruct",
        help="Model name or path from HuggingFace",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="TIGER-Lab/AceCode-87K",
        help="Dataset name from HuggingFace",
    )

    # Training arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs_3B-simple",
        help="Output directory for model and logs",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5000,
        help="Number of training samples (None for full dataset)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Training batch size per device"
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--num-epochs", type=int, default=1, help="Number of training epochs"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-5, help="Learning rate"
    )
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument(
        "--max-grad-norm", type=float, default=1.0, help="Maximum gradient norm"
    )

    # GRPO arguments
    parser.add_argument(
        "--num-generations", type=int, default=4, help="Number of generations for GRPO"
    )
    parser.add_argument(
        "--max-prompt-length", type=int, default=8196, help="Maximum prompt length"
    )
    parser.add_argument(
        "--max-completion-length",
        type=int,
        default=8192,
        help="Maximum completion length (increased for thinking)",
    )

    # Reward function toggles
    parser.add_argument(
        "--disable-thinking-reward",
        action="store_true",
        help="Disable deep thinking reward",
    )
    parser.add_argument(
        "--disable-ground-truth-reward",
        action="store_true",
        help="Disable ground truth test case reward",
    )

    # Optimization arguments
    parser.add_argument(
        "--optimizer", type=str, default="adamw_torch", help="Optimizer to use"
    )
    parser.add_argument(
        "--lr-scheduler", type=str, default="cosine", help="Learning rate scheduler"
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        default=True,
        help="Enable gradient checkpointing for memory efficiency",
    )
    parser.add_argument(
        "--no-gradient-checkpointing",
        dest="gradient_checkpointing",
        action="store_false",
        help="Disable gradient checkpointing",
    )

    # Logging arguments
    parser.add_argument(
        "--logging-steps", type=int, default=10, help="Logging frequency"
    )
    parser.add_argument(
        "--save-steps", type=int, default=1000, help="Model save frequency"
    )
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=5,
        help="Maximum number of checkpoints to keep",
    )
    parser.add_argument(
        "--run-name", type=str, default=None, help="Name for this training run"
    )
    parser.add_argument(
        "--no-wandb", action="store_true", help="Disable Weights & Biases logging"
    )

    # System arguments
    parser.add_argument("--cpu", action="store_true", help="Force CPU training")
    parser.add_argument(
        "--no-bf16", action="store_true", help="Disable bfloat16 training"
    )
    parser.add_argument(
        "--no-tf32", action="store_true", help="Disable TF32 on Ampere GPUs"
    )
    parser.add_argument(
        "--num-workers", type=int, default=64, help="Number of dataloader workers"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Cache directory for models and datasets",
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")
    parser.add_argument(
        "--use-vllm", action="store_true", help="use vllm for inference"
    )
    # Training control
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Resume training from checkpoint",
    )

    # Testing
    parser.add_argument(
        "--test-only", action="store_true", help="Only test the model, don't train"
    )
    parser.add_argument(
        "--test-model-path", type=str, default=None, help="Path to model for testing"
    )

    args = parser.parse_args()

    train_model(args)


if __name__ == "__main__":
    main()
