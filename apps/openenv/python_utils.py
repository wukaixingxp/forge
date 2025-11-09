# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Python coding task-specific utilities for OpenEnv training.
Contains prompt building, action creation, and reward evaluation functions.
"""

import re
from typing import Dict, Any

from envs.coding_env import CodingAction
from forge.observability.metrics import record_metric, Reduce


def get_python_system_prompt() -> str:
    """Get system prompt for Python coding tasks."""
    return """You are an expert Python programmer.

Write a Python function that correctly solves the problem described below.

Rules:
- The code must be syntactically correct and runnable
- Use proper Python conventions and best practices
- Include necessary imports
- Do not include test code in your response
- Return only the Python code

FORMAT YOUR RESPONSE AS:

```python
def function_name(args):
    # implementation
    return result
```
""".strip()


def build_python_prompt(sample: Dict[str, Any], tokenizer) -> str:
    """
    Build prompt for Python code generation.

    Args:
        sample: Dataset sample with 'prompt' field (e.g., from HumanEval)
        tokenizer: HuggingFace tokenizer for chat template

    Returns:
        Formatted prompt string ready for model generation
    """
    system_prompt = get_python_system_prompt()
    request = sample.get("prompt", "")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": request},
    ]

    formatted_request = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    return formatted_request


def build_python_action(response: str, sample: Dict[str, Any]) -> CodingAction:
    """
    Build CodingAction from model response and dataset sample.

    Args:
        response: Model's generated response
        sample: Dataset sample with test information

    Returns:
        CodingAction instance with code
    """
    # Extract code from markdown if present
    code = extract_python_code(response)

    # Get test code if available
    test_code = sample.get("target", "")

    return CodingAction(
        code=code,
        test_code=test_code,
    )


def evaluate_python_response(result, response: str, sample: Dict[str, Any]) -> float:
    """
    Evaluate Python code execution result and return reward.

    Uses a simple reward structure:
    - 1.0: All tests passed (exit code 0)
    - 0.1: Runtime error (code is syntactically valid)
    - 0.0: Syntax error or other failure

    Args:
        result: StepResult from environment execution
        response: Model's response (for logging)
        sample: Dataset sample (for logging)

    Returns:
        Reward score (0.0, 0.1, or 1.0)
    """
    try:
        print("=" * 80)
        print("RAW RESPONSE FROM MODEL:")
        print("-" * 80)
        print(response)
        print("-" * 80)

        # Extract code for validation
        code = extract_python_code(response)

        if not code:
            print("No Python code extracted - Reward: 0.0")
            print("=" * 80)
            record_metric("reward/python/no_code_extracted", 1, Reduce.SUM)
            return 0.0

        print("EXTRACTED PYTHON CODE:")
        print("-" * 80)
        print(code)
        print("-" * 80)

        obs = result.observation

        # Simple binary reward based on exit code
        if obs.exit_code == 0:
            reward = 1.0
            print("✓ All tests passed!")
            record_metric("reward/python/success", 1, Reduce.SUM)
        elif "SyntaxError" in obs.stderr or "syntax error" in obs.stderr.lower():
            reward = 0.0
            print("✗ Syntax error")
            record_metric("reward/python/syntax_error", 1, Reduce.SUM)
        else:
            reward = 0.1
            print("✗ Runtime error (but syntactically valid)")
            record_metric("reward/python/runtime_error", 1, Reduce.SUM)

        # Log execution details
        print("CodingEnv Execution Result:")
        print(f"  Reward: {reward:.3f}")
        print(f"  Exit Code: {obs.exit_code}")

        if obs.stderr:
            print(f"  Stderr: {obs.stderr[:500]}")

        if obs.stdout:
            print(f"  Stdout (first 200 chars): {obs.stdout[:200]}")

        record_metric("reward/python/reward", reward, Reduce.MEAN)

        print(f"Final Reward: {reward:.3f}")
        print("=" * 80)

        return reward

    except Exception as e:
        print(f"✗ Error evaluating response: {e} - Reward: 0.0")
        print("=" * 80)
        record_metric("reward/python/evaluation_errors", 1, Reduce.SUM)
        return 0.0


def extract_python_code(response: str) -> str:
    """
    Extract Python code from markdown code blocks.

    Args:
        response: Model's response text

    Returns:
        Extracted Python code
    """
    # Try to find ```python code block
    pattern = r"```python\n(.*?)```"
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try generic code block
    pattern = r"```\n(.*?)```"
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()

    # No markdown block, return as-is
    return response.strip()


def transform_python_sample(sample: Dict[str, Any], tokenizer) -> Dict[str, Any] | None:
    """
    Transform raw dataset sample into training format.

    Args:
        sample: Raw dataset sample (e.g., from HumanEval)
        tokenizer: HuggingFace tokenizer

    Returns:
        Transformed sample with 'request', 'target', 'task_id' or None if invalid
    """
    # Validate required fields
    if not sample.get("prompt"):
        return None

    # Build prompt
    formatted_request = build_python_prompt(sample, tokenizer)

    return {
        "request": formatted_request,
        "target": sample.get("test", ""),  # Test code for reward function
        "task_id": sample.get("task_id", ""),
    }
