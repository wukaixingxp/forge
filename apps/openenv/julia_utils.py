# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Julia task-specific utilities for OpenEnv training.
Contains prompt building, action creation, and reward evaluation functions.
"""

import re
from typing import Any, Dict

from forge.observability.metrics import record_metric, Reduce


def get_julia_system_prompt() -> str:
    """Get system prompt for Julia coding tasks."""
    return """You are a precise and pragmatic Julia programmer.

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
FORMAT YOUR RESPONSE AS:

```julia
function <function_name>(<argument_list>)
    <function_body>
end
```
""".strip()


def build_julia_prompt(sample: Dict[str, Any], tokenizer) -> str:
    """
    Build prompt for Julia code generation.

    Args:
        sample: Dataset sample with 'julia_prompt', 'julia_test', 'first_test_case', 'task_id'
        tokenizer: HuggingFace tokenizer for chat template

    Returns:
        Formatted prompt string ready for model generation
    """
    system_prompt = get_julia_system_prompt()
    request = sample.get("julia_prompt", "")

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


def build_julia_action(response: str, sample: Dict[str, Any]):
    """
    Build JuliaAction from model response and dataset sample.

    Args:
        response: Model's generated response
        sample: Dataset sample with 'julia_test' field

    Returns:
        JuliaAction instance with core code and test code
    """
    # Import AutoAction dynamically to avoid pickle issues
    from envs import AutoAction

    # Get JuliaAction class dynamically
    JuliaAction = AutoAction.from_env("julia")

    # Extract code from markdown if present
    code = extract_julia_code(response)

    # Get test code from dataset
    test_code = sample.get("target", "")

    return JuliaAction(
        core_code=code,
        test_code=test_code,
    )


def evaluate_julia_response(result, response: str, sample: Dict[str, Any]) -> float:
    """
    Evaluate Julia code execution result and return reward.

    Uses a dense reward structure:
    - 0.0: Code failed to execute or tests failed
    - reward > 0.0: Reward based on test success rate
    - 1.0: All tests passed

    Args:
        result: StepResult from environment execution
        response: Model's response (for logging)
        sample: Dataset sample (for logging)

    Returns:
        Reward score (0.0 to 1.0)
    """
    try:
        print("=" * 80)
        print("RAW RESPONSE FROM MODEL:")
        print("-" * 80)
        print(response)
        print("-" * 80)

        # Extract code for validation
        code = extract_julia_code(response)

        if not code:
            print("No Julia code extracted - Reward: 0.0")
            print("=" * 80)
            record_metric("reward/julia/no_code_extracted", 1, Reduce.SUM)
            return 0.0

        print("EXTRACTED JULIA CODE:")
        print("-" * 80)
        print(code)
        print("-" * 80)

        # Extract reward from result
        reward = result.reward if result.reward is not None else 0.0

        obs = result.observation
        passed = obs.tests_passed
        failed = obs.tests_failed
        total = passed + failed

        # Log execution details
        print("JuliaEnv Execution Result:")
        print(f"  Reward: {reward:.3f}")
        print(f"  Tests Passed: {passed}")
        print(f"  Tests Failed: {failed}")
        print(f"  Total Tests: {total}")
        print(f"  Exit Code: {obs.exit_code}")
        print(f"  Code Compiles: {obs.code_compiles}")

        if obs.stderr:
            print(f"  Stderr: {obs.stderr[:500]}")
            record_metric("reward/julia/has_errors", 1, Reduce.SUM)

        if obs.stdout:
            print(f"  Stdout (first 200 chars): {obs.stdout[:200]}")

        # Log metrics
        pass_rate = passed / total if total > 0 else 0.0
        record_metric("reward/julia/pass_rate", pass_rate, Reduce.MEAN)

        print(f"Final Reward: {reward:.3f}")
        print("=" * 80)

        return reward

    except Exception as e:
        print(f"✗ Error evaluating response: {e} - Reward: 0.0")
        print("=" * 80)
        record_metric("reward/julia/evaluation_errors", 1, Reduce.SUM)
        return 0.0


def extract_julia_code(response: str) -> str:
    """
    Extract Julia code from markdown code blocks.

    Args:
        response: Model's response text

    Returns:
        Extracted Julia code
    """
    # Remove markdown code blocks with regex (more robust)
    text = re.sub(r"^```julia\s*\n?", "", response, flags=re.IGNORECASE)
    text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


def transform_julia_sample(sample: Dict[str, Any], tokenizer) -> Dict[str, Any] | None:
    """
    Transform raw dataset sample into training format.

    Args:
        sample: Raw dataset sample
        tokenizer: HuggingFace tokenizer

    Returns:
        Transformed sample with 'request', 'target', 'task_id' or None if invalid
    """
    # Validate required fields
    if not sample.get("julia_test") or not sample.get("first_test_case"):
        # Debug: log why sample was rejected (only for first few)
        if not hasattr(transform_julia_sample, "_warned"):
            print(
                f"WARNING: Sample rejected - missing 'julia_test' or 'first_test_case' field. Sample keys: {list(sample.keys())}"
            )
            transform_julia_sample._warned = True
        return None

    # Build prompt
    formatted_request = build_julia_prompt(sample, tokenizer)

    return {
        "request": formatted_request,
        "target": sample.get("julia_test", ""),  # Full test code for reward function
        "task_id": sample.get("task_id", ""),
    }
