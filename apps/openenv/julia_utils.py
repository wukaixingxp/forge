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
from collections import Counter

from forge.observability.metrics import record_metric, Reduce


def is_gibberish_output(response: str, max_length: int = 5000) -> tuple[bool, str]:
    """
    Detect if model output is gibberish/degraded (vLLM cache corruption symptom).

    This prevents wasting compute on obviously corrupted outputs that occur when
    vLLM's KV cache becomes fragmented or corrupted during long training runs.

    Args:
        response: Model's generated text
        max_length: Maximum reasonable response length in characters

    Returns:
        Tuple of (is_gibberish: bool, reason: str)
    """
    # Check 1: Excessive length (vLLM degradation often produces 10k+ token garbage)
    if len(response) > max_length:
        return True, f"Response too long: {len(response)} chars (max: {max_length})"

    # Check 2: Too many non-ASCII characters (indicates random token sampling)
    non_ascii_count = sum(1 for c in response if ord(c) > 127)
    non_ascii_ratio = non_ascii_count / len(response) if response else 0
    if non_ascii_ratio > 0.3:  # More than 30% non-ASCII
        return True, f"Too many non-ASCII chars: {non_ascii_ratio:.1%}"

    # Check 3: Excessive repetition (same phrase repeated many times)
    # Check for sequences of 20+ chars repeated 3+ times
    if len(response) > 100:
        # Sample a few 20-char windows and count how often they appear
        sample_size = min(50, len(response) - 20)
        for i in range(0, sample_size, 10):
            if i + 20 <= len(response):
                window = response[i:i+20]
                count = response.count(window)
                if count >= 3:  # Same 20-char sequence appears 3+ times
                    return True, f"Excessive repetition detected: '{window[:20]}...' x{count}"

    # Check 4: No valid code markers at all (neither ```julia nor function keyword)
    has_code_markers = (
        '```julia' in response.lower() or
        'function ' in response or
        'end' in response
    )
    if not has_code_markers and len(response) > 100:
        return True, "No Julia code markers found in long response"

    # Check 5: Excessive special characters (corrupted outputs have many random symbols)
    special_chars = sum(1 for c in response if not c.isalnum() and not c.isspace() and c not in '.,;:!?()[]{}"\'-_=')
    special_ratio = special_chars / len(response) if response else 0
    if special_ratio > 0.4:  # More than 40% random symbols
        return True, f"Too many special chars: {special_ratio:.1%}"

    return False, ""


def get_julia_system_prompt() -> str:
    """Get system prompt for Julia coding tasks."""
    return """You are a precise and pragmatic Julia programmer.

Write a **single Julia function** that correctly solves the problem described below.

CRITICAL - Julia is NOT Python! Use correct Julia syntax:
- Use `lowercase()` NOT `tolower()`
- Use `uppercase()` NOT `upper()`
- Use `reverse()` NOT `rev()` or `reversed()`
- Use `parse(Int, x)` or `Int(x)` for type conversion, NOT `int(x)`
- Use `string()` for string conversion, NOT `str()`
- Use `filter()` NOT `subset()`
- Use `length()` NOT `len()`
- Use `push!()` to append to arrays, NOT `append()`
- String indexing: `str[i]` returns a Char, use `str[i:i]` for single-char String
- Arrays are 1-indexed, NOT 0-indexed
- Use `println()` NOT `print()` for line output
- Use `Dict()` NOT `dict()`
- Boolean operators: `&&` for AND, `||` for OR, `!` for NOT
- Check string contains: `occursin(needle, haystack)` NOT `in` or `contains(haystack, needle)`

Example - Convert string to uppercase and reverse:
```julia
function process_text(text::String)
    upper_text = uppercase(text)  # NOT upper()
    reversed_text = reverse(upper_text)  # NOT rev()
    return reversed_text
end
```

Example - Work with integers and arrays:
```julia
function sum_digits(n::Int)
    total = 0
    digits_arr = Int[]  # Empty array
    while n > 0
        digit = n % 10
        push!(digits_arr, digit)  # NOT append()
        total += digit
        n = div(n, 10)
    end
    return total
end
```

Rules:
- The code must be syntactically correct and runnable as is
- Use only the Julia standard library
- Do **not** wrap the code in a module or add a `main` function
- Do **not** include any test code in your response
- Do **not** hardcode specific test cases or outputs — the function must work for general inputs
- The **function name must exactly match** the one used in the provided tests
- Respond with **only the Julia function** and nothing else (no explanations, no comments, no extra text)
- Character literal should not contain multiple characters
- Take care of object types and mind that spaces matter in Julia

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
        # First check if output is gibberish (vLLM cache corruption)
        is_gibberish, reason = is_gibberish_output(response, max_length=5000)
        if is_gibberish:
            print("=" * 80)
            print("GIBBERISH OUTPUT DETECTED (SKIPPING EVALUATION)")
            print(f"Reason: {reason}")
            print(f"Response length: {len(response)} chars")
            print("First 500 chars of response:")
            print("-" * 80)
            print(response[:500])
            print("-" * 80)
            print("⚠ This suggests vLLM KV cache corruption - cache reset recommended")
            print("=" * 80)
            record_metric("reward/julia/gibberish_outputs", 1, Reduce.SUM)
            return 0.0

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

        # Validate for common Python-like syntax errors
        is_valid, validation_warnings = validate_julia_syntax(code)
        if not is_valid:
            print("SYNTAX VALIDATION WARNINGS:")
            for warning in validation_warnings:
                print(f"  {warning}")
            print("-" * 80)
            record_metric("reward/julia/syntax_warnings", len(validation_warnings), Reduce.SUM)

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


def validate_julia_syntax(code: str) -> tuple[bool, list[str]]:
    """
    Validate Julia code for common Python-like syntax errors.

    Args:
        code: Julia code string to validate

    Returns:
        Tuple of (is_valid, list of warning messages)
    """
    warnings = []

    # Common Python functions that don't exist in Julia
    python_functions = {
        r'\btolower\(': 'tolower() -> use lowercase()',
        r'\bupper\(': 'upper() -> use uppercase()',
        r'\brev\(': 'rev() -> use reverse()',
        r'\bint\(': 'int() -> use parse(Int, x) or Int(x)',
        r'\bstr\(': 'str() -> use string()',
        r'\blen\(': 'len() -> use length()',
        r'\bsubset\(': 'subset() -> use filter()',
        r'\bappend\(': 'append() -> use push!()',
        r'\bdict\(': 'dict() -> use Dict()',
        r'\breversed\(': 'reversed() -> use reverse()',
        r'\.append\(': '.append() -> use push!()',
        r'\.lower\(': '.lower() -> use lowercase()',
        r'\.upper\(': '.upper() -> use uppercase()',
    }

    for pattern, suggestion in python_functions.items():
        if re.search(pattern, code, re.IGNORECASE):
            warnings.append(f"⚠ Found Python-like syntax: {suggestion}")

    # Check for 0-indexing patterns (common Python mistake)
    if re.search(r'\[\s*0\s*\]', code):
        warnings.append("⚠ Found [0] indexing - Julia arrays are 1-indexed")

    # Check for incomplete function definitions
    if 'function' in code and not re.search(r'\bend\b', code):
        warnings.append("⚠ Function missing 'end' keyword")

    is_valid = len(warnings) == 0
    return is_valid, warnings


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
