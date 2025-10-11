# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import re
import asyncio
from typing import Optional

from forge.interfaces import Reward


class MathReward(Reward):
    """Reward class for evaluating math correctness."""

    def __init__(self, tolerance: float = 1e-6, partial_credit: float = 0.1):
        self.tolerance = tolerance
        self.partial_credit = partial_credit

    def __call__(self, prompt: str, response: str, target: str) -> float:
        """Compute math correctness reward."""
        target_number = self._to_float(target)
        if target_number is None:
            return 0.0

        # Look for answer in <answer></answer> tags
        answer_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)

        if answer_match:
            model_answer = self._to_float(answer_match.group(1).strip())
            if (
                model_answer is not None
                and abs(target_number - model_answer) < self.tolerance
            ):
                return 1.0  # Correct answer

        # Check for partial credit: target number appears elsewhere in response
        response_without_answer_tags = re.sub(
            r"<answer>.*?</answer>", "", response, flags=re.DOTALL
        )
        # Convert to int if it's a whole number to avoid "117.0" vs "117" mismatch
        target_str = (
            str(int(target_number))
            if target_number.is_integer()
            else str(target_number)
        )
        if target_str in response_without_answer_tags:
            return self.partial_credit

        return 0.0  # No match

    def _to_float(self, text: str) -> float | None:
        """Convert text to float, return None if invalid."""
        try:
            # Remove common non-numeric characters like $, commas, etc.
            cleaned_text = re.sub(r"[$,\s]", "", text.strip())
            return float(cleaned_text)
        except (ValueError, AttributeError):
            return None


class ThinkingReward(Reward):
    """Reward class for evaluating use of <think> tags in reasoning."""

    def __init__(self, partial_reward: float = 0.2, full_reward: float = 1.0):
        self.partial_reward = partial_reward
        self.full_reward = full_reward
        self._THINK_BLOCK_RE = re.compile(
            r"<\s*think\s*>(.*?)<\s*/\s*think\s*>", re.IGNORECASE | re.DOTALL
        )
        self._THINK_TAG_ATTEMPT_RE = re.compile(r"<\s*/?\s*think\s*>", re.IGNORECASE)

    def __call__(self, prompt: str, response: str, target: str | None = None) -> float:
        """Compute thinking reward."""
        if not response:
            return 0.0

        matches = self._THINK_BLOCK_RE.findall(response)
        has_well_formed = any(len(re.sub(r"\s+", "", m)) >= 1 for m in matches)
        has_attempt = bool(self._THINK_TAG_ATTEMPT_RE.search(response)) or bool(matches)
        if has_well_formed:
            return self.full_reward
        elif has_attempt:
            return self.partial_reward
        return 0.0


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


class GroundTruthTestReward(Reward):
    """Reward class for evaluating code against ground truth test cases using sandboxed execution."""

    def __init__(self, coder_actor=None):
        self.coder_actor = coder_actor

    async def evaluate_async(self, prompt: str, response: str, test_cases: list[str]) -> float:
        """Async evaluation of code against test cases."""
        if not self.coder_actor:
            return -5.0  # Penalty for missing execution environment

        raw_content = response
        text = remove_thinking_tags(raw_content)
        code = extract_python_code(text)

        if not code:
            return -10.0  # Strong penalty for no code

        if not test_cases:
            return -5.0  # Penalty for no test cases

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
total = {len(test_cases)}
failed_tests = []

"""
            # Add each test case with proper error handling
            for j, test_case in enumerate(test_cases):
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

            # Execute code using the coder actor
            output, error = await self.coder_actor.execute(test_script)
            results_output = output + "\n" + error
              
            print("=" * 80)
            print("[DEBUG] GroundTruthTestReward - RESULTS OUTPUT:")
            print("-" * 80)
            print(results_output)
            print("-" * 80)

            if output and "PASSED:" in output:
                # Parse results from output
                passed = 0
                total = len(test_cases)

                for line in results_output.split("\n"):
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
                  
                print(f"[DEBUG] Test Results: passed={passed}/{total}, success_rate={success_rate:.2%}")

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
                  
                print(f"[DEBUG] Final Reward: {reward}")
                print("=" * 80)

                return reward
            else:
                # Execution failed - check if it's a syntax error or runtime error
                if "SyntaxError" in error:
                    reward = -15.0  # Syntax error penalty
                elif "timeout" in error.lower():
                    reward = -12.0  # Timeout penalty
                else:
                    reward = -10.0  # General execution failure
                  
                print(f"[DEBUG] Execution failed - Final Reward: {reward}")
                print("=" * 80)
                return reward

        except Exception as e:
            print(f"Error in testing framework: {e}")
            return -10.0  # Error in testing framework itself

    async def __call__(self, prompt: str, response: str, test_cases: list[str] | None = None) -> float:
        """Async call method to be used in async contexts."""
        if test_cases is None:
            test_cases = []
        return await self.evaluate_async(prompt, response, test_cases)
