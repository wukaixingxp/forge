# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import re
import traceback
from typing import Any, Coroutine, Optional

from forge.interfaces import Reward

# Configure logger with process ID for multi-process debugging
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [PID:%(process)d] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


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
    """Reward class for evaluating code against ground truth test cases using sandboxed execution.

    Uses a dense reward structure (0.0 to 1.0) with progressive milestones:
    - 0.05: Code extracted from response
    - 0.10: Code is syntactically valid
    - 0.15: Code executes without immediate crash
    - 0.70: Scaled by test success rate (0-100%)

    Total possible reward: 1.0
    """

    def __init__(self, coder_actor=None):
        self.coder_actor = coder_actor

    def evaluate_sync(
        self, prompt: str, response: str, test_cases: list[str]
    ) -> Coroutine[Any, Any, float]:
        """Evaluation method - returns a coroutine that evaluates code against test cases."""

        # This function needs to handle the async coder_actor.execute() call
        # Since we're called from an async context, we'll create an async inner function
        # and return it as a coroutine
        async def _async_evaluate():
            reward = 0.0

            if not self.coder_actor:
                logger.info("No coder_actor available - Reward: 0.0")
                return 0.0

            if not test_cases:
                logger.info("No test cases provided - Reward: 0.0")
                return 0.0

            raw_content = response
            logger.info("=" * 80)
            logger.info("RAW CONTENT FROM MODEL:")
            logger.info("-" * 80)
            logger.info(raw_content)
            logger.info("-" * 80)

            text = remove_thinking_tags(raw_content)
            code = extract_python_code(text)

            if not code:
                logger.info("No code extracted - Reward: 0.0")
                return 0.0

            # Milestone 1: Code extracted successfully
            reward += 0.05
            logger.info(f"✓ Code extracted - Reward: {reward:.3f}")

            # First, check if code is syntactically valid
            try:
                compile(code, "<string>", "exec")
                # Milestone 2: Code is syntactically valid
                reward += 0.10
                logger.info(f"✓ Syntax valid - Reward: {reward:.3f}")
            except SyntaxError as e:
                logger.info(f"✗ Syntax error: {e} - Final Reward: {reward:.3f}")
                return reward

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

                # Execute code using the coder actor - this is the async call
                output, error = await self.coder_actor.execute.call_one(
                    code=test_script
                )
                results_output = output + "\n" + error

                logger.info("=" * 80)
                logger.info("GroundTruthTestReward - RESULTS OUTPUT:")
                logger.info("-" * 80)
                logger.info(results_output)
                logger.info("-" * 80)

                # Check for timeout or immediate crash
                if "timeout" in error.lower():
                    logger.info(f"✗ Timeout - Final Reward: {reward:.3f}")
                    logger.info("=" * 80)
                    return reward

                # Check if code executed without immediate crash (got to test execution stage)
                if output and ("PASSED:" in output or "Test " in output):
                    # Milestone 3: Code executed without immediate crash
                    reward += 0.15
                    logger.info(f"✓ Code executed - Reward: {reward:.3f}")

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

                    # Milestone 4: Test success (scaled by success rate)
                    test_reward = 0.70 * success_rate
                    reward += test_reward

                    logger.info(
                        f"✓ Tests: {passed}/{total} ({success_rate:.1%}) - Test reward: +{test_reward:.3f}"
                    )
                    logger.info(f"Final Reward: {reward:.3f}")
                    logger.info("=" * 80)

                    return reward
                else:
                    # Code crashed before getting to tests
                    logger.info(
                        f"✗ Runtime crash before tests - Final Reward: {reward:.3f}"
                    )
                    if error:
                        logger.info(f"Error: {error[:200]}")
                    logger.info("=" * 80)
                    return reward

            except Exception as e:
                logger.error(
                    f"✗ Testing framework error: {e} - Final Reward: {reward:.3f}"
                )
                logger.error("Full traceback:")
                logger.error(traceback.format_exc())
                logger.info("=" * 80)
                return reward

        # Return the coroutine - it will be awaited by the caller
        return _async_evaluate()

    def __call__(
        self, prompt: str, response: str, test_cases: list[str] | None = None
    ) -> Coroutine[Any, Any, float]:
        """Call method - returns a coroutine that evaluates code against test cases."""
        if test_cases is None:
            test_cases = []
        return self.evaluate_sync(prompt, response, test_cases)
