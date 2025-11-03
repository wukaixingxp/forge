# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import re
from typing import Any, Coroutine

import requests


def remove_ticks(text: str) -> str:
    """Remove markdown code blocks from Julia code."""
    text = re.sub(r"^```julia\s*\n?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n?```\s*$", "", text)
    return text


class JuliaEnvReward:
    """Reward class for evaluating Julia code against tests using JuliaEnv.

    Uses a dense reward structure:
    - 0.0: Code failed to execute or tests failed
    - reward > 0.0: Reward from JuliaEnv based on test success rate
    - 1.0: All tests passed

    The reward is directly obtained from the JuliaEnv environment which
    executes the Julia code and runs the tests in a sandboxed container.
    """

    def __init__(
        self, base_url: str = "http://localhost:8000", request_timeout_s: float = 60.0
    ):
        """Initialize JuliaEnvReward.

        Args:
            base_url: Base URL of the JuliaEnv server
            request_timeout_s: Timeout for requests to JuliaEnv server
        """
        self.base_url = base_url
        self.request_timeout_s = request_timeout_s

    def evaluate_sync(
        self, prompt: str, response: str, test_code: str
    ) -> Coroutine[Any, Any, float]:
        """Evaluation method - returns a coroutine that evaluates Julia code against tests."""

        async def _async_evaluate():
            reward = 0.0

            print("=" * 80)
            print("RAW RESPONSE FROM MODEL:")
            print("-" * 80)
            print(response)
            print("-" * 80)

            # Extract Julia code from response (remove markdown code blocks)
            core_code = remove_ticks(response)

            if not core_code:
                print("No Julia code extracted - Reward: 0.0")
                print("=" * 80)
                return 0.0

            print("EXTRACTED JULIA CODE:")
            print("-" * 80)
            print(core_code)
            print("-" * 80)

            try:
                # Reset the environment
                reset_response = requests.post(
                    f"{self.base_url}/reset", timeout=self.request_timeout_s
                )
                reset_response.raise_for_status()

                # Execute the code with tests
                step_response = requests.post(
                    f"{self.base_url}/step",
                    json={
                        "core_code": core_code,
                        "test_code": test_code,
                    },
                    timeout=self.request_timeout_s,
                )
                step_response.raise_for_status()

                result = step_response.json()

                # Extract reward from result
                reward = result.get("reward", 0.0)
                if reward is None:
                    reward = 0.0

                # Log execution details
                print(f"JuliaEnv Execution Result:")
                print(f"  Reward: {reward}")
                if "observation" in result:
                    obs = result["observation"]
                    print(f"  Tests Passed: {obs.get('tests_passed', 'N/A')}")
                    print(f"  Tests Failed: {obs.get('tests_failed', 'N/A')}")
                    if obs.get("error_message"):
                        print(f"  Error: {obs.get('error_message')[:200]}")

                print(f"Final Reward: {reward:.3f}")
                print("=" * 80)

                return reward

            except requests.exceptions.Timeout:
                print(f"✗ JuliaEnv request timeout - Reward: 0.0")
                print("=" * 80)
                return 0.0
            except requests.exceptions.RequestException as e:
                print(f"✗ JuliaEnv request error: {e} - Reward: 0.0")
                print("=" * 80)
                return 0.0
            except Exception as e:
                print(f"✗ Unexpected error: {e} - Reward: 0.0")
                print("=" * 80)
                return 0.0

        # Return the coroutine - it will be awaited by the caller
        return _async_evaluate()

    def __call__(
        self, prompt: str, response: str, test_code: str | None = None
    ) -> Coroutine[Any, Any, float]:
        """Call method - returns a coroutine that evaluates Julia code against tests."""
        if test_code is None:
            test_code = ""
        return self.evaluate_sync(prompt, response, test_code)
