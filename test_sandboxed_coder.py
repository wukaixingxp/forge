#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Test script to demonstrate SandboxedPythonCoder functionality.

This script runs various test cases to verify that the SandboxedPythonCoder
can execute Python code in a sandboxed environment and properly capture
stdout and stderr.
"""

import asyncio
import sys
from pathlib import Path

# Add the src directory to the path so we can import forge modules
sys.path.insert(0, str(Path(__file__).parent / "src"))

from forge.actors.coder import SandboxedPythonCoder


async def run_test_case(coder: SandboxedPythonCoder, name: str, code: str):
    """Run a single test case and display results."""
    print(f"\n{'='*80}")
    print(f"TEST CASE: {name}")
    print(f"{'='*80}")
    print(f"Code to execute:\n{code}\n")
    
    try:
        stdout, stderr = await coder.execute.call(code)
        print(f"✓ Test completed")
        print(f"\nCaptured stdout:\n{stdout if stdout else '(empty)'}")
        print(f"\nCaptured stderr:\n{stderr if stderr else '(empty)'}")
        return True
    except Exception as e:
        print(f"✗ Test failed with exception: {e}")
        return False


async def main():
    """Run all test cases."""
    print("Initializing SandboxedPythonCoder...")
    
    # Create the coder instance with a custom container name
    coder = SandboxedPythonCoder(
        docker_image="docker://python:3.10",
        sqsh_image_path="/tmp/python-test.sqsh",
        container_name="test_sandbox"
    )
    
    # Setup the container (import image and create container)
    print("Setting up container (this may take a while on first run)...")
    await coder.setup.call()
    print("✓ Container setup complete\n")
    
    # Test cases
    test_cases = [
        (
            "Simple Print Statement",
            """print("Hello from sandboxed Python!")"""
        ),
        (
            "Basic Arithmetic",
            """
result = 42 + 58
print(f"The answer is: {result}")
"""
        ),
        (
            "Multiple Outputs",
            """
for i in range(5):
    print(f"Iteration {i}")
"""
        ),
        (
            "Using Built-in Libraries",
            """
import math

result = math.sqrt(144)
print(f"Square root of 144 is: {result}")
print(f"Pi value: {math.pi:.4f}")
"""
        ),
        (
            "Error Handling - Division by Zero",
            """
try:
    result = 10 / 0
except ZeroDivisionError as e:
    print(f"Caught error: {e}")
"""
        ),
        (
            "Code with Syntax Error",
            """
print("This will cause an error"
# Missing closing parenthesis
"""
        ),
        (
            "Code with Runtime Error",
            """
x = [1, 2, 3]
print(x[10])  # IndexError
"""
        ),
        (
            "Working with Strings",
            """
text = "Hello, World!"
print(f"Original: {text}")
print(f"Upper: {text.upper()}")
print(f"Lower: {text.lower()}")
print(f"Length: {len(text)}")
"""
        ),
        (
            "List Comprehension",
            """
squares = [x**2 for x in range(10)]
print(f"Squares: {squares}")
"""
        ),
        (
            "Writing to stderr",
            """
import sys

print("This goes to stdout")
print("This goes to stderr", file=sys.stderr)
"""
        ),
    ]
    
    # Run all test cases
    results = []
    for name, code in test_cases:
        success = await run_test_case(coder, name, code)
        results.append((name, success))
    
    # Summary
    print(f"\n{'='*80}")
    print("TEST SUMMARY")
    print(f"{'='*80}")
    passed = sum(1 for _, success in results if success)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    print("\nDetailed Results:")
    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status}: {name}")
    
    # Test recreate functionality
    print(f"\n{'='*80}")
    print("Testing Container Recreate Functionality")
    print(f"{'='*80}")
    await coder.recreate.call()
    print("✓ Container recreated successfully")
    
    # Run a simple test after recreate
    stdout, stderr = await coder.execute.call('print("Container is working after recreate!")')
    print(f"Output after recreate: {stdout}")
    
    print(f"\n{'='*80}")
    print("All tests completed!")
    print(f"{'='*80}")


if __name__ == "__main__":
    asyncio.run(main())
