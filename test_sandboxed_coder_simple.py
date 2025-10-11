#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Simple test script to demonstrate SandboxedPythonCoder functionality.

This script bypasses the actor framework and directly tests the core
functionality of the SandboxedPythonCoder.
"""

import asyncio
import subprocess
import tempfile
from pathlib import Path


class SimpleSandboxedPythonCoder:
    """Simplified version for testing without the actor framework."""
    
    def __init__(
        self,
        docker_image: str = "docker://python:3.10",
        sqsh_image_path: str = "python-image.sqsh",
        container_name: str = "sandbox",
    ):
        self.docker_image = docker_image
        self.sqsh_image_path = sqsh_image_path
        self.container_name = container_name
        self._initialized = False
    
    def _maybe_create_image(self):
        """Ensure the enroot image exists, import it if necessary."""
        import os
        if not os.path.exists(self.sqsh_image_path):
            print(f"Image {self.sqsh_image_path} not found, importing from {self.docker_image}")
            result = subprocess.run(
                ["enroot", "import", "-o", self.sqsh_image_path, self.docker_image],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(f"Failed to import image: {result.stderr}")
            print(f"Successfully imported {self.docker_image} to {self.sqsh_image_path}")
        else:
            print(f"Using existing image: {self.sqsh_image_path}")
    
    def _recreate(self):
        """(Re)create a clean container instance from the base image."""
        # Remove any old container
        print(f"Removing old container {self.container_name}")
        subprocess.run(
            ["enroot", "remove", "-f", self.container_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        # Create new container from image
        result = subprocess.run(
            ["enroot", "create", "--name", self.container_name, self.sqsh_image_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to recreate container: {result.stderr}")
        self._initialized = True
        print("Successfully initialized container")
    
    def setup(self):
        """Setup the sandboxed environment."""
        print("Setting up sandboxed environment")
        self._maybe_create_image()
        self._recreate()
    
    def execute(self, code: str) -> tuple[str, str]:
        """Execute Python code inside the container."""
        print("=" * 80)
        print("[DEBUG] INPUT CODE:")
        print("-" * 80)
        print(code)
        print("-" * 80)
        
        if not self._initialized:
            raise RuntimeError("Container not initialized. Call setup() first.")
        
        # Write code to a temporary file that we can mount
        with tempfile.TemporaryDirectory() as tmpdir:
            code_path = Path(tmpdir) / "script.py"
            code_path.write_text(code)
            
            # Run the code inside the container, mounting tmpdir
            cmd = [
                "enroot",
                "start",
                "--mount",
                f"{tmpdir}:/work",
                self.container_name,
                "python3",
                "/work/script.py",
            ]
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            output = result.stdout
            error = result.stderr
            
            print("[DEBUG] EXECUTION OUTPUTS:")
            print("-" * 80)
            print(f"Return Code: {result.returncode}")
            print(f"\nSTDOUT:\n{output if output else '(empty)'}")
            print(f"\nSTDERR:\n{error if error else '(empty)'}")
            print("=" * 80)
            
            return output, error
    
    def recreate(self):
        """Recreate the container."""
        self._recreate()


def run_test_case(coder: SimpleSandboxedPythonCoder, name: str, code: str):
    """Run a single test case and display results."""
    print(f"\n{'='*80}")
    print(f"TEST CASE: {name}")
    print(f"{'='*80}")
    print(f"Code to execute:\n{code}\n")
    
    try:
        stdout, stderr = coder.execute(code)
        print(f"✓ Test completed")
        print(f"\nCaptured stdout:\n{stdout if stdout else '(empty)'}")
        print(f"\nCaptured stderr:\n{stderr if stderr else '(empty)'}")
        return True
    except Exception as e:
        print(f"✗ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all test cases."""
    print("Initializing SimpleSandboxedPythonCoder...")
    
    # Create the coder instance with a custom container name
    coder = SimpleSandboxedPythonCoder(
        docker_image="docker://python:3.10",
        sqsh_image_path="/tmp/python-test.sqsh",
        container_name="test_sandbox"
    )
    
    # Setup the container (import image and create container)
    print("Setting up container (this may take a while on first run)...")
    coder.setup()
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
        success = run_test_case(coder, name, code)
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
    coder.recreate()
    print("✓ Container recreated successfully")
    
    # Run a simple test after recreate
    stdout, stderr = coder.execute('print("Container is working after recreate!")')
    print(f"Output after recreate: {stdout}")
    
    print(f"\n{'='*80}")
    print("All tests completed!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
