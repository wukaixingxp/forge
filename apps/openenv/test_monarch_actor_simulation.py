# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Test to simulate the Monarch actor scenario where functions are pickled
and sent to remote actors that need to unpickle them.
"""

import pickle
import subprocess
import sys
from pathlib import Path
from textwrap import dedent


def test_remote_process_unpickling():
    """
    Simulate the exact scenario where:
    1. Main process loads functions from julia_utils
    2. Functions are pickled and sent to remote process
    3. Remote process unpickles them (this is where the error occurred)
    """
    print("Test: Remote process unpickling simulation")
    print("-" * 60)

    # Create a test script that simulates a remote actor
    remote_actor_script = Path(__file__).parent / "test_remote_actor_temp.py"

    script_content = dedent(
        '''
        """
        This script simulates a remote actor process that receives
        pickled functions and tries to unpickle them.
        """
        import sys
        import pickle
        from pathlib import Path

        # This is what happens when a remote actor imports the main module
        # The module-level code in main.py should add openenv to sys.path
        import main

        # Read pickled data from stdin
        pickled_data = sys.stdin.buffer.read()

        try:
            # This is where the error occurred - unpickling requires julia_utils to be importable
            func = pickle.loads(pickled_data)
            print(f"SUCCESS: Unpickled function: {func}")
            print(f"SUCCESS: Function name: {func.__name__}")
            print(f"SUCCESS: Function module: {func.__module__}")
            sys.exit(0)
        except ModuleNotFoundError as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    '''
    )

    remote_actor_script.write_text(script_content)

    try:
        # In the main process, load and pickle a function from julia_utils
        import julia_utils

        func = julia_utils.build_julia_action
        pickled_func = pickle.dumps(func)
        print(f"  Main process: Pickled function {func.__name__}")
        print(f"  Main process: Pickled data size: {len(pickled_func)} bytes")

        # Simulate sending to remote process by spawning a subprocess
        print("  Spawning remote actor process...")
        result = subprocess.run(
            [sys.executable, str(remote_actor_script)],
            input=pickled_func,
            capture_output=True,
            timeout=10,
        )

        print("\n  Remote process output:")
        if result.stdout:
            for line in result.stdout.decode().split("\n"):
                if line:
                    print(f"    {line}")

        if result.returncode == 0:
            print("\n✓ Test passed: Remote actor successfully unpickled function")
            return True
        else:
            print("\n✗ Test failed: Remote actor failed to unpickle function")
            if result.stderr:
                print("  Stderr:")
                for line in result.stderr.decode().split("\n"):
                    if line:
                        print(f"    {line}")
            return False

    except Exception as e:
        print(f"\n✗ Test failed with exception: {e}")
        import traceback

        traceback.print_exc()
        return False
    finally:
        # Clean up temp script
        if remote_actor_script.exists():
            remote_actor_script.unlink()


def test_multiple_functions():
    """Test pickling all three task-specific functions."""
    print("\nTest: Pickle all task-specific functions")
    print("-" * 60)

    try:
        import julia_utils

        functions = [
            julia_utils.build_julia_action,
            julia_utils.evaluate_julia_response,
            julia_utils.transform_julia_sample,
        ]

        for func in functions:
            pickled = pickle.dumps(func)
            unpickled = pickle.loads(pickled)
            assert unpickled is func, f"Function {func.__name__} not properly unpickled"
            print(f"  ✓ {func.__name__}: pickle/unpickle OK")

        print("\n✓ All task-specific functions can be pickled/unpickled")
        return True

    except Exception as e:
        print(f"\n✗ Failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Monarch Actor Simulation Tests")
    print("=" * 60)
    print()

    results = []
    results.append(test_multiple_functions())
    results.append(test_remote_process_unpickling())

    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("✓ All tests passed!")
        print("\nThe fix ensures that:")
        print("  1. julia_utils is added to sys.path at module level")
        print("  2. Remote actors can import main.py and get the path setup")
        print("  3. Functions from julia_utils can be unpickled in remote processes")
        sys.exit(0)
    else:
        print("✗ Some tests failed")
        sys.exit(1)
