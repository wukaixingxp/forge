"""
Test to verify that julia_utils can be imported from remote actors.
This simulates the pickling/unpickling that happens with Monarch actors.
"""

import pickle
import sys
from pathlib import Path

# Add openenv to path (simulating what happens in main.py)
openenv_dir = Path(__file__).parent
if str(openenv_dir) not in sys.path:
    sys.path.insert(0, str(openenv_dir))


def test_import_julia_utils():
    """Test that julia_utils can be imported."""
    print("Test 1: Direct import of julia_utils")
    try:
        import julia_utils

        print("✓ Successfully imported julia_utils")
        print(f"  Module path: {julia_utils.__file__}")
        return True
    except ModuleNotFoundError as e:
        print(f"✗ Failed to import julia_utils: {e}")
        return False


def test_pickle_function():
    """Test that functions from julia_utils can be pickled and unpickled."""
    print("\nTest 2: Pickle and unpickle julia_utils functions")
    try:
        import julia_utils

        # Get a function from julia_utils
        func = julia_utils.build_julia_action
        print(f"  Original function: {func}")

        # Pickle the function
        pickled = pickle.dumps(func)
        print(f"  Pickled successfully, size: {len(pickled)} bytes")

        # Unpickle the function
        unpickled_func = pickle.loads(pickled)
        print(f"  Unpickled function: {unpickled_func}")

        # Verify it's the same function
        assert unpickled_func is func, "Unpickled function is not the same object"
        print("✓ Successfully pickled and unpickled julia_utils function")
        return True
    except Exception as e:
        print(f"✗ Failed to pickle/unpickle: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_simulate_remote_import():
    """Simulate what happens when a remote actor imports the main module."""
    print("\nTest 3: Simulate remote actor importing main module")
    try:
        # This simulates what happens when a remote actor process
        # imports the main module
        import main

        print("✓ Successfully imported main module")

        # Verify that julia_utils is now importable
        import julia_utils

        print("✓ julia_utils is importable after main module import")
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Testing julia_utils import and pickling")
    print("=" * 60)

    results = []
    results.append(test_import_julia_utils())
    results.append(test_pickle_function())
    results.append(test_simulate_remote_import())

    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("✓ All tests passed!")
        sys.exit(0)
    else:
        print("✗ Some tests failed")
        sys.exit(1)
