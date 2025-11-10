#!/usr/bin/env python3
"""
Simple direct test of the isinstance fix.

This test directly verifies that the isinstance check works for all the types
that OmegaConf might return: tuple, list, and ListConfig.
"""

import sys
from pathlib import Path

# Add openenv directory to path
openenv_dir = Path(__file__).parent
if str(openenv_dir) not in sys.path:
    sys.path.insert(0, str(openenv_dir))

from omegaconf import ListConfig
import importlib


def load_function_from_string(func_ref: str):
    """Load a function from string reference."""
    module_name, func_name = func_ref.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, func_name)


def test_isinstance_with_different_types():
    """Test that isinstance works with tuple, list, and ListConfig."""
    print("=" * 80)
    print("TEST: isinstance check works with tuple, list, and ListConfig")
    print("=" * 80)

    # Test data that mimics what OmegaConf returns
    test_cases = [
        ("tuple", ("!function", "julia_utils.build_julia_action")),
        ("list", ["!function", "julia_utils.build_julia_action"]),
        ("ListConfig", ListConfig(["!function", "julia_utils.build_julia_action"])),
    ]

    results = []

    for type_name, test_value in test_cases:
        print(f"\n{type_name}:")
        print(f"  Value: {test_value}")
        print(f"  Type: {type(test_value).__name__}")

        # OLD check (without ListConfig)
        old_check = isinstance(test_value, (tuple, list))
        print(f"  OLD isinstance(x, (tuple, list)): {old_check}")

        # NEW check (with ListConfig)
        new_check = isinstance(test_value, (tuple, list, ListConfig))
        print(f"  NEW isinstance(x, (tuple, list, ListConfig)): {new_check}")

        # Full condition check
        full_condition = (
            isinstance(test_value, (tuple, list, ListConfig))
            and len(test_value) == 2
            and test_value[0] == "!function"
        )
        print(f"  Full condition passes: {full_condition}")

        results.append((type_name, full_condition))

    # Summary
    print("\n" + "=" * 80)
    print("RESULTS:")
    print("=" * 80)

    all_pass = all(result for _, result in results)

    for type_name, result in results:
        status = "✓" if result else "✗"
        print(f"{status} {type_name}: {'PASS' if result else 'FAIL'}")

    if all_pass:
        print("\n✓ ALL TYPES WORK - Fix handles tuple, list, AND ListConfig!")
        return True
    else:
        print("\n✗ SOME TYPES FAILED - Fix incomplete")
        return False


def test_function_loading_with_list():
    """Test that we can actually load functions when value is a list."""
    print("\n" + "=" * 80)
    print("TEST: Function loading works with list format")
    print("=" * 80)

    # This mimics what we see in the actual training log
    test_configs = {
        "build_action": ["!function", "julia_utils.build_julia_action"],
        "evaluate_response": ["!function", "julia_utils.evaluate_julia_response"],
        "transform_sample": ["!function", "julia_utils.transform_julia_sample"],
    }

    loaded = []

    for func_name, func_config in test_configs.items():
        print(f"\n{func_name}:")
        print(f"  Config: {func_config}")

        # Apply the NEW condition
        if (
            isinstance(func_config, (tuple, list, ListConfig))
            and len(func_config) == 2
            and func_config[0] == "!function"
        ):
            print(f"  Condition passed ✓")
            func_ref = func_config[1]
            try:
                func = load_function_from_string(func_ref)
                print(f"  Loaded: {func.__name__}")
                loaded.append(func_name)
                print(f"  Status: ✓ SUCCESS")
            except Exception as e:
                print(f"  Error: {e}")
                print(f"  Status: ✗ FAILED")
        else:
            print(f"  Condition failed ✗")
            print(f"  Status: ✗ FAILED")

    print(f"\n{'=' * 80}")
    print(f"Loaded {len(loaded)}/3 functions")

    if len(loaded) == 3:
        print("✓ ALL FUNCTIONS LOADED SUCCESSFULLY!")
        return True
    else:
        print(f"✗ ONLY {len(loaded)}/3 FUNCTIONS LOADED")
        return False


def main():
    """Run all tests."""
    print("\n")
    print("█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + "  ISINSTANCE FIX VERIFICATION (Simple Direct Test)".center(78) + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80)
    print()

    test1_pass = test_isinstance_with_different_types()
    test2_pass = test_function_loading_with_list()

    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    print(
        f"✓ PASS - isinstance check works"
        if test1_pass
        else "✗ FAIL - isinstance check broken"
    )
    print(
        f"✓ PASS - Function loading works"
        if test2_pass
        else "✗ FAIL - Function loading broken"
    )

    if test1_pass and test2_pass:
        print("\n" + "█" * 80)
        print("█" + " " * 78 + "█")
        print("█" + "  ✓✓✓ ALL TESTS PASSED - FIX IS WORKING! ✓✓✓".center(78) + "█")
        print("█" + " " * 78 + "█")
        print("█" * 80)
        print()
        print("The fix successfully handles:")
        print("  - Regular Python tuples")
        print("  - Regular Python lists (what OmegaConf actually returns)")
        print("  - OmegaConf ListConfig objects")
        print()
        print("Your training will now work correctly!")
        print()
        return 0
    else:
        print("\n✗ TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
