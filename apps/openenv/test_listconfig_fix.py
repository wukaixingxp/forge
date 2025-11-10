#!/usr/bin/env python3
"""
Test script to verify the fix for OmegaConf ListConfig issue.

This test proves that:
1. OmegaConf loads !function tags as ListConfig objects
2. The OLD code (checking only tuple/list) FAILS to detect them
3. The NEW code (checking tuple/list/ListConfig) SUCCEEDS
4. Functions can be successfully loaded using the fix
"""

import sys
from pathlib import Path

# Add openenv directory to path
openenv_dir = Path(__file__).parent
if str(openenv_dir) not in sys.path:
    sys.path.insert(0, str(openenv_dir))

from omegaconf import OmegaConf, ListConfig, DictConfig
from omegaconf import _utils as omegaconf_utils
import importlib
import yaml


# Register the !function tag constructor (same as in main.py)
def function_constructor(loader, node):
    """YAML constructor for !function tag - converts to list."""
    value = loader.construct_scalar(node)
    # Return as list: ['!function', 'module.function_name']
    return ["!function", value]


# Register with OmegaConf's YAML loader
yaml.add_constructor(
    "!function", function_constructor, Loader=omegaconf_utils.get_yaml_loader()
)


def load_function_from_string(func_ref: str):
    """Load a function from string reference like 'julia_utils.build_julia_action'."""
    module_name, func_name = func_ref.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, func_name)


def test_omegaconf_loads_as_listconfig():
    """Test 1: Verify that OmegaConf loads !function tags as ListConfig."""
    print("=" * 80)
    print("TEST 1: Verify OmegaConf loads !function tags as ListConfig")
    print("=" * 80)

    config_path = openenv_dir / "llama3_8b_julia.yaml"
    cfg = OmegaConf.load(config_path)
    task_config = cfg.task

    print(f"\nLoaded config from: {config_path}")
    print(f"task_config.build_action = {task_config.build_action}")
    print(f"Type: {type(task_config.build_action).__name__}")

    # Verify it's a ListConfig
    is_listconfig = isinstance(task_config.build_action, ListConfig)
    print(f"\nIs ListConfig? {is_listconfig}")

    if is_listconfig:
        print("✓ TEST 1 PASSED: OmegaConf loads !function as ListConfig")
        return True
    else:
        print(
            "✗ TEST 1 FAILED: Expected ListConfig, got", type(task_config.build_action)
        )
        return False


def test_old_code_fails():
    """Test 2: Verify that OLD code (without ListConfig check) FAILS."""
    print("\n" + "=" * 80)
    print("TEST 2: Verify OLD code (tuple/list check only) FAILS")
    print("=" * 80)

    config_path = openenv_dir / "llama3_8b_julia.yaml"
    cfg = OmegaConf.load(config_path)
    task_config = cfg.task

    # OLD CODE - what was originally in main.py
    print("\nTesting OLD condition (checking only tuple and list):")
    print("  isinstance(task_config.build_action, (tuple, list))")

    old_condition = (
        isinstance(task_config.build_action, (tuple, list))
        and len(task_config.build_action) == 2
        and task_config.build_action[0] == "!function"
    )

    print(f"\nResult: {old_condition}")

    if not old_condition:
        print("✓ TEST 2 PASSED: OLD code correctly FAILS (as expected)")
        return True
    else:
        print("✗ TEST 2 FAILED: OLD code unexpectedly passed")
        return False


def test_new_code_passes():
    """Test 3: Verify that NEW code (with ListConfig check) PASSES."""
    print("\n" + "=" * 80)
    print("TEST 3: Verify NEW code (tuple/list/ListConfig check) PASSES")
    print("=" * 80)

    config_path = openenv_dir / "llama3_8b_julia.yaml"
    cfg = OmegaConf.load(config_path)
    task_config = cfg.task

    # NEW CODE - what we fixed in main.py
    print("\nTesting NEW condition (checking tuple, list, AND ListConfig):")
    print("  isinstance(task_config.build_action, (tuple, list, ListConfig))")

    new_condition = (
        isinstance(task_config.build_action, (tuple, list, ListConfig))
        and len(task_config.build_action) == 2
        and task_config.build_action[0] == "!function"
    )

    print(f"\nResult: {new_condition}")

    if new_condition:
        print("✓ TEST 3 PASSED: NEW code correctly detects ListConfig")
        return True
    else:
        print("✗ TEST 3 FAILED: NEW code should have passed")
        return False


def test_function_loading():
    """Test 4: Verify that functions can actually be loaded."""
    print("\n" + "=" * 80)
    print("TEST 4: Verify functions can be loaded using the fix")
    print("=" * 80)

    config_path = openenv_dir / "llama3_8b_julia.yaml"
    cfg = OmegaConf.load(config_path)
    task_config = cfg.task

    functions_loaded = []

    # Test build_action
    print("\n1. Loading build_action function...")
    if (
        isinstance(task_config.build_action, (tuple, list, ListConfig))
        and len(task_config.build_action) == 2
        and task_config.build_action[0] == "!function"
    ):
        func_ref = task_config.build_action[1]
        print(f"   Function reference: {func_ref}")
        try:
            func = load_function_from_string(func_ref)
            print(f"   Loaded function: {func}")
            print(f"   Function name: {func.__name__}")
            functions_loaded.append("build_action")
            print("   ✓ build_action loaded successfully")
        except Exception as e:
            print(f"   ✗ Failed to load: {e}")
    else:
        print("   ✗ Condition failed - cannot load function")

    # Test evaluate_response
    print("\n2. Loading evaluate_response function...")
    if (
        isinstance(task_config.evaluate_response, (tuple, list, ListConfig))
        and len(task_config.evaluate_response) == 2
        and task_config.evaluate_response[0] == "!function"
    ):
        func_ref = task_config.evaluate_response[1]
        print(f"   Function reference: {func_ref}")
        try:
            func = load_function_from_string(func_ref)
            print(f"   Loaded function: {func}")
            print(f"   Function name: {func.__name__}")
            functions_loaded.append("evaluate_response")
            print("   ✓ evaluate_response loaded successfully")
        except Exception as e:
            print(f"   ✗ Failed to load: {e}")
    else:
        print("   ✗ Condition failed - cannot load function")

    # Test transform_sample
    print("\n3. Loading transform_sample function...")
    if hasattr(task_config, "transform_sample"):
        if (
            isinstance(task_config.transform_sample, (tuple, list, ListConfig))
            and len(task_config.transform_sample) == 2
            and task_config.transform_sample[0] == "!function"
        ):
            func_ref = task_config.transform_sample[1]
            print(f"   Function reference: {func_ref}")
            try:
                func = load_function_from_string(func_ref)
                print(f"   Loaded function: {func}")
                print(f"   Function name: {func.__name__}")
                functions_loaded.append("transform_sample")
                print("   ✓ transform_sample loaded successfully")
            except Exception as e:
                print(f"   ✗ Failed to load: {e}")
        else:
            print("   ✗ Condition failed - cannot load function")
    else:
        print("   ✗ transform_sample not found in config")

    print(f"\nFunctions successfully loaded: {len(functions_loaded)}/3")

    if len(functions_loaded) == 3:
        print("✓ TEST 4 PASSED: All functions loaded successfully")
        return True
    else:
        print(f"✗ TEST 4 FAILED: Only {len(functions_loaded)}/3 functions loaded")
        return False


def main():
    """Run all tests and report results."""
    print("\n")
    print("█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + "  OMEGACONF LISTCONFIG FIX VERIFICATION TEST SUITE".center(78) + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80)
    print()

    results = []

    # Run all tests
    results.append(
        ("Test 1: OmegaConf loads as ListConfig", test_omegaconf_loads_as_listconfig())
    )
    results.append(("Test 2: OLD code fails", test_old_code_fails()))
    results.append(("Test 3: NEW code passes", test_new_code_passes()))
    results.append(("Test 4: Functions load correctly", test_function_loading()))

    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8} - {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n" + "█" * 80)
        print("█" + " " * 78 + "█")
        print("█" + "  ✓ ALL TESTS PASSED - FIX IS WORKING CORRECTLY!".center(78) + "█")
        print("█" + " " * 78 + "█")
        print("█" * 80)
        print()
        return 0
    else:
        print("\n" + "█" * 80)
        print("█" + " " * 78 + "█")
        print(
            "█"
            + f"  ✗ {total - passed} TEST(S) FAILED - FIX NEEDS WORK!".center(78)
            + "█"
        )
        print("█" + " " * 78 + "█")
        print("█" * 80)
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())
