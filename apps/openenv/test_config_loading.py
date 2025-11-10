#!/usr/bin/env python3
"""Test script to verify OmegaConf config loading and function references."""

import sys
from pathlib import Path

# Add openenv directory to path
openenv_dir = Path(__file__).parent
if str(openenv_dir) not in sys.path:
    sys.path.insert(0, str(openenv_dir))

from omegaconf import OmegaConf, ListConfig, DictConfig
import argparse


def test_config_loading_with_parse():
    """Test loading config the way forge.util.config.parse does it."""

    config_path = openenv_dir / "llama3_8b_julia.yaml"
    print(f"Loading config from: {config_path}")
    print("Using OmegaConf.load() which is what forge.util.config uses...")

    # This mimics what happens in forge.util.config.ForgeRecipeArgParser.parse_known_args
    # Line 271: config = OmegaConf.load(namespace.config)
    cfg = OmegaConf.load(config_path)

    print("\n" + "=" * 80)
    print("TASK CONFIG ANALYSIS")
    print("=" * 80)

    task_config = cfg.task

    # Check build_action
    print(f"\n1. build_action:")
    print(f"   Value: {task_config.build_action}")
    print(f"   Type: {type(task_config.build_action)}")
    print(f"   Type name: {type(task_config.build_action).__name__}")
    print(f"   Is tuple: {isinstance(task_config.build_action, tuple)}")
    print(f"   Is list: {isinstance(task_config.build_action, list)}")
    print(f"   Is tuple or list: {isinstance(task_config.build_action, (tuple, list))}")

    if hasattr(task_config.build_action, "__len__"):
        print(f"   Length: {len(task_config.build_action)}")
        if len(task_config.build_action) > 0:
            print(f"   First element: {task_config.build_action[0]}")
            print(f"   First element type: {type(task_config.build_action[0])}")
            if len(task_config.build_action) > 1:
                print(f"   Second element: {task_config.build_action[1]}")

    # Check evaluate_response
    print(f"\n2. evaluate_response:")
    print(f"   Value: {task_config.evaluate_response}")
    print(f"   Type: {type(task_config.evaluate_response)}")
    print(f"   Type name: {type(task_config.evaluate_response).__name__}")
    print(f"   Is tuple: {isinstance(task_config.evaluate_response, tuple)}")
    print(f"   Is list: {isinstance(task_config.evaluate_response, list)}")
    print(
        f"   Is tuple or list: {isinstance(task_config.evaluate_response, (tuple, list))}"
    )

    # Check transform_sample
    print(f"\n3. transform_sample:")
    if hasattr(task_config, "transform_sample"):
        print(f"   Value: {task_config.transform_sample}")
        print(f"   Type: {type(task_config.transform_sample)}")
        print(f"   Type name: {type(task_config.transform_sample).__name__}")
        print(f"   Is tuple: {isinstance(task_config.transform_sample, tuple)}")
        print(f"   Is list: {isinstance(task_config.transform_sample, list)}")
        print(
            f"   Is tuple or list: {isinstance(task_config.transform_sample, (tuple, list))}"
        )
    else:
        print("   NOT FOUND in task_config")

    # Test the actual condition used in main.py
    print("\n" + "=" * 80)
    print("CONDITION TESTS (as used in main.py)")
    print("=" * 80)

    print("\n1. build_action condition:")
    condition1 = (
        isinstance(task_config.build_action, (tuple, list))
        and len(task_config.build_action) == 2
        and task_config.build_action[0] == "!function"
    )
    print(f"   Result: {condition1}")
    if not condition1:
        print("   FAILED - Function will NOT be loaded!")
        # Debug each part
        print(
            f"   - isinstance check: {isinstance(task_config.build_action, (tuple, list))}"
        )
        if hasattr(task_config.build_action, "__len__"):
            print(f"   - length == 2: {len(task_config.build_action) == 2}")
            if len(task_config.build_action) > 0:
                print(
                    f"   - first == '!function': {task_config.build_action[0] == '!function'}"
                )
    else:
        print("   PASSED - Function will be loaded")

    print("\n2. evaluate_response condition:")
    condition2 = (
        isinstance(task_config.evaluate_response, (tuple, list))
        and len(task_config.evaluate_response) == 2
        and task_config.evaluate_response[0] == "!function"
    )
    print(f"   Result: {condition2}")
    if not condition2:
        print("   FAILED - Function will NOT be loaded!")
    else:
        print("   PASSED - Function will be loaded")

    print("\n3. transform_sample condition:")
    if hasattr(task_config, "transform_sample"):
        condition3 = (
            isinstance(task_config.transform_sample, (tuple, list))
            and len(task_config.transform_sample) == 2
            and task_config.transform_sample[0] == "!function"
        )
        print(f"   Result: {condition3}")
        if not condition3:
            print("   FAILED - Function will NOT be loaded!")
        else:
            print("   PASSED - Function will be loaded")
    else:
        print("   SKIPPED - attribute not found")

    # Try to access as OmegaConf ListConfig
    print("\n" + "=" * 80)
    print("OMEGACONF SPECIFIC CHECKS")
    print("=" * 80)

    from omegaconf import ListConfig, DictConfig

    print(
        f"\nIs build_action a ListConfig? {isinstance(task_config.build_action, ListConfig)}"
    )
    print(
        f"Is build_action a DictConfig? {isinstance(task_config.build_action, DictConfig)}"
    )

    # Check if we need to convert
    if isinstance(task_config.build_action, ListConfig):
        print("\nConverting ListConfig to list:")
        as_list = list(task_config.build_action)
        print(f"   Converted value: {as_list}")
        print(f"   Converted type: {type(as_list)}")
        print(f"   Is list: {isinstance(as_list, list)}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    all_passed = (
        condition1
        and condition2
        and (not hasattr(task_config, "transform_sample") or condition3)
    )

    if all_passed:
        print("\n✓ ALL CONDITIONS PASSED - Functions will be loaded correctly")
    else:
        print("\n✗ SOME CONDITIONS FAILED - Functions will NOT be loaded!")
        print("\nRECOMMENDATION:")
        print("  The issue is likely that OmegaConf uses ListConfig instead of list.")
        print("  Solution: Import ListConfig and check for it:")
        print("    from omegaconf import ListConfig")
        print("    isinstance(task_config.build_action, (tuple, list, ListConfig))")


if __name__ == "__main__":
    test_config_loading()
