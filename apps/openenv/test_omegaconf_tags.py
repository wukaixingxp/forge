#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Test how OmegaConf handles YAML tags."""

import tempfile
from pathlib import Path

from omegaconf import OmegaConf

# Create a test YAML with various tag formats
test_yaml = """
test1: !function julia_utils.my_function
test2: !!python/tuple [!function, julia_utils.my_function]
test3:
  - !function
  - julia_utils.my_function
test4: [!function, julia_utils.my_function]
"""

# Write to temp file
with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
    f.write(test_yaml)
    temp_path = f.name

try:
    print("Loading YAML with OmegaConf...")
    cfg = OmegaConf.load(temp_path)
    print("SUCCESS! Config loaded.\n")

    print("=" * 80)
    print("CONTENTS:")
    print("=" * 80)
    print(OmegaConf.to_yaml(cfg))

    print("\n" + "=" * 80)
    print("ANALYSIS:")
    print("=" * 80)

    for key in ["test1", "test2", "test3", "test4"]:
        if key in cfg:
            val = cfg[key]
            print(f"\n{key}:")
            print(f"  Value: {val}")
            print(f"  Type: {type(val)}")
            print(f"  Type name: {type(val).__name__}")

except Exception as e:
    print(f"ERROR: {e}")
    import traceback

    traceback.print_exc()
finally:
    # Clean up
    Path(temp_path).unlink()
