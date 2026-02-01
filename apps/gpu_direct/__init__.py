# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
GPU-Direct Weight Sync GRPO Example

This app demonstrates GRPO training with GPU-direct weight synchronization
using CUDA IPC handles for 60x faster weight transfer between trainer and generator.

Key Features:
- CUDA IPC direct GPU-to-GPU weight transfer
- Supports FSDP (trainer) + TP (generator) configurations
- Side-by-side comparison with TorchStore baseline

Usage:
    # With IPC (recommended)
    export FORGE_IPC_GPU_VISIBILITY=1
    python -m apps.gpu_direct.main --config apps/gpu_direct/qwen3_32b_2x2.yaml

    # Baseline comparison
    python -m apps.gpu_direct.main --config apps/gpu_direct/qwen3_32b_2x2_baseline.yaml
"""

from apps.gpu_direct.data import DatasetActor
from apps.gpu_direct.grading import MathReward, ThinkingReward

__all__ = ["DatasetActor", "MathReward", "ThinkingReward"]
