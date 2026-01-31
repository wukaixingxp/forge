# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""
GPU-Direct Weight Sync Demo

Demonstrates the new GPU-direct weight synchronization feature that:
1. Stores FSDP shards directly without gathering (push_weights_sharded)
2. Fetches TP-aware slices from stored shards (update_weights_gpu_direct)
3. Eliminates CPU memory bottleneck from FSDP all_gather

Demo scenario:
- Trainer: 2 GPUs with FSDP (each GPU holds 1/2 of each parameter's rows)
- Generator: 2 GPUs with TP=2 (each GPU holds 1/2 of columns for QKV, rows for O)

Usage:
    # Simplified API test (no GPU required)
    python -m demos.gpu_direct_weight_sync.run_demo --simplified

    # Full demo with Llama 4 Scout (requires 4 GPUs)
    python -m demos.gpu_direct_weight_sync.run_demo
"""
