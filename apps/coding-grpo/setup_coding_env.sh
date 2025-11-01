#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Setup script for OpenEnv coding environment

set -e

echo "=========================================="
echo "OpenEnv Coding Environment Setup"
echo "=========================================="

# Stop and remove any existing coding-env containers
echo ""
echo "1. Cleaning up existing containers..."
podman ps -a | grep coding-env | awk '{print $1}' | xargs -r podman stop || true
podman ps -a | grep coding-env | awk '{print $1}' | xargs -r podman rm || true
echo "✓ Cleanup complete"

# # Build the Docker image
# echo ""
# echo "2. Building coding-env:latest Docker image..."
# cd /home/kaiwu/work/kaiwu/OpenEnv
# if [ ! -f "src/envs/coding_env/server/Dockerfile" ]; then
#     echo "ERROR: Dockerfile not found at src/envs/coding_env/server/Dockerfile"
#     exit 1
# fi

# podman build -t coding-env:latest -f src/envs/coding_env/server/Dockerfile .
# echo "✓ Docker image built successfully"

# Verify the image exists
echo ""
echo "3. Verifying Docker image..."
if podman images | grep -q "coding-env.*latest"; then
    echo "✓ coding-env:latest image is ready"
else
    echo "ERROR: coding-env:latest image not found"
    exit 1
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "You can now run your training script:"
echo "  python -m apps.coding-grpo.main --config apps/coding-grpo/llama3_8b_hard.yaml"
