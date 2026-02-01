#!/bin/bash
# GPU-Direct Weight Sync Benchmark
# Compares IPC vs TorchStore weight synchronization performance
#
# Usage: ./apps/gpu_direct/benchmark.sh [model_size]
#   model_size: 4b (default), 32b, 30b_moe

set -e

cd /home/dev/framework/torchforge

# Activate conda environment
source /opt/conda/etc/profile.d/conda.sh && conda activate vllm 2>/dev/null || true

# Set Python path
export PYTHONPATH="src:../torchstore:../torchtitan:$PYTHONPATH"

# Model size (default: 4b for quick testing)
MODEL_SIZE="${1:-4b}"

echo "=============================================="
echo "GPU-Direct Weight Sync Benchmark"
echo "Model: Qwen3-${MODEL_SIZE^^}"
echo "=============================================="
echo ""

# Select config based on model size
case $MODEL_SIZE in
    4b)
        IPC_CONFIG="apps/gpu_direct/qwen3_4b_2x2.yaml"
        # Create a baseline version for 4B
        BASELINE_CONFIG="apps/gpu_direct/qwen3_4b_2x2.yaml"
        ;;
    32b)
        IPC_CONFIG="apps/gpu_direct/qwen3_32b_2x2.yaml"
        BASELINE_CONFIG="apps/gpu_direct/qwen3_32b_2x2_baseline.yaml"
        ;;
    30b_moe)
        IPC_CONFIG="apps/gpu_direct/qwen3_30b_moe_2x2.yaml"
        BASELINE_CONFIG="apps/gpu_direct/qwen3_30b_moe_2x2.yaml"
        ;;
    *)
        echo "Unknown model size: $MODEL_SIZE"
        echo "Options: 4b, 32b, 30b_moe"
        exit 1
        ;;
esac

# Clean up stale CUDA shared memory
echo "Cleaning up stale CUDA shared memory..."
rm -f /dev/shm/cuda.shm.* /dev/shm/torch_* 2>/dev/null || true

echo ""
echo "----------------------------------------------"
echo "Running BASELINE (TorchStore) benchmark..."
echo "----------------------------------------------"
echo ""

# Disable IPC for baseline
unset FORGE_IPC_GPU_VISIBILITY

if [ "$MODEL_SIZE" = "4b" ]; then
    # Override IPC setting for 4B baseline test
    python -m apps.gpu_direct.main \
        --config "$IPC_CONFIG" \
        weight_sync.use_ipc=false \
        trainer.training.steps=10 \
        2>&1 | tee /tmp/baseline_output.log
else
    python -m apps.gpu_direct.main \
        --config "$BASELINE_CONFIG" \
        trainer.training.steps=10 \
        2>&1 | tee /tmp/baseline_output.log
fi

echo ""
echo "----------------------------------------------"
echo "Running IPC (GPU-Direct) benchmark..."
echo "----------------------------------------------"
echo ""

# Enable IPC
export FORGE_IPC_GPU_VISIBILITY=1

# Clean up CUDA shared memory between runs
rm -f /dev/shm/cuda.shm.* /dev/shm/torch_* 2>/dev/null || true

python -m apps.gpu_direct.main \
    --config "$IPC_CONFIG" \
    trainer.training.steps=10 \
    2>&1 | tee /tmp/ipc_output.log

echo ""
echo "=============================================="
echo "BENCHMARK COMPLETE"
echo "=============================================="
echo ""
echo "Check the Weight Sync Summary sections in the output above"
echo "to compare TorchStore vs IPC performance."
echo ""
echo "Logs saved to:"
echo "  Baseline: /tmp/baseline_output.log"
echo "  IPC:      /tmp/ipc_output.log"
