# Julia GRPO Training with Forge

This example demonstrates how to use GRPO (Grouped Relative Policy Optimization) to train a language model for Julia code generation using the Forge framework.

## Overview

This implementation is based on the unsloth.py training script but adapted for the Forge framework. It uses:
- **Dataset**: Julia code generation dataset (`julia_trainset.parquet`)
- **Reward Function**: JuliaEnv for testing generated Julia code
- **Model**: Llama-3.1-8B-Instruct (full fine-tuning, no LoRA)
- **Hyperparameters**: Tuned for Julia code generation tasks

## Key Components

### 1. Dataset (`main.py` - DatasetActor)
- Loads Julia training data from parquet file
- Formats prompts with Julia-specific system instructions
- Includes test cases as context for code generation

### 2. Reward Function (`src/forge/actors/julia_env_reward.py`)
- Evaluates generated Julia code using JuliaEnv
- Sends code to JuliaEnv server running in Docker
- Returns rewards based on test success rate

### 3. Configuration (`llama3_8b_julia.yaml`)
Key hyperparameters from unsloth.py:
- `group_size: 2` (num_generations)
- `batch_size: 1`
- `max_req_tokens: 2048` (max_prompt_length)
- `max_res_tokens: 1024` (max_completion_length)
- `learning_rate: 5e-5`
- `weight_decay: 0.01`
- `warmup_steps: 50` (10% of 500 steps)
- `max_steps: 500`
- `temperature: 1.0`

## Setup Instructions

### 1. Prepare JuliaEnv Server

Before running the training, you need to start the JuliaEnv server in Docker:

```bash
# Build the OpenEnv base image (one-time setup)
cd /path/to/OpenEnv
docker build -t openenv-base:latest -f src/core/containers/images/Dockerfile .

# Build Julia Environment Image
docker build -t julia-env:latest -f src/envs/julia_env/server/Dockerfile .

# Run the JuliaEnv server
docker run -d -p 8000:8000 --name julia-env-server julia-env:latest

# Verify the server is running
curl http://localhost:8000/health
```

### 2. Prepare Dataset

Ensure your Julia dataset is available at:
```
/home/kaiwu/work/julia_trainset.parquet
```

The dataset should have these columns:
- `julia_prompt`: The problem description
- `julia_test`: Full test code for evaluation
- `first_test_case`: First test case for prompt context
- `task_id`: Unique identifier for the task

### 3. Run Training

```bash
# From the forge root directory
python -m apps.julia-grpo.main --config apps/julia-grpo/llama3_8b_julia.yaml
```

## Differences from unsloth.py

### What was adapted:
1. **Dataset loading**: Changed from unsloth's Dataset to Forge's DatasetActor
2. **Reward function**: Replaced unsloth's inline reward function with JuliaEnvReward actor
3. **Training framework**: Using Forge's GRPO implementation instead of TRL's GRPOTrainer
4. **Full fine-tuning**: Forge only supports full fine-tuning (no LoRA)

### What was kept:
1. **Julia prompt template**: Exact same system prompt for code generation
2. **Hyperparameters**: Learning rate, weight decay, warmup ratio, max steps
3. **Token limits**: max_prompt_length=2048, max_completion_length=1024
4. **Temperature**: 1.0 for sampling
5. **JuliaEnv integration**: Same reward calculation logic

## File Structure

```
apps/julia-grpo/
├── main.py                      # Main training script with Julia-specific DatasetActor
├── llama3_8b_julia.yaml        # Configuration file with Julia hyperparameters
├── unsloth.py                  # Original reference implementation
└── README.md                   # This file

src/forge/actors/
└── julia_env_reward.py         # JuliaEnv reward function implementation
```

## Monitoring

The training logs metrics to:
- **WandB**: Project `kaiwu-julia-grpo`
- **Console**: Real-time logging with per-rank details

Key metrics to watch:
- `reward/evaluate_response/avg_JuliaEnvReward_reward`: Average reward from JuliaEnv
- `loss/total_loss`: Total training loss
- `advantages/mean_after`: Advantage normalization (should be ~0)

## Troubleshooting

### JuliaEnv Connection Issues
```bash
# Check if JuliaEnv server is running
docker ps | grep julia-env

# Check server logs
docker logs julia-env-server

# Restart if needed
docker restart julia-env-server
```

### Dataset Issues
- Ensure the parquet file exists at the specified path
- Verify it has the required columns: `julia_prompt`, `julia_test`, `first_test_case`, `task_id`

### Memory Issues
- Adjust `gpu_memory_utilization` in the config (default: 0.85)
- Reduce `batch_size` or `max_tokens` if OOM occurs

## Next Steps

After training completes:
1. Checkpoints are saved in `checkpoint_llama3_8b_julia/`
2. Final model can be converted to HuggingFace format
3. Evaluate on Julia code generation benchmarks
4. Compare with the unsloth.py baseline results

## Notes

- This implementation uses **full fine-tuning** (no LoRA) as Forge currently doesn't support LoRA
- The JuliaEnv server must be running and accessible at `http://localhost:8000`
- Training takes approximately X hours on Y GPUs (adjust based on your setup)
