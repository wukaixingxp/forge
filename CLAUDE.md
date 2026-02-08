# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TorchForge is a PyTorch-native agentic RL library for post-training generative AI models. It separates infrastructure concerns from algorithm concerns, allowing researchers to write RL code that reads like pseudocode while Monarch handles distribution, TorchTitan handles training parallelism, vLLM handles inference, and TorchStore handles weight synchronization.

**Status:** Early development - expect API changes and incomplete features.

## Common Commands

### Installation
```bash
conda create -n forge python=3.12
conda activate forge
./scripts/install.sh
```

### Running Tests
```bash
# All unit tests
pytest -s tests/unit_tests/

# Specific test file
pytest -s tests/unit_tests/test_config.py

# Specific test function
pytest -s tests/unit_tests/test_config.py::test_cache_hit_scenario

# Integration tests (requires GPUs)
pytest -s tests/integration_tests/
```

### Code Quality
```bash
# Run all pre-commit checks
pre-commit run --all-files

# Enable pre-commit hooks
pre-commit install
```

### Running Applications
```bash
# SFT training (requires 2+ GPUs)
python -m apps.sft.main --config apps/sft/llama3_8b.yaml

# GRPO training (requires 3+ GPUs)
python -m apps.grpo.main --config apps/grpo/qwen3_1_7b.yaml
```

## Architecture

### Core Components

**Controller Layer** (`src/forge/controller/`): Distributed orchestration using Monarch's single-controller model. `ForgeActor` is the base class for all distributed actors. Services are configured via `ServiceConfig` (replicas, GPU allocation).

**Actors** (`src/forge/actors/`): Async actors communicating via Monarch:
- `Generator` - vLLM-based policy inference with weight sync
- `TitanTrainer` - Model training with TorchTitan (FSDP, tensor parallelism)
- `ReplayBuffer` - Episode storage and batching
- `ReferenceModel` - Frozen baseline for RL losses

**RL Components** (`src/forge/rl/`): Modular loss functions (GRPO, DAPO, GSPO, CISPO, SAPO) that inherit from `BaseLossConfig`. DAPO is the default. `ComputeAdvantages` calculates group-relative advantages.

**Data Pipeline** (`src/forge/data/`): Tokenizers, datasets (Alpaca format, HF datasets), and collation strategies (padded, packed).

### Key Patterns

**Async/Await Everywhere:** All actor communication is asynchronous:
```python
response = await generator.generate.call_one(prompt)
responses = await generator.generate.route(prompts)  # Returns ValueMesh
```

**Configuration-Driven:** YAML configs define model, training, and parallelism parameters.

**Core Data Types** (`src/forge/types.py`):
- `TrainBatch` - Contains `model_inputs`, `loss_inputs`, and `meta`
- `Episode` - Stores prompt/response/reward/logprobs/advantages
- `Group` - List of episodes for group-relative advantage computation

### Data Flow (GRPO)
```
Prompt → Generator → Response (with logprobs)
       → RewardActor → Reward
       → ComputeAdvantages → Advantages
       → ReplayBuffer → Episodes
       → TitanTrainer → Loss (with ref_logprobs)
```

## Code Style

- Max line length: 120 characters
- Pre-commit runs: ufmt (Black + usort), flake8, license headers
- PyTorch conventions: `import torch.nn.functional as F` is acceptable (N812 ignored)
- Lambda assignments allowed (E731 ignored)

## Key Dependencies

- PyTorch 2.9.0, TorchTitan 0.2.0, Monarch (torchmonarch 0.2.0)
- vLLM 0.13.0+ for inference
- TorchStore for weight synchronization
- OmegaConf for configuration
