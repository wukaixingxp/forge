# OpenEnv - Generic Training Framework

A centralized framework for training language models on any OpenEnv task using GRPO (Grouped Relative Policy Optimization).

## 📁 Folder Structure

```
apps/openenv/
  ├── main.py                    # Generic training script
  ├── julia_utils.py             # Julia task utilities
  ├── python_utils.py            # Python/coding task utilities
  ├── llama3_8b_julia.yaml       # Julia training config
  └── llama3_8b_coding.yaml      # Python coding training config
```

## 🎯 Key Features

- **Single Main Script**: One `main.py` works for all OpenEnv tasks
- **Task-Specific Utils**: Language-specific logic in separate files (e.g., `julia_utils.py`, `python_utils.py`)
- **YAML Configuration**: Use `!function` references to load task-specific functions
- **AutoEnv Integration**: Automatic environment and action class loading
- **Easy Extension**: Add new languages by creating new utils files

## 🚀 Usage

### Run Julia Training

```bash
python -m apps.openenv.main --config apps/openenv/llama3_8b_julia.yaml
```

### Run Python Coding Training

```bash
python -m apps.openenv.main --config apps/openenv/llama3_8b_coding.yaml
```

## 📝 YAML Configuration

Each task config needs:

```yaml
task:
  env_name: "julia"  # Environment name for AutoEnv
  build_action: !function julia_utils.build_julia_action
  evaluate_response: !function julia_utils.evaluate_julia_response
  transform_sample: !function julia_utils.transform_julia_sample
```

The `!function` tag references functions from the utils files in the same directory.

## 🔧 Adding a New Language

To add support for a new language (e.g., Rust):

### 1. Create Utils File

Create `/home/kaiwu/work/kaiwu/forge/apps/openenv/rust_utils.py`:

```python
from envs.rust_env import RustAction

def build_rust_action(response: str, sample: dict) -> RustAction:
    """Build RustAction from model response."""
    code = extract_rust_code(response)
    return RustAction(code=code, test_code=sample.get("test", ""))

def evaluate_rust_response(result, response: str, sample: dict) -> float:
    """Evaluate Rust code execution and return reward."""
    if result.observation.exit_code == 0:
        return 1.0
    return 0.0

def transform_rust_sample(sample: dict, tokenizer) -> dict | None:
    """Transform dataset sample for Rust tasks."""
    # Build prompt using tokenizer
    prompt = build_rust_prompt(sample, tokenizer)
    return {
        "request": prompt,
        "target": sample.get("test", ""),
        "task_id": sample.get("task_id", ""),
    }

def extract_rust_code(response: str) -> str:
    """Extract Rust code from markdown blocks."""
    # Implementation...
    pass
```

### 2. Create YAML Config

Create `/home/kaiwu/work/kaiwu/forge/apps/openenv/llama3_8b_rust.yaml`:

```yaml
task:
  env_name: "rust"
  build_action: !function rust_utils.build_rust_action
  evaluate_response: !function rust_utils.evaluate_rust_response
  transform_sample: !function rust_utils.transform_rust_sample

dataset:
  path: "path/to/rust/dataset"
  # ... other dataset config

# ... rest of config (same as other tasks)
```

### 3. Run It

```bash
python -m apps.openenv.main --config apps/openenv/llama3_8b_rust.yaml
```

That's it! No changes to `main.py` needed.

## 📋 Task Utils API

Each task utils file should implement these functions:

### Required Functions

1. **`build_<lang>_action(response: str, sample: dict) -> Action`**
   - Builds environment action from model response
   - Example: `build_julia_action`, `build_python_action`

2. **`evaluate_<lang>_response(result, response: str, sample: dict) -> float`**
   - Evaluates execution result and returns reward (0.0 to 1.0)
   - Example: `evaluate_julia_response`, `evaluate_python_response`

3. **`transform_<lang>_sample(sample: dict, tokenizer) -> dict | None`**
   - Transforms raw dataset sample into training format
   - Returns dict with 'request', 'target', 'task_id' or None if invalid
   - Example: `transform_julia_sample`, `transform_python_sample`

### Optional Helper Functions

- **`get_<lang>_system_prompt() -> str`**: Get system prompt for the language
- **`build_<lang>_prompt(sample: dict, tokenizer) -> str`**: Build formatted prompt
- **`extract_<lang>_code(response: str) -> str`**: Extract code from markdown

## 🔍 How It Works

1. **Configuration Loading**: YAML config is loaded with `!function` references
2. **Function Loading**: `main.py` dynamically loads functions from utils files
3. **Environment Setup**: AutoEnv automatically loads correct env/action classes
4. **Training Loop**: Generic GRPO loop uses task-specific functions for:
   - Dataset transformation
   - Action building
   - Reward evaluation

## 📊 Dataset Format

Each transformed sample should have:

```python
{
    "request": str,   # Formatted prompt for model
    "target": str,    # Test code or target data
    "task_id": str,   # Unique task identifier
}
```

## 🎓 Examples

### Julia Utils

- System prompt with strict formatting rules
- Extract code from markdown blocks
- Dense reward based on test pass rate
- Handles Julia-specific syntax requirements

### Python Utils

- Simple system prompt for Python coding
- Binary/proportional reward structure
- Extracts code from markdown blocks
- Works with HumanEval dataset format

## 🔗 Integration with OpenEnv

This framework uses OpenEnv's AutoEnv feature:

```python
from envs import AutoEnv, AutoAction

env_class = AutoEnv.from_name("julia")      # Loads JuliaEnv
action_class = AutoAction.from_env("julia")  # Loads JuliaAction
```

Make sure your environment is registered in OpenEnv's registry.

## 🐛 Debugging

Enable debug logging in main.py to see:
- Function loading
- Environment setup
- Reward calculation
- Code extraction

Set log level in YAML:
```yaml
metric_logging:
  console:
    log_per_rank: True
```

## 📚 References

- **GRPO Algorithm**: Grouped Relative Policy Optimization
- **OpenEnv**: Generic environment framework
- **AutoEnv**: Automatic environment detection
- **GenericOpenEnvActor**: Docker-based environment execution
