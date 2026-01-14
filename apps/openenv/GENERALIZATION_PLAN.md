# Plan: Full Generalization Using GenericEnvClient & GenericAction

## Executive Summary

Transform the torchforge OpenEnv integration to be **fully generic** by leveraging OpenEnv's `GenericEnvClient` and `GenericAction` abstractions. This eliminates ALL hardcoded environment-specific logic, allowing new environments to be added via YAML configuration alone.

## Current State Analysis

### Hardcoded Elements

#### 1. `src/forge/actors/generic_openenv.py` (Lines 218-248)
```python
# Fallback imports for specific environments
if env_key in ["coding", "code"]:
    from coding_env import CodeAction, CodingEnv
    ...
elif env_key in ["julia"]:
    from julia_env import JuliaAction, JuliaEnv
    ...
```
**Problem**: Requires code changes for every new environment.

#### 2. `apps/openenv/main.py` (Lines 638-653)
```python
# Environment-specific env var handling
if env_name == "julia" and "JULIA_MAX_WORKERS" not in env_vars:
    env_vars["JULIA_MAX_WORKERS"] = ...
if env_name == "coding" and "PYTHON_ADDITIONAL_IMPORTS" not in env_vars:
    env_vars["PYTHON_ADDITIONAL_IMPORTS"] = ...
```
**Problem**: Every new environment requires code changes.

#### 3. Task-Specific Utility Files
- `apps/openenv/julia_utils.py`: Julia-specific prompt, action building, reward evaluation
- `apps/openenv/python_utils.py`: Python-specific prompt, action building, reward evaluation

**Problem**: Each task requires a new Python file with custom functions.

#### 4. YAML Configs Use `!function` References
```yaml
task:
  build_action: !function apps.openenv.julia_utils.build_julia_action
  evaluate_response: !function apps.openenv.julia_utils.evaluate_julia_response
```
**Problem**: Still requires Python code for each task.

---

## The Solution: GenericEnvClient + GenericAction

### Key OpenEnv Abstractions

From the OpenEnv exploration, we have:

```python
from openenv.core.generic_client import GenericEnvClient, GenericAction

# GenericEnvClient: Works with raw dictionaries (no typed classes)
env = GenericEnvClient.from_docker_image("any-env:latest")

# GenericAction: Dict subclass that can represent ANY action
action = GenericAction(code="...", test_code="...", timeout=30)
# OR for Julia:
action = GenericAction(core_code="...", test_code="...")
# OR for ANY environment:
action = GenericAction(**arbitrary_fields)

# Execute - observation is also a dict
result = env.step(action)
reward = result.observation.get("reward", 0.0)
```

**Key Benefits**:
1. ✅ No environment-specific imports needed
2. ✅ Works with all OpenEnv environments out of the box
3. ✅ Observations are dicts - access fields dynamically
4. ✅ Actions are dicts - construct from YAML config

---

## Implementation Plan

### Phase 1: Refactor `GenericOpenEnvActor` to Use GenericEnvClient

**File**: `src/forge/actors/generic_openenv.py`

**Changes**:

1. **Remove environment-specific fallback imports (Lines 218-248)**
   - Delete the entire `if env_key in ["coding", "code"]` block
   - Delete the entire `elif env_key in ["julia"]` block

2. **Use GenericEnvClient when AutoEnv fails**
   ```python
   from openenv.core.generic_client import GenericEnvClient, GenericAction

   @classmethod
   def get_init_kwargs_from_env_name(cls, env_name: str, docker_image: str, ...):
       # Try AutoEnv first (for typed environments)
       try:
           env_class = AutoEnv.get_env_class(env_name)
           action_class = AutoAction.from_env(env_name)
       except (ValueError, ImportError) as e:
           # Fall back to GenericEnvClient (works for ALL environments)
           logger.info(f"Using GenericEnvClient for '{env_name}'")
           env_class = GenericEnvClient
           action_class = GenericAction

           # Get default image from metadata if possible
           if docker_image is None:
               docker_image = f"{env_name}-env:latest"

       return {
           "env_class": env_class,
           "action_class": action_class,
           "docker_image": docker_image,
           ...
       }
   ```

3. **Make `create_action` helper work with both typed and generic actions**
   ```python
   def create_action(self, **kwargs) -> Action:
       """Create action from kwargs - works for both typed and generic."""
       return self.action_class(**kwargs)
   ```

**Result**: GenericOpenEnvActor becomes truly universal - no hardcoded environment knowledge.

---

### Phase 2: Create Generic Reward Evaluation System

**Goal**: Replace task-specific `*_utils.py` files with a generic, config-driven approach.

#### 2.1: Create Generic Action Builder

**New File**: `apps/openenv/generic_utils.py`

```python
from typing import Any, Dict
from openenv.core.generic_client import GenericAction

def build_generic_action(
    response: str,
    sample: Dict[str, Any],
    action_fields: Dict[str, str],  # Field mapping from config
    code_extraction: Dict[str, Any],  # Code extraction config
) -> GenericAction:
    """
    Build GenericAction from response using config-driven field mapping.

    Args:
        response: Model's generated response
        sample: Dataset sample
        action_fields: Mapping of action fields to data sources
            Example: {"code": "response_code", "test_code": "sample.target"}
        code_extraction: Config for extracting code from markdown
            Example: {"enabled": true, "language": "python", "pattern": "```python\\n(.+?)```"}

    Returns:
        GenericAction with fields populated from config
    """
    # Extract code from markdown if needed
    code = response
    if code_extraction.get("enabled", False):
        pattern = code_extraction.get("pattern")
        if pattern:
            import re
            match = re.search(pattern, response, re.DOTALL)
            if match:
                code = match.group(1).strip()
        else:
            # Simple markdown stripping
            lang = code_extraction.get("language", "")
            code = re.sub(rf"^```{lang}\s*\n?", "", response, flags=re.IGNORECASE)
            code = re.sub(r"\n?```\s*$", "", code).strip()

    # Build action fields from mapping
    action_data = {}
    for field_name, source in action_fields.items():
        if source == "response_code":
            action_data[field_name] = code
        elif source == "response_raw":
            action_data[field_name] = response
        elif source.startswith("sample."):
            # Extract from sample dict
            key = source[7:]  # Remove "sample." prefix
            action_data[field_name] = sample.get(key, "")
        elif source.startswith("literal:"):
            # Literal value
            action_data[field_name] = source[8:]
        else:
            # Direct sample key
            action_data[field_name] = sample.get(source, "")

    return GenericAction(**action_data)
```

#### 2.2: Create Generic Reward Evaluator

```python
def evaluate_generic_response(
    result,
    response: str,
    sample: Dict[str, Any],
    reward_config: Dict[str, Any],
) -> float:
    """
    Evaluate response using config-driven reward logic.

    Args:
        result: StepResult from environment
        response: Model's response
        sample: Dataset sample
        reward_config: Reward calculation config
            Example: {
                "source": "observation.reward",  # Where to get reward
                "fallback": 0.0,
                "logging": {
                    "print_response": true,
                    "print_observation": true,
                    "metrics": ["pass_rate", "compilation"]
                }
            }

    Returns:
        Reward score
    """
    obs = result.observation

    # Extract reward from configured source
    reward_source = reward_config.get("source", "observation.reward")
    if reward_source == "observation.reward":
        reward = obs.get("reward", reward_config.get("fallback", 0.0))
    elif reward_source == "result.reward":
        reward = result.reward if result.reward is not None else reward_config.get("fallback", 0.0)
    elif reward_source.startswith("observation."):
        # Extract from observation dict
        key = reward_source[12:]  # Remove "observation." prefix
        reward = obs.get(key, reward_config.get("fallback", 0.0))
    else:
        reward = reward_config.get("fallback", 0.0)

    # Optional logging
    if reward_config.get("logging", {}).get("print_response"):
        print("=" * 80)
        print("MODEL RESPONSE:")
        print(response)
        print("-" * 80)

    if reward_config.get("logging", {}).get("print_observation"):
        print("OBSERVATION:")
        for key, value in obs.items():
            if isinstance(value, str) and len(value) > 200:
                print(f"  {key}: {value[:200]}...")
            else:
                print(f"  {key}: {value}")
        print("-" * 80)
        print(f"Reward: {reward}")
        print("=" * 80)

    # Optional metric recording
    for metric_name in reward_config.get("logging", {}).get("metrics", []):
        if metric_name == "pass_rate" and "tests_passed" in obs and "tests_failed" in obs:
            total = obs["tests_passed"] + obs["tests_failed"]
            if total > 0:
                record_metric(f"reward/pass_rate", obs["tests_passed"] / total, Reduce.MEAN)
        elif metric_name in obs:
            record_metric(f"reward/{metric_name}", obs[metric_name], Reduce.MEAN)

    return float(reward)
```

#### 2.3: Create Generic Dataset Transformer

```python
def transform_generic_sample(
    sample: Dict[str, Any],
    tokenizer,
    transform_config: Dict[str, Any],
) -> Dict[str, Any] | None:
    """
    Transform dataset sample using config-driven logic.

    Args:
        sample: Raw dataset sample
        tokenizer: HuggingFace tokenizer
        transform_config: Transformation config
            Example: {
                "prompt_field": "julia_prompt",  # OR "prompt" for Python
                "target_field": "julia_test",    # OR "test" for Python
                "task_id_field": "task_id",
                "system_prompt": "You are a Julia expert...",
                "required_fields": ["julia_prompt", "julia_test"]
            }

    Returns:
        Transformed sample or None if invalid
    """
    # Validate required fields
    required = transform_config.get("required_fields", [])
    for field in required:
        if not sample.get(field):
            return None

    # Extract prompt
    prompt_field = transform_config.get("prompt_field", "prompt")
    prompt_text = sample.get(prompt_field, "")

    # Build messages
    system_prompt = transform_config.get("system_prompt", "")
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt_text},
    ]

    # Apply chat template
    formatted_request = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Extract target
    target_field = transform_config.get("target_field", "target")
    target = sample.get(target_field, "")

    # Extract task ID
    task_id_field = transform_config.get("task_id_field", "task_id")
    task_id = sample.get(task_id_field, "")

    return {
        "request": formatted_request,
        "target": target,
        "task_id": task_id,
    }
```

**Result**: All task-specific logic is replaced by config-driven generic functions.

---

### Phase 3: Simplify `main.py`

**File**: `apps/openenv/main.py`

**Changes**:

1. **Remove environment-specific env var handling (Lines 638-653)**

   Replace:
   ```python
   if env_name == "julia" and "JULIA_MAX_WORKERS" not in env_vars:
       env_vars["JULIA_MAX_WORKERS"] = ...
   if env_name == "coding" and "PYTHON_ADDITIONAL_IMPORTS" not in env_vars:
       env_vars["PYTHON_ADDITIONAL_IMPORTS"] = ...
   ```

   With:
   ```python
   # All env vars come from YAML config - no hardcoding
   env_vars = openenv_config.get("env_vars", {})
   ```

2. **Replace `!function` loading with config-based approach**

   Instead of:
   ```python
   build_action_fn = load_function_from_string(task_config.build_action[1])
   evaluate_response_fn = load_function_from_string(task_config.evaluate_response[1])
   transform_sample_fn = load_function_from_string(task_config.transform_sample[1])
   ```

   Use:
   ```python
   from apps.openenv.generic_utils import (
       build_generic_action,
       evaluate_generic_response,
       transform_generic_sample,
   )

   # Create closures that capture config
   action_config = task_config.get("action_config", {})
   reward_config = task_config.get("reward_config", {})
   transform_config = task_config.get("transform_config", {})

   build_action_fn = lambda response, sample: build_generic_action(
       response, sample, action_config, reward_config.get("code_extraction", {})
   )

   evaluate_response_fn = lambda result, response, sample: evaluate_generic_response(
       result, response, sample, reward_config
   )

   transform_sample_fn = lambda sample, tokenizer: transform_generic_sample(
       sample, tokenizer, transform_config
   )
   ```

3. **Simplify GenericRewardActor initialization**
   ```python
   reward_actor = await GenericRewardActor.options(...).as_service(
       env_actor=env_actor,
       build_action_fn=build_action_fn,
       evaluate_response_fn=evaluate_response_fn,
       evaluation_timeout_s=evaluation_timeout_s,
   )
   ```
   (This part stays the same - the functions just come from generic utils now)

**Result**: `main.py` has ZERO environment-specific code.

---

### Phase 4: Update YAML Configs

**Goal**: Move all task-specific logic from Python code to YAML config.

#### 4.1: New Julia Config Structure

**File**: `apps/openenv/llama3_8b_julia.yaml`

```yaml
# Task configuration - now fully declarative!
task:
  env_name: "julia"

  # Action building config (replaces build_julia_action function)
  action_config:
    code: "response_code"        # Extract code from response
    test_code: "sample.target"   # Get test from sample.target

  # Reward evaluation config (replaces evaluate_julia_response function)
  reward_config:
    source: "observation.reward"  # Where to find reward
    fallback: 0.0
    code_extraction:
      enabled: true
      language: "julia"
      pattern: "```julia\\n(.+?)```"
    logging:
      print_response: true
      print_observation: true
      metrics:
        - "pass_rate"
        - "tests_passed"
        - "tests_failed"

  # Dataset transformation config (replaces transform_julia_sample function)
  transform_config:
    prompt_field: "julia_prompt"
    target_field: "julia_test"
    task_id_field: "task_id"
    required_fields:
      - "julia_test"
      - "first_test_case"
    system_prompt: |
      You are a precise and pragmatic Julia programmer.

      Write a **single Julia function** that correctly solves the problem.

      CRITICAL - Julia is NOT Python! Use correct Julia syntax:
      - Use `lowercase()` NOT `tolower()`
      - Use `uppercase()` NOT `upper()`
      - Arrays are 1-indexed, NOT 0-indexed

      FORMAT YOUR RESPONSE AS:
      ```julia
      function <function_name>(<argument_list>)
          <function_body>
      end
      ```

# OpenEnv configuration
openenv_config:
  docker_image: "julia-env:latest"
  container_timeout_s: 180.0
  container_memory_gb: 4
  port: 8000
  request_timeout_s: 60.0

  # Environment-specific env vars (no more hardcoding in main.py!)
  env_vars:
    PORT: "8000"
    NUM_WORKER: "16"
    JULIA_MAX_WORKERS: "16"
    JULIA_EXECUTION_TIMEOUT: "60"

# ... rest of config stays the same
```

#### 4.2: New Python Config Structure

**File**: `apps/openenv/llama3_8b_coding.yaml`

```yaml
task:
  env_name: "coding"

  action_config:
    code: "response_code"
    test_code: "sample.target"

  reward_config:
    source: "observation.reward"
    fallback: 0.0
    code_extraction:
      enabled: true
      language: "python"
    logging:
      print_response: true
      print_observation: true
      metrics:
        - "pass_rate"

  transform_config:
    prompt_field: "prompt"
    target_field: "test"
    task_id_field: "task_id"
    required_fields:
      - "prompt"
    system_prompt: |
      You are an expert Python programmer.
      Write a Python function that correctly solves the problem.

      FORMAT YOUR RESPONSE AS:
      ```python
      def function_name(args):
          # implementation
          return result
      ```

openenv_config:
  docker_image: "coding-env:latest"
  container_timeout_s: 180.0
  container_memory_gb: 4
  env_vars:
    PORT: "8000"
    NUM_WORKER: "4"
    PYTHON_ADDITIONAL_IMPORTS: "numpy,pandas"  # No more hardcoding!
```

**Result**: All task-specific configuration is in YAML - no Python code needed!

---

## Adding a New Environment (Example: Rust)

With this new architecture, adding Rust support requires **ZERO** code changes:

### Step 1: Create YAML Config Only

**File**: `apps/openenv/llama3_8b_rust.yaml`

```yaml
task:
  env_name: "rust"

  action_config:
    code: "response_code"
    test_code: "sample.target"

  reward_config:
    source: "observation.reward"
    fallback: 0.0
    code_extraction:
      enabled: true
      language: "rust"
    logging:
      print_response: true
      print_observation: true

  transform_config:
    prompt_field: "problem"
    target_field: "tests"
    task_id_field: "id"
    system_prompt: |
      You are an expert Rust programmer.
      Write safe, idiomatic Rust code.

openenv_config:
  docker_image: "rust-env:latest"
  env_vars:
    PORT: "8000"
    RUST_BACKTRACE: "1"

# ... standard training config
```

### Step 2: Run Training

```bash
python -m apps.openenv.main --config apps/openenv/llama3_8b_rust.yaml
```

**That's it!** No code changes needed in forge.

---

## Migration Path

### Option A: Big Bang (Recommended)

1. Implement all phases at once
2. Update both Julia and Python configs
3. Delete old `*_utils.py` files
4. Test thoroughly

**Pros**: Clean, no hybrid state
**Cons**: Larger initial changeset

### Option B: Gradual

1. Phase 1 first (GenericEnvClient fallback)
2. Keep existing `*_utils.py` working
3. Add support for config-driven approach alongside `!function`
4. Migrate configs one by one
5. Delete old utils when all migrated

**Pros**: Lower risk, incremental validation
**Cons**: Temporary complexity, hybrid state

---

## Benefits Summary

### Before (Current State)
- ❌ Every new environment requires Python code changes
- ❌ Hardcoded environment checks in multiple files
- ❌ Task-specific utility files for each language
- ❌ `!function` references requiring Python modules
- ❌ Environment-specific imports scattered everywhere

### After (Generalized)
- ✅ New environments = YAML config only
- ✅ Zero hardcoded environment knowledge
- ✅ Single generic utility module
- ✅ Pure declarative config
- ✅ Only GenericEnvClient/GenericAction imports

---

## Testing Plan

### 1. Unit Tests
- Test `build_generic_action` with various configs
- Test `evaluate_generic_response` with different observation formats
- Test `transform_generic_sample` with different field mappings

### 2. Integration Tests
- Run Julia training with new config format
- Run Python training with new config format
- Verify metrics match old approach

### 3. End-to-End Validation
- Train for a few steps with Julia
- Train for a few steps with Python
- Compare rewards and logs with old implementation

### 4. New Environment Test
- Create a mock environment (e.g., Echo)
- Add YAML config
- Verify training works without code changes

---

## Files to Create/Modify

### New Files
- ✅ `apps/openenv/generic_utils.py` - Generic action/reward/transform functions
- ✅ `apps/openenv/GENERALIZATION_PLAN.md` - This document

### Modified Files
- `src/forge/actors/generic_openenv.py` - Remove fallback imports, use GenericEnvClient
- `apps/openenv/main.py` - Remove env-specific logic, use generic utils
- `apps/openenv/llama3_8b_julia.yaml` - Convert to declarative config
- `apps/openenv/llama3_8b_coding.yaml` - Convert to declarative config

### Deleted Files (after migration)
- `apps/openenv/julia_utils.py` - Replaced by generic_utils.py + YAML config
- `apps/openenv/python_utils.py` - Replaced by generic_utils.py + YAML config

---

## Open Questions

1. **Q**: Should we support both `!function` and config-driven approaches during migration?
   **A**: TBD - depends on migration strategy chosen

2. **Q**: Do we need environment-specific validation (like `validate_julia_syntax`)?
   **A**: Could be optional in config: `validation: {enabled: true, checks: [...]}`

3. **Q**: How to handle complex reward functions (e.g., gibberish detection)?
   **A**: Could add optional "reward_filters" in config with predefined filter types

4. **Q**: Should GenericEnvClient be the default even when AutoEnv succeeds?
   **A**: No - typed clients provide better validation. Use GenericEnvClient as fallback only.

---

## Next Steps

1. **Review this plan** - Get feedback on approach
2. **Choose migration strategy** - Big bang vs gradual
3. **Implement Phase 1** - GenericEnvClient fallback in `generic_openenv.py`
4. **Implement Phase 2** - Create `generic_utils.py`
5. **Implement Phase 3** - Update `main.py`
6. **Implement Phase 4** - Convert YAML configs
7. **Test thoroughly** - Unit, integration, E2E
8. **Delete old code** - Remove `*_utils.py` files
9. **Update documentation** - Update CLAUDE.md and README

---

## Conclusion

This plan eliminates ALL hardcoded environment-specific logic from torchforge by leveraging OpenEnv's GenericEnvClient and GenericAction abstractions. After implementation:

- **Adding a new environment = Creating a YAML config**
- **No code changes needed in forge**
- **Truly generic, scalable architecture**

The key insight: OpenEnv already solved the generalization problem with GenericEnvClient/GenericAction. We just need to use them!
