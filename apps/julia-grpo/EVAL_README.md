# Julia Code Generation Evaluation

This evaluation script evaluates Julia code generation models on the `julia_testset.parquet` dataset.

## Prerequisites

1. **Julia Environment Server must be running:**
   ```bash
   # If not already built, build the Julia environment image
   docker build -t julia-env:latest -f src/envs/julia_env/server/Dockerfile .

   # Run the server
   docker run -d -p 8000:8000 --name julia-env-server julia-env:latest
   ```

2. **OpenEnv must be available:**
   - The script expects OpenEnv to be in the parent directory (`../OpenEnv`)
   - Make sure the path is correct in your setup

## Usage

### Basic Baseline Evaluation

Run evaluation on the baseline model (Meta-Llama-3.1-8B-Instruct):

```bash
python eval.py \
    --model_name "meta-llama/Meta-Llama-3.1-8B-Instruct" \
    --test_file "julia_testset.parquet"
```

This will:
- Load the baseline model
- Evaluate on all examples in `julia_testset.parquet`
- Save results to `eval_results_meta_llama_Meta_Llama_3.1_8B_Instruct_baseline.parquet`
- Save summary metrics to `eval_results_meta_llama_Meta_Llama_3.1_8B_Instruct_baseline_summary.json`

### Evaluate Fine-tuned Model (LoRA)

```bash
python eval.py \
    --model_name "unsloth/Llama-3.2-3B-Instruct" \
    --lora_path "grpo_saved_lora_julia" \
    --test_file "julia_testset.parquet"
```

### Quick Test (Few Samples)

To quickly test on a small subset:

```bash
python eval.py \
    --model_name "meta-llama/Meta-Llama-3.1-8B-Instruct" \
    --test_file "julia_testset.parquet" \
    --num_samples 10
```

### All Available Options

```bash
python eval.py \
    --model_name "meta-llama/Meta-Llama-3.1-8B-Instruct" \
    --lora_path "grpo_saved_lora_julia" \
    --test_file "julia_testset.parquet" \
    --output_file "my_eval_results.parquet" \
    --localhost "http://localhost:8000" \
    --max_seq_length 2000 \
    --temperature 0.8 \
    --top_p 0.95 \
    --max_tokens 1024 \
    --num_samples 100
```

## Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_name` | `meta-llama/Meta-Llama-3.1-8B-Instruct` | Model name or path |
| `--lora_path` | `None` | Path to LoRA weights (optional) |
| `--test_file` | `julia_testset.parquet` | Path to test dataset |
| `--output_file` | Auto-generated | Output file path |
| `--localhost` | `http://localhost:8000` | Julia environment server URL |
| `--max_seq_length` | `2000` | Maximum sequence length |
| `--temperature` | `0.8` | Sampling temperature |
| `--top_p` | `0.95` | Nucleus sampling top_p |
| `--max_tokens` | `1024` | Maximum tokens to generate |
| `--num_samples` | `None` (all) | Number of samples to evaluate |

## Output

The script produces two files:

### 1. Results Parquet File
Contains detailed results for each test case:
- `task_id`: Unique identifier for the task
- `julia_code`: Generated Julia code
- `julia_test`: Test code used for evaluation
- `reward`: Reward score (0.0 to 1.0)
- `passed`: Whether the test passed (boolean)
- `error`: Error message if any
- `observation`: Environment observation

### 2. Summary JSON File
Contains overall metrics:
```json
{
  "model_name": "meta-llama/Meta-Llama-3.1-8B-Instruct",
  "lora_path": null,
  "test_file": "julia_testset.parquet",
  "num_examples": 100,
  "total_passed": 45,
  "pass_rate": 45.0,
  "avg_reward": 0.6234,
  "temperature": 0.8,
  "top_p": 0.95,
  "max_tokens": 1024
}
```

## Evaluation Metrics

- **Pass Rate**: Percentage of test cases where the generated code passed all tests (reward > 0.5)
- **Average Reward**: Mean reward score across all examples (0.0 to 1.0)

## Troubleshooting

### Julia Environment Connection Error

If you see:
```
Warning: Could not connect to Julia environment
```

Make sure the Julia environment server is running:
```bash
docker ps | grep julia-env
```

If not running:
```bash
docker run -d -p 8000:8000 --name julia-env-server julia-env:latest
```

### Import Errors

If you get import errors for `envs.julia_env`, make sure:
1. OpenEnv is in the correct location (`../OpenEnv` relative to this directory)
2. The path is added to `sys.path` correctly in the script

### Memory Issues

If you run out of GPU memory, try:
- Reducing `--max_seq_length`
- Using a smaller model
- Reducing batch processing (the script processes one at a time by default)

## Example Workflow

```bash
# 1. Start Julia environment
docker run -d -p 8000:8000 --name julia-env-server julia-env:latest

# 2. Run baseline evaluation
python eval.py \
    --model_name "meta-llama/Meta-Llama-3.1-8B-Instruct" \
    --test_file "julia_testset.parquet" \
    --num_samples 10

# 3. Check results
cat eval_results_meta_llama_Meta_Llama_3.1_8B_Instruct_baseline_summary.json

# 4. Run full evaluation after confirming it works
python eval.py \
    --model_name "meta-llama/Meta-Llama-3.1-8B-Instruct" \
    --test_file "julia_testset.parquet"
```
