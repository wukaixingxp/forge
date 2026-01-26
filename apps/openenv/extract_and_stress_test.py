#!/usr/bin/env python3
"""
Extract LLM-generated Julia code from training logs and stress test the Julia environment.

This script:
1. Extracts LLM-generated Julia code AND test codes from training logs
2. Runs each code sample through the Julia environment with proper test codes
3. Reports success/failure rates and identifies problematic patterns

IMPORTANT: Make sure your training uses the updated julia_utils_generic.py which logs
both the extracted code AND the test code together. The new log format is:

    EXTRACTED JULIA CODE:
    --------------------------------------------------------------------------------
    <llm_generated_code>
    --------------------------------------------------------------------------------
    TEST CODE:
    --------------------------------------------------------------------------------
    <test_code>
    --------------------------------------------------------------------------------
    END OF SAMPLE

Usage:
    # Extract from new format log (has both code and test_code)
    python -m apps.openenv.extract_and_stress_test --log 0126llama8b_julia_log_off1

    # With existing container
    python -m apps.openenv.extract_and_stress_test --log 0126llama8b_julia_log_off1 --use-existing --port 8000

    # Limit samples for quick test
    python -m apps.openenv.extract_and_stress_test --log 0126llama8b_julia_log_off1 --max-samples 100

    # Verbose output - log all raw results from server
    python -m apps.openenv.extract_and_stress_test --log 0126llama8b_julia_log_off1 -v --max-samples 10

    # Save all results to JSON for inspection
    python -m apps.openenv.extract_and_stress_test --log 0126llama8b_julia_log_off1 --results-file results.json

    # Use dataset directly with reference code (for environment validation)
    python -m apps.openenv.extract_and_stress_test --dataset /path/to/julia_trainset.parquet
"""

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple
import statistics


@dataclass
class CodeSample:
    """A single code sample extracted from the log."""
    index: int
    code: str
    test_code: str = "@test true"
    task_id: str = ""
    expected_reward: Optional[float] = None
    line_number: int = 0


@dataclass
class DatasetSample:
    """A sample from the Julia dataset."""
    index: int
    task_id: str
    julia_prompt: str
    julia_code: str  # Reference solution
    julia_test: str  # Test code to evaluate
    first_test_case: str


def load_dataset_samples(dataset_path: str, max_samples: Optional[int] = None) -> Tuple[List[DatasetSample], dict]:
    """Load samples from the Julia dataset parquet file.

    Returns:
        Tuple of (list of DatasetSample, dict mapping task_id to sample)
    """
    import pandas as pd

    print(f"[DATASET] Loading dataset from: {dataset_path}")

    df = pd.read_parquet(dataset_path)
    print(f"[DATASET] Loaded {len(df)} samples with columns: {df.columns.tolist()}")

    samples = []
    task_id_map = {}

    for i, row in df.iterrows():
        if max_samples is not None and len(samples) >= max_samples:
            break

        sample = DatasetSample(
            index=len(samples),
            task_id=row.get('task_id', f'sample_{i}'),
            julia_prompt=row.get('julia_prompt', ''),
            julia_code=row.get('julia_code', ''),
            julia_test=row.get('julia_test', ''),
            first_test_case=row.get('first_test_case', ''),
        )
        samples.append(sample)
        task_id_map[sample.task_id] = sample

    print(f"[DATASET] Created {len(samples)} samples")
    return samples, task_id_map


def create_code_samples_from_dataset(
    dataset_samples: List[DatasetSample],
    use_reference_code: bool = True,
) -> List[CodeSample]:
    """Create CodeSample objects from dataset samples.

    Args:
        dataset_samples: List of DatasetSample from the dataset
        use_reference_code: If True, use julia_code (reference solution).
                           If False, the code field will be empty (for matching with logs).

    Returns:
        List of CodeSample with proper test_code from dataset
    """
    samples = []
    for ds in dataset_samples:
        code = ds.julia_code if use_reference_code else ""
        samples.append(CodeSample(
            index=ds.index,
            code=code,
            test_code=ds.julia_test,  # Use actual test code from dataset!
            task_id=ds.task_id,
        ))
    return samples


@dataclass
class ExecutionResult:
    """Result of executing a code sample."""
    sample_index: int
    success: bool
    duration_s: float
    task_id: str = ""
    exit_code: Optional[int] = None
    error: Optional[str] = None
    timed_out: bool = False
    connection_error: bool = False
    # Julia-specific fields from StepResult
    reward: Optional[float] = None
    tests_passed: Optional[int] = None
    tests_failed: Optional[int] = None
    code_compiles: Optional[bool] = None
    stdout_preview: str = ""
    stderr_preview: str = ""


def extract_code_samples(log_path: str, max_samples: Optional[int] = None) -> List[CodeSample]:
    """Extract LLM-generated Julia code samples from a training log.

    Supports two log formats:
    1. New format (with test code):
       EXTRACTED JULIA CODE:
       --------------------------------------------------------------------------------
       <code>
       --------------------------------------------------------------------------------
       TEST CODE:
       --------------------------------------------------------------------------------
       <test_code>
       --------------------------------------------------------------------------------
       END OF SAMPLE

    2. Old format (code only, uses dummy test):
       EXTRACTED JULIA CODE:
       --------------------------------------------------------------------------------
       <code>
       --------------------------------------------------------------------------------
    """

    print(f"[EXTRACT] Reading log file: {log_path}")

    with open(log_path, 'r', errors='ignore') as f:
        content = f.read()

    samples = []

    # Try new format first: EXTRACTED JULIA CODE + TEST CODE + END OF SAMPLE
    # Pattern matches the full block from "EXTRACTED JULIA CODE:" to "END OF SAMPLE"
    new_pattern = (
        r'EXTRACTED JULIA CODE:\s*-{60,}\s*'
        r'(.*?)\s*-{60,}\s*'
        r'TEST CODE:\s*-{60,}\s*'
        r'(.*?)\s*-{60,}\s*'
        r'END OF SAMPLE'
    )
    new_matches = re.findall(new_pattern, content, re.DOTALL)

    if new_matches:
        print(f"[EXTRACT] Found {len(new_matches)} samples with test code (new format)")

        for i, (code, test_code) in enumerate(new_matches):
            if max_samples is not None and i >= max_samples:
                break

            code = code.strip()
            test_code = test_code.strip()

            if code and len(code) > 10:
                samples.append(CodeSample(
                    index=i,
                    code=code,
                    test_code=test_code if test_code else "@test true",
                ))

        print(f"[EXTRACT] Extracted {len(samples)} valid samples with real test codes")

    else:
        # Fall back to old format (code only)
        print(f"[EXTRACT] No new format samples found, trying old format...")

        old_pattern = r'EXTRACTED JULIA CODE:\s*-{60,}\s*(.*?)\s*-{60,}'
        old_matches = re.findall(old_pattern, content, re.DOTALL)

        print(f"[EXTRACT] Found {len(old_matches)} code samples (old format, no test codes)")
        print(f"[WARNING] Old format detected - using dummy '@test true' for all samples!")
        print(f"[WARNING] Re-run training with updated julia_utils_generic.py to get real test codes")

        for i, code in enumerate(old_matches):
            if max_samples is not None and i >= max_samples:
                break

            code = code.strip()
            if code and len(code) > 10:
                samples.append(CodeSample(
                    index=i,
                    code=code,
                    test_code="@test true",  # Dummy test
                ))

    # Look for potentially problematic patterns in the code
    problematic_patterns = [
        (r'\bwhile\s+true\b', 'while true (infinite loop)'),
        (r'\bwhile\s*\(\s*true\s*\)', 'while(true) (infinite loop)'),
        (r'\bfor\s+\w+\s+in\s+1:Inf\b', 'for in 1:Inf (infinite loop)'),
        (r'\bwhile\s+1\s*[^=]', 'while 1 (infinite loop)'),
        (r'sleep\s*\(\s*\d{2,}\s*\)', 'long sleep'),
        (r'@async\s+while', 'async infinite loop'),
    ]

    problematic_samples = []
    for sample in samples:
        for pattern, desc in problematic_patterns:
            if re.search(pattern, sample.code, re.IGNORECASE):
                print(f"  [!] Sample {sample.index}: Contains {desc}")
                problematic_samples.append((sample.index, desc))
                break

    print(f"[EXTRACT] {len(problematic_samples)} samples contain potentially problematic patterns")

    return samples


def save_samples_to_json(samples: List[CodeSample], output_path: str):
    """Save extracted samples to JSON for later analysis."""
    data = [
        {
            "index": s.index,
            "task_id": s.task_id,
            "code": s.code,
            "test_code": s.test_code,
        }
        for s in samples
    ]

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"[SAVE] Saved {len(samples)} samples to {output_path}")


class JuliaStressTest:
    """Stress test Julia environment with extracted code samples."""

    def __init__(
        self,
        docker_image: str = "julia-env:latest",
        port: int = 8000,
        timeout_s: float = 20.0,
        num_workers: int = 16,
        use_existing: bool = False,
        verbose: bool = False,
    ):
        self.docker_image = docker_image
        self.port = port
        self.timeout_s = timeout_s
        self.num_workers = num_workers
        self.use_existing = use_existing
        self.verbose = verbose
        self.container_id = None
        self.client = None

    async def setup(self):
        """Start the Julia container (unless using existing)."""
        if self.use_existing:
            print(f"[SETUP] Using existing container on port {self.port}")
            await self._wait_for_ready(timeout_s=30)
            return

        print(f"\n[SETUP] Starting Julia container on port {self.port}...")

        # Setup persistent logging with volume mount (same as training code)
        logs_dir = os.path.expanduser("~/julia_container_logs")
        os.makedirs(logs_dir, exist_ok=True)
        print(f"[SETUP] Container logs will be written to: {logs_dir}")

        # Start new container
        env_vars = [
            "-e", f"PORT={self.port}",
            "-e", f"JULIA_MAX_WORKERS={self.num_workers}",
            "-e", "JULIA_EXECUTION_TIMEOUT=15",
            "-e", "JULIA_LOG_LEVEL=DEBUG",  # Enable debug logging
        ]

        cmd = [
            "docker", "run", "-d",
            "-p", f"{self.port}:8000",
            "--memory", "4g",
            "-v", f"{logs_dir}:/tmp/julia_logs",  # Mount logs directory
            *env_vars,
            self.docker_image
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to start container: {result.stderr}")

        self.container_id = result.stdout.strip()
        print(f"[SETUP] Container started: {self.container_id[:12]}")

        # Wait for container to be ready
        print(f"[SETUP] Waiting for container to be ready...")
        await self._wait_for_ready(timeout_s=120)
        print(f"[SETUP] Container is ready!")

    async def _wait_for_ready(self, timeout_s: float = 120):
        """Wait for the container to be ready."""
        from openenv import GenericEnvClient

        start = time.time()
        last_error = None

        while time.time() - start < timeout_s:
            try:
                client = GenericEnvClient(
                    base_url=f"http://localhost:{self.port}",
                    connect_timeout_s=5.0,
                    message_timeout_s=10.0,
                )
                client.connect()
                client.reset()
                client.close()
                return
            except Exception as e:
                last_error = e
                await asyncio.sleep(2)

        raise RuntimeError(f"Container not ready after {timeout_s}s: {last_error}")

    async def teardown(self):
        """Stop and remove the container."""
        if self.container_id and not self.use_existing:
            print(f"\n[TEARDOWN] Stopping container {self.container_id[:12]}...")
            subprocess.run(["docker", "stop", self.container_id], capture_output=True)
            subprocess.run(["docker", "rm", self.container_id], capture_output=True)
            print(f"[TEARDOWN] Container stopped and removed")

    def _create_client(self):
        """Create a new GenericEnvClient."""
        from openenv import GenericEnvClient

        client = GenericEnvClient(
            base_url=f"http://localhost:{self.port}",
            connect_timeout_s=10.0,
            message_timeout_s=self.timeout_s,
        )
        client.connect()
        client.reset()
        return client

    async def execute_sample(self, client, sample: CodeSample, verbose: bool = False) -> ExecutionResult:
        """Execute a single code sample."""
        from openenv import GenericAction

        start = time.time()
        try:
            action = GenericAction(core_code=sample.code, test_code=sample.test_code)

            # Run with timeout
            result = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None, lambda: client.step(dict(action))
                ),
                timeout=self.timeout_s
            )

            duration = time.time() - start

            # Log raw result for debugging
            if verbose:
                print(f"\n[RESULT] Sample {sample.index}:")
                print(f"  Raw result type: {type(result)}")
                print(f"  Raw result: {result}")

            # Extract reward from StepResult
            reward = result.reward if hasattr(result, 'reward') and result.reward is not None else None

            # Extract observation
            obs = result.observation if hasattr(result, 'observation') else result

            if verbose:
                print(f"  Reward: {reward}")
                print(f"  Observation type: {type(obs)}")
                print(f"  Observation: {obs}")

            if isinstance(obs, dict):
                exit_code = obs.get('exit_code', -1)
                tests_passed = obs.get('tests_passed', 0)
                tests_failed = obs.get('tests_failed', 0)
                code_compiles = obs.get('code_compiles', False)
                stdout = str(obs.get('stdout', ''))
                stderr = str(obs.get('stderr', ''))

                if verbose:
                    print(f"  Exit code: {exit_code}")
                    print(f"  Tests passed: {tests_passed}")
                    print(f"  Tests failed: {tests_failed}")
                    print(f"  Code compiles: {code_compiles}")
                    print(f"  Stdout ({len(stdout)} chars): {stdout[:300]}")
                    print(f"  Stderr ({len(stderr)} chars): {stderr[:300]}")

                return ExecutionResult(
                    sample_index=sample.index,
                    task_id=sample.task_id,
                    success=True,
                    duration_s=duration,
                    exit_code=exit_code,
                    reward=reward,
                    tests_passed=tests_passed,
                    tests_failed=tests_failed,
                    code_compiles=code_compiles,
                    stdout_preview=stdout[:1000],
                    stderr_preview=stderr[:1000],
                )
            else:
                if verbose:
                    print(f"  WARNING: obs is not a dict, type={type(obs)}")

                # Try to extract fields from typed observation object
                try:
                    return ExecutionResult(
                        sample_index=sample.index,
                        task_id=sample.task_id,
                        success=True,
                        duration_s=duration,
                        exit_code=getattr(obs, 'exit_code', None),
                        reward=reward,
                        tests_passed=getattr(obs, 'tests_passed', None),
                        tests_failed=getattr(obs, 'tests_failed', None),
                        code_compiles=getattr(obs, 'code_compiles', None),
                        stdout_preview=str(getattr(obs, 'stdout', ''))[:1000],
                        stderr_preview=str(getattr(obs, 'stderr', ''))[:1000],
                    )
                except Exception:
                    return ExecutionResult(
                        sample_index=sample.index,
                        task_id=sample.task_id,
                        success=True,
                        duration_s=duration,
                        reward=reward,
                    )

        except asyncio.TimeoutError:
            if verbose:
                print(f"\n[TIMEOUT] Sample {sample.index} timed out after {time.time() - start:.1f}s")
            return ExecutionResult(
                sample_index=sample.index,
                task_id=sample.task_id,
                success=False,
                duration_s=time.time() - start,
                error="timeout",
                timed_out=True,
            )

        except Exception as e:
            error_str = str(e).lower()
            is_conn_error = any(kw in error_str for kw in ['connection', 'websocket', 'closed', 'refused'])

            if verbose:
                print(f"\n[ERROR] Sample {sample.index}: {e}")

            return ExecutionResult(
                sample_index=sample.index,
                task_id=sample.task_id,
                success=False,
                duration_s=time.time() - start,
                error=str(e)[:200],
                connection_error=is_conn_error,
            )

    async def run_sequential_test(
        self,
        samples: List[CodeSample],
        stop_on_connection_error: bool = True,
    ) -> Tuple[List[ExecutionResult], dict]:
        """Run samples sequentially with a single connection."""

        print(f"\n[TEST] Sequential test with {len(samples)} samples...")

        results = []
        stats = {
            "total": len(samples),
            "successful": 0,
            "failed": 0,
            "timeouts": 0,
            "connection_errors": 0,
            "durations": [],
        }

        client = self._create_client()
        reconnect_count = 0

        for i, sample in enumerate(samples):
            result = await self.execute_sample(client, sample, verbose=self.verbose)
            results.append(result)

            if result.success:
                stats["successful"] += 1
                stats["durations"].append(result.duration_s)
            else:
                stats["failed"] += 1
                if result.timed_out:
                    stats["timeouts"] += 1
                    print(f"  [{i+1}/{len(samples)}] TIMEOUT: Sample {sample.index} ({result.duration_s:.1f}s)")
                    # Wait for worker to recover
                    print(f"    Waiting 3s for worker recovery...")
                    await asyncio.sleep(3)
                elif result.connection_error:
                    stats["connection_errors"] += 1
                    print(f"  [{i+1}/{len(samples)}] CONNECTION ERROR: {result.error[:50]}")

                    if stop_on_connection_error:
                        print(f"  Stopping test due to connection error")
                        break

                    # Try to reconnect
                    try:
                        client.close()
                    except:
                        pass

                    print(f"    Attempting reconnection...")
                    await asyncio.sleep(5)

                    try:
                        client = self._create_client()
                        reconnect_count += 1
                        print(f"    Reconnected (attempt {reconnect_count})")
                    except Exception as e:
                        print(f"    Reconnection failed: {e}")
                        break

            # Progress every 10 samples
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{len(samples)} "
                      f"({stats['successful']} ok, {stats['timeouts']} timeouts, {stats['connection_errors']} conn errors)")

        try:
            client.close()
        except:
            pass

        return results, stats

    async def run_concurrent_test(
        self,
        samples: List[CodeSample],
        num_connections: int = 4,
    ) -> Tuple[List[ExecutionResult], dict]:
        """Run samples concurrently with multiple connections (like training)."""

        print(f"\n[TEST] Concurrent test with {len(samples)} samples across {num_connections} connections...")

        stats = {
            "total": len(samples),
            "successful": 0,
            "failed": 0,
            "timeouts": 0,
            "connection_errors": 0,
            "durations": [],
        }

        # Split samples across workers
        sample_chunks = [samples[i::num_connections] for i in range(num_connections)]
        progress_lock = asyncio.Lock()
        progress = {"completed": 0, "start_time": time.time()}

        async def worker(worker_id: int, chunk: List[CodeSample]) -> List[ExecutionResult]:
            """Worker that processes its chunk of samples."""
            results = []
            try:
                client = self._create_client()
                for sample in chunk:
                    result = await self.execute_sample(client, sample, verbose=self.verbose)
                    results.append(result)

                    # Update progress
                    async with progress_lock:
                        progress["completed"] += 1
                        completed = progress["completed"]
                        if completed % 100 == 0 or completed == len(samples):
                            elapsed = time.time() - progress["start_time"]
                            rate = completed / elapsed if elapsed > 0 else 0
                            eta = (len(samples) - completed) / rate if rate > 0 else 0
                            success_count = sum(1 for r in results if r.success)
                            timeout_count = sum(1 for r in results if r.timed_out)
                            print(f"  Progress: {completed}/{len(samples)} ({rate:.1f}/s, ETA: {eta:.0f}s) "
                                  f"- Worker {worker_id}: {success_count} ok, {timeout_count} timeouts")

                    if result.timed_out:
                        # Small delay after timeout
                        await asyncio.sleep(2)
                    elif result.connection_error:
                        # Try reconnect
                        try:
                            client.close()
                        except:
                            pass
                        await asyncio.sleep(3)
                        try:
                            client = self._create_client()
                        except:
                            break

                client.close()
            except Exception as e:
                print(f"  Worker {worker_id} failed: {e}")

            return results

        # Run workers concurrently
        all_results = await asyncio.gather(*[
            worker(i, chunk) for i, chunk in enumerate(sample_chunks)
        ])

        # Aggregate results
        results = []
        for worker_results in all_results:
            for result in worker_results:
                results.append(result)
                if result.success:
                    stats["successful"] += 1
                    stats["durations"].append(result.duration_s)
                    # Track Julia-specific metrics
                    if result.reward is not None:
                        stats.setdefault("rewards", []).append(result.reward)
                    if result.tests_passed is not None:
                        stats.setdefault("tests_passed", []).append(result.tests_passed)
                    if result.tests_failed is not None:
                        stats.setdefault("tests_failed", []).append(result.tests_failed)
                    if result.code_compiles is not None:
                        stats.setdefault("compiles_count", 0)
                        if result.code_compiles:
                            stats["compiles_count"] += 1
                else:
                    stats["failed"] += 1
                    if result.timed_out:
                        stats["timeouts"] += 1
                    elif result.connection_error:
                        stats["connection_errors"] += 1

        return results, stats

    async def run_burst_test(
        self,
        samples: List[CodeSample],
        burst_size: int = 8,
        num_bursts: int = 5,
    ) -> Tuple[List[ExecutionResult], dict]:
        """Run bursts of concurrent requests (simulates training batches)."""

        print(f"\n[TEST] Burst test: {num_bursts} bursts of {burst_size} concurrent requests...")

        all_results = []
        stats = {
            "total": 0,
            "successful": 0,
            "failed": 0,
            "timeouts": 0,
            "connection_errors": 0,
            "durations": [],
        }

        sample_idx = 0

        for burst_num in range(num_bursts):
            # Get samples for this burst
            burst_samples = samples[sample_idx:sample_idx + burst_size]
            if not burst_samples:
                break
            sample_idx += burst_size

            print(f"  Burst {burst_num + 1}/{num_bursts}: {len(burst_samples)} samples...")

            # Create connections for this burst
            async def execute_one(sample):
                client = None
                try:
                    client = self._create_client()
                    result = await self.execute_sample(client, sample, verbose=self.verbose)
                    return result
                except Exception as e:
                    return ExecutionResult(
                        sample_index=sample.index,
                        success=False,
                        duration_s=0,
                        error=str(e)[:200],
                        connection_error=True,
                    )
                finally:
                    if client:
                        try:
                            client.close()
                        except:
                            pass

            # Run burst concurrently
            burst_results = await asyncio.gather(*[
                execute_one(s) for s in burst_samples
            ])

            # Aggregate
            burst_timeouts = 0
            burst_conn_errors = 0
            for result in burst_results:
                all_results.append(result)
                stats["total"] += 1
                if result.success:
                    stats["successful"] += 1
                    stats["durations"].append(result.duration_s)
                else:
                    stats["failed"] += 1
                    if result.timed_out:
                        stats["timeouts"] += 1
                        burst_timeouts += 1
                    elif result.connection_error:
                        stats["connection_errors"] += 1
                        burst_conn_errors += 1

            print(f"    Results: {len(burst_samples) - burst_timeouts - burst_conn_errors} ok, "
                  f"{burst_timeouts} timeouts, {burst_conn_errors} conn errors")

            # Wait between bursts
            if burst_num < num_bursts - 1:
                print(f"    Waiting 5s before next burst...")
                await asyncio.sleep(5)

        return all_results, stats


def print_stats(stats: dict, test_name: str):
    """Print test statistics."""
    print(f"\n{'='*60}")
    print(f"Results: {test_name}")
    print(f"{'='*60}")
    print(f"Total:            {stats['total']}")
    print(f"Successful:       {stats['successful']} ({stats['successful']/stats['total']*100:.1f}%)" if stats['total'] > 0 else "")
    print(f"Failed:           {stats['failed']}")
    print(f"  - Timeouts:     {stats['timeouts']}")
    print(f"  - Conn errors:  {stats['connection_errors']}")

    if stats['durations']:
        print(f"Duration (avg):   {statistics.mean(stats['durations']):.3f}s")
        print(f"Duration (p50):   {statistics.median(stats['durations']):.3f}s")
        if len(stats['durations']) >= 2:
            sorted_d = sorted(stats['durations'])
            p95_idx = int(len(sorted_d) * 0.95)
            print(f"Duration (p95):   {sorted_d[p95_idx]:.3f}s")

    # Julia-specific metrics
    if 'rewards' in stats and stats['rewards']:
        rewards = stats['rewards']
        print(f"\n--- Julia Execution Results ---")
        print(f"Samples with rewards: {len(rewards)}")
        print(f"Reward (avg):     {statistics.mean(rewards):.3f}")
        print(f"Reward (min):     {min(rewards):.3f}")
        print(f"Reward (max):     {max(rewards):.3f}")
        non_zero = sum(1 for r in rewards if r > 0)
        print(f"Non-zero rewards: {non_zero} ({100*non_zero/len(rewards):.1f}%)")

    if 'tests_passed' in stats and stats['tests_passed']:
        total_passed = sum(stats['tests_passed'])
        total_failed = sum(stats.get('tests_failed', [0]))
        total_tests = total_passed + total_failed
        print(f"Tests passed:     {total_passed}")
        print(f"Tests failed:     {total_failed}")
        if total_tests > 0:
            print(f"Test pass rate:   {100*total_passed/total_tests:.1f}%")

    if 'compiles_count' in stats:
        compiles = stats['compiles_count']
        print(f"Code compiles:    {compiles} ({100*compiles/stats['successful']:.1f}% of successful)" if stats['successful'] > 0 else "")


async def main():
    parser = argparse.ArgumentParser(description="Extract and stress test Julia code samples")
    parser.add_argument("--log", type=str, help="Path to training log file (extracts LLM-generated code)")
    parser.add_argument("--dataset", type=str, help="Path to Julia dataset parquet file (uses reference code with real tests)")
    parser.add_argument("--port", type=int, default=8000, help="Port for Julia container")
    parser.add_argument("--image", type=str, default="julia-env:latest", help="Docker image")
    parser.add_argument("--workers", type=int, default=16, help="Number of Julia workers")
    parser.add_argument("--timeout", type=float, default=20.0, help="Request timeout in seconds")
    parser.add_argument("--max-samples", type=int, default=0, help="Maximum samples to extract (0 = all)")
    parser.add_argument("--use-existing", action="store_true", help="Use existing container")
    parser.add_argument("--test-type", choices=["sequential", "concurrent", "burst", "full", "all"],
                        default="full", help="Type of test to run (full = all samples through N workers)")
    parser.add_argument("--num-connections", type=int, default=16, help="Number of concurrent connections for full/concurrent test")
    parser.add_argument("--save-samples", type=str, help="Save extracted samples to JSON file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output - log all results from server")
    parser.add_argument("--results-file", type=str, help="Save all execution results to JSON file")
    args = parser.parse_args()

    # Validate arguments
    if not args.log and not args.dataset:
        parser.error("Either --log or --dataset is required")

    max_samples = args.max_samples if args.max_samples > 0 else None

    # Load samples based on mode
    if args.dataset:
        # Load from dataset - uses reference code with proper test codes
        print(f"[MODE] Loading from dataset with reference solutions and real tests")
        dataset_samples, task_id_map = load_dataset_samples(args.dataset, max_samples=max_samples)
        samples = create_code_samples_from_dataset(dataset_samples, use_reference_code=True)

        if args.log:
            # Also have log file - try to match LLM-generated code with dataset test codes
            print(f"[MODE] Also extracting LLM-generated code from log to match with dataset tests")
            log_samples = extract_code_samples(args.log, max_samples=max_samples)

            # Replace reference code with LLM-generated code where indices match
            for i, log_sample in enumerate(log_samples):
                if i < len(samples):
                    # Keep the test_code from dataset, but use code from log
                    samples[i].code = log_sample.code
                    print(f"  Matched sample {i}: using LLM code with dataset test")

            print(f"[MODE] Matched {min(len(log_samples), len(samples))} samples")

    else:
        # Log only mode - extract from log (WARNING: no proper test codes!)
        print(f"[MODE] Extracting from log file only (WARNING: using dummy tests!)")
        print(f"[WARNING] Without --dataset, tests will use '@test true' which always passes!")
        print(f"[WARNING] Use --dataset to get real test results")
        samples = extract_code_samples(args.log, max_samples=max_samples)

    if not samples:
        print("[ERROR] No code samples found")
        return

    print(f"\n[INFO] Total samples to test: {len(samples)}")
    if samples:
        print(f"[INFO] Sample 0 test_code preview: {samples[0].test_code[:200]}...")

    # Optionally save samples
    if args.save_samples:
        save_samples_to_json(samples, args.save_samples)

    # Run stress tests
    tester = JuliaStressTest(
        docker_image=args.image,
        port=args.port,
        timeout_s=args.timeout,
        num_workers=args.workers,
        use_existing=args.use_existing,
        verbose=args.verbose,
    )

    all_results = []

    try:
        await tester.setup()

        if args.test_type == "full":
            # Run ALL samples through N concurrent workers (mimics training)
            print(f"\n[FULL TEST] Running all {len(samples)} samples through {args.num_connections} workers...")
            results, stats = await tester.run_concurrent_test(samples, num_connections=args.num_connections)
            all_results.extend(results)
            print_stats(stats, f"Full Test ({len(samples)} samples, {args.num_connections} workers)")

        if args.test_type in ["sequential", "all"]:
            # Take first 50 samples for sequential test
            seq_samples = samples[:50]
            results, stats = await tester.run_sequential_test(seq_samples)
            all_results.extend(results)
            print_stats(stats, "Sequential Test")

        if args.test_type in ["concurrent", "all"]:
            # Take next 50 samples for concurrent test
            conc_samples = samples[50:100] if len(samples) > 50 else samples[:50]
            results, stats = await tester.run_concurrent_test(conc_samples, num_connections=args.num_connections)
            all_results.extend(results)
            print_stats(stats, f"Concurrent Test ({args.num_connections} connections)")

        if args.test_type in ["burst", "all"]:
            # Take remaining samples for burst test
            burst_samples = samples[100:] if len(samples) > 100 else samples
            results, stats = await tester.run_burst_test(burst_samples, burst_size=8, num_bursts=5)
            all_results.extend(results)
            print_stats(stats, "Burst Test (8 concurrent x 5 bursts)")

    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Test interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await tester.teardown()

    # Save results to file
    if args.results_file and all_results:
        results_data = [
            {
                "sample_index": r.sample_index,
                "task_id": r.task_id,
                "success": r.success,
                "duration_s": r.duration_s,
                "exit_code": r.exit_code,
                "error": r.error,
                "timed_out": r.timed_out,
                "connection_error": r.connection_error,
                # Julia-specific fields
                "reward": r.reward,
                "tests_passed": r.tests_passed,
                "tests_failed": r.tests_failed,
                "code_compiles": r.code_compiles,
                "stdout_preview": r.stdout_preview,
                "stderr_preview": r.stderr_preview,
            }
            for r in all_results
        ]
        with open(args.results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        print(f"\n[SAVE] Saved {len(all_results)} results to {args.results_file}")


if __name__ == "__main__":
    asyncio.run(main())
