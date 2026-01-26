#!/usr/bin/env python3
"""
Reproduce Julia environment failures using actual problematic code from training logs.

This script extracts the exact code that caused failures and reproduces the issue.

Usage:
    python -m apps.openenv.reproduce_julia_failures --log 0126llama8b_julia_log_off1
"""

import asyncio
import subprocess
import time
import sys
from dataclasses import dataclass
from typing import List, Optional


# Problematic code samples extracted from the training log
# These are the actual `while true` infinite loop samples that caused issues

PROBLEMATIC_SAMPLES = [
    # Sample 347: while true with inefficient divisor counting
    {
        "name": "triangular_divisors_v1",
        "description": "while true loop with sqrt-based divisor counting",
        "code": '''
function find_triangular_number_with_divisors(divisor_count_threshold::Int64)
    n = 1
    while true
        triangular_number = n * (n + 1) / 2
        sqrt_threshold = sqrt(triangular_number)
        divisor_count = Int(0)
        for i = 1:Int(sqrt_threshold)
            if triangular_number % i == 0
                divisor_count += 2
            end
        end
        if divisor_count > divisor_count_threshold
            return triangular_number
        end
        n += 1
    end
end
''',
        "test_code": "@test find_triangular_number_with_divisors(5) > 0"
    },

    # Sample 348: while true with helper function
    {
        "name": "triangular_divisors_v2",
        "description": "while true loop calling count_divisors helper",
        "code": '''
function find_triangular_number_with_divisors(divisor_count_threshold::Int64)
    n = 1
    while true
        triangular_number = n * (n + 1) / 2
        divisors = count_divisors(Int(triangular_number))
        if divisors > divisor_count_threshold
            return triangular_number
        end
        n += 1
    end
end

function count_divisors(n::Int64)
    count = 0
    for i = 1:n
        if n % i == 0
            count += 1
        end
    end
    return count
end
''',
        "test_code": "@test find_triangular_number_with_divisors(5) > 0"
    },

    # Sample 350: Extremely inefficient - O(n^2) per triangular number
    {
        "name": "triangular_divisors_v3_very_slow",
        "description": "VERY SLOW: while true with O(n^2) divisor check per number",
        "code": '''
function find_triangular_number_with_divisors(divisor_count_threshold::Int64)
    n = 1
    while true
        num_divisors = 0
        triangular_num = div(n * (n + 1), 2)
        for i = 1:Int(triangular_num)
            if triangular_num % i == 0
                num_divisors += 1
            end
        end
        if num_divisors > divisor_count_threshold
            return triangular_num
        end
        n += 1
    end
end
''',
        "test_code": "@test find_triangular_number_with_divisors(5) > 0"
    },

    # Simple infinite loop for baseline
    {
        "name": "simple_infinite_loop",
        "description": "Simple while true with no exit",
        "code": '''
function infinite()
    x = 0
    while true
        x += 1
    end
    return x
end
''',
        "test_code": "@test infinite() > 0"
    },

    # Long-running computation (not infinite, but very slow)
    {
        "name": "slow_computation",
        "description": "Slow but finite computation",
        "code": '''
function slow_sum(n::Int64)
    total = 0
    for i = 1:n
        for j = 1:i
            total += 1
        end
    end
    return total
end
''',
        "test_code": "@test slow_sum(10000) > 0"  # O(n^2) = 100M iterations
    },

    # Normal fast code for comparison
    {
        "name": "fast_simple",
        "description": "Fast simple function (baseline)",
        "code": '''
function fast_sum(n::Int64)
    return div(n * (n + 1), 2)
end
''',
        "test_code": "@test fast_sum(100) == 5050"
    },
]


@dataclass
class TestResult:
    name: str
    success: bool
    duration_s: float
    timed_out: bool = False
    connection_error: bool = False
    error: Optional[str] = None


class JuliaFailureReproducer:
    """Reproduce Julia environment failures."""

    def __init__(
        self,
        port: int = 8000,
        timeout_s: float = 20.0,
        use_existing: bool = False,
        docker_image: str = "julia-env:latest",
        num_workers: int = 16,
    ):
        self.port = port
        self.timeout_s = timeout_s
        self.use_existing = use_existing
        self.docker_image = docker_image
        self.num_workers = num_workers
        self.container_id = None

    async def setup(self):
        """Start container if needed."""
        if self.use_existing:
            print(f"[SETUP] Using existing container on port {self.port}")
            return

        print(f"[SETUP] Starting Julia container...")
        env_vars = [
            "-e", f"PORT={self.port}",
            "-e", f"JULIA_MAX_WORKERS={self.num_workers}",
            "-e", "JULIA_EXECUTION_TIMEOUT=15",
        ]

        cmd = [
            "docker", "run", "-d",
            "-p", f"{self.port}:8000",
            "--memory", "4g",
            *env_vars,
            self.docker_image
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to start container: {result.stderr}")

        self.container_id = result.stdout.strip()
        print(f"[SETUP] Container: {self.container_id[:12]}")

        # Wait for ready
        await self._wait_for_ready()
        print(f"[SETUP] Container ready!")

    async def _wait_for_ready(self, timeout_s: float = 120):
        """Wait for container to be ready."""
        from openenv import GenericEnvClient

        start = time.time()
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
            except:
                await asyncio.sleep(2)
        raise RuntimeError("Container not ready")

    async def teardown(self):
        """Stop container if we started it."""
        if self.container_id:
            print(f"[TEARDOWN] Stopping container...")
            subprocess.run(["docker", "stop", self.container_id], capture_output=True)
            subprocess.run(["docker", "rm", self.container_id], capture_output=True)

    def _create_client(self):
        """Create a new client."""
        from openenv import GenericEnvClient

        client = GenericEnvClient(
            base_url=f"http://localhost:{self.port}",
            connect_timeout_s=10.0,
            message_timeout_s=self.timeout_s,
        )
        client.connect()
        client.reset()
        return client

    async def test_single_sample(self, sample: dict, client=None) -> TestResult:
        """Test a single code sample."""
        from openenv import GenericAction

        name = sample["name"]
        code = sample["code"]
        test_code = sample["test_code"]

        own_client = client is None
        if own_client:
            try:
                client = self._create_client()
            except Exception as e:
                return TestResult(
                    name=name,
                    success=False,
                    duration_s=0,
                    connection_error=True,
                    error=str(e)
                )

        start = time.time()
        try:
            action = GenericAction(core_code=code, test_code=test_code)

            result = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None, lambda: client.step(dict(action))
                ),
                timeout=self.timeout_s
            )

            duration = time.time() - start
            return TestResult(name=name, success=True, duration_s=duration)

        except asyncio.TimeoutError:
            return TestResult(
                name=name,
                success=False,
                duration_s=time.time() - start,
                timed_out=True,
                error="timeout"
            )

        except Exception as e:
            error_str = str(e).lower()
            is_conn_error = any(kw in error_str for kw in ['connection', 'websocket', 'closed'])
            return TestResult(
                name=name,
                success=False,
                duration_s=time.time() - start,
                connection_error=is_conn_error,
                error=str(e)[:100]
            )

        finally:
            if own_client:
                try:
                    client.close()
                except:
                    pass

    async def run_sequential_test(self) -> List[TestResult]:
        """Run samples sequentially with single connection."""
        print(f"\n{'='*60}")
        print("TEST 1: Sequential execution (single connection)")
        print(f"{'='*60}")

        results = []
        client = self._create_client()

        for sample in PROBLEMATIC_SAMPLES:
            print(f"\n[{sample['name']}] {sample['description']}")
            print(f"  Executing... (timeout: {self.timeout_s}s)")

            result = await self.test_single_sample(sample, client)
            results.append(result)

            if result.success:
                print(f"  ✓ SUCCESS in {result.duration_s:.2f}s")
            elif result.timed_out:
                print(f"  ✗ TIMEOUT after {result.duration_s:.2f}s")
                print(f"    Waiting 5s for worker recovery...")
                await asyncio.sleep(5)
            elif result.connection_error:
                print(f"  ✗ CONNECTION ERROR: {result.error}")
                print(f"    Reconnecting...")
                try:
                    client.close()
                except:
                    pass
                await asyncio.sleep(3)
                try:
                    client = self._create_client()
                    print(f"    Reconnected")
                except Exception as e:
                    print(f"    Reconnection failed: {e}")
                    break
            else:
                print(f"  ✗ ERROR: {result.error}")

        try:
            client.close()
        except:
            pass

        return results

    async def run_concurrent_test(self, num_connections: int = 4) -> List[TestResult]:
        """Run problematic samples concurrently (like training)."""
        print(f"\n{'='*60}")
        print(f"TEST 2: Concurrent execution ({num_connections} connections)")
        print(f"{'='*60}")

        # Use infinite loop samples only for this test
        infinite_samples = [s for s in PROBLEMATIC_SAMPLES if "infinite" in s["name"] or "v3" in s["name"]]
        if not infinite_samples:
            infinite_samples = PROBLEMATIC_SAMPLES[:3]

        # Multiply samples to have more concurrent requests
        test_samples = infinite_samples * num_connections

        print(f"Running {len(test_samples)} requests across {num_connections} connections...")

        async def worker(samples):
            results = []
            try:
                client = self._create_client()
                for sample in samples:
                    result = await self.test_single_sample(sample, client)
                    results.append(result)
                    if result.timed_out:
                        await asyncio.sleep(2)
                    elif result.connection_error:
                        break
                client.close()
            except Exception as e:
                print(f"  Worker failed: {e}")
            return results

        # Split samples across workers
        chunks = [test_samples[i::num_connections] for i in range(num_connections)]

        all_results = await asyncio.gather(*[worker(chunk) for chunk in chunks])

        results = []
        for worker_results in all_results:
            results.extend(worker_results)

        return results

    async def run_burst_test(self) -> List[TestResult]:
        """Simulate training batch: burst of concurrent infinite loop code."""
        print(f"\n{'='*60}")
        print("TEST 3: Burst of infinite loops (simulates problematic training batch)")
        print(f"{'='*60}")

        # Create 8 concurrent requests with infinite loop code
        burst_size = 8
        infinite_sample = next((s for s in PROBLEMATIC_SAMPLES if s["name"] == "simple_infinite_loop"), PROBLEMATIC_SAMPLES[0])

        print(f"Sending burst of {burst_size} infinite loop executions...")

        async def execute_one(idx):
            try:
                client = self._create_client()
                result = await self.test_single_sample(infinite_sample, client)
                result.name = f"{result.name}_{idx}"
                client.close()
                return result
            except Exception as e:
                return TestResult(
                    name=f"infinite_{idx}",
                    success=False,
                    duration_s=0,
                    connection_error=True,
                    error=str(e)
                )

        results = await asyncio.gather(*[execute_one(i) for i in range(burst_size)])

        return list(results)

    async def run_recovery_test(self) -> List[TestResult]:
        """Test if environment recovers after timeouts."""
        print(f"\n{'='*60}")
        print("TEST 4: Recovery after timeouts")
        print(f"{'='*60}")

        results = []

        # Step 1: Normal request
        print("\n[Step 1] Normal request before stress...")
        fast_sample = next((s for s in PROBLEMATIC_SAMPLES if s["name"] == "fast_simple"), PROBLEMATIC_SAMPLES[-1])
        client = self._create_client()
        result = await self.test_single_sample(fast_sample, client)
        results.append(result)
        print(f"  {'✓ SUCCESS' if result.success else '✗ FAILED'} in {result.duration_s:.2f}s")

        # Step 2: Trigger timeouts
        print("\n[Step 2] Triggering multiple timeouts...")
        infinite_sample = next((s for s in PROBLEMATIC_SAMPLES if s["name"] == "simple_infinite_loop"), PROBLEMATIC_SAMPLES[0])

        for i in range(3):
            result = await self.test_single_sample(infinite_sample, client)
            results.append(result)
            status = "TIMEOUT" if result.timed_out else ("CONN_ERR" if result.connection_error else "SUCCESS")
            print(f"  Attempt {i+1}: {status}")

            if result.connection_error:
                print(f"    Connection error, trying to reconnect...")
                try:
                    client.close()
                except:
                    pass
                await asyncio.sleep(3)
                try:
                    client = self._create_client()
                except:
                    print(f"    Reconnection failed")
                    break

        # Step 3: Wait for recovery
        print("\n[Step 3] Waiting 10s for worker recovery...")
        await asyncio.sleep(10)

        # Step 4: Try normal request again
        print("\n[Step 4] Normal request after recovery...")
        try:
            client = self._create_client()
            result = await self.test_single_sample(fast_sample, client)
            results.append(result)
            print(f"  {'✓ SUCCESS' if result.success else '✗ FAILED'} in {result.duration_s:.2f}s")
            client.close()
        except Exception as e:
            print(f"  ✗ FAILED to connect: {e}")
            results.append(TestResult(name="recovery_test", success=False, duration_s=0, error=str(e)))

        return results

    async def run_all_tests(self):
        """Run all tests."""
        print("\n" + "="*70)
        print("JULIA ENVIRONMENT FAILURE REPRODUCTION")
        print("="*70)

        try:
            await self.setup()

            all_results = []

            results = await self.run_sequential_test()
            all_results.extend(results)

            results = await self.run_concurrent_test(num_connections=4)
            all_results.extend(results)

            results = await self.run_burst_test()
            all_results.extend(results)

            results = await self.run_recovery_test()
            all_results.extend(results)

            # Summary
            print("\n" + "="*70)
            print("SUMMARY")
            print("="*70)

            total = len(all_results)
            success = sum(1 for r in all_results if r.success)
            timeouts = sum(1 for r in all_results if r.timed_out)
            conn_errors = sum(1 for r in all_results if r.connection_error)

            print(f"Total executions:   {total}")
            print(f"Successful:         {success} ({success/total*100:.1f}%)")
            print(f"Timeouts:           {timeouts}")
            print(f"Connection errors:  {conn_errors}")

            if conn_errors > 0:
                print("\n[!] CONNECTION ERRORS indicate the environment is not recovering properly")
                print("    after timeouts. The Julia process pool may be exhausted.")

        finally:
            await self.teardown()


async def main():
    import argparse
    parser = argparse.ArgumentParser(description="Reproduce Julia environment failures")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--use-existing", action="store_true")
    parser.add_argument("--image", type=str, default="julia-env:latest")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    reproducer = JuliaFailureReproducer(
        port=args.port,
        timeout_s=args.timeout,
        use_existing=args.use_existing,
        docker_image=args.image,
        num_workers=args.workers,
    )

    await reproducer.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())
