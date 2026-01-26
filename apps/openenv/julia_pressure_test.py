#!/usr/bin/env python3
"""
Julia Environment Pressure Test

This script tests the Julia environment Docker container under various conditions
to identify stability issues, connection problems, and performance bottlenecks.

Usage:
    python -m apps.openenv.julia_pressure_test

Tests:
    1. Basic connectivity - Can we connect and execute simple code?
    2. Concurrent load - Multiple simultaneous requests
    3. Timeout behavior - Code that runs for varying durations
    4. Infinite loop handling - Does the timeout kill stuck workers?
    5. Connection stability - Long-running test with many requests
    6. Reconnection after failure - Can we recover from errors?
"""

import asyncio
import time
import statistics
import subprocess
import sys
from dataclasses import dataclass, field
from typing import List, Optional
import argparse


@dataclass
class TestResult:
    """Result of a single test execution."""
    success: bool
    duration_s: float
    error: Optional[str] = None
    exit_code: Optional[int] = None
    stdout_len: int = 0
    stderr_len: int = 0


@dataclass
class TestSuiteResult:
    """Aggregated results from a test suite."""
    test_name: str
    total_requests: int
    successful: int
    failed: int
    timeouts: int
    connection_errors: int
    durations: List[float] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        return self.successful / self.total_requests if self.total_requests > 0 else 0

    @property
    def avg_duration(self) -> float:
        return statistics.mean(self.durations) if self.durations else 0

    @property
    def p50_duration(self) -> float:
        return statistics.median(self.durations) if self.durations else 0

    @property
    def p95_duration(self) -> float:
        if len(self.durations) < 2:
            return self.avg_duration
        sorted_d = sorted(self.durations)
        idx = int(len(sorted_d) * 0.95)
        return sorted_d[idx]

    def print_summary(self):
        print(f"\n{'='*60}")
        print(f"Test: {self.test_name}")
        print(f"{'='*60}")
        print(f"Total requests:     {self.total_requests}")
        print(f"Successful:         {self.successful} ({self.success_rate*100:.1f}%)")
        print(f"Failed:             {self.failed}")
        print(f"  - Timeouts:       {self.timeouts}")
        print(f"  - Connection err: {self.connection_errors}")
        if self.durations:
            print(f"Duration (avg):     {self.avg_duration:.3f}s")
            print(f"Duration (p50):     {self.p50_duration:.3f}s")
            print(f"Duration (p95):     {self.p95_duration:.3f}s")
        if self.errors:
            print(f"Sample errors:")
            for err in self.errors[:3]:
                print(f"  - {err[:100]}...")


class JuliaPressureTest:
    """Pressure test for Julia environment."""

    def __init__(
        self,
        docker_image: str = "julia-env:latest",
        port: int = 8000,
        timeout_s: float = 20.0,
        num_workers: int = 16,
    ):
        self.docker_image = docker_image
        self.port = port
        self.timeout_s = timeout_s
        self.num_workers = num_workers
        self.client = None
        self.container_id = None

    async def setup(self):
        """Start the Julia container."""
        print(f"\n[SETUP] Starting Julia container on port {self.port}...")

        # Stop any existing container on this port
        subprocess.run(
            ["docker", "ps", "-q", "--filter", f"publish={self.port}"],
            capture_output=True
        )

        # Start new container
        env_vars = [
            "-e", f"PORT={self.port}",
            "-e", f"JULIA_MAX_WORKERS={self.num_workers}",
            "-e", "JULIA_EXECUTION_TIMEOUT=15",
            "-e", "JULIA_LOG_LEVEL=DEBUG",
        ]

        cmd = [
            "docker", "run", "-d",
            "-p", f"{self.port}:8000",
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
        if self.container_id:
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

    async def _execute_code(self, client, code: str, test_code: str = "@test true") -> TestResult:
        """Execute Julia code and return result."""
        from openenv import GenericAction

        start = time.time()
        try:
            action = GenericAction(core_code=code, test_code=test_code)
            result = client.step(dict(action))
            duration = time.time() - start

            obs = result.observation if hasattr(result, 'observation') else result
            if isinstance(obs, dict):
                return TestResult(
                    success=True,
                    duration_s=duration,
                    exit_code=obs.get('exit_code', 0),
                    stdout_len=len(obs.get('stdout', '')),
                    stderr_len=len(obs.get('stderr', '')),
                )
            else:
                return TestResult(success=True, duration_s=duration)

        except asyncio.TimeoutError:
            return TestResult(
                success=False,
                duration_s=time.time() - start,
                error="timeout"
            )
        except Exception as e:
            error_str = str(e).lower()
            return TestResult(
                success=False,
                duration_s=time.time() - start,
                error=str(e)[:200]
            )

    # =========================================================================
    # Test Cases
    # =========================================================================

    async def test_basic_connectivity(self, num_requests: int = 10) -> TestSuiteResult:
        """Test 1: Basic connectivity with simple code."""
        print(f"\n[TEST] Basic Connectivity ({num_requests} requests)...")

        result = TestSuiteResult(
            test_name="Basic Connectivity",
            total_requests=num_requests,
            successful=0,
            failed=0,
            timeouts=0,
            connection_errors=0,
        )

        client = self._create_client()

        for i in range(num_requests):
            code = f'println("Hello from request {i}")'
            test_code = f'@test 1 + 1 == 2'

            res = await self._execute_code(client, code, test_code)

            if res.success:
                result.successful += 1
                result.durations.append(res.duration_s)
            else:
                result.failed += 1
                if res.error and "timeout" in res.error.lower():
                    result.timeouts += 1
                elif res.error and ("connection" in res.error.lower() or "websocket" in res.error.lower()):
                    result.connection_errors += 1
                result.errors.append(res.error or "unknown")

            # Progress indicator
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{num_requests} ({result.successful} ok, {result.failed} failed)")

        client.close()
        return result

    async def test_concurrent_load(self, num_connections: int = 8, requests_per_conn: int = 10) -> TestSuiteResult:
        """Test 2: Concurrent load with multiple connections."""
        print(f"\n[TEST] Concurrent Load ({num_connections} connections x {requests_per_conn} requests)...")

        result = TestSuiteResult(
            test_name=f"Concurrent Load ({num_connections} conns)",
            total_requests=num_connections * requests_per_conn,
            successful=0,
            failed=0,
            timeouts=0,
            connection_errors=0,
        )

        async def worker(worker_id: int):
            """Worker that sends requests on its own connection."""
            worker_results = []
            try:
                client = self._create_client()
                for i in range(requests_per_conn):
                    code = f'x = {worker_id * 1000 + i}; println("Worker {worker_id}, request {i}: x = $x")'
                    test_code = '@test true'
                    res = await self._execute_code(client, code, test_code)
                    worker_results.append(res)
                client.close()
            except Exception as e:
                # Connection failed entirely
                for _ in range(requests_per_conn - len(worker_results)):
                    worker_results.append(TestResult(
                        success=False,
                        duration_s=0,
                        error=f"Connection failed: {e}"
                    ))
            return worker_results

        # Run all workers concurrently
        all_results = await asyncio.gather(*[worker(i) for i in range(num_connections)])

        # Aggregate results
        for worker_results in all_results:
            for res in worker_results:
                if res.success:
                    result.successful += 1
                    result.durations.append(res.duration_s)
                else:
                    result.failed += 1
                    if res.error and "timeout" in res.error.lower():
                        result.timeouts += 1
                    elif res.error and ("connection" in res.error.lower() or "websocket" in res.error.lower()):
                        result.connection_errors += 1
                    result.errors.append(res.error or "unknown")

        return result

    async def test_timeout_behavior(self, num_tests: int = 5) -> TestSuiteResult:
        """Test 3: Code that runs for varying durations to test timeout."""
        print(f"\n[TEST] Timeout Behavior ({num_tests} tests with varying durations)...")

        result = TestSuiteResult(
            test_name="Timeout Behavior",
            total_requests=num_tests * 4,  # 4 duration tests each
            successful=0,
            failed=0,
            timeouts=0,
            connection_errors=0,
        )

        client = self._create_client()

        # Test different sleep durations
        sleep_durations = [1, 5, 10, 20]  # seconds

        for test_idx in range(num_tests):
            for sleep_s in sleep_durations:
                # Code that sleeps for a specific duration
                code = f'sleep({sleep_s}); println("Slept for {sleep_s} seconds")'
                test_code = '@test true'

                expected_timeout = sleep_s > 15  # Julia timeout is 15s

                print(f"  Test {test_idx+1}/{num_tests}: sleep({sleep_s}s) - expecting {'TIMEOUT' if expected_timeout else 'SUCCESS'}")

                res = await self._execute_code(client, code, test_code)

                if res.success:
                    result.successful += 1
                    result.durations.append(res.duration_s)
                    if expected_timeout:
                        print(f"    WARNING: Expected timeout but got success!")
                else:
                    result.failed += 1
                    if res.error and "timeout" in res.error.lower():
                        result.timeouts += 1
                        if not expected_timeout:
                            print(f"    WARNING: Unexpected timeout!")
                    elif res.error and ("connection" in res.error.lower() or "websocket" in res.error.lower()):
                        result.connection_errors += 1
                        print(f"    ERROR: Connection error: {res.error[:50]}")
                    result.errors.append(res.error or "unknown")

        client.close()
        return result

    async def test_infinite_loop_handling(self, num_tests: int = 5) -> TestSuiteResult:
        """Test 4: Infinite loops - does the timeout kill them?"""
        print(f"\n[TEST] Infinite Loop Handling ({num_tests} tests)...")

        result = TestSuiteResult(
            test_name="Infinite Loop Handling",
            total_requests=num_tests,
            successful=0,
            failed=0,
            timeouts=0,
            connection_errors=0,
        )

        client = self._create_client()

        for i in range(num_tests):
            # Code with infinite loop
            code = '''
            x = 0
            while true
                x += 1
            end
            println("Should never reach here: x = $x")
            '''
            test_code = '@test true'

            print(f"  Test {i+1}/{num_tests}: Infinite loop (expecting timeout)...")
            start = time.time()

            res = await self._execute_code(client, code, test_code)
            elapsed = time.time() - start

            if res.success:
                result.successful += 1
                result.durations.append(res.duration_s)
                print(f"    WARNING: Infinite loop returned success in {elapsed:.1f}s!")
            else:
                result.failed += 1
                if res.error and "timeout" in res.error.lower():
                    result.timeouts += 1
                    print(f"    OK: Timeout after {elapsed:.1f}s (expected)")
                elif res.error and ("connection" in res.error.lower() or "websocket" in res.error.lower()):
                    result.connection_errors += 1
                    print(f"    ERROR: Connection error after {elapsed:.1f}s: {res.error[:50]}")
                else:
                    print(f"    ERROR: Other error after {elapsed:.1f}s: {res.error[:50] if res.error else 'unknown'}")
                result.errors.append(res.error or "unknown")

            # Wait between tests to let worker recover
            print(f"    Waiting 5s for worker recovery...")
            await asyncio.sleep(5)

        client.close()
        return result

    async def test_connection_stability(self, duration_s: float = 60, requests_per_sec: float = 2) -> TestSuiteResult:
        """Test 5: Long-running test to check connection stability."""
        print(f"\n[TEST] Connection Stability ({duration_s}s at {requests_per_sec} req/s)...")

        expected_requests = int(duration_s * requests_per_sec)
        result = TestSuiteResult(
            test_name=f"Connection Stability ({duration_s}s)",
            total_requests=0,
            successful=0,
            failed=0,
            timeouts=0,
            connection_errors=0,
        )

        client = self._create_client()
        start_time = time.time()
        request_interval = 1.0 / requests_per_sec
        request_count = 0

        while time.time() - start_time < duration_s:
            code = f'x = {request_count}; println("Request $x at $(time())")'
            test_code = '@test true'

            res = await self._execute_code(client, code, test_code)
            result.total_requests += 1
            request_count += 1

            if res.success:
                result.successful += 1
                result.durations.append(res.duration_s)
            else:
                result.failed += 1
                if res.error and "timeout" in res.error.lower():
                    result.timeouts += 1
                elif res.error and ("connection" in res.error.lower() or "websocket" in res.error.lower()):
                    result.connection_errors += 1
                    # Try to reconnect
                    print(f"  Connection error at request {request_count}, reconnecting...")
                    try:
                        client.close()
                    except:
                        pass
                    await asyncio.sleep(2)
                    try:
                        client = self._create_client()
                        print(f"  Reconnected successfully")
                    except Exception as e:
                        print(f"  Reconnection failed: {e}")
                        break
                result.errors.append(res.error or "unknown")

            # Progress every 10 seconds
            elapsed = time.time() - start_time
            if request_count % int(requests_per_sec * 10) == 0:
                print(f"  Progress: {elapsed:.0f}s, {request_count} requests ({result.successful} ok, {result.failed} failed)")

            # Maintain request rate
            await asyncio.sleep(request_interval)

        try:
            client.close()
        except:
            pass

        return result

    async def test_reconnection_after_failure(self, num_cycles: int = 5) -> TestSuiteResult:
        """Test 6: Can we reconnect and continue after failures?"""
        print(f"\n[TEST] Reconnection After Failure ({num_cycles} cycles)...")

        result = TestSuiteResult(
            test_name="Reconnection After Failure",
            total_requests=num_cycles * 3,  # 3 requests per cycle
            successful=0,
            failed=0,
            timeouts=0,
            connection_errors=0,
        )

        for cycle in range(num_cycles):
            print(f"\n  Cycle {cycle + 1}/{num_cycles}:")

            # Step 1: Normal request
            print(f"    Step 1: Normal request...")
            client = self._create_client()
            res = await self._execute_code(client, 'println("Normal")', '@test true')
            result.total_requests = result.total_requests  # Already counted
            if res.success:
                result.successful += 1
                result.durations.append(res.duration_s)
                print(f"      OK ({res.duration_s:.2f}s)")
            else:
                result.failed += 1
                print(f"      FAILED: {res.error}")

            # Step 2: Trigger timeout (infinite loop)
            print(f"    Step 2: Trigger timeout (infinite loop)...")
            res = await self._execute_code(client, 'while true; end', '@test true')
            if res.success:
                result.successful += 1
                result.durations.append(res.duration_s)
                print(f"      Unexpected success")
            else:
                result.failed += 1
                if "timeout" in (res.error or "").lower():
                    result.timeouts += 1
                    print(f"      Expected timeout")
                else:
                    result.connection_errors += 1
                    print(f"      Error: {res.error}")

            # Close and wait
            try:
                client.close()
            except:
                pass
            print(f"    Waiting 5s for recovery...")
            await asyncio.sleep(5)

            # Step 3: Try to reconnect and execute
            print(f"    Step 3: Reconnect and normal request...")
            try:
                client = self._create_client()
                res = await self._execute_code(client, 'println("After recovery")', '@test true')
                if res.success:
                    result.successful += 1
                    result.durations.append(res.duration_s)
                    print(f"      OK - Recovery successful! ({res.duration_s:.2f}s)")
                else:
                    result.failed += 1
                    print(f"      FAILED after recovery: {res.error}")
                client.close()
            except Exception as e:
                result.failed += 1
                result.connection_errors += 1
                print(f"      Connection failed: {e}")

        return result

    async def run_all_tests(self):
        """Run all tests and print summary."""
        print("\n" + "=" * 70)
        print("JULIA ENVIRONMENT PRESSURE TEST")
        print("=" * 70)

        results = []

        try:
            await self.setup()

            # Run each test
            results.append(await self.test_basic_connectivity(num_requests=20))
            results.append(await self.test_concurrent_load(num_connections=8, requests_per_conn=5))
            results.append(await self.test_timeout_behavior(num_tests=3))
            results.append(await self.test_infinite_loop_handling(num_tests=3))
            results.append(await self.test_connection_stability(duration_s=30, requests_per_sec=2))
            results.append(await self.test_reconnection_after_failure(num_cycles=3))

        except Exception as e:
            print(f"\n[ERROR] Test failed with exception: {e}")
            import traceback
            traceback.print_exc()

        finally:
            await self.teardown()

        # Print summary
        print("\n" + "=" * 70)
        print("TEST SUMMARY")
        print("=" * 70)

        total_requests = sum(r.total_requests for r in results)
        total_success = sum(r.successful for r in results)
        total_timeouts = sum(r.timeouts for r in results)
        total_conn_errors = sum(r.connection_errors for r in results)

        for r in results:
            r.print_summary()

        print("\n" + "=" * 70)
        print("OVERALL RESULTS")
        print("=" * 70)
        print(f"Total requests:       {total_requests}")
        print(f"Total successful:     {total_success} ({total_success/total_requests*100:.1f}%)")
        print(f"Total timeouts:       {total_timeouts}")
        print(f"Total conn errors:    {total_conn_errors}")

        if total_conn_errors > 0:
            print("\n[!] CONNECTION ERRORS DETECTED - This indicates WebSocket/container stability issues")
        if total_timeouts > total_requests * 0.1:
            print("\n[!] HIGH TIMEOUT RATE - Workers may not be recovering properly")

        return results


async def main():
    parser = argparse.ArgumentParser(description="Julia Environment Pressure Test")
    parser.add_argument("--port", type=int, default=8000, help="Port to use for container")
    parser.add_argument("--image", type=str, default="julia-env:latest", help="Docker image")
    parser.add_argument("--workers", type=int, default=16, help="Number of Julia workers")
    parser.add_argument("--timeout", type=float, default=20.0, help="Request timeout in seconds")
    args = parser.parse_args()

    tester = JuliaPressureTest(
        docker_image=args.image,
        port=args.port,
        timeout_s=args.timeout,
        num_workers=args.workers,
    )

    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())
