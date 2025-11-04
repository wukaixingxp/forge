# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import logging
import subprocess
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from forge.controller import ForgeActor

from monarch.actor import endpoint

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class PodmanPythonCoder(ForgeActor):
    """A sandboxed code execution environment using podman containers.

    This actor provides a sandboxed environment for executing Python code
    using podman container technology.

    It automatically manages the entire container lifecycle including image
    pulling, container creation, and cleanup.

    The actor follows a three-stage workflow:
    1. Image Management: Uses podman to pull images from registries
    2. Container Lifecycle: Creates fresh container instances for isolated execution
    3. Code Execution: Safely runs Python code with proper error handling and output capture

    Dependencies:
    - podman: Container engine for pulling images and running containers (must be installed on host)
    - Container images: Accessible via standard container registries

    Args:
        container_image: Container image name to pull (e.g., "python:3.10").
                        Can be any Docker Hub image or custom registry URL.
        container_name: Unique name for the podman container instance. Used for
                        container lifecycle management (create/remove operations).
        max_workers: Maximum number of concurrent subprocess executions (default: 4).

    """

    def __init__(
        self,
        container_image: str = "python:3.10",
        container_name: str = "sandbox",
        max_workers: int = 4,
    ):
        self.container_image = container_image
        self.container_name = container_name
        self._initialized = False
        # Thread pool for running subprocess calls without blocking event loop
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="podman_exec_"
        )

    @endpoint
    async def setup(self):
        logging.debug("Setting up sandboxed actor")
        await self._maybe_create_image()
        self._recreate()

    @endpoint
    async def recreate(self):
        """Recreates the container instance from the base image."""
        self._recreate()

    async def _maybe_create_image(self):
        """Ensure the container image is pulled and available locally."""
        logging.debug(f"Checking if image {self.container_image} is available")

        # Check if image already exists locally
        inspect_result = subprocess.run(
            ["podman", "image", "exists", self.container_image],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        if inspect_result.returncode != 0:
            logging.debug(f"Image {self.container_image} not found locally, pulling")
            pull_result = subprocess.run(
                ["podman", "pull", self.container_image],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if pull_result.returncode != 0:
                raise RuntimeError(
                    f"Failed to pull image with podman: {pull_result.stderr}"
                )
            logging.debug(f"Successfully pulled {self.container_image}")
        else:
            logging.info(f"Using existing image: {self.container_image}")

    def _recreate(self):
        """(Re)create a clean container instance from the base image."""
        # CRITICAL: Remove any old container AND clean up any stray containers
        logging.debug(f"Removing container {self.container_name}")
        subprocess.run(
            ["podman", "rm", "-f", self.container_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Additional cleanup: Remove any exited containers to prevent accumulation
        logging.debug("Cleaning up exited containers")
        subprocess.run(
            ["podman", "container", "prune", "-f"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Create new container from image
        # We create the container in a stopped state, ready to be started
        result = subprocess.run(
            [
                "podman",
                "create",
                "--name",
                self.container_name,
                "--rm=false",  # We'll manage removal ourselves
                self.container_image,
                "sleep",
                "infinity",  # Keep container alive for exec commands
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        logging.debug(f"Container creation result: {result}")
        if result.returncode != 0:
            raise RuntimeError(f"Failed to recreate container: {result.stderr}")

        # Start the container
        start_result = subprocess.run(
            ["podman", "start", self.container_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if start_result.returncode != 0:
            raise RuntimeError(f"Failed to start container: {start_result.stderr}")

        self._initialized = True
        logging.debug("Successfully initialized container")

    def _run_subprocess_with_retry(
        self, cmd: list[str], max_retries: int = 3, timeout: int | None = None
    ) -> subprocess.CompletedProcess:
        """Run subprocess with exponential backoff retry on resource exhaustion.

        Args:
            cmd: Command to run as list of strings
            max_retries: Maximum number of retry attempts (default: 3)
            timeout: Timeout in seconds for subprocess (default: None)

        Returns:
            subprocess.CompletedProcess result

        Raises:
            RuntimeError: If all retries fail
        """
        last_error = None

        for attempt in range(max_retries):
            try:
                return subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=timeout,
                )
            except BlockingIOError as e:
                last_error = e
                if attempt < max_retries - 1:
                    # Exponential backoff: 0.1s, 0.2s, 0.4s
                    wait_time = 0.1 * (2**attempt)
                    logger.warning(
                        f"BlockingIOError on attempt {attempt + 1}/{max_retries} for cmd {cmd[0:2]}, "
                        f"retrying in {wait_time}s: {e}"
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(
                        f"BlockingIOError persisted after {max_retries} attempts for cmd {cmd[0:2]}: {e}"
                    )
            except Exception as e:
                logger.error(f"Unexpected error running subprocess {cmd[0:2]}: {e}")
                raise

        # If we get here, all retries failed
        raise RuntimeError(
            f"Failed to run subprocess after {max_retries} attempts. "
            f"Last error: {last_error}. This indicates system resource exhaustion. "
            f"Consider reducing max_workers or increasing system process limits."
        )

    def _execute_sync(self, code: str) -> tuple[str, str]:
        """Synchronous code execution - runs in thread pool via run_in_executor.

        This method contains the actual subprocess calls and is designed to be
        run in a thread pool to avoid blocking the async event loop.

        Uses UUID-based filenames to avoid race conditions when multiple threads
        execute code concurrently in the same container.
        """
        # Generate unique script name to avoid race conditions between concurrent executions
        execution_id = uuid.uuid4().hex[:8]  # Short UUID for readability
        script_name = f"script_{execution_id}.py"
        container_script_path = f"/tmp/{script_name}"

        logging.debug(f"Executing code in thread pool (execution_id={execution_id})")

        max_retries = 2
        for attempt in range(max_retries):
            try:
                # Write code to a temporary file and copy it into the container
                with tempfile.TemporaryDirectory() as tmpdir:
                    code_path = Path(tmpdir) / script_name
                    code_path.write_text(code)

                    # Copy the script into the container with retry logic
                    copy_result = self._run_subprocess_with_retry(
                        [
                            "podman",
                            "cp",
                            str(code_path),
                            f"{self.container_name}:{container_script_path}",
                        ],
                        max_retries=3,
                    )
                    if copy_result.returncode != 0:
                        error_msg = copy_result.stderr.lower()
                        # Check for container state errors
                        if any(
                            keyword in error_msg
                            for keyword in [
                                "container",
                                "not running",
                                "state improper",
                                "no such container",
                                "exec session",
                            ]
                        ):
                            raise RuntimeError(f"Container error: {copy_result.stderr}")
                        raise RuntimeError(
                            f"Failed to copy script to container: {copy_result.stderr}"
                        )

                    # Execute the code inside the container with 30 second timeout and retry logic
                    # CRITICAL: Track process to kill it on timeout
                    exec_proc = None
                    try:
                        # Start the process without waiting, so we can kill it on timeout
                        exec_proc = subprocess.Popen(
                            [
                                "podman",
                                "exec",
                                self.container_name,
                                "python3",
                                container_script_path,
                            ],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True,
                        )

                        # Wait for completion with timeout
                        output, error = exec_proc.communicate(timeout=30)

                        # Check for container state errors in stderr
                        if error and any(
                            keyword in error.lower()
                            for keyword in [
                                "container state improper",
                                "exec session",
                                "not running",
                                "no such container",
                            ]
                        ):
                            raise RuntimeError(f"Container error: {error}")

                    except subprocess.TimeoutExpired:
                        logging.warning(
                            f"Code execution timed out after 30 seconds (execution_id={execution_id})"
                        )

                        # CRITICAL FIX: Kill the timed-out process
                        if exec_proc:
                            logging.warning(
                                f"Killing timed-out podman exec process (PID={exec_proc.pid})"
                            )
                            try:
                                exec_proc.kill()
                                exec_proc.wait(timeout=2)
                            except Exception as kill_error:
                                logging.error(f"Failed to kill process: {kill_error}")

                        # Kill ALL python3 processes running this specific script inside container
                        # This ensures the zombie process inside the container is killed
                        logging.warning(
                            f"Killing zombie python3 processes for {script_name}"
                        )
                        try:
                            kill_result = subprocess.run(
                                [
                                    "podman",
                                    "exec",
                                    self.container_name,
                                    "pkill",
                                    "-9",
                                    "-f",
                                    script_name,
                                ],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                timeout=5,
                            )
                            if kill_result.returncode == 0:
                                logging.info(
                                    f"Successfully killed zombie process for {script_name}"
                                )
                            else:
                                logging.warning(
                                    f"pkill returned {kill_result.returncode}: {kill_result.stderr}"
                                )
                        except Exception as kill_error:
                            logging.error(
                                f"Failed to kill zombie process: {kill_error}"
                            )

                        output = ""
                        error = "Error: Code execution timed out after 30 seconds (possible infinite loop)"
                    finally:
                        # Clean up the script file from container to avoid clutter
                        # Use check=False to not fail if file doesn't exist
                        subprocess.run(
                            [
                                "podman",
                                "exec",
                                self.container_name,
                                "rm",
                                "-f",
                                container_script_path,
                            ],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            check=False,
                        )

                    return output, error

            except RuntimeError as e:
                error_msg = str(e).lower()
                # Check for container-related errors
                if any(
                    keyword in error_msg
                    for keyword in [
                        "container",
                        "exec session",
                        "not running",
                        "state improper",
                        "no such container",
                    ]
                ):
                    logging.warning(
                        f"Container error on attempt {attempt + 1}/{max_retries}: {e}"
                    )
                    if attempt < max_retries - 1:
                        logging.info("Attempting to recreate container...")
                        try:
                            self._recreate()
                            logging.info("Container recreated successfully")
                            continue
                        except Exception as recreate_error:
                            logging.error(
                                f"Failed to recreate container: {recreate_error}"
                            )
                            if attempt == max_retries - 1:
                                raise RuntimeError(
                                    f"Container failed and could not be recreated: {e}"
                                ) from e
                    else:
                        raise RuntimeError(
                            f"Container failed after {max_retries} attempts: {e}"
                        ) from e
                else:
                    # Non-container error, propagate immediately
                    raise

        # Should never reach here, but for type safety
        raise RuntimeError("Execution failed after all retry attempts")

    @endpoint
    async def execute(self, code: str) -> tuple[str, str]:
        """Executes Python code inside the container and returns the output.

        Uses ThreadPoolExecutor with run_in_executor to run synchronous subprocess
        calls without blocking the async event loop. This is simpler and more
        reliable than trying to make subprocess async.

        Args:
            code: Python source code string to execute.

        Returns:
            The captured stdout and stderr from the execution, as a
            (stdout, stderr) tuple of strings.
        """
        if not self._initialized:
            raise RuntimeError("Container not initialized. Call recreate() first.")

        # Run synchronous code execution in thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self._execute_sync, code)

    @endpoint
    async def cleanup_zombie_processes(self) -> int:
        """Kill any zombie python3 processes in the container that might be consuming memory.

        Returns:
            Number of processes killed
        """
        logging.debug(f"Checking for zombie python3 processes in {self.container_name}")

        try:
            # Get list of all python3 processes except PID 1
            ps_result = subprocess.run(
                [
                    "podman",
                    "exec",
                    self.container_name,
                    "sh",
                    "-c",
                    "ps aux | grep 'python3 /tmp/script_' | grep -v grep | awk '{print $2}'",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=5,
            )

            if ps_result.returncode == 0 and ps_result.stdout.strip():
                pids = ps_result.stdout.strip().split("\n")
                pid_count = len(pids)
                logging.warning(
                    f"Found {pid_count} zombie python3 processes in container, killing them"
                )

                # Kill all zombie processes at once
                kill_result = subprocess.run(
                    [
                        "podman",
                        "exec",
                        self.container_name,
                        "pkill",
                        "-9",
                        "-f",
                        "python3 /tmp/script_",
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=5,
                )

                if kill_result.returncode == 0:
                    logging.info(f"Successfully killed {pid_count} zombie processes")
                    return pid_count
                else:
                    logging.warning(
                        f"pkill returned {kill_result.returncode}: {kill_result.stderr}"
                    )
                    return 0
            else:
                logging.debug("No zombie processes found")
                return 0

        except Exception as e:
            logging.error(f"Error during zombie process cleanup: {e}")
            return 0

    @endpoint
    async def shutdown(self):
        """Cleanup resources - shutdown thread pool executor and remove container."""
        logging.debug("Shutting down PodmanPythonCoder thread pool")
        try:
            self._executor.shutdown(wait=True, cancel_futures=False)
            logging.info("Thread pool executor shutdown successfully")
        except Exception as e:
            logging.error(f"Error during thread pool shutdown: {e}")

        # CRITICAL: Remove the container to prevent memory leaks
        logging.debug(f"Removing container {self.container_name}")
        try:
            result = subprocess.run(
                ["podman", "rm", "-f", self.container_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                logging.info(f"Successfully removed container {self.container_name}")
            else:
                logging.warning(f"Failed to remove container: {result.stderr}")
        except Exception as e:
            logging.error(f"Error removing container during shutdown: {e}")
